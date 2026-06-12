#!/usr/bin/env python3
"""
TCP Receiver for TPX3 data stream.

Handles TCP socket management, connection accept/disconnect,
and data receiving with ring buffers.
"""
import numpy as np
from SERVAL.core.utils import find_last_pattern

import socket
import struct
import threading
import queue
import time
from typing import Callable, Optional, Tuple

import zmq

from SERVAL.utils import EventBus, Events
from SERVAL.utils.logging import get_logger

# SERVAL chunk header constants
TPX3_SIGNATURE = 0x33585054  # "TPX3" in little-endian


class TCPReceiver:
    """
    TCP receiver with connection management and ring buffers.

    Handles:
    - Server socket creation and binding
    - Connection accept loop with reconnection support
    - Zero-copy receive into ring buffers
    - Distribution to save queue and ZMQ socket

    Parameters
    ----------
    host : str
        TCP bind address
    port : int
        TCP bind port
    recv_buffer_size : int
        Size of each recv_into() call (bytes)
    socket_buffer_size : int
        OS socket buffer size (bytes)
    num_ring_buffers : int
        Number of ring buffers for zero-copy receive
    chunk_size : int
        Flush when this many bytes accumulated
    flush_timeout : float
        Flush after this many seconds even if chunk_size not reached
    """

    def __init__(
        self,
        host: str = '192.168.1.2',
        port: int = 8088,
        recv_buffer_size: int = 2 * 1024 * 1024,
        socket_buffer_size: int = 128 * 1024 * 1024,
        num_ring_buffers: int = 10,
        chunk_size: int = 10_000_000,
        flush_timeout: float = 0.3,
        event_bus: Optional[EventBus] = None,
        recording_flag=None,  # multiprocessing.Value('b', 0) — gates raw save queue
    ):
        self.logger = get_logger('SERVAL.TCPReceiver')
        self.host = host
        self.port = port
        self.recv_buffer_size = recv_buffer_size
        self.socket_buffer_size = socket_buffer_size
        self.num_ring_buffers = num_ring_buffers
        self.chunk_size = chunk_size
        self.flush_timeout = flush_timeout
        self.bus = event_bus
        self.recording_flag = recording_flag

        # Ring buffers
        buffer_size = int(1.5 * chunk_size)
        self.ring_buffers = [bytearray(buffer_size) for _ in range(num_ring_buffers)]
        self.ring_buffer_views = [memoryview(buf) for buf in self.ring_buffers]
        self.ring_buffer_index = 0

        # Socket state
        self.server_socket: Optional[socket.socket] = None
        self.current_connection: Optional[socket.socket] = None
        self.is_connected = False
        self._connection_lock = threading.Lock()

        # Output targets (set by pipeline)
        self.save_queues: list = []  # List of save queues for round-robin
        self.save_queue_index: int = 0  # Current queue index
        self.zmq_socket: Optional[zmq.Socket] = None

        # Thread control
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def _publish(self, event: str, *args, **kwargs):
        """Publish event to bus if available."""
        if self.bus:
            self.bus.publish(event, *args, **kwargs)

    def set_targets(self, save_queues, zmq_socket: zmq.Socket):
        """Set output targets for received data.

        Parameters
        ----------
        save_queues : queue.Queue or list of queues or None
            Single queue, list of queues for round-robin, or None to disable
        zmq_socket : zmq.Socket
            ZMQ socket for sending to extractors
        """
        if save_queues is None:
            self.save_queues = []
        elif isinstance(save_queues, list):
            self.save_queues = save_queues
        else:
            self.save_queues = [save_queues]  # Single queue -> list of one
        self.save_queue_index = 0
        self.zmq_socket = zmq_socket

    def _set_connected(self, connected: bool, client_address: Optional[tuple] = None):
        """Update connection state and publish event."""
        with self._connection_lock:
            self.is_connected = connected
            self._publish(Events.CONNECTION_CHANGED, connected, client_address)

    def bind(self) -> int:
        """
        Create and bind server socket.

        Returns
        -------
        int
            Actual socket buffer size
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        actual_buffer = self.server_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self.logger.info(f"Listening on {self.host}:{self.port} (buffer: {actual_buffer/1024/1024:.1f} MB)")
        return actual_buffer

    def start(self):
        """Start receiver thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, name="TCPReceiver")
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """Stop receiver thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def close(self):
        """Close server socket."""
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None

    def _run(self):
        """Main receiver loop with connection management."""
        self.logger.info("Starting - waiting for connections...")

        while self.running:
            connection = self._accept_connection()
            if connection is None:
                continue

            self._receive_loop(connection)

        self.logger.info("Shutting down")

    def _accept_connection(self) -> Optional[socket.socket]:
        """Wait for and accept a connection."""
        self.server_socket.settimeout(1.0)

        while self.running:
            try:
                connection, client_address = self.server_socket.accept()
                connection.settimeout(1.0)
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.socket_buffer_size)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.current_connection = connection
                self._set_connected(True, client_address)
                self.logger.info(f"Connected from {client_address}")
                return connection
            except socket.timeout:
                continue
            except OSError as e:
                if self.running:
                    self.logger.error(f"Accept error: {e}")
                return None

        return None

    def _receive_loop(self, connection: socket.socket):
        """Receive data from connection until it closes."""
        last_flush_time = time.time()
        bytes_in_buffer = 0
        current_view = self.ring_buffer_views[self.ring_buffer_index]

        try:
            while self.running:
                try:
                    nbytes = connection.recv_into(
                        current_view[bytes_in_buffer:],
                        self.recv_buffer_size
                    )
                    if nbytes == 0:
                        self.logger.info("Connection closed by remote")
                        break

                    bytes_in_buffer += nbytes
                    self._publish(Events.BYTES_RECEIVED, nbytes)
                    current_time = time.time()
                    time_since_flush = current_time - last_flush_time

                    if bytes_in_buffer >= self.chunk_size or time_since_flush >= self.flush_timeout:
                        if bytes_in_buffer > 0:
                            data_unflushed = self._flush_buffer(current_view, bytes_in_buffer)
                            self.ring_buffer_index = (self.ring_buffer_index + 1) % self.num_ring_buffers
                            current_view = self.ring_buffer_views[self.ring_buffer_index]
                            # Copy unflushed data to start of new buffer
                            unflushed_len = len(data_unflushed)
                            if unflushed_len:
                                current_view[:unflushed_len] = data_unflushed
                            # Continue writing after unflushed bytes
                            bytes_in_buffer = unflushed_len
                            last_flush_time = current_time
                            # bytes_in_buffer = 0
                            # last_flush_time = current_time

                except socket.timeout:
                    # Flush on timeout for low latency
                    if bytes_in_buffer > 0:
                        data_unflushed = self._flush_buffer(current_view, bytes_in_buffer)
                        self.ring_buffer_index = (self.ring_buffer_index + 1) % self.num_ring_buffers
                        current_view = self.ring_buffer_views[self.ring_buffer_index]
                        # Carry forward any unflushed partial-chunk data (same as non-timeout path)
                        unflushed_len = len(data_unflushed)
                        if unflushed_len:
                            current_view[:unflushed_len] = data_unflushed
                        bytes_in_buffer = unflushed_len
                        last_flush_time = time.time()

                except (ConnectionResetError, BrokenPipeError) as e:
                    self.logger.warning(f"Connection lost: {e}")
                    break

                except Exception as e:
                    self.logger.exception(f"Receive error: {e}")
                    break

        finally:
            # Flush remaining data
            if bytes_in_buffer > 0:
                data_unflushed = self._flush_buffer(current_view, bytes_in_buffer)

            # Clean up connection
            try:
                connection.close()
            except Exception:
                pass

            self.current_connection = None
            self._set_connected(False)

            if self.running:
                self.logger.info("Waiting for new connection...")

    def _flush_buffer(self, view: memoryview, nbytes: int):
        """Send buffered data to save queues (round-robin) and ZMQ."""
        data = bytes(view[:nbytes])
        last_chunk_index = find_last_pattern(data, pattern=b"TPX3")
        if last_chunk_index == -1:
            return data  # No complete chunk found, keep all data unflushed
        else:
            data_flushed = data[:last_chunk_index]
            data_unflushed = data[last_chunk_index:]

            # Nothing complete to flush yet (e.g. the only "TPX3" marker found is at
            # offset 0). An empty payload sent to ZMQ is indistinguishable from the
            # worker shutdown signal (zmq_socket.send(b"")), so skip sending entirely.
            if not data_flushed:
                return data_unflushed

            # Send to save queue (round-robin if multiple), gated by recording_flag
            if self.save_queues and (self.recording_flag is None or self.recording_flag.value):
                current_queue = self.save_queues[self.save_queue_index]
                self.save_queue_index = (self.save_queue_index + 1) % len(self.save_queues)
                try:
                    current_queue.put_nowait(data_flushed)
                except queue.Full:
                    self._publish(Events.CHUNK_DROPPED_SAVE)
                    self.logger.warning("Save queue full, dropped chunk")

            # Send to ZMQ
            if self.zmq_socket:
                try:
                    self.zmq_socket.send(data_flushed, flags=zmq.NOBLOCK, copy=False)
                    self._publish(Events.CHUNK_SENT, nbytes)
                except zmq.Again:
                    self._publish(Events.CHUNK_DROPPED_ZMQ)
                    self.logger.warning("ZMQ send would block, dropped chunk")

            # Move unflushed data to start of next buffer
            return data_unflushed



