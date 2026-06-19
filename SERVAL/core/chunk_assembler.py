"""
Chunk Assembler

Sits between TCPReceiver and the ZMQ extractor workers.  Accumulates
raw TPX3-aligned chunks and forwards super-chunks whose boundaries fall
just before every N-th rising edge of the configured TDC channel.

Each forwarded super-chunk is therefore guaranteed to start with a
trigger and contain exactly N complete laser shots, so every extractor
worker gets a self-contained unit that can be correlated without any
cross-chunk carry-over state.

When triggers_per_chunk == 0 the assembler operates in pass-through
mode: chunks are forwarded immediately with no accumulation, preserving
the original size/time-based behaviour.

The assembler duck-types zmq.Socket.send() so TCPReceiver can hand it
to set_targets() without any modification.
"""

import queue
import threading
import time

import zmq

from SERVAL.core.utils import find_nth_trigger_boundary
from SERVAL.utils.logging import get_logger


class ChunkAssembler(threading.Thread):
    """
    Thread that accumulates raw TPX3 chunks and emits trigger-aligned
    super-chunks to a ZMQ PUSH socket.

    Parameters
    ----------
    zmq_socket : zmq.Socket
        Downstream PUSH socket connected to extractor workers.
    triggers_per_chunk : int
        Flush every N rising edges of the selected TDC.  0 = pass-through.
    tdc_rising_subheader : int
        Raw subheader byte that identifies the rising edge of interest:
        0xF = TDC1 rising, 0xE = TDC2 rising.
    flush_timeout : float
        Maximum seconds to hold a partial buffer before flushing anyway.
        This is the fallback when triggers are absent or infrequent.
    max_buffer_bytes : int
        Safety flush if the buffer grows past this size regardless of
        trigger count (prevents unbounded memory use at very high rates
        with no triggers).
    input_queue_maxsize : int
        Depth of the internal queue that buffers incoming raw chunks from
        the TCPReceiver thread.
    """

    def __init__(
        self,
        zmq_socket: zmq.Socket,
        triggers_per_chunk: int = 0,
        tdc_rising_subheader: int = 0xF,
        flush_timeout: float = 0.3,
        max_buffer_bytes: int = 20_000_000,
        input_queue_maxsize: int = 200,
    ):
        super().__init__(name="ChunkAssembler", daemon=True)
        self._zmq_socket = zmq_socket
        self.triggers_per_chunk = triggers_per_chunk
        self.tdc_rising_subheader = tdc_rising_subheader
        self.flush_timeout = flush_timeout
        self.max_buffer_bytes = max_buffer_bytes
        self._input_queue: queue.Queue = queue.Queue(maxsize=input_queue_maxsize)
        self._stop_event = threading.Event()
        self.logger = get_logger("SERVAL.ChunkAssembler")

        self._chunks_forwarded = 0
        self._chunks_dropped = 0

    # ------------------------------------------------------------------
    # Duck-typed zmq.Socket interface — TCPReceiver needs no changes
    # ------------------------------------------------------------------

    def send(self, data, flags: int = 0, copy: bool = True):
        """Accept a raw chunk from TCPReceiver (mirrors zmq.Socket.send)."""
        try:
            self._input_queue.put_nowait(bytes(data))
        except queue.Full:
            self._chunks_dropped += 1
            self.logger.warning("Assembler input queue full — dropped chunk")

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def stop(self):
        """Signal the thread to drain and exit."""
        self._stop_event.set()
        try:
            self._input_queue.put_nowait(None)  # unblock a waiting get()
        except queue.Full:
            pass

    def run(self):
        passthrough = (self.triggers_per_chunk == 0)
        self.logger.info(
            "Started — " + (
                f"trigger-aligned ({self.triggers_per_chunk} rising edges/chunk, "
                f"subheader=0x{self.tdc_rising_subheader:X})"
                if not passthrough else "pass-through mode"
            )
        )

        buffer = bytearray()
        last_flush = time.monotonic()

        while not self._stop_event.is_set():
            try:
                chunk = self._input_queue.get(timeout=0.05)
            except queue.Empty:
                # Periodic timeout flush so the pipeline never stalls when
                # triggers are absent (e.g. laser off, testing dark counts).
                if buffer and (time.monotonic() - last_flush) >= self.flush_timeout:
                    self._forward(buffer)
                    buffer = bytearray()
                    last_flush = time.monotonic()
                continue

            if chunk is None:  # stop sentinel
                break

            # --- Pass-through: forward immediately, no accumulation ---
            if passthrough:
                self._forward_bytes(chunk)
                continue

            # --- Trigger-aligned accumulation ---
            buffer.extend(chunk)

            boundary = find_nth_trigger_boundary(
                bytes(buffer), self.triggers_per_chunk, self.tdc_rising_subheader
            )

            if boundary != -1:
                # Flush up to (but not including) the (N+1)-th trigger's chunk
                self._forward(buffer[:boundary])
                buffer = bytearray(buffer[boundary:])
                last_flush = time.monotonic()
            elif len(buffer) >= self.max_buffer_bytes:
                # Safety flush: buffer has grown too large without a trigger
                self.logger.debug(
                    f"Safety flush at {len(buffer):,} B "
                    f"(no trigger boundary found in {self.triggers_per_chunk}-trigger window)"
                )
                self._forward(buffer)
                buffer = bytearray()
                last_flush = time.monotonic()

        # Drain any remaining data before the thread exits
        if buffer:
            self._forward(buffer)

        self.logger.info(
            f"Stopped — forwarded {self._chunks_forwarded}, "
            f"dropped {self._chunks_dropped}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(self, data: bytearray):
        self._forward_bytes(bytes(data))

    def _forward_bytes(self, data: bytes):
        if not data:
            return
        try:
            self._zmq_socket.send(data, flags=zmq.NOBLOCK, copy=False)
            self._chunks_forwarded += 1
        except zmq.Again:
            self.logger.warning("ZMQ HWM reached — dropped assembled chunk")
        except Exception as e:
            self.logger.error(f"ZMQ send error: {e}")
