#!/usr/bin/env python3
"""
Extraction workers for TPX3 pipeline.

ExtractorWorker: Process that extracts and correlates TPX3 data
ExtractorPool: Manages multiple extraction workers
"""

import multiprocessing
import queue
import time
from pathlib import Path
from typing import Optional

import numpy as np
import zmq

from SERVAL.utils.logging import get_logger
from SERVAL.core.data_types import TDCChannel, TriggerEdge
from SERVAL.core.extractors.parallel_processor import (
    TPX3Extractor, PixelData,
    _correlate_pixels_jit, _correlate_pixels_parallel, _centroid_hits,
)
from .savers import EventSaverProcess, PixelSaverProcess, TriggerSaverProcess


class ExtractorWorker(multiprocessing.Process):
    """
    Worker that extracts pixels/triggers and correlates them to events.

    Receives raw TPX3 chunks via ZMQ, processes them using JIT-compiled
    functions, and outputs correlated events.
    """

    # # Re-export for backwards compatibility
    # EVENT_DTYPE = EVENT_DTYPE

    def __init__(
        self,
        worker_id: int,
        zmq_port: int,
        stats_interval: int = 100,  # Log stats every N chunks
        use_fast_extract: bool = False,
        log_level: str = "INFO",
        tdc_id: TDCChannel = TDCChannel.TDC1,
        edge: TriggerEdge = TriggerEdge.RISING,
        event_window: tuple = (0.0, 100_000_000.0),  # ns
        write_buffer_size: int = 500_000,
        correlate_func = _correlate_pixels_parallel,
        use_centroiding: bool = False,
        eps_space: int = 2,       # pixels (Manhattan distance)
        eps_time_ns: float = 100.0,  # nanoseconds
        b_size: int = 16,
        event_queue: Optional[multiprocessing.Queue] = None,  # For correlated events (saving)
        pixel_queue: Optional[multiprocessing.Queue] = None,  # For raw pixels (saving)
        trigger_queue: Optional[multiprocessing.Queue] = None,  # For all triggers (saving)
        callback_event_queue: Optional[multiprocessing.Queue] = None,  # For GUI callbacks
        callback_pixel_queue: Optional[multiprocessing.Queue] = None,  # For GUI callbacks
        callback_trigger_queue: Optional[multiprocessing.Queue] = None,  # For GUI callbacks
        recording_flag=None,  # multiprocessing.Value('b', 0) — gates saver queue puts
        callback_display_flag=None,  # multiprocessing.Value('i') — 0=events, 1=pixels
        callback_display_fraction=None,  # multiprocessing.Value('d', 1.0) — GUI subsample fraction
        chunks_processed_counter=None,  # multiprocessing.Value('L', 0) — incremented per chunk
    ):
        super().__init__(name=f"Extractor-{worker_id}")
        self.worker_id = worker_id
        self.zmq_port = zmq_port
        self.chunks_processed_counter = chunks_processed_counter
        self.event_queue = event_queue
        self.pixel_queue = pixel_queue
        self.trigger_queue = trigger_queue
        self.callback_event_queue = callback_event_queue
        self.callback_pixel_queue = callback_pixel_queue
        self.callback_trigger_queue = callback_trigger_queue
        self.stats_interval = stats_interval
        self.use_fast_extract = use_fast_extract
        self.log_level = log_level
        self.tdc_id = tdc_id
        self.edge = edge
        self.event_window_min = event_window[0] * 1e-9
        self.event_window_max = event_window[1] * 1e-9
        self.write_buffer_size = write_buffer_size
        self.recording_flag = recording_flag
        self.callback_display_flag = callback_display_flag
        self.callback_display_fraction = callback_display_fraction
        self.use_centroiding = use_centroiding
        self.eps_space = eps_space
        self.eps_time = eps_time_ns * 1e-9  # convert to seconds
        self.b_size = b_size
        self.daemon = True
        self._correlate_func = correlate_func
    
    @property
    def correlate_func(self):
        return self._correlate_func
    
    @correlate_func.setter
    def correlate_func(self, func):
        self._correlate_func = func

    def _subsample_for_display(self, arrays: tuple) -> tuple:
        """Stride-subsample arrays for the GUI callback queue, per ``callback_display_fraction``.

        Shrinks the IPC payload (and thus the multiprocessing.Queue transport
        cost) proportionally to the "Display %" setting, instead of sending
        the full chunk and discarding most of it client-side. Saving to disk
        is unaffected — this only touches the GUI callback queue payload.
        """
        _dfrac = self.callback_display_fraction
        frac = _dfrac.value if _dfrac is not None else 1.0
        if frac >= 1.0 or len(arrays[0]) == 0:
            return arrays
        stride = max(1, round(1.0 / frac))
        return tuple(a[::stride] for a in arrays)

    def run(self):
        from SERVAL.utils.logging import set_log_level

        set_log_level(self.log_level)
        logger = get_logger(f"SERVAL.Extractor-{self.worker_id}")

        mode = "FAST" if self.use_fast_extract else "STANDARD"
        logger.info(f"Started ({mode} mode, PID: {multiprocessing.current_process().pid})")

        # ZMQ setup
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.connect(f"tcp://127.0.0.1:{self.zmq_port}")

        # Extractor (stateless, fast)
        extractor = TPX3Extractor(debug_log_interval=0)  # Disable internal logging
        extract_fn = extractor.extract_fast if self.use_fast_extract else extractor.extract

        # Stats
        chunks_processed = 0
        total_pixels = 0
        total_triggers = 0
        total_events = 0
        start_time = time.time()
        
        diagnostics = {'t_extract': np.empty(self.stats_interval),
                        't_correlate': np.empty(self.stats_interval),
                        'pixels_per_chunk': np.empty(self.stats_interval),
                        'triggers_per_chunk': np.empty(self.stats_interval),
                        'valid_events': np.empty(self.stats_interval)
                        }
        try:
            while True:
                raw_bytes = socket.recv(copy=False).bytes

                if len(raw_bytes) == 0:
                    logger.info("Shutdown signal received")
                    break

                t0 = time.perf_counter()

                # Extract pixels and triggers
                pixels, triggers, _, _ = extract_fn(raw_bytes)
                t_extract = time.perf_counter()

                # Optional greedy centroiding (replaces raw pixels with centroids)
                if self.use_centroiding and len(pixels) > 0:
                    cx, cy, ctoa, ctot, _ = _centroid_hits(
                        pixels.x, pixels.y, pixels.toa, pixels.tot,
                        self.eps_space, self.eps_time, self.b_size,
                    )
                    pixels = PixelData(x=cx, y=cy, toa=ctoa, tot=ctot)

                chunks_processed += 1
                total_pixels += len(pixels)
                total_triggers += len(triggers)

                if self.chunks_processed_counter is not None:
                    with self.chunks_processed_counter.get_lock():
                        self.chunks_processed_counter.value += 1

                # Send all triggers to trigger queue (before TDC filter)
                if len(triggers) > 0:
                    trigger_data = (triggers.toa, triggers.tdc_id, triggers.edge)
                    if self.trigger_queue is not None and (
                        self.recording_flag is None or self.recording_flag.value
                    ):
                        try:
                            self.trigger_queue.put(trigger_data, timeout=1.0)
                        except queue.Full:
                            logger.warning("Trigger saver queue full, dropped chunk")
                    if self.callback_trigger_queue is not None:
                        try:
                            self.callback_trigger_queue.put_nowait(trigger_data)
                        except Exception:
                            pass

                # Send raw pixels to queues if enabled
                if len(pixels) > 0:
                    pixel_data = (pixels.x, pixels.y, pixels.toa, pixels.tot)
                    # Saver queue (gated by recording_flag)
                    if self.pixel_queue is not None and (
                        self.recording_flag is None or self.recording_flag.value
                    ):
                        try:
                            self.pixel_queue.put(pixel_data, timeout=1.0)
                        except queue.Full:
                            logger.warning("Pixel saver queue full, dropped chunk")
                    # Callback queue (for GUI) — only when display_flag selects pixels (1)
                    _dflag = self.callback_display_flag
                    if self.callback_pixel_queue is not None and (
                        _dflag is None or _dflag.value == 1
                    ):
                        try:
                            cb_data = self._subsample_for_display(pixel_data)
                            # Append a monotonic timestamp (callback path only) so the
                            # GUI can measure end-to-end display latency.
                            self.callback_pixel_queue.put_nowait(cb_data + (time.monotonic(),))
                        except Exception:
                            pass  # Drop if queue full - GUI can handle missing frames

                # Skip correlation if no event queues configured
                if self.event_queue is None and self.callback_event_queue is None:
                    continue

                if len(pixels) == 0 or len(triggers) == 0:
                    continue

                # Filter triggers by TDC and edge
                if self.tdc_id == 0:
                    mask = triggers.edge == self.edge
                else:
                    mask = (triggers.tdc_id == self.tdc_id) & (triggers.edge == self.edge)

                trigger_times = triggers.toa[mask]

                if len(trigger_times) < 2:
                    continue

                # JIT correlation
                t_trigger, ex, ey, etof, etot, n_valid = self.correlate_func(
                    pixels.toa,
                    pixels.x,
                    pixels.y,
                    pixels.tot,
                    trigger_times,
                    self.event_window_min,
                    self.event_window_max,
                )

                t_corr = time.perf_counter()

                if n_valid == 0:
                    continue

                # Send to event queues
                event_data = (t_trigger, ex, ey, etof, etot)
                # Saver queue (gated by recording_flag)
                if self.event_queue is not None and (
                    self.recording_flag is None or self.recording_flag.value
                ):
                    try:
                        self.event_queue.put(event_data, timeout=1.0)
                    except queue.Full:
                        logger.warning("Event saver queue full, dropped chunk")
                # Callback queue (for GUI) — only when display_flag selects events (0)
                _dflag = self.callback_display_flag
                if self.callback_event_queue is not None and (
                    _dflag is None or _dflag.value == 0
                ):
                    try:
                        cb_data = self._subsample_for_display(event_data)
                        # Append a monotonic timestamp (callback path only) so the
                        # GUI can measure end-to-end display latency.
                        self.callback_event_queue.put_nowait(cb_data + (time.monotonic(),))
                    except Exception:
                        pass  # Drop if queue full - GUI can handle missing frames

                total_events += n_valid

                index = (chunks_processed-1) % self.stats_interval
                diagnostics['t_extract'][index] = (t_extract - t0)
                diagnostics['t_correlate'][index] = (t_corr - t_extract)
                diagnostics['pixels_per_chunk'][index] = len(pixels)
                diagnostics['triggers_per_chunk'][index] = len(triggers)
                diagnostics['valid_events'][index] = n_valid

                # Periodic stats (reduced frequency)
                if index  == 0:
                    t_extract = diagnostics['t_extract'].mean() * 1000
                    t_correlate = diagnostics['t_correlate'].mean() * 1000
                    pixels_per_chunk = diagnostics['pixels_per_chunk'].sum()
                    n_event = int(diagnostics['valid_events'].sum())
                    triggers_per_chunk = diagnostics['triggers_per_chunk'].sum()
                    t_total = t_extract + t_correlate
                    logger.info(
                        f"[W{self.worker_id}] {pixels_per_chunk:,} px / {triggers_per_chunk:,} tr → {n_event:,} ev | "
                        f"{t_total:.1f}ms (ext:{t_extract:.1f} corr:{t_correlate:.1f})"
                    )
                    diagnostics = {key: np.empty(self.stats_interval) for key in diagnostics.keys()}

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
        finally:
            elapsed = time.time() - start_time
            logger.info(
                f"Final: {chunks_processed} chunks, {total_pixels:,} px, "
                f"{total_triggers:,} trig, {total_events:,} events in {elapsed:.1f}s"
            )
            socket.close()
            context.term()


class ExtractorPool:
    """
    Manages a pool of extraction workers and their associated saver processes.

    Handles:
    - ZMQ socket setup for distributing work to workers
    - Starting/stopping worker processes
    - Starting/stopping saver processes
    """

    # Default save configuration
    DEFAULT_SAVE_CONFIG = {
        "events": {"enabled": True, "num_savers": 1, "buffer_size": 500_000},
        "pixels": {"enabled": False, "num_savers": 1, "buffer_size": 500_000},
        "triggers": {"enabled": True, "num_savers": 1, "buffer_size": 500_000},
    }

    def __init__(
        self,
        num_workers: int,
        zmq_port: int,
        use_fast_extract: bool = False,
        log_level: str = "INFO",
        tdc_id: TDCChannel = TDCChannel.TDC1,
        edge: TriggerEdge = TriggerEdge.RISING,
        event_window: tuple = (0.0, 10_000.0),
        zmq_hwm: int = 1000,
        output_dir: Optional[Path] = None,
        save_config: Optional[dict] = None,
        callback_event_queue: Optional[multiprocessing.Queue] = None,
        callback_pixel_queue: Optional[multiprocessing.Queue] = None,
        recording_flag=None,  # multiprocessing.Value — gates saver queue puts
        callback_display_flag=None,  # multiprocessing.Value('i') — 0=events, 1=pixels
        callback_display_fraction=None,  # multiprocessing.Value('d', 1.0) — GUI subsample fraction
        use_centroiding: bool = False,
        eps_space: int = 2,
        eps_time_ns: float = 100.0,
        b_size: int = 16,
    ):
        self.num_workers = num_workers
        self.zmq_port = zmq_port
        self.use_fast_extract = use_fast_extract
        self.log_level = log_level
        self.tdc_id = tdc_id
        self.edge = edge
        self.event_window = event_window
        self.output_dir = output_dir
        self.zmq_hwm = zmq_hwm
        self.callback_event_queue = callback_event_queue
        self.callback_pixel_queue = callback_pixel_queue
        self.recording_flag = recording_flag
        self.callback_display_flag = callback_display_flag
        self.callback_display_fraction = callback_display_fraction
        self.use_centroiding = use_centroiding
        self.eps_space = eps_space
        self.eps_time_ns = eps_time_ns
        self.b_size = b_size

        # Merge user config with defaults
        self.save_config = {
            key: {**self.DEFAULT_SAVE_CONFIG[key], **(save_config or {}).get(key, {})}
            for key in self.DEFAULT_SAVE_CONFIG
        }

        self.zmq_context = None
        self.zmq_socket = None
        self.workers = []
        # Per-worker shared counters of chunks pulled off the ZMQ socket —
        # used to estimate the extraction backlog (chunks_sent - sum(counters)).
        self.chunk_counters = [multiprocessing.Value('L', 0) for _ in range(num_workers)]
        # Saver processes/queues keyed by type ("events", "pixels", "triggers")
        self.saver_processes = {"events": [], "pixels": [], "triggers": []}
        self.saver_queues = {"events": [], "pixels": [], "triggers": []}
        self.logger = get_logger("SERVAL.ExtractorPool")

        # Create saver queues and processes upfront (started idle)
        self._init_savers()

    def _init_savers(self):
        """Create saver queues and processes (idle, not yet started)."""
        type_to_class = {
            "events": EventSaverProcess,
            "pixels": PixelSaverProcess,
            "triggers": TriggerSaverProcess,
        }
        for saver_type, saver_class in type_to_class.items():
            config = self.save_config[saver_type]
            if not config["enabled"] or config.get("num_savers", 0) == 0:
                continue

            num_savers = config["num_savers"]
            buffer_size = config["buffer_size"]
            queue_size = config.get("queue_size", 1000)

            self.saver_queues[saver_type] = [
                multiprocessing.Queue(maxsize=queue_size) for _ in range(num_savers)
            ]

            for q in self.saver_queues[saver_type]:
                saver = saver_class(
                    input_queue=q,
                    buffer_size=buffer_size,
                    log_level=self.log_level,
                )
                self.saver_processes[saver_type].append(saver)

    def start_savers(self):
        """Start pre-created saver processes."""
        for saver_type, savers in self.saver_processes.items():
            for saver in savers:
                saver.start()
        total = sum(len(s) for s in self.saver_processes.values())
        if total:
            self.logger.info(f"Started {total} saver process(es)")

    def setup_zmq(self):
        """Setup ZMQ PUSH socket for distributing work to workers."""
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_socket.setsockopt(zmq.SNDHWM, self.zmq_hwm)
        self.zmq_socket.bind(f"tcp://127.0.0.1:{self.zmq_port}")
        actual_hwm = self.zmq_socket.getsockopt(zmq.SNDHWM)
        self.logger.info(f"ZMQ bound to port {self.zmq_port} (HWM: {actual_hwm})")
        time.sleep(0.3)  # Allow workers to connect
        return self.zmq_socket

    def start_workers(self):
        """Start extraction workers (savers must already be started via start_savers())."""
        mode = "FAST" if self.use_fast_extract else "STANDARD"
        self.logger.info(f"Starting {self.num_workers} workers ({mode})")

        for i in range(self.num_workers):
            events_queues = self.saver_queues["events"]
            pixels_queues = self.saver_queues["pixels"]
            triggers_queues = self.saver_queues["triggers"]

            event_queue = (
                events_queues[i % len(events_queues)] if events_queues else None
            )
            pixel_queue = (
                pixels_queues[i % len(pixels_queues)] if pixels_queues else None
            )
            trigger_queue = (
                triggers_queues[i % len(triggers_queues)] if triggers_queues else None
            )

            worker = ExtractorWorker(
                worker_id=i,
                zmq_port=self.zmq_port,
                event_queue=event_queue,
                pixel_queue=pixel_queue,
                trigger_queue=trigger_queue,
                callback_event_queue=self.callback_event_queue,
                callback_pixel_queue=self.callback_pixel_queue,
                use_fast_extract=self.use_fast_extract,
                log_level=self.log_level,
                tdc_id=self.tdc_id,
                edge=self.edge,
                event_window=self.event_window,
                recording_flag=self.recording_flag,
                callback_display_flag=self.callback_display_flag,
                callback_display_fraction=self.callback_display_fraction,
                use_centroiding=self.use_centroiding,
                eps_space=self.eps_space,
                eps_time_ns=self.eps_time_ns,
                b_size=self.b_size,
                chunks_processed_counter=self.chunk_counters[i],
            )
            worker.start()
            self.workers.append(worker)

    def get_total_processed(self) -> int:
        """Total chunks pulled off the ZMQ socket by all workers (for backlog estimation)."""
        return sum(c.value for c in self.chunk_counters)

    def shutdown(self, timeout: float = 5.0):
        """Shutdown all workers and savers gracefully."""
        # Send shutdown signals to workers
        for _ in range(self.num_workers):
            try:
                if self.zmq_socket:
                    self.zmq_socket.send(b"", flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                pass

        # Wait for workers
        for w in self.workers:
            w.join(timeout=timeout)
            if w.is_alive():
                w.terminate()
        self.workers.clear()

        # Close ZMQ
        if self.zmq_socket:
            self.zmq_socket.close()
        if self.zmq_context:
            self.zmq_context.term()

        # Stop all saver processes
        for saver_type in self.saver_queues:
            for q in self.saver_queues[saver_type]:
                try:
                    q.put("STOP", timeout=1.0)
                except Exception:
                    pass

        for saver_type in self.saver_processes:
            for saver in self.saver_processes[saver_type]:
                saver.stop()
                saver.join(timeout=timeout)
                if saver.is_alive():
                    saver.terminate()
            self.saver_processes[saver_type].clear()

        # Release semaphores held by each Queue after all consumers have exited
        for saver_type in self.saver_queues:
            for q in self.saver_queues[saver_type]:
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass
            self.saver_queues[saver_type].clear()

        self.logger.info("Shutdown complete")
