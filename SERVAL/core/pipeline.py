#!/usr/bin/env python3
"""
TPX3 Pipeline V3 - Two-Stage Architecture

Stage 1: Parallel Extraction (Workers)
  - Fast bit manipulation in parallel
  - JIT-compiled correlation

Stage 2: Parallel Saving
  - Multiple saver processes for raw and event data
  - Configurable parallelism

Saving is decoupled from pipeline operation: savers are always-running
processes that start idle. Call start_record() / stop_record() to
begin and end a recording session without restarting the pipeline.
"""

import json
import multiprocessing
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from SERVAL.utils.logging import get_logger
from SERVAL.utils import EventBus, Events
from SERVAL.core.data_types import TDCChannel, TriggerEdge
from SERVAL.core.tcp_receiver import TCPReceiver
from SERVAL.core.stats_reporter import StatsReporter
from SERVAL.core.workers import ExtractorPool, RawSaverProcess


class TPX3PipelineV3:
    """
    Two-stage pipeline for TPX3 data acquisition.

    Handles:
    - TCP reception of raw data
    - Parallel extraction and correlation via worker processes
    - Dynamic recording control: start_record() / stop_record()

    Configuration is organized into four dicts:
    - connection_config: TCP receiver settings
    - save_config: Output directory and saver settings
    - extract_config: Worker count, fast mode, and correlation settings
    - command_config: ZMQ command server settings (for PyMoDAQ integration)
    """

    DEFAULT_CONNECTION_CONFIG = {
        "host": "192.168.1.2",
        "port": 8088,
        "recv_buffer_size": 2 * 1024 * 1024,
        "socket_buffer_size": 128 * 1024 * 1024,
        "num_ring_buffers": 10,
        "chunk_size": 10_000_000,
        "flush_timeout": 0.3,
    }

    DEFAULT_SAVE_CONFIG = {
        "output_dir": "./data",
        "raw": {"enabled": True, "num_savers": 1, "buffer_size": 8 * 1024 * 1024, "queue_size": 200},
        "events": {"enabled": True, "num_savers": 2, "buffer_size":500_000, "queue_size": 1000},
        "pixels": {"enabled": False, "num_savers": 0, "buffer_size": 500_000, "queue_size": 1000},
        "triggers": {"enabled": True, "num_savers": 1, "buffer_size": 500_000, "queue_size": 1000},
    }

    DEFAULT_EXTRACT_CONFIG = {
        "num_workers": 4,
        "use_fast_extract": False,
        # "zmq_port": 9001,
        "zmq_port": 9200,
        "zmq_hwm": 1000,
        "event_window": (0.0, 10_000.0),  # ns
        "tdc_id": TDCChannel.TDC1,
        "edge": TriggerEdge.RISING,
        "events": True,
        "pixels": True,
        # Greedy centroiding (applied to raw pixels before trigger correlation)
        "use_centroiding": False,
        "eps_space": 2,        # pixels, Manhattan distance
        "eps_time_ns": 100.0,  # nanoseconds
        "b_size": 16,          # lookback buffer depth
    }

    DEFAULT_CALLBACK_CONFIG = {
        "mode": "events",  # "events" | "pixels" | None (disabled)
    }

    DEFAULT_COMMAND_CONFIG = {
        "enabled": True,
        "port": 9100,
    }

    def __init__(
        self,
        connection_config: Optional[dict] = None,
        save_config: Optional[dict] = None,
        extract_config: Optional[dict] = None,
        callback_config: Optional[dict] = None,
        command_config: Optional[dict] = None,
        log_level: str = "INFO",
    ):
        self.log_level = log_level

        # Merge configs with defaults
        self.connection_config = {**self.DEFAULT_CONNECTION_CONFIG, **(connection_config or {})}
        self.extract_config = {**self.DEFAULT_EXTRACT_CONFIG, **(extract_config or {})}
        self.callback_config = {**self.DEFAULT_CALLBACK_CONFIG, **(callback_config or {})}
        self.command_config = {**self.DEFAULT_COMMAND_CONFIG, **(command_config or {})}

        # Save config needs special handling for nested dicts
        user_save = save_config or {}
        self.save_config = {
            "output_dir": user_save.get("output_dir", self.DEFAULT_SAVE_CONFIG["output_dir"]),
            "raw": {**self.DEFAULT_SAVE_CONFIG["raw"], **user_save.get("raw", {})},
            "events": {**self.DEFAULT_SAVE_CONFIG["events"], **user_save.get("events", {})},
            "pixels": {**self.DEFAULT_SAVE_CONFIG["pixels"], **user_save.get("pixels", {})},
            "triggers": {**self.DEFAULT_SAVE_CONFIG["triggers"], **user_save.get("triggers", {})},
        }

        self.output_dir = Path(self.save_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = get_logger("SERVAL.Pipeline")

        # Recording state (main-process only, no cross-process sharing needed)
        self._recording_flag = multiprocessing.Value('b', 0)
        self._recording_state = {
            "active": False,
            "filename": None,
            "output_dir": None,
            "active_savers": [],
        }

        # EventBus for inter-component communication
        self.event_bus = EventBus()

        # Callback queues + display-mode flag for GUI live display.
        # Both queues are created when callbacks are enabled; the shared flag tells
        # workers which one to write to, so only one queue is ever filled at a time.
        # The consumer loop reads the flag and switches queues dynamically.
        callback_mode = self.callback_config.get("mode")
        _callbacks_enabled = callback_mode is not None
        self.callback_event_queue = multiprocessing.Queue(maxsize=1000) if _callbacks_enabled else None
        self.callback_pixel_queue = multiprocessing.Queue(maxsize=1000) if _callbacks_enabled else None
        # 0 = events (TOF), 1 = pixels (TOA)
        _initial_flag = 1 if callback_mode == "pixels" else 0
        self._callback_display_flag = (
            multiprocessing.Value('i', _initial_flag) if _callbacks_enabled else None
        )
        # Fraction (0-1] of each chunk's pixels/events forwarded to the GUI
        # callback queues. Subsampling happens in the worker, before the
        # multiprocessing.Queue transfer, so the IPC payload (and its pipe-
        # transport cost) shrinks proportionally. 1.0 = no subsampling.
        _initial_fraction = self.callback_config.get("display_fraction", 1.0)
        self._callback_display_fraction = (
            multiprocessing.Value('d', _initial_fraction) if _callbacks_enabled else None
        )
        self._callback_consumer_thread = None
        self._callback_stop_event = threading.Event()

        # Raw saver processes and queues (created here, started in _start_components)
        self.raw_saver_queues = []
        self.raw_saver_processes = []
        self._init_raw_savers()

        # TCP Receiver
        conn = self.connection_config
        self.receiver = TCPReceiver(
            host=conn["host"],
            port=conn["port"],
            recv_buffer_size=conn["recv_buffer_size"],
            socket_buffer_size=conn["socket_buffer_size"],
            num_ring_buffers=conn["num_ring_buffers"],
            chunk_size=conn["chunk_size"],
            flush_timeout=conn["flush_timeout"],
            event_bus=self.event_bus,
            recording_flag=self._recording_flag,
        )

        self.stats = StatsReporter(
            event_bus=self.event_bus,
            report_interval=5.0,
        )

        # ExtractorPool handles events/pixels savers
        ext = self.extract_config
        extractor_save_config = {
            k: v for k, v in self.save_config.items() if k in ("events", "pixels", "triggers")
        }
        self.extractors = ExtractorPool(
            num_workers=ext["num_workers"],
            zmq_port=ext["zmq_port"],
            use_fast_extract=ext["use_fast_extract"],
            log_level=log_level,
            tdc_id=ext["tdc_id"],
            edge=ext["edge"],
            event_window=ext["event_window"],
            zmq_hwm=ext["zmq_hwm"],
            output_dir=self.output_dir,
            save_config=extractor_save_config,
            callback_event_queue=self.callback_event_queue,
            callback_pixel_queue=self.callback_pixel_queue,
            recording_flag=self._recording_flag,
            callback_display_flag=self._callback_display_flag,
            callback_display_fraction=self._callback_display_fraction,
            use_centroiding=ext.get("use_centroiding", False),
            eps_space=ext.get("eps_space", 2),
            eps_time_ns=ext.get("eps_time_ns", 100.0),
            b_size=ext.get("b_size", 16),
        )

        # Callbacks for events and pixels
        self._user_event_callback = None
        self._user_pixel_callback = None
        self.event_bus.subscribe(Events.EVENT_BATCH, self._on_event_batch)
        self.event_bus.subscribe(Events.PIXEL_BATCH, self._on_pixel_batch)

        # Command server (created on demand in _start_components)
        self._command_server = None

        self.running = False

    def _init_raw_savers(self):
        """Create raw saver queues and processes (idle, not yet started)."""
        raw_cfg = self.save_config["raw"]
        if not raw_cfg["enabled"] or raw_cfg.get("num_savers", 0) == 0:
            return

        num_savers = raw_cfg["num_savers"]
        for _ in range(num_savers):
            q = multiprocessing.Queue(maxsize=raw_cfg["queue_size"])
            self.raw_saver_queues.append(q)
            saver = RawSaverProcess(
                input_queue=q,
                buffer_size=raw_cfg["buffer_size"],
                log_level=self.log_level,
            )
            self.raw_saver_processes.append(saver)

        self.logger.info(f"Created {num_savers} raw saver(s) (idle)")

    # =========================================================================
    # Recording control
    # =========================================================================

    def start_record(
        self,
        filename: str,
        output_dir: Optional[str] = None,
        save_raw: bool = True,
        save_events: bool = True,
        save_pixels: bool = False,
        save_triggers: bool = True,
    ) -> bool:
        """
        Begin a recording session.

        Sends NEW_FILE to the appropriate saver queues, then enables the
        recording flag so workers start feeding data into those queues.

        Parameters
        ----------
        filename : str
            Run name. A subdirectory {output_dir}/{filename}/ is created and
            all files are placed inside it: {filename}.tpx3, {filename}_events.dat, etc.
        output_dir : str, optional
            Directory for output files. Defaults to self.output_dir.
        save_raw : bool
            Write raw .tpx3 file (requires raw saver to be enabled).
        save_events : bool
            Write correlated events .dat file (requires events saver).
        save_pixels : bool
            Write raw pixel .dat file (requires pixels saver).
        save_triggers : bool
            Write triggers .trg file (requires triggers saver).

        Returns
        -------
        bool
            True if recording started successfully.
        """
        if not self.running:
            self.logger.warning("Cannot start recording: pipeline not running")
            return False

        if self.is_recording:
            self.stop_record()

        base_dir = Path(output_dir) if output_dir else self.output_dir
        # Each recording gets its own subdirectory named after the run
        run_dir = base_dir / filename
        run_dir.mkdir(parents=True, exist_ok=True)

        active_savers = []
        file_map = {}

        # Send NEW_FILE to raw savers
        if save_raw and self.raw_saver_queues:
            for i, q in enumerate(self.raw_saver_queues):
                suffix = f"_raw{i}" if len(self.raw_saver_queues) > 1 else ""
                filepath = str(run_dir / f"{filename}{suffix}.tpx3")
                try:
                    q.put(("NEW_FILE", filepath), timeout=1.0)
                except Exception as e:
                    self.logger.error(f"Failed to open raw saver file: {e}")
            active_savers.append("raw")
            file_map["raw"] = f"{filename}.tpx3"

        # Send NEW_FILE to event savers
        if save_events and self.extractors.saver_queues.get("events"):
            event_queues = self.extractors.saver_queues["events"]
            for i, q in enumerate(event_queues):
                suffix = f"_saver{i}" if len(event_queues) > 1 else ""
                filepath = str(run_dir / f"{filename}{suffix}_events.dat")
                try:
                    q.put(("NEW_FILE", filepath), timeout=1.0)
                except Exception as e:
                    self.logger.error(f"Failed to open event saver file: {e}")
            active_savers.append("events")
            file_map["events"] = f"{filename}_events.dat"

        # Send NEW_FILE to pixel savers
        if save_pixels and self.extractors.saver_queues.get("pixels"):
            pixel_queues = self.extractors.saver_queues["pixels"]
            for i, q in enumerate(pixel_queues):
                suffix = f"_saver{i}" if len(pixel_queues) > 1 else ""
                filepath = str(run_dir / f"{filename}{suffix}_pixels.dat")
                try:
                    q.put(("NEW_FILE", filepath), timeout=1.0)
                except Exception as e:
                    self.logger.error(f"Failed to open pixel saver file: {e}")
            active_savers.append("pixels")
            file_map["pixels"] = f"{filename}_pixels.dat"

        # Send NEW_FILE to trigger savers
        if save_triggers and self.extractors.saver_queues.get("triggers"):
            trigger_queues = self.extractors.saver_queues["triggers"]
            for i, q in enumerate(trigger_queues):
                suffix = f"_saver{i}" if len(trigger_queues) > 1 else ""
                filepath = str(run_dir / f"{filename}{suffix}_triggers.trg")
                try:
                    q.put(("NEW_FILE", filepath), timeout=1.0)
                except Exception as e:
                    self.logger.error(f"Failed to open trigger saver file: {e}")
            active_savers.append("triggers")
            file_map["triggers"] = f"{filename}_triggers.trg"

        # Enable recording flag — workers start feeding saver queues
        self._recording_flag.value = 1
        self._recording_state = {
            "active": True,
            "filename": filename,
            "output_dir": str(run_dir),
            "active_savers": active_savers,
        }

        # Write metadata file
        self._write_metadata(filename, run_dir, file_map)

        self.logger.info(f"Recording started: {filename} (savers: {active_savers})")
        return True

    def _write_metadata(self, filename: str, base_dir: Path, file_map: dict):
        """Write run metadata JSON file."""
        ext = self.extract_config
        meta = {
            "format_version": "2.0",
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tdc_id": int(ext["tdc_id"]),
            "tdc_label": TDCChannel(ext["tdc_id"]).label,
            "edge": int(ext["edge"]),
            "edge_label": TriggerEdge(ext["edge"]).label,
            "event_window_ns": list(ext["event_window"]),
            "num_workers": ext["num_workers"],
            "centroiding": {
                "enabled": ext.get("use_centroiding", False),
                "eps_space_px": ext.get("eps_space", 2),
                "eps_time_ns": ext.get("eps_time_ns", 100.0),
                "b_size": ext.get("b_size", 16),
            },
            "files": {
                "events": file_map.get("events"),
                "triggers": file_map.get("triggers"),
                "pixels": file_map.get("pixels"),
                "raw": file_map.get("raw"),
            },
            "event_dtype": "t_trigger(f8),x(u2),y(u2),tof(f8),tot(u4)",
            "trigger_dtype": "toa(f8),tdc_id(u1),edge(u1)",
        }
        meta_path = base_dir / f"{filename}_meta.json"
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to write metadata: {e}")

    def stop_record(self) -> bool:
        """
        End the current recording session.

        Sets the recording flag to 0 first (stops workers from feeding queues),
        then sends CLOSE_FILE to savers so they flush and close cleanly.

        Returns
        -------
        bool
            True if a recording was active and has been stopped.
        """
        if not self.is_recording:
            return False

        # Stop feeding queues first
        self._recording_flag.value = 0

        active_savers = self._recording_state.get("active_savers", [])

        if "raw" in active_savers:
            for q in self.raw_saver_queues:
                try:
                    q.put(("CLOSE_FILE",), timeout=1.0)
                except Exception as e:
                    self.logger.error(f"Error closing raw saver: {e}", exc_info=True)
                    pass

        if "events" in active_savers:
            for q in self.extractors.saver_queues.get("events", []):
                try:
                    q.put(("CLOSE_FILE",), timeout=1.0)
                except Exception as e:
                    self.logger.error(f"Error closing event saver: {e}", exc_info=True)
                    pass

        if "pixels" in active_savers:
            for q in self.extractors.saver_queues.get("pixels", []):
                try:
                    q.put(("CLOSE_FILE",), timeout=1.0)
                except Exception as e:
                    self.logger.error(f"Error closing pixel saver: {e}", exc_info=True)
                    pass

        if "triggers" in active_savers:
            for q in self.extractors.saver_queues.get("triggers", []):
                try:
                    q.put(("CLOSE_FILE",), timeout=1.0)
                except Exception as e:
                    self.logger.error(f"Error closing trigger saver: {e}", exc_info=True)
                    pass

        # Wait for all active savers to flush and close their files (max 10 s)
        waiters = []
        if "raw" in active_savers:
            waiters.extend(self.raw_saver_processes)
        for stype in ("events", "pixels", "triggers"):
            if stype in active_savers:
                waiters.extend(self.extractors.saver_processes.get(stype, []))

        deadline = time.time() + 10.0
        for saver in waiters:
            remaining = max(0.0, deadline - time.time())
            if not saver._file_closed_event.wait(timeout=remaining):
                self.logger.warning(f"{saver.name}: timed out waiting for file close")

        filename = self._recording_state.get("filename")
        self._recording_state = {
            "active": False,
            "filename": None,
            "output_dir": None,
            "active_savers": [],
        }

        self.logger.info(f"Recording stopped: {filename}")
        return True

    @property
    def is_recording(self) -> bool:
        """True if a recording session is active."""
        return bool(self._recording_state.get("active", False))

    # =========================================================================
    # Pipeline lifecycle
    # =========================================================================

    def _on_event_batch(self, _event, t_trigger, x, y, tof, tot, t_sent):
        """Forward event batches to user callback if registered."""
        if self._user_event_callback:
            try:
                self._user_event_callback(t_trigger, x, y, tof, tot, t_sent)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")

    def _on_pixel_batch(self, _event, x, y, toa, tot, t_sent):
        """Forward pixel batches to user callback if registered."""
        if self._user_pixel_callback:
            try:
                self._user_pixel_callback(x, y, toa, tot, t_sent)
            except Exception as e:
                self.logger.error(f"Pixel callback error: {e}")

    def _callback_consumer_loop(self):
        """Consumer thread: reads from the active callback queue and publishes to EventBus.

        The active queue is determined by ``_callback_display_flag`` (0=events, 1=pixels).
        On a mode change the newly-active queue is drained first to discard any data
        produced before the switch, preventing stale data from reaching the GUI.
        """
        import queue as _queue

        _prev_mode = -1  # sentinel — forces initial drain on first iteration

        # Periodic timing summary (DEBUG): how much of this thread's time goes to
        # blocking on the queue (get) vs CPU work (publish -> user callback,
        # which does np.concatenate + Qt signal emit and holds the GIL).
        _n = 0
        _get_time = 0.0
        _get_cpu_time = 0.0
        _publish_time = 0.0
        _last_log = time.monotonic()

        while not self._callback_stop_event.is_set():
            flag = self._callback_display_flag
            mode = flag.value if flag is not None else 0

            # On mode change: flush the newly-active queue to discard pre-switch data
            if mode != _prev_mode:
                _prev_mode = mode
                flush_q = (self.callback_event_queue if mode == 0
                           else self.callback_pixel_queue)
                if flush_q is not None:
                    while True:
                        try:
                            flush_q.get_nowait()
                        except Exception:
                            break

            active_q = (self.callback_event_queue if mode == 0
                        else self.callback_pixel_queue)
            if active_q is None:
                time.sleep(0.05)
                continue

            try:
                t0 = time.perf_counter()
                c0 = time.thread_time()
                data = active_q.get(timeout=0.05)
                t1 = time.perf_counter()
                c1 = time.thread_time()
                if data is None:
                    continue
                if mode == 0:
                    t_trigger, x, y, tof, tot, t_sent = data
                    self.event_bus.publish(Events.EVENT_BATCH, t_trigger, x, y, tof, tot, t_sent)
                else:
                    x, y, toa, tot, t_sent = data
                    self.event_bus.publish(Events.PIXEL_BATCH, x, y, toa, tot, t_sent)
                t2 = time.perf_counter()

                _n += 1
                _get_time += (t1 - t0)
                _get_cpu_time += (c1 - c0)
                _publish_time += (t2 - t1)
                now = time.monotonic()
                if now - _last_log >= 2.0:
                    try:
                        qsize = active_q.qsize()
                    except Exception:
                        qsize = -1
                    self.logger.debug(
                        f"callback consumer: {_n} items/{now - _last_log:.2f}s, "
                        f"get wall={_get_time*1000:.1f} ms (cpu={_get_cpu_time*1000:.1f} ms), "
                        f"publish={_publish_time*1000:.1f} ms, queue={qsize}")
                    _n = 0
                    _get_time = 0.0
                    _get_cpu_time = 0.0
                    _publish_time = 0.0
                    _last_log = now
            except _queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Callback consumer error: {e}")

    def set_event_callback(self, callback):
        """Set callback for receiving event batches: callback(t_trigger, x, y, tof, tot, t_sent)

        ``t_sent`` is the ``time.monotonic()`` value recorded by the extractor worker
        when the batch was queued, for end-to-end latency measurement.
        """
        self._user_event_callback = callback

    def set_pixel_callback(self, callback):
        """Set callback for receiving pixel batches: callback(x, y, toa, tot, t_sent)

        ``t_sent`` is the ``time.monotonic()`` value recorded by the extractor worker
        when the batch was queued, for end-to-end latency measurement.
        """
        self._user_pixel_callback = callback

    def set_status_callback(self, callback):
        """Set callback for status updates."""
        self.stats.status_callback = callback

    def set_callback_display_mode(self, mode: str):
        """Switch the live-display queue without restarting the pipeline.

        Parameters
        ----------
        mode : str
            ``'events'`` — workers write to the event (TOF) callback queue.
            ``'pixels'`` — workers write to the pixel (TOA) callback queue.
        """
        if self._callback_display_flag is not None:
            self._callback_display_flag.value = 0 if mode == 'events' else 1

    def set_callback_display_fraction(self, fraction: float):
        """Set the fraction of pixels/events forwarded to the GUI callback queues.

        Workers stride-subsample each chunk to roughly ``fraction`` of its
        original size before queuing it for the GUI, shrinking the
        multiprocessing.Queue IPC payload proportionally. Saved data is
        always full-resolution and unaffected. Live-updatable without
        restarting the pipeline.

        Parameters
        ----------
        fraction : float
            Value in (0, 1]. 1.0 = no subsampling.
        """
        if self._callback_display_fraction is not None:
            self._callback_display_fraction.value = max(0.01, min(1.0, fraction))

    @property
    def is_connected(self):
        """Check if TCP connection is active."""
        return self.receiver.is_connected

    def get_extraction_backlog(self) -> int:
        """Estimate chunks queued in ZMQ awaiting extraction (sent - pulled by workers)."""
        chunks_sent = self.stats.get_stats().get('chunks_sent', 0)
        return max(0, chunks_sent - self.extractors.get_total_processed())

    def get_callback_queue_size(self):
        """Return (size, maxsize) of the active GUI callback queue, or None if disabled.

        This is the multiprocessing.Queue between extractor workers and the
        ``_callback_consumer_loop`` thread. A growing size here (while the
        extraction backlog stays low) means the consumer/GUI side can't keep
        up draining batches — a common source of increasing display latency.
        """
        flag = self._callback_display_flag
        mode = flag.value if flag is not None else 0
        cb_queue = self.callback_event_queue if mode == 0 else self.callback_pixel_queue
        if cb_queue is None:
            return None
        try:
            return (cb_queue.qsize(), cb_queue._maxsize)
        except Exception:
            return (0, 0)

    def start(
        self,
        duration: Optional[float] = None,
        stop_event: Optional[threading.Event] = None,
        run_name: Optional[str] = None,
        auto_record: bool = False,
        ready_callback=None,
    ):
        """
        Start the pipeline (blocking until stopped).

        Parameters
        ----------
        duration : float, optional
            Run for this many seconds then stop.
        stop_event : threading.Event, optional
            External stop signal.
        run_name : str, optional
            Used as auto-record filename when auto_record=True.
        auto_record : bool
            If True, immediately start recording with run_name as filename.
        ready_callback : callable, optional
            Called once the TCP socket is bound and all workers are running,
            just before entering the run loop.  Safe to use from a QThread to
            emit a "pipeline ready" signal.
        """
        self._print_config()

        try:
            self._setup_components()
            self._start_components()

            if ready_callback is not None:
                ready_callback()

            if auto_record:
                record_name = run_name or datetime.now().strftime("acquisition_%Y%m%d_%H%M%S")
                self.start_record(record_name)

            self._run_loop(duration, stop_event)

        except KeyboardInterrupt:
            print("\n\nCtrl+C - stopping")
        finally:
            if self.is_recording:
                self.stop_record()
            self._shutdown()

    def _setup_components(self):
        """Bind sockets and start workers (savers already created in __init__)."""
        self.receiver.bind()

        zmq_socket = self.extractors.setup_zmq()
        self.extractors.start_workers()

        # Connect receiver to raw saver queues and ZMQ extractor socket
        self.receiver.set_targets(self.raw_saver_queues, zmq_socket)

        # Pass queue references to stats reporter (with declared maxsizes)
        raw_qs = self.save_config["raw"].get("queue_size", 100)
        ext_cfg = self.extractors.save_config
        self.stats.set_queues('raw',      self.raw_saver_queues,
                              maxsize=raw_qs)
        self.stats.set_queues('events',   self.extractors.saver_queues.get('events', []),
                              maxsize=ext_cfg['events'].get('queue_size', 1000))
        self.stats.set_queues('pixels',   self.extractors.saver_queues.get('pixels', []),
                              maxsize=ext_cfg['pixels'].get('queue_size', 1000))
        self.stats.set_queues('triggers', self.extractors.saver_queues.get('triggers', []),
                              maxsize=ext_cfg['triggers'].get('queue_size', 1000))

    def _start_components(self):
        """Start all saver processes, receiver thread, and ancillary threads."""
        self.running = True

        # Start raw saver processes
        for saver in self.raw_saver_processes:
            saver.start()

        # Start event/pixel saver processes
        self.extractors.start_savers()

        self.receiver.start()
        self.stats.start()

        # Callback consumer thread for GUI updates
        self._callback_stop_event.clear()
        self._callback_consumer_thread = threading.Thread(
            target=self._callback_consumer_loop,
            name="CallbackConsumer",
            daemon=True,
        )
        self._callback_consumer_thread.start()

        # ZMQ command server for PyMoDAQ / external control
        if self.command_config.get("enabled"):
            from SERVAL.core.command_server import CommandServer
            self._command_server = CommandServer(self, port=self.command_config["port"])
            self._command_server.start()
            self.logger.info(f"Command server listening on port {self.command_config['port']}")

    def stop(self):
        """Signal the pipeline to stop."""
        self.running = False

    def _run_loop(self, duration, stop_event):
        """Main run loop."""
        start_time = time.time()
        last_diag = start_time

        while self.running:
            if stop_event and stop_event.is_set():
                break
            if duration and (time.time() - start_time) >= duration:
                break

            if time.time() - last_diag >= 30.0:
                self._print_diagnostics()
                last_diag = time.time()

            time.sleep(0.5)

    def _print_diagnostics(self):
        """Log queue fill levels and throughput stats."""
        stats = self.stats.get_stats()
        bytes_received = stats.get("bytes_received", 0)
        elapsed = stats.get("elapsed", 1)
        mb_per_sec = (bytes_received / 1e6) / elapsed if elapsed > 0 else 0
        self.logger.info(f"Rate: {mb_per_sec:.1f} MB/s")

    def _shutdown(self):
        """Gracefully shutdown all components."""
        self.logger.info("Shutting down...")
        self.running = False

        # Stop command server
        if self._command_server:
            self._command_server.stop()
            self._command_server = None

        # Stop callback consumer thread
        self._callback_stop_event.set()
        if self._callback_consumer_thread and self._callback_consumer_thread.is_alive():
            self._callback_consumer_thread.join(timeout=2.0)

        self.receiver.stop()
        self.receiver.close()

        self.extractors.shutdown()

        # Stop raw savers
        for q in self.raw_saver_queues:
            try:
                q.put("STOP", timeout=1.0)
            except Exception as e:
                self.logger.error(f"Error stopping raw saver: {e}", exc_info=True)
                pass

        for saver in self.raw_saver_processes:
            saver.stop()
            saver.join(timeout=5.0)
            if saver.is_alive():
                saver.terminate()

        self.raw_saver_processes.clear()

        # Release semaphores held by each raw saver Queue
        for q in self.raw_saver_queues:
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass
        self.raw_saver_queues.clear()

        # Release semaphores held by callback queues
        for q in (self.callback_event_queue, self.callback_pixel_queue):
            if q is not None:
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass
        self.callback_event_queue = None
        self.callback_pixel_queue = None

        self.stats.stop()
        self.logger.info("Shutdown complete.")

    def _print_config(self):
        """Print pipeline configuration."""
        conn = self.connection_config
        ext = self.extract_config
        raw_cfg = self.save_config["raw"]
        events_cfg = self.save_config["events"]
        pixels_cfg = self.save_config["pixels"]
        triggers_cfg = self.save_config["triggers"]
        tdc_id = ext["tdc_id"]

        sep = "=" * 60
        lines = [
            sep,
            "TPX3 Pipeline V3",
            sep,
            f"Connection:  {conn['host']}:{conn['port']}",
            f"Workers:     {ext['num_workers']} ({'fast' if ext['use_fast_extract'] else 'standard'})",
        ]
        if events_cfg["enabled"]:
            lines.append(f"TDC:         {'TDC1' if tdc_id == 1 else 'TDC2' if tdc_id == 2 else 'Both'}")
            lines.append(f"Event window:{ext['event_window']} ns")
        lines += [
            "Savers (idle until start_record()):",
            f"  Output dir: {self.output_dir}",
            f"  Raw:      {raw_cfg['enabled']} ({raw_cfg['num_savers']} savers)",
            f"  Events:   {events_cfg['enabled']} ({events_cfg['num_savers']} savers)",
            f"  Pixels:   {pixels_cfg['enabled']} ({pixels_cfg['num_savers']} savers)",
            f"  Triggers: {triggers_cfg['enabled']} ({triggers_cfg['num_savers']} savers)",
            f"Callbacks:   {self.callback_config.get('mode') or 'disabled'}",
        ]
        if self.command_config.get("enabled"):
            lines.append(f"Command server: port {self.command_config['port']}")
        lines.append(sep)
        for line in lines:
            self.logger.info(line)


def main():
    from SERVAL.utils.logging import set_log_level

    set_log_level("INFO")

    pipeline = TPX3PipelineV3(
        connection_config={
            "host": "192.168.1.2",
            "port": 8088,
        },
        extract_config={
            "num_workers": 4,
            "use_fast_extract": True,
            "event_window": (0.0, 300_000.0),
            "tdc_id": 1,
            "use_centroiding": True,
            "eps_space": 2,        # pixels, Manhattan distance
            "eps_time_ns": 100.0,  # nanoseconds
            "b_size": 16,          # lookback buffer depth            
        },
        save_config={
            "output_dir": "./data",
            "raw": {"enabled": True, "num_savers": 2},
            "events": {"enabled": True, "num_savers": 2},
            "pixels": {"enabled": False, "num_savers": 0},
        },
        command_config={"enabled": True, "port": 9100},
        log_level="INFO",
    )

    # auto_record=True starts recording immediately with a timestamped filename
    pipeline.start(auto_record=True)


if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver', force=False)
    main()
