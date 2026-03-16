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

import multiprocessing
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from SERVAL.utils.logging import get_logger
from SERVAL.utils import EventBus, Events
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
    }

    DEFAULT_EXTRACT_CONFIG = {
        "num_workers": 4,
        "use_fast_extract": False,
        # "zmq_port": 9001,
        "zmq_port": 9200,
        "zmq_hwm": 1000,
        "event_window": (0.0, 10_000.0),  # ns
        "tdc_id": 1,  # 1=TDC1, 2=TDC2, 0=both
        "events": True,
        "pixels": True
    }

    DEFAULT_CALLBACK_CONFIG = {
        "mode": "events",  # "events" | "pixels" | None (disabled)
    }

    DEFAULT_COMMAND_CONFIG = {
        "enabled": False,
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

        # Callback queue for GUI/live display (workers → main process)
        callback_mode = self.callback_config.get("mode")
        self.callback_event_queue = multiprocessing.Queue(maxsize=1000) if callback_mode == "events" else None
        self.callback_pixel_queue = multiprocessing.Queue(maxsize=1000) if callback_mode == "pixels" else None
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
            k: v for k, v in self.save_config.items() if k in ("events", "pixels")
        }
        self.extractors = ExtractorPool(
            num_workers=ext["num_workers"],
            zmq_port=ext["zmq_port"],
            use_fast_extract=ext["use_fast_extract"],
            log_level=log_level,
            tdc_id=ext["tdc_id"],
            event_window=ext["event_window"],
            zmq_hwm=ext["zmq_hwm"],
            output_dir=self.output_dir,
            save_config=extractor_save_config,
            callback_event_queue=self.callback_event_queue,
            callback_pixel_queue=self.callback_pixel_queue,
            recording_flag=self._recording_flag,
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
    ) -> bool:
        """
        Begin a recording session.

        Sends NEW_FILE to the appropriate saver queues, then enables the
        recording flag so workers start feeding data into those queues.

        Parameters
        ----------
        filename : str
            Base filename (without extension). Files will be created as
            {output_dir}/{filename}.tpx3, {filename}_events.dat, etc.
        output_dir : str, optional
            Directory for output files. Defaults to self.output_dir.
        save_raw : bool
            Write raw .tpx3 file (requires raw saver to be enabled).
        save_events : bool
            Write correlated events .dat file (requires events saver).
        save_pixels : bool
            Write raw pixel .dat file (requires pixels saver).

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
        base_dir.mkdir(parents=True, exist_ok=True)

        active_savers = []

        # Send NEW_FILE to raw savers
        if save_raw and self.raw_saver_queues:
            for i, q in enumerate(self.raw_saver_queues):
                suffix = f"_raw{i}" if len(self.raw_saver_queues) > 1 else ""
                filepath = str(base_dir / f"{filename}{suffix}.tpx3")
                try:
                    q.put(("NEW_FILE", filepath), timeout=1.0)
                except Exception as e:
                    self.logger.error(f"Failed to open raw saver file: {e}")
            active_savers.append("raw")

        # Send NEW_FILE to event savers
        if save_events and self.extractors.saver_queues.get("events"):
            event_queues = self.extractors.saver_queues["events"]
            for i, q in enumerate(event_queues):
                suffix = f"_saver{i}" if len(event_queues) > 1 else ""
                filepath = str(base_dir / f"{filename}{suffix}_events.dat")
                try:
                    q.put(("NEW_FILE", filepath), timeout=1.0)
                except Exception as e:
                    self.logger.error(f"Failed to open event saver file: {e}")
            active_savers.append("events")

        # Send NEW_FILE to pixel savers
        if save_pixels and self.extractors.saver_queues.get("pixels"):
            pixel_queues = self.extractors.saver_queues["pixels"]
            for i, q in enumerate(pixel_queues):
                suffix = f"_saver{i}" if len(pixel_queues) > 1 else ""
                filepath = str(base_dir / f"{filename}{suffix}_pixels.dat")
                try:
                    q.put(("NEW_FILE", filepath), timeout=1.0)
                except Exception as e:
                    self.logger.error(f"Failed to open pixel saver file: {e}")
            active_savers.append("pixels")

        # Enable recording flag — workers start feeding saver queues
        self._recording_flag.value = 1
        self._recording_state = {
            "active": True,
            "filename": filename,
            "output_dir": str(base_dir),
            "active_savers": active_savers,
        }

        self.logger.info(f"Recording started: {filename} (savers: {active_savers})")
        return True

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

    def _on_event_batch(self, _event, event_num, x, y, tof, tot):
        """Forward event batches to user callback if registered."""
        if self._user_event_callback:
            try:
                self._user_event_callback(event_num, x, y, tof, tot)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")

    def _on_pixel_batch(self, _event, x, y, toa, tot):
        """Forward pixel batches to user callback if registered."""
        if self._user_pixel_callback:
            try:
                self._user_pixel_callback(x, y, toa, tot)
            except Exception as e:
                self.logger.error(f"Pixel callback error: {e}")

    def _callback_consumer_loop(self):
        """Consumer thread that reads from callback queue and publishes to EventBus."""
        import queue

        while not self._callback_stop_event.is_set():
            if self.callback_event_queue is not None:
                try:
                    event_data = self.callback_event_queue.get(timeout=0.05)
                    if event_data is not None:
                        event_num, x, y, tof, tot = event_data
                        self.event_bus.publish(Events.EVENT_BATCH, event_num, x, y, tof, tot)
                except queue.Empty:
                    pass
                except Exception as e:
                    self.logger.error(f"Callback consumer error (events): {e}")

            elif self.callback_pixel_queue is not None:
                try:
                    pixel_data = self.callback_pixel_queue.get(timeout=0.05)
                    if pixel_data is not None:
                        x, y, toa, tot = pixel_data
                        self.event_bus.publish(Events.PIXEL_BATCH, x, y, toa, tot)
                except queue.Empty:
                    pass
                except Exception as e:
                    self.logger.error(f"Callback consumer error (pixels): {e}")
            else:
                time.sleep(0.05)

    def set_event_callback(self, callback):
        """Set callback for receiving event batches: callback(event_num, x, y, tof, tot)"""
        self._user_event_callback = callback

    def set_pixel_callback(self, callback):
        """Set callback for receiving pixel batches: callback(x, y, toa, tot)"""
        self._user_pixel_callback = callback

    def set_status_callback(self, callback):
        """Set callback for status updates."""
        self.stats.status_callback = callback

    @property
    def is_connected(self):
        """Check if TCP connection is active."""
        return self.receiver.is_connected

    def start(
        self,
        duration: Optional[float] = None,
        stop_event: Optional[threading.Event] = None,
        run_name: Optional[str] = None,
        auto_record: bool = False,
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
        """
        self._print_config()

        try:
            self._setup_components()
            self._start_components()

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

        # Pass queue references to stats reporter
        self.stats.set_queues('raw', self.raw_saver_queues)
        self.stats.set_queues('events', self.extractors.saver_queues.get('events', []))
        self.stats.set_queues('pixels', self.extractors.saver_queues.get('pixels', []))

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
        print("\nShutting down...")
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
        self.raw_saver_queues.clear()

        self.stats.stop()
        print("Shutdown complete.")

    def _print_config(self):
        """Print pipeline configuration."""
        conn = self.connection_config
        ext = self.extract_config
        raw_cfg = self.save_config["raw"]
        events_cfg = self.save_config["events"]
        pixels_cfg = self.save_config["pixels"]
        tdc_id = ext["tdc_id"]

        print(f"\n{'=' * 60}")
        print("TPX3 Pipeline V3")
        print(f"{'=' * 60}")
        print("Connection:")
        print(f"  Host: {conn['host']}:{conn['port']}")
        print("Extraction:")
        print(f"  Workers: {ext['num_workers']} ({'fast' if ext['use_fast_extract'] else 'standard'})")
        if events_cfg["enabled"]:
            print(f"  TDC: {'TDC1' if tdc_id == 1 else 'TDC2' if tdc_id == 2 else 'Both'}")
            print(f"  Event window: {ext['event_window']} ns")
        print("Savers (idle until start_record()):")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Raw: {raw_cfg['enabled']} ({raw_cfg['num_savers']} savers)")
        print(f"  Events: {events_cfg['enabled']} ({events_cfg['num_savers']} savers)")
        print(f"  Pixels: {pixels_cfg['enabled']} ({pixels_cfg['num_savers']} savers)")
        callback_mode = self.callback_config.get("mode")
        print(f"Callbacks: {callback_mode or 'disabled'}")
        if self.command_config.get("enabled"):
            print(f"Command server: port {self.command_config['port']}")
        print(f"{'=' * 60}\n")


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
    main()
