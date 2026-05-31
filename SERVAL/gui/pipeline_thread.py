"""
Pipeline Thread

QThread wrapper for running TPX3PipelineV3 in a background thread.
"""

import sys
import threading
import time
import traceback

import numpy as np
from qtpy.QtCore import QThread, Signal

from SERVAL.core.pipeline import TPX3PipelineV3

# Maximum interval between coalesced signal emissions (seconds).
# Batches that arrive faster than this are merged into one signal,
# capping the GUI update rate at ~1/_EMIT_INTERVAL Hz regardless of data rate.
_EMIT_INTERVAL = 0.05  # 20 Hz max


class PipelineThread(QThread):
    """
    Background thread for running TPX3PipelineV3.

    Emits Qt signals for thread-safe GUI updates.

    Signals
    -------
    event_data_ready : Signal(object, object, object, object, object)
        Emitted when events are available: (event_num, x, y, tof, tot)
    pixel_data_ready : Signal(object, object, object, object)
        Emitted when raw pixels are available: (x, y, toa, tot)
    stats_updated : Signal(dict)
        Emitted with throughput statistics
    status_changed : Signal(dict)
        Emitted when pipeline status changes
    error_occurred : Signal(str)
        Emitted on errors
    pipeline_started : Signal()
        Emitted when pipeline starts successfully
    pipeline_stopped : Signal()
        Emitted when pipeline stops
    """

    event_data_ready = Signal(object, object, object, object, object)
    pixel_data_ready = Signal(object, object, object, object)
    stats_updated = Signal(dict)
    status_changed = Signal(dict)
    error_occurred = Signal(str)
    pipeline_started = Signal()
    pipeline_stopped = Signal()

    def __init__(
        self,
        connection_config=None,
        save_config=None,
        extract_config=None,
        callback_config=None,
        command_config=None,
        log_level="INFO",
    ):
        """
        Initialize pipeline thread.

        Parameters
        ----------
        connection_config : dict, optional
            TCP receiver configuration
        save_config : dict, optional
            Save configuration for raw/events/pixels
        extract_config : dict, optional
            Extraction and correlation configuration
        callback_config : dict, optional
            Callback configuration: {"mode": "events" | "pixels" | None}
        command_config : dict, optional
            ZMQ command server configuration: {"enabled": bool, "port": int}
        log_level : str
            Logging level
        """
        super().__init__()

        self.connection_config = connection_config or {}
        self.save_config = save_config or {}
        self.extract_config = extract_config or {}
        self.callback_config = callback_config or {"mode": "events"}
        self.command_config = command_config or {}
        self.log_level = log_level

        self.stop_event = threading.Event()
        self.duration = None
        self.run_name = None

        self._pipeline = None

        # Coalescing accumulators — batches are merged here between emissions.
        # Accessed only from the consumer thread, so no lock required.
        self._event_acc: list = []
        self._pixel_acc: list = []
        self._last_event_emit: float = 0.0
        self._last_pixel_emit: float = 0.0

    def set_duration(self, duration):
        """Set acquisition duration (None = run until stopped)."""
        self.duration = duration

    def set_run_name(self, run_name):
        """Set custom run directory name."""
        self.run_name = run_name

    def request_stop(self):
        """Request graceful pipeline stop."""
        self.stop_event.set()
        if self._pipeline:
            self._pipeline.stop()

    def run(self):
        """Run pipeline in background thread."""
        try:
            self.stop_event.clear()

            # Create pipeline
            self._pipeline = TPX3PipelineV3(
                connection_config=self.connection_config,
                save_config=self.save_config,
                extract_config=self.extract_config,
                callback_config=self.callback_config,
                command_config=self.command_config,
                log_level=self.log_level,
            )

            # Set callbacks
            self._pipeline.set_event_callback(self._on_events)
            self._pipeline.set_pixel_callback(self._on_pixels)
            self._pipeline.set_status_callback(self._on_status)

            # Run pipeline (blocking); emit pipeline_started once TCP is bound
            self._pipeline.start(
                duration=self.duration,
                stop_event=self.stop_event,
                run_name=self.run_name,
                ready_callback=self.pipeline_started.emit,
            )

        except Exception as e:
            error_msg = f"Pipeline error: {e}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)

        finally:
            self._pipeline = None
            self.pipeline_stopped.emit()

    def _on_events(self, event_num, x, y, tof, tot):
        """Accumulate event batch; emit coalesced signal at most every _EMIT_INTERVAL s."""
        self._event_acc.append((event_num, x, y, tof, tot))
        now = time.monotonic()
        if now - self._last_event_emit >= _EMIT_INTERVAL:
            self._last_event_emit = now
            if len(self._event_acc) == 1:
                en, x_, y_, tof_, tot_ = self._event_acc[0]
            else:
                en   = np.concatenate([b[0] for b in self._event_acc])
                x_   = np.concatenate([b[1] for b in self._event_acc])
                y_   = np.concatenate([b[2] for b in self._event_acc])
                tof_ = np.concatenate([b[3] for b in self._event_acc])
                tot_ = np.concatenate([b[4] for b in self._event_acc])
            self._event_acc.clear()
            self.event_data_ready.emit(en, x_, y_, tof_, tot_)

    def _on_pixels(self, x, y, toa, tot):
        """Accumulate pixel batch; emit coalesced signal at most every _EMIT_INTERVAL s."""
        self._pixel_acc.append((x, y, toa, tot))
        now = time.monotonic()
        if now - self._last_pixel_emit >= _EMIT_INTERVAL:
            self._last_pixel_emit = now
            if len(self._pixel_acc) == 1:
                x_, y_, toa_, tot_ = self._pixel_acc[0]
            else:
                x_   = np.concatenate([b[0] for b in self._pixel_acc])
                y_   = np.concatenate([b[1] for b in self._pixel_acc])
                toa_ = np.concatenate([b[2] for b in self._pixel_acc])
                tot_ = np.concatenate([b[3] for b in self._pixel_acc])
            self._pixel_acc.clear()
            self.pixel_data_ready.emit(x_, y_, toa_, tot_)

    def _on_status(self, status_dict):
        """Callback for status updates - emits Qt signal."""
        self.status_changed.emit(status_dict)

    def set_display_mode(self, mode: str):
        """Switch live display between 'events' (TOF) and 'pixels' (TOA).

        Forwards the request to the pipeline's shared flag so workers immediately
        stop writing to the old queue and start writing to the new one.
        Clears the local coalescing accumulators to prevent stale data leaking
        into the next emitted signal.
        """
        pipeline = self._pipeline
        if pipeline is not None:
            pipeline.set_callback_display_mode(mode)
        self._event_acc.clear()
        self._pixel_acc.clear()

    def start_record(self, filename, save_raw=True, save_events=True,
                     save_pixels=False, save_triggers=True):
        """Start a recording session. Returns True on success."""
        if self._pipeline is None:
            return False
        return self._pipeline.start_record(
            filename=filename,
            save_raw=save_raw,
            save_events=save_events,
            save_pixels=save_pixels,
            save_triggers=save_triggers,
        )

    def stop_record(self):
        """Stop the active recording session."""
        if self._pipeline is not None:
            self._pipeline.stop_record()

    @property
    def is_connected(self):
        """Check if pipeline TCP connection is active."""
        if self._pipeline:
            return self._pipeline.is_connected
        return False

    def get_live_status(self):
        """Return current queue fill levels and worker alive states (thread-safe).

        Returns
        -------
        dict or None
            ``{'queues': {type: [(size, maxsize), ...]}, 'workers': [(name, alive), ...]}``
            or *None* when the pipeline is not running.
        """
        pipeline = self._pipeline  # atomic attribute read
        if pipeline is None:
            return None
        try:
            queues = {}
            for qtype, qlist in pipeline.stats.queues.items():
                maxsize = pipeline.stats._queue_maxsizes.get(qtype, 0)
                if qlist:
                    try:
                        queues[qtype] = [(q.qsize(), maxsize) for q in qlist]
                    except Exception:
                        queues[qtype] = [(0, maxsize) for _ in qlist]
                else:
                    queues[qtype] = None

            workers = []
            if pipeline.extractors and pipeline.extractors.workers:
                for w in pipeline.extractors.workers:
                    workers.append((w.name, w.is_alive()))

            return {'queues': queues, 'workers': workers}
        except Exception:
            return None
