"""
Pipeline Thread

QThread wrapper for running TPX3PipelineV3 in a background thread.
"""

import sys
import threading
import traceback
from pathlib import Path

from qtpy.QtCore import QThread, Signal

# Add parent directory to path for SERVAL imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from SERVAL.core.pipeline import TPX3PipelineV3


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
        log_level : str
            Logging level
        """
        super().__init__()

        self.connection_config = connection_config or {}
        self.save_config = save_config or {}
        self.extract_config = extract_config or {}
        self.callback_config = callback_config or {"mode": "events"}
        self.log_level = log_level

        self.stop_event = threading.Event()
        self.duration = None
        self.run_name = None

        self._pipeline = None

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
                log_level=self.log_level,
            )

            # Set callbacks
            self._pipeline.set_event_callback(self._on_events)
            self._pipeline.set_pixel_callback(self._on_pixels)
            self._pipeline.set_status_callback(self._on_status)

            self.pipeline_started.emit()

            # Run pipeline (blocking)
            self._pipeline.start(
                duration=self.duration,
                stop_event=self.stop_event,
                run_name=self.run_name,
            )

        except Exception as e:
            error_msg = f"Pipeline error: {e}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)

        finally:
            self._pipeline = None
            self.pipeline_stopped.emit()

    def _on_events(self, event_num, x, y, tof, tot):
        """Callback for event data - emits Qt signal."""
        self.event_data_ready.emit(event_num, x, y, tof, tot)

    def _on_pixels(self, x, y, toa, tot):
        """Callback for pixel data - emits Qt signal."""
        self.pixel_data_ready.emit(x, y, toa, tot)

    def _on_status(self, status_dict):
        """Callback for status updates - emits Qt signal."""
        self.status_changed.emit(status_dict)

    @property
    def is_connected(self):
        """Check if pipeline TCP connection is active."""
        if self._pipeline:
            return self._pipeline.is_connected
        return False
