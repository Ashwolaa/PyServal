"""
_PipelineMixin — acquisition lifecycle for ServalAcquisitionGUI.

Mixed into ServalAcquisitionGUI; all methods access GUI state via ``self``.
Covers pipeline config builders, start/stop, recording control, and error
handling.
"""

import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from qtpy.QtWidgets import QMessageBox

from SERVAL.core.data_types import TDCChannel, TriggerEdge
from SERVAL.gui.pipeline_thread import PipelineThread
from SERVAL.utils.logging import enable_file_logging, disable_file_logging


def _increment_filename(name: str) -> str:
    """Bump the trailing run of digits in *name* by one, preserving zero-padding.

    e.g. 'run_007' -> 'run_008', 'scan42' -> 'scan43', 'run' -> 'run_1'.
    """
    m = re.search(r'(\d+)$', name)
    if m:
        next_num = str(int(m.group(1)) + 1).zfill(len(m.group(1)))
        return name[:m.start()] + next_num
    return f'{name}_1'


def _set_group_enabled(param, enabled: bool):
    """Recursively enable/disable every leaf parameter under *param*."""
    for child in param.children():
        if child.hasChildren():
            _set_group_enabled(child, enabled)
        else:
            child.setOpts(enabled=enabled)


class _PipelineMixin:
    """Acquisition start/stop, recording control, and config dict builders."""

    # =========================================================================
    # Config dict builders
    # =========================================================================

    def _build_connection_config(self):
        s   = self.settings.child('serval', 'serval_destination')
        adv = self.settings.child('pipeline', 'processing', 'advanced')
        return {
            'host': s['dest_host'],
            'port': s['dest_port'],
            'triggers_per_chunk': adv['triggers_per_chunk'],
            'chunk_size': adv['chunk_size'],
            'flush_timeout': adv['flush_timeout'],
            'recv_buffer_size': adv['recv_buffer_mb'] * 1024 * 1024,
        }

    def _build_extract_config(self):
        p   = self.settings.child('pipeline', 'processing')
        cor = self.settings.child('pipeline', 'correlation')
        c   = p.child('centroiding')
        adv = p.child('advanced')
        return {
            'num_workers':      p['num_workers'],
            'use_fast_extract': p['use_fast_extract'],
            'tdc_id':           TDCChannel.from_label(cor['tdc_id']),
            'edge':             TriggerEdge.from_label(cor['edge']),
            'event_window':     (cor['event_window_min'], cor['event_window_max']),
            'use_centroiding':  c['use_centroiding'],
            'eps_space':        c['eps_space'],
            'eps_time_ns':      c['eps_time_ns'],
            'zmq_port':         adv['zmq_port'],
            'zmq_hwm':          adv['zmq_hwm'],
            # Column correction — populated by _start_acquisition from SERVAL chip config
            'adjusted_columns': getattr(self, '_chip_adjust_columns', []),
            'chip_config':      getattr(self, '_chip_config', {}),
            # Full /detector tree, archived verbatim into run metadata
            'detector_info':    getattr(self, '_detector_info', {}),
        }

    def _build_save_config(self):
        p   = self.settings.child('pipeline', 'saving')
        adv = p.child('advanced')
        return {
            'output_dir': p['output_dir'],
            'raw':      {'enabled': p['save_raw'],     'num_savers': adv['raw_num_savers']},
            'events':   {'enabled': p['save_events'],  'num_savers': adv['events_num_savers']},
            'pixels':   {'enabled': p['save_pixels'],  'num_savers': adv['pixels_num_savers']},
            'triggers': {'enabled': p['save_triggers'], 'num_savers': adv['triggers_num_savers']},
        }

    def _build_command_config(self):
        s = self.settings.child('pipeline', 'external_control', 'command_server')
        return {'enabled': s['cmd_enabled'], 'port': s['cmd_port']}

    # =========================================================================
    # Acquisition Control
    # =========================================================================

    def _on_run_clicked(self, checked: bool):
        if checked:
            self._start_acquisition()
        else:
            self._on_stop_clicked()

    def _start_acquisition(self):
        if self.is_acquiring:
            return
        if not self.serval.is_connected:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to SERVAL first.")
            self.get_action('run').setChecked(False)
            return

        self.histogram.clear()
        self.histogram.clear_timeseries()

        # Fetch chip configuration for column timing correction.
        # Done here (before workers start) so adjusted_columns is available at
        # ExtractorWorker construction time.  A failure is non-fatal: we proceed
        # with no correction and log a warning.
        self._chip_config = {}
        self._chip_adjust_columns = []
        try:
            chip_cfg = self.serval.get_chip_config(chip_id=0)
            if chip_cfg:
                self._chip_config = chip_cfg
                self._chip_adjust_columns = self.serval.get_adjusted_columns(chip_id=0)
                if self._chip_adjust_columns:
                    self._log(
                        f"Column correction: {len(self._chip_adjust_columns)} adjusted "
                        f"double-column(s): {self._chip_adjust_columns}"
                    )
                else:
                    self._log("Column correction: no adjusted double-columns reported by SERVAL")
        except Exception as e:
            self._log(f"Could not fetch chip config (column correction disabled): {e}", level=30)

        # Fetch the full /detector info tree, archived verbatim into run metadata.
        self._detector_info = {}
        try:
            self._detector_info = self.serval.get_detector_info()
        except Exception as e:
            self._log(f"Could not fetch detector info (metadata will omit it): {e}", level=30)

        callback_mode = self.settings.child('pipeline', 'live', 'callback_mode').value()
        display_fraction = self.settings.child('display_settings', 'live_feed', 'display_fraction').value() / 100.0
        self.histogram.set_display_fraction(display_fraction)

        self.pipeline_thread = PipelineThread(
            connection_config=self._build_connection_config(),
            save_config=self._build_save_config(),
            extract_config=self._build_extract_config(),
            callback_config={
                'mode': callback_mode if callback_mode != 'disabled' else None,
                'display_fraction': display_fraction,
            },
            command_config=self._build_command_config(),
        )
        self.pipeline_thread.event_data_ready.connect(self._on_event_data)
        self.pipeline_thread.pixel_data_ready.connect(self._on_pixel_data)
        self.pipeline_thread.pipeline_started.connect(self._on_pipeline_started)
        self.pipeline_thread.pipeline_stopped.connect(self._on_pipeline_stopped)
        self.pipeline_thread.error_occurred.connect(self._on_pipeline_error)
        self.pipeline_thread.status_changed.connect(self._on_status_update)

        self.pipeline_thread.start()
        self._log("Starting pipeline...")
        self._set_pipeline_controls_enabled(False)

    def _on_pipeline_started(self):
        self._log("TCP socket bound — configuring SERVAL...")
        try:
            p = self.settings.child('pipeline', 'saving')
            log_path = Path(p['output_dir']) / 'serval.log'
            enable_file_logging(log_path)
            self._log(f"Logging to file: {log_path}")
        except Exception as e:
            self._log(f"Could not start file logging: {e}", level=30)
        threading.Thread(target=self._configure_and_start_serval, daemon=True).start()

    def _configure_and_start_serval(self):
        """HTTP configuration — runs in a background thread."""
        s = self.settings.child('serval', 'serval_destination')
        host = s['dest_host']
        port = s['dest_port']

        self._log(f"Setting SERVAL destination → {host}:{port}")
        if not self.serval.set_destination(host, port):
            self._serval_sig.failed.emit("Failed to set SERVAL destination")
            self.pipeline_thread.request_stop()
            return

        self._log("Starting SERVAL measurement...")
        if not self.serval.start_measurement():
            self._serval_sig.failed.emit("Failed to start SERVAL measurement")
            self.pipeline_thread.request_stop()
            return

        self._serval_sig.started.emit()

    def _on_serval_started(self):
        """Called on the main thread after SERVAL HTTP configuration succeeds."""
        self.is_acquiring = True
        self.record_btn.setEnabled(self.serval.is_connected)
        self._set_led(self.led_acquiring, True)
        self._last_clear_time = time.time()
        refresh_rate_s = self.settings.child('display_settings', 'live_feed', 'refresh_rate_s').value()
        if refresh_rate_s > 0:
            self.refresh_timer.start(int(refresh_rate_s * 1000))
        callback_mode = self.settings.child('pipeline', 'live', 'callback_mode').value()
        if callback_mode == 'disabled':
            callback_mode = 'events'
        self._display_mode = callback_mode
        self._apply_display_mode(callback_mode)
        self._log("Acquisition running")

    def _validate_output_dir(self, path: Path) -> bool:
        if not path.exists():
            reply = QMessageBox.question(
                self, "Output Directory",
                f"Directory does not exist:\n{path}\n\nCreate it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if reply == QMessageBox.No:
                return False
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                QMessageBox.critical(self, "Error", f"Could not create directory:\n{e}")
                return False
        if not os.access(path, os.W_OK):
            QMessageBox.critical(self, "Error",
                                 f"Output directory is not writable:\n{path}")
            return False
        return True

    def _set_record_btn_state(self, recording: bool):
        if recording:
            self.record_btn.setText("Stop")
            self.record_btn.setStyleSheet(
                "QPushButton { font-weight: bold; padding: 6px; }"
                "QPushButton:enabled { background-color: #f44336; color: white; }")
        else:
            self.record_btn.setText("Save")
            self.record_btn.setStyleSheet(
                "QPushButton { font-weight: bold; padding: 6px; }"
                "QPushButton:enabled { background-color: #2196F3; color: white; }")

    def _on_save_clicked(self):
        if not self.is_acquiring or self.pipeline_thread is None:
            return
        if self._is_recording:
            self._save_timer.stop()
            self._stop_save()
        else:
            p = self.settings.child('pipeline', 'saving')
            output_dir = Path(p['output_dir'])
            if not self._validate_output_dir(output_dir):
                return
            filename = self.record_filename_edit.text().strip()
            if not filename:
                filename = datetime.now().strftime("rec_%Y%m%d_%H%M%S")
            ok = self.pipeline_thread.start_record(
                filename=filename,
                save_raw=p['save_raw'],
                save_events=p['save_events'],
                save_pixels=p['save_pixels'],
                save_triggers=p['save_triggers'],
            )
            if ok:
                self._last_record_filename = filename
                self._is_recording = True
                self._record_start_time = time.monotonic()
                self._set_record_btn_state(True)
                self._set_led(self.led_recording, True, color='#ff4444')
                self.record_filename_edit.setEnabled(False)
                self.save_duration_spin.setEnabled(False)
                self._log(f"Recording started: {filename}")
                duration = self.save_duration_spin.value()
                if duration > 0:
                    self._save_timer.start(int(duration * 1000))

    def _auto_stop_save(self):
        if self._is_recording:
            self._stop_save()

    def _stop_save(self):
        if self.pipeline_thread is not None:
            self.pipeline_thread.stop_record()
        self._is_recording = False
        self._record_start_time = None
        self.record_time_label.setText("")
        self._set_record_btn_state(False)
        self._set_led(self.led_recording, False)
        self._prefill_next_record_filename()
        self.record_filename_edit.setEnabled(True)
        self.save_duration_spin.setEnabled(True)
        self._log("Recording stopped")

    def _prefill_next_record_filename(self):
        """After a recording finishes, either clear the name field or, if
        auto-increment is enabled, prefill it with the previous name's
        trailing number bumped by one."""
        if self.auto_increment_check.isChecked() and self._last_record_filename:
            self.record_filename_edit.setText(
                _increment_filename(self._last_record_filename))
        else:
            self.record_filename_edit.clear()

    def _on_stop_clicked(self):
        if not self.is_acquiring:
            return
        if self._is_recording:
            self._save_timer.stop()
            self.pipeline_thread.stop_record()
            self._is_recording = False
            self._record_start_time = None
            self.record_time_label.setText("")
        self._log("Stopping acquisition...")
        self.serval.stop_measurement()
        if self.pipeline_thread:
            self.pipeline_thread.request_stop()

    def _on_pipeline_stopped(self):
        disable_file_logging()
        self._reset_pipeline_status()
        self.is_acquiring = False
        self._is_recording = False
        self._record_start_time = None
        self.record_time_label.setText("")
        self._save_timer.stop()
        self.record_btn.setEnabled(False)
        self._set_record_btn_state(False)
        self._set_led(self.led_acquiring, False)
        self._set_led(self.led_recording, False)
        self._prefill_next_record_filename()
        self.record_filename_edit.setEnabled(True)
        self.save_duration_spin.setEnabled(True)
        self._set_pipeline_controls_enabled(True)
        self.refresh_timer.stop()
        self._last_clear_time = None
        self._lag_ms = None
        self._log("Acquisition stopped")
        self._update_histograms()

    def _on_pipeline_error(self, msg: str):
        self._log(f"Pipeline error: {msg}", level=40)
        QMessageBox.critical(self, "Pipeline Error", msg)

    def _set_pipeline_controls_enabled(self, enabled: bool):
        _set_group_enabled(self.settings.child('pipeline'), enabled)
        _set_group_enabled(
            self.settings.child('serval', 'serval_destination'), enabled)
        self.get_action('run').setChecked(not enabled)
