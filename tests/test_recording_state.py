"""
Tests for pipeline recording state management: start_record / stop_record /
is_recording, recording_flag, and the file-path logic — without starting
the full TCP / ZMQ / process stack.
"""

import multiprocessing
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from SERVAL.core.pipeline import TPX3PipelineV3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(tmp_path, save_raw=True, save_events=True, save_pixels=False):
    """Build a pipeline with savers configured but NOT started."""
    return TPX3PipelineV3(
        connection_config={"host": "127.0.0.1", "port": 19088},
        save_config={
            "output_dir": str(tmp_path),
            "raw": {"enabled": save_raw, "num_savers": 1, "queue_size": 10},
            "events": {"enabled": save_events, "num_savers": 1, "queue_size": 10},
            "pixels": {"enabled": save_pixels, "num_savers": 1, "queue_size": 10},
        },
        extract_config={"num_workers": 1, "zmq_port": 19001},
        callback_config={"mode": None},
    )


# ---------------------------------------------------------------------------
# is_recording / _recording_flag initial state
# ---------------------------------------------------------------------------

class TestRecordingInitialState:
    def test_not_recording_at_init(self, tmp_path):
        p = _make_pipeline(tmp_path)
        assert p.is_recording is False

    def test_flag_zero_at_init(self, tmp_path):
        p = _make_pipeline(tmp_path)
        assert p._recording_flag.value == 0

    def test_recording_state_dict_at_init(self, tmp_path):
        p = _make_pipeline(tmp_path)
        assert p._recording_state["active"] is False
        assert p._recording_state["filename"] is None


# ---------------------------------------------------------------------------
# start_record / stop_record behaviour
# ---------------------------------------------------------------------------

class TestStartStopRecord:
    def test_start_record_returns_false_when_not_running(self, tmp_path):
        p = _make_pipeline(tmp_path)
        # pipeline.running is False (start() not called)
        result = p.start_record("my_run")
        assert result is False
        assert p.is_recording is False

    def test_start_record_sets_flag_and_state(self, tmp_path):
        p = _make_pipeline(tmp_path)
        p.running = True  # Fake "running" without starting processes

        result = p.start_record("my_run")

        assert result is True
        assert p.is_recording is True
        assert p._recording_flag.value == 1
        assert p._recording_state["filename"] == "my_run"

    def test_stop_record_clears_flag_and_state(self, tmp_path):
        p = _make_pipeline(tmp_path)
        p.running = True
        p.start_record("my_run")

        result = p.stop_record()

        assert result is True
        assert p.is_recording is False
        assert p._recording_flag.value == 0
        assert p._recording_state["filename"] is None

    def test_stop_record_returns_false_when_not_recording(self, tmp_path):
        p = _make_pipeline(tmp_path)
        assert p.stop_record() is False

    def test_second_start_record_stops_first(self, tmp_path):
        p = _make_pipeline(tmp_path)
        p.running = True
        p.start_record("run_a")
        assert p._recording_state["filename"] == "run_a"

        p.start_record("run_b")
        assert p._recording_state["filename"] == "run_b"
        assert p.is_recording is True

    def test_active_savers_raw_only(self, tmp_path):
        p = _make_pipeline(tmp_path, save_raw=True, save_events=True)
        p.running = True
        p.start_record("x", save_raw=True, save_events=False, save_pixels=False)
        assert "raw" in p._recording_state["active_savers"]
        assert "events" not in p._recording_state["active_savers"]

    def test_active_savers_events_only(self, tmp_path):
        p = _make_pipeline(tmp_path, save_raw=True, save_events=True)
        p.running = True
        p.start_record("x", save_raw=False, save_events=True)
        assert "raw" not in p._recording_state["active_savers"]
        assert "events" in p._recording_state["active_savers"]


# ---------------------------------------------------------------------------
# NEW_FILE messages sent to saver queues
# ---------------------------------------------------------------------------

class TestSaverQueueMessages:
    def test_new_file_sent_to_raw_queue(self, tmp_path):
        p = _make_pipeline(tmp_path, save_raw=True, save_events=False)
        p.running = True
        p.start_record("scan_001")

        msg = p.raw_saver_queues[0].get(timeout=1.0)
        assert msg[0] == "NEW_FILE"
        assert "scan_001.tpx3" in msg[1]

    def test_new_file_sent_to_event_queue(self, tmp_path):
        p = _make_pipeline(tmp_path, save_raw=False, save_events=True)
        p.running = True
        p.start_record("scan_001")

        msg = p.extractors.saver_queues["events"][0].get(timeout=1.0)
        assert msg[0] == "NEW_FILE"
        assert "scan_001" in msg[1]
        assert "_events.dat" in msg[1]

    def test_close_file_sent_on_stop(self, tmp_path):
        p = _make_pipeline(tmp_path, save_raw=True, save_events=True)
        p.running = True
        p.start_record("scan_001")
        # Drain the NEW_FILE messages (use timeout: mp.Queue background thread may not have flushed yet)
        p.raw_saver_queues[0].get(timeout=1.0)
        p.extractors.saver_queues["events"][0].get(timeout=1.0)

        p.stop_record()

        raw_msg = p.raw_saver_queues[0].get(timeout=1.0)
        evt_msg = p.extractors.saver_queues["events"][0].get(timeout=1.0)
        assert raw_msg == ("CLOSE_FILE",)
        assert evt_msg == ("CLOSE_FILE",)

    def test_output_dir_used_in_filepath(self, tmp_path):
        custom_dir = tmp_path / "custom_output"
        p = _make_pipeline(tmp_path, save_raw=True, save_events=False)
        p.running = True
        p.start_record("myfile", output_dir=str(custom_dir))

        msg = p.raw_saver_queues[0].get(timeout=1.0)
        assert str(custom_dir) in msg[1]

    def test_no_message_when_saver_type_disabled(self, tmp_path):
        """With save_pixels=True but no pixel saver created, no crash."""
        p = _make_pipeline(tmp_path, save_raw=False, save_events=False, save_pixels=False)
        p.running = True
        # Should not raise even though pixel saver wasn't created
        result = p.start_record("x", save_raw=False, save_events=False, save_pixels=True)
        assert result is True
        assert p.is_recording is True
