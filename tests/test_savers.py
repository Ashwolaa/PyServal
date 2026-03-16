"""
Tests for saver processes: idle startup, NEW_FILE/CLOSE_FILE control messages,
data discard when idle, and correct file contents.
"""

import multiprocessing
import time
from pathlib import Path

import numpy as np
import pytest

from SERVAL.core.workers.savers import (
    EVENT_DTYPE,
    PIXEL_DTYPE,
    EventSaverProcess,
    PixelSaverProcess,
    RawSaverProcess,
)


def _drain_and_stop(queue, saver, timeout=3.0):
    """Send STOP and join saver process."""
    queue.put("STOP")
    saver.join(timeout=timeout)
    if saver.is_alive():
        saver.terminate()
        saver.join(timeout=1.0)


# ---------------------------------------------------------------------------
# EventSaverProcess
# ---------------------------------------------------------------------------

class TestEventSaverProcess:
    def test_starts_idle_no_file_created(self, tmp_path):
        """Saver should not create any file on startup."""
        q = multiprocessing.Queue()
        saver = EventSaverProcess(input_queue=q)
        saver.start()
        time.sleep(0.3)
        _drain_and_stop(q, saver)
        assert list(tmp_path.iterdir()) == []

    def test_new_file_creates_file(self, tmp_path):
        filepath = str(tmp_path / "events.dat")
        q = multiprocessing.Queue()
        saver = EventSaverProcess(input_queue=q)
        saver.start()
        q.put(("NEW_FILE", filepath))
        time.sleep(0.3)
        _drain_and_stop(q, saver)
        assert Path(filepath).exists()

    def test_data_discarded_when_idle(self, tmp_path):
        """Data sent before NEW_FILE must not cause a file to appear."""
        q = multiprocessing.Queue()
        saver = EventSaverProcess(input_queue=q)
        saver.start()
        # Send data with no file open
        q.put((np.array([0], dtype=np.uint64),
               np.array([10], dtype=np.uint16),
               np.array([20], dtype=np.uint16),
               np.array([1e-6], dtype=np.float64),
               np.array([100], dtype=np.uint32)))
        time.sleep(0.3)
        _drain_and_stop(q, saver)
        assert list(tmp_path.iterdir()) == []

    def test_writes_correct_records(self, tmp_path):
        filepath = str(tmp_path / "events.dat")
        q = multiprocessing.Queue()
        saver = EventSaverProcess(input_queue=q)
        saver.start()

        q.put(("NEW_FILE", filepath))
        time.sleep(0.1)

        event_num = np.array([0, 1, 2], dtype=np.uint64)
        x = np.array([10, 20, 30], dtype=np.uint16)
        y = np.array([11, 21, 31], dtype=np.uint16)
        tof = np.array([1e-6, 2e-6, 3e-6], dtype=np.float64)
        tot = np.array([100, 200, 300], dtype=np.uint32)
        q.put((event_num, x, y, tof, tot))

        q.put(("CLOSE_FILE",))
        time.sleep(0.3)
        _drain_and_stop(q, saver)

        records = np.fromfile(filepath, dtype=EVENT_DTYPE)
        assert len(records) == 3
        np.testing.assert_array_equal(records["x"], x)
        np.testing.assert_array_equal(records["y"], y)
        np.testing.assert_array_almost_equal(records["tof"], tof)
        np.testing.assert_array_equal(records["tot"], tot)

    def test_close_file_then_idle(self, tmp_path):
        """After CLOSE_FILE, subsequent data must not be written."""
        filepath = str(tmp_path / "events.dat")
        q = multiprocessing.Queue()
        saver = EventSaverProcess(input_queue=q)
        saver.start()

        q.put(("NEW_FILE", filepath))
        time.sleep(0.1)
        q.put(("CLOSE_FILE",))
        time.sleep(0.1)

        # Data sent after CLOSE_FILE should be discarded
        q.put((np.array([99], dtype=np.uint64),
               np.array([5], dtype=np.uint16),
               np.array([5], dtype=np.uint16),
               np.array([9e-6], dtype=np.float64),
               np.array([999], dtype=np.uint32)))

        time.sleep(0.2)
        _drain_and_stop(q, saver)

        records = np.fromfile(filepath, dtype=EVENT_DTYPE)
        assert len(records) == 0

    def test_new_file_switches_file(self, tmp_path):
        """NEW_FILE on an open saver closes the old file and opens a new one."""
        path1 = str(tmp_path / "first.dat")
        path2 = str(tmp_path / "second.dat")
        q = multiprocessing.Queue()
        saver = EventSaverProcess(input_queue=q)
        saver.start()

        q.put(("NEW_FILE", path1))
        time.sleep(0.1)
        ev = (np.array([0], dtype=np.uint64), np.array([1], dtype=np.uint16),
              np.array([2], dtype=np.uint16), np.array([1e-6], dtype=np.float64),
              np.array([10], dtype=np.uint32))
        q.put(ev)

        q.put(("NEW_FILE", path2))
        time.sleep(0.1)
        ev2 = (np.array([1, 2], dtype=np.uint64), np.array([3, 4], dtype=np.uint16),
               np.array([5, 6], dtype=np.uint16), np.array([2e-6, 3e-6], dtype=np.float64),
               np.array([20, 30], dtype=np.uint32))
        q.put(ev2)

        q.put(("CLOSE_FILE",))
        time.sleep(0.3)
        _drain_and_stop(q, saver)

        r1 = np.fromfile(path1, dtype=EVENT_DTYPE)
        r2 = np.fromfile(path2, dtype=EVENT_DTYPE)
        assert len(r1) == 1
        assert len(r2) == 2


# ---------------------------------------------------------------------------
# PixelSaverProcess
# ---------------------------------------------------------------------------

class TestPixelSaverProcess:
    def test_starts_idle(self, tmp_path):
        q = multiprocessing.Queue()
        saver = PixelSaverProcess(input_queue=q)
        saver.start()
        time.sleep(0.3)
        _drain_and_stop(q, saver)
        assert list(tmp_path.iterdir()) == []

    def test_writes_correct_records(self, tmp_path):
        filepath = str(tmp_path / "pixels.dat")
        q = multiprocessing.Queue()
        saver = PixelSaverProcess(input_queue=q)
        saver.start()

        q.put(("NEW_FILE", filepath))
        time.sleep(0.1)

        x = np.array([1, 2, 3], dtype=np.uint16)
        y = np.array([4, 5, 6], dtype=np.uint16)
        toa = np.array([1e-7, 2e-7, 3e-7], dtype=np.float64)
        tot = np.array([10, 20, 30], dtype=np.uint32)
        q.put((x, y, toa, tot))

        q.put(("CLOSE_FILE",))
        time.sleep(0.3)
        _drain_and_stop(q, saver)

        records = np.fromfile(filepath, dtype=PIXEL_DTYPE)
        assert len(records) == 3
        np.testing.assert_array_equal(records["x"], x)
        np.testing.assert_array_equal(records["y"], y)
        np.testing.assert_array_almost_equal(records["toa"], toa)


# ---------------------------------------------------------------------------
# RawSaverProcess
# ---------------------------------------------------------------------------

class TestRawSaverProcess:
    def test_starts_idle(self, tmp_path):
        q = multiprocessing.Queue()
        saver = RawSaverProcess(input_queue=q)
        saver.start()
        time.sleep(0.3)
        _drain_and_stop(q, saver)
        assert list(tmp_path.iterdir()) == []

    def test_writes_bytes(self, tmp_path):
        filepath = str(tmp_path / "data.tpx3")
        q = multiprocessing.Queue()
        saver = RawSaverProcess(input_queue=q)
        saver.start()

        payload = b"TPX3" + b"\xAB" * 64
        q.put(("NEW_FILE", filepath))
        time.sleep(0.1)
        q.put(payload)
        q.put(("CLOSE_FILE",))
        time.sleep(0.3)
        _drain_and_stop(q, saver)

        assert Path(filepath).read_bytes() == payload

    def test_data_discarded_when_idle(self, tmp_path):
        q = multiprocessing.Queue()
        saver = RawSaverProcess(input_queue=q)
        saver.start()
        q.put(b"should be discarded")
        time.sleep(0.2)
        _drain_and_stop(q, saver)
        assert list(tmp_path.iterdir()) == []

    def test_multiple_recordings(self, tmp_path):
        """Two sequential NEW_FILE sessions produce independent files."""
        path1 = str(tmp_path / "run1.tpx3")
        path2 = str(tmp_path / "run2.tpx3")
        q = multiprocessing.Queue()
        saver = RawSaverProcess(input_queue=q)
        saver.start()

        q.put(("NEW_FILE", path1))
        time.sleep(0.1)
        q.put(b"run1data")
        q.put(("CLOSE_FILE",))
        time.sleep(0.1)

        q.put(("NEW_FILE", path2))
        time.sleep(0.1)
        q.put(b"run2data_longer")
        q.put(("CLOSE_FILE",))
        time.sleep(0.2)
        _drain_and_stop(q, saver)

        assert Path(path1).read_bytes() == b"run1data"
        assert Path(path2).read_bytes() == b"run2data_longer"

    def test_creates_parent_directories(self, tmp_path):
        filepath = str(tmp_path / "subdir" / "nested" / "data.tpx3")
        q = multiprocessing.Queue()
        saver = RawSaverProcess(input_queue=q)
        saver.start()

        q.put(("NEW_FILE", filepath))
        time.sleep(0.1)
        q.put(b"hello")
        q.put(("CLOSE_FILE",))
        time.sleep(0.2)
        _drain_and_stop(q, saver)

        assert Path(filepath).exists()
