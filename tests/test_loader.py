"""
Tests for SERVAL/io/loader.py :: TPX3Run

All tests use tmp_path to create synthetic run directories or single files
so they run without a real detector and without touching the project tree.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from SERVAL.core.data_types import (
    EVENT_DTYPE, PIXEL_DTYPE, TRIGGER_DTYPE,
    TRIGGER_BIT_TDC1_RISING, TRIGGER_BIT_TDC1_FALLING,
    TRIGGER_BIT_TDC2_RISING, TRIGGER_BIT_TDC2_FALLING,
)
from SERVAL.io.loader import TPX3Run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _events(t_trigger, x=None, y=None, tof=None, tot=None) -> np.ndarray:
    n = len(t_trigger)
    arr = np.zeros(n, dtype=EVENT_DTYPE)
    arr["t_trigger"] = t_trigger
    arr["x"]         = x   if x   is not None else np.arange(n, dtype=np.uint16)
    arr["y"]         = y   if y   is not None else np.arange(n, dtype=np.uint16)
    arr["tof"]       = tof if tof is not None else np.zeros(n)
    arr["tot"]       = tot if tot is not None else np.zeros(n, dtype=np.uint32)
    return arr


def _triggers(toa, tdc_id=1, edge=0) -> np.ndarray:
    n = len(toa)
    arr = np.zeros(n, dtype=TRIGGER_DTYPE)
    arr["toa"]    = toa
    arr["tdc_id"] = tdc_id
    arr["edge"]   = edge
    return arr


def _pixels(toa, x=None, y=None) -> np.ndarray:
    n = len(toa)
    arr = np.zeros(n, dtype=PIXEL_DTYPE)
    arr["toa"] = toa
    arr["x"]   = x if x is not None else np.zeros(n, dtype=np.uint16)
    arr["y"]   = y if y is not None else np.zeros(n, dtype=np.uint16)
    return arr


def _write_run(d, name, events=None, triggers=None, pixels=None, meta=None):
    """Write a complete run directory under ``d / name``."""
    run = d / name
    run.mkdir(exist_ok=True)
    if events is not None:
        events.tofile(run / f"{name}_events.dat")
    if triggers is not None:
        triggers.tofile(run / f"{name}_triggers.trg")
    if pixels is not None:
        pixels.tofile(run / f"{name}_pixels.dat")
    if meta is not None:
        (run / f"{name}_meta.json").write_text(json.dumps(meta))
    return run


# ---------------------------------------------------------------------------
# Empty / missing data
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_empty_directory_returns_empty_events(self, tmp_path):
        run = tmp_path / "empty_run"
        run.mkdir()
        r = TPX3Run(run)
        assert len(r.events) == 0
        assert r.events.dtype == EVENT_DTYPE

    def test_empty_directory_returns_empty_triggers(self, tmp_path):
        run = tmp_path / "empty_run"
        run.mkdir()
        r = TPX3Run(run)
        assert len(r.triggers) == 0
        assert r.triggers.dtype == TRIGGER_DTYPE

    def test_empty_directory_pixels_is_none(self, tmp_path):
        run = tmp_path / "empty_run"
        run.mkdir()
        r = TPX3Run(run)
        assert r.pixels is None

    def test_t0_is_zero_when_no_data(self, tmp_path):
        run = tmp_path / "empty_run"
        run.mkdir()
        r = TPX3Run(run)
        assert r.t0 == 0.0


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

class TestBasicLoading:
    def test_events_loaded_correctly(self, tmp_path):
        ev = _events([1.0, 2.0, 3.0], tof=[1e-9, 2e-9, 3e-9])
        tr = _triggers([0.5])
        run = _write_run(tmp_path, "r", events=ev, triggers=tr)
        r = TPX3Run(run)
        # t0 is 0.5 from primary triggers; events rebased by 0.5
        np.testing.assert_allclose(r.events["t_trigger"], [0.5, 1.5, 2.5])
        np.testing.assert_allclose(r.events["tof"], [1e-9, 2e-9, 3e-9])

    def test_events_sorted_by_t_trigger(self, tmp_path):
        ev = _events([3.0, 1.0, 2.0])
        run = _write_run(tmp_path, "r", events=ev)
        r = TPX3Run(run)
        assert list(r.events["t_trigger"]) == sorted(r.events["t_trigger"])

    def test_triggers_loaded_correctly(self, tmp_path):
        tr = _triggers([10.0, 20.0, 30.0], tdc_id=1, edge=0)
        run = _write_run(tmp_path, "r", triggers=tr)
        r = TPX3Run(run)
        np.testing.assert_allclose(r.triggers["toa"], [10.0, 20.0, 30.0])

    def test_triggers_sorted_by_toa(self, tmp_path):
        tr = _triggers([30.0, 10.0, 20.0])
        run = _write_run(tmp_path, "r", triggers=tr)
        r = TPX3Run(run)
        assert list(r.triggers["toa"]) == sorted(r.triggers["toa"])

    def test_pixels_loaded_correctly(self, tmp_path):
        px = _pixels([1.0, 2.0])
        run = _write_run(tmp_path, "r", pixels=px)
        r = TPX3Run(run)
        assert r.pixels is not None
        np.testing.assert_allclose(r.pixels["toa"], [1.0, 2.0])

    def test_pixels_is_none_without_pixel_files(self, tmp_path):
        ev = _events([1.0])
        run = _write_run(tmp_path, "r", events=ev)
        r = TPX3Run(run)
        assert r.pixels is None


# ---------------------------------------------------------------------------
# t0 rebasing
# ---------------------------------------------------------------------------

class TestT0:
    def test_t0_from_primary_trigger(self, tmp_path):
        tr = _triggers([5.0, 6.0, 7.0], tdc_id=1, edge=0)
        run = _write_run(tmp_path, "r", triggers=tr)
        r = TPX3Run(run)
        assert r.t0 == pytest.approx(5.0)

    def test_t0_from_events_when_no_triggers(self, tmp_path):
        ev = _events([3.0, 4.0, 5.0])
        run = _write_run(tmp_path, "r", events=ev)
        r = TPX3Run(run)
        assert r.t0 == pytest.approx(3.0)

    def test_events_rebased_to_zero_at_first_trigger(self, tmp_path):
        t0 = 1000.0
        tr = _triggers([t0, t0 + 1.0, t0 + 2.0])
        ev = _events([t0 + 0.5e-6, t0 + 1.0 + 0.5e-6])
        run = _write_run(tmp_path, "r", events=ev, triggers=tr)
        r = TPX3Run(run)
        # First event should be ~0.5 µs after t0 = 0
        assert r.events["t_trigger"][0] == pytest.approx(0.5e-6, rel=1e-6)

    def test_triggers_not_rebased(self, tmp_path):
        t0 = 500.0
        tr = _triggers([t0, t0 + 1.0])
        run = _write_run(tmp_path, "r", triggers=tr)
        r = TPX3Run(run)
        # raw chip-clock values preserved
        np.testing.assert_allclose(r.triggers["toa"], [t0, t0 + 1.0])


# ---------------------------------------------------------------------------
# Multi-saver concatenation
# ---------------------------------------------------------------------------

class TestMultiSaver:
    def _write_multi_saver(self, tmp_path, name, ev_list):
        run = tmp_path / name
        run.mkdir()
        for i, ev in enumerate(ev_list):
            ev.tofile(run / f"{name}_saver{i}_events.dat")
        return run

    def test_two_saver_files_concatenated(self, tmp_path):
        ev0 = _events([1.0, 3.0])
        ev1 = _events([2.0, 4.0])
        run = self._write_multi_saver(tmp_path, "r", [ev0, ev1])
        r = TPX3Run(run)
        assert len(r.events) == 4

    def test_multi_saver_sorted_by_t_trigger(self, tmp_path):
        ev0 = _events([3.0, 5.0])
        ev1 = _events([2.0, 4.0])
        run = self._write_multi_saver(tmp_path, "r", [ev0, ev1])
        r = TPX3Run(run)
        t = r.events["t_trigger"]
        assert all(t[i] <= t[i + 1] for i in range(len(t) - 1))

    def test_single_file_path_discovers_siblings(self, tmp_path):
        """Pointing TPX3Run at a single *_events.dat file still finds its sibling trigger file."""
        name = "r"
        run = tmp_path / name
        run.mkdir()
        ev = _events([10.0, 11.0])
        tr = _triggers([9.0])
        ev.tofile(run / f"{name}_events.dat")
        tr.tofile(run / f"{name}_triggers.trg")

        r = TPX3Run(run / f"{name}_events.dat")
        assert len(r.events) == 2
        assert len(r.triggers) == 1

    def test_single_saver_file_path_discovers_trigger_sibling(self, tmp_path):
        name = "r"
        run = tmp_path / name
        run.mkdir()
        ev = _events([5.0])
        tr = _triggers([4.0])
        ev.tofile(run / f"{name}_saver0_events.dat")
        tr.tofile(run / f"{name}_saver0_triggers.trg")

        r = TPX3Run(run / f"{name}_saver0_events.dat")
        assert len(r.events) == 1
        assert len(r.triggers) == 1


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_overrides_tdc_id(self, tmp_path):
        tr_tdc2 = _triggers([1.0, 2.0], tdc_id=2, edge=0)
        meta = {"tdc_id": 2}
        run = _write_run(tmp_path, "r", triggers=tr_tdc2, meta=meta)
        r = TPX3Run(run)
        assert len(r.primary_triggers) == 2

    def test_metadata_overrides_edge(self, tmp_path):
        tr = np.zeros(3, dtype=TRIGGER_DTYPE)
        tr["toa"]    = [1.0, 2.0, 3.0]
        tr["tdc_id"] = [1,   1,   1]
        tr["edge"]   = [0,   1,   0]  # mixed edges
        meta = {"edge": 1}
        run = _write_run(tmp_path, "r", triggers=tr, meta=meta)
        r = TPX3Run(run)
        # Only the falling edge trigger should be selected
        assert len(r.primary_triggers) == 1
        assert r.primary_triggers["edge"][0] == 1

    def test_metadata_loaded_from_single_file(self, tmp_path):
        name = "r"
        run = tmp_path / name
        run.mkdir()
        tr = _triggers([1.0], tdc_id=2, edge=0)
        tr.tofile(run / f"{name}_triggers.trg")
        (run / f"{name}_meta.json").write_text(json.dumps({"tdc_id": 2}))
        ev = _events([2.0])
        ev.tofile(run / f"{name}_events.dat")

        r = TPX3Run(run / f"{name}_events.dat")
        assert r._tdc_id == 2

    def test_no_metadata_uses_defaults(self, tmp_path):
        run = _write_run(tmp_path, "r")
        r = TPX3Run(run)
        assert r._tdc_id == 1
        assert r._edge == 0


# ---------------------------------------------------------------------------
# primary_triggers
# ---------------------------------------------------------------------------

class TestPrimaryTriggers:
    def _mixed_triggers(self, tmp_path):
        """Run with TDC1 rising, TDC1 falling, TDC2 rising, TDC2 falling."""
        tr = np.zeros(4, dtype=TRIGGER_DTYPE)
        tr["toa"]    = [1.0, 2.0, 3.0, 4.0]
        tr["tdc_id"] = [1,   1,   2,   2]
        tr["edge"]   = [0,   1,   0,   1]
        run = _write_run(tmp_path, "r", triggers=tr)
        return run

    def test_filters_by_tdc_id_and_edge(self, tmp_path):
        run = self._mixed_triggers(tmp_path)
        r = TPX3Run(run, tdc_id=1)  # default edge=0 (rising)
        assert len(r.primary_triggers) == 1
        assert r.primary_triggers["tdc_id"][0] == 1
        assert r.primary_triggers["edge"][0] == 0

    def test_tdc2_rising_selection(self, tmp_path):
        run = self._mixed_triggers(tmp_path)
        r = TPX3Run(run, tdc_id=2)
        assert len(r.primary_triggers) == 1
        assert r.primary_triggers["tdc_id"][0] == 2
        assert r.primary_triggers["edge"][0] == 0

    def test_tdc_id_zero_selects_by_edge_only(self, tmp_path):
        run = self._mixed_triggers(tmp_path)
        # tdc_id=0 means "both channels, filter by edge only"
        r = TPX3Run(run, tdc_id=0)
        # both rising (edge=0) triggers should be selected regardless of channel
        assert len(r.primary_triggers) == 2
        assert all(r.primary_triggers["edge"] == 0)

    def test_no_triggers_returns_empty(self, tmp_path):
        run = _write_run(tmp_path, "r")
        r = TPX3Run(run)
        assert len(r.primary_triggers) == 0


# ---------------------------------------------------------------------------
# events_shot_index and per-shot slicing
# ---------------------------------------------------------------------------

class TestShotIndex:
    def _setup(self, tmp_path):
        # 3 triggers at t0, t0+1µs, t0+2µs.
        # t_trigger in EVENT_DTYPE stores the trigger timestamp (not the hit time),
        # so each event's t_trigger equals its corresponding trigger's toa.
        t0 = 100.0
        tr = _triggers([t0, t0 + 1e-6, t0 + 2e-6])
        ev = _events(
            [t0, t0 + 1e-6, t0 + 2e-6],   # t_trigger = trigger time
            tof=[0.5e-6, 0.5e-6, 0.5e-6],
        )
        run = _write_run(tmp_path, "r", events=ev, triggers=tr)
        return TPX3Run(run)

    def test_shot_index_length_matches_events(self, tmp_path):
        r = self._setup(tmp_path)
        assert len(r.events_shot_index) == len(r.events)

    def test_shot_index_values(self, tmp_path):
        r = self._setup(tmp_path)
        # After rebasing t_trigger values are [0, 1µs, 2µs]; triggers are also
        # [0, 1µs, 2µs] after rebasing → exact searchsorted gives [0, 1, 2].
        np.testing.assert_array_equal(r.events_shot_index, [0, 1, 2])

    def test_get_events_for_shot(self, tmp_path):
        r = self._setup(tmp_path)
        shot1_ev = r.get_events_for_shot(1)
        assert len(shot1_ev) == 1
        assert shot1_ev["t_trigger"][0] == pytest.approx(1e-6, rel=1e-6)

    def test_get_events_in_shot_range(self, tmp_path):
        r = self._setup(tmp_path)
        ev = r.get_events_in_shot_range(0, 2)
        assert len(ev) == 2

    def test_shot_with_no_events_returns_empty(self, tmp_path):
        # Events only for shots 0 and 2 (t_trigger = trigger time).
        t0 = 100.0
        tr = _triggers([t0, t0 + 1e-6, t0 + 2e-6])
        ev = _events([t0, t0 + 2e-6])   # no event for shot 1
        run = _write_run(tmp_path, "r", events=ev, triggers=tr)
        r = TPX3Run(run)
        assert len(r.get_events_for_shot(1)) == 0

    def test_get_events_in_shot_range_empty_on_invalid_range(self, tmp_path):
        r = self._setup(tmp_path)
        ev = r.get_events_in_shot_range(5, 3)  # stop < start
        assert len(ev) == 0

    def test_events_shot_index_empty_when_no_data(self, tmp_path):
        run = _write_run(tmp_path, "r")
        r = TPX3Run(run)
        assert len(r.events_shot_index) == 0


# ---------------------------------------------------------------------------
# trigger_mask_per_shot
# ---------------------------------------------------------------------------

class TestTriggerMaskPerShot:
    def test_single_trigger_type_each_shot(self, tmp_path):
        # 2 primary triggers; each shot has only TDC1-rising
        tr = _triggers([1.0, 2.0], tdc_id=1, edge=0)
        run = _write_run(tmp_path, "r", triggers=tr)
        r = TPX3Run(run)
        mask = r.trigger_mask_per_shot()
        assert len(mask) == 2
        assert all(m == TRIGGER_BIT_TDC1_RISING for m in mask)

    def test_mask_accumulates_extra_triggers_in_shot(self, tmp_path):
        # Primary: TDC1 rising at 1.0 and 10.0
        # Between shots: TDC2 rising at 5.0 and TDC1 falling at 6.0
        tr = np.zeros(4, dtype=TRIGGER_DTYPE)
        tr["toa"]    = [1.0, 5.0, 6.0, 10.0]
        tr["tdc_id"] = [1,   2,   1,   1]
        tr["edge"]   = [0,   0,   1,   0]
        run = _write_run(tmp_path, "r", triggers=tr)
        r = TPX3Run(run)
        mask = r.trigger_mask_per_shot()
        # shot 0 spans [TDC1-rising@1.0, TDC2-rising@5.0, TDC1-falling@6.0]
        expected = TRIGGER_BIT_TDC1_RISING | TRIGGER_BIT_TDC2_RISING | TRIGGER_BIT_TDC1_FALLING
        assert mask[0] == expected
        # shot 1 spans only [TDC1-rising@10.0]
        assert mask[1] == TRIGGER_BIT_TDC1_RISING

    def test_empty_triggers_returns_empty_mask(self, tmp_path):
        run = _write_run(tmp_path, "r")
        r = TPX3Run(run)
        assert len(r.trigger_mask_per_shot()) == 0


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

class TestDerivedQuantities:
    def test_absolute_times(self, tmp_path):
        # t_trigger stores the trigger time; tof is time after trigger.
        t0 = 50.0
        tr = _triggers([t0, t0 + 1e-6])
        # Event correlated to second trigger: t_trigger = t0+1µs, tof = 2 ns
        ev = _events([t0 + 1e-6], tof=[2e-9])
        run = _write_run(tmp_path, "r", events=ev, triggers=tr)
        r = TPX3Run(run)
        # absolute_times = t_trigger_rebased + tof = 1e-6 + 2e-9
        np.testing.assert_allclose(r.absolute_times(), [1e-6 + 2e-9], rtol=1e-6)

    def test_repr_does_not_crash(self, tmp_path):
        tr = _triggers([1.0, 2.0, 3.0])
        ev = _events([1.5e-6, 2.5e-6])
        run = _write_run(tmp_path, "run_abc", events=ev, triggers=tr)
        r = TPX3Run(run)
        s = repr(r)
        assert "run_abc" in s
        assert "events" in s.lower() or "M events" in s

    def test_repr_no_triggers(self, tmp_path):
        ev = _events([1.0, 2.0, 3.0])
        run = _write_run(tmp_path, "r", events=ev)
        r = TPX3Run(run)
        s = repr(r)
        assert "r" in s
