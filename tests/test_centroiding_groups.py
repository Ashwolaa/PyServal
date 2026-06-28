"""
Tests for run-group discovery and per-group centroiding.

Covers the fix for scan folders holding multiple steps' *_events.dat files
side by side (e.g. PyMoDAQ scan steps "00001", "00002", ...) which must NOT
be merged together, while parallel-saver splits of one take ("_saver0",
"_saver1") still must be.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from SERVAL.core.data_types import EVENT_DTYPE, TRIGGER_DTYPE
from SERVAL.postprocessing.centroiding import (
    CentroidProcessor,
    MERGED_CENTROID_DTYPE,
    discover_run_groups,
    get_run_info,
    get_run_status,
    RunStatus,
    step_key,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_events_file(path: Path, t_triggers: list, hits_per_shot=2, x=10, y=10, tot=100):
    """Write `hits_per_shot` events per given t_trigger value, all at the same
    (x, y, tof) — dbscan_main groups points by shot (t_trigger) before
    clustering, and a lone point has no neighbors (self-excluded) with the
    default min_points=1, so each shot needs >=2 coincident hits to produce
    one centroid."""
    n_shots = len(t_triggers)
    n = n_shots * hits_per_shot
    raw = np.empty(n, dtype=EVENT_DTYPE)
    raw["t_trigger"] = np.repeat(np.asarray(t_triggers, dtype=np.float64), hits_per_shot)
    raw["x"] = x
    raw["y"] = y
    raw["tof"] = 1e-6
    raw["tot"] = tot
    raw.tofile(path)


def make_triggers_file(path: Path, toas: list, tdc_id=1, edge=0):
    raw = np.empty(len(toas), dtype=TRIGGER_DTYPE)
    raw["toa"] = np.asarray(toas, dtype=np.float64)
    raw["tdc_id"] = tdc_id
    raw["edge"] = edge
    raw.tofile(path)


def make_meta_file(path: Path, tdc_id=1, edge=0):
    with open(path, "w") as f:
        json.dump({"tdc_id": tdc_id, "edge": edge}, f)


# ---------------------------------------------------------------------------
# step_key / discover_run_groups
# ---------------------------------------------------------------------------

class TestStepKey:
    def test_plain_step(self):
        assert step_key(Path("00001_events.dat")) == "00001"

    def test_saver_split(self):
        assert step_key(Path("00001_saver0_events.dat")) == "00001"
        assert step_key(Path("00001_saver1_events.dat")) == "00001"

    def test_named_run(self):
        assert step_key(Path("my_run_events.dat")) == "my_run"


class TestDiscoverRunGroups:
    def test_distinct_steps_are_separate_groups(self, tmp_path):
        (tmp_path / "00001_events.dat").touch()
        (tmp_path / "00002_events.dat").touch()
        groups = discover_run_groups(tmp_path)
        assert set(groups.keys()) == {"00001", "00002"}
        assert len(groups["00001"]) == 1
        assert len(groups["00002"]) == 1

    def test_saver_splits_grouped_together(self, tmp_path):
        (tmp_path / "take_saver0_events.dat").touch()
        (tmp_path / "take_saver1_events.dat").touch()
        groups = discover_run_groups(tmp_path)
        assert set(groups.keys()) == {"take"}
        assert len(groups["take"]) == 2

    def test_empty_folder(self, tmp_path):
        assert discover_run_groups(tmp_path) == {}


# ---------------------------------------------------------------------------
# get_run_status / get_run_info
# ---------------------------------------------------------------------------

class TestRunStatus:
    def test_empty_when_no_event_files(self, tmp_path):
        assert get_run_status([], tmp_path / "x_centroids.datbin") == RunStatus.EMPTY

    def test_ready_when_no_centroid_file(self, tmp_path):
        ev = tmp_path / "00001_events.dat"
        make_events_file(ev, [1.0])
        assert get_run_status([ev], tmp_path / "00001_centroids.datbin") == RunStatus.READY

    def test_done_when_centroid_newer(self, tmp_path):
        ev = tmp_path / "00001_events.dat"
        make_events_file(ev, [1.0])
        centroid = tmp_path / "00001_centroids.datbin"
        centroid.write_bytes(b"\x00" * MERGED_CENTROID_DTYPE.itemsize)
        assert get_run_status([ev], centroid) == RunStatus.DONE

    def test_get_run_info_fields(self, tmp_path):
        ev = tmp_path / "00001_events.dat"
        make_events_file(ev, [1.0, 2.0])
        centroid = tmp_path / "00001_centroids.datbin"
        info = get_run_info("00001", [ev], centroid)
        assert info["name"] == "00001"
        assert info["n_event_files"] == 1
        assert info["status"] == RunStatus.READY
        assert info["n_centroids"] is None


# ---------------------------------------------------------------------------
# process_event_group / process_run_dir_merged
# ---------------------------------------------------------------------------

class TestProcessEventGroup:
    def test_single_step_shot_index(self, tmp_path):
        """One step's own triggers give shot_index 0, 1, 2 — sanity check."""
        ev = tmp_path / "00001_events.dat"
        make_events_file(ev, [1.0, 2.0, 3.0])
        make_triggers_file(tmp_path / "00001_triggers.trg", [1.0, 2.0, 3.0])
        make_meta_file(tmp_path / "00001_meta.json")

        proc = CentroidProcessor(tof_min=0.0, tof_max=1.0)
        out = proc.process_event_group([ev], tmp_path / "00001_centroids.datbin")

        result = np.fromfile(out, dtype=MERGED_CENTROID_DTYPE)
        assert len(result) == 3
        assert sorted(result["shot_index"].tolist()) == [0, 1, 2]

    def test_saver_split_merges_into_one_shot_index_space(self, tmp_path):
        """Parallel-saver splits of ONE take must still merge together."""
        ev0 = tmp_path / "take_saver0_events.dat"
        ev1 = tmp_path / "take_saver1_events.dat"
        make_events_file(ev0, [1.0, 3.0])
        make_events_file(ev1, [2.0, 4.0])
        make_triggers_file(tmp_path / "take_saver0_triggers.trg", [1.0, 2.0, 3.0, 4.0])
        make_meta_file(tmp_path / "take_meta.json")

        proc = CentroidProcessor(tof_min=0.0, tof_max=1.0)
        out = proc.process_event_group([ev0, ev1], tmp_path / "take_centroids.datbin")

        result = np.fromfile(out, dtype=MERGED_CENTROID_DTYPE)
        assert len(result) == 4
        assert sorted(result["shot_index"].tolist()) == [0, 1, 2, 3]

    def test_distinct_steps_do_not_leak_shot_index(self, tmp_path):
        """Regression test: two different scan steps sharing a folder must
        each get their own shot_index space, not be merged together."""
        ev1 = tmp_path / "00001_events.dat"
        ev2 = tmp_path / "00002_events.dat"
        # Step 2's triggers start at a LATER absolute time than step 1's —
        # if they were merged, step 1's shots would all sort before step 2's.
        make_events_file(ev1, [1.0, 2.0])
        make_events_file(ev2, [101.0, 102.0])
        make_triggers_file(tmp_path / "00001_triggers.trg", [1.0, 2.0])
        make_triggers_file(tmp_path / "00002_triggers.trg", [101.0, 102.0])
        make_meta_file(tmp_path / "_scan_meta.json")  # one centralized metadata file

        proc = CentroidProcessor(tof_min=0.0, tof_max=1.0)
        results = proc.process_run_dir_merged(tmp_path)

        assert set(results.keys()) == {"00001", "00002"}
        r1 = np.fromfile(results["00001"], dtype=MERGED_CENTROID_DTYPE)
        r2 = np.fromfile(results["00002"], dtype=MERGED_CENTROID_DTYPE)
        # Each step's own 2 events get shot_index 0 and 1 *within that step* —
        # if the bug were present, step 2 would instead get index 2, 3 (continuing
        # from a merged, shared index space across both steps).
        assert sorted(r1["shot_index"].tolist()) == [0, 1]
        assert sorted(r2["shot_index"].tolist()) == [0, 1]

    def test_force_reprocesses_existing_output(self, tmp_path):
        ev = tmp_path / "00001_events.dat"
        make_events_file(ev, [1.0])
        make_triggers_file(tmp_path / "00001_triggers.trg", [1.0])
        make_meta_file(tmp_path / "00001_meta.json")
        out_path = tmp_path / "00001_centroids.datbin"

        proc = CentroidProcessor(tof_min=0.0, tof_max=1.0)
        proc.process_event_group([ev], out_path)
        mtime1 = out_path.stat().st_mtime_ns

        # Without force, an existing output is returned untouched
        proc.process_event_group([ev], out_path)
        assert out_path.stat().st_mtime_ns == mtime1
