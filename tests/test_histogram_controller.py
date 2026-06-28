"""
Tests for HistogramController — accumulation, ROI filtering, and spatial-TOF.
"""

import numpy as np
import pytest

# Import directly to avoid pulling in the full Qt-laden gui/__init__.py
import importlib.util, sys, types

for _mod in [
    "qtpy", "qtpy.QtCore", "qtpy.QtWidgets", "qtpy.QtGui",
    "pyqtgraph", "pymodaq_gui",
    "pymodaq_gui.managers", "pymodaq_gui.managers.action_manager",
    "pymodaq_gui.managers.parameter_manager",
    "pymodaq_gui.parameter", "pymodaq_gui.utils", "pymodaq_gui.utils.styling",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

_spec = importlib.util.spec_from_file_location(
    "histogram_controller",
    "SERVAL/gui/histogram_controller.py",
)
_hc_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hc_mod)
HistogramController = _hc_mod.HistogramController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_events(n=200, x_range=(0, 255), y_range=(0, 255),
                 tof_range=(0.0, 100e-9), seed=0):
    rng = np.random.default_rng(seed)
    x   = rng.integers(*x_range, n, endpoint=True).astype(np.int32)
    y   = rng.integers(*y_range, n, endpoint=True).astype(np.int32)
    tof = rng.uniform(*tof_range, n).astype(np.float64)
    event_num = np.zeros(n, dtype=np.uint64)
    tot       = np.zeros(n, dtype=np.uint32)
    return event_num, x, y, tof, tot


# ---------------------------------------------------------------------------
# Basic accumulation
# ---------------------------------------------------------------------------

class TestBasicAccumulation:
    def test_tof_histogram_counts_all_events(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        ev, x, y, tof, tot = _make_events(500)
        hc.add_events(ev, x, y, tof * 1e9, tot)   # tof already in ns here? No — add_events takes seconds
        # Reset: use seconds
        hc2 = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        ev, x, y, tof, tot = _make_events(500)
        hc2.add_events(ev, x, y, tof, tot)
        _, counts = hc2.get_tof_histogram()
        assert counts.sum() == 500

    def test_pixel_histogram_shape(self):
        hc = HistogramController()
        ev, x, y, tof, tot = _make_events(100)
        hc.add_events(ev, x, y, tof, tot)
        img = hc.get_pixel_image()
        assert img.shape == (256, 256)
        assert img.sum() == 100

    def test_empty_batch_is_noop(self):
        hc = HistogramController(tof_bins=50, tof_range=(0.0, 1000.0))
        hc.add_events(
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.uint32),
        )
        _, counts = hc.get_tof_histogram()
        assert counts.sum() == 0

    def test_clear_resets_histograms_keeps_rois(self):
        hc = HistogramController(tof_bins=50, tof_range=(0.0, 100_000.0))
        hc.add_roi("r1", 0.0, 50_000.0)
        ev, x, y, tof, tot = _make_events(200)
        hc.add_events(ev, x, y, tof, tot)
        hc.clear()
        img = hc.get_pixel_image()
        _, counts = hc.get_tof_histogram()
        assert img.sum() == 0
        assert counts.sum() == 0
        assert "r1" in hc.get_roi_names()


# ---------------------------------------------------------------------------
# TOF ROI (time-window filtering → 2D images)
# ---------------------------------------------------------------------------

class TestTofRoi:
    def test_roi_image_is_subset_of_total(self):
        # tof_range in ns; events are uniform over 0-100 ns, ROI covers 0-25 ns
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100.0))
        hc.add_roi("early", 0.0, 25.0)
        ev, x, y, tof, tot = _make_events(500, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        total = hc.get_pixel_image()
        roi   = hc.get_roi_image("early")
        # Every pixel in the ROI image must be <= total image
        assert np.all(roi <= total)
        assert roi.sum() < total.sum()

    def test_full_range_roi_equals_total(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_roi("all", 0.0, 100_000.0)
        ev, x, y, tof, tot = _make_events(300, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        total = hc.get_pixel_image()
        roi   = hc.get_roi_image("all")
        np.testing.assert_array_equal(roi, total)

    def test_empty_range_roi_is_zero(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_roi("none", 200_000.0, 300_000.0)  # outside tof_range
        ev, x, y, tof, tot = _make_events(200, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        roi = hc.get_roi_image("none")
        assert roi.sum() == 0

    def test_update_roi_clears_histogram(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_roi("r", 0.0, 50_000.0)
        ev, x, y, tof, tot = _make_events(200, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        before = hc.get_roi_image("r").sum()
        assert before > 0
        hc.update_roi("r", 0.0, 50_000.0)   # same range — still resets
        assert hc.get_roi_image("r").sum() == 0

    def test_remove_roi(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_roi("r", 0.0, 50_000.0)
        hc.remove_roi("r")
        assert "r" not in hc.get_roi_names()
        assert hc.get_roi_image("r") is None


# ---------------------------------------------------------------------------
# Spatial ROI — image-space masks
# ---------------------------------------------------------------------------

class TestSpatialRoi:
    def test_spatial_roi_counts_subset(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_spatial_roi("", "box", "rect", "+", 0, 0, 128, 128)
        ev, x, y, tof, tot = _make_events(400)
        hc.add_events(ev, x, y, tof, tot)
        total  = hc.get_pixel_image().sum()
        in_box = hc.get_spatial_roi_counts("", "box")
        assert 0 < in_box <= total

    def test_full_image_spatial_roi_equals_total(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_spatial_roi("", "all", "rect", "+", 0, 0, 256, 256)
        ev, x, y, tof, tot = _make_events(300)
        hc.add_events(ev, x, y, tof, tot)
        total  = hc.get_pixel_image().sum()
        in_all = hc.get_spatial_roi_counts("", "all")
        assert in_all == total

    def test_exclude_roi_reduces_combined(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_spatial_roi("", "full", "rect", "+", 0, 0, 256, 256)
        hc.add_spatial_roi("", "hole", "rect", "-", 64, 64, 128, 128)
        ev, x, y, tof, tot = _make_events(400)
        hc.add_events(ev, x, y, tof, tot)
        combined = hc.get_combined_counts("")
        full     = hc.get_spatial_roi_counts("", "full")
        assert combined < full


# ---------------------------------------------------------------------------
# Spatial-ROI TOF histograms (the new feature)
# ---------------------------------------------------------------------------

class TestSpatialRoiTof:
    def test_tof_hist_sum_leq_total_tof(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_spatial_roi("", "S1", "rect", "+", 0, 0, 128, 128)
        ev, x, y, tof, tot = _make_events(500, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        _, total_counts = hc.get_tof_histogram()
        _, s1_counts    = hc.get_spatial_roi_tof("", "S1")
        assert s1_counts.sum() <= total_counts.sum()
        assert s1_counts.sum() > 0

    def test_combined_tof_leq_total_tof(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_spatial_roi("", "A", "rect", "+", 0, 0, 128, 256)
        hc.add_spatial_roi("", "B", "rect", "+", 128, 0, 128, 256)
        ev, x, y, tof, tot = _make_events(400, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        _, total    = hc.get_tof_histogram()
        _, combined = hc.get_combined_tof("")
        assert combined.sum() <= total.sum()

    def test_full_coverage_roi_tof_equals_total(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_spatial_roi("", "all", "rect", "+", 0, 0, 256, 256)
        ev, x, y, tof, tot = _make_events(300, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        _, total  = hc.get_tof_histogram()
        _, s_tof  = hc.get_spatial_roi_tof("", "all")
        _, c_tof  = hc.get_combined_tof("")
        np.testing.assert_array_equal(s_tof, total)
        np.testing.assert_array_equal(c_tof, total)

    def test_exclude_reduces_combined_tof(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_spatial_roi("", "full", "rect", "+", 0, 0, 256, 256)
        hc.add_spatial_roi("", "hole", "rect", "-", 0, 0, 128, 256)
        ev, x, y, tof, tot = _make_events(400, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        _, full_tof  = hc.get_spatial_roi_tof("", "full")
        _, comb_tof  = hc.get_combined_tof("")
        assert comb_tof.sum() < full_tof.sum()
        assert comb_tof.sum() > 0

    def test_tof_hist_zeros_after_clear(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_spatial_roi("", "S", "rect", "+", 0, 0, 128, 128)
        ev, x, y, tof, tot = _make_events(200, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        hc.clear()
        _, s_tof = hc.get_spatial_roi_tof("", "S")
        _, c_tof = hc.get_combined_tof("")
        assert s_tof.sum() == 0
        assert c_tof.sum() == 0

    def test_tof_hist_resets_on_roi_update(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_spatial_roi("", "S", "rect", "+", 0, 0, 128, 128)
        ev, x, y, tof, tot = _make_events(200, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        hc.update_spatial_roi("", "S", "rect", "+", 64, 64, 64, 64)
        _, s_tof = hc.get_spatial_roi_tof("", "S")
        _, c_tof = hc.get_combined_tof("")
        assert s_tof.sum() == 0
        assert c_tof.sum() == 0

    def test_tof_hist_resizes_with_set_tof_config(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_spatial_roi("", "S", "rect", "+", 0, 0, 128, 128)
        ev, x, y, tof, tot = _make_events(200, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        hc.set_tof_config(tof_bins=200)
        _, s_tof = hc.get_spatial_roi_tof("", "S")
        assert len(s_tof) == 200
        assert s_tof.sum() == 0

    def test_tof_roi_on_parent_spatial_ignored(self):
        """Spatial ROIs on TOF-ROI images (parent != '') get no tof_hist."""
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_roi("tof_roi", 0.0, 50_000.0)
        hc.add_spatial_roi("tof_roi", "S", "rect", "+", 0, 0, 128, 128)
        ev, x, y, tof, tot = _make_events(200, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        # Should return zeros (no tof_hist stored for non-"" parents)
        _, s_tof = hc.get_spatial_roi_tof("tof_roi", "S")
        assert s_tof.sum() == 0

    def test_remove_only_roi_discards_combined_tof(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_spatial_roi("", "S", "rect", "+", 0, 0, 128, 128)
        ev, x, y, tof, tot = _make_events(200, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        hc.remove_spatial_roi("", "S")
        assert "" not in hc._combined_tof_hists

    def test_add_pixels_accumulates_toa_in_tof_hist(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        hc.add_spatial_roi("", "S", "rect", "+", 0, 0, 256, 256)
        rng = np.random.default_rng(7)
        x   = rng.integers(0, 256, 300).astype(np.int32)
        y   = rng.integers(0, 256, 300).astype(np.int32)
        toa = rng.uniform(0.0, 100e-9, 300)
        tot = np.zeros(300, dtype=np.uint32)
        hc.add_pixels(x, y, toa, tot)
        _, s_toa = hc.get_spatial_roi_tof("", "S")
        _, total = hc.get_tof_histogram()
        # Full-coverage ROI should match the main TOA histogram
        np.testing.assert_array_equal(s_toa, total)


# ---------------------------------------------------------------------------
# set_tof_config
# ---------------------------------------------------------------------------

class TestSetTofConfig:
    def test_bins_change_resets_histogram(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        ev, x, y, tof, tot = _make_events(100, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        hc.set_tof_config(tof_bins=200)
        _, counts = hc.get_tof_histogram()
        assert len(counts) == 200
        assert counts.sum() == 0

    def test_range_change_resets_histogram(self):
        hc = HistogramController(tof_bins=100, tof_range=(0.0, 100_000.0))
        ev, x, y, tof, tot = _make_events(100, tof_range=(0.0, 100e-9))
        hc.add_events(ev, x, y, tof, tot)
        hc.set_tof_config(tof_range=(0.0, 200_000.0))
        _, counts = hc.get_tof_histogram()
        assert counts.sum() == 0
