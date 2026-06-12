"""
Histogram Controller

Thread-safe histogram accumulation for real-time visualization.
Supports multiple TOF ROI regions for filtered 2D images.
"""

import time
import threading
import numpy as np
from collections import OrderedDict

from SERVAL.utils.logging import get_logger

# Log a timing breakdown when add_events/add_pixels takes longer than this
# (milliseconds). Visible at DEBUG level (set_log_level('DEBUG')).
_SLOW_UPDATE_MS = 5.0


class HistogramController:
    """
    Manages histogram accumulation for pixel and TOF data.

    Thread-safe for concurrent updates from pipeline callbacks.
    Supports multiple named TOF ROI regions.

    Parameters
    ----------
    tof_bins : int
        Number of bins for TOF histogram (default: 1000)
    tof_range : tuple
        (min, max) range for TOF in nanoseconds (default: 0-100000 ns = 0-100 us)
    max_timeseries_points : int
        Maximum number of time series points to keep (default: 1000)
    """

    def __init__(self, tof_bins=1000, tof_range=(0.0, 100_000.0), max_timeseries_points=1000):
        self._logger = get_logger('SERVAL.Histogram')
        self._lock = threading.Lock()

        # Pixel histogram: 256x256 2D array (total counts)
        self._pixel_hist = np.zeros((256, 256), dtype=np.int64)

        # TOF histogram
        self._tof_bins = tof_bins
        self._tof_range = tof_range
        self._tof_counts = np.zeros(tof_bins, dtype=np.int64)
        self._tof_edges = np.linspace(tof_range[0], tof_range[1], tof_bins + 1)
        self._tof_centers = (self._tof_edges[:-1] + self._tof_edges[1:]) / 2

        # Multiple ROIs: name -> {"tof_min": float, "tof_max": float, "hist": np.ndarray}
        self._rois = OrderedDict()

        # Statistics
        self._total_events = 0
        self._total_pixels = 0

        # Trigger counting for counts/shot normalization
        self._last_trigger_num = 0

        # Time series tracking
        self._max_timeseries_points = max_timeseries_points
        self._timeseries_start = None
        self._total_timeseries = []  # List of (time, counts_per_shot)
        # ROI time series stored in roi_data["timeseries"]

        # Baselines for computing per-sample deltas
        self._last_sampled_pixel_count = 0
        self._last_sampled_trigger_num = 0

    @staticmethod
    def _bincount_2d(x_clipped, y_clipped):
        """Fast 256x256 histogram via bincount (~18x faster than np.histogram2d)."""
        flat = x_clipped.astype(np.int64) * 256 + y_clipped.astype(np.int64)
        return np.bincount(flat, minlength=256 * 256).reshape(256, 256)

    def add_events(self, event_num, x, y, tof, _tot):
        """
        Add event data to histograms.

        Parameters
        ----------
        event_num : np.ndarray
            Event numbers (uint64)
        x : np.ndarray
            X coordinates (0-255)
        y : np.ndarray
            Y coordinates (0-255)
        tof : np.ndarray
            Time of flight in seconds (converted to ns internally)
        tot : np.ndarray
            Time over threshold
        """
        if len(x) == 0:
            return

        t0 = time.perf_counter()
        with self._lock:
            # Clip to valid range
            x_clipped = np.clip(x.astype(np.int32), 0, 255)
            y_clipped = np.clip(y.astype(np.int32), 0, 255)

            # Convert TOF from seconds to nanoseconds
            tof_ns = tof * 1e9
            # Update total pixel histogram
            self._pixel_hist += self._bincount_2d(x_clipped, y_clipped)

            # Update TOF histogram
            tof_hist, _ = np.histogram(tof_ns, bins=self._tof_edges)
            self._tof_counts += tof_hist.astype(np.int64)

            # Update all ROI histograms
            for _roi_name, roi_data in self._rois.items():
                tof_min = roi_data["tof_min"]
                tof_max = roi_data["tof_max"]
                roi_mask = (tof_ns >= tof_min) & (tof_ns <= tof_max)
                if np.any(roi_mask):
                    roi_data["hist"] += self._bincount_2d(
                        x_clipped[roi_mask], y_clipped[roi_mask])


            # Update trigger counter (event_num is the laser-shot / trigger index)
            if len(event_num) > 0:
                self._last_trigger_num = max(self._last_trigger_num, int(event_num.max()))

            # Update stats
            self._total_events += len(x)

        dt_ms = (time.perf_counter() - t0) * 1000
        if dt_ms > _SLOW_UPDATE_MS:
            self._logger.debug(
                f"add_events: {len(x):,} hits, {len(self._rois)} ROI(s) -> {dt_ms:.2f} ms")

    def add_pixels(self, x, y, toa, _tot):
        """
        Add raw pixel data (without event correlation).

        Parameters
        ----------
        x : np.ndarray
            X coordinates (0-255)
        y : np.ndarray
            Y coordinates (0-255)
        toa : np.ndarray
            Time of arrival in seconds (converted to ns internally)
        tot : np.ndarray
            Time over threshold
        """
        if len(x) == 0:
            return

        t0 = time.perf_counter()
        with self._lock:
            x_clipped = np.clip(x.astype(np.int32), 0, 255)
            y_clipped = np.clip(y.astype(np.int32), 0, 255)

            self._pixel_hist += self._bincount_2d(x_clipped, y_clipped)

            # Fill TOA histogram (reuses the same axis as TOF)
            toa_ns = toa * 1e9
            toa_hist, _ = np.histogram(toa_ns, bins=self._tof_edges)
            self._tof_counts += toa_hist.astype(np.int64)

            # Update ROI histograms filtered by TOA range
            for _roi_name, roi_data in self._rois.items():
                roi_mask = (toa_ns >= roi_data["tof_min"]) & (toa_ns <= roi_data["tof_max"])
                if np.any(roi_mask):
                    roi_data["hist"] += self._bincount_2d(
                        x_clipped[roi_mask], y_clipped[roi_mask])

            self._total_pixels += len(x)

        dt_ms = (time.perf_counter() - t0) * 1000
        if dt_ms > _SLOW_UPDATE_MS:
            self._logger.debug(
                f"add_pixels: {len(x):,} hits, {len(self._rois)} ROI(s) -> {dt_ms:.2f} ms")

    def get_pixel_image(self):
        """
        Get current pixel histogram as 2D array.

        Returns
        -------
        np.ndarray
            256x256 int64 array of pixel counts
        """
        with self._lock:
            return self._pixel_hist.copy()

    def get_tof_histogram(self):
        """
        Get current TOF histogram.

        Returns
        -------
        bin_centers : np.ndarray
            Center of each bin in nanoseconds
        counts : np.ndarray
            Counts in each bin
        """
        with self._lock:
            return self._tof_centers.copy(), self._tof_counts.copy()

    def get_stats(self):
        """
        Get current statistics.

        Returns
        -------
        dict
            Dictionary with 'total_events', 'total_pixels', 'pixel_sum'
        """
        with self._lock:
            roi_stats = {}
            for name, roi_data in self._rois.items():
                roi_stats[name] = int(roi_data["hist"].sum())

            return {
                'total_events': self._total_events,
                'total_pixels': self._total_pixels,
                'pixel_sum': int(self._pixel_hist.sum()),
                'roi_counts': roi_stats,
            }

    def sample_timeseries(self):
        """
        Record a time series sample of counts per laser shot.

        Computes the incremental counts since the last sample divided by the
        incremental number of trigger events (laser shots) to yield counts/shot.
        Should be called periodically (e.g., every refresh).
        """
        with self._lock:
            now = time.time()
            if self._timeseries_start is None:
                self._timeseries_start = now

            elapsed = now - self._timeseries_start
            current_pixel_count = int(self._pixel_hist.sum())
            current_trigger_num = self._last_trigger_num

            delta_counts = current_pixel_count - self._last_sampled_pixel_count
            delta_triggers = current_trigger_num - self._last_sampled_trigger_num
            # In trigger mode: counts/shot.  In pixel mode (no triggers): raw count delta.
            rate = delta_counts / delta_triggers if delta_triggers > 0 else float(delta_counts)

            self._last_sampled_pixel_count = current_pixel_count
            self._last_sampled_trigger_num = current_trigger_num

            self._total_timeseries.append((elapsed, rate))
            if len(self._total_timeseries) > self._max_timeseries_points:
                self._total_timeseries.pop(0)

            # Add ROI counts/shot (or counts/refresh in pixel mode) samples
            for roi_data in self._rois.values():
                roi_counts = int(roi_data["hist"].sum())
                last_roi_count = roi_data.get("last_sampled_count", 0)

                delta_roi = roi_counts - last_roi_count
                roi_rate = delta_roi / delta_triggers if delta_triggers > 0 else float(delta_roi)

                roi_data["last_sampled_count"] = roi_counts
                roi_data["last_sampled_trigger"] = current_trigger_num

                if "timeseries" not in roi_data:
                    roi_data["timeseries"] = []
                roi_data["timeseries"].append((elapsed, roi_rate))
                if len(roi_data["timeseries"]) > self._max_timeseries_points:
                    roi_data["timeseries"].pop(0)

    def get_timeseries(self, name=None):
        """
        Get time series data.

        Parameters
        ----------
        name : str, optional
            ROI name. If None, returns total counts time series.

        Returns
        -------
        times : np.ndarray
            Time points in seconds since start
        counts : np.ndarray
            Counts at each time point
        """
        with self._lock:
            if name is None:
                data = self._total_timeseries
            elif name in self._rois and "timeseries" in self._rois[name]:
                data = self._rois[name]["timeseries"]
            else:
                return np.array([]), np.array([])

            if not data:
                return np.array([]), np.array([])

            times = np.array([t for t, _ in data])
            counts = np.array([c for _, c in data])
            return times, counts

    def clear_timeseries(self):
        """Clear all time series data."""
        with self._lock:
            self._timeseries_start = None
            self._total_timeseries.clear()
            for roi_data in self._rois.values():
                if "timeseries" in roi_data:
                    roi_data["timeseries"].clear()

    def clear(self):
        """Clear all histogram data (keeps ROI definitions and timeseries)."""
        with self._lock:
            self._pixel_hist.fill(0)
            self._tof_counts.fill(0)
            self._total_events = 0
            self._total_pixels = 0

            # Reset per-sample baselines so next timeseries point measures from zero
            self._last_sampled_pixel_count = 0

            # Clear ROI histograms but keep definitions
            for roi_data in self._rois.values():
                roi_data["hist"].fill(0)
                roi_data["last_sampled_count"] = 0

    def set_tof_config(self, tof_range=None, tof_bins=None):
        """
        Update TOF histogram configuration (clears TOF data).

        Parameters
        ----------
        tof_range : tuple, optional
            (min, max) in nanoseconds
        tof_bins : int, optional
            Number of bins
        """
        with self._lock:
            if tof_range is not None:
                self._tof_range = tof_range
            if tof_bins is not None:
                self._tof_bins = tof_bins

            self._tof_edges = np.linspace(
                self._tof_range[0], self._tof_range[1], self._tof_bins + 1
            )
            self._tof_centers = (self._tof_edges[:-1] + self._tof_edges[1:]) / 2
            self._tof_counts = np.zeros(self._tof_bins, dtype=np.int64)

    # =========================================================================
    # Multiple ROI Support
    # =========================================================================

    def add_roi(self, name, tof_min, tof_max):
        """
        Add a new TOF ROI region.

        Parameters
        ----------
        name : str
            Unique name for this ROI
        tof_min : float
            Minimum TOF in nanoseconds
        tof_max : float
            Maximum TOF in nanoseconds
        """
        with self._lock:
            self._rois[name] = {
                "tof_min": tof_min,
                "tof_max": tof_max,
                "hist": np.zeros((256, 256), dtype=np.int64),
                "timeseries": [],
            }

    def update_roi(self, name, tof_min, tof_max):
        """
        Update an existing ROI's range.

        Parameters
        ----------
        name : str
            ROI name
        tof_min : float
            New minimum TOF
        tof_max : float
            New maximum TOF
        """
        with self._lock:
            if name not in self._rois:
                return

            self._rois[name]["tof_min"] = tof_min
            self._rois[name]["tof_max"] = tof_max
            self._rois[name]["hist"].fill(0)
            # Reset timeseries baselines so the next sample measures from zero
            self._rois[name]["last_sampled_count"] = 0
            self._rois[name]["last_sampled_trigger"] = 0

    def remove_roi(self, name):
        """
        Remove a ROI.

        Parameters
        ----------
        name : str
            ROI name to remove
        """
        with self._lock:
            if name in self._rois:
                del self._rois[name]

    def get_roi_names(self):
        """Get list of ROI names."""
        with self._lock:
            return list(self._rois.keys())

    def get_roi_image(self, name):
        """
        Get ROI-filtered pixel histogram.

        Parameters
        ----------
        name : str
            ROI name

        Returns
        -------
        np.ndarray or None
            256x256 int64 array of pixel counts, or None if ROI doesn't exist
        """
        with self._lock:
            if name in self._rois:
                return self._rois[name]["hist"].copy()
            return None

    def get_roi_range(self, name):
        """
        Get ROI range.

        Parameters
        ----------
        name : str
            ROI name

        Returns
        -------
        tuple or None
            (tof_min, tof_max) or None if ROI doesn't exist
        """
        with self._lock:
            if name in self._rois:
                return (self._rois[name]["tof_min"], self._rois[name]["tof_max"])
            return None

    def get_roi_counts(self, name):
        """Get total counts in a ROI."""
        with self._lock:
            if name in self._rois:
                return int(self._rois[name]["hist"].sum())
            return 0

    # Legacy compatibility
    def set_tof_range(self, tof_range, tof_bins=None):
        """Legacy method - use set_tof_config instead."""
        self.set_tof_config(tof_range=tof_range, tof_bins=tof_bins)

    # Legacy single ROI methods (for backwards compatibility)
    def has_roi(self):
        """Check if any ROI is active."""
        return len(self._rois) > 0

    def set_roi(self, tof_min, tof_max):
        """Legacy: set a single ROI named 'default'."""
        self.add_roi("default", tof_min, tof_max)

    def clear_roi(self):
        """Legacy: clear the 'default' ROI."""
        self.remove_roi("default")

    def get_roi_pixel_image(self):
        """Legacy: get the 'default' ROI image."""
        return self.get_roi_image("default")
