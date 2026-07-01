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

        # Mass (m/z) calibration, re-binned independently of the TOF histogram.
        # Standard TOF-MS relation: tof_ns = coeff * sqrt(mass) + t0
        #                       =>  mass    = ((tof_ns - t0) / coeff) ** 2
        self._mass_calib_enabled = False
        self._mass_coeff = 1.0
        self._mass_t0 = 0.0
        self._mass_bins = 1000
        self._mass_range = (0.0, 200.0)
        self._mass_edges = np.linspace(self._mass_range[0], self._mass_range[1], self._mass_bins + 1)
        self._mass_centers = (self._mass_edges[:-1] + self._mass_edges[1:]) / 2
        self._mass_counts = np.zeros(self._mass_bins, dtype=np.int64)

        # Per-shot covariance accumulation.
        # Uses the same TOF range as the main histogram but at a configurable
        # (typically smaller) bin count.  For each laser shot we compute the
        # 1-D TOF spectrum n_s (shape [B]) and accumulate:
        #   S1 = Σ_s  n_s                    (sum of shot spectra)
        #   S2 = Σ_s  n_s ⊗ n_s  =  A.T @ A  (sum of outer products)
        # where A is the (shots × bins) matrix.  At display time:
        #   corr(m1,m2) = S2 / N
        #   cov(m1,m2)  = S2/N - outer(S1/N, S1/N)
        self._cov_enabled = False
        self._cov_bins = 200
        self._cov_edges = np.linspace(tof_range[0], tof_range[1], 201)
        self._cov_centers = (self._cov_edges[:-1] + self._cov_edges[1:]) / 2
        self._cov_S1 = np.zeros(200, dtype=np.float64)
        self._cov_S2 = np.zeros((200, 200), dtype=np.float64)
        self._cov_n_shots = 0

        # Multiple ROIs: name -> {"tof_min": float, "tof_max": float, "hist": np.ndarray}
        self._rois = OrderedDict()

        # Spatial (image-space) ROIs: (parent, name) -> {
        #   "mask": np.ndarray(256,256,bool),
        #   "op":   "+" include | "-" exclude (affects the combined mask),
        #   "last_sampled_count": int,
        #   "timeseries": [(elapsed, rate), ...],
        #   "tof_hist": np.ndarray(tof_bins) — only populated for parent="" ROIs
        # }
        # "parent" is "" for the main pixel histogram, or a TOF-ROI name.
        self._spatial_rois = OrderedDict()

        # Per-parent combined-mask timeseries (union(+) AND NOT union(-)).
        # keyed by parent str -> list of (elapsed, rate)
        self._combined_timeseries: dict[str, list] = {}
        self._combined_last_count: dict[str, int] = {}

        # Per-parent combined-mask TOF histograms (only populated for parent="").
        self._combined_tof_hists: dict[str, np.ndarray] = {}

        # Statistics
        self._total_events = 0
        self._total_pixels = 0

        # Trigger counting for counts/shot normalization.
        # Incremented by the number of unique trigger pulses seen in each add_events() call.
        # (event_num passed to add_events is t_trigger in seconds, not an integer index.)
        self._n_triggers_seen = 0

        # Time series tracking
        self._max_timeseries_points = max_timeseries_points
        self._timeseries_start = None
        self._total_timeseries = []  # List of (time, counts_per_shot)
        # ROI time series stored in roi_data["timeseries"]

        # Baselines for computing per-sample deltas
        self._last_sampled_pixel_count = 0
        self._last_sampled_trigger_count = 0

        # Display subsampling compensation. The extractor strides pixel/event
        # arrays before they reach the callback queue (see
        # ExtractorWorker._subsample_for_display) to shrink IPC payload size
        # when "Display %" < 100. Each surviving sample then represents
        # `_display_stride` real hits, so histogram/total increments are
        # scaled up by this factor to approximate true counts. Mirrors the
        # extractor's own stride formula so the two stay in lockstep.
        self._display_stride = 1

    def set_display_fraction(self, frac: float):
        """Record the current GUI display subsampling fraction (0 < frac <= 1).

        Must match the fraction the extractor uses to stride callback data,
        so that add_events/add_pixels can scale counts back up to estimate
        true (unsampled) totals.
        """
        with self._lock:
            self._display_stride = max(1, round(1.0 / frac)) if frac > 0 else 1

    def _tof_ns_to_mass(self, tof_ns):
        """Convert TOF (ns) to mass via the calibration tof_ns = coeff*sqrt(mass) + t0."""
        return np.clip((tof_ns - self._mass_t0) / self._mass_coeff, 0.0, None) ** 2

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
            stride = self._display_stride

            # Clip to valid range
            x_clipped = np.clip(x.astype(np.int32), 0, 255)
            y_clipped = np.clip(y.astype(np.int32), 0, 255)

            # Convert TOF from seconds to nanoseconds
            tof_ns = tof * 1e9
            # Update total pixel histogram (scaled to compensate for display
            # subsampling — see _display_stride)
            self._pixel_hist += self._bincount_2d(x_clipped, y_clipped) * stride

            # Update TOF histogram
            tof_hist, _ = np.histogram(tof_ns, bins=self._tof_edges)
            self._tof_counts += tof_hist.astype(np.int64) * stride

            # Update mass (m/z) histogram, re-binned on its own mass-uniform grid
            if self._mass_calib_enabled:
                mass = self._tof_ns_to_mass(tof_ns)
                mass_hist, _ = np.histogram(mass, bins=self._mass_edges)
                self._mass_counts += mass_hist.astype(np.int64) * stride

            # Update all ROI histograms
            for _roi_name, roi_data in self._rois.items():
                tof_min = roi_data["tof_min"]
                tof_max = roi_data["tof_max"]
                roi_mask = (tof_ns >= tof_min) & (tof_ns <= tof_max)
                if np.any(roi_mask):
                    roi_data["hist"] += self._bincount_2d(
                        x_clipped[roi_mask], y_clipped[roi_mask]) * stride


            # Per-spatial-ROI TOF histograms (total-image ROIs only).
            has_total_spatial = any(p == "" for (p, _) in self._spatial_rois)
            if has_total_spatial:
                for (p, _n), roi_data in self._spatial_rois.items():
                    if p != "":
                        continue
                    ev_mask = roi_data["mask"][x_clipped, y_clipped]
                    if np.any(ev_mask):
                        h, _ = np.histogram(tof_ns[ev_mask], bins=self._tof_edges)
                        roi_data["tof_hist"] += h.astype(np.int64) * stride
                combined = self._compute_combined_mask_locked("")
                ev_mask = combined[x_clipped, y_clipped]
                if np.any(ev_mask):
                    h, _ = np.histogram(tof_ns[ev_mask], bins=self._tof_edges)
                    if "" not in self._combined_tof_hists:
                        self._combined_tof_hists[""] = np.zeros(self._tof_bins, dtype=np.int64)
                    self._combined_tof_hists[""] += h.astype(np.int64) * stride

            # Per-shot covariance accumulation.
            # Build the (shots × bins) matrix A via np.add.at, then
            # S1 += A.sum(axis=0)  and  S2 += A.T @ A  (single BLAS call).
            if self._cov_enabled and len(event_num) > 0:
                b = np.searchsorted(self._cov_edges[1:], tof_ns).clip(0, self._cov_bins - 1)
                unique_shots, shot_ids = np.unique(event_num, return_inverse=True)
                n_batch = len(unique_shots)
                A = np.zeros((n_batch, self._cov_bins), dtype=np.float64)
                np.add.at(A, (shot_ids, b), 1.0)
                self._cov_S1 += A.sum(axis=0)
                self._cov_S2 += A.T @ A
                self._cov_n_shots += n_batch

            # Count unique trigger pulses in this batch.
            # event_num holds t_trigger — the absolute trigger TIME in seconds (float64),
            # one entry per correlated pixel hit.  Unique values = distinct laser shots.
            # Not scaled by stride: a subsampled shot with fewer hits than the
            # stride could vanish from the batch entirely, so a flat multiply
            # would overstate the shot count rather than correct it.
            if len(event_num) > 0:
                self._n_triggers_seen += len(np.unique(event_num))

            # Update stats
            self._total_events += len(x) * stride

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
            stride = self._display_stride

            x_clipped = np.clip(x.astype(np.int32), 0, 255)
            y_clipped = np.clip(y.astype(np.int32), 0, 255)

            self._pixel_hist += self._bincount_2d(x_clipped, y_clipped) * stride

            # Fill TOA histogram (reuses the same axis as TOF)
            toa_ns = toa * 1e9
            toa_hist, _ = np.histogram(toa_ns, bins=self._tof_edges)
            self._tof_counts += toa_hist.astype(np.int64) * stride

            if self._mass_calib_enabled:
                mass = self._tof_ns_to_mass(toa_ns)
                mass_hist, _ = np.histogram(mass, bins=self._mass_edges)
                self._mass_counts += mass_hist.astype(np.int64) * stride

            # Update ROI histograms filtered by TOA range
            for _roi_name, roi_data in self._rois.items():
                roi_mask = (toa_ns >= roi_data["tof_min"]) & (toa_ns <= roi_data["tof_max"])
                if np.any(roi_mask):
                    roi_data["hist"] += self._bincount_2d(
                        x_clipped[roi_mask], y_clipped[roi_mask]) * stride

            # Per-spatial-ROI TOA histograms (total-image ROIs only).
            has_total_spatial = any(p == "" for (p, _) in self._spatial_rois)
            if has_total_spatial:
                for (p, _n), roi_data in self._spatial_rois.items():
                    if p != "":
                        continue
                    ev_mask = roi_data["mask"][x_clipped, y_clipped]
                    if np.any(ev_mask):
                        h, _ = np.histogram(toa_ns[ev_mask], bins=self._tof_edges)
                        roi_data["tof_hist"] += h.astype(np.int64) * stride
                combined = self._compute_combined_mask_locked("")
                ev_mask = combined[x_clipped, y_clipped]
                if np.any(ev_mask):
                    h, _ = np.histogram(toa_ns[ev_mask], bins=self._tof_edges)
                    if "" not in self._combined_tof_hists:
                        self._combined_tof_hists[""] = np.zeros(self._tof_bins, dtype=np.int64)
                    self._combined_tof_hists[""] += h.astype(np.int64) * stride

            self._total_pixels += len(x) * stride

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

    def get_mass_histogram(self):
        """
        Get current mass (m/z) histogram, re-binned via the TOF→mass calibration.

        Returns
        -------
        bin_centers : np.ndarray
            Center of each bin, in the calibration's mass units (e.g. Da).
        counts : np.ndarray
            Counts in each bin.
        """
        with self._lock:
            return self._mass_centers.copy(), self._mass_counts.copy()

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
            current_trigger_count = self._n_triggers_seen

            delta_counts = current_pixel_count - self._last_sampled_pixel_count
            delta_triggers = current_trigger_count - self._last_sampled_trigger_count
            # In trigger mode: counts/shot.  In pixel mode (no triggers): raw count delta.
            rate = delta_counts / delta_triggers if delta_triggers > 0 else float(delta_counts)

            self._last_sampled_pixel_count = current_pixel_count
            self._last_sampled_trigger_count = current_trigger_count

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
                roi_data["last_sampled_trigger"] = current_trigger_count

                if "timeseries" not in roi_data:
                    roi_data["timeseries"] = []
                roi_data["timeseries"].append((elapsed, roi_rate))
                if len(roi_data["timeseries"]) > self._max_timeseries_points:
                    roi_data["timeseries"].pop(0)

            # Add spatial ROI counts/shot samples (same delta_triggers/elapsed as above)
            for key, roi_data in self._spatial_rois.items():
                parent_arr = self._get_parent_array_locked(key[0])
                if parent_arr is None:
                    continue
                roi_counts = self._sum_mask(parent_arr, roi_data["mask"])
                last_count = roi_data.get("last_sampled_count", 0)

                delta_roi = roi_counts - last_count
                roi_rate = delta_roi / delta_triggers if delta_triggers > 0 else float(delta_roi)

                roi_data["last_sampled_count"] = roi_counts

                roi_data["timeseries"].append((elapsed, roi_rate))
                if len(roi_data["timeseries"]) > self._max_timeseries_points:
                    roi_data["timeseries"].pop(0)

            # Sample the combined mask for every parent that has spatial ROIs
            seen_parents = set(k[0] for k in self._spatial_rois)
            for parent in seen_parents:
                parent_arr = self._get_parent_array_locked(parent)
                if parent_arr is None:
                    continue
                combined_mask = self._compute_combined_mask_locked(parent)
                combined_counts = int((parent_arr * combined_mask).sum())
                last_c = self._combined_last_count.get(parent, 0)
                delta_c = combined_counts - last_c
                rate_c = delta_c / delta_triggers if delta_triggers > 0 else float(delta_c)
                self._combined_last_count[parent] = combined_counts
                ts = self._combined_timeseries.setdefault(parent, [])
                ts.append((elapsed, rate_c))
                if len(ts) > self._max_timeseries_points:
                    ts.pop(0)

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
            for roi_data in self._spatial_rois.values():
                roi_data["timeseries"].clear()

    def clear(self):
        """Clear all histogram data (keeps ROI definitions and timeseries)."""
        with self._lock:
            self._pixel_hist.fill(0)
            self._tof_counts.fill(0)
            self._mass_counts.fill(0)
            self._total_events = 0
            self._total_pixels = 0

            # Reset per-sample baselines so next timeseries point measures from zero.
            # Trigger count is NOT reset (we keep counting shots), but the baseline is
            # advanced to the current count so the delta restarts cleanly from zero.
            self._last_sampled_pixel_count = 0
            self._last_sampled_trigger_count = self._n_triggers_seen

            # Clear ROI histograms but keep definitions
            for roi_data in self._rois.values():
                roi_data["hist"].fill(0)
                roi_data["last_sampled_count"] = 0

            # Reset covariance accumulators
            self._cov_S1.fill(0)
            self._cov_S2.fill(0)
            self._cov_n_shots = 0

            # Reset combined-mask timeseries
            for k in self._combined_timeseries:
                self._combined_timeseries[k] = []
            for k in self._combined_last_count:
                self._combined_last_count[k] = 0

            # Spatial ROIs share the data already in self._pixel_hist / self._rois[*]["hist"],
            # so just reset their sampling baseline and TOF histograms (geometry is untouched).
            for roi_data in self._spatial_rois.values():
                roi_data["last_sampled_count"] = 0
                if "tof_hist" in roi_data:
                    roi_data["tof_hist"].fill(0)
            for arr in self._combined_tof_hists.values():
                arr.fill(0)

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

            # Re-bin spatial ROI TOF histograms to new size
            for roi_data in self._spatial_rois.values():
                if "tof_hist" in roi_data:
                    roi_data["tof_hist"] = np.zeros(self._tof_bins, dtype=np.int64)
            self._combined_tof_hists = {
                k: np.zeros(self._tof_bins, dtype=np.int64)
                for k in self._combined_tof_hists
            }

            # Re-bin covariance to match the new TOF range
            if tof_range is not None:
                self._cov_edges = np.linspace(
                    self._tof_range[0], self._tof_range[1], self._cov_bins + 1)
                self._cov_centers = (self._cov_edges[:-1] + self._cov_edges[1:]) / 2
                self._cov_S1.fill(0)
                self._cov_S2.fill(0)
                self._cov_n_shots = 0

    # =========================================================================
    # TOF -> Mass (m/z) Calibration
    # =========================================================================

    def set_mass_calibration(self, coeff, t0, enabled=True):
        """
        Configure the TOF->mass calibration: tof_ns = coeff * sqrt(mass) + t0.

        Parameters
        ----------
        coeff : float
            Calibration slope (ns / sqrt(mass unit)).
        t0 : float
            Time offset (ns).
        enabled : bool
            Whether incoming events should also be re-binned into the mass histogram.
        """
        with self._lock:
            self._mass_coeff = coeff if coeff else 1.0
            self._mass_t0 = t0
            self._mass_calib_enabled = enabled

    def is_mass_calibration_enabled(self):
        """Return True if the TOF->mass calibration is currently enabled."""
        with self._lock:
            return self._mass_calib_enabled

    def set_mass_config(self, mass_range=None, mass_bins=None):
        """
        Update mass histogram configuration (clears mass data).

        Parameters
        ----------
        mass_range : tuple, optional
            (min, max) in the calibration's mass units.
        mass_bins : int, optional
            Number of bins.
        """
        with self._lock:
            if mass_range is not None:
                self._mass_range = mass_range
            if mass_bins is not None:
                self._mass_bins = mass_bins

            self._mass_edges = np.linspace(
                self._mass_range[0], self._mass_range[1], self._mass_bins + 1
            )
            self._mass_centers = (self._mass_edges[:-1] + self._mass_edges[1:]) / 2
            self._mass_counts = np.zeros(self._mass_bins, dtype=np.int64)

    # =========================================================================
    # Per-shot covariance
    # =========================================================================

    def enable_covariance(self, enabled: bool):
        with self._lock:
            self._cov_enabled = enabled

    def set_covariance_config(self, bins: int = None, cov_range=None):
        """Resize the covariance accumulators and reset all accumulated data."""
        with self._lock:
            if bins is not None:
                self._cov_bins = bins
            if cov_range is not None:
                self._cov_range = cov_range
            else:
                self._cov_range = getattr(self, '_cov_range', self._tof_range)
            self._cov_edges = np.linspace(
                self._cov_range[0], self._cov_range[1], self._cov_bins + 1)
            self._cov_centers = (self._cov_edges[:-1] + self._cov_edges[1:]) / 2
            self._cov_S1 = np.zeros(self._cov_bins, dtype=np.float64)
            self._cov_S2 = np.zeros((self._cov_bins, self._cov_bins), dtype=np.float64)
            self._cov_n_shots = 0

    def get_covariance_map(self):
        """Return (centers_ns, covariance_2d, correlation_2d, n_shots).

        centers_ns  : bin centres in nanoseconds (shape [B])
        covariance  : C = <n·n>/N - outer(<n>/N, <n>/N)  (shape [B,B])
        correlation : <n·n>/N  (shape [B,B])
        n_shots     : number of laser shots accumulated
        """
        with self._lock:
            n = self._cov_n_shots
            centers = self._cov_centers.copy()
            if n < 2:
                z = np.zeros((self._cov_bins, self._cov_bins))
                return centers, z, z, 0
            corr = self._cov_S2 / n
            mean = self._cov_S1 / n
            cov  = corr - np.outer(mean, mean)
            return centers, cov, corr, n

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

    # =========================================================================
    # Spatial (image-space) ROI support
    # =========================================================================
    # NOTE on indexing: _bincount_2d builds flat = x*256 + y then
    # reshape(256, 256), so every 2D histogram array here is indexed [x, y]
    # (not the usual [row=y, col=x]). A rectangle drawn on an image plot at
    # pos=(x, y), size=(w, h) is therefore summed as arr[x:x+w, y:y+h] with
    # no transpose.  Ellipse masks use the same (x, y) convention.

    def _get_parent_array_locked(self, parent):
        """Return the 2D histogram array identified by *parent* (caller must
        already hold self._lock). "" -> main pixel histogram; otherwise a
        TOF-ROI name -> that ROI's filtered histogram. None if not found."""
        if parent == "":
            return self._pixel_hist
        roi_data = self._rois.get(parent)
        return roi_data["hist"] if roi_data is not None else None

    @staticmethod
    def _make_mask(shape: str, x: int, y: int, w: int, h: int,
                   size: int = 256) -> np.ndarray:
        """Compute a boolean pixel mask for *shape* in a *size*×*size* grid.

        shape: "rect" or "ellipse".  (x, y) is the top-left corner of the
        bounding box in [x, y] histogram indexing (x = column, y = row).
        """
        if shape == "rect":
            mask = np.zeros((size, size), dtype=bool)
            x0, x1 = max(0, x), min(size, x + w)
            y0, y1 = max(0, y), min(size, y + h)
            if x1 > x0 and y1 > y0:
                mask[x0:x1, y0:y1] = True
        elif shape == "ellipse":
            cx, cy = x + w / 2.0, y + h / 2.0
            rx, ry = w / 2.0, h / 2.0
            if rx > 0 and ry > 0:
                X, Y = np.ogrid[0:size, 0:size]
                mask = ((X - cx) ** 2 / rx ** 2 + (Y - cy) ** 2 / ry ** 2) <= 1.0
            else:
                mask = np.zeros((size, size), dtype=bool)
        else:
            mask = np.zeros((size, size), dtype=bool)
        return mask

    @staticmethod
    def _sum_mask(arr, mask: np.ndarray) -> int:
        return int((arr * mask).sum())

    def _compute_combined_mask_locked(self, parent: str) -> np.ndarray:
        """Compute union(include) AND NOT union(exclude) for *parent*'s ROIs.
        Returns all-False if there are no include ROIs."""
        N = 256
        include = np.zeros((N, N), dtype=bool)
        exclude = np.zeros((N, N), dtype=bool)
        has_include = False
        for (p, _n), roi_data in self._spatial_rois.items():
            if p != parent:
                continue
            if roi_data.get("op", "+") == "+":
                include |= roi_data["mask"]
                has_include = True
            else:
                exclude |= roi_data["mask"]
        if not has_include:
            return np.zeros((N, N), dtype=bool)
        return include & ~exclude

    def add_spatial_roi(self, parent, name, shape, op, x, y, w, h):
        """Add a spatial ROI (rect or ellipse) on the image identified by *parent*."""
        with self._lock:
            entry = {
                "mask": self._make_mask(shape, x, y, w, h),
                "op": op,
                "last_sampled_count": 0,
                "timeseries": [],
            }
            if parent == "":
                entry["tof_hist"] = np.zeros(self._tof_bins, dtype=np.int64)
            self._spatial_rois[(parent, name)] = entry

    def update_spatial_roi(self, parent, name, shape, op, x, y, w, h):
        """Update an existing spatial ROI's geometry and/or operation."""
        with self._lock:
            key = (parent, name)
            if key not in self._spatial_rois:
                return
            roi_data = self._spatial_rois[key]
            roi_data["mask"] = self._make_mask(shape, x, y, w, h)
            roi_data["op"] = op
            roi_data["last_sampled_count"] = 0
            if parent == "" and "tof_hist" in roi_data:
                roi_data["tof_hist"].fill(0)
            # Combined mask changed — reset its TOF histogram too
            if parent in self._combined_tof_hists:
                self._combined_tof_hists[parent].fill(0)

    def remove_spatial_roi(self, parent, name):
        """Remove a single spatial ROI."""
        with self._lock:
            self._spatial_rois.pop((parent, name), None)
            # Clear combined timeseries cache for this parent so it restarts clean
            self._combined_timeseries.pop(parent, None)
            self._combined_last_count.pop(parent, None)
            # If no more ROIs for this parent, discard combined TOF histogram
            if not any(p == parent for (p, _) in self._spatial_rois):
                self._combined_tof_hists.pop(parent, None)
            elif parent in self._combined_tof_hists:
                # Combined mask changed — restart accumulation
                self._combined_tof_hists[parent].fill(0)

    def remove_spatial_rois_for_parent(self, parent):
        """Remove every spatial ROI belonging to *parent* (used when a TOF
        ROI dock is closed or renamed)."""
        with self._lock:
            for key in [k for k in self._spatial_rois if k[0] == parent]:
                del self._spatial_rois[key]
            self._combined_timeseries.pop(parent, None)
            self._combined_last_count.pop(parent, None)
            self._combined_tof_hists.pop(parent, None)

    def get_spatial_roi_counts(self, parent, name):
        """Get total counts currently inside a spatial ROI."""
        with self._lock:
            roi_data = self._spatial_rois.get((parent, name))
            if roi_data is None:
                return 0
            arr = self._get_parent_array_locked(parent)
            if arr is None:
                return 0
            return self._sum_mask(arr, roi_data["mask"])

    def get_spatial_roi_timeseries(self, parent, name):
        """Get the counts/shot time series for a spatial ROI."""
        with self._lock:
            roi_data = self._spatial_rois.get((parent, name))
            if roi_data is None or not roi_data["timeseries"]:
                return np.array([]), np.array([])
            times = np.array([t for t, _ in roi_data["timeseries"]])
            counts = np.array([c for _, c in roi_data["timeseries"]])
            return times, counts

    def get_spatial_roi_tof(self, parent, name):
        """Return (bin_centers_ns, counts) for the spatially-filtered TOF histogram."""
        with self._lock:
            roi_data = self._spatial_rois.get((parent, name))
            if roi_data is None or "tof_hist" not in roi_data:
                return self._tof_centers.copy(), np.zeros(self._tof_bins, dtype=np.int64)
            return self._tof_centers.copy(), roi_data["tof_hist"].copy()

    def get_combined_tof(self, parent):
        """Return (bin_centers_ns, counts) for the combined-mask TOF histogram."""
        with self._lock:
            counts = self._combined_tof_hists.get(parent)
            if counts is None:
                return self._tof_centers.copy(), np.zeros(self._tof_bins, dtype=np.int64)
            return self._tof_centers.copy(), counts.copy()

    def get_combined_counts(self, parent) -> int:
        """Counts under the combined (include AND NOT exclude) mask for *parent*."""
        with self._lock:
            arr = self._get_parent_array_locked(parent)
            if arr is None:
                return 0
            mask = self._compute_combined_mask_locked(parent)
            return self._sum_mask(arr, mask)

    def get_combined_timeseries(self, parent):
        """Counts/shot time series for the combined mask."""
        with self._lock:
            ts = self._combined_timeseries.get(parent, [])
            if not ts:
                return np.array([]), np.array([])
            times = np.array([t for t, _ in ts])
            counts = np.array([c for _, c in ts])
            return times, counts

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
