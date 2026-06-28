#!/usr/bin/env python3
"""
TPX3Run: loader for TPX3 run data produced by TPX3PipelineV3.

Discovers and loads *_events.dat, *_triggers.trg, *_pixels.dat files
from a run directory. With multiple saver files, concatenates and sorts
by t_trigger / toa.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from SERVAL.core.data_types import EVENT_DTYPE, PIXEL_DTYPE, TRIGGER_DTYPE, trigger_combo_bits


class TPX3Run:
    """
    Load a complete TPX3 run from a directory.

    Auto-discovers *_events.dat, *_triggers.trg, *_pixels.dat files.
    With multiple saver files, concatenates and sorts by t_trigger / toa.

    Parameters
    ----------
    path : str or Path
        Path to the run directory (or a single events .dat file).
    tdc_id : int
        TDC channel used for correlation (1 or 2). Used to filter
        primary_triggers. Default: 1 (or read from _meta.json if present).
    """

    def __init__(self, path: Union[str, Path], tdc_id: int = 1):
        self._path = Path(path)
        self._tdc_id = tdc_id
        self._edge: int = 0  # default: rising (overridden by metadata if present)
        self._meta: Optional[dict] = None

        # Lazy-loaded arrays
        self._events: Optional[np.ndarray] = None
        self._triggers: Optional[np.ndarray] = None
        self._pixels: Optional[np.ndarray] = None
        self._centroids: Optional[np.ndarray] = None

        # Cached per-row shot index for events (see events_shot_index)
        self._events_shot_index: Optional[np.ndarray] = None

        self._load_meta()

    def _load_meta(self):
        """Load metadata JSON if present."""
        if self._path.is_dir():
            candidates = sorted(self._path.glob("*_meta.json"))
            if candidates:
                with open(candidates[0]) as f:
                    self._meta = json.load(f)
                # Override tdc_id and edge from metadata
                if "tdc_id" in self._meta:
                    self._tdc_id = self._meta["tdc_id"]
                if "edge" in self._meta:
                    self._edge = self._meta["edge"]
        elif self._path.is_file():
            # Single file — look for metadata alongside it. Prefer an
            # exact-name match (one metadata file per take), but fall back to
            # any *_meta.json in the same directory — e.g. a PyMoDAQ scan's
            # centralized "_scan_meta.json" shared by every step file, which
            # has no per-step counterpart.
            meta_path = self._path.parent / (
                self._path.name.replace("_events.dat", "_meta.json")
            )
            if not meta_path.exists():
                candidates = sorted(self._path.parent.glob("*_meta.json"))
                meta_path = candidates[0] if candidates else meta_path
            if meta_path.exists():
                with open(meta_path) as f:
                    self._meta = json.load(f)
                if "tdc_id" in self._meta:
                    self._tdc_id = self._meta["tdc_id"]
                if "edge" in self._meta:
                    self._edge = self._meta["edge"]

    def _discover_files(self, pattern: str) -> list[Path]:
        """Find files matching pattern in the run directory."""
        if self._path.is_dir():
            return sorted(self._path.glob(pattern))
        elif self._path.is_file():
            # Treat as a single events file; derive siblings by name
            base = self._path.parent
            stem = self._path.stem  # e.g. "run001_events" or "run001_saver0_events"
            # Strip _saverN suffix to get the base run name
            import re
            base_name = re.sub(r"_saver\d+_events$", "", stem)
            base_name = re.sub(r"_events$", "", base_name)
            suffix = pattern.lstrip("*")  # e.g. "_events.dat"
            return sorted(base.glob(f"{base_name}*{suffix}"))
        return []

    def _load_array(self, pattern: str, dtype: np.dtype,
                    sort_field: Optional[str] = None) -> Optional[np.ndarray]:
        """Load and concatenate all files matching pattern."""
        files = self._discover_files(pattern)
        if not files:
            return None
        arrays = [np.fromfile(f, dtype=dtype) for f in files]
        result = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        if sort_field is not None and len(result) > 0:
            order = np.argsort(result[sort_field], kind="stable")
            result = result[order]
        return result

    # -------------------------------------------------------------------------
    # Properties (lazy, cached)
    # -------------------------------------------------------------------------

    @property
    def events(self) -> np.ndarray:
        """
        EVENT_DTYPE array sorted by ``t_trigger``. ``t_trigger`` is rebased
        to be relative to this run's start (``t0``) rather than the chip's
        free-running internal clock — see ``t0`` for why that distinction
        matters. Use ``t0 + events["t_trigger"]`` to recover the original
        chip-clock value (e.g. to cross-reference against ``triggers``).
        """
        if self._events is None:
            raw = self._load_array("*_events.dat", EVENT_DTYPE, "t_trigger")
            if raw is None:
                raw = np.empty(0, dtype=EVENT_DTYPE)
            elif len(raw):
                raw = raw.copy()
                raw["t_trigger"] = raw["t_trigger"] - self.t0
            self._events = raw
        return self._events

    @property
    def triggers(self) -> np.ndarray:
        """TRIGGER_DTYPE array — all channels + edges, sorted by toa."""
        if self._triggers is None:
            self._triggers = self._load_array("*_triggers.trg", TRIGGER_DTYPE, "toa")
            if self._triggers is None:
                self._triggers = np.empty(0, dtype=TRIGGER_DTYPE)
        return self._triggers

    @property
    def primary_triggers(self) -> np.ndarray:
        """Subset of triggers used for correlation (tdc_id + edge)."""
        t = self.triggers
        if len(t) == 0:
            return t
        if self._tdc_id == 0:  # TDCChannel.BOTH — "either channel", never a real packet tag
            mask = t["edge"] == self._edge
        else:
            mask = (t["tdc_id"] == self._tdc_id) & (t["edge"] == self._edge)
        return t[mask]

    def trigger_mask_per_shot(self) -> np.ndarray:
        """
        Per-shot bitmask of which (tdc_id, edge) combos occurred.

        Shot ``i`` spans from ``primary_triggers[i]`` up to (but not
        including) ``primary_triggers[i + 1]``. This returns, for each shot,
        the OR of ``trigger_combo_bits`` over every raw trigger record (any
        channel/edge) falling in that window — i.e. whether an "extra"
        trigger (e.g. a second TDC2 pulse) occurred during the shot used for
        ToF correlation. See ``TRIGGER_BIT_*`` in ``SERVAL.core.data_types``.

        Returns
        -------
        np.ndarray[uint8]
            Same length as ``primary_triggers``.
        """
        t = self.triggers
        if len(t) == 0:
            return np.empty(0, dtype=np.uint8)
        if self._tdc_id == 0:
            ref_mask = t["edge"] == self._edge
        else:
            ref_mask = (t["tdc_id"] == self._tdc_id) & (t["edge"] == self._edge)
        ref_positions = np.flatnonzero(ref_mask)
        if len(ref_positions) == 0:
            return np.empty(0, dtype=np.uint8)
        bit_vals = trigger_combo_bits(t["tdc_id"], t["edge"])
        return np.bitwise_or.reduceat(bit_vals, ref_positions)

    @property
    def pixels(self) -> Optional[np.ndarray]:
        """PIXEL_DTYPE array if *_pixels.dat exists, else None."""
        if self._pixels is None:
            self._pixels = self._load_array("*_pixels.dat", PIXEL_DTYPE, "toa")
        return self._pixels  # may still be None

    @property
    def centroids(self) -> np.ndarray:
        """
        MERGED_CENTROID_DTYPE array from ``<run_name>_centroids.datbin``, as
        written by ``CentroidProcessor.process_run_dir_merged`` (see
        ``SERVAL.postprocessing.centroiding``). Already sorted by
        ``shot_index`` on disk; empty array if the file doesn't exist yet.

        ``shot_index`` indexes into the same ``primary_triggers`` array as
        ``events`` does via ``t_trigger`` — i.e. row ``i`` of
        ``primary_triggers`` corresponds to ``shot_index == i`` here.

        Like ``events``, ``t_trigger`` is rebased relative to this run's
        start (``t0``) — see ``events`` / ``t0`` for why.
        """
        if self._centroids is None:
            # Deferred import: SERVAL.postprocessing.centroiding imports
            # TPX3Run from this module, so importing it at module load time
            # would deadlock.
            from SERVAL.postprocessing.centroiding import MERGED_CENTROID_DTYPE

            if self._path.is_dir():
                run_name = self._path.name
                centroid_file = self._path / f"{run_name}_centroids.datbin"
            else:
                stem = self._path.stem
                import re
                stem = re.sub(r"_saver\d+_events$", "", stem)
                stem = re.sub(r"_events$", "", stem)
                centroid_file = self._path.parent / f"{stem}_centroids.datbin"

            if centroid_file.exists() and centroid_file.stat().st_size > 0:
                c = np.fromfile(str(centroid_file), dtype=MERGED_CENTROID_DTYPE)
                if len(c):
                    c["t_trigger"] = c["t_trigger"] - self.t0
                self._centroids = c
            else:
                self._centroids = np.empty(0, dtype=MERGED_CENTROID_DTYPE)
        return self._centroids

    @property
    def t0(self) -> float:
        """
        Run time origin (seconds) in the chip's free-running clock — the
        clock keeps counting from chip power-on, not from "start
        measurement", so without this offset every timestamp carries an
        arbitrary, often large, baseline (e.g. ~2536 s into a session that
        started well before this particular recording).

        ``events`` and ``centroids`` are pre-rebased by this value (so they
        read close to 0 at the start of the run); ``triggers`` /
        ``primary_triggers`` are left in raw chip-clock seconds, since
        ``t0`` is derived from them.

        First primary trigger's ``toa`` if trigger files are present, else
        the first raw event's ``t_trigger`` (read directly, bypassing the
        ``events`` property, to avoid a circular rebase), else 0.0.
        """
        primary = self.primary_triggers
        if len(primary):
            return float(primary["toa"][0])
        raw = self._load_array("*_events.dat", EVENT_DTYPE, "t_trigger")
        if raw is not None and len(raw):
            return float(raw["t_trigger"][0])
        return 0.0

    @property
    def events_shot_index(self) -> np.ndarray:
        """
        Absolute shot index (0-based index into ``primary_triggers``) for
        each row of ``events`` — computed the same way as
        ``centroids["shot_index"]``: an exact ``np.searchsorted`` of
        ``t_trigger`` against ``primary_triggers["toa"]``. A shot with zero
        events simply doesn't appear here, so this is directly comparable
        to ``centroids["shot_index"]`` (e.g. to find shots with zero counts
        in either array: ``set(range(len(primary_triggers))) - set(idx)``).
        """
        if self._events_shot_index is None:
            ev = self.events
            primary = self.primary_triggers
            if len(ev) == 0 or len(primary) == 0:
                self._events_shot_index = np.empty(0, dtype=np.int64)
            else:
                # ev["t_trigger"] is rebased relative to t0 (see `events`);
                # primary["toa"] is not, so shift it the same way before
                # comparing — searchsorted needs both sides on one basis.
                self._events_shot_index = np.searchsorted(
                    primary["toa"] - self.t0, ev["t_trigger"]
                )
        return self._events_shot_index

    # -------------------------------------------------------------------------
    # Derived quantities
    # -------------------------------------------------------------------------

    def absolute_times(self) -> np.ndarray:
        """
        Pixel hit time = t_trigger + tof (seconds), relative to this run's
        start (``t0``) since ``t_trigger`` is already rebased — add ``t0``
        back for the true chip-clock value.
        """
        ev = self.events
        return ev["t_trigger"] + ev["tof"]

    # -------------------------------------------------------------------------
    # Fast per-shot slicing — events and centroids both indexed by the
    # absolute shot index (position in primary_triggers), via searchsorted
    # on the already-sorted index column. A shot with zero rows just leaves
    # a gap rather than shifting later shots down.
    # -------------------------------------------------------------------------

    def get_events_for_shot(self, i: int) -> np.ndarray:
        """Return all events belonging to the i-th primary trigger / shot."""
        return self.get_events_in_shot_range(i, i + 1)

    def get_events_in_shot_range(self, start: int, stop: int) -> np.ndarray:
        """Return events for primary trigger / shot indices [start, stop)."""
        ev = self.events
        if len(ev) == 0 or stop <= start:
            return np.empty(0, dtype=ev.dtype)
        shot_idx = self.events_shot_index
        idx_start = np.searchsorted(shot_idx, start)
        idx_end = np.searchsorted(shot_idx, stop)
        return ev[idx_start:idx_end]

    def get_centroids_for_shot(self, i: int) -> np.ndarray:
        """Return all centroids belonging to the i-th primary trigger / shot."""
        return self.get_centroids_in_shot_range(i, i + 1)

    def get_centroids_in_shot_range(self, start: int, stop: int) -> np.ndarray:
        """Return centroids for primary trigger / shot indices [start, stop)."""
        c = self.centroids
        if len(c) == 0 or stop <= start:
            return np.empty(0, dtype=c.dtype)
        idx_start = np.searchsorted(c["shot_index"], start)
        idx_end = np.searchsorted(c["shot_index"], stop)
        return c[idx_start:idx_end]

    # -------------------------------------------------------------------------
    # Repr
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        n_ev = len(self.events)
        n_tr = len(self.triggers)
        # Duration estimate from trigger range
        if n_tr > 1:
            duration = self.triggers["toa"][-1] - self.triggers["toa"][0]
            dur_str = f"{duration:.2f}s"
        elif n_ev > 1:
            duration = self.events["t_trigger"][-1] - self.events["t_trigger"][0]
            dur_str = f"{duration:.2f}s"
        else:
            dur_str = "?"
        name = self._path.name
        return (
            f"TPX3Run('{name}'): "
            f"{n_ev/1e6:.2f}M events, {n_tr/1e3:.1f}k triggers, {dur_str}"
        )
