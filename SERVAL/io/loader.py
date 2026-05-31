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

from SERVAL.core.data_types import EVENT_DTYPE, PIXEL_DTYPE, TRIGGER_DTYPE


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

        # Index for fast per-trigger slicing
        self._trigger_vals: Optional[np.ndarray] = None
        self._event_starts: Optional[np.ndarray] = None

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
            # Single file — look for metadata alongside it
            meta_path = self._path.parent / (
                self._path.name.replace("_events.dat", "_meta.json")
            )
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
        """EVENT_DTYPE array sorted by t_trigger."""
        if self._events is None:
            self._events = self._load_array("*_events.dat", EVENT_DTYPE, "t_trigger")
            if self._events is None:
                self._events = np.empty(0, dtype=EVENT_DTYPE)
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
        """Subset of triggers used for correlation (tdc_id + rising edge)."""
        t = self.triggers
        if len(t) == 0:
            return t
        mask = (t["tdc_id"] == self._tdc_id) & (t["edge"] == self._edge)
        return t[mask]

    @property
    def pixels(self) -> Optional[np.ndarray]:
        """PIXEL_DTYPE array if *_pixels.dat exists, else None."""
        if self._pixels is None:
            self._pixels = self._load_array("*_pixels.dat", PIXEL_DTYPE, "toa")
        return self._pixels  # may still be None

    # -------------------------------------------------------------------------
    # Derived quantities
    # -------------------------------------------------------------------------

    def absolute_times(self) -> np.ndarray:
        """Absolute pixel hit time = t_trigger + tof (seconds)."""
        ev = self.events
        return ev["t_trigger"] + ev["tof"]

    # -------------------------------------------------------------------------
    # Fast per-trigger slicing
    # -------------------------------------------------------------------------

    def build_event_index(self):
        """
        Build index: unique sorted t_trigger values + searchsorted positions.
        Called automatically on first get_events_for_trigger call.
        """
        ev = self.events
        if len(ev) == 0:
            self._trigger_vals = np.empty(0, dtype=np.float64)
            self._event_starts = np.empty(0, dtype=np.intp)
            return
        unique_vals, first_idx = np.unique(ev["t_trigger"], return_index=True)
        self._trigger_vals = unique_vals
        self._event_starts = first_idx

    def get_events_for_trigger(self, i: int) -> np.ndarray:
        """Return all events for the i-th primary trigger (by index)."""
        if self._trigger_vals is None:
            self.build_event_index()
        starts = self._event_starts
        n = len(starts)
        if i < 0 or i >= n:
            return np.empty(0, dtype=EVENT_DTYPE)
        start = starts[i]
        end = starts[i + 1] if i + 1 < n else len(self.events)
        return self.events[start:end]

    def get_events_in_range(self, start: int, stop: int) -> np.ndarray:
        """Return events for primary trigger indices [start, stop)."""
        if self._trigger_vals is None:
            self.build_event_index()
        starts = self._event_starts
        n = len(starts)
        if start >= n or stop <= 0:
            return np.empty(0, dtype=EVENT_DTYPE)
        start = max(start, 0)
        stop = min(stop, n)
        idx_start = starts[start]
        idx_end = starts[stop] if stop < n else len(self.events)
        return self.events[idx_start:idx_end]

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
