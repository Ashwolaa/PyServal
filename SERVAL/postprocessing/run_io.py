"""
Shared utilities for discovering and grouping PyServal run files.

Both ``centroiding`` (groups ``*_events.dat`` by pipeline-saver split) and
``raw_extraction`` (groups ``*.tpx3`` by raw-saver split) need the same
"group files by step key" pattern.  This module is the single source of truth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional


def _group_files(folder: Path, pattern: str, key_fn: Callable[[Path], str]) -> dict:
    groups: dict = {}
    for f in sorted(Path(folder).glob(pattern)):
        groups.setdefault(key_fn(f), []).append(f)
    return groups


# ---------------------------------------------------------------------------
# Processed-events groups  (*_events.dat, keyed by pipeline-saver split)
# ---------------------------------------------------------------------------

def step_key(event_file: Path) -> str:
    """Run/step identity shared by parallel-saver splits of one take.

    Examples
    --------
    '00001_events.dat'       -> '00001'
    '00001_saver0_events.dat'-> '00001'
    """
    stem = Path(event_file).stem
    if stem.endswith("_events"):
        stem = stem[: -len("_events")]
    return re.sub(r"_saver\d+$", "", stem)


def discover_run_groups(folder: Path) -> dict:
    """Group the ``*_events.dat`` files in *folder* by :func:`step_key`.

    Returns
    -------
    dict[str, list[Path]]
        step_key -> sorted list of that run's event files (``_saver{i}``
        order).  Iteration order matches the sorted glob.
    """
    return _group_files(folder, "*_events.dat", step_key)


# ---------------------------------------------------------------------------
# Raw groups  (*.tpx3, keyed by raw-saver split)
# ---------------------------------------------------------------------------

def raw_step_key(raw_file: Path) -> str:
    """Run/step identity shared by parallel raw-saver splits of one take.

    Examples
    --------
    '00001.tpx3'    -> '00001'
    '00001_raw0.tpx3' -> '00001'
    """
    return re.sub(r"_raw\d+$", "", Path(raw_file).stem)


def discover_raw_groups(folder: Path) -> dict:
    """Group the ``*.tpx3`` files in *folder* by :func:`raw_step_key`.

    Returns
    -------
    dict[str, list[Path]]
        step_key -> sorted list of that run's raw files (``_raw{i}`` order).
        Iteration order matches the sorted glob.
    """
    return _group_files(folder, "*.tpx3", raw_step_key)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_run_meta(folder: Path) -> Optional[dict]:
    """Load a single ``*_meta.json`` in *folder*, or None if absent/ambiguous."""
    candidates = sorted(Path(folder).glob("*_meta.json"))
    if len(candidates) != 1:
        return None
    with open(candidates[0]) as f:
        return json.load(f)
