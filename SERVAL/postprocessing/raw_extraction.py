"""
Raw .tpx3 -> pixels/triggers/events extraction.

Recomputes pixels, triggers, and correlated (non-centroided) events from raw
``.tpx3`` files, using the same JIT extraction (``SERVAL.core.extractors``)
and correlation (``TPX3Correlator``) code the live pipeline uses. Mirrors
``SERVAL.postprocessing.centroiding``'s run-group conventions, but keyed off
a run's raw ``.tpx3`` file(s) rather than its ``*_events.dat`` outputs — this
is the stage that *produces* those event files in the first place.

Run the existing DBSCAN centroiding post-processor
(``SERVAL.postprocessing.centroiding``) on the resulting ``*_events.dat`` if
you want centroided events as a separate, clearly-named file.
"""

import json
import re
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from SERVAL.core.data_types import (
    EVENT_DTYPE,
    PIXEL_DTYPE,
    TRIGGER_DTYPE,
    PixelData,
    TriggerData,
    TDCChannel,
    TriggerEdge,
    merge_pixels,
    merge_triggers,
)
from SERVAL.core.extractors.parallel_processor import TPX3Extractor, TPX3Correlator
from SERVAL.postprocessing.centroiding import RunStatus

# =============================================================================
# Run-group discovery
# =============================================================================
#
# A "run group" is the set of *.tpx3 files belonging to one take. Within one
# take, the only thing that can multiply a single recording into several
# files is being split across parallel raw-saver processes (pipeline.py's
# `start_record` suffixes each with `_raw{i}`, distinct from events' own
# `_saver{i}` — see centroiding.step_key) — those splits are interleaved
# chunks of one continuous stream and must be concatenated before
# correlation, not processed independently. Different scan steps sharing a
# folder must NOT be merged together.

def raw_step_key(raw_file: Path) -> str:
    """Run/step identity shared by parallel-raw-saver splits of one take.

    Examples
    --------
    '00001.tpx3' -> '00001'
    '00001_raw0.tpx3' -> '00001'
    """
    return re.sub(r"_raw\d+$", "", Path(raw_file).stem)


def discover_raw_groups(folder: Path) -> dict:
    """Group the *.tpx3 files directly inside `folder` by raw_step_key.

    Returns
    -------
    dict[str, list[Path]]
        Mapping of step_key -> sorted list of that run's raw files (in
        ``_raw{i}`` order when split). Iteration order matches the sorted
        glob, so single-file folders keep their natural order.
    """
    groups: dict = {}
    for f in sorted(Path(folder).glob("*.tpx3")):
        groups.setdefault(raw_step_key(f), []).append(f)
    return groups


def load_run_meta(folder: Path) -> Optional[dict]:
    """Auto-detect and load a single ``*_meta.json`` in `folder`.

    Returns None if none or more than one candidate is found — callers fall
    back to explicit settings/defaults in that case.
    """
    candidates = sorted(Path(folder).glob("*_meta.json"))
    if len(candidates) != 1:
        return None
    with open(candidates[0]) as f:
        return json.load(f)


def get_raw_status(raw_files: list, events_file: Path) -> RunStatus:
    """Return the raw->events extraction status of one run group.

    Parameters
    ----------
    raw_files : list[Path]
        The run group's ``*.tpx3`` files (see ``discover_raw_groups``).
    events_file : Path
        Where the extracted ``_events.dat`` for this group is (or would be).
    """
    raw_files = [Path(f) for f in raw_files]
    if not raw_files:
        return RunStatus.EMPTY

    events_file = Path(events_file)
    if not events_file.exists() or events_file.stat().st_size == 0:
        return RunStatus.READY

    newest_raw = max(f.stat().st_mtime for f in raw_files)
    if events_file.stat().st_mtime < newest_raw:
        return RunStatus.STALE

    return RunStatus.DONE


# =============================================================================
# Extraction + correlation
# =============================================================================

class RawEventProcessor:
    """
    Recomputes pixels/triggers/events from one run group's raw ``.tpx3``
    file(s), using the same JIT extraction and correlation the live pipeline
    uses.

    Parameters
    ----------
    tdc_id : TDCChannel
        Reference TDC channel used for ToF correlation.
    edge : TriggerEdge
        Reference trigger edge.
    window_ns : tuple[float, float]
        Correlation window (min, max), in nanoseconds: a pixel is kept only
        if its time-of-flight from the nearest preceding reference trigger
        falls in this window. This is the correlation window, not the
        post-hoc ToF filter applied later by centroiding.
    save_pixels, save_triggers : bool
        If True, also persist the extracted ``*_pixels.dat`` /
        ``*_triggers.trg`` files (``PIXEL_DTYPE`` / ``TRIGGER_DTYPE``),
        alongside the always-written ``*_events.dat``. Off by default —
        most uses only need correlated events.
    chunk_mb : int
        Read block size (MiB) when streaming each raw file through
        extraction.
    """

    def __init__(
        self,
        tdc_id: TDCChannel = TDCChannel.TDC1,
        edge: TriggerEdge = TriggerEdge.RISING,
        window_ns: Tuple[float, float] = (0.0, 100_000.0),
        save_pixels: bool = False,
        save_triggers: bool = False,
        chunk_mb: int = 64,
    ):
        self.tdc_id = tdc_id
        self.edge = edge
        self.window_ns = window_ns
        self.save_pixels = save_pixels
        self.save_triggers = save_triggers
        self.chunk_bytes = chunk_mb * 2**20

    def extract_raw_group(
        self,
        raw_files: list,
        progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    ) -> Tuple[Optional[PixelData], Optional[TriggerData]]:
        """
        Extract pixels/triggers from one run group's raw ``.tpx3`` file(s).

        Splits (parallel-raw-saver ``_raw{i}`` files of the same take) are
        concatenated into single pixel/trigger arrays — they are interleaved
        chunks of one continuous stream, not independent recordings.

        Parameters
        ----------
        raw_files : list[Path]
            The run group's raw files (see ``discover_raw_groups``).
        progress_callback : callable(file_idx, n_files, pct, phase), optional

        Returns
        -------
        (pixels, triggers) : tuple[PixelData | None, TriggerData | None]
            None for either if no data of that kind was found.
        """
        raw_files = sorted(Path(f) for f in raw_files)
        extractor = TPX3Extractor()
        pixel_chunks, trigger_chunks = [], []

        for file_idx, raw_path in enumerate(raw_files):
            remaining = b""
            file_size = raw_path.stat().st_size
            bytes_read = 0
            with open(raw_path, "rb") as fraw:
                while True:
                    block = fraw.read(self.chunk_bytes)
                    if not block:
                        break
                    bytes_read += len(block)
                    data = remaining + block
                    pixels, triggers, remaining, _ = extractor.extract_fast(data)
                    if len(pixels) > 0:
                        pixel_chunks.append(pixels)
                    if len(triggers) > 0:
                        trigger_chunks.append(triggers)
                    if progress_callback:
                        pct = int(bytes_read / file_size * 100) if file_size else 100
                        progress_callback(file_idx, len(raw_files), pct, "extracting")

        pixels = merge_pixels(*pixel_chunks) if pixel_chunks else None
        triggers = merge_triggers(*trigger_chunks) if trigger_chunks else None
        return pixels, triggers

    def correlate(self, pixels: Optional[PixelData], triggers: Optional[TriggerData]) -> np.ndarray:
        """Correlate extracted pixels to the reference trigger -> EVENT_DTYPE array."""
        if pixels is None or triggers is None:
            return np.empty(0, dtype=EVENT_DTYPE)

        correlator = TPX3Correlator(
            triggers, event_window=self.window_ns, tdc_id=int(self.tdc_id), edge=int(self.edge)
        )
        result = correlator.correlate(pixels)
        if result is None:
            return np.empty(0, dtype=EVENT_DTYPE)

        event_indices, x, y, tof, tot = result
        events = np.zeros(len(x), dtype=EVENT_DTYPE)
        events["t_trigger"] = correlator.triggers[event_indices]
        events["x"] = x
        events["y"] = y
        events["tof"] = tof
        events["tot"] = tot
        events.sort(order="t_trigger")
        return events

    def process_raw_group(
        self,
        raw_files: list,
        events_out: Path,
        pixels_out: Optional[Path] = None,
        triggers_out: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
        force: bool = False,
    ) -> Path:
        """
        Extract + correlate one run group's raw file(s) into ``events_out``.

        Parameters
        ----------
        raw_files : list[Path]
            The run group's ``*.tpx3`` files (see ``discover_raw_groups``).
        events_out : Path
            Destination for the extracted ``_events.dat`` file.
        pixels_out, triggers_out : Path, optional
            Destinations for the optional ``_pixels.dat`` / ``_triggers.trg``
            outputs (only written if ``save_pixels``/``save_triggers`` are
            set). Default next to ``events_out`` if not given.
        progress_callback : callable(file_idx, n_files, pct, phase), optional
        force : bool
            If False and ``events_out`` is already up to date (see
            ``get_raw_status``), return immediately without re-extracting.

        Returns
        -------
        Path
            ``events_out``.
        """
        raw_files = sorted(Path(f) for f in raw_files)
        if not raw_files:
            raise RuntimeError("No raw files given to process_raw_group")
        events_out = Path(events_out)

        if not force and get_raw_status(raw_files, events_out) == RunStatus.DONE:
            return events_out

        pixels, triggers = self.extract_raw_group(raw_files, progress_callback=progress_callback)

        if progress_callback:
            progress_callback(len(raw_files) - 1, len(raw_files), 0, "correlating")
        events = self.correlate(pixels, triggers)

        events_out.parent.mkdir(parents=True, exist_ok=True)
        events.tofile(str(events_out))

        base = re.sub(r"_events$", "", events_out.stem)

        if self.save_pixels and pixels is not None:
            pixels_out = Path(pixels_out) if pixels_out else events_out.with_name(f"{base}_pixels.dat")
            pixel_arr = np.zeros(len(pixels), dtype=PIXEL_DTYPE)
            pixel_arr["x"] = pixels.x
            pixel_arr["y"] = pixels.y
            pixel_arr["toa"] = pixels.toa
            pixel_arr["tot"] = pixels.tot
            pixel_arr.tofile(str(pixels_out))

        if self.save_triggers and triggers is not None:
            triggers_out = Path(triggers_out) if triggers_out else events_out.with_name(f"{base}_triggers.trg")
            trig_arr = np.zeros(len(triggers), dtype=TRIGGER_DTYPE)
            trig_arr["toa"] = triggers.toa
            trig_arr["tdc_id"] = triggers.tdc_id
            trig_arr["edge"] = triggers.edge
            trig_arr.tofile(str(triggers_out))

        if progress_callback:
            progress_callback(len(raw_files) - 1, len(raw_files), 100, "done")

        return events_out
