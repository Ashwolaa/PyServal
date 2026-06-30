"""
DBSCAN Pixel Centroiding Post-Processor

Standalone post-processing tool that applies DBSCAN centroiding
to saved PyServal _events.dat files using the pymepixcentroider C++ backend.
"""

import os
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from SERVAL.core.data_types import (
    EVENT_DTYPE,
    TRIGGER_BIT_TDC1_RISING,
    TRIGGER_BIT_TDC1_FALLING,
    TRIGGER_BIT_TDC2_RISING,
    TRIGGER_BIT_TDC2_FALLING,
    decode_trigger_mask,
)
from SERVAL.io import TPX3Run

# Output dtype matching the C++ binary output format: 5 x float64
CENTROID_DTYPE = np.dtype([
    ("t_trigger", "<f8"),  # shot/trigger time (seconds)
    ("x",         "<f8"),  # TOT-weighted centroid x
    ("y",         "<f8"),  # TOT-weighted centroid y
    ("tof",       "<f8"),  # TOT-weighted centroid TOF (seconds)
    ("tot",       "<f8"),  # max TOT in cluster
])

# Merged output dtype — extends CENTROID_DTYPE with an integer shot index
# derived from the run's trigger files, plus a bitmask of which (tdc_id, edge)
# combinations occurred during that shot (see TRIGGER_BIT_* below).
# Sorted by shot_index.
MERGED_CENTROID_DTYPE = np.dtype([
    ("shot_index",   "<u8"),  # 0-based index into the run's reference-trigger array
    ("t_trigger",    "<f8"),
    ("x",            "<f8"),
    ("y",            "<f8"),
    ("tof",          "<f8"),
    ("tot",          "<f8"),
    ("trigger_mask", "<u1"),  # bitmask of (tdc_id, edge) combos seen during this shot
])

# trigger_mask bit meanings: one shot's window runs from its reference trigger
# (the one used for ToF correlation, e.g. TDC1 rising) up to the next
# reference trigger; any (tdc_id, edge) combo seen from the raw *_triggers.trg
# records in that window sets the corresponding TRIGGER_BIT_* (re-exported
# from SERVAL.core.data_types above) — including the reference combo itself,
# which is always set. decode_trigger_mask() is also re-exported from there.


# =============================================================================
# Run-group discovery
# =============================================================================
# step_key / discover_run_groups live in run_io; re-exported here for
# backward compatibility with existing importers.
from SERVAL.postprocessing.run_io import step_key, discover_run_groups  # noqa: F401, E402


# =============================================================================
# Run status
# =============================================================================

class RunStatus(Enum):
    EMPTY = "empty"   # no *_events.dat found
    READY = "ready"   # has *_events.dat, no merged centroid output yet
    DONE  = "done"    # centroid file exists and is non-empty
    STALE = "stale"   # centroid file is older than the newest events file


def get_run_status(event_files: list, centroid_file: Path) -> RunStatus:
    """Return the centroiding status of one run group.

    Parameters
    ----------
    event_files : list[Path]
        The run group's *_events.dat files (see `discover_run_groups`).
    centroid_file : Path
        Where the merged centroid output for this group is (or would be).
    """
    event_files = [Path(f) for f in event_files]
    if not event_files:
        return RunStatus.EMPTY

    centroid_file = Path(centroid_file)
    if not centroid_file.exists() or centroid_file.stat().st_size == 0:
        return RunStatus.READY

    newest_events = max(f.stat().st_mtime for f in event_files)
    if centroid_file.stat().st_mtime < newest_events:
        return RunStatus.STALE

    return RunStatus.DONE


def get_run_info(step_key_: str, event_files: list, centroid_file: Path) -> dict:
    """Return a summary dict for one run group (used to populate the GUI tree).

    Parameters
    ----------
    step_key_ : str
        The run group's step key (see `step_key`/`discover_run_groups`).
    event_files : list[Path]
        The run group's *_events.dat files.
    centroid_file : Path
        Where the merged centroid output for this group is (or would be).
    """
    event_files = [Path(f) for f in event_files]
    centroid_file = Path(centroid_file)

    status = get_run_status(event_files, centroid_file)

    n_centroids = None
    if centroid_file.exists() and centroid_file.stat().st_size > 0:
        n_centroids = centroid_file.stat().st_size // MERGED_CENTROID_DTYPE.itemsize

    mtime = max((f.stat().st_mtime for f in event_files), default=None)

    return {
        "name":          step_key_,
        "path":          event_files[0].parent if event_files else None,
        "centroid_path": centroid_file,
        "status":        status,
        "n_event_files": len(event_files),
        "n_centroids":   n_centroids,
        "mtime":         mtime,
    }

# C++ source and compiled executable (moved to cpp/ subdirectory)
_POSTPROCESSING_DIR = Path(__file__).parent
_EXECUTABLE_PATH = _POSTPROCESSING_DIR / "cpp" / "dbscan_main.exe"
_CPP_SOURCE_PATH = _POSTPROCESSING_DIR / "cpp" / "dbscan_main.cpp"


def detect_tof_peak(tof: np.ndarray, n_bins: int = 2000, width_multiplier: float = 4.0) -> tuple:
    """
    Estimate the (center, half_width) of the dominant ToF peak in seconds.

    The real ToF signal usually sits at an instrument-dependent offset (e.g.
    a time-of-flight delay after the trigger), not at zero, so a simple
    "tof <= threshold" cutoff from zero is the wrong shape of filter — it
    either clips the real peak or, if loosened enough to include it, lets in
    most of the background tail anyway. This instead finds the histogram bin
    with the most points (the peak), measures its FWHM, and returns a window
    of ``width_multiplier`` times that FWHM centered on the peak — meant as a
    starting point for ``tof_min``/``tof_max``, not a final answer.

    Parameters
    ----------
    tof : np.ndarray
        ToF values in seconds (e.g. ``events["tof"]``).
    n_bins : int
        Number of histogram bins spanning the full data range.
    width_multiplier : float
        How many FWHMs wide the returned window should be.

    Returns
    -------
    (center, half_width) : tuple[float, float]
        Both in seconds. The suggested window is
        ``[center - half_width, center + half_width]``.
    """
    tof = np.asarray(tof)
    if len(tof) == 0:
        return 0.0, 0.0

    counts, edges = np.histogram(tof, bins=n_bins)
    peak_bin = int(np.argmax(counts))
    peak_count = counts[peak_bin]
    half_max = peak_count / 2.0

    left = peak_bin
    while left > 0 and counts[left] > half_max:
        left -= 1
    right = peak_bin
    while right < n_bins - 1 and counts[right] > half_max:
        right += 1

    center = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])
    bin_width = edges[1] - edges[0]
    fwhm = max(edges[right + 1] - edges[left], bin_width)
    half_width = fwhm * width_multiplier / 2.0
    return center, half_width


def detect_tof_window(events_dat_path, n_bins: int = 2000, width_multiplier: float = 4.0) -> tuple:
    """
    Convenience wrapper: load a ``*_events.dat`` file and return a suggested
    ``(tof_min, tof_max)`` window around its dominant ToF peak (seconds).
    See ``detect_tof_peak`` for details.
    """
    events = np.fromfile(str(events_dat_path), dtype=EVENT_DTYPE)
    center, half_width = detect_tof_peak(events["tof"], n_bins=n_bins, width_multiplier=width_multiplier)
    return center - half_width, center + half_width


class CentroidProcessor:
    """
    Wraps dbscan_main.exe for post-processing PyServal event files.

    Parameters
    ----------
    executable_path : str or Path, optional
        Path to the compiled dbscan_main.exe. Defaults to
        pymepixcentroider/dbscan_main.exe relative to this package.
    epsilon : float
        Spatial (x, y) clustering radius, in pixels. Two pixel hits can only
        join the same cluster if they are within this distance of each other
        AND within ``eps_time`` of each other in ToF — these are independent
        criteria. Default: 2.0.
    eps_time : float
        ToF clustering radius, in seconds. Controls the minimum ToF
        separation needed for two pixel hits to be treated as *different*
        ions/clusters (e.g. two different ion masses arriving at different
        times within the same shot) rather than the same one. This is
        unrelated to ``tof_min``/``tof_max`` below — it's the clustering
        granularity, not a pre-filter. Default: 100 ns, matching the live
        pipeline's greedy centroiding default.
    tof_min, tof_max : float
        ToF acceptance window (seconds) — a point is kept only if
        ``tof_min <= tof <= tof_max``. The real ToF peak usually sits at a
        nonzero, instrument-dependent offset, so this should be a window
        around that peak (see ``detect_tof_window``), not just an upper
        bound from zero. Defaults to an effectively unbounded window
        (0, 1 second) — i.e. no filtering — until set explicitly.
    min_points : int
        Minimum cluster size for DBSCAN. Default: 1.
    backend : str
        ``"cpp"`` (default) runs the compiled ``dbscan_main.exe`` via
        subprocess on a temp/output file — same algorithm, but pays
        process-spawn + file I/O cost on every call. ``"numba"`` runs the
        identical algorithm in-process via ``dbscan_numba`` — no subprocess,
        but the *first* call in a given process pays a JIT compile/cache-load
        cost; subsequent calls in the same (long-lived) process are
        substantially faster. For a one-shot CLI invocation the two are
        comparable; for a long-lived process calling this repeatedly (e.g.
        the centroiding GUI processing many files in one session), "numba"
        wins decisively once warm.
    """

    def __init__(
        self,
        executable_path: Optional[str] = None,
        epsilon: float = 2.0,
        eps_time: float = 100e-9,
        tof_min: float = 0.0,
        tof_max: float = 1.0,
        min_points: int = 1,
        backend: str = "cpp",
    ):
        if backend not in ("cpp", "numba"):
            raise ValueError(f"backend must be 'cpp' or 'numba', got {backend!r}")
        self.executable_path = Path(executable_path) if executable_path else _EXECUTABLE_PATH
        self.epsilon = epsilon
        self.eps_time = eps_time
        self.tof_min = tof_min
        self.tof_max = tof_max
        self.min_points = min_points
        self.backend = backend

    def compile(self, force: bool = False) -> bool:
        """
        Compile dbscan_main.cpp to dbscan_main.exe.

        Parameters
        ----------
        force : bool
            If True, recompile even if the executable already exists.

        Returns
        -------
        bool
            True if compilation succeeded or executable already exists.
        """
        if self.executable_path.exists() and not force:
            return True

        cpp_file = str(_CPP_SOURCE_PATH)
        out_file = str(self.executable_path)

        try:
            subprocess.run(
                ["g++", cpp_file, "-o", out_file, "-O2"],
                capture_output=True,
                text=True,
                check=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed:\n{e.stderr}")
            return False
        except FileNotFoundError:
            print("g++ not found. Please install a C++ compiler.")
            return False

    def process_file(
        self,
        events_dat_path: str,
        output_path: Optional[str] = None,
        correction_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        diagnostics: bool = False,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> np.ndarray:
        """
        Run DBSCAN centroiding on a single _events.dat file.

        Parameters
        ----------
        events_dat_path : str
            Path to the _events.dat file.
        output_path : str, optional
            Path for the output .datbin file. If None, uses
            <stem>_centroids.datbin in the same directory.
        correction_path : str, optional
            Path to a TOF correction .txt file (tof,correction CSV format).
        labels_path : str, optional
            If provided, write per-point cluster labels to this path.
        diagnostics : bool
            If True, print timing output from the C++ executable.
        progress_callback : callable(percent: int, phase: str), optional
            Called with (0-100, phase_name) as the C++ reports progress.
            Phase names: "reading", "grouping", "dbscan", "clustering",
            "writing", "done".

        Returns
        -------
        np.ndarray
            Structured array with dtype CENTROID_DTYPE.
        """
        events_dat_path = Path(events_dat_path)

        if output_path is None:
            stem = events_dat_path.stem
            if stem.endswith("_events"):
                stem = stem[: -len("_events")]
            output_path = events_dat_path.parent / f"{stem}_centroids.datbin"
        output_path = Path(output_path)

        if events_dat_path.stat().st_size == 0:
            return np.array([], dtype=CENTROID_DTYPE)

        if self.backend == "numba":
            if labels_path is not None:
                raise NotImplementedError("labels_path is only supported by the 'cpp' backend")
            from SERVAL.postprocessing import dbscan_numba

            if progress_callback:
                progress_callback(0, "reading")
            centroids = dbscan_numba.process_file(
                events_dat_path,
                epsilon=self.epsilon,
                eps_time=self.eps_time,
                min_points=self.min_points,
                tof_min=self.tof_min,
                tof_max=self.tof_max,
                correction_path=correction_path,
            )
            centroids.tofile(str(output_path))
            if progress_callback:
                progress_callback(100, "done")
            return centroids

        # Compile if needed
        if not self.executable_path.exists():
            if not self.compile():
                raise RuntimeError(
                    f"Executable not found and compilation failed: {self.executable_path}"
                )

        # dbscan_main.exe reads _events.dat directly — its on-disk EVENT_DTYPE
        # layout (data_types.py) is read byte-for-byte by the C++ RawRecord
        # struct, so no conversion or intermediate file is needed.
        cmd = [
            str(self.executable_path),
            str(events_dat_path),
            str(output_path),
            "--epsilon", str(self.epsilon),
            "--eps-time", str(self.eps_time),
            "--min-points", str(self.min_points),
            "--tof-min", str(self.tof_min),
            "--tof-max", str(self.tof_max),
        ]
        if correction_path is not None:
            cmd.append(str(correction_path))
        if labels_path is not None:
            cmd.append(str(labels_path))

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )

        # Collect stderr in background to avoid deadlock
        stderr_lines = []

        def _read_stderr():
            for line in process.stderr:
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        # Read stdout line by line, parsing PROGRESS and PHASE markers
        current_phase = "reading"
        for line in process.stdout:
            line = line.rstrip()
            if line.startswith("PROGRESS:"):
                try:
                    pct = int(line.split(":")[1].strip())
                    if progress_callback:
                        progress_callback(pct, current_phase)
                except ValueError:
                    pass
            elif line.startswith("PHASE:"):
                current_phase = line.split(":")[1].strip()
            elif diagnostics and line:
                print(line)

        process.wait()
        stderr_thread.join()
        stderr = "".join(stderr_lines)

        if stderr:
            print(f"[centroiding stderr] {stderr}", end="")

        if process.returncode != 0:
            raise RuntimeError(
                f"dbscan_main.exe exited with code {process.returncode}:\n{stderr}"
            )

        if progress_callback:
            progress_callback(100, "done")

        # Labels file format differs; caller reads it directly
        if labels_path is not None:
            return np.array([], dtype=CENTROID_DTYPE)

        if not output_path.exists() or output_path.stat().st_size == 0:
            return np.array([], dtype=CENTROID_DTYPE)

        return np.fromfile(str(output_path), dtype=CENTROID_DTYPE)

    def process_event_group(
        self,
        event_files,
        output_path,
        correction_path: Optional[str] = None,
        labels: bool = False,
        diagnostics: bool = False,
        progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
        force: bool = False,
    ) -> Path:
        """
        Process one run group's *_events.dat files in parallel, then merge
        results into a single ``output_path`` file sorted by ``shot_index``.

        A "run group" is one take's event files — normally just one, or
        several only when split across parallel saver processes (the
        ``_saverN`` suffix from ``pipeline.py``'s ``start_record``); see
        ``discover_run_groups``. Shot indices come from the *_triggers.trg
        file(s) belonging to this same run (matched via ``TPX3Run`` in
        single-file mode, which scopes sibling-file discovery to this run's
        own name — not any other run's files that might share the folder).

        Parameters
        ----------
        event_files : list[Path]
            The run group's ``*_events.dat`` files (e.g. from
            ``discover_run_groups``).
        output_path : Path
            Destination for the merged centroid ``.datbin`` file.
        correction_path : str, optional
            TOF correction file passed to each C++ worker.
        labels : bool
            If True, generate per-file ``.toflabels`` files alongside the
            event files (labels are not merged).
        diagnostics : bool
            If True, print C++ timing output.
        progress_callback : callable(file_idx, n_files, overall_pct, phase)
            Called from worker threads; must be thread-safe.
        force : bool
            If False and output_path already exists, return immediately.

        Returns
        -------
        Path
            output_path.
        """
        event_files = sorted(Path(f) for f in event_files)
        if not event_files:
            raise RuntimeError("No event files given to process_event_group")
        output_path = Path(output_path)

        if output_path.exists() and not force:
            return output_path

        n_files = len(event_files)
        file_progress = [0] * n_files
        progress_lock = threading.Lock()

        def make_cb(idx):
            def cb(pct, phase):
                with progress_lock:
                    file_progress[idx] = pct
                    overall = sum(file_progress) // n_files
                if progress_callback:
                    progress_callback(idx, n_files, overall, phase)
            return cb

        # --- parallel per-file centroiding -----------------------------------
        tmp_paths = [None] * n_files

        def process_one(idx):
            event_file = event_files[idx]
            tmp = tempfile.NamedTemporaryFile(suffix=".datbin", delete=False)
            tmp.close()
            tmp_path = Path(tmp.name)
            tmp_paths[idx] = tmp_path

            labels_path = None
            if labels:
                stem = event_file.stem
                if stem.endswith("_events"):
                    stem = stem[: -len("_events")]
                labels_path = str(event_file.parent / f"{stem}.toflabels")

            self.process_file(
                str(event_file),
                output_path=str(tmp_path),
                correction_path=correction_path,
                labels_path=labels_path,
                diagnostics=diagnostics,
                progress_callback=make_cb(idx),
            )

            # Always read from the output file (process_file returns [] when
            # labels_path is set, but the .datbin is still written).
            if tmp_path.exists() and tmp_path.stat().st_size > 0:
                return np.fromfile(str(tmp_path), dtype=CENTROID_DTYPE)
            return np.array([], dtype=CENTROID_DTYPE)

        centroid_arrays = {}
        max_workers = min(n_files, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(process_one, i): i for i in range(n_files)}
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    centroid_arrays[idx] = future.result()
                except Exception as e:
                    centroid_arrays[idx] = np.array([], dtype=CENTROID_DTYPE)
                    if diagnostics:
                        print(f"[worker {idx}] error: {e}")

        # --- build shot index + per-shot trigger-presence mask ---------------
        # TPX3Run in single-file mode derives sibling files (here,
        # *_triggers.trg) from event_files[0]'s own stripped base name, so
        # this only ever sees this run's trigger files — not any other run's,
        # even if they share the same folder (e.g. other scan steps).
        # It also reads this run's *_meta.json for the (tdc_id, edge) combo
        # used as the ToF correlation reference (e.g. TDC1 rising) — that
        # defines "one shot". primary_triggers is the array t_trigger values
        # are matched against; trigger_mask_per_shot() is a parallel array
        # recording, for each shot's window, which of the 4 (tdc_id, edge)
        # combos appeared anywhere in the raw trigger stream (TRIGGER_BIT_*).
        ref_toa = None
        shot_trigger_mask = None
        run = TPX3Run(event_files[0])
        primary = run.primary_triggers
        if len(primary):
            ref_toa = primary["toa"]
            shot_trigger_mask = run.trigger_mask_per_shot()

        # --- concatenate, assign shot_index + trigger_mask, sort, write -------
        valid = [centroid_arrays[i] for i in range(n_files) if len(centroid_arrays[i])]
        if not valid:
            output_path.write_bytes(b"")
            return output_path

        all_c = np.concatenate(valid)
        merged = np.empty(len(all_c), dtype=MERGED_CENTROID_DTYPE)
        merged["t_trigger"] = all_c["t_trigger"]
        merged["x"]         = all_c["x"]
        merged["y"]         = all_c["y"]
        merged["tof"]       = all_c["tof"]
        merged["tot"]       = all_c["tot"]

        if ref_toa is not None and len(ref_toa):
            # searchsorted is exact: t_trigger values come directly from the
            # same reference trigger packets stored in the trigger files.
            merged["shot_index"]   = np.searchsorted(ref_toa, all_c["t_trigger"])
            merged["trigger_mask"] = shot_trigger_mask[merged["shot_index"]]
        else:
            # Fallback when no trigger files (or no matching reference triggers) exist
            merged["shot_index"]   = np.argsort(all_c["t_trigger"], kind="stable")
            merged["trigger_mask"] = 0

        order = np.argsort(merged["shot_index"], kind="stable")
        merged[order].tofile(str(output_path))

        # --- cleanup ---------------------------------------------------------
        for tmp_path in tmp_paths:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        return output_path

    def process_run_dir_merged(
        self,
        run_dir,
        correction_path: Optional[str] = None,
        labels: bool = False,
        diagnostics: bool = False,
        progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
        force: bool = False,
    ) -> dict:
        """
        Process every run group in a folder (see ``discover_run_groups``),
        writing one merged ``{step_key}_centroids.datbin`` per group — never
        merging different scan steps/takes together even if they share the
        folder.

        Parameters
        ----------
        run_dir : Path
            Folder containing one or more runs' ``*_events.dat`` files.
        correction_path, labels, diagnostics, force :
            See ``process_event_group``.
        progress_callback : callable(file_idx, n_files, overall_pct, phase)
            Forwarded to each group's ``process_event_group`` call in turn.

        Returns
        -------
        dict[str, Path]
            Mapping of step_key -> path to that group's merged centroid file.
        """
        run_dir = Path(run_dir)
        groups = discover_run_groups(run_dir)
        if not groups:
            raise RuntimeError(f"No *_events.dat files found in {run_dir}")

        results = {}
        for key, event_files in groups.items():
            output_path = run_dir / f"{key}_centroids.datbin"
            results[key] = self.process_event_group(
                event_files,
                output_path,
                correction_path=correction_path,
                labels=labels,
                diagnostics=diagnostics,
                progress_callback=progress_callback,
                force=force,
            )
        return results

    def process_run_dir(
        self,
        run_dir: str,
        output_dir: Optional[str] = None,
        correction_path: Optional[str] = None,
        labels: bool = False,
        diagnostics: bool = False,
    ) -> dict:
        """
        Process all *_events.dat files in a run directory.

        Parameters
        ----------
        run_dir : str
            Directory containing *_events.dat files.
        output_dir : str, optional
            Directory for output files. Defaults to run_dir.
        correction_path : str, optional
            Path to TOF correction file.
        labels : bool
            If True, also generate .toflabels files.
        diagnostics : bool
            If True, print C++ timing output.

        Returns
        -------
        dict
            Mapping of input file path to centroid array (or None on error).
        """
        run_dir = Path(run_dir)
        out_dir = Path(output_dir) if output_dir else run_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        event_files = sorted(run_dir.glob("*_events.dat"))

        if not event_files:
            print(f"No *_events.dat files found in {run_dir}")
            return results

        for event_file in event_files:
            stem = event_file.stem[: -len("_events")]
            output_path = out_dir / f"{stem}_centroids.datbin"
            labels_path = str(out_dir / f"{stem}.toflabels") if labels else None

            try:
                centroids = self.process_file(
                    str(event_file),
                    output_path=str(output_path),
                    correction_path=correction_path,
                    labels_path=labels_path,
                    diagnostics=diagnostics,
                )
                results[str(event_file)] = centroids
                print(f"Processed {event_file.name}: {len(centroids)} centroids")
            except Exception as e:
                print(f"Error processing {event_file.name}: {e}")
                results[str(event_file)] = None

        return results


def main():
    """Command-line entry point for headless processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DBSCAN centroiding post-processor for PyServal _events.dat files"
    )
    parser.add_argument("input", help="Path to _events.dat file or run directory")
    parser.add_argument("-o", "--output", help="Output path or directory")
    parser.add_argument("--epsilon", type=float, default=2.0,
                         help="Spatial clustering radius, pixels")
    parser.add_argument("--eps-time", type=float, default=100e-9,
                         help="ToF clustering radius, seconds — minimum ToF "
                              "separation for two hits to be different clusters")
    parser.add_argument("--tof-min", type=float, default=0.0,
                         help="ToF window lower bound, seconds")
    parser.add_argument("--tof-max", type=float, default=1.0,
                         help="ToF window upper bound, seconds")
    parser.add_argument("--detect-tof-peak", action="store_true",
                         help="Ignore --tof-min/--tof-max and auto-detect a window "
                              "around the dominant ToF peak (single-file input only)")
    parser.add_argument("--min-points", type=int, default=1)
    parser.add_argument("--correction", help="Path to TOF correction .txt file")
    parser.add_argument("--labels", action="store_true", help="Generate .toflabels output")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    tof_min, tof_max = args.tof_min, args.tof_max
    if args.detect_tof_peak:
        if Path(args.input).is_dir():
            print("--detect-tof-peak requires a single _events.dat file, not a directory")
            return
        tof_min, tof_max = detect_tof_window(args.input)
        print(f"Detected ToF window: [{tof_min*1e9:.1f}, {tof_max*1e9:.1f}] ns")

    proc = CentroidProcessor(
        epsilon=args.epsilon,
        eps_time=args.eps_time,
        tof_min=tof_min,
        tof_max=tof_max,
        min_points=args.min_points,
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        proc.process_run_dir(
            str(input_path),
            output_dir=args.output,
            correction_path=args.correction,
            labels=args.labels,
            diagnostics=args.diagnostics,
        )
    else:
        centroids = proc.process_file(
            str(input_path),
            output_path=args.output,
            correction_path=args.correction,
            diagnostics=args.diagnostics,
        )
        print(f"Done: {len(centroids)} centroids")


if __name__ == "__main__":
    main()
