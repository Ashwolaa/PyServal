"""
DBSCAN Pixel Centroiding Post-Processor

Standalone post-processing tool that applies DBSCAN centroiding
to saved PyServal _events.dat files using the pymepixcentroider C++ backend.
"""

import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from SERVAL.core.data_types import EVENT_DTYPE

# Output dtype matching the C++ binary output format: 5 x float64
CENTROID_DTYPE = np.dtype([
    ("t_trigger", "<f8"),  # shot/trigger time (seconds)
    ("x",         "<f8"),  # TOT-weighted centroid x
    ("y",         "<f8"),  # TOT-weighted centroid y
    ("tof",       "<f8"),  # TOT-weighted centroid TOF (seconds)
    ("tot",       "<f8"),  # max TOT in cluster
])

# Merged output dtype — extends CENTROID_DTYPE with an integer shot index
# derived from the run's trigger files. Sorted by shot_index.
MERGED_CENTROID_DTYPE = np.dtype([
    ("shot_index", "<u8"),  # index into the run's sorted trigger array (0-based)
    ("t_trigger",  "<f8"),
    ("x",          "<f8"),
    ("y",          "<f8"),
    ("tof",        "<f8"),
    ("tot",        "<f8"),
])


# =============================================================================
# Run status
# =============================================================================

class RunStatus(Enum):
    EMPTY = "empty"   # no *_events.dat found
    READY = "ready"   # has *_events.dat, no merged centroid output yet
    DONE  = "done"    # <folder>_centroids.datbin exists and is non-empty
    STALE = "stale"   # centroid file is older than the newest events file


def get_run_status(run_dir: Path) -> RunStatus:
    """Return the centroiding status of a run folder."""
    run_dir = Path(run_dir)
    event_files = list(run_dir.glob("*_events.dat"))
    if not event_files:
        return RunStatus.EMPTY

    centroid_file = run_dir / f"{run_dir.name}_centroids.datbin"
    if not centroid_file.exists() or centroid_file.stat().st_size == 0:
        return RunStatus.READY

    newest_events = max(f.stat().st_mtime for f in event_files)
    if centroid_file.stat().st_mtime < newest_events:
        return RunStatus.STALE

    return RunStatus.DONE


def get_run_info(run_dir: Path) -> dict:
    """Return a summary dict for a run folder (used to populate the GUI table)."""
    run_dir = Path(run_dir)
    event_files = sorted(run_dir.glob("*_events.dat"))
    centroid_file = run_dir / f"{run_dir.name}_centroids.datbin"

    status = get_run_status(run_dir)

    n_centroids = None
    if centroid_file.exists() and centroid_file.stat().st_size > 0:
        n_centroids = centroid_file.stat().st_size // MERGED_CENTROID_DTYPE.itemsize

    mtime = max((f.stat().st_mtime for f in event_files), default=None)

    return {
        "name":         run_dir.name,
        "path":         run_dir,
        "status":       status,
        "n_event_files": len(event_files),
        "n_centroids":  n_centroids,
        "mtime":        mtime,
    }

# C++ source and compiled executable live alongside this module
_POSTPROCESSING_DIR = Path(__file__).parent
_EXECUTABLE_PATH = _POSTPROCESSING_DIR / "dbscan_main.exe"
_CPP_SOURCE_PATH = _POSTPROCESSING_DIR / "dbscan_main.cpp"


def convert_events_to_tofbin(events_dat_path: str, tofbin_path: str) -> int:
    """
    Convert a PyServal _events.dat file to the binary .tofbin format
    expected by dbscan_main.exe.

    The .tofbin format is 5 x float64 per record: (shot, x, y, tof, tot),
    matching the write_bin() format from helperfuncs.py.

    Parameters
    ----------
    events_dat_path : str
        Path to the _events.dat file (EVENT_DTYPE binary).
    tofbin_path : str
        Output path for the .tofbin file.

    Returns
    -------
    int
        Number of records written.
    """
    events = np.fromfile(events_dat_path, dtype=EVENT_DTYPE)
    if len(events) == 0:
        return 0

    # Build a contiguous (N, 5) float64 array and write in one call —
    # avoids a per-row Python loop that was taking ~13 s for large files.
    out = np.empty((len(events), 5), dtype=np.float64)
    out[:, 0] = events["t_trigger"]
    out[:, 1] = events["x"]
    out[:, 2] = events["y"]
    out[:, 3] = events["tof"]
    out[:, 4] = events["tot"]
    out.tofile(tofbin_path)

    return len(events)


class CentroidProcessor:
    """
    Wraps dbscan_main.exe for post-processing PyServal event files.

    Parameters
    ----------
    executable_path : str or Path, optional
        Path to the compiled dbscan_main.exe. Defaults to
        pymepixcentroider/dbscan_main.exe relative to this package.
    epsilon : float
        DBSCAN spatial epsilon (pixel units). Default: 2.0.
    tof_threshold : float
        Maximum TOF (seconds) for a point to be included. Default: 2e-4.
    min_points : int
        Minimum cluster size for DBSCAN. Default: 1.
    """

    def __init__(
        self,
        executable_path: Optional[str] = None,
        epsilon: float = 2.0,
        tof_threshold: float = 2e-4,
        min_points: int = 1,
    ):
        self.executable_path = Path(executable_path) if executable_path else _EXECUTABLE_PATH
        self.epsilon = epsilon
        self.tof_threshold = tof_threshold
        self.min_points = min_points

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
            Phase names: "converting", "reading", "grouping", "dbscan",
            "clustering", "writing", "done".

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

        # Compile if needed
        if not self.executable_path.exists():
            if not self.compile():
                raise RuntimeError(
                    f"Executable not found and compilation failed: {self.executable_path}"
                )

        # Convert _events.dat to temp .tofbin
        with tempfile.NamedTemporaryFile(suffix=".tofbin", delete=False) as tmp:
            tofbin_path = tmp.name

        try:
            if progress_callback:
                progress_callback(0, "converting")
            n = convert_events_to_tofbin(str(events_dat_path), tofbin_path)
            if n == 0:
                return np.array([], dtype=CENTROID_DTYPE)

            # Build command
            cmd = [
                str(self.executable_path),
                tofbin_path,
                str(output_path),
                "--epsilon", str(self.epsilon),
                "--min-points", str(self.min_points),
                "--tof-threshold", str(self.tof_threshold),
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

        finally:
            Path(tofbin_path).unlink(missing_ok=True)

    def process_run_dir_merged(
        self,
        run_dir,
        correction_path: Optional[str] = None,
        labels: bool = False,
        diagnostics: bool = False,
        progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
        force: bool = False,
    ) -> Path:
        """
        Process all *_events.dat files in a run folder in parallel, then merge
        results into a single ``<folder_name>_centroids.datbin`` file sorted by
        ``shot_index``.

        Shot indices are derived from the run's ``*_triggers.trg`` files: all
        trigger files are merged and sorted by ``toa``, giving a canonical
        0-based index for each laser shot. Each centroid's ``t_trigger`` is
        matched against this array via binary search.

        Parameters
        ----------
        run_dir : Path
            Folder containing ``*_events.dat`` and ``*_triggers.trg`` files.
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
            If False and the merged output already exists, return immediately.

        Returns
        -------
        Path
            Path to the merged ``_centroids.datbin`` file.
        """
        run_dir = Path(run_dir)
        event_files = sorted(run_dir.glob("*_events.dat"))
        if not event_files:
            raise RuntimeError(f"No *_events.dat files found in {run_dir}")

        output_path = run_dir / f"{run_dir.name}_centroids.datbin"
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
                labels_path = str(run_dir / f"{stem}.toflabels")

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

        # --- build shot_index from trigger files -----------------------------
        trigger_toa = None
        trigger_files = sorted(run_dir.glob("*_triggers.trg"))
        if trigger_files:
            from SERVAL.core.data_types import TRIGGER_DTYPE, TriggerData, merge_triggers

            trigger_list = []
            for tf in trigger_files:
                raw = np.fromfile(str(tf), dtype=TRIGGER_DTYPE)
                if len(raw):
                    trigger_list.append(
                        TriggerData(toa=raw["toa"], tdc_id=raw["tdc_id"], edge=raw["edge"])
                    )
            if trigger_list:
                merged_trig = merge_triggers(*trigger_list)
                trigger_toa = merged_trig.toa  # already sorted by merge_triggers

        # --- concatenate, assign shot_index, sort, write ---------------------
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

        if trigger_toa is not None and len(trigger_toa):
            # searchsorted is exact: t_trigger values come directly from the
            # same trigger packets stored in the trigger files.
            merged["shot_index"] = np.searchsorted(trigger_toa, all_c["t_trigger"])
        else:
            # Fallback when no trigger files are present
            merged["shot_index"] = np.argsort(all_c["t_trigger"], kind="stable")

        order = np.argsort(merged["shot_index"], kind="stable")
        merged[order].tofile(str(output_path))

        # --- cleanup ---------------------------------------------------------
        for tmp_path in tmp_paths:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        return output_path

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
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--tof-threshold", type=float, default=2e-4)
    parser.add_argument("--min-points", type=int, default=1)
    parser.add_argument("--correction", help="Path to TOF correction .txt file")
    parser.add_argument("--labels", action="store_true", help="Generate .toflabels output")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    proc = CentroidProcessor(
        epsilon=args.epsilon,
        tof_threshold=args.tof_threshold,
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
