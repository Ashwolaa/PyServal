"""
Postprocessing GUI — run-group centric raw->events->centroids pipeline

Discovers run groups anywhere beneath a chosen parent directory and shows
them as a checkable file tree: each directory is a branch, each run group
within it is a leaf. This lets a folder holding several scan steps' sibling
files (e.g. ``Scan000/00001.tpx3``, ``00002.tpx3``, ...) be selected/processed
at the granularity of a single step, the whole scan folder, or an entire
dataset subtree — without ever merging different steps' data together.
Checking a branch checks every run group beneath it (and vice versa) via the
usual tri-state parent/child checkbox cascade.

Each run group goes through up to two stages, run only if needed (or
forced):

    raw .tpx3 --[extract+correlate]--> _events.dat --[DBSCAN]--> _centroids.datbin

The raw stage is skipped for groups that have no raw ``.tpx3`` file (e.g.
legacy data where only ``_events.dat`` was kept) and already have events.
Parallel-raw-saver splits of one take (``_raw{i}.tpx3``, see
``SERVAL.postprocessing.raw_extraction``) are extracted and concatenated
into a single merged ``_events.dat``; the centroiding stage then runs its
own per-file parallel C++ workers (for parallel *event*-saver splits,
``_saver{i}``) and merges into one ``{step_key}_centroids.datbin`` sorted by
shot_index.

Launch with:
    python -m SERVAL.postprocessing.postprocessing_gui
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from SERVAL.core.data_types import TDCChannel, TriggerEdge
from SERVAL.postprocessing.centroiding import (
    MERGED_CENTROID_DTYPE,
    CentroidProcessor,
    RunStatus,
    detect_tof_window,
    discover_run_groups,
    get_run_status,
)
from SERVAL.postprocessing.raw_extraction import (
    RawEventProcessor,
    discover_raw_groups,
    get_raw_status,
    load_run_meta,
)

# ---------------------------------------------------------------------------
# Tree column indices
#
# The checkbox lives on the Name column itself (item.setCheckState(COL_NAME,
# ...)) rather than a separate narrow column — Qt draws it inline with the
# icon/text at full row height, scaling naturally with tree indentation,
# instead of being squeezed into a fixed-width sliver next to the expand
# arrows.
# ---------------------------------------------------------------------------
COL_NAME        = 0
COL_RAW_STATUS  = 1
COL_CENT_STATUS = 2
COL_RAW_FILES   = 3
COL_EVENT_FILES = 4
COL_CENTROIDS   = 5
COL_DATE        = 6

#: Qt.UserRole data on a leaf item: a dict describing the run group (see
#: ``PostprocessingGUI._make_leaf_item``). Branch items carry no such data —
#: that's how leaves are told apart.
_GROUP_DATA_ROLE = Qt.UserRole

#: Sentinel status for "this stage doesn't apply" (e.g. no raw .tpx3 found,
#: or events not extracted yet so centroiding can't be assessed), shown as
#: a dash rather than a status word.
_NA = None

_STATUS_COLOR = {
    RunStatus.READY: QColor(70,  130, 180),  # steel blue
    RunStatus.DONE:  QColor(50,  160,  50),  # green
    RunStatus.STALE: QColor(200, 120,   0),  # orange
    RunStatus.EMPTY: QColor(150, 150, 150),  # gray
    _NA:             QColor(150, 150, 150),  # gray
}
_STATUS_LABEL = {
    RunStatus.READY: "Ready",
    RunStatus.DONE:  "Done",
    RunStatus.STALE: "Stale",
    RunStatus.EMPTY: "Empty",
    _NA:             "—",
}

#: Which tree column shows progress/status for each processing stage.
_STAGE_COLUMN = {"raw": COL_RAW_STATUS, "cent": COL_CENT_STATUS}

_AUTO_LABEL = "Auto (from *_meta.json)"


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

class _RunWorker(QThread):
    """Processes selected run groups sequentially (parallel within each group's stage)."""

    group_started  = Signal(str, str, str)               # group_key, display_name, stage
    group_progress = Signal(str, str, int, int, int, str) # group_key, stage, file_idx, n_files, pct, phase
    group_raw_done = Signal(str)                          # group_key — raw stage finished, events ready
    group_done     = Signal(str, int)                     # group_key, n_centroids — whole group finished
    group_error    = Signal(str, str, str)                # group_key, stage, error_msg
    log_message    = Signal(str)
    finished       = Signal()

    def __init__(
        self,
        run_groups,
        cent_processor,
        correction_path,
        labels,
        diagnostics,
        force_raw,
        force_cent,
        tdc_override,
        edge_override,
        window_override,
        save_pixels,
        save_triggers,
    ):
        """
        Parameters
        ----------
        run_groups : list[dict]
            One dict per checked run group (see
            ``PostprocessingGUI._make_leaf_item``).
        tdc_override, edge_override : str, optional
            GUI label overriding each run's own ``*_meta.json`` for the raw
            extraction stage. None means "use that run's metadata, or the
            pipeline default if it has none".
        window_override : tuple[float, float], optional
            Same idea, for the correlation ToF window (ns).
        """
        super().__init__()
        self._run_groups       = run_groups
        self._cent_processor   = cent_processor
        self._correction_path  = correction_path
        self._labels           = labels
        self._diagnostics      = diagnostics
        self._force_raw        = force_raw
        self._force_cent       = force_cent
        self._tdc_override     = tdc_override
        self._edge_override    = edge_override
        self._window_override  = window_override
        self._save_pixels      = save_pixels
        self._save_triggers    = save_triggers
        self._stop_requested   = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            for group in self._run_groups:
                if self._stop_requested:
                    self.log_message.emit("Stopped by user.")
                    break

                display_name = group["key"]
                group_key = str(group["centroid_path"])
                events_files = group["event_files"]
                need_raw = bool(group["raw_files"]) and (
                    self._force_raw
                    or group["raw_status"] in (RunStatus.READY, RunStatus.STALE)
                    or not events_files
                )

                try:
                    if need_raw:
                        self.group_started.emit(group_key, display_name, "raw")
                        self.log_message.emit(f"Extracting {display_name} (raw → events) …")

                        meta = load_run_meta(group["raw_files"][0].parent) or {}
                        tdc_label = self._tdc_override or meta.get("tdc_label") or TDCChannel.TDC1.label
                        edge_label = self._edge_override or meta.get("edge_label") or TriggerEdge.RISING.label
                        window = self._window_override or tuple(meta.get("event_window_ns", (0.0, 100_000.0)))

                        raw_processor = RawEventProcessor(
                            tdc_id=TDCChannel.from_label(tdc_label),
                            edge=TriggerEdge.from_label(edge_label),
                            window_ns=window,
                            save_pixels=self._save_pixels,
                            save_triggers=self._save_triggers,
                        )

                        def raw_progress_cb(file_idx, n_files, pct, phase):
                            self.group_progress.emit(group_key, "raw", file_idx, n_files, pct, phase)

                        events_out = raw_processor.process_raw_group(
                            group["raw_files"],
                            group["events_target"],
                            progress_callback=raw_progress_cb,
                            force=self._force_raw,
                        )
                        events_files = [events_out]
                        self.group_raw_done.emit(group_key)
                        self.log_message.emit(f"  ✓ {display_name}: extracted → {events_out.name}")

                    if not events_files:
                        raise RuntimeError(
                            "No events available — no raw .tpx3 found and no existing _events.dat"
                        )

                    self.group_started.emit(group_key, display_name, "cent")
                    self.log_message.emit(f"Centroiding {display_name} …")

                    def cent_progress_cb(file_idx, n_files, pct, phase):
                        self.group_progress.emit(group_key, "cent", file_idx, n_files, pct, phase)

                    out = self._cent_processor.process_event_group(
                        events_files,
                        group["centroid_path"],
                        correction_path=self._correction_path or None,
                        labels=self._labels,
                        diagnostics=self._diagnostics,
                        progress_callback=cent_progress_cb,
                        force=self._force_cent,
                    )

                    n_centroids = 0
                    if out and out.exists() and out.stat().st_size > 0:
                        n_centroids = out.stat().st_size // MERGED_CENTROID_DTYPE.itemsize

                    self.group_done.emit(group_key, n_centroids)
                    self.log_message.emit(
                        f"  ✓ {display_name}: {n_centroids:,} centroids → {out.name}"
                    )

                except Exception as e:
                    stage = "raw" if need_raw and not events_files else "cent"
                    self.group_error.emit(group_key, stage, str(e))
                    self.log_message.emit(f"  ✗ {display_name}: {e}")
                    if self._diagnostics:
                        self.log_message.emit(traceback.format_exc())

        finally:
            self.finished.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class PostprocessingGUI(QMainWindow):
    """Run-group centric raw->events->centroids GUI, with a checkable file tree."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyServal Postprocessing")
        self.resize(1300, 760)

        self._parent_dir     = None
        self._item_map        = {}   # group_key (str(centroid_path)) -> QTreeWidgetItem
        self._worker          = None
        self._n_selected       = 0
        self._n_done           = 0
        self._updating_checks = False  # re-entrancy guard for the checkbox cascade

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Directory bar ──────────────────────────────────────────────────
        dir_bar = QHBoxLayout()
        dir_bar.addWidget(QLabel("Data directory:"))
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Select parent directory containing run folders…")
        self._dir_edit.setReadOnly(True)
        dir_bar.addWidget(self._dir_edit, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_dir)
        dir_bar.addWidget(browse_btn)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh_tree)
        dir_bar.addWidget(self._refresh_btn)
        root.addLayout(dir_bar)

        # ── Main splitter: table (left) | controls + log (right) ───────────
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # ── Left: run-group tree ────────────────────────────────────────────
        tree_container = QWidget()
        tree_layout = QVBoxLayout(tree_container)
        tree_layout.setContentsMargins(0, 0, 0, 0)

        sel_bar = QHBoxLayout()
        sel_all = QPushButton("Select All")
        sel_all.clicked.connect(self._select_all)
        sel_none = QPushButton("Select None")
        sel_none.clicked.connect(self._select_none)
        sel_bar.addWidget(sel_all)
        sel_bar.addWidget(sel_none)
        sel_bar.addStretch()
        tree_layout.addLayout(sel_bar)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(7)
        self._tree.setHeaderLabels(
            ["Name", "Raw→Events", "Events→Centroids", "Raw files", "Event files", "Centroids", "Modified"]
        )
        hh = self._tree.header()
        hh.setSectionResizeMode(COL_NAME,        QHeaderView.Stretch)
        hh.setSectionResizeMode(COL_RAW_STATUS,   QHeaderView.Fixed)
        hh.setSectionResizeMode(COL_CENT_STATUS,  QHeaderView.Fixed)
        hh.setSectionResizeMode(COL_RAW_FILES,    QHeaderView.Fixed)
        hh.setSectionResizeMode(COL_EVENT_FILES,  QHeaderView.Fixed)
        hh.setSectionResizeMode(COL_CENTROIDS,    QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_DATE,         QHeaderView.ResizeToContents)
        self._tree.setColumnWidth(COL_RAW_STATUS,  110)
        self._tree.setColumnWidth(COL_CENT_STATUS, 130)
        self._tree.setColumnWidth(COL_RAW_FILES,   70)
        self._tree.setColumnWidth(COL_EVENT_FILES, 70)
        self._tree.setEditTriggers(QTreeWidget.NoEditTriggers)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        tree_layout.addWidget(self._tree)

        splitter.addWidget(tree_container)

        # ── Right: parameters + actions + progress + log ───────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)

        # Raw extraction parameters
        raw_group = QGroupBox("Raw extraction (raw .tpx3 → events)")
        raw_layout = QVBoxLayout(raw_group)

        raw_layout.addWidget(QLabel("TDC channel (reference trigger):"))
        self._tdc_combo = QComboBox()
        self._tdc_combo.addItems([_AUTO_LABEL] + TDCChannel.labels())
        raw_layout.addWidget(self._tdc_combo)

        raw_layout.addWidget(QLabel("Trigger edge:"))
        self._edge_combo = QComboBox()
        self._edge_combo.addItems([_AUTO_LABEL] + TriggerEdge.labels())
        raw_layout.addWidget(self._edge_combo)

        self._window_override_check = QCheckBox("Override correlation ToF window (ns):")
        raw_layout.addWidget(self._window_override_check)
        win_row = QHBoxLayout()
        self._raw_window_min_spin = QDoubleSpinBox()
        self._raw_window_min_spin.setRange(0.0, 1_000_000_000.0)
        self._raw_window_min_spin.setSingleStep(10.0)
        self._raw_window_min_spin.setDecimals(1)
        self._raw_window_min_spin.setValue(0.0)
        self._raw_window_max_spin = QDoubleSpinBox()
        self._raw_window_max_spin.setRange(0.0, 1_000_000_000.0)
        self._raw_window_max_spin.setSingleStep(10.0)
        self._raw_window_max_spin.setDecimals(1)
        self._raw_window_max_spin.setValue(100_000.0)
        win_row.addWidget(self._raw_window_min_spin)
        win_row.addWidget(QLabel("–"))
        win_row.addWidget(self._raw_window_max_spin)
        raw_layout.addLayout(win_row)

        self._save_pixels_check = QCheckBox("Also save *_pixels.dat")
        raw_layout.addWidget(self._save_pixels_check)
        self._save_triggers_check = QCheckBox("Also save *_triggers.trg")
        raw_layout.addWidget(self._save_triggers_check)

        right_layout.addWidget(raw_group)

        # DBSCAN centroiding parameters
        params_group = QGroupBox("Centroiding (events → centroids)")
        params_layout = QVBoxLayout(params_group)

        # Backend — same algorithm either way. "cpp" runs the compiled binary
        # via subprocess (process-spawn + file I/O cost on every call).
        # "numba" runs in-process: pays a JIT warm-up on the first call in
        # this GUI session, then is substantially faster for every call after
        # — the GUI is a long-lived process, so "numba" wins after the first run.
        params_layout.addWidget(QLabel("Backend:"))
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["cpp", "numba"])
        params_layout.addWidget(self._backend_combo)

        params_layout.addWidget(QLabel("Epsilon (pixels) — spatial cluster radius:"))
        self._epsilon_spin = QDoubleSpinBox()
        self._epsilon_spin.setRange(0.1, 100.0)
        self._epsilon_spin.setSingleStep(0.5)
        self._epsilon_spin.setDecimals(2)
        self._epsilon_spin.setValue(2.0)
        params_layout.addWidget(self._epsilon_spin)

        # Independent from epsilon: how far apart in ToF two pixel hits can be
        # and still be the same ion/cluster. Two ions of different mass arriving
        # at noticeably different ToF on the same shot should land in separate
        # clusters — that separation is controlled here, not by epsilon.
        params_layout.addWidget(QLabel("ToF cluster radius (ns) — min. ToF separation between clusters:"))
        self._eps_time_spin = QDoubleSpinBox()
        self._eps_time_spin.setRange(0.1, 1_000_000.0)  # up to 1 ms
        self._eps_time_spin.setSingleStep(10.0)
        self._eps_time_spin.setDecimals(1)
        self._eps_time_spin.setValue(100.0)  # 100 ns default, matches live pipeline
        params_layout.addWidget(self._eps_time_spin)

        # ToF acceptance window — the real ToF peak sits at an instrument-dependent
        # offset (not at zero), so this is a [min, max] window around that peak,
        # not a single "distance from zero" threshold. Use "Detect peak" to seed
        # min/max from the actual data instead of guessing. Independent from the
        # raw-extraction correlation window above — this is a post-hoc filter.
        params_layout.addWidget(QLabel("ToF window (ns):"))
        tof_row = QHBoxLayout()
        self._tof_min_spin = QDoubleSpinBox()
        self._tof_min_spin.setRange(0.0, 1_000_000_000.0)  # up to 1 s
        self._tof_min_spin.setSingleStep(10.0)
        self._tof_min_spin.setDecimals(1)
        self._tof_min_spin.setValue(0.0)
        self._tof_max_spin = QDoubleSpinBox()
        self._tof_max_spin.setRange(0.0, 1_000_000_000.0)  # up to 1 s
        self._tof_max_spin.setSingleStep(10.0)
        self._tof_max_spin.setDecimals(1)
        self._tof_max_spin.setValue(1_000_000_000.0)  # effectively unbounded by default
        tof_row.addWidget(self._tof_min_spin)
        tof_row.addWidget(QLabel("–"))
        tof_row.addWidget(self._tof_max_spin)
        params_layout.addLayout(tof_row)

        params_layout.addWidget(QLabel("Min points:"))
        self._minpts_spin = QSpinBox()
        self._minpts_spin.setRange(1, 100)
        self._minpts_spin.setValue(1)
        params_layout.addWidget(self._minpts_spin)

        params_layout.addWidget(QLabel("Correction file (.txt, optional):"))
        corr_row = QHBoxLayout()
        self._corr_edit = QLineEdit()
        self._corr_edit.setPlaceholderText("Optional…")
        corr_browse = QPushButton("Browse")
        corr_browse.clicked.connect(self._browse_correction)
        corr_row.addWidget(self._corr_edit, 1)
        corr_row.addWidget(corr_browse)
        params_layout.addLayout(corr_row)

        self._labels_check = QCheckBox("Generate per-file labels (.toflabels)")
        params_layout.addWidget(self._labels_check)
        self._diag_check = QCheckBox("Show C++ timing diagnostics")
        params_layout.addWidget(self._diag_check)

        right_layout.addWidget(params_group)

        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        self._compile_btn = QPushButton("Compile C++")
        self._compile_btn.clicked.connect(self._on_compile)
        actions_layout.addWidget(self._compile_btn)

        self._run_btn = QPushButton("Process selected")
        self._run_btn.setStyleSheet(
            "QPushButton:enabled { background-color: #4CAF50; color: white; font-weight: bold; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        actions_layout.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setStyleSheet(
            "QPushButton:enabled { background-color: #f44336; color: white; font-weight: bold; }"
        )
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        actions_layout.addWidget(self._stop_btn)

        self._force_raw_check = QCheckBox("Force raw → events (overwrite existing)")
        actions_layout.addWidget(self._force_raw_check)
        self._force_cent_check = QCheckBox("Force events → centroids (overwrite existing)")
        actions_layout.addWidget(self._force_cent_check)

        self._benchmark_btn = QPushButton("Benchmark centroiding backends (first selected run)")
        self._benchmark_btn.clicked.connect(self._on_benchmark)
        actions_layout.addWidget(self._benchmark_btn)

        right_layout.addWidget(actions_group)

        # Overall progress
        progress_group = QGroupBox("Overall progress")
        progress_layout = QVBoxLayout(progress_group)
        self._progress_label = QLabel("Idle")
        self._overall_bar = QProgressBar()
        self._overall_bar.setRange(0, 100)
        progress_layout.addWidget(self._progress_label)
        progress_layout.addWidget(self._overall_bar)
        right_layout.addWidget(progress_group)

        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setFontFamily("monospace")
        log_layout.addWidget(self._log_edit)
        right_layout.addWidget(log_group, 1)

        splitter.addWidget(right)
        splitter.setSizes([780, 480])

    # ── Directory ──────────────────────────────────────────────────────────

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select data directory", "",
            QFileDialog.Option.DontUseNativeDialog,
        )
        if d:
            self._parent_dir = Path(d)
            self._dir_edit.setText(d)
            self._refresh_tree()

    def _browse_correction(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select correction file", "", "Text files (*.txt)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self._corr_edit.setText(path)

    # ── Tree ───────────────────────────────────────────────────────────────

    def _refresh_tree(self):
        if not self._parent_dir or not self._parent_dir.is_dir():
            return

        self._updating_checks = True
        try:
            self._tree.clear()
            self._item_map.clear()
            self._build_tree(None, self._parent_dir)
            self._tree.expandAll()
        finally:
            self._updating_checks = False

    def _has_run_files_recursive(self, directory: Path) -> bool:
        if any(directory.glob("*_events.dat")) or any(directory.glob("*.tpx3")):
            return True
        return any(
            self._has_run_files_recursive(sub)
            for sub in directory.iterdir() if sub.is_dir()
        )

    def _build_tree(self, parent_item, directory: Path) -> bool:
        """Recursively add `directory`'s own run groups as leaves, then
        its subdirectories (that contain raw or event files anywhere
        beneath them) as branches. Returns whether anything was added."""
        added_any = False

        raw_groups = discover_raw_groups(directory)
        event_groups = discover_run_groups(directory)
        for key in sorted(set(raw_groups) | set(event_groups)):
            leaf = self._make_leaf_item(
                directory, key, raw_groups.get(key, []), event_groups.get(key, [])
            )
            self._add_item(parent_item, leaf)
            added_any = True

        for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
            if not self._has_run_files_recursive(sub):
                continue
            branch = QTreeWidgetItem()
            branch.setText(COL_NAME, sub.name)
            branch.setFlags(branch.flags() | Qt.ItemIsUserCheckable)
            branch.setCheckState(COL_NAME, Qt.Unchecked)
            self._add_item(parent_item, branch)
            self._build_tree(branch, sub)
            added_any = True

        return added_any

    def _add_item(self, parent_item, item):
        if parent_item is None:
            self._tree.addTopLevelItem(item)
        else:
            parent_item.addChild(item)

    def _make_leaf_item(self, directory: Path, key: str, raw_files: list, event_files: list) -> QTreeWidgetItem:
        events_target = directory / f"{key}_events.dat"
        centroid_path = directory / f"{key}_centroids.datbin"

        if raw_files:
            raw_status = get_raw_status(raw_files, event_files[0] if event_files else events_target)
        else:
            raw_status = _NA  # nothing to extract — legacy events-only data

        n_centroids = None
        if event_files:
            cent_status = get_run_status(event_files, centroid_path)
            if centroid_path.exists() and centroid_path.stat().st_size > 0:
                n_centroids = centroid_path.stat().st_size // MERGED_CENTROID_DTYPE.itemsize
        elif raw_files:
            cent_status = _NA  # pending raw extraction first
        else:
            cent_status = RunStatus.EMPTY

        item = QTreeWidgetItem()
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        auto_check = (
            raw_status in (RunStatus.READY, RunStatus.STALE)
            or cent_status in (RunStatus.READY, RunStatus.STALE)
        )
        item.setCheckState(COL_NAME, Qt.Checked if auto_check else Qt.Unchecked)

        item.setText(COL_NAME, key)
        self._set_status(item, COL_RAW_STATUS, raw_status)
        self._set_status(item, COL_CENT_STATUS, cent_status)

        item.setText(COL_RAW_FILES, str(len(raw_files)))
        item.setTextAlignment(COL_RAW_FILES, Qt.AlignCenter)
        item.setText(COL_EVENT_FILES, str(len(event_files)))
        item.setTextAlignment(COL_EVENT_FILES, Qt.AlignCenter)

        cent_text = f"{n_centroids:,}" if n_centroids is not None else "—"
        item.setText(COL_CENTROIDS, cent_text)
        item.setTextAlignment(COL_CENTROIDS, Qt.AlignRight | Qt.AlignVCenter)

        mtimes = [f.stat().st_mtime for f in (raw_files + event_files)]
        if mtimes:
            item.setText(COL_DATE, datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d %H:%M"))

        group = {
            "key":            key,
            "raw_files":      raw_files,
            "event_files":    event_files,
            "events_target":  events_target,
            "centroid_path":  centroid_path,
            "raw_status":     raw_status,
            "cent_status":    cent_status,
        }
        item.setData(COL_NAME, _GROUP_DATA_ROLE, group)
        self._item_map[str(centroid_path)] = item
        return item

    def _set_status(self, item: QTreeWidgetItem, column: int, status):
        item.setText(column, _STATUS_LABEL[status])
        item.setTextAlignment(column, Qt.AlignCenter)
        item.setForeground(column, _STATUS_COLOR[status])

    def _set_leaf_progress(self, item: QTreeWidgetItem, column: int, label: str, pct: int):
        bar = self._tree.itemWidget(item, column)
        if not isinstance(bar, QProgressBar):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            self._tree.setItemWidget(item, column, bar)
        bar.setValue(pct)
        bar.setFormat(f"{label}  {pct}%")

    # ── Checkbox cascade (parent <-> children) ──────────────────────────────

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        if column != COL_NAME or self._updating_checks:
            return
        self._updating_checks = True
        try:
            state = item.checkState(COL_NAME)
            if state != Qt.PartiallyChecked:
                self._cascade_to_children(item, state)
            self._update_ancestors(item.parent())
        finally:
            self._updating_checks = False

    def _cascade_to_children(self, item: QTreeWidgetItem, state):
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(COL_NAME, state)
            self._cascade_to_children(child, state)

    def _update_ancestors(self, item):
        while item is not None:
            states = {item.child(i).checkState(COL_NAME) for i in range(item.childCount())}
            item.setCheckState(COL_NAME, states.pop() if len(states) == 1 else Qt.PartiallyChecked)
            item = item.parent()

    def _select_all(self):
        self._updating_checks = True
        try:
            for i in range(self._tree.topLevelItemCount()):
                top = self._tree.topLevelItem(i)
                top.setCheckState(COL_NAME, Qt.Checked)
                self._cascade_to_children(top, Qt.Checked)
        finally:
            self._updating_checks = False

    def _select_none(self):
        self._updating_checks = True
        try:
            for i in range(self._tree.topLevelItemCount()):
                top = self._tree.topLevelItem(i)
                top.setCheckState(COL_NAME, Qt.Unchecked)
                self._cascade_to_children(top, Qt.Unchecked)
        finally:
            self._updating_checks = False

    # ── Worker signal handlers (main-thread slots) ─────────────────────────

    def _on_group_started(self, group_key: str, display_name: str, stage: str):
        item = self._item_map.get(group_key)
        if item is not None:
            self._set_leaf_progress(item, _STAGE_COLUMN[stage], "starting", 0)
        verb = "Extracting" if stage == "raw" else "Centroiding"
        self._progress_label.setText(f"{verb}: {display_name}")

    def _on_group_progress(self, group_key: str, stage: str, file_idx: int, n_files: int, pct: int, phase: str):
        item = self._item_map.get(group_key)
        if item is not None:
            label = f"[{file_idx + 1}/{n_files}] {phase}"
            self._set_leaf_progress(item, _STAGE_COLUMN[stage], label, pct)
        # overall = completed groups + fraction of current group
        overall = int((self._n_done + pct / 100) / max(self._n_selected, 1) * 100)
        self._overall_bar.setValue(overall)

    def _on_group_raw_done(self, group_key: str):
        item = self._item_map.get(group_key)
        if item is not None:
            self._tree.removeItemWidget(item, COL_RAW_STATUS)
            self._set_status(item, COL_RAW_STATUS, RunStatus.DONE)

    def _on_group_done(self, group_key: str, n_centroids: int):
        self._n_done += 1
        item = self._item_map.get(group_key)
        if item is not None:
            self._tree.removeItemWidget(item, COL_CENT_STATUS)
            self._set_status(item, COL_CENT_STATUS, RunStatus.DONE)
            item.setText(COL_CENTROIDS, f"{n_centroids:,}")
        self._overall_bar.setValue(int(self._n_done / max(self._n_selected, 1) * 100))

    def _on_group_error(self, group_key: str, stage: str, error: str):
        self._n_done += 1
        item = self._item_map.get(group_key)
        if item is not None:
            column = _STAGE_COLUMN[stage]
            self._tree.removeItemWidget(item, column)
            item.setText(column, "Error")
            item.setTextAlignment(column, Qt.AlignCenter)
            item.setForeground(column, QColor(200, 0, 0))
            item.setToolTip(column, error)
        self._overall_bar.setValue(int(self._n_done / max(self._n_selected, 1) * 100))

    def _on_finished(self):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._overall_bar.setValue(100)
        self._progress_label.setText("Done")
        self._log("All done.")

    # ── Button slots ───────────────────────────────────────────────────────

    def _log(self, msg: str):
        self._log_edit.append(msg)

    def _on_compile(self):
        self._log("Compiling dbscan_main.cpp…")
        proc = CentroidProcessor(
            epsilon=self._epsilon_spin.value(),
            eps_time=self._eps_time_spin.value() * 1e-9,  # ns → s
            tof_min=self._tof_min_spin.value() * 1e-9,  # ns → s
            tof_max=self._tof_max_spin.value() * 1e-9,
            min_points=self._minpts_spin.value(),
            backend=self._backend_combo.currentText(),
        )
        if proc.compile(force=True):
            self._log("Compilation successful.")
        else:
            self._log("Compilation failed. Check that g++ is installed.")

    def _checked_run_groups(self) -> list:
        """Walk the tree and return the group dict for every checked leaf."""
        groups = []

        def visit(item):
            data = item.data(COL_NAME, _GROUP_DATA_ROLE)
            if data is not None:  # leaf
                if item.checkState(COL_NAME) == Qt.Checked:
                    groups.append(data)
            for i in range(item.childCount()):
                visit(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i))
        return groups

    def _on_run(self):
        if not self._parent_dir:
            self._log("No directory selected.")
            return

        selected = self._checked_run_groups()
        if not selected:
            self._log("No run groups selected.")
            return

        cent_processor = CentroidProcessor(
            epsilon=self._epsilon_spin.value(),
            eps_time=self._eps_time_spin.value() * 1e-9,  # ns → s
            tof_min=self._tof_min_spin.value() * 1e-9,  # ns → s
            tof_max=self._tof_max_spin.value() * 1e-9,
            min_points=self._minpts_spin.value(),
            backend=self._backend_combo.currentText(),
        )

        tdc_text = self._tdc_combo.currentText()
        tdc_override = None if tdc_text == _AUTO_LABEL else tdc_text
        edge_text = self._edge_combo.currentText()
        edge_override = None if edge_text == _AUTO_LABEL else edge_text
        window_override = None
        if self._window_override_check.isChecked():
            window_override = (self._raw_window_min_spin.value(), self._raw_window_max_spin.value())

        self._worker = _RunWorker(
            run_groups=selected,
            cent_processor=cent_processor,
            correction_path=self._corr_edit.text().strip() or None,
            labels=self._labels_check.isChecked(),
            diagnostics=self._diag_check.isChecked(),
            force_raw=self._force_raw_check.isChecked(),
            force_cent=self._force_cent_check.isChecked(),
            tdc_override=tdc_override,
            edge_override=edge_override,
            window_override=window_override,
            save_pixels=self._save_pixels_check.isChecked(),
            save_triggers=self._save_triggers_check.isChecked(),
        )
        self._worker.group_started.connect(self._on_group_started)
        self._worker.group_progress.connect(self._on_group_progress)
        self._worker.group_raw_done.connect(self._on_group_raw_done)
        self._worker.group_done.connect(self._on_group_done)
        self._worker.group_error.connect(self._on_group_error)
        self._worker.log_message.connect(self._log)
        self._worker.finished.connect(self._on_finished)

        self._n_selected = len(selected)
        self._n_done = 0
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._overall_bar.setValue(0)
        self._progress_label.setText("Starting…")
        self._log(f"Starting {len(selected)} run group(s)…")
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)
            self._log("Stop requested…")

    def _on_benchmark(self):
        groups = self._checked_run_groups()
        if not groups:
            self._log("No run groups selected — check at least one to benchmark.")
            return

        group = groups[0]
        event_files = group["event_files"]
        if not event_files:
            self._log(f"No *_events.dat available yet for {group['key']} — run raw extraction first.")
            return
        event_file = event_files[0]

        kwargs = dict(
            epsilon=self._epsilon_spin.value(),
            eps_time=self._eps_time_spin.value() * 1e-9,
            tof_min=self._tof_min_spin.value() * 1e-9,
            tof_max=self._tof_max_spin.value() * 1e-9,
            min_points=self._minpts_spin.value(),
        )

        self._log(f"Benchmarking both backends on {event_file.name}…")
        self._benchmark_btn.setEnabled(False)

        self._bench_thread = _BenchmarkWorker(event_file, kwargs)
        self._bench_thread.result_ready.connect(self._on_benchmark_result)
        self._bench_thread.start()

    def _on_benchmark_result(self, summary: str):
        self._benchmark_btn.setEnabled(True)
        self._log(summary)


class _BenchmarkWorker(QThread):
    """Runs both backends once each on a single file and reports timing + agreement."""

    result_ready = Signal(str)

    def __init__(self, event_file: Path, kwargs: dict):
        super().__init__()
        self._event_file = event_file
        self._kwargs = kwargs

    def run(self):
        import time
        import numpy as np

        lines = []
        results = {}
        for backend in ("cpp", "numba"):
            proc = CentroidProcessor(backend=backend, **self._kwargs)
            try:
                t0 = time.perf_counter()
                centroids = proc.process_file(str(self._event_file), output_path="/tmp/_bench_centroids.datbin")
                dt = time.perf_counter() - t0
                results[backend] = centroids
                lines.append(f"  {backend:>6}: {len(centroids):,} centroids in {dt*1000:.1f} ms")
            except Exception as e:
                lines.append(f"  {backend:>6}: failed ({e})")

        if "cpp" in results and "numba" in results:
            a, b = results["cpp"], results["numba"]
            if len(a) == len(b):
                order_a = np.lexsort((a["y"], a["x"], a["t_trigger"]))
                order_b = np.lexsort((b["y"], b["x"], b["t_trigger"]))
                a, b = a[order_a], b[order_b]
                max_dx = float(np.max(np.abs(a["x"] - b["x"]))) if len(a) else 0.0
                max_dtof_ns = float(np.max(np.abs(a["tof"] - b["tof"]))) * 1e9 if len(a) else 0.0
                lines.append(f"  agreement: same count, max|dx|={max_dx:.6g} px, max|dtof|={max_dtof_ns:.6g} ns")
            else:
                lines.append(f"  agreement: COUNT MISMATCH ({len(a)} vs {len(b)})")

        self.result_ready.emit("Benchmark results:\n" + "\n".join(lines))


def main():
    app = QApplication(sys.argv)
    window = PostprocessingGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
