"""
DBSCAN Centroiding GUI — run-folder centric

Discovers acquisition run folders inside a parent directory, shows their
centroid status, and processes selected folders.  Within each folder, one
C++ worker runs per *_events.dat file (parallel); folders are processed
sequentially.  Results are merged into a single
``<folder_name>_centroids.datbin`` sorted by shot_index.

Launch with:
    python -m SERVAL.postprocessing.centroiding_gui
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
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from SERVAL.postprocessing.centroiding import (
    MERGED_CENTROID_DTYPE,
    CentroidProcessor,
    RunStatus,
    get_run_info,
)

# ---------------------------------------------------------------------------
# Table column indices
# ---------------------------------------------------------------------------
COL_CHECK     = 0
COL_NAME      = 1
COL_STATUS    = 2
COL_FILES     = 3
COL_CENTROIDS = 4
COL_DATE      = 5

_STATUS_COLOR = {
    RunStatus.READY: QColor(70,  130, 180),  # steel blue
    RunStatus.DONE:  QColor(50,  160,  50),  # green
    RunStatus.STALE: QColor(200, 120,   0),  # orange
    RunStatus.EMPTY: QColor(150, 150, 150),  # gray
}
_STATUS_LABEL = {
    RunStatus.READY: "Ready",
    RunStatus.DONE:  "Done",
    RunStatus.STALE: "Stale",
    RunStatus.EMPTY: "Empty",
}


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

class _RunWorker(QThread):
    """Processes selected run folders sequentially (parallel within each folder)."""

    folder_started  = Signal(str)                    # folder_name
    folder_progress = Signal(str, int, int, int, str)  # name, file_idx, n_files, pct, phase
    folder_done     = Signal(str, int)               # folder_name, n_centroids
    folder_error    = Signal(str, str)               # folder_name, error_msg
    log_message     = Signal(str)
    finished        = Signal()

    def __init__(self, folders, processor, correction_path, labels, diagnostics, force):
        super().__init__()
        self._folders         = folders
        self._processor       = processor
        self._correction_path = correction_path
        self._labels          = labels
        self._diagnostics     = diagnostics
        self._force           = force
        self._stop_requested  = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            for folder in self._folders:
                if self._stop_requested:
                    self.log_message.emit("Stopped by user.")
                    break

                self.folder_started.emit(folder.name)
                self.log_message.emit(f"Processing {folder.name} …")

                try:
                    def progress_cb(file_idx, n_files, pct, phase):
                        self.folder_progress.emit(
                            folder.name, file_idx, n_files, pct, phase
                        )

                    out = self._processor.process_run_dir_merged(
                        folder,
                        correction_path=self._correction_path or None,
                        labels=self._labels,
                        diagnostics=self._diagnostics,
                        progress_callback=progress_cb,
                        force=self._force,
                    )

                    n_centroids = 0
                    if out and out.exists() and out.stat().st_size > 0:
                        n_centroids = out.stat().st_size // MERGED_CENTROID_DTYPE.itemsize

                    self.folder_done.emit(folder.name, n_centroids)
                    self.log_message.emit(
                        f"  ✓ {folder.name}: {n_centroids:,} centroids → {out.name}"
                    )

                except Exception as e:
                    self.folder_error.emit(folder.name, str(e))
                    self.log_message.emit(f"  ✗ {folder.name}: {e}")
                    if self._diagnostics:
                        self.log_message.emit(traceback.format_exc())

        finally:
            self.finished.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class CentroidingGUI(QMainWindow):
    """Run-folder centric DBSCAN centroiding GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DBSCAN Centroiding")
        self.resize(1150, 720)

        self._parent_dir  = None
        self._row_map     = {}   # folder_name -> row index
        self._worker      = None
        self._n_selected  = 0
        self._n_done      = 0

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
        self._refresh_btn.clicked.connect(self._refresh_table)
        dir_bar.addWidget(self._refresh_btn)
        root.addLayout(dir_bar)

        # ── Main splitter: table (left) | controls + log (right) ───────────
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # ── Left: run table ────────────────────────────────────────────────
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        sel_bar = QHBoxLayout()
        sel_all = QPushButton("Select All")
        sel_all.clicked.connect(self._select_all)
        sel_none = QPushButton("Select None")
        sel_none.clicked.connect(self._select_none)
        sel_bar.addWidget(sel_all)
        sel_bar.addWidget(sel_none)
        sel_bar.addStretch()
        table_layout.addLayout(sel_bar)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["", "Run folder", "Status", "Event files", "Centroids", "Modified"]
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(COL_CHECK,     QHeaderView.Fixed)
        hh.setSectionResizeMode(COL_NAME,      QHeaderView.Stretch)
        hh.setSectionResizeMode(COL_STATUS,    QHeaderView.Fixed)
        hh.setSectionResizeMode(COL_FILES,     QHeaderView.Fixed)
        hh.setSectionResizeMode(COL_CENTROIDS, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_DATE,      QHeaderView.ResizeToContents)
        self._table.setColumnWidth(COL_CHECK,  28)
        self._table.setColumnWidth(COL_STATUS, 100)
        self._table.setColumnWidth(COL_FILES,  80)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        table_layout.addWidget(self._table)

        splitter.addWidget(table_container)

        # ── Right: parameters + actions + progress + log ───────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)

        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        params_layout.addWidget(QLabel("Epsilon (pixels):"))
        self._epsilon_spin = QDoubleSpinBox()
        self._epsilon_spin.setRange(0.1, 100.0)
        self._epsilon_spin.setSingleStep(0.5)
        self._epsilon_spin.setDecimals(2)
        self._epsilon_spin.setValue(2.0)
        params_layout.addWidget(self._epsilon_spin)

        params_layout.addWidget(QLabel("TOF threshold (ns):"))
        self._tof_spin = QDoubleSpinBox()
        self._tof_spin.setRange(0.0, 1_000_000.0)  # up to 1 ms
        self._tof_spin.setSingleStep(1.0)
        self._tof_spin.setDecimals(0)
        self._tof_spin.setValue(200.0)              # 200 ns default
        params_layout.addWidget(self._tof_spin)

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

        self._force_check = QCheckBox("Force re-process (overwrite existing)")
        actions_layout.addWidget(self._force_check)

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
        splitter.setSizes([680, 420])

    # ── Directory ──────────────────────────────────────────────────────────

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select data directory")
        if d:
            self._parent_dir = Path(d)
            self._dir_edit.setText(d)
            self._refresh_table()

    def _browse_correction(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select correction file", "", "Text files (*.txt)"
        )
        if path:
            self._corr_edit.setText(path)

    # ── Table ──────────────────────────────────────────────────────────────

    def _refresh_table(self):
        if not self._parent_dir or not self._parent_dir.is_dir():
            return

        run_dirs = sorted(
            d for d in self._parent_dir.iterdir()
            if d.is_dir() and any(d.glob("*_events.dat"))
        )

        self._table.setRowCount(0)
        self._row_map.clear()

        for row, run_dir in enumerate(run_dirs):
            self._table.insertRow(row)
            self._row_map[run_dir.name] = row
            self._fill_row(row, get_run_info(run_dir))

    def _fill_row(self, row: int, info: dict):
        status = info["status"]

        # Col 0: checkbox — pre-check READY and STALE folders
        chk = QTableWidgetItem()
        chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        auto_check = status in (RunStatus.READY, RunStatus.STALE)
        chk.setCheckState(Qt.Checked if auto_check else Qt.Unchecked)
        self._table.setItem(row, COL_CHECK, chk)

        # Col 1: name
        name_item = QTableWidgetItem(info["name"])
        name_item.setFlags(Qt.ItemIsEnabled)
        self._table.setItem(row, COL_NAME, name_item)

        # Col 2: status text
        self._set_status_text(row, status)

        # Col 3: event file count
        files_item = QTableWidgetItem(str(info["n_event_files"]))
        files_item.setTextAlignment(Qt.AlignCenter)
        files_item.setFlags(Qt.ItemIsEnabled)
        self._table.setItem(row, COL_FILES, files_item)

        # Col 4: centroid count
        cent_text = f"{info['n_centroids']:,}" if info["n_centroids"] is not None else "—"
        cent_item = QTableWidgetItem(cent_text)
        cent_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        cent_item.setFlags(Qt.ItemIsEnabled)
        self._table.setItem(row, COL_CENTROIDS, cent_item)

        # Col 5: modification date of newest events file
        date_text = ""
        if info["mtime"]:
            date_text = datetime.fromtimestamp(info["mtime"]).strftime("%Y-%m-%d %H:%M")
        date_item = QTableWidgetItem(date_text)
        date_item.setFlags(Qt.ItemIsEnabled)
        self._table.setItem(row, COL_DATE, date_item)

    def _set_status_text(self, row: int, status: RunStatus):
        self._table.removeCellWidget(row, COL_STATUS)
        item = QTableWidgetItem(_STATUS_LABEL[status])
        item.setTextAlignment(Qt.AlignCenter)
        item.setForeground(_STATUS_COLOR[status])
        item.setFlags(Qt.ItemIsEnabled)
        self._table.setItem(row, COL_STATUS, item)

    def _set_status_progress(self, row: int, label: str, pct: int):
        bar = self._table.cellWidget(row, COL_STATUS)
        if not isinstance(bar, QProgressBar):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            self._table.setCellWidget(row, COL_STATUS, bar)
        bar.setValue(pct)
        bar.setFormat(f"{label}  {pct}%")

    def _select_all(self):
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_CHECK)
            if item:
                item.setCheckState(Qt.Checked)

    def _select_none(self):
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_CHECK)
            if item:
                item.setCheckState(Qt.Unchecked)

    # ── Worker signal handlers (main-thread slots) ─────────────────────────

    def _on_folder_started(self, name: str):
        row = self._row_map.get(name)
        if row is not None:
            self._set_status_progress(row, "starting", 0)
        self._progress_label.setText(f"Processing: {name}")

    def _on_folder_progress(self, name: str, file_idx: int, n_files: int, pct: int, phase: str):
        row = self._row_map.get(name)
        if row is not None:
            label = f"[{file_idx + 1}/{n_files}] {phase}"
            self._set_status_progress(row, label, pct)
        # overall = completed folders + fraction of current folder
        overall = int((self._n_done + pct / 100) / max(self._n_selected, 1) * 100)
        self._overall_bar.setValue(overall)

    def _on_folder_done(self, name: str, n_centroids: int):
        self._n_done += 1
        row = self._row_map.get(name)
        if row is not None:
            self._set_status_text(row, RunStatus.DONE)
            cent_item = self._table.item(row, COL_CENTROIDS)
            if cent_item:
                cent_item.setText(f"{n_centroids:,}")
        self._overall_bar.setValue(int(self._n_done / max(self._n_selected, 1) * 100))

    def _on_folder_error(self, name: str, error: str):
        self._n_done += 1
        row = self._row_map.get(name)
        if row is not None:
            self._table.removeCellWidget(row, COL_STATUS)
            item = QTableWidgetItem("Error")
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(QColor(200, 0, 0))
            item.setFlags(Qt.ItemIsEnabled)
            item.setToolTip(error)
            self._table.setItem(row, COL_STATUS, item)

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
            tof_threshold=self._tof_spin.value() * 1e-9,  # ns → s
            min_points=self._minpts_spin.value(),
        )
        if proc.compile(force=True):
            self._log("Compilation successful.")
        else:
            self._log("Compilation failed. Check that g++ is installed.")

    def _on_run(self):
        if not self._parent_dir:
            self._log("No directory selected.")
            return

        selected = []
        for row in range(self._table.rowCount()):
            chk = self._table.item(row, COL_CHECK)
            if chk and chk.checkState() == Qt.Checked:
                name = self._table.item(row, COL_NAME).text()
                folder = self._parent_dir / name
                if folder.is_dir():
                    selected.append(folder)

        if not selected:
            self._log("No run folders selected.")
            return

        processor = CentroidProcessor(
            epsilon=self._epsilon_spin.value(),
            tof_threshold=self._tof_spin.value() * 1e-9,  # ns → s
            min_points=self._minpts_spin.value(),
        )

        self._worker = _RunWorker(
            folders=selected,
            processor=processor,
            correction_path=self._corr_edit.text().strip() or None,
            labels=self._labels_check.isChecked(),
            diagnostics=self._diag_check.isChecked(),
            force=self._force_check.isChecked(),
        )
        self._worker.folder_started.connect(self._on_folder_started)
        self._worker.folder_progress.connect(self._on_folder_progress)
        self._worker.folder_done.connect(self._on_folder_done)
        self._worker.folder_error.connect(self._on_folder_error)
        self._worker.log_message.connect(self._log)
        self._worker.finished.connect(self._on_finished)

        self._n_selected = len(selected)
        self._n_done = 0
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._overall_bar.setValue(0)
        self._progress_label.setText("Starting…")
        self._log(f"Starting {len(selected)} run folder(s)…")
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)
            self._log("Stop requested…")


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = CentroidingGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
