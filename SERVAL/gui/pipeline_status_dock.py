"""
PipelineStatusDock — dock widget showing throughput, queue fill levels, and worker health.

Usage::

    dock = PipelineStatusDock()
    dock_area.addDock(dock, 'bottom')

    # From StatsReporter callback (every ~5 s):
    dock.update_throughput(status_dict)

    # From refresh timer (every refresh interval):
    dock.update_queues_and_workers(pipeline_thread.get_live_status())

    # When pipeline stops:
    dock.reset()
"""

from pyqtgraph.dockarea import Dock
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class PipelineStatusDock(Dock):
    """Dock showing throughput stats, queue fill bars, and per-worker health indicators."""

    def __init__(self, parent=None):
        super().__init__("Pipeline Status", closable=False, size=(600, 100))

        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        main_layout.setContentsMargins(6, 4, 6, 4)
        main_layout.setSpacing(12)

        # --- Throughput section ---
        throughput_box = QGroupBox("Throughput")
        tbox_layout = QVBoxLayout(throughput_box)
        tbox_layout.setSpacing(3)
        self._rate_label = QLabel("Rate: —")
        self._data_label = QLabel("Data: — | Elapsed: —")
        self._chunks_label = QLabel("Chunks: —")
        for lbl in (self._rate_label, self._data_label, self._chunks_label):
            lbl.setStyleSheet("font-family: monospace; font-size: 11px;")
            tbox_layout.addWidget(lbl)
        tbox_layout.addStretch()
        main_layout.addWidget(throughput_box)

        # --- Queues section ---
        queues_box = QGroupBox("Queues")
        qbox_layout = QVBoxLayout(queues_box)
        qbox_layout.setSpacing(3)
        self._queue_bars = {}  # qtype -> (QProgressBar, QLabel)
        for qtype in ('raw', 'events', 'pixels', 'triggers'):
            row = QHBoxLayout()
            name_lbl = QLabel(f"{qtype}:")
            name_lbl.setFixedWidth(56)
            name_lbl.setStyleSheet("font-family: monospace; font-size: 11px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(14)
            bar.setTextVisible(False)
            bar.setEnabled(False)
            count_lbl = QLabel("—")
            count_lbl.setFixedWidth(80)
            count_lbl.setStyleSheet("font-family: monospace; font-size: 11px;")
            row.addWidget(name_lbl)
            row.addWidget(bar, stretch=1)
            row.addWidget(count_lbl)
            qbox_layout.addLayout(row)
            self._queue_bars[qtype] = (bar, count_lbl)
        qbox_layout.addStretch()
        main_layout.addWidget(queues_box, stretch=1)

        # --- Workers section ---
        workers_box = QGroupBox("Workers")
        self._workers_grid = QHBoxLayout()
        wbox_layout = QVBoxLayout(workers_box)
        wbox_layout.addLayout(self._workers_grid)
        wbox_layout.addStretch()
        self._worker_labels = []
        main_layout.addWidget(workers_box)

        self.addWidget(widget)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def update_throughput(self, status_dict: dict):
        """Refresh throughput labels from a StatsReporter status dict."""
        if not self.isVisible():
            return
        rate = status_dict.get('rate_mbs', 0.0)
        avg = status_dict.get('avg_rate_mbs', 0.0)
        elapsed = status_dict.get('elapsed', 0.0)
        gb = status_dict.get('bytes_received', 0) / 1e9
        chunks_sent = status_dict.get('chunks_sent', 0)
        d_save = status_dict.get('chunks_dropped_save', 0)
        d_zmq = status_dict.get('chunks_dropped_zmq', 0)
        self._rate_label.setText(f"Rate: {rate:.1f} MB/s (avg: {avg:.1f})")
        self._data_label.setText(f"Data: {gb:.3f} GB | Elapsed: {elapsed:.0f} s")
        self._chunks_label.setText(
            f"Chunks: {chunks_sent} sent | {d_save + d_zmq} dropped "
            f"({d_save} save / {d_zmq} zmq)"
        )

    def update_queues_and_workers(self, live: dict | None):
        """Refresh queue bars and worker LEDs from pipeline_thread.get_live_status()."""
        if not self.isVisible() or live is None:
            return

        queues = live.get('queues', {})
        for qtype, (bar, count_lbl) in self._queue_bars.items():
            qs = queues.get(qtype)
            if qs:
                total_size = sum(s for s, _ in qs)
                total_max = sum(m for _, m in qs) or 1
                pct = min(100, int(100 * total_size / total_max))
                bar.setValue(pct)
                bar.setEnabled(True)
                if len(qs) == 1:
                    count_lbl.setText(f"{qs[0][0]}/{qs[0][1]}")
                else:
                    count_lbl.setText(f"{total_size}/{total_max}")
                if pct > 80:
                    style = "QProgressBar::chunk { background-color: #f44336; }"
                elif pct > 50:
                    style = "QProgressBar::chunk { background-color: #ff9800; }"
                else:
                    style = "QProgressBar::chunk { background-color: #4caf50; }"
                bar.setStyleSheet(style)
            else:
                bar.setValue(0)
                bar.setEnabled(False)
                bar.setStyleSheet("")
                count_lbl.setText("—")

        workers = live.get('workers', [])
        if len(workers) != len(self._worker_labels):
            for lbl in self._worker_labels:
                self._workers_grid.removeWidget(lbl)
                lbl.deleteLater()
            self._worker_labels = []
            for _ in workers:
                lbl = QLabel()
                lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
                self._workers_grid.addWidget(lbl)
                self._worker_labels.append(lbl)

        for lbl, (name, alive) in zip(self._worker_labels, workers):
            idx = name.split('-')[-1]
            color = '#4caf50' if alive else '#f44336'
            lbl.setText(f"● W{idx}")
            lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")
            lbl.setToolTip(f"{name}: {'alive' if alive else 'DEAD'}")

    def reset(self):
        """Return to idle state (no active pipeline)."""
        self._rate_label.setText("Rate: —")
        self._data_label.setText("Data: — | Elapsed: —")
        self._chunks_label.setText("Chunks: —")
        for bar, count_lbl in self._queue_bars.values():
            bar.setValue(0)
            bar.setEnabled(False)
            bar.setStyleSheet("")
            count_lbl.setText("—")
        for lbl in self._worker_labels:
            self._workers_grid.removeWidget(lbl)
            lbl.deleteLater()
        self._worker_labels = []
