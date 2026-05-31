"""
Log Dock

Floating log panel that displays filtered messages from SERVAL's Python
logging system.  Attach its handler to the logging infrastructure via
``SERVAL.utils.add_log_handler(log_dock.handler)``.
"""

import logging
from collections import deque
from datetime import datetime

import pyqtgraph as pg
from pyqtgraph.dockarea import Dock
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class _LogSignaller(QObject):
    """Tiny helper to marshal log records from any thread to the GUI thread."""
    record_received = Signal(object)


class QtLogHandler(logging.Handler):
    """logging.Handler that forwards records as Qt signals (thread-safe)."""

    def __init__(self):
        super().__init__()
        self._signaller = _LogSignaller()
        self.record_received = self._signaller.record_received

    def emit(self, record: logging.LogRecord):
        try:
            self._signaller.record_received.emit(record)
        except Exception:
            self.handleError(record)


class LogDock(Dock):
    """
    Dock widget that shows log output from SERVAL components.

    Categories map to logger name prefixes; each has a checkbox so the
    user can filter what they see.  Errors (WARNING+) are always shown
    unless explicitly unchecked.
    """

    # category label -> set of logger-name prefixes that feed it
    CATEGORIES = {
        'Connection': {'SERVAL.TCPReceiver'},
        'Pipeline':   {'SERVAL.Pipeline', 'SERVAL.ExtractorPool', 'SERVAL.Extractor'},
        'Stats':      {'SERVAL.StatsReporter'},
        'Recording':  {
            'SERVAL.RawSaverProcess',
            'SERVAL.EventSaverProcess',
            'SERVAL.PixelSaverProcess',
            'SERVAL.TriggerSaverProcess',
        },
        'GUI':        {'SERVAL.GUI'},
    }

    _LEVEL_COLORS = {
        logging.DEBUG:    '#888888',
        logging.INFO:     '#dddddd',
        logging.WARNING:  '#ffcc44',
        logging.ERROR:    '#ff6666',
        logging.CRITICAL: '#ff2222',
    }

    def __init__(self, parent=None):
        super().__init__("Log", closable=False, size=(600, 150))

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Filter row
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show:"))

        self._checks: dict[str, QCheckBox] = {}
        for cat in self.CATEGORIES:
            cb = QCheckBox(cat)
            cb.setChecked(True)
            cb.toggled.connect(self._rebuild_visible)
            self._checks[cat] = cb
            filter_row.addWidget(cb)

        self._errors_check = QCheckBox("Errors")
        self._errors_check.setChecked(True)
        self._errors_check.toggled.connect(self._rebuild_visible)
        filter_row.addWidget(self._errors_check)

        filter_row.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(52)
        clear_btn.clicked.connect(self._on_clear)
        filter_row.addWidget(clear_btn)
        layout.addLayout(filter_row)

        # Log text area
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.document().setMaximumBlockCount(2000)
        self._text.setFont(QFont("Monospace", 8))
        self._text.setStyleSheet("background-color: #1e1e1e; color: #dddddd;")
        layout.addWidget(self._text)

        self.addWidget(widget)

        # Qt logging handler
        self.handler = QtLogHandler()
        self.handler.record_received.connect(self._on_record)

        # Buffer of received records for filter rebuilds (capped to avoid unbounded growth)
        self._records: deque[logging.LogRecord] = deque(maxlen=10_000)

    def _category_for(self, name: str) -> str | None:
        for cat, prefixes in self.CATEGORIES.items():
            for prefix in prefixes:
                if name.startswith(prefix):
                    return cat
        return None

    def _should_show(self, record: logging.LogRecord) -> bool:
        is_error = record.levelno >= logging.WARNING
        if is_error:
            return self._errors_check.isChecked()
        cat = self._category_for(record.name)
        if cat is None:
            return False
        return self._checks.get(cat, self._errors_check).isChecked()

    def _record_to_html(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLORS.get(record.levelno, '#dddddd')
        ts = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        short_name = record.name.replace('SERVAL.', '')
        msg = record.getMessage().replace('&', '&amp;').replace('<', '&lt;')
        return f'<span style="color:{color}">[{ts}] <b>{short_name}</b>: {msg}</span>'

    def _on_record(self, record: logging.LogRecord):
        self._records.append(record)
        if self._should_show(record):
            self._text.append(self._record_to_html(record))

    def _rebuild_visible(self):
        """Rebuild the log view after a filter checkbox changes."""
        self._text.clear()
        for record in self._records:
            if self._should_show(record):
                self._text.append(self._record_to_html(record))

    def _on_clear(self):
        self._records.clear()
        self._text.clear()
