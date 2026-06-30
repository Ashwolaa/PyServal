"""
CollapsiblePane — ▶/▼ header + content widget for use inside a QSplitter.

Unlike CollapsibleSection (animation-based), visibility is toggled with
setVisible() so the splitter can freely resize the expanded content without
fighting a maximumHeight constraint.
"""
from __future__ import annotations

from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

__all__ = ["CollapsiblePane"]


class CollapsiblePane(QWidget):
    """▶/▼ title header + content widget, designed to live inside a QSplitter.

    Collapse/expand via setVisible() so the pane height is fully under
    splitter control when expanded.
    """

    def __init__(self, title: str, content: QWidget, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._title = title
        self._content = content
        self._expanded = expanded

        self._header = QPushButton()
        self._header.setFlat(True)
        self._header.setStyleSheet("text-align: left; padding: 2px 4px;")
        self._header.clicked.connect(self._toggle)
        self._update_header()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(content)

        content.setVisible(expanded)

    @property
    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool):
        if expanded != self._expanded:
            self._expanded = expanded
            self._content.setVisible(expanded)
            self._update_header()

    def _toggle(self):
        self.set_expanded(not self._expanded)

    def _update_header(self):
        arrow = "▼" if self._expanded else "▶"
        self._header.setText(f"{arrow}  {self._title}")
