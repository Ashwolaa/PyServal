"""
_BasePlotDock — shared base for dock widgets that contain a plot + toolbar.

Provides:
  • _build_header_toolbar()  — standard left-widgets / spacer / right-actions / Clear layout
  • _wire_coord_mouse()      — connects sigMouseMoved to the coord label
  • _format_coord()          — override in subclasses to customise coord text
  • clear_requested signal   — emitted by the Clear toolbar button
"""

from pyqtgraph.dockarea import Dock
from pymodaq_gui.utils.styling import create_icon

import pyqtgraph as pg
from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtWidgets import (
    QAction,
    QLabel,
    QSizePolicy,
    QToolBar,
    QWidget,
)

__all__ = ["_BasePlotDock"]


class _BasePlotDock(Dock):
    """Base class for plot docks; provides toolbar factory and coord-label mouse handler."""

    clear_requested = Signal()

    def _build_header_toolbar(
        self,
        left_widgets=(),
        right_actions=(),
        icon_size=(16, 16),
        tool_button_style=Qt.ToolButtonStyle.ToolButtonIconOnly,
        coord_min_width: int = 150,
    ) -> QToolBar:
        """Build and return the standard header toolbar.

        Creates ``self.coord_label`` and appends it after *left_widgets*.
        The Clear action is always appended last (after a separator).
        """
        tb = QToolBar()
        tb.setIconSize(QSize(*icon_size))
        tb.setMovable(False)
        tb.setToolButtonStyle(tool_button_style)

        for w in left_widgets:
            tb.addWidget(w)

        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: gray;")
        self.coord_label.setMinimumWidth(coord_min_width)
        tb.addWidget(self.coord_label)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        for action in right_actions:
            tb.addAction(action)

        tb.addSeparator()

        clear_action = QAction(create_icon('ink_eraser'), 'Clear', self)
        clear_action.setToolTip("Clear histograms and time series")
        clear_action.triggered.connect(self.clear_requested)
        tb.addAction(clear_action)

        return tb

    def _wire_coord_mouse(self, plot_widget: pg.PlotWidget):
        """Connect *plot_widget*'s sigMouseMoved to update ``self.coord_label``."""
        plot_widget.scene().sigMouseMoved.connect(
            lambda pos: self._update_coord_label(plot_widget, pos)
        )

    def _update_coord_label(self, plot_widget: pg.PlotWidget, scene_pos):
        if not plot_widget.sceneBoundingRect().contains(scene_pos):
            self.coord_label.setText("")
            return
        view_pos = plot_widget.getPlotItem().vb.mapSceneToView(scene_pos)
        self.coord_label.setText(self._format_coord(view_pos))

    def _format_coord(self, view_pos) -> str:
        """Override to customise the coordinate label text."""
        return f"x={view_pos.x():.1f}, y={view_pos.y():.0f}"
