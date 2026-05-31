"""
ImageDockWidget — dockable 2-D histogram image panel with time-series strip.

Also exports ROI_COLORS, the default colour palette for TOF ROI regions.
"""

import pyqtgraph as pg
from pyqtgraph.dockarea import Dock
from qtpy.QtCore import Qt, QSize
from qtpy.QtWidgets import (
    QAction,
    QLabel,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pymodaq_gui.utils.styling import create_icon


# ROI colours palette (R, G, B, alpha)
ROI_COLORS = [
    (255, 100, 100, 80),   # Red
    (100, 255, 100, 80),   # Green
    (100, 100, 255, 80),   # Blue
    (255, 255, 100, 80),   # Yellow
    (255, 100, 255, 80),   # Magenta
    (100, 255, 255, 80),   # Cyan
    (255, 180, 100, 80),   # Orange
    (180, 100, 255, 80),   # Purple
]


class ImageDockWidget(Dock):
    """Dock widget containing a 2D histogram image with counts display and time series plot."""

    def __init__(self, title, color=None, parent=None):
        super().__init__(title, closable=True, size=(300, 400))
        self._color = color

        # Main widget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        # Header toolbar: [color●] [Counts] [yield] ─── [image] [chart_data]
        header_tb = QToolBar()
        header_tb.setIconSize(QSize(18, 18))
        header_tb.setMovable(False)
        header_tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)

        self._color_indicator = None
        if color:
            self._color_indicator = QLabel()
            self._color_indicator.setFixedSize(14, 14)
            self._color_indicator.setStyleSheet(
                f"background-color: rgba({color[0]},{color[1]},{color[2]},200);"
                f" border: 1px solid black;"
            )
            header_tb.addWidget(self._color_indicator)

        self.counts_label = QLabel("Counts: 0")
        self.counts_label.setStyleSheet("font-weight: bold;")
        header_tb.addWidget(self.counts_label)

        self.yield_label = QLabel("  (0.0%)")
        self.yield_label.setStyleSheet("color: gray;")
        header_tb.addWidget(self.yield_label)

        _spacer = QWidget()
        _spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        header_tb.addWidget(_spacer)

        # Image toggle (orange = hidden, green = visible)
        self.image_check = QAction(
            create_icon('image', icon_color='orange', icon_checked_color='green'),
            'Image', self
        )
        self.image_check.setCheckable(True)
        self.image_check.setChecked(True)
        self.image_check.setToolTip("Show/hide 2D image")
        self.image_check.toggled.connect(lambda checked: self.plot.setVisible(checked))
        header_tb.addAction(self.image_check)

        # Time-series toggle (orange = hidden, green = visible)
        self.timeseries_check = QAction(
            create_icon('monitoring', icon_color='orange', icon_checked_color='green'),
            'Plot', self
        )
        self.timeseries_check.setCheckable(True)
        self.timeseries_check.setChecked(True)
        self.timeseries_check.setToolTip("Show/hide counts over time")
        self.timeseries_check.toggled.connect(self._on_timeseries_toggled)
        header_tb.addAction(self.timeseries_check)

        layout.addWidget(header_tb)

        # Image plot
        self.plot = pg.PlotWidget()
        self.plot.setLabel('left', 'Y')
        self.plot.setLabel('bottom', 'X')
        self.plot.setAspectLocked(True)

        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.image.setColorMap(pg.colormap.get('viridis'))
        self.plot.setXRange(0, 256)
        self.plot.setYRange(0, 256)

        layout.addWidget(self.plot, stretch=3)

        # Time series plot (visible by default)
        self.timeseries_plot = pg.PlotWidget()
        self.timeseries_plot.setLabel('left', 'Counts/Shot')
        self.timeseries_plot.setLabel('bottom', 'Time', units='s')
        self.timeseries_plot.showGrid(x=True, y=True, alpha=0.3)
        pen_color = color[:3] if color else (100, 100, 255)
        self.timeseries_curve = self.timeseries_plot.plot(pen=pg.mkPen(color=pen_color, width=2))
        self.timeseries_plot.setVisible(True)
        layout.addWidget(self.timeseries_plot, stretch=1)

        self.addWidget(widget)

    def update_color_indicator(self, color: tuple):
        """Update the colour swatch in the header toolbar."""
        self._color = color
        if self._color_indicator is not None:
            r, g, b = color[:3]
            self._color_indicator.setStyleSheet(
                f"background-color: rgba({r},{g},{b},200); border: 1px solid black;"
            )

    def _on_timeseries_toggled(self, checked):
        self.timeseries_plot.setVisible(checked)

    def update_image(self, data):
        """Update the displayed image."""
        self.image.setImage(data.T)

    def update_counts(self, counts, label_suffix=""):
        """Update the counts label."""
        self.counts_label.setText(f"Counts: {counts:,}{label_suffix}")

    def update_yield(self, pct):
        """Update the yield fraction label."""
        self.yield_label.setText(f"  ({pct:.1f}%)" if pct is not None else "  (0.0%)")

    def update_timeseries(self, times, counts):
        """Update the time series plot."""
        if len(times) > 0:
            self.timeseries_curve.setData(times, counts)

    def set_colormap(self, name):
        """Set colormap by name."""
        try:
            colormap = pg.colormap.get(name)
            self.image.setColorMap(colormap)
        except Exception:
            pass

    def set_timeseries_label(self, label: str):
        """Update the Y-axis label on the timeseries plot."""
        self.timeseries_plot.setLabel('left', label)

    def is_timeseries_visible(self):
        """Return True if the time series plot is currently shown."""
        return self.timeseries_check.isChecked()
