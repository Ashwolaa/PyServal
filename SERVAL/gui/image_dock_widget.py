"""
ImageDockWidget — dockable 2-D histogram image panel with time-series strip.

Also exports ROI_COLORS, the default colour palette for TOF ROI regions.
"""

import pyqtgraph as pg
from pyqtgraph.dockarea import Dock
from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtWidgets import (
    QAction,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pymodaq_gui.utils.styling import create_icon

from SERVAL.gui.widgets.collapsible_pane import CollapsiblePane

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

    # Emitted by the ROI mini-toolbar; wire to a SpatialROIManager's add_roi/remove_roi.
    add_roi_clicked = Signal()
    remove_roi_clicked = Signal()
    clear_requested = Signal()

    def __init__(self, title, color=None, closable=True, parent=None):
        super().__init__(title, closable=closable, size=(300, 400))
        self._color = color

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        # ── Toolbar: color● | Counts | Yield% | coord | spacer | Clear ─────────
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

        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: gray;")
        self.coord_label.setMinimumWidth(150)
        header_tb.addWidget(self.coord_label)

        _spacer = QWidget()
        _spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        header_tb.addWidget(_spacer)

        self.lock_levels_check = QAction(
            create_icon('lock', icon_color='orange', icon_checked_color='green'),
            'Lock Levels', self
        )
        self.lock_levels_check.setCheckable(True)
        self.lock_levels_check.setChecked(False)
        self.lock_levels_check.setToolTip(
            "Lock the colorbar levels so they don't auto-rescale on every update")
        header_tb.addAction(self.lock_levels_check)

        header_tb.addSeparator()

        clear_action = QAction(create_icon('ink_eraser'), 'Clear', self)
        clear_action.setToolTip("Clear image and time series")
        clear_action.triggered.connect(self.clear_requested)
        header_tb.addAction(clear_action)

        layout.addWidget(header_tb)

        # ── Image + LUT ───────────────────────────────────────────────────────
        self.plot = pg.PlotWidget()
        self.plot.setLabel('left', 'Y')
        self.plot.setLabel('bottom', 'X')
        self.plot.setAspectLocked(True)

        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.image.setColorMap(pg.colormap.get('viridis'))
        self.plot.setXRange(0, 256)
        self.plot.setYRange(0, 256)
        self._last_data = None
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # Histogram LUT widget — drag the lower/upper handles to tune intensity scaling
        self.histogram_lut = pg.HistogramLUTWidget()
        self.histogram_lut.setImageItem(self.image)
        self.histogram_lut.gradient.loadPreset('viridis')

        self._image_container = QWidget()
        image_row = QHBoxLayout(self._image_container)
        image_row.setContentsMargins(0, 0, 0, 0)
        image_row.setSpacing(2)
        image_row.addWidget(self.plot, stretch=4)
        image_row.addWidget(self.histogram_lut, stretch=1)

        # ── Splitter: image | counts-over-time | spatial ROIs ─────────────────
        # Each sub-panel is a _CollapsiblePane: drag the handle to resize,
        # click the ▶/▼ header to collapse/expand.
        self.timeseries_plot = pg.PlotWidget()
        self.timeseries_plot.setLabel('left', 'Counts/Shot')
        self.timeseries_plot.setLabel('bottom', 'Time', units='s')
        self.timeseries_plot.showGrid(x=True, y=True, alpha=0.3)
        pen_color = color[:3] if color else (100, 100, 255)
        self.timeseries_curve = self.timeseries_plot.plot(
            pen=pg.mkPen(color=pen_color, width=2), name='Total')
        self._timeseries_legend = self.timeseries_plot.addLegend()
        self._roi_curves = {}  # name -> PlotDataItem
        self._time_window_s: float = -1.0  # -1 = show all history

        self._roi_container = self._build_roi_container()

        self._timeseries_pane = CollapsiblePane(
            "Counts over time", self.timeseries_plot, expanded=True
        )
        self._roi_pane = CollapsiblePane(
            "Spatial ROIs", self._roi_container, expanded=False
        )

        self._vsplitter = QSplitter(Qt.Orientation.Vertical)
        self._vsplitter.addWidget(self._image_container)
        self._vsplitter.addWidget(self._timeseries_pane)
        self._vsplitter.addWidget(self._roi_pane)
        self._vsplitter.setStretchFactor(0, 3)
        self._vsplitter.setStretchFactor(1, 1)
        self._vsplitter.setStretchFactor(2, 0)  # starts collapsed → no extra space

        layout.addWidget(self._vsplitter)
        self.addWidget(widget)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_roi_container(self) -> QWidget:
        roi_widget = QWidget()
        roi_layout = QVBoxLayout(roi_widget)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(1)

        roi_tb = QToolBar()
        roi_tb.setIconSize(QSize(14, 14))
        roi_tb.setMovable(False)
        roi_tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        add_roi_action = QAction(create_icon('add_circle'), 'Add ROI', self)
        add_roi_action.setToolTip("Add a spatial ROI rectangle on this image")
        add_roi_action.triggered.connect(self.add_roi_clicked)
        roi_tb.addAction(add_roi_action)

        remove_roi_action = QAction(create_icon('remove'), 'Remove ROI', self)
        remove_roi_action.setToolTip("Remove a spatial ROI")
        remove_roi_action.triggered.connect(self.remove_roi_clicked)
        roi_tb.addAction(remove_roi_action)

        roi_layout.addWidget(roi_tb)

        # ROI table — cols: color | name | shape | op | x | y | w | h | counts | yield% | vis | lock
        # Shape (col 2): "▭" rect  / "○" ellipse  — click to cycle
        # Op    (col 3): "⊕" include / "⊖" exclude — click to toggle; drives the combined mask
        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(12)
        self.roi_table.setHorizontalHeaderLabels(
            ["", "Name", "▭/○", "⊕/⊖", "X", "Y", "W", "H",
             "Counts", "Yield %", "Vis", "\U0001f512"]
        )
        hh = self.roi_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for col in (2, 3, 4, 5, 6, 7, 8, 9):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(10, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(11, QHeaderView.ResizeMode.Fixed)
        self.roi_table.setColumnWidth(0, 20)
        self.roi_table.setColumnWidth(10, 30)
        self.roi_table.setColumnWidth(11, 30)
        self.roi_table.verticalHeader().setVisible(False)
        roi_layout.addWidget(self.roi_table)

        return roi_widget

    # ─────────────────────────────────────────────────────────────────────────
    # Internal handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_mouse_moved(self, scene_pos):
        """Show the pixel coordinate (and value) under the mouse cursor."""
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.coord_label.setText("")
            return
        view_pos = self.plot.getPlotItem().vb.mapSceneToView(scene_pos)
        ix, iy = int(view_pos.x()), int(view_pos.y())
        text = f"x={ix}, y={iy}"
        if (self._last_data is not None
                and 0 <= iy < self._last_data.shape[0] and 0 <= ix < self._last_data.shape[1]):
            text += f", val={self._last_data[iy, ix]:,}"
        self.coord_label.setText(text)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update_color_indicator(self, color: tuple):
        """Update the colour swatch in the header toolbar."""
        self._color = color
        if self._color_indicator is not None:
            r, g, b = color[:3]
            self._color_indicator.setStyleSheet(
                f"background-color: rgba({r},{g},{b},200); border: 1px solid black;"
            )

    def update_image(self, data):
        """Update the displayed image."""
        self._last_data = data
        self.image.setImage(data.T, autoLevels=not self.lock_levels_check.isChecked())

    def update_counts(self, counts, label_suffix=""):
        """Update the counts label."""
        self.counts_label.setText(f"Counts: {counts:,}{label_suffix}")

    def update_yield(self, pct):
        """Update the yield fraction label."""
        self.yield_label.setText(f"  ({pct:.1f}%)" if pct is not None else "  (0.0%)")

    def set_time_window(self, max_s: float):
        """Set the trailing time window shown on the timeseries X-axis (-1 = show all)."""
        self._time_window_s = max_s
        if max_s <= 0:
            self.timeseries_plot.enableAutoRange(axis='x')

    def update_timeseries(self, times, counts):
        """Update the time series plot."""
        if len(times) > 0:
            self.timeseries_curve.setData(times, counts)
            if self._time_window_s > 0:
                t_max = float(times[-1])
                self.timeseries_plot.setXRange(
                    max(0.0, t_max - self._time_window_s), t_max, padding=0
                )

    def set_colormap(self, name):
        """Set colormap by name."""
        try:
            colormap = pg.colormap.get(name)
            self.image.setColorMap(colormap)
            self.histogram_lut.gradient.loadPreset(name)
        except Exception:
            pass

    def set_timeseries_label(self, label: str):
        """Update the Y-axis label on the timeseries plot."""
        self.timeseries_plot.setLabel('left', label)

    def set_timeseries_visible(self, visible: bool):
        """Expand or collapse the counts-over-time pane."""
        self._timeseries_pane.set_expanded(visible)

    def is_timeseries_visible(self) -> bool:
        """Return True if the counts-over-time pane is expanded."""
        return self._timeseries_pane.is_expanded

    # ─────────────────────────────────────────────────────────────────────────
    # Spatial ROI curves — one extra curve per ROI, overlaid on timeseries_plot
    # ─────────────────────────────────────────────────────────────────────────

    def add_roi_curve(self, name, color):
        """Add a new counts/shot curve for a spatial ROI named *name*."""
        curve = self.timeseries_plot.plot(
            pen=pg.mkPen(color=color[:3], width=2), name=name)
        self._roi_curves[name] = curve

    def remove_roi_curve(self, name):
        """Remove a spatial ROI's curve."""
        curve = self._roi_curves.pop(name, None)
        if curve is not None:
            self.timeseries_plot.removeItem(curve)
            self._timeseries_legend.removeItem(name)

    def update_roi_curve(self, name, times, counts):
        """Update a spatial ROI's curve data."""
        curve = self._roi_curves.get(name)
        if curve is not None and len(times) > 0:
            curve.setData(times, counts)

    def set_roi_curve_visible(self, name, visible):
        """Show/hide a spatial ROI's curve without removing it."""
        curve = self._roi_curves.get(name)
        if curve is not None:
            curve.setVisible(visible)

    def rename_roi_curve(self, old_name, new_name, color):
        """Swap a curve's identity (used when a spatial ROI is renamed)."""
        curve = self._roi_curves.pop(old_name, None)
        if curve is not None:
            self.timeseries_plot.removeItem(curve)
            self._timeseries_legend.removeItem(old_name)
        self.add_roi_curve(new_name, color)

    # ── Combined-mask curve ───────────────────────────────────────────────────

    def ensure_combined_curve(self):
        """Create the 'Combined' timeseries curve if it doesn't exist yet."""
        if "Combined" not in self._roi_curves:
            curve = self.timeseries_plot.plot(
                pen=pg.mkPen(color=(220, 220, 220), width=2,
                             style=pg.QtCore.Qt.PenStyle.DashLine),
                name="Combined",
            )
            self._roi_curves["Combined"] = curve

    def remove_combined_curve(self):
        self.remove_roi_curve("Combined")

    def update_combined_curve(self, times, counts):
        self.update_roi_curve("Combined", times, counts)
