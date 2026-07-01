"""
TofHistogramDock — dock widget containing the TOF histogram and the ROI table panel.

Sub-panels (TOF Binning, ROI table) collapse/expand via CollapsiblePane headers
inside a QSplitter so the user can freely resize them.

Mass calibration settings live in the main settings tree (acquisition_params.py).
"""

import pyqtgraph as pg
from datpx3.analysis.binning import BinSpec
from datpx3.gui.plots.bin_spec_widget import BinSpecWidget

from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtWidgets import (
    QAction,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pymodaq_gui.utils.styling import create_icon

from SERVAL.gui.base_plot_dock import _BasePlotDock
from SERVAL.gui.widgets.collapsible_pane import CollapsiblePane


class TofHistogramDock(_BasePlotDock):
    """
    Dock widget containing the TOF histogram and ROI table.

    Signals
    -------
    clear_requested
        Emitted when the user clicks the Clear button.
    add_roi_clicked / remove_roi_clicked / raise_roi_clicked / zoom_roi_clicked / zoom_out_clicked
        Emitted by the ROI mini-toolbar; wire to ROIManager methods.
    tof_bin_spec_changed
        Emitted when the TOF binning widget changes.
    """

    add_roi_clicked = Signal()
    remove_roi_clicked = Signal()
    raise_roi_clicked = Signal()
    zoom_roi_clicked = Signal()
    zoom_out_clicked = Signal()
    tof_bin_spec_changed = Signal()

    def __init__(self, parent=None):
        super().__init__("TOF Histogram", closable=False, size=(500, 600))

        # ── TOF bin spec widget ───────────────────────────────────────────────
        self.tof_bin_spec_widget = BinSpecWidget(start=0.0, end=100_000.0, n_bins=1000)
        self.tof_bin_spec_widget.changed.connect(self.tof_bin_spec_changed)

        # ── Main layout ───────────────────────────────────────────────────────
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # ── Toolbar ───────────────────────────────────────────────────────────
        self._legend_toggle = QAction(
            create_icon('legend_toggle', icon_color='orange', icon_checked_color='green'),
            'Legend', self
        )
        self._legend_toggle.setCheckable(True)
        self._legend_toggle.setToolTip("Show/hide plot legend")
        self._legend_toggle.toggled.connect(self._on_legend_toggled)

        toolbar = self._build_header_toolbar(
            right_actions=[self._legend_toggle],
            icon_size=(16, 16),
            tool_button_style=Qt.ToolButtonStyle.ToolButtonTextUnderIcon,
            coord_min_width=180,
        )
        layout.addWidget(toolbar)

        # ── TOF histogram plot ────────────────────────────────────────────────
        self.tof_plot = pg.PlotWidget()
        self.tof_plot.setLabel('left', 'Counts')
        self.tof_plot.setLabel('bottom', 'TOF (ns)')
        self.tof_plot.showGrid(x=True, y=True)

        # Legend — hidden by default; toggled with the Legend toolbar button.
        self._tof_legend = self.tof_plot.addLegend()
        self._tof_legend.setVisible(False)

        self.tof_curve = self.tof_plot.plot(
            pen=pg.mkPen('y', width=2),
            fillLevel=0,
            brush=(100, 100, 200, 100),
            name='All',
        )
        self._wire_coord_mouse(self.tof_plot)

        self._spatial_tof_curves: dict[str, pg.PlotDataItem] = {}
        self._combined_tof_curve: pg.PlotDataItem | None = None

        # ── ROI container ─────────────────────────────────────────────────────
        self._roi_container = self._build_roi_container()

        # ── Splitter: binning pane | plot | ROI pane ─────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(CollapsiblePane("TOF Binning", self.tof_bin_spec_widget, expanded=False))
        splitter.addWidget(self.tof_plot)
        self._roi_pane = CollapsiblePane("TOF ROIs", self._roi_container, expanded=False)
        splitter.addWidget(self._roi_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

        layout.addWidget(splitter)
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

        add_action = QAction(create_icon('add_circle'), 'Add ROI', self)
        add_action.setToolTip("Add TOF ROI")
        add_action.triggered.connect(self.add_roi_clicked)
        roi_tb.addAction(add_action)

        remove_action = QAction(create_icon('remove'), 'Remove ROI', self)
        remove_action.setToolTip("Remove TOF ROI")
        remove_action.triggered.connect(self.remove_roi_clicked)
        roi_tb.addAction(remove_action)

        roi_tb.addSeparator()

        raise_btn = QPushButton("Raise")
        raise_btn.setToolTip("Bring the selected ROI dock to front")
        raise_btn.setFixedHeight(24)
        raise_btn.setIcon(create_icon('image_arrow_up'))
        raise_btn.clicked.connect(self.raise_roi_clicked)
        roi_tb.addWidget(raise_btn)

        zoom_btn = QPushButton("Zoom to ROI")
        zoom_btn.setToolTip("Zoom the TOF histogram to the selected ROI's range")
        zoom_btn.setFixedHeight(24)
        zoom_btn.setIcon(create_icon('zoom_in'))
        zoom_btn.clicked.connect(self.zoom_roi_clicked)
        roi_tb.addWidget(zoom_btn)

        zoom_out_btn = QPushButton("Zoom out")
        zoom_out_btn.setToolTip("Zoom out to full range")
        zoom_out_btn.setFixedHeight(24)
        zoom_out_btn.setIcon(create_icon('zoom_out'))
        zoom_out_btn.clicked.connect(self.zoom_out_clicked)
        roi_tb.addWidget(zoom_out_btn)

        roi_layout.addWidget(roi_tb)

        # ROI table — cols: color | name | min | max | vis | lock
        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(6)
        self.roi_table.setHorizontalHeaderLabels(
            ["", "Name", "Min (ns)", "Max (ns)", "Vis", "\U0001f512"]
        )
        hh = self.roi_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.roi_table.setColumnWidth(0, 20)
        self.roi_table.setColumnWidth(4, 30)
        self.roi_table.setColumnWidth(5, 30)
        self.roi_table.verticalHeader().setVisible(False)
        roi_layout.addWidget(self.roi_table)

        return roi_widget

    # ─────────────────────────────────────────────────────────────────────────
    # Internal handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_legend_toggled(self, checked: bool):
        self._tof_legend.setVisible(checked)

    # ─────────────────────────────────────────────────────────────────────────
    # Public helpers
    # ─────────────────────────────────────────────────────────────────────────

    def tof_bin_spec(self) -> BinSpec | None:
        """Return the current TOF BinSpec, or None if the widget state is invalid."""
        return self.tof_bin_spec_widget.bin_spec()

    def update_tof(self, centers, counts):
        """Update the TOF histogram curve."""
        self.tof_curve.setData(centers, counts)

    def set_plot_title(self, title: str):
        self.tof_plot.setTitle(title)

    def set_x_label(self, label: str):
        self.tof_plot.setLabel('bottom', label)

    def set_main_curve_label(self, label: str):
        """Update the legend entry for the main (All) curve."""
        self._tof_legend.removeItem(self.tof_curve)
        self._tof_legend.addItem(self.tof_curve, label)

    # ─────────────────────────────────────────────────────────────────────────
    # Spatially-filtered TOF overlay curves
    # ─────────────────────────────────────────────────────────────────────────

    def add_spatial_tof_curve(self, name: str, color):
        if name in self._spatial_tof_curves:
            return
        curve = self.tof_plot.plot(
            pen=pg.mkPen(color=color[:3], width=1.5),
            name=name,
        )
        self._spatial_tof_curves[name] = curve

    def update_spatial_tof_curve(self, name: str, centers, counts):
        curve = self._spatial_tof_curves.get(name)
        if curve is not None:
            curve.setData(centers, counts)

    def remove_spatial_tof_curve(self, name: str):
        curve = self._spatial_tof_curves.pop(name, None)
        if curve is not None:
            self.tof_plot.removeItem(curve)

    def update_combined_tof_curve(self, centers, counts):
        if self._combined_tof_curve is None:
            self._combined_tof_curve = self.tof_plot.plot(
                pen=pg.mkPen(color='w', width=2,
                             style=pg.QtCore.Qt.PenStyle.DashLine),
                name="Spatial (combined)",
            )
        self._combined_tof_curve.setData(centers, counts)

    def remove_combined_tof_curve(self):
        if self._combined_tof_curve is not None:
            self.tof_plot.removeItem(self._combined_tof_curve)
            self._combined_tof_curve = None
