"""
TofHistogramDock — dock widget containing the TOF histogram, display settings,
and the ROI table panel.

Owns the ``display`` Parameter group (TOF range, bins, mass calibration).
Internal toolbar buttons emit signals; the caller wires them to ROIManager methods.
Sub-panels collapse/expand via CollapsibleSection accordion headers.
"""

import pyqtgraph as pg
from pyqtgraph.dockarea import Dock
from pyqtgraph.parametertree.Parameter import Parameter
from pymodaq_gui.parameter import ParameterTree
from pymodaq_gui.utils.styling import create_icon
from datpx3.gui.widgets.collapsible import CollapsibleSection

from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtWidgets import (
    QAction,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


class TofHistogramDock(Dock):
    """
    Dock widget containing the TOF histogram, display-settings panel, and ROI table.

    Signals
    -------
    clear_requested
        Emitted when the user clicks the Clear button.
    add_roi_clicked / remove_roi_clicked / raise_roi_clicked / zoom_roi_clicked / zoom_out_clicked
        Emitted by the ROI mini-toolbar; wire to ROIManager methods.
    """

    clear_requested = Signal()
    add_roi_clicked = Signal()
    remove_roi_clicked = Signal()
    raise_roi_clicked = Signal()
    zoom_roi_clicked = Signal()
    zoom_out_clicked = Signal()

    display_params = [
        {'title': 'TOF Min (ns)', 'name': 'tof_min_ns', 'type': 'float',
         'value': 0.0, 'limits': (0.0, 1e9),
         'tip': 'Lower bound of the TOF histogram axis (ns)'},
        {'title': 'TOF Max (ns)', 'name': 'tof_max_ns', 'type': 'float',
         'value': 100000.0, 'limits': (0.0, 1e9),
         'tip': 'Upper bound of the TOF histogram axis (ns)'},
        {'title': 'TOF Bins', 'name': 'tof_bins', 'type': 'int', 'value': 1000,
         'limits': (100, 10000),
         'tip': 'Number of bins in the TOF histogram'},
        {'title': 'Mass Calibration', 'name': 'mass_calib', 'type': 'group', 'children': [
            {'title': 'Enable (show m/z)', 'name': 'enabled', 'type': 'bool', 'value': False,
             'tip': 'Display the histogram in calibrated mass units instead of TOF/TOA'},
            {'title': 'Coeff (ns / sqrt(mass))', 'name': 'coeff', 'type': 'float',
             'value': 1.0, 'tip': 'Calibration slope: tof_ns = coeff * sqrt(mass) + t0'},
            {'title': 't0 (ns)', 'name': 't0', 'type': 'float', 'value': 0.0,
             'tip': 'Calibration time offset: tof_ns = coeff * sqrt(mass) + t0'},
            {'title': 'Mass Min', 'name': 'mass_min', 'type': 'float', 'value': 0.0,
             'tip': 'Lower bound of the mass histogram axis'},
            {'title': 'Mass Max', 'name': 'mass_max', 'type': 'float', 'value': 200.0,
             'tip': 'Upper bound of the mass histogram axis'},
            {'title': 'Mass Bins', 'name': 'mass_bins', 'type': 'int', 'value': 1000,
             'limits': (100, 10000), 'tip': 'Number of bins in the mass histogram'},
        ]},
    ]

    def __init__(self, parent=None):
        super().__init__("TOF Histogram", closable=False, size=(500, 600))

        # ── Display parameters ────────────────────────────────────────────────
        self.display = Parameter.create(
            name='display', type='group', children=self.display_params
        )

        # ── Main layout ───────────────────────────────────────────────────────
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # ── Toolbar: coord | spacer | Legend | sep | Clear ────────────────────
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: gray;")
        self.coord_label.setMinimumWidth(180)
        toolbar.addWidget(self.coord_label)

        _spacer = QWidget()
        _spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(_spacer)

        self._legend_toggle = QAction(
            create_icon('legend_toggle', icon_color='orange', icon_checked_color='green'),
            'Legend', self
        )
        self._legend_toggle.setCheckable(True)
        self._legend_toggle.setToolTip("Show/hide plot legend")
        self._legend_toggle.toggled.connect(self._on_legend_toggled)
        toolbar.addAction(self._legend_toggle)

        toolbar.addSeparator()

        clear_action = QAction(create_icon('ink_eraser'), 'Clear', self)
        clear_action.setToolTip("Clear histograms and time series")
        clear_action.triggered.connect(self.clear_requested)
        toolbar.addAction(clear_action)

        layout.addWidget(toolbar)

        # ── Histogram settings accordion ──────────────────────────────────────
        self.display_tree = ParameterTree()
        self.display_tree.setParameters(self.display, showTop=False)
        layout.addWidget(CollapsibleSection("Histogram settings", self.display_tree))

        # ── TOF histogram plot ────────────────────────────────────────────────
        self.tof_plot = pg.PlotWidget()
        self.tof_plot.setLabel('left', 'Counts')
        self.tof_plot.setLabel('bottom', 'TOF (ns)')
        self.tof_plot.showGrid(x=True, y=True)

        # Legend — hidden by default; toggled with the Legend toolbar button.
        # Must be created before the curves so named curves auto-register.
        self._tof_legend = self.tof_plot.addLegend()
        self._tof_legend.setVisible(False)

        self.tof_curve = self.tof_plot.plot(
            pen=pg.mkPen('y', width=2),
            fillLevel=0,
            brush=(100, 100, 200, 100),
            name='All',
        )
        self.tof_plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self._spatial_tof_curves: dict[str, pg.PlotDataItem] = {}
        self._combined_tof_curve: pg.PlotDataItem | None = None

        layout.addWidget(self.tof_plot, stretch=1)

        # ── ROI Table accordion ───────────────────────────────────────────────
        self._roi_container = self._build_roi_container()
        layout.addWidget(CollapsibleSection("TOF ROIs", self._roi_container))

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

    def _on_mouse_moved(self, scene_pos):
        """Show the cursor's (x, y) position in plot coordinates."""
        if not self.tof_plot.sceneBoundingRect().contains(scene_pos):
            self.coord_label.setText("")
            return
        view_pos = self.tof_plot.getPlotItem().vb.mapSceneToView(scene_pos)
        self.coord_label.setText(f"x={view_pos.x():.1f}, y={view_pos.y():.0f}")

    def _on_legend_toggled(self, checked: bool):
        self._tof_legend.setVisible(checked)

    # ─────────────────────────────────────────────────────────────────────────
    # Public helpers
    # ─────────────────────────────────────────────────────────────────────────

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
