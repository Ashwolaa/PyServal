"""
TofHistogramDock — dock widget containing the TOF histogram, display settings,
and the ROI table panel.

Owns the ``display`` Parameter group (refresh rate, colourmap, TOF range, etc.).
Internal toolbar buttons emit signals; the caller wires them to ROIManager methods.
"""

import pyqtgraph as pg
from pyqtgraph.dockarea import Dock
from pyqtgraph.parametertree.Parameter import Parameter
from pymodaq_gui.parameter import ParameterTree
from pymodaq_gui.utils.styling import create_icon

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
        {'title': 'Refresh Rate (s)', 'name': 'refresh_rate_s', 'type': 'float',
         'value': 1.0, 'limits': (0.1, 60.0)},
        {'title': 'Auto-clear (s, -1=off)', 'name': 'clear_interval', 'type': 'float',
         'value': -1.0, 'limits': (-1.0, 3600.0),
         'tip': 'Automatically clear histogram every N seconds during acquisition. -1 disables.'},
        {'title': 'Colormap', 'name': 'colormap', 'type': 'list',
         'limits': ['viridis', 'plasma', 'inferno', 'magma', 'thermal'],
         'value': 'viridis', 'tip': 'Colour scale for the 2D pixel images'},
        {'title': 'TOF Min (ns)', 'name': 'tof_min_ns', 'type': 'float',
         'value': 0.0, 'limits': (0.0, 1e9),
         'tip': 'Lower bound of the TOF histogram axis (ns)'},
        {'title': 'TOF Max (ns)', 'name': 'tof_max_ns', 'type': 'float',
         'value': 100000.0, 'limits': (0.0, 1e9),
         'tip': 'Upper bound of the TOF histogram axis (ns)'},
        {'title': 'TOF Bins', 'name': 'tof_bins', 'type': 'int', 'value': 1000,
         'limits': (100, 10000),
         'tip': 'Number of bins in the TOF histogram'},
        {'title': 'Display % (throttle)', 'name': 'display_fraction', 'type': 'float',
         'value': 100.0, 'limits': (1.0, 100.0), 'step': 5.0, 'decimals': 0,
         'tip': ('Percentage of incoming events/pixels fed to the live display. '
                 'Subsampling happens in the extractor workers, before the data '
                 'is sent to the GUI, so reducing this also shrinks the '
                 'inter-process transfer cost (lowers display lag at high rates). '
                 'Saving to disk is always full-resolution and unaffected.')},
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

        # ── Internal toolbar ──────────────────────────────────────────────────
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        self._display_toggle = QAction(
            create_icon('settings_applications',
                        icon_color='orange', icon_checked_color='green'),
            'Display', self
        )
        self._display_toggle.setCheckable(True)
        self._display_toggle.setToolTip("Show/hide display settings")
        self._display_toggle.toggled.connect(self._on_display_toggled)
        toolbar.addAction(self._display_toggle)

        toolbar.addSeparator()

        self._roi_table_toggle = QAction(
            create_icon('table_rows',
                        icon_color='orange', icon_checked_color='green'),
            'ROI Table', self
        )
        self._roi_table_toggle.setCheckable(True)
        self._roi_table_toggle.setToolTip("Show/hide ROI table")
        self._roi_table_toggle.toggled.connect(self._on_roi_table_toggled)
        toolbar.addAction(self._roi_table_toggle)

        _spacer = QWidget()
        _spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(_spacer)

        clear_action = QAction(
            create_icon('ink_eraser'), 'Clear', self
        )
        clear_action.setToolTip("Clear histograms and time series")
        clear_action.triggered.connect(self.clear_requested)
        toolbar.addAction(clear_action)

        layout.addWidget(toolbar)

        # ── TOF histogram plot ────────────────────────────────────────────────
        self.tof_plot = pg.PlotWidget()
        self.tof_plot.setLabel('left', 'Counts')
        self.tof_plot.setLabel('bottom', 'TOF (ns)')
        self.tof_plot.showGrid(x=True, y=True)
        self.tof_curve = self.tof_plot.plot(
            pen=pg.mkPen('y', width=2),
            fillLevel=0,
            brush=(100, 100, 200, 100),
        )
        layout.addWidget(self.tof_plot, stretch=3)

        # ── Display settings panel (hidden by default) ────────────────────────
        self.display_tree = ParameterTree()
        self.display_tree.setParameters(self.display, showTop=False)
        self.display_tree.setMaximumHeight(180)
        self.display_tree.setVisible(False)
        layout.addWidget(self.display_tree)

        # ── ROI container (hidden by default) ────────────────────────────────
        self._roi_container = QWidget()
        self._roi_container.setVisible(False)
        roi_layout = QVBoxLayout(self._roi_container)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(1)

        # ROI mini-toolbar
        roi_tb = QToolBar()
        roi_tb.setIconSize(QSize(14, 14))
        roi_tb.setMovable(False)
        roi_tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        add_action = QAction(
            create_icon('add_circle'), 'Add ROI', self
        )
        add_action.setToolTip("Add TOF ROI")
        add_action.triggered.connect(self.add_roi_clicked)
        roi_tb.addAction(add_action)

        remove_action = QAction(
            create_icon('remove'), 'Remove ROI', self
        )
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
        self.roi_table.setMaximumHeight(150)
        roi_layout.addWidget(self.roi_table)

        layout.addWidget(self._roi_container)
        self.addWidget(widget)

    # -------------------------------------------------------------------------
    # Internal toggle handlers
    # -------------------------------------------------------------------------

    def _on_display_toggled(self, checked: bool):
        self.display_tree.setVisible(checked)

    def _on_roi_table_toggled(self, checked: bool):
        self._roi_container.setVisible(checked)

    # -------------------------------------------------------------------------
    # Public helpers
    # -------------------------------------------------------------------------

    def update_tof(self, centers, counts):
        """Update the TOF histogram curve."""
        self.tof_curve.setData(centers, counts)

    def set_plot_title(self, title: str):
        self.tof_plot.setTitle(title)

    def set_x_label(self, label: str):
        self.tof_plot.setLabel('bottom', label)
