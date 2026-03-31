#!/usr/bin/env python3
"""
SERVAL Acquisition GUI

PyQt GUI for SERVAL detector control and TPX3 data acquisition
with real-time histogram visualization and multiple TOF ROI filters.

Uses dock widgets for flexible layout with dynamic ROI image panels.

Usage:
    python -m SERVAL.gui.acquisition_gui
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pymodaq_gui.managers.action_manager import ActionManager
from pymodaq_gui.managers.parameter_manager import ParameterManager

from SERVAL.controllers.serval_control import SERVALController
from SERVAL.core.data_types import TDCChannel, TriggerEdge
from SERVAL.gui.histogram_controller import HistogramController
from SERVAL.gui.pipeline_thread import PipelineThread


# ROI colors palette
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


class ImageDockWidget(QDockWidget):
    """Dock widget containing a 2D histogram image with counts display and time series plot."""

    def __init__(self, title, color=None, parent=None):
        super().__init__(title, parent)
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        self._color = color

        # Main widget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)

        # Header with color indicator, counts, and timeseries toggle
        header = QHBoxLayout()
        if color:
            color_label = QLabel()
            color_label.setFixedSize(16, 16)
            color_label.setStyleSheet(
                f"background-color: rgba({color[0]},{color[1]},{color[2]},200); border: 1px solid black;"
            )
            header.addWidget(color_label)

        self.counts_label = QLabel("Counts: 0")
        self.counts_label.setStyleSheet("font-weight: bold;")
        header.addWidget(self.counts_label)

        self.yield_label = QLabel("  (0.0%)")
        self.yield_label.setStyleSheet("color: gray;")
        header.addWidget(self.yield_label)
        header.addStretch()

        # Image toggle checkbox
        self.image_check = QCheckBox("Image")
        self.image_check.setChecked(True)
        self.image_check.toggled.connect(lambda checked: self.plot.setVisible(checked))
        header.addWidget(self.image_check)

        # Time series toggle checkbox
        self.timeseries_check = QCheckBox("Plot")
        self.timeseries_check.setChecked(True)
        self.timeseries_check.setToolTip("Show counts over time")
        self.timeseries_check.toggled.connect(self._on_timeseries_toggled)
        header.addWidget(self.timeseries_check)

        layout.addLayout(header)

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

        self.setWidget(widget)

    def _on_timeseries_toggled(self, checked):
        """Show/hide time series plot."""
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
        """Set colormap."""
        try:
            colormap = pg.colormap.get(name)
            self.image.setColorMap(colormap)
        except Exception:
            pass

    def is_timeseries_visible(self):
        """Check if time series plot is visible."""
        return self.timeseries_check.isChecked()


class ServalAcquisitionGUI(QMainWindow, ParameterManager, ActionManager):
    """
    Main GUI for SERVAL acquisition with live histograms.

    Features:
    - Dockable image panels
    - Multiple TOF ROI filters with individual images
    - Real-time histogram display
    - ParameterManager-based settings with save/load/search
    """

    settings_name = 'serval_acquisition'
    params = [
        {'title': 'SERVAL Connection', 'name': 'serval', 'type': 'group', 'children': [
            {'title': 'Host', 'name': 'host', 'type': 'str', 'value': '192.168.1.1'},
            {'title': 'Port', 'name': 'port', 'type': 'int', 'value': 8080,
             'limits': (1, 65535)},
            {'title': 'Bias Voltage (V)', 'name': 'bias_voltage', 'type': 'int',
             'value': 40, 'limits': (0, 200)},
            {'title': 'Bias Enabled', 'name': 'bias_enabled', 'type': 'bool',
             'value': True},
            {'title': 'Trigger Mode', 'name': 'trigger_mode', 'type': 'list',
             'limits': ['CONTINUOUS', 'AUTOTRIGSTART_TIMERSTOP', 'EXTERNAL'],
             'value': 'CONTINUOUS'},
            {'title': 'N Triggers', 'name': 'n_triggers', 'type': 'int', 'value': -1,
             'limits': (-1, 1000000)},
            {'title': 'Period (s)', 'name': 'trigger_period', 'type': 'float',
             'value': 0.5, 'limits': (0.001, 1000.0)},
            {'title': 'Exposure (s)', 'name': 'trigger_exposure', 'type': 'float',
             'value': 0.01, 'limits': (0.0001, 100.0)},
            {'title': 'Dest. Host', 'name': 'dest_host', 'type': 'str',
             'value': '192.168.1.2'},
            {'title': 'Dest. Port', 'name': 'dest_port', 'type': 'int', 'value': 8088,
             'limits': (1, 65535)},
        ]},
        {'title': 'Pipeline', 'name': 'pipeline', 'type': 'group', 'children': [
            {'title': 'Workers', 'name': 'num_workers', 'type': 'int', 'value': 4,
             'limits': (1, 16)},
            {'title': 'Fast Extract', 'name': 'use_fast_extract', 'type': 'bool',
             'value': True},
            {'title': 'TDC', 'name': 'tdc_id', 'type': 'list',
             'limits': TDCChannel.labels(), 'value': TDCChannel.TDC1.label},
            {'title': 'Edge', 'name': 'edge', 'type': 'list',
             'limits': TriggerEdge.labels(), 'value': TriggerEdge.RISING.label},
            {'title': 'Event Window Min (ns)', 'name': 'event_window_min', 'type': 'float',
             'value': 0.0},
            {'title': 'Event Window Max (ns)', 'name': 'event_window_max', 'type': 'float',
             'value': 100000.0},
            {'title': 'Output Directory', 'name': 'output_dir', 'type': 'str',
             'value': './data'},
            {'title': 'Run Name', 'name': 'run_name', 'type': 'str', 'value': ''},
            {'title': 'Save Raw', 'name': 'save_raw', 'type': 'bool', 'value': True},
            {'title': 'Save Events', 'name': 'save_events', 'type': 'bool', 'value': True},
            {'title': 'Save Pixels', 'name': 'save_pixels', 'type': 'bool', 'value': False},
            {'title': 'Callback Mode', 'name': 'callback_mode', 'type': 'list',
             'limits': ['events', 'pixels', 'disabled'], 'value': 'events'},
            {'title': 'Centroiding', 'name': 'centroiding', 'type': 'group', 'children': [
                {'title': 'Enable', 'name': 'use_centroiding', 'type': 'bool', 'value': False},
                {'title': 'Spatial eps (px)', 'name': 'eps_space', 'type': 'int',
                 'value': 2, 'limits': (1, 10)},
                {'title': 'Time eps (ns)', 'name': 'eps_time_ns', 'type': 'float',
                 'value': 100.0, 'limits': (1.0, 10000.0), 'step': 1.0, 'decimals': 0},
                {'title': 'Buffer depth', 'name': 'b_size', 'type': 'int',
                 'value': 16, 'limits': (4, 128)},
            ]},
        ]},
        {'title': 'Display', 'name': 'display', 'type': 'group', 'children': [
            {'title': 'Refresh Rate (s)', 'name': 'refresh_rate_s', 'type': 'float',
             'value': 1.0, 'limits': (0.1, 60.0)},
            {'title': 'Auto-clear (s, -1=off)', 'name': 'clear_interval', 'type': 'float',
             'value': -1.0, 'limits': (-1.0, 3600.0)},
            {'title': 'Colormap', 'name': 'colormap', 'type': 'list',
             'limits': ['viridis', 'plasma', 'inferno', 'magma', 'thermal'],
             'value': 'viridis'},
            {'title': 'TOF Max (ns)', 'name': 'tof_max_ns', 'type': 'float',
             'value': 100000.0},
            {'title': 'TOF Bins', 'name': 'tof_bins', 'type': 'int', 'value': 1000,
             'limits': (100, 10000)},
        ]},
    ]

    def __init__(self):
        QMainWindow.__init__(self)
        ParameterManager.__init__(self, action_list=('search', 'save', 'load'))
        ActionManager.__init__(self)

        self.setWindowTitle("SERVAL Acquisition")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 1000)

        # Controllers
        self.serval = SERVALController()
        self.histogram = HistogramController()
        self.pipeline_thread = None

        # State
        self.is_acquiring = False
        self._is_recording = False

        # ROI tracking: name -> {"region": LinearRegionItem, "dock": ImageDockWidget, "color": tuple}
        self._rois = {}
        self._roi_counter = 0

        # Timers
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._update_histograms)

        # Auto-clear: tracked via timestamp, triggered inside _update_histograms
        self._last_clear_time = None

        # Setup UI
        self._setup_ui()
        self._setup_toolbar()
        self._connect_signals()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    # =========================================================================
    # ParameterManager hook
    # =========================================================================
    def value_changed(self, param):
        """React immediately to parameter tree changes."""
        name = param.name()
        if name == 'refresh_rate_s':
            if self.is_acquiring:
                self.refresh_timer.start(int(param.value() * 1000))
        elif name == 'clear_interval':
            # Handled inside _update_histograms; reset the timer so the new
            # interval starts from the current moment.
            if self.is_acquiring and param.value() > 0:
                self._last_clear_time = time.time()
        elif name == 'colormap':
            self._on_colormap_changed(param.value())
        elif name in ('tof_bins', 'tof_max_ns'):
            self._on_tof_config_changed()

    # =========================================================================
    # ActionManager
    # =========================================================================
    def setup_actions(self):
        """Define all toolbar actions."""
        self.add_action('connect', 'Connect', 'cable',
                        tip='Connect / Disconnect SERVAL',
                        checkable=True, shortcut='F2', auto_toolbar=False)
        self.add_action('start', 'Start', 'motion_play',
                        tip='Start acquisition', shortcut='F5', auto_toolbar=False)
        self.add_action('stop', 'Stop', 'stop_circle',
                        tip='Stop acquisition', shortcut='F6',
                        enabled=False, auto_toolbar=False)
        self.add_action('record', 'Record', 'camera_snap',
                        tip='Toggle disk recording', checkable=True,
                        enabled=False, auto_toolbar=False)
        self.add_action('apply_bias', 'Apply Bias', 'Approve',
                        tip='Send bias settings to SERVAL',
                        enabled=False, auto_toolbar=False)
        self.add_action('apply_triggers', 'Apply Triggers', 'gear2',
                        tip='Send trigger settings to SERVAL',
                        enabled=False, auto_toolbar=False)
        self.add_action('add_roi', 'Add ROI', 'add_circle',
                        tip='Add TOF ROI', auto_toolbar=False)
        self.add_action('remove_roi', 'Remove ROI', 'remove',
                        tip='Remove TOF ROI', auto_toolbar=False)
        self.add_action('clear', 'Clear', 'ink_eraser',
                        tip='Clear histograms and time series', auto_toolbar=False)

    def _setup_toolbar(self):
        """Build the main-window toolbar and wire actions to handlers."""
        self.setup_actions()
        tb = self.addToolBar('Acquisition')
        tb.setMovable(False)

        tb.addAction(self.get_action('connect'))
        tb.addSeparator()
        tb.addAction(self.get_action('start'))
        tb.addAction(self.get_action('stop'))
        tb.addAction(self.get_action('record'))
        tb.addSeparator()
        tb.addAction(self.get_action('apply_bias'))
        tb.addAction(self.get_action('apply_triggers'))
        tb.addSeparator()
        tb.addAction(self.get_action('add_roi'))
        tb.addAction(self.get_action('remove_roi'))
        tb.addAction(self.get_action('clear'))

        self.get_action('connect').connect_to(self._on_connect_clicked)
        self.get_action('start').connect_to(self._on_start_clicked)
        self.get_action('stop').connect_to(self._on_stop_clicked)
        self.get_action('record').connect_to(self._on_record_clicked)
        self.get_action('apply_bias').connect_to(self._on_apply_bias)
        self.get_action('apply_triggers').connect_to(self._on_apply_triggers)
        self.get_action('add_roi').connect_to(self._on_add_roi)
        self.get_action('remove_roi').connect_to(self._on_remove_roi)
        self.get_action('clear').connect_to(self._on_clear_all)

    # =========================================================================
    # UI Setup
    # =========================================================================
    def _setup_ui(self):
        """Setup the user interface with dock widgets."""
        # Central widget: TOF histogram
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(5, 5, 5, 5)

        self.tof_plot = pg.PlotWidget(title="Time of Flight (TOF)")
        self.tof_plot.setLabel('left', 'Counts')
        self.tof_plot.setLabel('bottom', 'TOF (ns)')
        self.tof_plot.showGrid(x=True, y=True)

        self.tof_curve = self.tof_plot.plot(
            pen=pg.mkPen('y', width=2),
            fillLevel=0,
            brush=(100, 100, 200, 100)
        )

        central_layout.addWidget(self.tof_plot)
        self.setCentralWidget(central)

        # Left dock: Settings + controls
        self._create_settings_dock()

        # Right dock: Total image
        self.total_dock = ImageDockWidget("Total", parent=self)
        self.total_dock.timeseries_check.setChecked(True)  # Show time series by default
        self.addDockWidget(Qt.RightDockWidgetArea, self.total_dock)

    def _create_settings_dock(self):
        """Create settings dock widget with ParameterTree and action controls."""
        dock = QDockWidget("Settings", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # ParameterTree (replaces 3 tabs)
        layout.addWidget(self.settings_tree, stretch=1)

        # SERVAL action buttons
        serval_group = QGroupBox("SERVAL Actions")
        serval_layout = QVBoxLayout(serval_group)

        conn_row = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        conn_row.addWidget(self.connect_btn)
        self.connection_status = QLabel("Disconnected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold;")
        conn_row.addWidget(self.connection_status)
        serval_layout.addLayout(conn_row)

        apply_row = QHBoxLayout()
        self.apply_bias_btn = QPushButton("Apply Bias")
        self.apply_bias_btn.clicked.connect(self._on_apply_bias)
        self.apply_bias_btn.setEnabled(False)
        apply_row.addWidget(self.apply_bias_btn)

        self.apply_triggers_btn = QPushButton("Apply Triggers")
        self.apply_triggers_btn.clicked.connect(self._on_apply_triggers)
        self.apply_triggers_btn.setEnabled(False)
        apply_row.addWidget(self.apply_triggers_btn)
        serval_layout.addLayout(apply_row)

        layout.addWidget(serval_group)

        # ROI controls (moved from Display tab)
        roi_group = QGroupBox("TOF ROI Filters")
        roi_layout = QVBoxLayout(roi_group)

        btn_layout = QHBoxLayout()
        self.add_roi_btn = QPushButton("Add ROI")
        self.add_roi_btn.clicked.connect(self._on_add_roi)
        btn_layout.addWidget(self.add_roi_btn)

        self.remove_roi_btn = QPushButton("Remove ROI")
        self.remove_roi_btn.clicked.connect(self._on_remove_roi)
        btn_layout.addWidget(self.remove_roi_btn)

        self.clear_btn = QPushButton("Clear Now")
        self.clear_btn.clicked.connect(self._on_clear_all)
        btn_layout.addWidget(self.clear_btn)
        roi_layout.addLayout(btn_layout)

        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(4)
        self.roi_table.setHorizontalHeaderLabels(["", "Name", "Min (ns)", "Max (ns)"])
        self.roi_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.roi_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.roi_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.roi_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.roi_table.setColumnWidth(0, 20)
        self.roi_table.verticalHeader().setVisible(False)
        self.roi_table.setMaximumHeight(150)
        self.roi_table.cellChanged.connect(self._on_roi_table_changed)
        roi_layout.addWidget(self.roi_table)

        layout.addWidget(roi_group)

        # Acquisition controls
        acq_panel = self._create_acquisition_panel()
        layout.addWidget(acq_panel)

        dock.setWidget(widget)
        dock.setMinimumWidth(350)
        dock.setMaximumWidth(420)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _create_acquisition_panel(self):
        """Create acquisition control panel."""
        panel = QGroupBox("Acquisition")
        layout = QVBoxLayout(panel)

        # Start / Stop row
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 8px; }"
            "QPushButton:enabled { background-color: #4CAF50; color: white; }"
        )
        self.start_btn.clicked.connect(self._on_start_clicked)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 8px; }"
            "QPushButton:enabled { background-color: #f44336; color: white; }"
        )
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

        # Record row
        rec_layout = QHBoxLayout()

        self.record_filename_edit = QLineEdit()
        self.record_filename_edit.setPlaceholderText("auto-timestamp")
        self.record_filename_edit.setToolTip("Recording filename (no extension)")
        rec_layout.addWidget(self.record_filename_edit)

        self.record_btn = QPushButton("Record")
        self.record_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }"
            "QPushButton:enabled { background-color: #2196F3; color: white; }"
        )
        self.record_btn.clicked.connect(self._on_record_clicked)
        self.record_btn.setEnabled(False)
        rec_layout.addWidget(self.record_btn)

        layout.addLayout(rec_layout)

        self.stats_label = QLabel("Events: 0")
        self.stats_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.stats_label)

        return panel

    # =========================================================================
    # Signal Connections
    # =========================================================================
    def _connect_signals(self):
        """Connect SERVAL controller signals."""
        self.serval.connected.connect(self._on_serval_connected)
        self.serval.disconnected.connect(self._on_serval_disconnected)
        self.serval.error_occurred.connect(self._on_serval_error)

    # =========================================================================
    # SERVAL Handlers
    # =========================================================================
    def _on_connect_clicked(self):
        if self.serval.is_connected:
            self.serval.disconnect()
        else:
            host = self.settings.child('serval', 'host').value()
            port = self.settings.child('serval', 'port').value()
            self.serval.host = host
            self.serval.port = port
            self.serval.base_url = f'http://{host}:{port}'
            self.status_bar.showMessage(f"Connecting to {host}:{port}...")
            self.serval.connect()

    def _on_serval_connected(self):
        self.connection_status.setText("Connected")
        self.connection_status.setStyleSheet("color: green; font-weight: bold;")
        self.connect_btn.setText("Disconnect")
        self.apply_bias_btn.setEnabled(True)
        self.apply_triggers_btn.setEnabled(True)
        self.get_action('connect').setChecked(True)
        self.get_action('apply_bias').setEnabled(True)
        self.get_action('apply_triggers').setEnabled(True)
        self.status_bar.showMessage("Connected to SERVAL")

    def _on_serval_disconnected(self):
        self.connection_status.setText("Disconnected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold;")
        self.connect_btn.setText("Connect")
        self.apply_bias_btn.setEnabled(False)
        self.apply_triggers_btn.setEnabled(False)
        self.get_action('connect').setChecked(False)
        self.get_action('apply_bias').setEnabled(False)
        self.get_action('apply_triggers').setEnabled(False)
        self.status_bar.showMessage("Disconnected")

    def _on_serval_error(self, msg):
        self.status_bar.showMessage(f"Error: {msg}")

    def _on_apply_bias(self):
        s = self.settings.child('serval')
        voltage = s['bias_voltage']
        enabled = s['bias_enabled']
        if self.serval.set_bias(voltage, enabled):
            self.status_bar.showMessage(f"Bias: {voltage}V (enabled={enabled})")
        else:
            self.status_bar.showMessage("Failed to set bias")

    def _on_apply_triggers(self):
        s = self.settings.child('serval')
        mode = s['trigger_mode']
        n_triggers = s['n_triggers']
        period = s['trigger_period']
        exposure = s['trigger_exposure']

        if self.serval.set_trigger_settings(mode, n_triggers, period, exposure):
            self.status_bar.showMessage(f"Triggers: {mode}")
        else:
            self.status_bar.showMessage("Failed to set triggers")

    # =========================================================================
    # Display Handlers
    # =========================================================================
    def _on_clear_all(self):
        """Clear histograms and time series (called by button)."""
        self.histogram.clear()
        self.histogram.clear_timeseries()
        self._update_histograms()
        self.status_bar.showMessage("Histograms and time series cleared")

    def _on_colormap_changed(self, name):
        self.total_dock.set_colormap(name)
        for roi_data in self._rois.values():
            roi_data["dock"].set_colormap(name)

    def _on_tof_config_changed(self):
        tof_bins = self.settings.child('display', 'tof_bins').value()
        tof_max = self.settings.child('display', 'tof_max_ns').value()
        self.histogram.set_tof_config(tof_range=(0, tof_max), tof_bins=tof_bins)
        self._update_histograms()

    # =========================================================================
    # ROI Management
    # =========================================================================
    def _on_add_roi(self):
        """Add a new TOF ROI."""
        name, ok = QInputDialog.getText(
            self, "Add ROI", "ROI Name:",
            text=f"ROI_{self._roi_counter + 1}"
        )
        if not ok or not name.strip():
            return

        name = name.strip()
        if name in self._rois:
            QMessageBox.warning(self, "Duplicate", f"ROI '{name}' already exists")
            return

        color_idx = self._roi_counter % len(ROI_COLORS)
        color = ROI_COLORS[color_idx]
        self._roi_counter += 1

        tof_max = self.settings.child('display', 'tof_max_ns').value()
        tof_min = tof_max * 0.2
        tof_max_roi = tof_max * 0.4

        region = pg.LinearRegionItem(
            values=[tof_min, tof_max_roi],
            brush=color,
            movable=True
        )
        region.sigRegionChanged.connect(lambda *args, n=name: self._on_roi_region_changing(n))
        region.sigRegionChangeFinished.connect(lambda *args, n=name: self._on_roi_region_changed(n))
        self.tof_plot.addItem(region)

        dock = ImageDockWidget(name, color=color, parent=self)
        dock.setObjectName(f"roi_dock_{name}")
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        if self._rois:
            last_dock = list(self._rois.values())[-1]["dock"]
            self.tabifyDockWidget(last_dock, dock)
        else:
            self.tabifyDockWidget(self.total_dock, dock)

        self._rois[name] = {
            "region": region,
            "dock": dock,
            "color": color,
        }

        self.histogram.add_roi(name, tof_min, tof_max_roi)
        self._update_roi_list()
        self._update_histograms()
        self.status_bar.showMessage(f"Added ROI: {name}")

    def _on_remove_roi(self):
        """Remove a TOF ROI."""
        if not self._rois:
            return

        names = list(self._rois.keys())
        name, ok = QInputDialog.getItem(
            self, "Remove ROI", "Select ROI:",
            names, 0, False
        )
        if not ok:
            return

        region = self._rois[name]["region"]
        self.tof_plot.removeItem(region)

        dock = self._rois[name]["dock"]
        self.removeDockWidget(dock)
        dock.deleteLater()

        self.histogram.remove_roi(name)
        del self._rois[name]

        self._update_roi_list()
        self.status_bar.showMessage(f"Removed ROI: {name}")

    def _on_roi_region_changing(self, name):
        """Live update during drag: refresh ROI range, table, and displays."""
        if name not in self._rois:
            return
        region = self._rois[name]["region"]
        tof_min, tof_max = region.getRegion()
        self.histogram.update_roi(name, tof_min, tof_max)
        self._update_roi_list()
        self._update_histograms()

    def _on_roi_region_changed(self, name):
        """Handle ROI region change on plot."""
        if name not in self._rois:
            return

        region = self._rois[name]["region"]
        tof_min, tof_max = region.getRegion()
        self.histogram.update_roi(name, tof_min, tof_max)
        self._update_roi_list()
        self._update_histograms()

    def _update_roi_list(self):
        """Update the ROI table."""
        self.roi_table.blockSignals(True)
        self.roi_table.setRowCount(len(self._rois))

        for row, (name, roi_data) in enumerate(self._rois.items()):
            color = roi_data["color"]
            color_item = QTableWidgetItem("")
            color_item.setBackground(pg.mkColor(color[0], color[1], color[2], 200))
            color_item.setFlags(Qt.ItemIsEnabled)
            self.roi_table.setItem(row, 0, color_item)

            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.UserRole, name)
            self.roi_table.setItem(row, 1, name_item)

            region = roi_data["region"]
            tof_min, tof_max = region.getRegion()

            min_item = QTableWidgetItem(f"{tof_min:.0f}")
            min_item.setData(Qt.UserRole, name)
            self.roi_table.setItem(row, 2, min_item)

            max_item = QTableWidgetItem(f"{tof_max:.0f}")
            max_item.setData(Qt.UserRole, name)
            self.roi_table.setItem(row, 3, max_item)

        self.roi_table.blockSignals(False)

    def _on_roi_table_changed(self, row, col):
        """Handle ROI table edits."""
        if row >= self.roi_table.rowCount():
            return

        name_item = self.roi_table.item(row, 1)
        if not name_item:
            return

        original_name = name_item.data(Qt.UserRole)
        if original_name not in self._rois:
            return

        if col == 1:
            new_name = name_item.text().strip()
            if new_name and new_name != original_name and new_name not in self._rois:
                self._rename_roi(original_name, new_name)
        elif col in (2, 3):
            try:
                min_item = self.roi_table.item(row, 2)
                max_item = self.roi_table.item(row, 3)
                tof_min = float(min_item.text())
                tof_max = float(max_item.text())
                if tof_min < tof_max:
                    region = self._rois[original_name]["region"]
                    region.blockSignals(True)
                    region.setRegion([tof_min, tof_max])
                    region.blockSignals(False)
                    self.histogram.update_roi(original_name, tof_min, tof_max)
                    self._update_histograms()
            except ValueError:
                pass

    def _rename_roi(self, old_name, new_name):
        """Rename an ROI."""
        if old_name not in self._rois:
            return

        roi_data = self._rois[old_name]
        del self._rois[old_name]
        self._rois[new_name] = roi_data

        roi_data["dock"].setWindowTitle(new_name)
        roi_data["dock"].setObjectName(f"roi_dock_{new_name}")

        tof_min, tof_max = roi_data["region"].getRegion()
        self.histogram.remove_roi(old_name)
        self.histogram.add_roi(new_name, tof_min, tof_max)

        roi_data["region"].sigRegionChanged.disconnect()
        roi_data["region"].sigRegionChangeFinished.disconnect()
        roi_data["region"].sigRegionChanged.connect(
            lambda *args, n=new_name: self._on_roi_region_changing(n))
        roi_data["region"].sigRegionChangeFinished.connect(
            lambda *args, n=new_name: self._on_roi_region_changed(n)
        )

        self._update_roi_list()
        self.status_bar.showMessage(f"Renamed ROI: {old_name} → {new_name}")

    # =========================================================================
    # Config dict builders
    # =========================================================================
    def _build_connection_config(self):
        s = self.settings.child('serval')
        return {
            'host': s['dest_host'],
            'port': s['dest_port'],
        }

    def _build_extract_config(self):
        p = self.settings.child('pipeline')
        c = p.child('centroiding')
        return {
            'num_workers': p['num_workers'],
            'use_fast_extract': p['use_fast_extract'],
            'tdc_id': TDCChannel.from_label(p['tdc_id']),
            'edge': TriggerEdge.from_label(p['edge']),
            'event_window': (
                p['event_window_min'],
                p['event_window_max'],
            ),
            'use_centroiding': c['use_centroiding'],
            'eps_space': c['eps_space'],
            'eps_time_ns': c['eps_time_ns'],
            'b_size': c['b_size'],
        }

    def _build_save_config(self):
        p = self.settings.child('pipeline')
        return {
            'output_dir': p['output_dir'],
            'raw': {'enabled': p['save_raw'], 'num_savers': 1},
            'events': {'enabled': p['save_events'], 'num_savers': 2},
            'pixels': {'enabled': p['save_pixels'], 'num_savers': 1},
        }

    # =========================================================================
    # Acquisition Control
    # =========================================================================
    def _on_start_clicked(self):
        if self.is_acquiring:
            return

        if not self.serval.is_connected:
            QMessageBox.warning(self, "Not Connected", "Please connect to SERVAL first.")
            return

        self.histogram.clear()
        self.histogram.clear_timeseries()

        connection_config = self._build_connection_config()
        extract_config = self._build_extract_config()
        save_config = self._build_save_config()

        callback_mode = self.settings.child('pipeline', 'callback_mode').value()
        callback_config = {
            'mode': callback_mode if callback_mode != 'disabled' else None,
        }

        self.pipeline_thread = PipelineThread(
            connection_config=connection_config,
            save_config=save_config,
            extract_config=extract_config,
            callback_config=callback_config,
        )

        run_name = self.settings.child('pipeline', 'run_name').value().strip()
        if run_name:
            self.pipeline_thread.set_run_name(run_name)

        self.pipeline_thread.event_data_ready.connect(self._on_event_data)
        self.pipeline_thread.pixel_data_ready.connect(self._on_pixel_data)
        self.pipeline_thread.pipeline_started.connect(self._on_pipeline_started)
        self.pipeline_thread.pipeline_stopped.connect(self._on_pipeline_stopped)
        self.pipeline_thread.error_occurred.connect(self._on_pipeline_error)

        self.pipeline_thread.start()
        self.status_bar.showMessage("Starting pipeline...")
        self.start_btn.setEnabled(False)
        self._set_pipeline_controls_enabled(False)

    def _on_pipeline_started(self):
        self.status_bar.showMessage("Pipeline listening...")
        QTimer.singleShot(2000, self._configure_and_start_serval)

    def _configure_and_start_serval(self):
        s = self.settings.child('serval')
        pipeline_host = s['dest_host']
        pipeline_port = s['dest_port']

        if not self.serval.set_destination(pipeline_host, pipeline_port):
            self._on_pipeline_error("Failed to set SERVAL destination")
            self.pipeline_thread.request_stop()
            return

        if not self.serval.start_measurement():
            self._on_pipeline_error("Failed to start SERVAL measurement")
            self.pipeline_thread.request_stop()
            return

        self.is_acquiring = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.record_btn.setEnabled(True)

        self._last_clear_time = time.time()

        refresh_rate_s = self.settings.child('display', 'refresh_rate_s').value()
        if refresh_rate_s > 0:
            self.refresh_timer.start(int(refresh_rate_s * 1000))

        self.status_bar.showMessage("Acquisition running")

    def _on_record_clicked(self):
        """Toggle recording on/off."""
        if not self.is_acquiring:
            return
        pipeline = self.pipeline_thread._pipeline if self.pipeline_thread else None
        if pipeline is None:
            return

        if self._is_recording:
            pipeline.stop_record()
            self._is_recording = False
            self.record_btn.setText("Record")
            self.record_btn.setStyleSheet(
                "QPushButton { font-weight: bold; padding: 6px; }"
                "QPushButton:enabled { background-color: #2196F3; color: white; }"
            )
            self.record_filename_edit.setEnabled(True)
            self.get_action('record').setChecked(False)
            self.status_bar.showMessage("Recording stopped")
        else:
            filename = self.record_filename_edit.text().strip()
            if not filename:
                filename = datetime.now().strftime("rec_%Y%m%d_%H%M%S")
                self.record_filename_edit.setText(filename)
            p = self.settings.child('pipeline')
            ok = pipeline.start_record(
                filename=filename,
                save_raw=p['save_raw'],
                save_events=p['save_events'],
                save_pixels=p['save_pixels'],
            )
            if ok:
                self._is_recording = True
                self.record_btn.setText("Stop Rec")
                self.record_btn.setStyleSheet(
                    "QPushButton { font-weight: bold; padding: 6px; }"
                    "QPushButton:enabled { background-color: #f44336; color: white; }"
                )
                self.record_filename_edit.setEnabled(False)
                self.get_action('record').setChecked(True)
                self.status_bar.showMessage(f"[REC] {filename}")

    def _on_stop_clicked(self):
        if not self.is_acquiring:
            return

        if self._is_recording:
            pipeline = self.pipeline_thread._pipeline if self.pipeline_thread else None
            if pipeline:
                pipeline.stop_record()
            self._is_recording = False

        self.status_bar.showMessage("Stopping...")
        self.serval.stop_measurement()

        if self.pipeline_thread:
            self.pipeline_thread.request_stop()

    def _on_pipeline_stopped(self):
        self.is_acquiring = False
        self._is_recording = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.record_btn.setText("Record")
        self.record_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }"
            "QPushButton:enabled { background-color: #2196F3; color: white; }"
        )
        self.record_filename_edit.setEnabled(True)
        self.get_action('record').setChecked(False)
        self.get_action('record').setEnabled(False)
        self._set_pipeline_controls_enabled(True)
        self.refresh_timer.stop()
        self._last_clear_time = None
        self.status_bar.showMessage("Acquisition stopped")
        self._update_histograms()

    def _on_pipeline_error(self, msg):
        self.status_bar.showMessage(f"Error: {msg}")
        QMessageBox.critical(self, "Pipeline Error", msg)

    def _set_pipeline_controls_enabled(self, enabled):
        """Enable/disable pipeline and SERVAL destination parameters during acquisition."""
        p = self.settings.child('pipeline')
        for name in ['num_workers', 'use_fast_extract', 'tdc_id', 'edge', 'event_window_min',
                     'event_window_max', 'output_dir', 'run_name', 'save_raw',
                     'save_events', 'save_pixels', 'callback_mode']:
            p.child(name).setOpts(enabled=enabled)
        c = p.child('centroiding')
        for name in ['use_centroiding', 'eps_space', 'eps_time_ns', 'b_size']:
            c.child(name).setOpts(enabled=enabled)
        s = self.settings.child('serval')
        for name in ['dest_host', 'dest_port']:
            s.child(name).setOpts(enabled=enabled)
        self.get_action('start').setEnabled(enabled)
        self.get_action('stop').setEnabled(not enabled)
        self.get_action('record').setEnabled(not enabled)

    # =========================================================================
    # Data Handlers
    # =========================================================================
    def _on_event_data(self, event_num, x, y, tof, tot):
        self.histogram.add_events(event_num, x, y, tof, tot)

    def _on_pixel_data(self, x, y, toa, tot):
        self.histogram.add_pixels(x, y, toa, tot)

    def _update_histograms(self):
        """Update all histogram displays."""
        if self.is_acquiring:
            self.histogram.sample_timeseries()

        pixel_data = self.histogram.get_pixel_image()
        total_counts = int(pixel_data.sum())
        self.total_dock.update_image(pixel_data)
        self.total_dock.update_counts(total_counts)

        if self.total_dock.is_timeseries_visible():
            times, counts = self.histogram.get_timeseries(None)
            self.total_dock.update_timeseries(times, counts)

        for name, roi_data in self._rois.items():
            roi_image = self.histogram.get_roi_image(name)
            if roi_image is not None:
                roi_counts = int(roi_image.sum())
                roi_range = self.histogram.get_roi_range(name)
                roi_data["dock"].update_image(roi_image)
                if roi_range:
                    roi_data["dock"].update_counts(roi_counts, f" ({roi_range[0]:.0f}-{roi_range[1]:.0f} ns)")
                else:
                    roi_data["dock"].update_counts(roi_counts)
                pct = (roi_counts / total_counts * 100) if total_counts > 0 else None
                roi_data["dock"].update_yield(pct)

                if roi_data["dock"].is_timeseries_visible():
                    times, counts = self.histogram.get_timeseries(name)
                    roi_data["dock"].update_timeseries(times, counts)

        tof_centers, tof_counts = self.histogram.get_tof_histogram()
        self.tof_curve.setData(tof_centers, tof_counts)

        stats = self.histogram.get_stats()
        self.stats_label.setText(f"Events: {stats['total_events']:,} | Total: {stats['pixel_sum']:,}")

        # Auto-clear: check interval AFTER displaying so we never show a 0-count frame.
        # The 2D histogram accumulates normally; only the per-refresh rate drives the
        # timeseries, so clearing here does not affect counts/shot accuracy.
        if self.is_acquiring and self._last_clear_time is not None:
            clear_interval = self.settings.child('display', 'clear_interval').value()
            if clear_interval > 0 and (time.time() - self._last_clear_time) >= clear_interval:
                self.histogram.clear()
                self._last_clear_time = time.time()
                self.status_bar.showMessage("Histograms cleared")

    # =========================================================================
    # Window Events
    # =========================================================================
    def closeEvent(self, event):
        if self.is_acquiring:
            reply = QMessageBox.question(
                self, "Acquisition Running",
                "Stop acquisition and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

            self._on_stop_clicked()
            if self.pipeline_thread:
                self.pipeline_thread.wait(5000)

        event.accept()


def main():
    # Must be set before any threads (Qt, ZMQ, …) are created.
    # 'forkserver' forks workers from a clean, single-threaded helper process,
    # avoiding the deadlock risk that arises when fork() is called from a
    # multi-threaded parent (Python 3.12+ DeprecationWarning).
    import multiprocessing
    multiprocessing.set_start_method('forkserver', force=False)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ServalAcquisitionGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
