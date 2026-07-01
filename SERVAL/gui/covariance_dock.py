"""
CovarianceDock — dock widget displaying the per-shot mass covariance map.

The covariance between two TOF bins m1 and m2 is defined as:

    C(m1, m2) = <n(m1)·n(m2)> - <n(m1)>·<n(m2)>

where n(m) is the count at bin m in a single laser shot and <> averages over
shots.  Positive values indicate species that tend to appear together;
negative values indicate anti-correlated species (e.g. competing channels).
The diagonal C(m, m) = Var(n(m)) is the per-bin variance.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.dockarea import Dock
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QTransform
from qtpy.QtWidgets import (
    QAction,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from datpx3.analysis.binning import BinSpec
from datpx3.gui.plots.bin_spec_widget import BinSpecWidget
from datpx3.gui.widgets.collapsible import CollapsibleSection
from pymodaq_gui.utils.styling import create_icon


def _make_diverging_cmap():
    """Blue → white → red diverging colour map centred at zero."""
    return pg.ColorMap(
        pos=np.array([0.0, 0.5, 1.0]),
        color=np.array([
            [0,   0,   200, 255],
            [255, 255, 255, 255],
            [200, 0,   0,   255],
        ], dtype=np.uint8),
    )


class CovarianceDock(Dock):
    """Dock widget displaying the 2D per-shot covariance (or correlation) map."""

    clear_requested = Signal()
    cov_bin_spec_changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Covariance Map", closable=False, size=(500, 500))

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # ── Toolbar: N shots | coord | spacer | Hide diag | Raw corr | sep | Clear
        tb = QToolBar()
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(False)

        self._shots_label = QLabel("N shots: 0")
        self._shots_label.setStyleSheet("font-family: monospace;")
        tb.addWidget(self._shots_label)

        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: gray;")
        self.coord_label.setMinimumWidth(150)
        tb.addWidget(self.coord_label)

        _spacer = QWidget()
        _spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(_spacer)

        self._mask_diag = QAction('Hide diagonal', self)
        self._mask_diag.setCheckable(True)
        self._mask_diag.setChecked(False)
        self._mask_diag.setToolTip(
            "Zero the diagonal (per-bin variance) to reveal weaker off-diagonal "
            "covariance features without colour-scale saturation")
        tb.addAction(self._mask_diag)
        tb.widgetForAction(self._mask_diag).setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly)

        self._show_corr = QAction('Raw correlation', self)
        self._show_corr.setCheckable(True)
        self._show_corr.setChecked(False)
        self._show_corr.setToolTip(
            "Show <n(m1)·n(m2)> instead of the background-subtracted covariance")
        tb.addAction(self._show_corr)
        tb.widgetForAction(self._show_corr).setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly)

        self._lock_levels = QAction(
            create_icon('lock', icon_color='orange', icon_checked_color='green'),
            'Lock Levels', self
        )
        self._lock_levels.setCheckable(True)
        self._lock_levels.setChecked(False)
        self._lock_levels.setToolTip(
            "Lock the colorbar levels so they don't auto-rescale on every update")
        tb.addAction(self._lock_levels)

        tb.addSeparator()

        clear_action = QAction(create_icon('ink_eraser'), 'Clear', self)
        clear_action.setToolTip("Clear the covariance map")
        clear_action.triggered.connect(self.clear_requested)
        tb.addAction(clear_action)

        layout.addWidget(tb)

        # ── Binning accordion ─────────────────────────────────────────────────
        self.cov_bin_spec_widget = BinSpecWidget(start=0.0, end=100_000.0, n_bins=200)
        self.cov_bin_spec_widget.changed.connect(self.cov_bin_spec_changed)
        layout.addWidget(CollapsibleSection("Covariance Binning", self.cov_bin_spec_widget,
                                           expanded=False))

        # ── Image + LUT ───────────────────────────────────────────────────────
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'TOF (ns)')
        self.plot.setLabel('left', 'TOF (ns)')
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self.image = pg.ImageItem()
        self.plot.addItem(self.image)

        self.histogram_lut = pg.HistogramLUTWidget()
        self.histogram_lut.setImageItem(self.image)
        try:
            self.histogram_lut.gradient.setColorMap(_make_diverging_cmap())
        except Exception:
            pass

        image_row = QWidget()
        row_layout = QHBoxLayout(image_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)
        row_layout.addWidget(self.plot, stretch=4)
        row_layout.addWidget(self.histogram_lut, stretch=1)

        layout.addWidget(image_row)
        self.addWidget(widget)

    # ── Internal handlers ─────────────────────────────────────────────────────

    def _on_mouse_moved(self, scene_pos):
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.coord_label.setText("")
            return
        view_pos = self.plot.getPlotItem().vb.mapSceneToView(scene_pos)
        self.coord_label.setText(f"x={view_pos.x():.1f}, y={view_pos.y():.1f}")

    # ── Public API ────────────────────────────────────────────────────────────

    def update_map(self, centers: np.ndarray, cov_2d: np.ndarray,
                   corr_2d: np.ndarray, n_shots: int):
        """Refresh the displayed map.

        Parameters
        ----------
        centers   : bin centres in current display units (ns or m/z)
        cov_2d    : background-subtracted covariance matrix (B×B)
        corr_2d   : raw correlation matrix  <n(m1)·n(m2)>  (B×B)
        n_shots   : total number of shots accumulated
        """
        n = len(centers)
        if n < 2:
            return

        data = corr_2d if self._show_corr.isChecked() else cov_2d
        display = data.copy()
        if self._mask_diag.isChecked():
            np.fill_diagonal(display, 0.0)

        # Place image pixels at the correct plot coordinates.
        # display[i, j] ↔ (centers[i], centers[j]).
        step = (centers[-1] - centers[0]) / (n - 1)
        tr = QTransform()
        tr.translate(centers[0] - step / 2, centers[0] - step / 2)
        tr.scale(step, step)
        self.image.setTransform(tr)

        locked = self._lock_levels.isChecked()
        # Transpose: pyqtgraph ImageItem has x along columns, y along rows.
        self.image.setImage(display.T, autoLevels=not locked)
        if not locked and not self._show_corr.isChecked():
            vmax = np.abs(display).max()
            if vmax > 0:
                self.image.setLevels([-vmax, vmax])

        self._shots_label.setText(f"N shots: {n_shots:,}")

    def cov_bin_spec(self) -> BinSpec | None:
        """Return the current covariance BinSpec, or None if the widget state is invalid."""
        return self.cov_bin_spec_widget.bin_spec()

    def set_tof_range(self, start: float, end: float):
        """Update the Auto button's memory without forcing a re-bin."""
        self.cov_bin_spec_widget.set_data_range(start, end)

    def set_axis_label(self, label: str):
        self.plot.setLabel('bottom', label)
        self.plot.setLabel('left', label)

    def clear(self):
        self.image.clear()
        self._shots_label.setText("N shots: 0")
