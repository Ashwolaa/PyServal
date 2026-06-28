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
from qtpy.QtCore import QSize
from qtpy.QtGui import QTransform
from qtpy.QtWidgets import (
    QAction,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

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

    def __init__(self, parent=None):
        super().__init__("Covariance Map", closable=False, size=(500, 500))

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # ── Toolbar ───────────────────────────────────────────────────────────
        tb = QToolBar()
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(False)

        self._shots_label = QLabel("N shots: 0")
        self._shots_label.setStyleSheet("font-family: monospace;")
        tb.addWidget(self._shots_label)

        _spacer = QWidget()
        _spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(_spacer)

        self._mask_diag = QCheckBox("Hide diagonal")
        self._mask_diag.setToolTip(
            "Zero the diagonal (per-bin variance) to reveal weaker off-diagonal "
            "covariance features without colour-scale saturation")
        self._mask_diag.setChecked(False)
        tb.addWidget(self._mask_diag)

        self._show_corr = QCheckBox("Raw correlation")
        self._show_corr.setToolTip(
            "Show <n(m1)·n(m2)> instead of the background-subtracted covariance")
        self._show_corr.setChecked(False)
        tb.addWidget(self._show_corr)

        self._lock_levels = QAction(
            create_icon('lock', icon_color='orange', icon_checked_color='green'),
            'Lock Levels', self,
        )
        self._lock_levels.setCheckable(True)
        self._lock_levels.setChecked(False)
        self._lock_levels.setToolTip("Lock colour-bar levels (stop auto-rescaling)")
        tb.addAction(self._lock_levels)

        layout.addWidget(tb)

        # ── Image + LUT ───────────────────────────────────────────────────────
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'TOF (ns)')
        self.plot.setLabel('left', 'TOF (ns)')
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.2)

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

        auto = not self._lock_levels.isChecked()
        # Transpose: pyqtgraph ImageItem has x along columns, y along rows.
        self.image.setImage(display.T, autoLevels=auto)
        if auto and not self._show_corr.isChecked():
            vmax = np.abs(display).max()
            if vmax > 0:
                self.image.setLevels([-vmax, vmax])

        self._shots_label.setText(f"N shots: {n_shots:,}")

    def set_axis_label(self, label: str):
        self.plot.setLabel('bottom', label)
        self.plot.setLabel('left', label)

    def clear(self):
        self.image.clear()
        self._shots_label.setText("N shots: 0")
