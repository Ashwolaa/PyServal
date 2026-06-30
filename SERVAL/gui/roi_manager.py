"""
ROIManager — manages TOF LinearRegionItems on the histogram and their
associated ImageDockWidgets.

The caller wires the three data-flow signals:

    roi_manager.roi_added.connect(lambda n,mn,mx: histogram.add_roi(n, mn, mx))
    roi_manager.roi_removed.connect(histogram.remove_roi)
    roi_manager.roi_changed.connect(histogram.update_roi)
"""

import math

import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QMenu, QTableWidget, QTableWidgetItem

from SERVAL.gui.base_roi_manager import BaseROIManager
from SERVAL.gui.image_dock_widget import ImageDockWidget, ROI_COLORS
from SERVAL.gui.spatial_roi_manager import SpatialROIManager


class ROIManager(BaseROIManager):
    """
    Manages TOF ROI regions: LinearRegionItems on the histogram and their
    associated ImageDockWidgets.

    Parameters
    ----------
    tof_plot : pg.PlotWidget
        The TOF histogram plot where LinearRegionItems are added.
    roi_table : QTableWidget
        The 6-column table widget (owned by TofHistogramDock).
    display : Parameter
        The display Parameter group (used to read tof_min/max_ns for new ROIs).
    main_window : QMainWindow
        Used as the parent for dock-related dialogs.
    dock_area : DockArea
        The pyqtgraph DockArea that owns all docks.
    total_dock : ImageDockWidget
        The "Total" dock; the first ROI dock is tabified next to it.
    """

    roi_added = Signal(str, float, float)    # name, min_ns, max_ns
    roi_removed = Signal(str)                # name
    roi_changed = Signal(str, float, float)  # name, min_ns, max_ns
    clear_requested = Signal()               # re-emitted from any ROI dock's Clear button
    dock_created = Signal(object)            # emitted when a new ImageDockWidget is added

    # Forwarded from each TOF ROI's own SpatialROIManager
    spatial_roi_added = Signal(str, str, str, str, int, int, int, int)  # parent,name,shape,op,x,y,w,h
    spatial_roi_removed = Signal(str, str)                               # parent, name
    spatial_roi_changed = Signal(str, str, str, str, int, int, int, int) # parent,name,shape,op,x,y,w,h
    spatial_rois_cleared = Signal(str)                                   # parent

    def __init__(self, tof_plot, roi_table: QTableWidget, display, main_window,
                 dock_area, total_dock, parent=None):
        super().__init__(parent)
        self._tof_plot = tof_plot
        self._roi_table = roi_table
        self._display = display
        self._main_window = main_window
        self._dockarea = dock_area
        self._total_dock = total_dock

        self._rois: dict = {}       # name -> {"region", "dock", "color", "locked", "spatial_manager",
                                  #           "tof_min_ns", "tof_max_ns"}
        self._roi_counter: int = 0

        # Mass-calibration state — kept in sync by update_mass_calibration()
        self._mass_enabled = False
        self._mass_coeff = 1.0
        self._mass_t0 = 0.0

        self._roi_table.cellChanged.connect(self._on_table_changed)
        self._roi_table.cellClicked.connect(self._on_table_cell_clicked)
        self._roi_table.doubleClicked.connect(self.raise_selected)
        self._roi_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._roi_table.customContextMenuRequested.connect(self._on_context_menu)

    # ── BaseROIManager contract ───────────────────────────────────────────────

    @property
    def _table(self) -> QTableWidget:
        return self._roi_table

    @property
    def docks(self) -> list:
        """Return the ImageDockWidget for every live TOF ROI."""
        return [r["dock"] for r in self._rois.values()]

    def _pg_item(self, name: str):
        return self._rois[name]["region"]

    def _apply_color(self, name: str, color: tuple):
        roi_data = self._rois[name]
        roi_data["region"].setBrush(pg.mkBrush(*color))
        roi_data["dock"].update_color_indicator(color)
        r, g, b, _a = color
        roi_data["dock"].timeseries_curve.setPen(pg.mkPen(color=(r, g, b), width=2))

    # ── Mass calibration ──────────────────────────────────────────────────────

    def _ns_to_display(self, tof_ns: float) -> float:
        """Convert ns to the current plot display unit (m/z or ns)."""
        if self._mass_enabled:
            return max(0.0, (tof_ns - self._mass_t0) / self._mass_coeff) ** 2
        return tof_ns

    def _display_to_ns(self, display_val: float) -> float:
        """Convert the current plot display unit back to ns."""
        if self._mass_enabled:
            return self._mass_coeff * math.sqrt(max(0.0, display_val)) + self._mass_t0
        return display_val

    def update_mass_calibration(self, coeff: float, t0: float, enabled: bool):
        """Called when the mass-calibration parameters or enable state changes.

        Re-projects every existing LinearRegionItem from its stored ns position
        to the new display units, and updates the table column headers.
        """
        self._mass_coeff = coeff if coeff else 1.0
        self._mass_t0 = t0
        self._mass_enabled = enabled

        for roi_data in self._rois.values():
            lo_display = self._ns_to_display(roi_data["tof_min_ns"])
            hi_display = self._ns_to_display(roi_data["tof_max_ns"])
            region = roi_data["region"]
            region.blockSignals(True)
            region.setRegion([lo_display, hi_display])
            region.blockSignals(False)

        unit = "m/z" if enabled else "ns"
        self._roi_table.setHorizontalHeaderLabels(
            ["", "Name", f"Min ({unit})", f"Max ({unit})", "Vis", "\U0001f512"])
        self._refresh_table()

    # ── Context-menu extras ───────────────────────────────────────────────────

    def _add_extra_menu_actions(self, menu: QMenu, name: str, row: int, acts: dict):
        acts["raise"]    = menu.addAction("Raise dock")
        acts["zoom"]     = menu.addAction("Zoom to ROI")
        acts["zoom_out"] = menu.addAction("Zoom out")

    def _handle_extra_menu_action(self, action, name: str, row: int, acts: dict) -> bool:
        if action == acts.get("raise"):
            self._roi_table.selectRow(row)
            self.raise_selected()
            return True
        if action == acts.get("zoom"):
            self._roi_table.selectRow(row)
            self.zoom_to_selected()
            return True
        if action == acts.get("zoom_out"):
            self.zoom_out()
            return True
        return False

    # ── Public API ────────────────────────────────────────────────────────────

    def add_roi(self):
        """Create a new TOF ROI and put the name cell into edit mode immediately."""
        name = f"ROI_{self._roi_counter + 1}"
        while name in self._rois:
            self._roi_counter += 1
            name = f"ROI_{self._roi_counter + 1}"

        color = ROI_COLORS[self._roi_counter % len(ROI_COLORS)]
        self._roi_counter += 1

        tof_ns_lo = self._display.child('tof_min_ns').value()
        tof_ns_hi = self._display.child('tof_max_ns').value()
        span_ns = tof_ns_hi - tof_ns_lo
        tof_min_ns = tof_ns_lo + span_ns * 0.2
        tof_max_ns = tof_ns_lo + span_ns * 0.4

        region = pg.LinearRegionItem(
            values=[self._ns_to_display(tof_min_ns), self._ns_to_display(tof_max_ns)],
            brush=color, movable=True)
        region.sigRegionChanged.connect(
            lambda *_args, n=name: self._on_region_changing(n))
        region.sigRegionChangeFinished.connect(
            lambda *_args, n=name: self._on_region_changed(n))
        self._tof_plot.addItem(region)

        dock = ImageDockWidget(name, color=color)
        dock.sigClosed.connect(lambda _d, n=name: self._on_dock_closed(n))
        dock.clear_requested.connect(self.clear_requested)

        if self._rois:
            self._dockarea.addDock(dock, 'above', list(self._rois.values())[-1]["dock"])
        else:
            self._dockarea.addDock(dock, 'above', self._total_dock)

        spatial_manager = SpatialROIManager(name, dock, self._main_window)
        spatial_manager.roi_added.connect(self.spatial_roi_added.emit)
        spatial_manager.roi_removed.connect(self.spatial_roi_removed.emit)
        spatial_manager.roi_changed.connect(self.spatial_roi_changed.emit)

        self._rois[name] = {"region": region, "dock": dock,
                            "color": color, "locked": False,
                            "spatial_manager": spatial_manager,
                            "tof_min_ns": tof_min_ns, "tof_max_ns": tof_max_ns}

        self.dock_created.emit(dock)
        self._refresh_table()
        self.roi_added.emit(name, tof_min_ns, tof_max_ns)

        row = len(self._rois) - 1
        name_item = self._roi_table.item(row, 1)
        if name_item is not None:
            self._roi_table.setCurrentItem(name_item)
            self._roi_table.editItem(name_item)

    def raise_selected(self, _index=None):
        """Show and raise the dock for the currently selected table row."""
        name = self._selected_name()
        if name is None:
            return
        dock = self._rois[name]["dock"]
        dock.show()
        dock.raiseDock()
        self._refresh_table()

    def zoom_to_selected(self):
        """Zoom the TOF histogram X axis to the selected ROI's range."""
        name = self._selected_name()
        if name is None:
            return
        tof_min, tof_max = self._rois[name]["region"].getRegion()
        self._tof_plot.setXRange(tof_min, tof_max, padding=0.05)

    def zoom_out(self):
        """Zoom the TOF histogram X axis to full range."""
        self._tof_plot.autoRange()

    def update_displays(self, histogram, total_counts: int):
        """Refresh all ROI ImageDockWidgets from *histogram*."""
        for name, roi_data in self._rois.items():
            roi_image = histogram.get_roi_image(name)
            if roi_image is None:
                continue
            roi_counts = int(roi_image.sum())
            roi_range = histogram.get_roi_range(name)
            roi_data["dock"].update_image(roi_image)
            if roi_range:
                roi_data["dock"].update_counts(
                    roi_counts, f" ({roi_range[0]:.0f}-{roi_range[1]:.0f} ns)")
            else:
                roi_data["dock"].update_counts(roi_counts)
            pct = (roi_counts / total_counts * 100) if total_counts > 0 else None
            roi_data["dock"].update_yield(pct)
            if roi_data["dock"].is_timeseries_visible():
                times, counts = histogram.get_timeseries(name)
                roi_data["dock"].update_timeseries(times, counts)
            roi_data["spatial_manager"].update_displays(histogram, roi_counts)

    def set_timeseries_label(self, label: str):
        for roi_data in self._rois.values():
            roi_data["dock"].set_timeseries_label(label)

    def set_colormap(self, name: str):
        for roi_data in self._rois.values():
            roi_data["dock"].set_colormap(name)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _remove_roi_by_name(self, name: str):
        if name not in self._rois:
            return
        roi_data = self._rois.pop(name)
        self._tof_plot.removeItem(roi_data["region"])
        roi_data["dock"].close()
        roi_data["dock"].deleteLater()
        self._refresh_table()
        self.roi_removed.emit(name)
        self.spatial_rois_cleared.emit(name)

    def _on_dock_closed(self, name: str):
        """Called when a ROI dock's close button is clicked."""
        if name not in self._rois:
            return
        roi_data = self._rois.pop(name)
        self._tof_plot.removeItem(roi_data["region"])
        roi_data["dock"].deleteLater()
        self._refresh_table()
        self.roi_removed.emit(name)
        self.spatial_rois_cleared.emit(name)

    def _rename_roi(self, old_name: str, new_name: str):
        if old_name not in self._rois:
            return
        roi_data = self._rois.pop(old_name)
        self._rois[new_name] = roi_data

        roi_data["dock"].setTitle(new_name)

        self.roi_removed.emit(old_name)
        self.roi_added.emit(new_name, roi_data["tof_min_ns"], roi_data["tof_max_ns"])

        spatial_manager = roi_data["spatial_manager"]
        geometries = spatial_manager.get_roi_geometries()
        self.spatial_rois_cleared.emit(old_name)
        spatial_manager.set_parent_key(new_name)
        for roi_name, shape, op, x, y, w, h in geometries:
            self.spatial_roi_added.emit(new_name, roi_name, shape, op, x, y, w, h)

        roi_data["region"].sigRegionChanged.disconnect()
        roi_data["region"].sigRegionChangeFinished.disconnect()
        roi_data["region"].sigRegionChanged.connect(
            lambda *_: self._on_region_changing(new_name))
        roi_data["region"].sigRegionChangeFinished.connect(
            lambda *_: self._on_region_changed(new_name))

        roi_data["dock"].sigClosed.disconnect()
        roi_data["dock"].sigClosed.connect(
            lambda _d, n=new_name: self._on_dock_closed(n))

        self._refresh_table()

    def _on_region_changing(self, name: str):
        if name not in self._rois:
            return
        lo, hi = self._rois[name]["region"].getRegion()
        tof_min_ns = self._display_to_ns(lo)
        tof_max_ns = self._display_to_ns(hi)
        self._rois[name]["tof_min_ns"] = tof_min_ns
        self._rois[name]["tof_max_ns"] = tof_max_ns
        self._refresh_table()
        self.roi_changed.emit(name, tof_min_ns, tof_max_ns)

    def _on_region_changed(self, name: str):
        if name not in self._rois:
            return
        lo, hi = self._rois[name]["region"].getRegion()
        tof_min_ns = self._display_to_ns(lo)
        tof_max_ns = self._display_to_ns(hi)
        self._rois[name]["tof_min_ns"] = tof_min_ns
        self._rois[name]["tof_max_ns"] = tof_max_ns
        self._refresh_table()
        self.roi_changed.emit(name, tof_min_ns, tof_max_ns)

    def _refresh_table(self):
        self._roi_table.blockSignals(True)
        self._roi_table.setRowCount(len(self._rois))

        editable = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
        readonly = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

        for row, (name, roi_data) in enumerate(self._rois.items()):
            color  = roi_data["color"]
            locked = roi_data.get("locked", False)

            self._roi_table.setItem(row, 0, self._make_color_swatch(color))

            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.ItemDataRole.UserRole, name)
            self._roi_table.setItem(row, 1, name_item)

            lo_display, hi_display = roi_data["region"].getRegion()
            for col, val in ((2, lo_display), (3, hi_display)):
                item = QTableWidgetItem(f"{val:.0f}")
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setFlags(readonly if locked else editable)
                self._roi_table.setItem(row, col, item)

            vis_item = QTableWidgetItem()
            vis_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            vis_item.setCheckState(
                Qt.CheckState.Checked if roi_data["dock"].isVisible() else Qt.CheckState.Unchecked)
            vis_item.setToolTip("Show / hide this ROI's image dock")
            self._roi_table.setItem(row, 4, vis_item)

            lock_item = QTableWidgetItem()
            lock_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            lock_item.setCheckState(Qt.CheckState.Checked if locked else Qt.CheckState.Unchecked)
            lock_item.setToolTip("Lock / unlock the ROI drag handles on the histogram")
            self._roi_table.setItem(row, 5, lock_item)

        self._roi_table.blockSignals(False)

    def _on_table_changed(self, row: int, col: int):
        if row >= self._roi_table.rowCount():
            return
        name_item = self._roi_table.item(row, 1)
        if name_item is None:
            return
        original_name = name_item.data(Qt.UserRole)
        if original_name not in self._rois:
            return

        if col == 1:
            new_name = name_item.text().strip()
            if new_name and new_name != original_name and new_name not in self._rois:
                self._rename_roi(original_name, new_name)
        elif col in (2, 3):
            if self._rois[original_name].get("locked", False):
                return
            try:
                lo_display = float(self._roi_table.item(row, 2).text())
                hi_display = float(self._roi_table.item(row, 3).text())
                if lo_display < hi_display:
                    tof_min_ns = self._display_to_ns(lo_display)
                    tof_max_ns = self._display_to_ns(hi_display)
                    self._rois[original_name]["tof_min_ns"] = tof_min_ns
                    self._rois[original_name]["tof_max_ns"] = tof_max_ns
                    region = self._rois[original_name]["region"]
                    region.blockSignals(True)
                    region.setRegion([lo_display, hi_display])
                    region.blockSignals(False)
                    self.roi_changed.emit(original_name, tof_min_ns, tof_max_ns)
            except (ValueError, AttributeError):
                pass
        elif col == 4:
            vis_item = self._roi_table.item(row, 4)
            if vis_item:
                self._rois[original_name]["dock"].setVisible(
                    vis_item.checkState() == Qt.CheckState.Checked)
        elif col == 5:
            lock_item = self._roi_table.item(row, 5)
            if lock_item:
                locked = lock_item.checkState() == Qt.CheckState.Checked
                self._rois[original_name]["locked"] = locked
                self._rois[original_name]["region"].setMovable(not locked)
                self._refresh_table()
