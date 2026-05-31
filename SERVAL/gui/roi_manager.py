"""
ROIManager — standalone QObject managing TOF ROI regions.

Receives references to the plot, table, display parameters, dock area, and the
parent window at construction time.  Connects ROI-table signals internally.
The caller (main frame) only needs to wire the three data-flow signals:

    roi_manager.roi_added.connect(lambda n,mn,mx: histogram.add_roi(n, mn, mx))
    roi_manager.roi_removed.connect(histogram.remove_roi)
    roi_manager.roi_changed.connect(histogram.update_roi)
"""

import pyqtgraph as pg
from qtpy.QtCore import Qt, QObject, Signal
from qtpy.QtWidgets import (
    QColorDialog,
    QInputDialog,
    QMenu,
    QMessageBox,
    QTableWidgetItem,
    QTableWidget,
)

from SERVAL.gui.image_dock_widget import ImageDockWidget, ROI_COLORS


class ROIManager(QObject):
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
        Used for dialog parents (QInputDialog, QMessageBox, etc.).
    dock_area : DockArea
        The pyqtgraph DockArea that owns all docks.
    total_dock : ImageDockWidget
        The "Total" dock; the first ROI dock is tabified next to it.
    """

    roi_added = Signal(str, float, float)    # name, min_ns, max_ns
    roi_removed = Signal(str)                # name
    roi_changed = Signal(str, float, float)  # name, min_ns, max_ns

    def __init__(self, tof_plot, roi_table: QTableWidget, display, main_window,
                 dock_area, total_dock, parent=None):
        super().__init__(parent)
        self._tof_plot = tof_plot
        self._roi_table = roi_table
        self._display = display
        self._main_window = main_window
        self._dockarea = dock_area
        self._total_dock = total_dock

        self._rois: dict = {}       # name -> {"region", "dock", "color", "locked"}
        self._roi_counter: int = 0

        # Wire table signals
        self._roi_table.cellChanged.connect(self._on_table_changed)
        self._roi_table.cellClicked.connect(self._on_table_cell_clicked)
        self._roi_table.doubleClicked.connect(self.raise_selected)
        self._roi_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._roi_table.customContextMenuRequested.connect(self._on_context_menu)

    # =========================================================================
    # Public API
    # =========================================================================

    def add_roi(self):
        """Prompt for a name and create a new TOF ROI."""
        name, ok = QInputDialog.getText(
            self._main_window, "Add ROI", "ROI Name:",
            text=f"ROI_{self._roi_counter + 1}"
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self._rois:
            QMessageBox.warning(self._main_window, "Duplicate",
                                f"ROI '{name}' already exists")
            return

        color = ROI_COLORS[self._roi_counter % len(ROI_COLORS)]
        self._roi_counter += 1

        tof_min_display = self._display.child('tof_min_ns').value()
        tof_max_display = self._display.child('tof_max_ns').value()
        span = tof_max_display - tof_min_display
        tof_min = tof_min_display + span * 0.2
        tof_max = tof_min_display + span * 0.4

        region = pg.LinearRegionItem(values=[tof_min, tof_max], brush=color, movable=True)
        region.sigRegionChanged.connect(
            lambda *_args, n=name: self._on_region_changing(n))
        region.sigRegionChangeFinished.connect(
            lambda *_args, n=name: self._on_region_changed(n))
        self._tof_plot.addItem(region)

        dock = ImageDockWidget(name, color=color)
        dock.sigClosed.connect(lambda _d, n=name: self._on_dock_closed(n))

        # Tab the new dock next to the last ROI dock (or total_dock if first)
        if self._rois:
            self._dockarea.addDock(dock, 'above', list(self._rois.values())[-1]["dock"])
        else:
            self._dockarea.addDock(dock, 'above', self._total_dock)

        self._rois[name] = {"region": region, "dock": dock,
                            "color": color, "locked": False}

        self._refresh_table()
        self.roi_added.emit(name, tof_min, tof_max)

    def remove_roi(self):
        """Prompt for a ROI name and remove it."""
        if not self._rois:
            return
        name, ok = QInputDialog.getItem(
            self._main_window, "Remove ROI", "Select ROI:",
            list(self._rois.keys()), 0, False
        )
        if ok:
            self._remove_roi_by_name(name)

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
        # self._tof_plot.setXRange(0, 1, padding=0.05)
        # self._tof_plot.setYRange(0, 1, padding=0.05)
        
    def update_displays(self, histogram, total_counts: int):
        """
        Refresh all ROI ImageDockWidgets from *histogram*.

        Call this once per GUI refresh tick, after the total image has been updated.
        """
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

    def set_timeseries_label(self, label: str):
        """Propagate display-mode label to all ROI docks."""
        for roi_data in self._rois.values():
            roi_data["dock"].set_timeseries_label(label)

    def set_colormap(self, name: str):
        """Apply a colourmap to all ROI docks."""
        for roi_data in self._rois.values():
            roi_data["dock"].set_colormap(name)

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _selected_name(self) -> str | None:
        """Return the ROI name for the currently selected table row, or None."""
        selected = self._roi_table.selectedItems()
        if not selected:
            return None
        name_item = self._roi_table.item(selected[0].row(), 1)
        if name_item is None:
            return None
        name = name_item.data(Qt.UserRole)
        return name if name in self._rois else None

    def _remove_roi_by_name(self, name: str):
        if name not in self._rois:
            return
        roi_data = self._rois.pop(name)
        self._tof_plot.removeItem(roi_data["region"])
        # Pop from dict first so sigClosed callback (if triggered by close()) is a no-op
        roi_data["dock"].close()
        roi_data["dock"].deleteLater()
        self._refresh_table()
        self.roi_removed.emit(name)

    def _on_dock_closed(self, name: str):
        """Called when a ROI dock's close button (X) is clicked by the user."""
        if name not in self._rois:
            return  # Already removed programmatically
        roi_data = self._rois.pop(name)
        self._tof_plot.removeItem(roi_data["region"])
        roi_data["dock"].deleteLater()
        self._refresh_table()
        self.roi_removed.emit(name)

    def _rename_roi(self, old_name: str, new_name: str):
        if old_name not in self._rois:
            return
        roi_data = self._rois.pop(old_name)
        self._rois[new_name] = roi_data

        roi_data["dock"].setTitle(new_name)

        tof_min, tof_max = roi_data["region"].getRegion()
        # Update histogram via remove+add
        self.roi_removed.emit(old_name)
        self.roi_added.emit(new_name, tof_min, tof_max)

        roi_data["region"].sigRegionChanged.disconnect()
        roi_data["region"].sigRegionChangeFinished.disconnect()
        roi_data["region"].sigRegionChanged.connect(
            lambda *_: self._on_region_changing(new_name))
        roi_data["region"].sigRegionChangeFinished.connect(
            lambda *_: self._on_region_changed(new_name))

        # Reconnect sigClosed with updated name
        roi_data["dock"].sigClosed.disconnect()
        roi_data["dock"].sigClosed.connect(
            lambda _d, n=new_name: self._on_dock_closed(n))

        self._refresh_table()

    # ── Region signals ────────────────────────────────────────────────────────

    def _on_region_changing(self, name: str):
        if name not in self._rois:
            return
        tof_min, tof_max = self._rois[name]["region"].getRegion()
        self._refresh_table()
        self.roi_changed.emit(name, tof_min, tof_max)

    def _on_region_changed(self, name: str):
        if name not in self._rois:
            return
        tof_min, tof_max = self._rois[name]["region"].getRegion()
        self._refresh_table()
        self.roi_changed.emit(name, tof_min, tof_max)

    # ── Table management ──────────────────────────────────────────────────────

    def _refresh_table(self):
        """Rebuild the ROI table from current state."""
        self._roi_table.blockSignals(True)
        self._roi_table.setRowCount(len(self._rois))

        for row, (name, roi_data) in enumerate(self._rois.items()):
            color = roi_data["color"]

            color_item = QTableWidgetItem("")
            color_item.setBackground(pg.mkColor(color[0], color[1], color[2], 200))
            color_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._roi_table.setItem(row, 0, color_item)

            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.ItemDataRole.UserRole, name)
            self._roi_table.setItem(row, 1, name_item)

            tof_min, tof_max = roi_data["region"].getRegion()

            min_item = QTableWidgetItem(f"{tof_min:.0f}")
            min_item.setData(Qt.ItemDataRole.UserRole, name)
            self._roi_table.setItem(row, 2, min_item)

            max_item = QTableWidgetItem(f"{tof_max:.0f}")
            max_item.setData(Qt.ItemDataRole.UserRole, name)
            self._roi_table.setItem(row, 3, max_item)

            vis_item = QTableWidgetItem()
            vis_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            vis_item.setCheckState(
                Qt.CheckState.Checked if roi_data["dock"].isVisible() else Qt.CheckState.Unchecked)
            vis_item.setToolTip("Show / hide this ROI's image dock")
            self._roi_table.setItem(row, 4, vis_item)

            locked = roi_data.get("locked", False)
            lock_item = QTableWidgetItem()
            lock_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            lock_item.setCheckState(Qt.CheckState.Checked if locked else Qt.CheckState.Unchecked)
            lock_item.setToolTip("Lock / unlock the ROI drag handles on the histogram")
            self._roi_table.setItem(row, 5, lock_item)

            editable = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
            readonly = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            for col in (2, 3):
                item = self._roi_table.item(row, col)
                if item:
                    item.setFlags(readonly if locked else editable)

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
                tof_min = float(self._roi_table.item(row, 2).text())
                tof_max = float(self._roi_table.item(row, 3).text())
                if tof_min < tof_max:
                    region = self._rois[original_name]["region"]
                    region.blockSignals(True)
                    region.setRegion([tof_min, tof_max])
                    region.blockSignals(False)
                    self.roi_changed.emit(original_name, tof_min, tof_max)
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

    def _on_table_cell_clicked(self, row: int, col: int):
        """Open colour picker when the colour swatch cell (col 0) is clicked."""
        if col != 0:
            return
        name_item = self._roi_table.item(row, 1)
        if name_item is None:
            return
        name = name_item.data(Qt.ItemDataRole.UserRole)
        if name in self._rois:
            self._pick_color(name)

    def _pick_color(self, name: str):
        old_color = self._rois[name]["color"]
        initial = pg.mkColor(*old_color[:3])
        color = QColorDialog.getColor(initial, self._main_window,
                                      f"Pick colour for {name}")
        if not color.isValid():
            return
        new_color = (color.red(), color.green(), color.blue(), 80)
        roi_data = self._rois[name]
        roi_data["color"] = new_color
        roi_data["region"].setBrush(pg.mkBrush(*new_color))
        roi_data["dock"].update_color_indicator(new_color)
        r, g, b, _a = new_color
        roi_data["dock"].timeseries_curve.setPen(pg.mkPen(color=(r, g, b), width=2))
        self._refresh_table()

    def _on_context_menu(self, pos):
        item = self._roi_table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        name_item = self._roi_table.item(row, 1)
        if name_item is None:
            return
        name = name_item.data(Qt.ItemDataRole.UserRole)
        if name not in self._rois:
            return

        roi_data = self._rois[name]
        locked = roi_data.get("locked", False)

        menu = QMenu(self._main_window)
        raise_act  = menu.addAction("Raise dock")
        zoom_act   = menu.addAction("Zoom to ROI")
        zoom_out_act = menu.addAction("Zoom out")
        menu.addSeparator()
        color_act  = menu.addAction("Pick colour…")
        menu.addSeparator()
        lock_act   = menu.addAction("Unlock range" if locked else "Lock range")
        menu.addSeparator()
        rename_act = menu.addAction("Rename…")
        remove_act = menu.addAction("Remove")

        action = menu.exec_(self._roi_table.viewport().mapToGlobal(pos))
        if action is None:
            return

        if action == raise_act:
            self._roi_table.selectRow(row)
            self.raise_selected()
        elif action == zoom_act:
            self._roi_table.selectRow(row)
            self.zoom_to_selected()
        elif action == zoom_out_act:
            self.zoom_out()
        elif action == color_act:
            self._pick_color(name)
        elif action == lock_act:
            roi_data["locked"] = not locked
            roi_data["region"].setMovable(locked)   # locked → not movable
            self._refresh_table()
        elif action == rename_act:
            new_name, ok = QInputDialog.getText(
                self._main_window, "Rename ROI", "New name:", text=name)
            if ok and new_name.strip() and new_name.strip() != name:
                if new_name.strip() in self._rois:
                    QMessageBox.warning(self._main_window, "Duplicate",
                                        f"ROI '{new_name.strip()}' already exists")
                else:
                    self._rename_roi(name, new_name.strip())
        elif action == remove_act:
            self._remove_roi_by_name(name)
