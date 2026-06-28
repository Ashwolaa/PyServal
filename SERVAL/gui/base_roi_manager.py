"""
BaseROIManager — shared table interaction logic for ROIManager and SpatialROIManager.

Both managers drive a QTableWidget whose first column is a colour swatch and
second column is the ROI name.  The ROI type differs (LinearRegionItem vs
RectROI), but the table/colour/selection/context-menu behaviour is identical.

Subclasses implement the four type-specific hooks:
    _table          — the QTableWidget this manager drives (property)
    _pg_item(name)  — returns the pyqtgraph item for a named ROI
    _apply_color(name, color) — push a new colour to the ROI's visuals
    _remove_roi_by_name(name) — full teardown for one ROI
    _refresh_table()          — rebuild table rows from _rois

Optionally, subclasses can inject extra entries at the top of the right-click
context menu by overriding _add_extra_menu_actions / _handle_extra_menu_action.
"""

import pyqtgraph as pg
from qtpy.QtCore import QObject, Qt
from qtpy.QtWidgets import QColorDialog, QMenu, QTableWidget, QTableWidgetItem


class BaseROIManager(QObject):

    # ── Subclass contract ─────────────────────────────────────────────────────

    @property
    def _table(self) -> QTableWidget:
        raise NotImplementedError

    def _pg_item(self, name: str):
        """Return the pyqtgraph item (LinearRegionItem or RectROI) for *name*."""
        raise NotImplementedError

    def _apply_color(self, name: str, color: tuple):
        """Push *color* to all visual elements of the named ROI."""
        raise NotImplementedError

    def _remove_roi_by_name(self, name: str):
        raise NotImplementedError

    def _refresh_table(self):
        raise NotImplementedError

    # ── Shared API ────────────────────────────────────────────────────────────

    def remove_roi(self):
        """Remove the currently selected ROI (no dialog)."""
        name = self._selected_name()
        if name is not None:
            self._remove_roi_by_name(name)

    # ── Shared internals ──────────────────────────────────────────────────────

    def _selected_name(self) -> str | None:
        """Return the ROI name for the currently selected table row, or None."""
        selected = self._table.selectedItems()
        if not selected:
            return None
        name_item = self._table.item(selected[0].row(), 1)
        if name_item is None:
            return None
        name = name_item.data(Qt.UserRole)
        return name if name in self._rois else None

    def _on_table_cell_clicked(self, row: int, col: int):
        """Open colour picker when the colour swatch cell (col 0) is clicked."""
        if col != 0:
            return
        name_item = self._table.item(row, 1)
        if name_item is None:
            return
        name = name_item.data(Qt.ItemDataRole.UserRole)
        if name in self._rois:
            self._pick_color(name)

    def _pick_color(self, name: str):
        old_color = self._rois[name]["color"]
        color = QColorDialog.getColor(
            pg.mkColor(*old_color[:3]), self._table, f"Pick colour for {name}")
        self._table.setFocus()
        if not color.isValid():
            return
        new_color = (color.red(), color.green(), color.blue(), 80)
        self._rois[name]["color"] = new_color
        self._apply_color(name, new_color)
        self._refresh_table()

    def _on_context_menu(self, pos):
        item = self._table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        name_item = self._table.item(row, 1)
        if name_item is None:
            return
        name = name_item.data(Qt.ItemDataRole.UserRole)
        if name not in self._rois:
            return

        locked = self._rois[name].get("locked", False)
        menu = QMenu(self._table)

        extra_acts: dict = {}
        self._add_extra_menu_actions(menu, name, row, extra_acts)
        if extra_acts:
            menu.addSeparator()

        color_act = menu.addAction("Pick colour…")
        menu.addSeparator()
        lock_act  = menu.addAction("Unlock" if locked else "Lock")
        menu.addSeparator()
        remove_act = menu.addAction("Remove")

        action = menu.exec_(self._table.viewport().mapToGlobal(pos))
        if action is None:
            return

        if self._handle_extra_menu_action(action, name, row, extra_acts):
            return
        if action == color_act:
            self._pick_color(name)
        elif action == lock_act:
            self._rois[name]["locked"] = not locked
            self._pg_item(name).setMovable(locked)   # old value: was unlocked → lock it
            self._refresh_table()
        elif action == remove_act:
            self._remove_roi_by_name(name)

    def _add_extra_menu_actions(self, menu: QMenu, name: str, row: int, acts: dict):
        """Override to prepend type-specific items at the top of the context menu."""

    def _handle_extra_menu_action(self, action, name: str, row: int, acts: dict) -> bool:
        """Override to handle type-specific items. Return True if the action was consumed."""
        return False

    # ── Table utility ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_color_swatch(color: tuple) -> QTableWidgetItem:
        item = QTableWidgetItem("")
        item.setBackground(pg.mkColor(color[0], color[1], color[2], 200))
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        return item
