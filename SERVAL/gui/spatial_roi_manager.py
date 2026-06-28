"""
SpatialROIManager — manages rect/ellipse image-space ROIs on an ImageDockWidget.

Each ROI has:
  shape  : "rect" | "ellipse"
  op     : "+" include | "-" exclude  (used for the combined mask)

The combined mask for a group of ROIs is  union(+) AND NOT union(-),
which allows constructions like an annulus (large circle ⊕ minus small circle ⊖).

Signal signatures
-----------------
roi_added   (parent, name, shape, op, x, y, w, h)
roi_removed (parent, name)
roi_changed (parent, name, shape, op, x, y, w, h)
"""

import pyqtgraph as pg
from qtpy.QtCore import QObject, QRectF, Qt, Signal
from qtpy.QtWidgets import QMenu, QTableWidget, QTableWidgetItem

from SERVAL.gui.base_roi_manager import BaseROIManager
from SERVAL.gui.image_dock_widget import ROI_COLORS

_DEFAULT_RECT = (64, 64, 64, 64)
_IMAGE_SIZE   = 256

_SHAPE_GLYPH = {"rect": "▭", "ellipse": "○"}
_OP_GLYPH    = {"+": "⊕", "-": "⊖"}


class SpatialROIManager(BaseROIManager):
    """
    Manages rectangular/elliptical spatial ROIs drawn on one ImageDockWidget.

    Parameters
    ----------
    parent_key : str
        "" for the main pixel histogram, or a TOF-ROI name for its histogram.
    image_dock : ImageDockWidget
    main_window : QMainWindow  (used as parent for dialogs / menus)
    """

    roi_added   = Signal(str, str, str, str, int, int, int, int)  # parent,name,shape,op,x,y,w,h
    roi_removed = Signal(str, str)
    roi_changed = Signal(str, str, str, str, int, int, int, int)

    def __init__(self, parent_key: str, image_dock, main_window, parent=None):
        super().__init__(parent)
        self._parent_key  = parent_key
        self._image_dock  = image_dock
        self._main_window = main_window

        self._rois: dict = {}   # name -> {"roi", "shape", "op", "color", "locked"}
        self._counter = 0

        image_dock.add_roi_clicked.connect(self.add_roi)
        image_dock.remove_roi_clicked.connect(self.remove_roi)

        table = image_dock.roi_table
        table.cellChanged.connect(self._on_table_changed)
        table.cellClicked.connect(self._on_table_cell_clicked)
        table.setContextMenuPolicy(Qt.CustomContextMenu)
        table.customContextMenuRequested.connect(self._on_context_menu)

    # ── BaseROIManager contract ───────────────────────────────────────────────

    @property
    def _table(self) -> QTableWidget:
        return self._image_dock.roi_table

    def _pg_item(self, name: str):
        return self._rois[name]["roi"]

    def _apply_color(self, name: str, color: tuple):
        roi_data = self._rois[name]
        roi_data["roi"].setPen(pg.mkPen(color=color[:3], width=2))
        self._image_dock.remove_roi_curve(name)
        self._image_dock.add_roi_curve(name, color)

    # ── Shape/op cell clicks (extend base handler) ────────────────────────────

    def _on_table_cell_clicked(self, row: int, col: int):
        if col == 0:
            # Color picker (inherited logic)
            super()._on_table_cell_clicked(row, col)
            return
        name_item = self._table.item(row, 1)
        if name_item is None:
            return
        name = name_item.data(Qt.ItemDataRole.UserRole)
        if name not in self._rois:
            return

        if col == 2:
            # Cycle shape: rect → ellipse → rect
            current = self._rois[name]["shape"]
            new_shape = "ellipse" if current == "rect" else "rect"
            self._change_shape(name, new_shape)
        elif col == 3:
            # Toggle op: + ↔ -
            current_op = self._rois[name]["op"]
            new_op = "-" if current_op == "+" else "+"
            self._set_op(name, new_op)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_roi(self, shape: str = "rect"):
        name = f"Spatial_{self._counter + 1}"
        while name in self._rois:
            self._counter += 1
            name = f"Spatial_{self._counter + 1}"

        color = ROI_COLORS[self._counter % len(ROI_COLORS)]
        self._counter += 1

        x, y, w, h = _DEFAULT_RECT
        roi = self._create_pg_roi(name, shape, x, y, w, h, color)
        self._image_dock.add_roi_curve(name, color)
        self._rois[name] = {
            "roi": roi, "shape": shape, "op": "+",
            "color": color, "locked": False,
        }

        self._refresh_table()
        # Ensure the combined curve exists
        self._image_dock.ensure_combined_curve()
        self.roi_added.emit(self._parent_key, name, shape, "+", x, y, w, h)

        # Start the name cell in edit mode
        table = self._image_dock.roi_table
        row = len(self._rois) - 1
        name_item = table.item(row, 1)
        if name_item is not None:
            table.setCurrentItem(name_item)
            table.editItem(name_item)

    def update_displays(self, histogram, total_counts: int):
        """Refresh counts/yield/curve for every ROI and the combined mask."""
        table = self._image_dock.roi_table
        for row, (name, roi_data) in enumerate(self._rois.items()):
            counts = histogram.get_spatial_roi_counts(self._parent_key, name)
            pct    = (counts / total_counts * 100) if total_counts > 0 else None

            counts_item = table.item(row, 8)
            if counts_item is not None:
                counts_item.setText(f"{counts:,}")
            yield_item = table.item(row, 9)
            if yield_item is not None:
                yield_item.setText(f"{pct:.1f}" if pct is not None else "—")

            times, rates = histogram.get_spatial_roi_timeseries(self._parent_key, name)
            self._image_dock.update_roi_curve(name, times, rates)

        # Combined mask
        times_c, rates_c = histogram.get_combined_timeseries(self._parent_key)
        self._image_dock.update_combined_curve(times_c, rates_c)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def set_parent_key(self, new_key: str):
        self._parent_key = new_key

    def get_roi_names(self) -> list[str]:
        return list(self._rois.keys())

    def get_roi_color(self, name: str):
        roi_data = self._rois.get(name)
        return roi_data["color"] if roi_data is not None else None

    def get_roi_geometries(self):
        out = []
        for name, roi_data in self._rois.items():
            x, y = roi_data["roi"].pos()
            w, h = roi_data["roi"].size()
            out.append((name, roi_data["shape"], roi_data["op"],
                        int(x), int(y), int(w), int(h)))
        return out

    def _create_pg_roi(self, name, shape, x, y, w, h, color):
        """Create a RectROI or EllipseROI and wire its signals."""
        cls = pg.EllipseROI if shape == "ellipse" else pg.RectROI
        roi = cls(
            [x, y], [w, h],
            pen=pg.mkPen(color=color[:3], width=2),
            removable=True,
            maxBounds=QRectF(0, 0, _IMAGE_SIZE, _IMAGE_SIZE),
        )
        roi.sigRegionChangeFinished.connect(
            lambda *_a, n=name: self._on_region_changed(n))
        roi.sigRemoveRequested.connect(
            lambda *_a, n=name: self._remove_roi_by_name(n))
        self._image_dock.plot.addItem(roi)
        return roi

    def _change_shape(self, name: str, new_shape: str):
        roi_data = self._rois.get(name)
        if roi_data is None or roi_data["shape"] == new_shape:
            return
        old_roi = roi_data["roi"]
        x, y = old_roi.pos()
        w, h = old_roi.size()
        # Disconnect old signals before removing to avoid stale callbacks
        try:
            old_roi.sigRegionChangeFinished.disconnect()
            old_roi.sigRemoveRequested.disconnect()
        except Exception:
            pass
        self._image_dock.plot.removeItem(old_roi)
        new_roi = self._create_pg_roi(name, new_shape, int(x), int(y), int(w), int(h),
                                      roi_data["color"])
        roi_data["roi"]   = new_roi
        roi_data["shape"] = new_shape
        self._refresh_table()
        self.roi_changed.emit(
            self._parent_key, name, new_shape, roi_data["op"],
            int(x), int(y), int(w), int(h),
        )

    def _set_op(self, name: str, new_op: str):
        roi_data = self._rois.get(name)
        if roi_data is None:
            return
        roi_data["op"] = new_op
        # Draw exclude ROIs with a dashed pen to signal their role visually
        style = (pg.QtCore.Qt.PenStyle.DashLine if new_op == "-"
                 else pg.QtCore.Qt.PenStyle.SolidLine)
        roi_data["roi"].setPen(
            pg.mkPen(color=roi_data["color"][:3], width=2, style=style))
        self._refresh_table()
        x, y = roi_data["roi"].pos()
        w, h = roi_data["roi"].size()
        self.roi_changed.emit(
            self._parent_key, name, roi_data["shape"], new_op,
            int(x), int(y), int(w), int(h),
        )

    def _remove_roi_by_name(self, name: str):
        roi_data = self._rois.pop(name, None)
        if roi_data is None:
            return
        self._image_dock.plot.removeItem(roi_data["roi"])
        self._image_dock.remove_roi_curve(name)
        if not self._rois:
            self._image_dock.remove_combined_curve()
        self._refresh_table()
        self.roi_removed.emit(self._parent_key, name)

    def _rename_roi(self, old_name: str, new_name: str):
        if old_name not in self._rois:
            return
        roi_data = self._rois.pop(old_name)
        self._rois[new_name] = roi_data

        x, y = roi_data["roi"].pos()
        w, h = roi_data["roi"].size()
        self._image_dock.rename_roi_curve(old_name, new_name, roi_data["color"])

        self.roi_removed.emit(self._parent_key, old_name)
        self.roi_added.emit(
            self._parent_key, new_name,
            roi_data["shape"], roi_data["op"],
            int(x), int(y), int(w), int(h),
        )

        roi_data["roi"].sigRegionChangeFinished.disconnect()
        roi_data["roi"].sigRegionChangeFinished.connect(
            lambda *_: self._on_region_changed(new_name))
        roi_data["roi"].sigRemoveRequested.disconnect()
        roi_data["roi"].sigRemoveRequested.connect(
            lambda *_, n=new_name: self._remove_roi_by_name(n))

        self._refresh_table()

    def _on_region_changed(self, name: str):
        roi_data = self._rois.get(name)
        if roi_data is None:
            return
        x, y = roi_data["roi"].pos()
        w, h = roi_data["roi"].size()
        self._refresh_table()
        self.roi_changed.emit(
            self._parent_key, name,
            roi_data["shape"], roi_data["op"],
            int(x), int(y), int(w), int(h),
        )

    def _refresh_table(self):
        table = self._image_dock.roi_table
        table.blockSignals(True)
        table.setRowCount(len(self._rois))

        editable = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
        readonly = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        clickable = Qt.ItemFlag.ItemIsEnabled   # col 2/3 — no selection highlight needed

        for row, (name, roi_data) in enumerate(self._rois.items()):
            color  = roi_data["color"]
            locked = roi_data.get("locked", False)
            shape  = roi_data["shape"]
            op     = roi_data["op"]

            # col 0 — colour swatch
            table.setItem(row, 0, self._make_color_swatch(color))

            # col 1 — name (editable)
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.ItemDataRole.UserRole, name)
            table.setItem(row, 1, name_item)

            # col 2 — shape glyph (click cycles shape)
            shape_item = QTableWidgetItem(_SHAPE_GLYPH.get(shape, shape))
            shape_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            shape_item.setFlags(clickable)
            shape_item.setToolTip("Click to toggle shape (▭ rect / ○ ellipse)")
            table.setItem(row, 2, shape_item)

            # col 3 — op glyph (click toggles include/exclude)
            op_item = QTableWidgetItem(_OP_GLYPH.get(op, op))
            op_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            op_item.setFlags(clickable)
            op_item.setToolTip(
                "⊕ include — ⊖ exclude (from combined mask)\n"
                "Combined = union(⊕) AND NOT union(⊖)")
            if op == "-":
                op_item.setForeground(pg.mkColor(255, 80, 80))
            table.setItem(row, 3, op_item)

            # cols 4-7 — geometry
            x, y = roi_data["roi"].pos()
            w, h = roi_data["roi"].size()
            for col, val in ((4, x), (5, y), (6, w), (7, h)):
                item = QTableWidgetItem(f"{val:.0f}")
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setFlags(readonly if locked else editable)
                table.setItem(row, col, item)

            # col 8 — counts (readonly)
            counts_item = QTableWidgetItem("0")
            counts_item.setFlags(readonly)
            table.setItem(row, 8, counts_item)

            # col 9 — yield (readonly)
            yield_item = QTableWidgetItem("—")
            yield_item.setFlags(readonly)
            table.setItem(row, 9, yield_item)

            # col 10 — visibility checkbox
            vis_item = QTableWidgetItem()
            vis_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            vis_item.setCheckState(
                Qt.CheckState.Checked if roi_data["roi"].isVisible()
                else Qt.CheckState.Unchecked)
            vis_item.setToolTip("Show / hide this ROI's rectangle and curve")
            table.setItem(row, 10, vis_item)

            # col 11 — lock checkbox
            lock_item = QTableWidgetItem()
            lock_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            lock_item.setCheckState(Qt.CheckState.Checked if locked else Qt.CheckState.Unchecked)
            lock_item.setToolTip("Lock / unlock the ROI drag handles")
            table.setItem(row, 11, lock_item)

        table.blockSignals(False)

    def _on_table_changed(self, row: int, col: int):
        table = self._image_dock.roi_table
        if row >= table.rowCount():
            return
        name_item = table.item(row, 1)
        if name_item is None:
            return
        original_name = name_item.data(Qt.UserRole)
        if original_name not in self._rois:
            return

        if col == 1:
            new_name = name_item.text().strip()
            if new_name and new_name != original_name and new_name not in self._rois:
                self._rename_roi(original_name, new_name)

        elif col in (4, 5, 6, 7):
            if self._rois[original_name].get("locked", False):
                return
            try:
                x = float(table.item(row, 4).text())
                y = float(table.item(row, 5).text())
                w = float(table.item(row, 6).text())
                h = float(table.item(row, 7).text())
                if w > 0 and h > 0:
                    roi = self._rois[original_name]["roi"]
                    roi.blockSignals(True)
                    roi.setPos([x, y])
                    roi.setSize([w, h])
                    roi.blockSignals(False)
                    rd = self._rois[original_name]
                    self.roi_changed.emit(
                        self._parent_key, original_name,
                        rd["shape"], rd["op"],
                        int(x), int(y), int(w), int(h),
                    )
            except (ValueError, AttributeError):
                pass

        elif col == 10:
            vis_item = table.item(row, 10)
            if vis_item:
                visible = vis_item.checkState() == Qt.CheckState.Checked
                self._rois[original_name]["roi"].setVisible(visible)
                self._image_dock.set_roi_curve_visible(original_name, visible)

        elif col == 11:
            lock_item = table.item(row, 11)
            if lock_item:
                locked = lock_item.checkState() == Qt.CheckState.Checked
                self._rois[original_name]["locked"] = locked
                self._rois[original_name]["roi"].setMovable(not locked)
                self._refresh_table()

    # ── Extra context menu items (shape submenu) ──────────────────────────────

    def _add_extra_menu_actions(self, menu: QMenu, name: str, row: int, acts: dict):
        shape_menu = menu.addMenu("Shape")
        rect_act    = shape_menu.addAction("▭  Rectangle")
        ellipse_act = shape_menu.addAction("○  Ellipse")
        current = self._rois[name]["shape"]
        rect_act.setEnabled(current != "rect")
        ellipse_act.setEnabled(current != "ellipse")
        acts["shape_rect"]    = rect_act
        acts["shape_ellipse"] = ellipse_act

        op_menu  = menu.addMenu("Operation")
        inc_act  = op_menu.addAction("⊕  Include")
        exc_act  = op_menu.addAction("⊖  Exclude")
        current_op = self._rois[name]["op"]
        inc_act.setEnabled(current_op != "+")
        exc_act.setEnabled(current_op != "-")
        acts["op_include"] = inc_act
        acts["op_exclude"] = exc_act

    def _handle_extra_menu_action(self, action, name: str, row: int, acts: dict) -> bool:
        if action == acts.get("shape_rect"):
            self._change_shape(name, "rect")
            return True
        if action == acts.get("shape_ellipse"):
            self._change_shape(name, "ellipse")
            return True
        if action == acts.get("op_include"):
            self._set_op(name, "+")
            return True
        if action == acts.get("op_exclude"):
            self._set_op(name, "-")
            return True
        return False
