"""
_SessionMixin — QSettings persistence for ServalAcquisitionGUI.

Mixed into ServalAcquisitionGUI; all methods access GUI state via ``self``.
"""

import json

from qtpy.QtCore import Qt, QSettings


class _SessionMixin:
    """Saves and restores window geometry, dock layout, and parameter values."""

    # Parameter types whose value is a persistable scalar config setting
    # (as opposed to 'group'/'action_led' container nodes or transient
    # action/status indicators).
    _CONFIG_LEAF_TYPES = ('str', 'int', 'float', 'bool', 'list')

    # ── Iterators ─────────────────────────────────────────────────────────────

    def _iter_config_params(self, parent=None):
        """Recursively yield every leaf Parameter under *parent* (default
        ``self.settings``) representing a user-configurable setting."""
        if parent is None:
            parent = self.settings
        for child in parent.children():
            if child.hasChildren():
                yield from self._iter_config_params(child)
            elif child.type() in self._CONFIG_LEAF_TYPES:
                yield child

    # ── QSettings handle ──────────────────────────────────────────────────────

    def _qsettings(self) -> QSettings:
        return QSettings('SERVAL', 'AcquisitionGUI')

    # ── Save / restore ────────────────────────────────────────────────────────

    def _restore_session(self):
        s = self._qsettings()
        if geom := s.value('geometry'):
            self.restoreGeometry(geom)
        if state := s.value('dockarea_state'):
            try:
                self.dock_area.restoreState(state)
                # Sanity-check: if all main docks ended up hidden, the saved
                # state is stale — drop it and let the default layout stand.
                core_docks = [self._settings_dock, self.tof_dock, self.total_dock]
                if all(not d.isVisible() for d in core_docks):
                    self.dock_area.restoreState(None)
                    s.remove('dockarea_state')
            except Exception:
                s.remove('dockarea_state')  # Discard incompatible saved state
        # BinSpec widgets — restore before display params so the histogram is
        # already sized correctly when mass-calib params fire their signals.
        for key, widget in [('tof_bin_spec', self.tof_dock.tof_bin_spec_widget),
                            ('cov_bin_spec', self._cov_dock.cov_bin_spec_widget)]:
            if state_str := s.value(key):
                try:
                    widget.set_state(json.loads(state_str))
                except Exception:
                    pass
        # Acquisition / pipeline configuration (SERVAL connection, saving
        # options, processing parameters, ...)
        for child in self._iter_config_params():
            key = 'config/' + '/'.join(self.settings.childPath(child))
            if s.contains(key):
                try:
                    child.setValue(s.value(key, child.value(), type(child.value())))
                except Exception:
                    pass
        # Toolbar style — applied after all docks are built so every toolbar is covered
        if (saved := s.value('toolbar_style')) is not None:
            restored = Qt.ToolButtonStyle(int(saved))
            self._apply_toolbar_style(restored)
            for act, (_lbl, style) in zip(
                    self._toolbar_style_group.actions(), self._TOOLBAR_STYLES):
                act.setChecked(style == restored)

    def _save_session(self):
        s = self._qsettings()
        s.setValue('geometry', self.saveGeometry())
        s.setValue('dockarea_state', self.dock_area.saveState())
        for key, widget in [('tof_bin_spec', self.tof_dock.tof_bin_spec_widget),
                            ('cov_bin_spec', self._cov_dock.cov_bin_spec_widget)]:
            s.setValue(key, json.dumps(widget.get_state()))
        for child in self._iter_config_params():
            key = 'config/' + '/'.join(self.settings.childPath(child))
            s.setValue(key, child.value())
