"""
Build PyServal's extra Material Icons resource file.

Run from the PyServal root:
    python -m SERVAL.resources.build_icons

This reads SERVAL/resources/icons.toml, fetches the SVGs from Google Fonts
via qt_material_icons, and generates:
    SERVAL/resources/material_icons/resources/icons_rounded_40.py

That file is imported at GUI startup to register the icons in Qt's resource
system, making them available to pymodaq_gui.utils.styling.create_icon().

Only run this when icons.toml changes — the generated file is committed to git.
"""

import shutil
import sys
import tomllib
from pathlib import Path

resource_folder = Path(__file__).parent

# qt_material_icons creates a sub-package with the same name in the output dir;
# remove it first to avoid import conflicts (mirrors pymodaq's check_icons_dev.py).
wrong_path = resource_folder / 'qt_material_icons'
shutil.rmtree(wrong_path, ignore_errors=True)

try:
    from qt_material_icons import extract
except ImportError:
    sys.exit(
        "qt_material_icons is not installed.\n"
        "Install it with:  pip install qt-material-icons"
    )

with open(resource_folder / 'icons.toml', 'rb') as f:
    cfg = tomllib.load(f)

names  = cfg['icons']['names']
styles = tuple(extract.MaterialIcon.Style(s) for s in cfg['icons']['style'])
sizes  = cfg['icons']['size']

print(f"Fetching {len(names)} icons  styles={[s.value for s in styles]}  sizes={sizes}")

extract.extract_package(output=str(resource_folder))
extract.extract_icons_multi(names=names, styles=styles, sizes=sizes,
                            output=str(resource_folder))

good_path = resource_folder / 'material_icons'
shutil.rmtree(good_path, ignore_errors=True)
wrong_path.rename(good_path)

print(f"Done — resource file written to {good_path}/resources/")
