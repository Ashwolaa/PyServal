#!/usr/bin/env python3
"""
Copy TPX3 acquisition data into an existing PyMoDAQ HDF5 dataset.

PyMoDAQ writes one Dataset_*.h5 per session with a group per scan node (e.g.
"Scan001"). The TPX3 plugin (daq_0Dviewer_Tpx3Serval) writes its own flat
folder of *_events.dat / *_pixels.dat / *.tpx3 files named by step index next
to (not inside) that h5. This module copies the correlated event/pixel arrays
for a given scan node into the matching h5 group after acquisition, so
everything is addressable from one file.

Requires h5py (not a core SERVAL dependency — install separately if needed).

CLI usage:
    python -m SERVAL.pymodaq_plugin.h5_export <h5_path> <scan_node_name> <data_dir>
"""

from pathlib import Path

import h5py
import numpy as np

from SERVAL.core.data_types import EVENT_DTYPE, PIXEL_DTYPE


def _find_scan_group(h5file: h5py.File, scan_node_name: str) -> h5py.Group:
    """Find the (first) group in the h5 tree named exactly `scan_node_name`."""
    match = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name.rsplit("/", 1)[-1] == scan_node_name:
            match.setdefault("group", obj)

    h5file.visititems(visitor)
    if "group" not in match:
        raise KeyError(
            f"No group named {scan_node_name!r} found in {h5file.filename}. "
            "Check that the scan actually ran under this name."
        )
    return match["group"]


def export_scan_to_h5(
    h5_path,
    scan_node_name: str,
    data_dir,
    include_pixels: bool = False,
    overwrite: bool = False,
) -> int:
    """
    Copy correlated TPX3 events (and optionally pixels) from `data_dir` into
    the PyMoDAQ h5 file's scan-node group, one subgroup per acquisition step.

    Parameters
    ----------
    h5_path : str or Path
        Path to the PyMoDAQ Dataset_*.h5 file.
    scan_node_name : str
        Name of the scan node group to attach data under (e.g. "Scan001") —
        must match the folder name the plugin used during acquisition.
    data_dir : str or Path
        Folder containing the plugin's *_events.dat (and *_pixels.dat) files
        for that scan node (flat layout: one file per step, no subfolders).
    include_pixels : bool
        Also copy *_pixels.dat files if present.
    overwrite : bool
        Replace existing step subgroups instead of skipping them.

    Returns
    -------
    int
        Number of steps copied.
    """
    data_dir = Path(data_dir)
    event_files = sorted(data_dir.glob("*_events.dat"))
    if not event_files:
        raise FileNotFoundError(f"No *_events.dat files found in {data_dir}")

    count = 0
    with h5py.File(h5_path, "a") as h5:
        scan_group = _find_scan_group(h5, scan_node_name)
        tpx3_group = scan_group.require_group("tpx3")
        tpx3_group.attrs["source_dir"] = str(data_dir.resolve())

        for ev_file in event_files:
            step = ev_file.stem.replace("_events", "")
            if step in tpx3_group:
                if not overwrite:
                    continue
                del tpx3_group[step]
            step_group = tpx3_group.create_group(step)

            events = np.fromfile(ev_file, dtype=EVENT_DTYPE)
            step_group.create_dataset("events", data=events, compression="gzip")

            if include_pixels:
                px_file = data_dir / f"{step}_pixels.dat"
                if px_file.exists():
                    pixels = np.fromfile(px_file, dtype=PIXEL_DTYPE)
                    step_group.create_dataset("pixels", data=pixels, compression="gzip")

            raw_file = data_dir / f"{step}.tpx3"
            if raw_file.exists():
                step_group.attrs["raw_file"] = str(raw_file.resolve())

            count += 1

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", help="Path to the PyMoDAQ Dataset_*.h5 file")
    parser.add_argument("scan_node_name", help="Scan node name, e.g. Scan001")
    parser.add_argument("data_dir", help="Folder with the TPX3 *_events.dat files for that scan node")
    parser.add_argument("--pixels", action="store_true", help="Also copy *_pixels.dat files")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing step subgroups")
    args = parser.parse_args()

    n = export_scan_to_h5(
        args.h5_path, args.scan_node_name, args.data_dir,
        include_pixels=args.pixels, overwrite=args.overwrite,
    )
    print(f"Copied {n} step(s) into {args.h5_path}:.../{args.scan_node_name}/tpx3")
