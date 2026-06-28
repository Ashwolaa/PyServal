#!/usr/bin/env python3
"""
PyMoDAQ 0D Viewer plugin for TPX3/SERVAL pipeline.

Returns the scan index as a scalar signal. The actual TPX3 data is
written to .tpx3/.dat files on disk, synchronized with PyMoDAQ's
scan metadata via the filename embedded in HDF5.

Usage in PyMoDAQ:
  - Add as DAQ_0DViewer with plugin "Tpx3Serval"
  - Set command_host/port to match the pipeline's command_server settings
  - ini_detector() connects and pings the pipeline
  - grab_data() sends start_record, waits acquisition_time, sends stop_record
  - The TPX3 filename is logged in the PyMoDAQ HDF5 as an attribute

During a DAQ_Scan run, the output folder is derived from ScanInfo (see
https://github.com/PyMoDAQ/PyMoDAQ/pull/1086): PyMoDAQ passes
``scan_info=ScanInfo(ind_scan, ind_average, scan_node_name, h5_file_path)``
as a grab_data() kwarg. We use ``h5_file_path`` to create one folder per
dataset, named after the h5 file itself (e.g. ``Dataset_20260620_001/``),
right next to PyMoDAQ's own .h5, with one subfolder per scan node inside it
(e.g. ``Scan000/``) — mirroring PyMoDAQ's own Dataset/Scan000 h5 grouping.
Step index alone names each file (e.g. ``00001_events.dat``). Until that PR
is merged, scan_info is absent and grab_data() falls back to
manual/standalone-style folder naming under the configured Output Dir.

The pipeline's recording metadata (TDC config, event window, centroiding,
column correction — see ``pipeline._write_metadata``) is identical for every
step of a scan, so it's written once per scan (``Scan000/_scan_meta.json``,
on the first step) rather than once per step file, and once per manual
session (``_session_meta.json``) rather than once per manual grab.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np

from pymodaq_utils.utils import ThreadCommand
from pymodaq_data.data import DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.data import DataFromPlugins

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False


PARAMS = comon_parameters + [
    {
        "title": "Connection",
        "name": "connection",
        "type": "group",
        "children": [
            {"title": "Command Host:", "name": "command_host", "type": "str", "value": "localhost"},
            {"title": "Command Port:", "name": "command_port", "type": "int", "value": 9100},
        ],
    },
    {
        "title": "Recording",
        "name": "recording",
        "type": "group",
        "children": [
            {"title": "Output Dir:", "name": "output_dir", "type": "str", "value": "./data",
             "tip": "Used for manual/standalone grabs only (outside a DAQ_Scan run). "
                    "During a scan, the folder is derived directly from ScanInfo.h5_file_path: "
                    "<h5 file's dir>/<h5 file's stem>/<scan_node_name>/, with step index as filename. "
                    "Manual grabs get an auto-timestamped subfolder under Output Dir, "
                    "fresh on every detector initialisation."},
            {"title": "Filename Prefix:", "name": "filename_prefix", "type": "str", "value": "scan"},
            {
                "title": "Acquisition Time (s):",
                "name": "acquisition_time",
                "type": "float",
                "value": 10.0,
                "tip": "0 = manual stop via stop()",
            },
            {"title": "Save Raw (.tpx3):", "name": "save_raw", "type": "bool", "value": True},
            {"title": "Save Events (.dat):", "name": "save_events", "type": "bool", "value": True},
            {"title": "Save Pixels (.dat):", "name": "save_pixels", "type": "bool", "value": False},
        ],
    },
]


class DAQ_0DViewer_Tpx3Serval(DAQ_Viewer_base):
    """
    PyMoDAQ 0D viewer plugin for TPX3 pipeline control via SERVAL.

    Connects to the pipeline's ZMQ command server and orchestrates
    per-scan file recording. Returns the scan index as a scalar value,
    allowing PyMoDAQ to embed the TPX3 filename in its HDF5 metadata.

    Each grab_data() call:
      1. Resolves a folder from ScanInfo.h5_file_path: one per dataset, one
         per scan node inside it, and a filename from the zero-padded step
         index
      2. Sends start_record (flat=True) to the pipeline
      3. Waits acquisition_time seconds (if > 0), then sends stop_record
      4. Emits the scan index as a Data0D value

    Attributes
    ----------
    controller : None
        Not used (no shared controller).
    """

    params = PARAMS

    def ini_attributes(self):
        self.controller = None
        self._socket = None
        self._context = None
        self._scan_index = 0
        self._current_filename = None
        self._scan_dir = None  # global folder for the current scan session

    def ini_detector(self, controller=None):
        """Connect to the pipeline command server and verify with a ping."""
        if not HAS_ZMQ:
            return "pyzmq not installed", False

        # Reset session state so each initialisation starts a fresh global folder
        self._scan_index = 0
        self._scan_dir = None

        try:
            host = self.settings.child("connection", "command_host").value()
            port = self.settings.child("connection", "command_port").value()

            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 s timeout
            self._socket.setsockopt(zmq.SNDTIMEO, 5000)
            self._socket.connect(f"tcp://{host}:{port}")

            reply = self._send_cmd({"cmd": "ping"})
            if reply.get("status") != "pong":
                return f"Unexpected ping reply: {reply}", False

            # Emit a zero placeholder so PyMoDAQ knows the data shape
            self.dte_signal_temp.emit(DataToExport(
                name='tpx3serval',
                data=[DataFromPlugins(
                    name='scan_index',
                    data=[np.array([0.0])],
                    dim='Data0D',
                    labels=['scan_index'],
                )],
            ))

            return f"Connected to TPX3 pipeline at {host}:{port}", True

        except Exception as e:
            return f"Connection failed: {e}", False

    def close(self):
        """Disconnect from the pipeline, stopping any active recording first."""
        if self._socket:
            self._send_cmd({"cmd": "stop_record"})
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    def grab_data(self, Naverage=1, **kwargs):
        """
        Trigger one recording cycle and emit the scan index as 0D data.

        In scan mode (PyMoDAQ >= the ScanInfo PR, see
        https://github.com/PyMoDAQ/PyMoDAQ/pull/1086), ``kwargs['scan_info']``
        is a ``ScanInfo(ind_scan, ind_average, scan_node_name, h5_file_path)``.
        ``h5_file_path`` points at the .h5 *file* itself (e.g.
        ``.../20260620/Dataset_20260620_001.h5``), so the folder is
        ``<same dir>/<h5 file's stem>/<scan_node_name>/`` (e.g.
        ``Dataset_20260620_001/Scan000/``) — one folder per dataset, one
        subfolder per scan node, mirroring PyMoDAQ's own
        Dataset/Scan000 h5 grouping. Step index alone is enough inside that
        folder (e.g. ``00001_events.dat``) — no extra per-step subfolder.
        Resolved statelessly on every call (no session bookkeeping), so it
        doesn't matter how many times the detector gets re-initialised.

        Outside scan mode (manual/standalone use, no ScanInfo available), a
        single session folder under the configured Output Dir is created on
        first use and kept until close()/stop().
        """
        if self._socket is None:
            self.emit_status(ThreadCommand("Update_Status", ["Not connected"]))
            return

        prefix = self.settings.child("recording", "filename_prefix").value()
        acq_time = self.settings.child("recording", "acquisition_time").value()
        save_raw = self.settings.child("recording", "save_raw").value()
        save_events = self.settings.child("recording", "save_events").value()
        save_pixels = self.settings.child("recording", "save_pixels").value()

        scan_info = kwargs.get('scan_info')
        in_scan_mode = scan_info is not None

        if in_scan_mode:
            # Stateless: dataset folder (named after the h5 file itself) +
            # scan-node subfolder, resolved fresh every call — no need to
            # guess PyMoDAQ's save path. Step index alone names the file.
            h5_path = Path(scan_info.h5_file_path)
            scan_dir = h5_path.parent / h5_path.stem / scan_info.scan_node_name
            scan_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{scan_info.ind_scan + 1:05d}"
            # Pipeline config is identical for every step of a scan — write
            # the metadata file once (on the first step) instead of once per
            # step, at the scan folder level rather than per-step filename.
            write_metadata = scan_info.ind_scan == 0
            metadata_name = "_scan"
        else:
            # Manual / standalone: one session folder for the run, created
            # on first use, under the configured Output Dir. Always
            # timestamped (not user-nameable) so repeated manual sessions
            # never silently overwrite a previous one's files.
            if self._scan_dir is None:
                base_output_dir = Path(self.settings.child("recording", "output_dir").value())
                folder_name = datetime.now().strftime("scan_%Y%m%d_%H%M%S")
                self._scan_dir = base_output_dir / folder_name
                self._scan_dir.mkdir(parents=True, exist_ok=True)
            scan_dir = self._scan_dir
            self._scan_index += 1
            filename = f"{prefix}_{self._scan_index:05d}"
            # Same rationale as scan mode: one metadata file per session.
            write_metadata = self._scan_index == 1
            metadata_name = "_session"

        self._scan_dir = scan_dir
        self._current_filename = filename

        reply = self._send_cmd({
            "cmd": "start_record",
            "filename": filename,
            "output_dir": str(scan_dir),
            "save_raw": save_raw,
            "save_events": save_events,
            "save_pixels": save_pixels,
            "flat": True,
            "write_metadata": write_metadata,
            "metadata_name": metadata_name,
        })

        if reply.get("status") != "ok":
            msg = f"start_record failed: {reply.get('message', reply)}"
            self.emit_status(ThreadCommand("Update_Status", [msg]))
            return

        if acq_time > 0:
            time.sleep(acq_time)
            self._send_cmd({"cmd": "stop_record"})

        emitted_index = float(scan_info.ind_scan + 1) if in_scan_mode else float(self._scan_index)
        self.dte_signal.emit(DataToExport(
            name='tpx3serval',
            data=[DataFromPlugins(
                name='scan_index',
                data=[np.array([emitted_index])],
                dim='Data0D',
                labels=['scan_index'],
            )],
        ))

    def stop(self):
        """Stop any active recording (called by PyMoDAQ on scan abort)."""
        self._send_cmd({"cmd": "stop_record"})
        # Reset session state so the next grab creates a fresh global folder
        self._scan_dir = None
        self._scan_index = 0
        return ""

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _send_cmd(self, cmd_dict: dict) -> dict:
        """Send a JSON command over ZMQ REQ and return the parsed reply."""
        if self._socket is None:
            return {"status": "error", "message": "not connected"}
        try:
            self._socket.send(json.dumps(cmd_dict).encode())
            raw = self._socket.recv()
            return json.loads(raw)
        except zmq.Again:
            return {"status": "error", "message": "timeout"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


if __name__ == '__main__':
    main(__file__)
