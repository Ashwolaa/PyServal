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

Dependencies:
  - pymodaq (>=4.x recommended)
  - pyzmq
  - numpy
"""

import json
import time
from pathlib import Path

import numpy as np

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

try:
    from pymodaq.daq_viewer.utility_classes import DAQ_Viewer_base
    from pymodaq.daq_utils.daq_utils import ThreadCommand
    HAS_PYMODAQ = True
except ImportError:
    # Allow the module to be imported without PyMoDAQ installed
    HAS_PYMODAQ = False

    class DAQ_Viewer_base:
        """Stub base class when PyMoDAQ is not installed."""
        def __init__(self, parent=None, params_state=None):
            pass


PARAMS = [
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
            {"title": "Output Dir:", "name": "output_dir", "type": "str", "value": "./data"},
            {"title": "Filename Prefix:", "name": "filename_prefix", "type": "str", "value": "scan"},
            {
                "title": "Acquisition Time (s):",
                "name": "acquisition_time",
                "type": "float",
                "value": 0.0,
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
    PyMoDAQ 0D viewer plugin for TPX3 pipeline control.

    Connects to the pipeline's ZMQ command server and orchestrates
    per-scan file recording. Returns the scan index as a scalar value,
    allowing PyMoDAQ to embed the TPX3 filename in its HDF5 metadata.
    """

    params = PARAMS

    def __init__(self, parent=None, params_state=None):
        super().__init__(parent, params_state)
        self._socket = None
        self._context = None
        self._scan_index = 0
        self._current_filename = None

    # =========================================================================
    # PyMoDAQ lifecycle
    # =========================================================================

    def ini_detector(self, controller=None):
        """Connect to the pipeline command server."""
        if not HAS_ZMQ:
            return "pyzmq not installed", False

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

            return f"Connected to TPX3 pipeline at {host}:{port}", True

        except Exception as e:
            return f"Connection failed: {e}", False

    def close(self):
        """Disconnect from the pipeline."""
        if self._socket:
            # Best-effort: stop any active recording
            self._send_cmd({"cmd": "stop_record"})
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    def grab_data(self, Naverage=1, **kwargs):
        """
        Trigger one recording cycle and return the scan index.

        - Builds filename from prefix + scan index
        - Sends start_record to pipeline
        - Waits acquisition_time seconds (if > 0)
        - Sends stop_record
        - Emits scan_index as a 0D data point

        The filename is logged in the PyMoDAQ HDF5 dataset as an attribute,
        providing cross-reference between scan position and TPX3 files.
        """
        if self._socket is None:
            self.emit_status(ThreadCommand("Update_Status", ["Not connected"]))
            return

        prefix = self.settings.child("recording", "filename_prefix").value()
        base_output_dir = self.settings.child("recording", "output_dir").value()
        acq_time = self.settings.child("recording", "acquisition_time").value()
        save_raw = self.settings.child("recording", "save_raw").value()
        save_events = self.settings.child("recording", "save_events").value()
        save_pixels = self.settings.child("recording", "save_pixels").value()

        self._scan_index += 1
        filename = f"{prefix}_{self._scan_index:05d}"
        self._current_filename = filename

        # Each scan position gets its own subfolder so the centroiding GUI can
        # treat it as an independent acquisition unit.
        output_dir = str(Path(base_output_dir) / filename)

        # Start recording
        reply = self._send_cmd({
            "cmd": "start_record",
            "filename": filename,
            "output_dir": output_dir,
            "save_raw": save_raw,
            "save_events": save_events,
            "save_pixels": save_pixels,
        })

        if reply.get("status") != "ok":
            msg = f"start_record failed: {reply.get('message', reply)}"
            self.emit_status(ThreadCommand("Update_Status", [msg]))
            return

        # Wait for acquisition time
        if acq_time > 0:
            time.sleep(acq_time)
            self._send_cmd({"cmd": "stop_record"})

        # Emit scan index as the 0D data value
        # PyMoDAQ will store this alongside scan position in HDF5
        data = np.array([float(self._scan_index)])
        self.data_grabed_signal.emit([{"name": "scan_index", "data": [data], "dim": "Data0D"}])

    def stop(self):
        """Stop any active recording (called by PyMoDAQ on scan abort)."""
        self._send_cmd({"cmd": "stop_record"})
        return ""

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _send_cmd(self, cmd_dict: dict) -> dict:
        """Send a JSON command and return the parsed reply dict."""
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
