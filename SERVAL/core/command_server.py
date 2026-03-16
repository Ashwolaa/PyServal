#!/usr/bin/env python3
"""
ZMQ command server for external pipeline control.

Listens on a REP socket for JSON commands from PyMoDAQ or any other
client. Commands are dispatched to the TPX3PipelineV3 instance.

Protocol (JSON over ZMQ REQ/REP):
  → {"cmd": "ping"}
  ← {"status": "pong"}

  → {"cmd": "start_record", "filename": "scan_001", "output_dir": "/data",
       "save_raw": true, "save_events": true, "save_pixels": false}
  ← {"status": "ok", "recording": true}

  → {"cmd": "stop_record"}
  ← {"status": "ok", "recording": false}

  → {"cmd": "status"}
  ← {"status": "ok", "recording": true, "filename": "scan_001"}
"""

import json
import threading

import zmq

from SERVAL.utils.logging import get_logger


class CommandServer:
    """
    ZMQ REP command server for dynamic pipeline control.

    Runs in a daemon thread. Thread-safe: command dispatch calls
    pipeline.start_record() / stop_record() which are designed to be
    called from any thread.
    """

    def __init__(self, pipeline, port: int = 9100):
        """
        Parameters
        ----------
        pipeline : TPX3PipelineV3
            Pipeline instance to control.
        port : int
            Port to bind the REP socket on (default 9100).
        """
        self.pipeline = pipeline
        self.port = port
        self.logger = get_logger("SERVAL.CommandServer")
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start the command server in a daemon thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._serve,
            name="CommandServer",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Signal the server to stop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _serve(self):
        """REP loop: receive JSON command, dispatch, send JSON reply."""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.RCVTIMEO, 500)  # 500 ms poll interval
        try:
            socket.bind(f"tcp://*:{self.port}")
            self.logger.info(f"Bound on port {self.port}")

            while not self._stop_event.is_set():
                try:
                    raw = socket.recv()
                except zmq.Again:
                    continue  # Timeout — check stop event

                try:
                    msg = json.loads(raw)
                    reply = self._dispatch(msg)
                except Exception as e:
                    reply = {"status": "error", "message": f"Parse error: {e}"}

                try:
                    socket.send(json.dumps(reply).encode())
                except Exception as e:
                    self.logger.error(f"Failed to send reply: {e}")

        except Exception as e:
            self.logger.error(f"Server error: {e}", exc_info=True)
        finally:
            socket.close()
            context.term()
            self.logger.info("Stopped.")

    def _dispatch(self, msg: dict) -> dict:
        """Dispatch a command dict and return a reply dict."""
        cmd = msg.get("cmd", "")

        if cmd == "ping":
            return {"status": "pong"}

        elif cmd == "start_record":
            filename = msg.get("filename", "")
            if not filename:
                return {"status": "error", "message": "filename required"}
            success = self.pipeline.start_record(
                filename=filename,
                output_dir=msg.get("output_dir", None),
                save_raw=msg.get("save_raw", True),
                save_events=msg.get("save_events", True),
                save_pixels=msg.get("save_pixels", False),
            )
            return {
                "status": "ok" if success else "error",
                "recording": self.pipeline.is_recording,
            }

        elif cmd == "stop_record":
            self.pipeline.stop_record()
            return {"status": "ok", "recording": False}

        elif cmd == "status":
            return {
                "status": "ok",
                "recording": self.pipeline.is_recording,
                "filename": self.pipeline._recording_state.get("filename"),
                "pipeline_running": self.pipeline.running,
            }

        else:
            return {"status": "error", "message": f"Unknown command: {cmd!r}"}
