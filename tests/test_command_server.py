"""
Tests for the ZMQ command server.
"""

import json
import threading
import time
from unittest.mock import MagicMock, PropertyMock

import pytest
import zmq

from SERVAL.core.command_server import CommandServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_pipeline(is_recording=False, filename=None, running=True):
    p = MagicMock()
    type(p).is_recording = PropertyMock(return_value=is_recording)
    p.running = running
    p._recording_state = {"filename": filename, "active": is_recording}
    p.start_record.return_value = True
    return p


def _client_send(port, cmd_dict, timeout_ms=2000):
    """Send one command and return the parsed reply."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        sock.connect(f"tcp://127.0.0.1:{port}")
        sock.send(json.dumps(cmd_dict).encode())
        raw = sock.recv()
        return json.loads(raw)
    finally:
        sock.close()
        ctx.term()


# Use a port in the ephemeral range to avoid conflicts
BASE_PORT = 19200


class TestCommandServer:
    """Each test uses a fresh port to avoid bind conflicts."""

    def _start_server(self, pipeline, port):
        server = CommandServer(pipeline, port=port)
        server.start()
        time.sleep(0.1)  # Allow REP socket to bind
        return server

    def test_ping(self):
        p = _make_mock_pipeline()
        s = self._start_server(p, BASE_PORT)
        try:
            reply = _client_send(BASE_PORT, {"cmd": "ping"})
            assert reply == {"status": "pong"}
        finally:
            s.stop()

    def test_status_not_recording(self):
        p = _make_mock_pipeline(is_recording=False)
        s = self._start_server(p, BASE_PORT + 1)
        try:
            reply = _client_send(BASE_PORT + 1, {"cmd": "status"})
            assert reply["status"] == "ok"
            assert reply["recording"] is False
        finally:
            s.stop()

    def test_status_recording(self):
        p = _make_mock_pipeline(is_recording=True, filename="scan_001")
        s = self._start_server(p, BASE_PORT + 2)
        try:
            reply = _client_send(BASE_PORT + 2, {"cmd": "status"})
            assert reply["recording"] is True
            assert reply["filename"] == "scan_001"
        finally:
            s.stop()

    def test_start_record_dispatches_to_pipeline(self):
        p = _make_mock_pipeline()
        s = self._start_server(p, BASE_PORT + 3)
        try:
            reply = _client_send(BASE_PORT + 3, {
                "cmd": "start_record",
                "filename": "my_scan",
                "save_raw": True,
                "save_events": True,
                "save_pixels": False,
            })
            assert reply["status"] == "ok"
            p.start_record.assert_called_once_with(
                filename="my_scan",
                output_dir=None,
                save_raw=True,
                save_events=True,
                save_pixels=False,
            )
        finally:
            s.stop()

    def test_start_record_missing_filename(self):
        p = _make_mock_pipeline()
        s = self._start_server(p, BASE_PORT + 4)
        try:
            reply = _client_send(BASE_PORT + 4, {"cmd": "start_record"})
            assert reply["status"] == "error"
            assert "filename" in reply["message"]
            p.start_record.assert_not_called()
        finally:
            s.stop()

    def test_stop_record(self):
        p = _make_mock_pipeline(is_recording=True)
        s = self._start_server(p, BASE_PORT + 5)
        try:
            reply = _client_send(BASE_PORT + 5, {"cmd": "stop_record"})
            assert reply["status"] == "ok"
            assert reply["recording"] is False
            p.stop_record.assert_called_once()
        finally:
            s.stop()

    def test_unknown_command(self):
        p = _make_mock_pipeline()
        s = self._start_server(p, BASE_PORT + 6)
        try:
            reply = _client_send(BASE_PORT + 6, {"cmd": "explode"})
            assert reply["status"] == "error"
            assert "explode" in reply["message"]
        finally:
            s.stop()

    def test_start_record_with_output_dir(self):
        p = _make_mock_pipeline()
        s = self._start_server(p, BASE_PORT + 7)
        try:
            _client_send(BASE_PORT + 7, {
                "cmd": "start_record",
                "filename": "scan_x",
                "output_dir": "/data/exp1",
            })
            _, kwargs = p.start_record.call_args
            assert kwargs["output_dir"] == "/data/exp1"
        finally:
            s.stop()

    def test_multiple_sequential_commands(self):
        """Server must handle multiple req/rep cycles correctly."""
        p = _make_mock_pipeline()
        s = self._start_server(p, BASE_PORT + 8)
        try:
            for i in range(5):
                reply = _client_send(BASE_PORT + 8, {"cmd": "ping"})
                assert reply["status"] == "pong"
        finally:
            s.stop()

    def test_stop_cleans_up(self):
        """After stop(), the port should be releasable immediately."""
        p = _make_mock_pipeline()
        s = self._start_server(p, BASE_PORT + 9)
        s.stop()
        time.sleep(0.1)
        # Should be able to start a new server on the same port
        s2 = self._start_server(p, BASE_PORT + 9)
        try:
            reply = _client_send(BASE_PORT + 9, {"cmd": "ping"})
            assert reply["status"] == "pong"
        finally:
            s2.stop()
