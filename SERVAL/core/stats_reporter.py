#!/usr/bin/env python3
"""
Statistics Reporter for TPX3 pipeline.

Tracks and reports pipeline statistics via EventBus subscriptions.
"""

import threading
import time
from typing import Optional, Callable, Dict, List

from SERVAL.utils import EventBus, Events
from SERVAL.utils.logging import get_logger


class StatsReporter:
    """
    Tracks and reports pipeline statistics.

    Subscribes to EventBus events and aggregates stats
    from various pipeline components.

    Parameters
    ----------
    event_bus : EventBus
        Event bus to subscribe to
    report_interval : float
        Seconds between periodic reports
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        report_interval: float = 5.0,
    ):
        self.logger = get_logger('SERVAL.StatsReporter')
        self.bus = event_bus
        self.report_interval = report_interval

        # Stats storage
        self._stats: Dict[str, int] = {
            'bytes_received': 0,
            'chunks_sent': 0,
            'chunks_dropped_save': 0,
            'chunks_dropped_zmq': 0,
        }
        self._lock = threading.Lock()

        # Timing
        self.start_time: Optional[float] = None
        self._last_bytes = 0
        self._last_time = 0.0

        # Connection state
        self.is_connected = False

        # Queue references for size reporting (list of queues per type)
        self.queues: Dict[str, List] = {
            'raw': [],
            'events': [],
            'pixels': [],
            'triggers': [],
        }
        # Corresponding maxsizes (set via set_queues)
        self._queue_maxsizes: Dict[str, int] = {}

        # Thread control
        self.running = False
        self._thread: Optional[threading.Thread] = None

        # Status callback for GUI
        self.status_callback: Optional[Callable[[dict], None]] = None

        # Subscribe to events if bus provided
        if self.bus:
            self._subscribe_events()

    def _subscribe_events(self):
        """Subscribe to all relevant events."""
        self.bus.subscribe(Events.BYTES_RECEIVED, self._on_bytes_received)
        self.bus.subscribe(Events.CHUNK_SENT, self._on_chunk_sent)
        self.bus.subscribe(Events.CHUNK_DROPPED_SAVE, self._on_chunk_dropped_save)
        self.bus.subscribe(Events.CHUNK_DROPPED_ZMQ, self._on_chunk_dropped_zmq)
        self.bus.subscribe(Events.CONNECTION_CHANGED, self._on_connection_changed)

    def _on_bytes_received(self, _event: str, nbytes: int):
        """Handle bytes received event."""
        self._increment('bytes_received', nbytes)

    def _on_chunk_sent(self, _event: str, _chunk_size: int):
        """Handle chunk sent event."""
        self._increment('chunks_sent', 1)

    def _on_chunk_dropped_save(self, _event: str):
        """Handle chunk dropped (save) event."""
        self._increment('chunks_dropped_save', 1)

    def _on_chunk_dropped_zmq(self, _event: str):
        """Handle chunk dropped (zmq) event."""
        self._increment('chunks_dropped_zmq', 1)

    def _on_connection_changed(self, _event: str, connected: bool, _address):
        """Handle connection state change."""
        self.is_connected = connected

    def _increment(self, stat_name: str, value: int):
        """Increment a stat value (thread-safe)."""
        with self._lock:
            self._stats[stat_name] = self._stats.get(stat_name, 0) + value

    def set_queues(self, queue_type: str, queues: List, maxsize: int = 0):
        """Set queue references for size reporting.

        Parameters
        ----------
        queue_type : str
            Type of queue ('raw', 'events', 'pixels', 'triggers')
        queues : List
            List of queue objects
        maxsize : int
            Declared maxsize of each queue (used in fill-level display)
        """
        self.queues[queue_type] = queues
        self._queue_maxsizes[queue_type] = maxsize

    def get_stats(self) -> dict:
        """Get current stats snapshot."""
        with self._lock:
            stats = self._stats.copy()

        # Add timing
        if self.start_time:
            stats['elapsed'] = time.time() - self.start_time
            stats['start_time'] = self.start_time

        return stats

    def start(self):
        """Start reporter thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self.running = True
        self.start_time = time.time()
        self._last_time = self.start_time
        self._last_bytes = 0

        self._thread = threading.Thread(target=self._run, name="StatsReporter")
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """Stop reporter thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self):
        """Main reporter loop."""
        self.logger.debug("Starting")

        try:
            while self.running:
                time.sleep(self.report_interval)
                self._report()

        finally:
            self.logger.debug("Shutting down")

    def _report(self):
        """Generate and output stats report."""
        stats = self.get_stats()
        current_time = time.time()
        elapsed = stats.get('elapsed', 0)

        # Calculate rates (MB/s)
        bytes_delta = stats['bytes_received'] - self._last_bytes
        time_delta = current_time - self._last_time

        instant_rate_mbs = (bytes_delta / time_delta) / 1e6 if time_delta > 0 else 0
        avg_rate_mbs = (stats['bytes_received'] / elapsed) / 1e6 if elapsed > 0 else 0

        # Get queue sizes for all types (per-queue breakdown)
        queue_stats = {}
        for qtype, qlist in self.queues.items():
            if qlist:
                try:
                    maxsize = self._queue_maxsizes.get(qtype, 0)
                    queue_stats[qtype] = [(q.qsize(), maxsize) for q in qlist]
                except NotImplementedError:
                    queue_stats[qtype] = [(0, 0) for _ in qlist]
            else:
                queue_stats[qtype] = None

        # Build queue status string with per-process breakdown
        queue_parts = []
        for qtype in ['raw', 'events', 'pixels', 'triggers']:
            qs = queue_stats.get(qtype)
            if qs:
                if len(qs) == 1:
                    queue_parts.append(f"{qtype}={qs[0][0]}/{qs[0][1]}")
                else:
                    per_q = ",".join(f"{s}/{m}" for s, m in qs)
                    queue_parts.append(f"{qtype}=[{per_q}]")
        queue_str = " | ".join(queue_parts) if queue_parts else "none"

        self.logger.info(
            f"Session: {elapsed:.1f}s | {stats['bytes_received']/1e9:.2f} GB | "
            f"{instant_rate_mbs:.1f} MB/s ({avg_rate_mbs:.1f} avg) | "
            f"Chunks: {stats['chunks_sent']} sent, "
            f"dropped: {stats['chunks_dropped_save']} save / {stats['chunks_dropped_zmq']} zmq | "
            f"Queues: {queue_str}"
        )

        # Callback for GUI
        if self.status_callback:
            try:
                status_dict = {
                    'running': self.running,
                    'connected': self.is_connected,
                    'elapsed': elapsed,
                    'bytes_received': stats['bytes_received'],
                    'rate_mbs': instant_rate_mbs,
                    'avg_rate_mbs': avg_rate_mbs,
                    'chunks_sent': stats['chunks_sent'],
                    'chunks_dropped_save': stats['chunks_dropped_save'],
                    'chunks_dropped_zmq': stats['chunks_dropped_zmq'],
                    'queues': queue_stats,
                }
                self.status_callback(status_dict)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")

        # Update for next iteration
        self._last_bytes = stats['bytes_received']
        self._last_time = current_time

    def print_final_stats(self):
        """Print final statistics summary."""
        stats = self.get_stats()
        elapsed = stats.get('elapsed', 0)

        avg = f"{(stats['bytes_received']/elapsed)/1e6:.1f} MB/s" if elapsed > 0 else "n/a"
        self.logger.info(
            f"Final — duration: {elapsed:.1f}s | data: {stats['bytes_received']/1e9:.2f} GB | "
            f"avg rate: {avg} | chunks sent: {stats['chunks_sent']} | "
            f"dropped: {stats['chunks_dropped_save']} save / {stats['chunks_dropped_zmq']} zmq"
        )
