#!/usr/bin/env python3
"""
Statistics Reporter for TPX3 pipeline.

Tracks and reports pipeline statistics via EventBus subscriptions.
"""

import threading
import time
from typing import Optional, Callable, Dict, List

from SERVAL.utils import EventBus, Events


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

        # Queue references for size reporting
        self.queues: Dict[str, List] = {
            'raw': [],
            'events': [],
            'pixels': [],
        }

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

    def set_queues(self, queue_type: str, queues: List):
        """Set queue references for size reporting.

        Parameters
        ----------
        queue_type : str
            Type of queue ('raw', 'events', 'pixels')
        queues : List
            List of queue objects
        """
        self.queues[queue_type] = queues

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
        print("[Stats] Starting")

        try:
            while self.running:
                time.sleep(self.report_interval)
                self._report()

        finally:
            print("[Stats] Shutting down")

    def _report(self):
        """Generate and output stats report."""
        stats = self.get_stats()
        current_time = time.time()
        elapsed = stats.get('elapsed', 0)

        # Calculate rates
        bytes_delta = stats['bytes_received'] - self._last_bytes
        time_delta = current_time - self._last_time

        instant_rate_mbps = (bytes_delta / time_delta) / 1e6 * 8 if time_delta > 0 else 0
        avg_rate_mbps = (stats['bytes_received'] / elapsed) / 1e6 * 8 if elapsed > 0 else 0

        # Get queue sizes for all types (per-queue breakdown)
        queue_stats = {}
        for qtype, qlist in self.queues.items():
            if qlist:
                try:
                    per_queue = [(q.qsize(), q._maxsize) for q in qlist]
                    queue_stats[qtype] = per_queue
                except (NotImplementedError, AttributeError):
                    queue_stats[qtype] = [(0, 0) for _ in qlist]
            else:
                queue_stats[qtype] = None

        # Build queue status string with per-process breakdown
        queue_parts = []
        for qtype in ['raw', 'events', 'pixels']:
            qs = queue_stats.get(qtype)
            if qs is not None and len(qs) > 0:
                if len(qs) == 1:
                    # Single queue - simple format
                    queue_parts.append(f"{qtype}={qs[0][0]}/{qs[0][1]}")
                else:
                    # Multiple queues - show each
                    per_q = ",".join(f"{s}/{m}" for s, m in qs)
                    queue_parts.append(f"{qtype}=[{per_q}]")
        queue_str = " | ".join(queue_parts) if queue_parts else "none"

        # Console output
        print(f"\n{'='*80}")
        print(f"[Stats] Session: {elapsed:.1f}s | Total: {stats['bytes_received']/1e9:.2f} GB")
        print(f"[Stats] Rate: {instant_rate_mbps:.1f} Mbps (instant) | {avg_rate_mbps:.1f} Mbps (avg)")
        print(f"[Stats] Chunks: {stats['chunks_sent']} sent | "
              f"Dropped: {stats['chunks_dropped_save']} save, {stats['chunks_dropped_zmq']} zmq")
        print(f"[Stats] Queues: {queue_str}")
        print(f"[Stats] Connected: {self.is_connected}")
        print(f"{'='*80}\n")

        # Callback for GUI
        if self.status_callback:
            try:
                status_dict = {
                    'running': self.running,
                    'connected': self.is_connected,
                    'elapsed': elapsed,
                    'bytes_received': stats['bytes_received'],
                    'rate_mbps': instant_rate_mbps,
                    'avg_rate_mbps': avg_rate_mbps,
                    'chunks_sent': stats['chunks_sent'],
                    'chunks_dropped_save': stats['chunks_dropped_save'],
                    'chunks_dropped_zmq': stats['chunks_dropped_zmq'],
                    'queues': queue_stats,
                }
                self.status_callback(status_dict)
            except Exception as e:
                print(f"[Stats] Callback error: {e}")

        # Update for next iteration
        self._last_bytes = stats['bytes_received']
        self._last_time = current_time

    def print_final_stats(self):
        """Print final statistics summary."""
        stats = self.get_stats()
        elapsed = stats.get('elapsed', 0)

        print(f"\n{'='*80}")
        print("FINAL STATISTICS")
        print(f"{'='*80}")
        print(f"Duration:       {elapsed:.1f}s")
        print(f"Total data:     {stats['bytes_received']/1e9:.2f} GB")
        if elapsed > 0:
            print(f"Average rate:   {(stats['bytes_received']/elapsed)/1e6*8:.1f} Mbps")
        print(f"Chunks sent:    {stats['chunks_sent']}")
        print(f"Chunks dropped: {stats['chunks_dropped_save']} save, {stats['chunks_dropped_zmq']} zmq")
        print(f"{'='*80}\n")
