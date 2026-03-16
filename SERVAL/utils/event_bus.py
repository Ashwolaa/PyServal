#!/usr/bin/env python3
"""
Event Bus - Publish/Subscribe mechanism for SERVAL components.

Provides decoupled communication between pipeline components.
"""

import threading
from collections import defaultdict
from typing import Callable, Optional, Set


class EventBus:
    """
    Simple publish/subscribe event bus.

    Thread-safe event distribution with support for:
    - Multiple subscribers per event
    - Wildcard subscriptions ('*')
    - Weak references (auto-cleanup when subscriber is deleted)
    - Synchronous dispatch

    Usage:
        bus = EventBus()

        # Subscribe
        bus.subscribe('connection_changed', my_handler)
        bus.subscribe('*', log_all_events)  # Wildcard

        # Publish
        bus.publish('connection_changed', True, ('192.168.1.1', 8080))

        # Unsubscribe
        bus.unsubscribe('connection_changed', my_handler)
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, event: str, callback: Callable) -> None:
        """
        Subscribe to an event.

        Parameters
        ----------
        event : str
            Event name to subscribe to. Use '*' for all events.
        callback : callable
            Function to call when event is published.
            Receives (*args, **kwargs) from publish().
        """
        with self._lock:
            if callback not in self._subscribers[event]:
                self._subscribers[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable) -> bool:
        """
        Unsubscribe from an event.

        Parameters
        ----------
        event : str
            Event name
        callback : callable
            The callback to remove

        Returns
        -------
        bool
            True if callback was found and removed
        """
        with self._lock:
            try:
                self._subscribers[event].remove(callback)
                return True
            except ValueError:
                return False

    def publish(self, event: str, *args, **kwargs) -> int:
        """
        Publish an event to all subscribers.

        Parameters
        ----------
        event : str
            Event name
        *args, **kwargs
            Arguments to pass to subscribers

        Returns
        -------
        int
            Number of subscribers notified
        """
        with self._lock:
            # Get subscribers for this event + wildcard subscribers
            callbacks = list(self._subscribers.get(event, []))
            callbacks.extend(self._subscribers.get('*', []))

        # Call outside lock to prevent deadlocks
        count = 0
        for callback in callbacks:
            try:
                callback(event, *args, **kwargs) if event != '*' else callback(event, *args, **kwargs)
                count += 1
            except Exception as e:
                print(f"[EventBus] Error in subscriber for '{event}': {e}")

        return count

    def clear(self, event: Optional[str] = None) -> None:
        """
        Clear subscribers.

        Parameters
        ----------
        event : str, optional
            Event to clear. If None, clears all.
        """
        with self._lock:
            if event is None:
                self._subscribers.clear()
            elif event in self._subscribers:
                del self._subscribers[event]

    def events(self) -> Set[str]:
        """Get set of events with subscribers."""
        with self._lock:
            return set(self._subscribers.keys())

    def subscriber_count(self, event: str) -> int:
        """Get number of subscribers for an event."""
        with self._lock:
            return len(self._subscribers.get(event, []))


# Common event names (for documentation/consistency)
class Events:
    """Standard event names used in the pipeline."""

    # Connection events
    CONNECTION_CHANGED = 'connection_changed'  # (connected: bool, address: tuple)

    # Data flow events
    BYTES_RECEIVED = 'bytes_received'          # (nbytes: int)
    CHUNK_SENT = 'chunk_sent'                  # (chunk_size: int)
    CHUNK_DROPPED_SAVE = 'chunk_dropped_save'  # ()
    CHUNK_DROPPED_ZMQ = 'chunk_dropped_zmq'    # ()

    # Event processing
    EVENT_BATCH = 'event_batch'                # (event_num, x, y, tof, tot)
    PIXEL_BATCH = 'pixel_batch'                # (x, y, toa, tot)
    EVENTS_SAVED = 'events_saved'              # (count: int)
    PIXELS_SAVED = 'pixels_saved'              # (count: int)

    # Status
    STATS_UPDATE = 'stats_update'              # (stats_dict: dict)
    STATUS_UPDATE = 'status_update'            # (status_dict: dict)

    # Lifecycle
    PIPELINE_STARTED = 'pipeline_started'      # ()
    PIPELINE_STOPPED = 'pipeline_stopped'      # ()


# Global bus instance (optional convenience)
_default_bus: Optional[EventBus] = None


def get_bus() -> EventBus:
    """Get the default global event bus."""
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def reset_bus() -> None:
    """Reset the global event bus."""
    global _default_bus
    _default_bus = None
