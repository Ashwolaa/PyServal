#!/usr/bin/env python3
"""
Data containers and enumerations for the TPX3 pipeline.

Dataclasses hold extracted pixel and trigger data.
Enumerations bridge equipment-level integer codes and human-readable labels,
used consistently across the pipeline, workers, and GUI.
"""

from dataclasses import dataclass
from enum import IntEnum
import numpy as np


# =============================================================================
# Enumerations
# =============================================================================

class TDCChannel(IntEnum):
    """TDC channel selector.

    The integer value is the equipment code used in the data stream and all
    pipeline configuration dicts (``extract_config["tdc_id"]``).

    ``BOTH`` (0) accepts triggers from either channel.
    """
    BOTH = 0
    TDC1 = 1
    TDC2 = 2

    @property
    def label(self) -> str:
        """Human-readable label used in the GUI."""
        return {
            TDCChannel.BOTH: "Both",
            TDCChannel.TDC1: "TDC1",
            TDCChannel.TDC2: "TDC2",
        }[self]

    @classmethod
    def from_label(cls, label: str) -> "TDCChannel":
        """Return the TDCChannel matching a GUI label string."""
        mapping = {ch.label: ch for ch in cls}
        if label not in mapping:
            raise ValueError(f"Unknown TDC label {label!r}. Valid: {list(mapping)}")
        return mapping[label]

    @classmethod
    def labels(cls) -> list:
        """Ordered list of labels for populating GUI drop-downs."""
        return [ch.label for ch in cls]


class TriggerEdge(IntEnum):
    """Trigger edge selector.

    The integer value matches the ``edge`` field in ``TRIGGER_DTYPE`` and the
    classification produced by ``extract_triggers()``:
    ``0`` = rising, ``1`` = falling.
    """
    RISING  = 0
    FALLING = 1

    @property
    def label(self) -> str:
        """Human-readable label used in the GUI."""
        return self.name.capitalize()   # "Rising" / "Falling"

    @classmethod
    def from_label(cls, label: str) -> "TriggerEdge":
        """Return the TriggerEdge matching a GUI label string."""
        mapping = {e.label: e for e in cls}
        if label not in mapping:
            raise ValueError(f"Unknown edge label {label!r}. Valid: {list(mapping)}")
        return mapping[label]

    @classmethod
    def labels(cls) -> list:
        """Ordered list of labels for populating GUI drop-downs."""
        return [e.label for e in cls]


@dataclass
class PixelData:
    """
    Container for extracted pixel data.

    Attributes
    ----------
    x : np.ndarray[uint16]
        X coordinates (0-255)
    y : np.ndarray[uint16]
        Y coordinates (0-255)
    toa : np.ndarray[float64]
        Time of arrival in seconds
    tot : np.ndarray[uint32]
        Time over threshold in ns
    """
    x: np.ndarray
    y: np.ndarray
    toa: np.ndarray
    tot: np.ndarray

    def __len__(self):
        return len(self.x) if self.x is not None else 0

    def concatenate(self, other: 'PixelData') -> 'PixelData':
        """Merge two PixelData objects."""
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self
        return PixelData(
            x=np.concatenate([self.x, other.x]),
            y=np.concatenate([self.y, other.y]),
            toa=np.concatenate([self.toa, other.toa]),
            tot=np.concatenate([self.tot, other.tot]),
        )

    @classmethod
    def empty(cls) -> 'PixelData':
        """Create an empty PixelData container."""
        return cls(
            x=np.array([], dtype=np.uint16),
            y=np.array([], dtype=np.uint16),
            toa=np.array([], dtype=np.float64),
            tot=np.array([], dtype=np.uint32),
        )


@dataclass
class TriggerData:
    """
    Container for extracted trigger data.

    Attributes
    ----------
    toa : np.ndarray[float64]
        Time of arrival in seconds
    tdc_id : np.ndarray[uint8]
        TDC identifier (1 for TDC1, 2 for TDC2)
    edge : np.ndarray[uint8]
        Edge type (0 for rising, 1 for falling)
    """
    toa: np.ndarray
    tdc_id: np.ndarray
    edge: np.ndarray

    def __len__(self):
        return len(self.toa) if self.toa is not None else 0

    def concatenate(self, other: 'TriggerData') -> 'TriggerData':
        """Merge two TriggerData objects."""
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self
        return TriggerData(
            toa=np.concatenate([self.toa, other.toa]),
            tdc_id=np.concatenate([self.tdc_id, other.tdc_id]),
            edge=np.concatenate([self.edge, other.edge]),
        )

    @classmethod
    def empty(cls) -> 'TriggerData':
        """Create an empty TriggerData container."""
        return cls(
            toa=np.array([], dtype=np.float64),
            tdc_id=np.array([], dtype=np.uint8),
            edge=np.array([], dtype=np.uint8),
        )


def merge_triggers(*trigger_lists: TriggerData) -> TriggerData:
    """
    Merge TriggerData from multiple sources and sort by timestamp.

    Parameters
    ----------
    *trigger_lists : TriggerData
        Variable number of TriggerData objects to merge

    Returns
    -------
    TriggerData
        Merged and sorted trigger data
    """
    if not trigger_lists:
        return TriggerData.empty()

    result = trigger_lists[0]
    for triggers in trigger_lists[1:]:
        result = result.concatenate(triggers)

    if len(result) > 0:
        sort_idx = np.argsort(result.toa)
        result = TriggerData(
            toa=result.toa[sort_idx],
            tdc_id=result.tdc_id[sort_idx],
            edge=result.edge[sort_idx],
        )

    return result


def merge_pixels(*pixel_lists: PixelData) -> PixelData:
    """
    Merge PixelData from multiple sources.

    Parameters
    ----------
    *pixel_lists : PixelData
        Variable number of PixelData objects to merge

    Returns
    -------
    PixelData
        Merged pixel data (not sorted)
    """
    if not pixel_lists:
        return PixelData.empty()

    result = pixel_lists[0]
    for pixels in pixel_lists[1:]:
        result = result.concatenate(pixels)

    return result
