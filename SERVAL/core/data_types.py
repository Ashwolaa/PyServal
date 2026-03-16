#!/usr/bin/env python3
"""
Data containers for TPX3 pipeline.

These dataclasses hold extracted pixel and trigger data with
helper methods for concatenation and basic operations.
"""

from dataclasses import dataclass
import numpy as np


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
