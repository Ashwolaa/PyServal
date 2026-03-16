#!/usr/bin/env python3
"""
TPX3 Packet Processing

Provides TPX3Extractor for extracting pixels and triggers from raw data,
and TPX3Correlator for correlating pixels to trigger events.

Usage:
    # Extraction
    extractor = TPX3Extractor()
    pixels, triggers, remaining, timestamps = extractor.extract_fast(raw_bytes)

    # Correlation (after merging triggers from all workers)
    correlator = TPX3Correlator(all_triggers, event_window=(0, 10_000))
    events = correlator.correlate(pixels)
"""

import time
from typing import Tuple, Optional

import numpy as np

from SERVAL.utils.logging import get_logger
from SERVAL.core.data_types import PixelData, TriggerData, merge_triggers, merge_pixels
from SERVAL.core.extractors.jit_functions import (
    NUMBA_AVAILABLE,
    classify_packets,
    parse_chunks,
    extract_pixels,
    extract_triggers,
    correlate_pixels as _correlate_pixels_jit,
    correlate_pixels_parallel as _correlate_pixels_parallel,
)

# Re-export for backwards compatibility
__all__ = [
    'TPX3Extractor',
    'TPX3Correlator',
    'PixelData',
    'TriggerData',
    'merge_triggers',
    'merge_pixels',
    '_correlate_pixels_jit',
    '_correlate_pixels_parallel',
]


class TPX3Extractor:
    """
    Stateless packet extractor - safe for parallel use.

    Extracts pixels and triggers from raw TPX3 data chunks.
    Each worker can have its own instance.
    """

    # TPX3 constants
    TPX3_SIGNATURE = 0x33585054
    TDC1_RISING = 0xF
    TDC1_FALLING = 0xA
    TDC2_RISING = 0xE
    TDC2_FALLING = 0xB

    def __init__(self, debug_log_interval: int = 0):
        """
        Initialize the extractor.

        Parameters
        ----------
        debug_log_interval : int
            Log timing stats every N chunks. Set to 0 to disable.
        """
        self.logger = get_logger('SERVAL.Extractor')
        self.debug_log_interval = debug_log_interval
        self._chunk_counter = 0

    def extract_fast(
        self, raw_bytes: bytes
    ) -> Tuple[PixelData, TriggerData, bytes, Tuple[int, int]]:
        """
        Extract pixels and triggers using JIT-compiled functions.

        Parameters
        ----------
        raw_bytes : bytes
            Raw TPX3 data

        Returns
        -------
        pixels : PixelData
            Extracted pixel data with timestamps in seconds
        triggers : TriggerData
            Extracted trigger data with timestamps in seconds
        remaining : bytes
            Incomplete chunk data for next call
        timestamps : tuple[int, int]
            (min_timestamp, max_timestamp) of processed chunks
        """
        if not NUMBA_AVAILABLE:
            self.logger.warning("Numba not available, falling back to standard extract()")
            return self.extract(raw_bytes)

        t_start = time.perf_counter()

        # Convert to uint64 words
        n_complete_words = len(raw_bytes) // 8
        words = np.frombuffer(raw_bytes[:n_complete_words * 8], dtype=np.uint64)

        # Parse chunks
        all_packets, all_min_ts, n_packets, global_min_ts, global_max_ts, remaining_offset = \
            parse_chunks(words, np.uint32(self.TPX3_SIGNATURE))

        t_parse = time.perf_counter()

        # Handle remaining bytes
        remaining_byte_offset = remaining_offset * 8
        remaining = raw_bytes[remaining_byte_offset:] if remaining_byte_offset < len(raw_bytes) else b''

        # Empty results
        if n_packets == 0:
            return PixelData.empty(), TriggerData.empty(), remaining, (0, 0)

        # Classify packets
        is_pixel, is_tdc, subheaders = classify_packets(all_packets)
        t_classify = time.perf_counter()

        # Extract pixels
        pixels = PixelData.empty()
        if np.any(is_pixel):
            pixel_packets = all_packets[is_pixel]
            pixel_min_ts = all_min_ts[is_pixel]
            x, y, toa, tot = extract_pixels(pixel_packets, pixel_min_ts)
            pixels = PixelData(
                x=x,
                y=y,
                toa=toa.astype(np.float64) * 1.5625e-9,
                tot=tot,
            )
        t_pixels = time.perf_counter()

        # Extract triggers
        triggers = TriggerData.empty()
        if np.any(is_tdc):
            tdc_packets = all_packets[is_tdc]
            tdc_subheaders = subheaders[is_tdc]
            tdc_min_ts = all_min_ts[is_tdc]
            toa, tdc_id, edge = extract_triggers(tdc_packets, tdc_subheaders, tdc_min_ts)
            triggers = TriggerData(
                toa=toa.astype(np.float64) * 260.41666e-12,
                tdc_id=tdc_id,
                edge=edge,
            )
        t_triggers = time.perf_counter()

        # Optional timing diagnostics
        self._chunk_counter += 1
        if self.debug_log_interval > 0 and self._chunk_counter % self.debug_log_interval == 0:
            total_ms = (t_triggers - t_start) * 1000
            self.logger.debug(
                f"[FAST] {len(pixels)} px, {len(triggers)} trig in {total_ms:.2f}ms | "
                f"parse:{(t_parse-t_start)*1000:.1f} classify:{(t_classify-t_parse)*1000:.1f} "
                f"px:{(t_pixels-t_classify)*1000:.1f} trig:{(t_triggers-t_pixels)*1000:.1f}ms"
            )

        return pixels, triggers, remaining, (int(global_min_ts), int(global_max_ts))

    def extract(
        self, raw_bytes: bytes
    ) -> Tuple[PixelData, TriggerData, bytes, Tuple[int, int]]:
        """
        Extract pixels and triggers using standard numpy (no JIT).

        Same parameters and return values as extract_fast().
        """
        t_start = time.perf_counter()
        packets_list, n_chunks, remaining = self._parse_chunks(raw_bytes)

        if packets_list is None:
            return PixelData.empty(), TriggerData.empty(), remaining, (0, 0)

        # Collect arrays
        pixel_x_list, pixel_y_list, pixel_toa_list, pixel_tot_list = [], [], [], []
        trigger_toa_list, trigger_tdc_id_list, trigger_edge_list = [], [], []
        global_min_ts = None
        global_max_ts = None

        for packets, (min_ts, max_ts) in packets_list:
            if global_min_ts is None or min_ts < global_min_ts:
                global_min_ts = min_ts
            if global_max_ts is None or max_ts > global_max_ts:
                global_max_ts = max_ts

            # Classify
            header = (packets >> 60) & 0xF
            subheader = (packets >> 56) & 0xF

            # Pixels
            pixel_mask = (header == 0xA) | (header == 0xB)
            if np.any(pixel_mask):
                px = self._extract_pixels(packets[pixel_mask], min_ts)
                pixel_x_list.append(px.x)
                pixel_y_list.append(px.y)
                pixel_toa_list.append(px.toa)
                pixel_tot_list.append(px.tot)

            # Triggers
            tdc_mask = header == 0x6
            if np.any(tdc_mask):
                tr = self._extract_triggers(packets[tdc_mask], subheader[tdc_mask], min_ts)
                trigger_toa_list.append(tr.toa)
                trigger_tdc_id_list.append(tr.tdc_id)
                trigger_edge_list.append(tr.edge)

        # Concatenate
        pixels = PixelData.empty()
        if pixel_x_list:
            pixels = PixelData(
                x=np.concatenate(pixel_x_list),
                y=np.concatenate(pixel_y_list),
                toa=np.concatenate(pixel_toa_list),
                tot=np.concatenate(pixel_tot_list),
            )

        triggers = TriggerData.empty()
        if trigger_toa_list:
            triggers = TriggerData(
                toa=np.concatenate(trigger_toa_list),
                tdc_id=np.concatenate(trigger_tdc_id_list),
                edge=np.concatenate(trigger_edge_list),
            )

        # Optional timing
        self._chunk_counter += 1
        if self.debug_log_interval > 0 and self._chunk_counter % self.debug_log_interval == 0:
            elapsed = (time.perf_counter() - t_start) * 1000
            self.logger.debug(f"[STD] {len(pixels)} px, {len(triggers)} trig in {elapsed:.2f}ms")

        return pixels, triggers, remaining, (global_min_ts or 0, global_max_ts or 0)

    def _parse_chunks(self, raw_bytes: bytes):
        """Parse SERVAL chunk format."""
        packet_lists = []
        offset = 0
        n_chunks = 0

        while offset + 8 <= len(raw_bytes):
            header_word = np.frombuffer(raw_bytes[offset:offset + 8], dtype=np.uint64)[0]
            signature = header_word & 0xFFFFFFFF

            if signature != self.TPX3_SIGNATURE:
                break

            chunk_size = (header_word >> 48) & 0xFFFF
            if chunk_size == 0 or chunk_size % 8 != 0:
                break

            offset += 8
            if offset + chunk_size > len(raw_bytes):
                offset -= 8
                break

            if chunk_size > 40:
                chunk_data = raw_bytes[offset:offset + chunk_size]
                chunk_packets = np.frombuffer(chunk_data, dtype=np.uint64)
                min_timestamp = chunk_packets[-2] & 0x003FFFFFFFFFFFFF
                max_timestamp = chunk_packets[-1] & 0x003FFFFFFFFFFFFF
                data_packets = chunk_packets[1:-3]
                packet_lists.append((data_packets, (min_timestamp, max_timestamp)))
                n_chunks += 1

            offset += chunk_size

        remaining = raw_bytes[offset:] if offset < len(raw_bytes) else b''
        return (packet_lists, n_chunks, remaining) if packet_lists else (None, n_chunks, remaining)

    def _extract_pixels(self, pixel_packets: np.ndarray, min_timestamp: int) -> PixelData:
        """Extract pixel data with timestamp extension."""
        dcol = ((pixel_packets & 0x0FE0000000000000) >> 53).astype(np.uint16)
        spix = ((pixel_packets & 0x001F800000000000) >> 47).astype(np.uint16)
        pix = ((pixel_packets & 0x0000700000000000) >> 44).astype(np.uint8)

        x = dcol * 2 + pix // 4
        y = spix * 4 + (pix & 0x3)

        data = (pixel_packets & 0x00000FFFFFFF0000) >> 16
        spidr_time = (pixel_packets & 0x000000000000FFFF).astype(np.uint64)
        toa_coarse = ((data & 0x0FFFC000) >> 14).astype(np.uint64)
        ftoa = (data & 0xF).astype(np.uint64)
        tot = ((data & 0x00003FF0) >> 4).astype(np.uint32) * 25

        toa = ((((spidr_time << 14) + toa_coarse) << 4) - ftoa)
        toa = self._extend_timestamp(toa, min_timestamp, n_bits=34)
        toa = toa.astype(np.float64) * 1.5625e-9

        return PixelData(x=x, y=y, toa=toa, tot=tot)

    def _extract_triggers(
        self,
        tdc_packets: np.ndarray,
        subheaders: np.ndarray,
        min_timestamp: int
    ) -> TriggerData:
        """Extract trigger data with timestamp extension."""
        coarse_time = ((tdc_packets >> 9) & 0x7FFFFFFFF).astype(np.uint64) * 2
        tmp_fine = ((tdc_packets >> 5) & 0xF).astype(np.uint64)

        tdc_extended = self._extend_timestamp(coarse_time, min_timestamp, n_bits=36)
        tdc_time = (tdc_extended * 6 + tmp_fine - 1).astype(np.float64) * 260.41666e-12

        tdc_id = np.zeros(len(tdc_packets), dtype=np.uint8)
        edge = np.zeros(len(tdc_packets), dtype=np.uint8)

        tdc_id[(subheaders == self.TDC1_RISING) | (subheaders == self.TDC1_FALLING)] = 1
        tdc_id[(subheaders == self.TDC2_RISING) | (subheaders == self.TDC2_FALLING)] = 2
        edge[(subheaders == self.TDC1_FALLING) | (subheaders == self.TDC2_FALLING)] = 1

        return TriggerData(toa=tdc_time, tdc_id=tdc_id, edge=edge)

    @staticmethod
    def _extend_timestamp(timestamp: np.ndarray, min_timestamp: int, n_bits: int) -> np.ndarray:
        """Extend truncated timestamp using chunk's minimum timestamp."""
        bit_mask = (1 << n_bits) - 1
        half_range = 1 << (n_bits - 1)
        min_ts_truncated = min_timestamp & bit_mask

        delta_t = (timestamp - min_ts_truncated) & bit_mask
        delta_signed = delta_t.astype(np.int64)
        delta_signed = np.where(delta_t >= half_range, delta_signed - (bit_mask + 1), delta_signed)

        return (min_timestamp + delta_signed).astype(np.uint64)


class TPX3Correlator:
    """
    Correlates pixels to triggers using a global sorted trigger list.

    Use after merging triggers from all workers.
    """

    def __init__(
        self,
        triggers: TriggerData,
        event_window: Tuple[float, float] = (0.0, 10_000.0),
        tdc_id: int = 1,
        edge: int = 0,
    ):
        """
        Initialize correlator.

        Parameters
        ----------
        triggers : TriggerData
            Trigger data (will be filtered and sorted)
        event_window : tuple[float, float]
            Valid ToF window in nanoseconds
        tdc_id : int
            TDC to use (1, 2, or 0 for both)
        edge : int
            Edge to use (0=rising, 1=falling)
        """
        self.event_window_min = event_window[0] * 1e-9
        self.event_window_max = event_window[1] * 1e-9
        self.logger = get_logger('SERVAL.Correlator')

        self.triggers = self._prepare_triggers(triggers, tdc_id, edge)
        self.logger.info(
            f"Initialized with {len(self.triggers)} triggers, "
            f"window=[{event_window[0]:.1f}, {event_window[1]:.1f}] ns"
        )

    def _prepare_triggers(self, triggers: TriggerData, tdc_id: int, edge: int) -> np.ndarray:
        """Filter and sort triggers."""
        if len(triggers) == 0:
            return np.array([], dtype=np.float64)

        mask = np.ones(len(triggers), dtype=bool)
        if tdc_id != 0:
            mask &= triggers.tdc_id == tdc_id
        mask &= triggers.edge == edge

        return np.sort(triggers.toa[mask])

    def correlate(self, pixels: PixelData) -> Optional[Tuple[np.ndarray, ...]]:
        """
        Correlate pixels to triggers.

        Parameters
        ----------
        pixels : PixelData
            Pixel data to correlate

        Returns
        -------
        tuple or None
            (event_num, x, y, tof, tot) or None if no events found
        """
        if len(pixels) == 0 or len(self.triggers) < 2:
            return None

        event_indices = np.digitize(pixels.toa, self.triggers) - 1
        valid_mask = (event_indices >= 0) & (event_indices < len(self.triggers))

        if not np.any(valid_mask):
            return None

        event_indices = event_indices[valid_mask]
        x = pixels.x[valid_mask]
        y = pixels.y[valid_mask]
        toa = pixels.toa[valid_mask]
        tot = pixels.tot[valid_mask]

        trigger_times = self.triggers[event_indices]
        tof = toa - trigger_times

        window_mask = (tof >= self.event_window_min) & (tof <= self.event_window_max)
        if not np.any(window_mask):
            return None

        return (
            event_indices[window_mask].astype(np.uint64),
            x[window_mask],
            y[window_mask],
            tof[window_mask],
            tot[window_mask],
        )
