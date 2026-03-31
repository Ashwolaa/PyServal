#!/usr/bin/env python3
"""
JIT-compiled functions for TPX3 data processing.

These functions use Numba for high-performance extraction and correlation.
Falls back to pure Python if Numba is not available.
"""

from typing import Tuple
import numpy as np

# Try to import numba for JIT compilation
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback decorators that do nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args or not callable(args[0]) else args[0]

    prange = range


# =============================================================================
# Packet Classification
# =============================================================================

@njit(cache=True)
def classify_packets(packets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify packets by type (pixel or TDC).

    Parameters
    ----------
    packets : np.ndarray[uint64]
        Raw 64-bit packets

    Returns
    -------
    is_pixel : np.ndarray[bool]
        True for pixel packets (header 0xA or 0xB)
    is_tdc : np.ndarray[bool]
        True for TDC packets (header 0x6)
    subheaders : np.ndarray[uint8]
        Subheader values for all packets
    """
    n = len(packets)
    is_pixel = np.empty(n, dtype=np.bool_)
    is_tdc = np.empty(n, dtype=np.bool_)
    subheaders = np.empty(n, dtype=np.uint8)

    for i in range(n):
        header = (packets[i] >> 60) & 0xF
        subheaders[i] = (packets[i] >> 56) & 0xF
        is_pixel[i] = (header == 0xA) or (header == 0xB)
        is_tdc[i] = (header == 0x6)

    return is_pixel, is_tdc, subheaders


# =============================================================================
# Chunk Parsing
# =============================================================================

@njit(cache=True)
def parse_chunks(words: np.ndarray, tpx3_signature: np.uint32) -> Tuple[
    np.ndarray, np.ndarray, int, np.uint64, np.uint64, int
]:
    """
    Parse TPX3 chunks from raw uint64 words.

    Parameters
    ----------
    words : np.ndarray[uint64]
        Raw data as uint64 array
    tpx3_signature : uint32
        TPX3 signature (0x33585054)

    Returns
    -------
    packet_data : np.ndarray[uint64]
        All data packets
    packet_min_ts : np.ndarray[uint64]
        Min timestamp for each packet
    n_packets : int
        Number of valid packets
    chunk_min_ts : uint64
        Global minimum timestamp
    chunk_max_ts : uint64
        Global maximum timestamp
    remaining_offset : int
        Word offset where remaining data starts
    """
    n_words_total = len(words)

    # First pass: count packets to pre-allocate
    word_offset = 0
    total_packets = 0

    while word_offset < n_words_total:
        header_word = words[word_offset]
        signature = header_word & 0xFFFFFFFF

        if signature != tpx3_signature:
            break

        chunk_size_bytes = (header_word >> 48) & 0xFFFF
        if chunk_size_bytes == 0 or chunk_size_bytes % 8 != 0:
            break

        chunk_size_words = chunk_size_bytes // 8
        word_offset += 1

        if word_offset + chunk_size_words > n_words_total:
            word_offset -= 1
            break

        if chunk_size_words > 5:
            n_data_packets = chunk_size_words - 4
            total_packets += n_data_packets

        word_offset += chunk_size_words

    # Pre-allocate output arrays
    packet_data = np.empty(total_packets, dtype=np.uint64)
    packet_min_ts = np.empty(total_packets, dtype=np.uint64)

    # Second pass: extract data
    word_offset = 0
    packet_idx = 0
    global_min_ts = np.uint64(0xFFFFFFFFFFFFFFFF)
    global_max_ts = np.uint64(0)

    while word_offset < n_words_total:
        header_word = words[word_offset]
        signature = header_word & 0xFFFFFFFF

        if signature != tpx3_signature:
            break

        chunk_size_bytes = (header_word >> 48) & 0xFFFF
        if chunk_size_bytes == 0 or chunk_size_bytes % 8 != 0:
            break

        chunk_size_words = chunk_size_bytes // 8
        word_offset += 1

        if word_offset + chunk_size_words > n_words_total:
            word_offset -= 1
            break

        if chunk_size_words > 5:
            chunk_end = word_offset + chunk_size_words
            min_ts = words[chunk_end - 2] & 0x003FFFFFFFFFFFFF
            max_ts = words[chunk_end - 1] & 0x003FFFFFFFFFFFFF

            if min_ts < global_min_ts:
                global_min_ts = min_ts
            if max_ts > global_max_ts:
                global_max_ts = max_ts

            n_data = chunk_size_words - 4
            data_start = word_offset + 1

            for i in range(n_data):
                packet_data[packet_idx] = words[data_start + i]
                packet_min_ts[packet_idx] = min_ts
                packet_idx += 1

        word_offset += chunk_size_words

    return packet_data, packet_min_ts, packet_idx, global_min_ts, global_max_ts, word_offset


# =============================================================================
# Pixel Extraction
# =============================================================================

@njit(cache=True)
def extract_pixels(
    packets: np.ndarray,
    min_timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract pixel data from packets with per-packet timestamps.

    Parameters
    ----------
    packets : np.ndarray[uint64]
        Pixel packets
    min_timestamps : np.ndarray[uint64]
        Min timestamp for each packet

    Returns
    -------
    x : np.ndarray[uint16]
        X coordinates
    y : np.ndarray[uint16]
        Y coordinates
    toa : np.ndarray[uint64]
        Time of arrival (raw units)
    tot : np.ndarray[uint32]
        Time over threshold (ns)
    """
    n = len(packets)
    x = np.empty(n, dtype=np.uint16)
    y = np.empty(n, dtype=np.uint16)
    toa = np.empty(n, dtype=np.uint64)
    tot = np.empty(n, dtype=np.uint32)

    bit_mask_34 = np.uint64((1 << 34) - 1)
    half_range_34 = np.uint64(1 << 33)

    for i in range(n):
        p = packets[i]
        min_ts = min_timestamps[i]

        # Bit field extraction
        dcol = (p >> 53) & 0x7F
        spix = (p >> 47) & 0x3F
        pix = (p >> 44) & 0x7

        x[i] = dcol * 2 + pix // 4
        y[i] = spix * 4 + (pix & 0x3)

        # Timing extraction
        data = (p >> 16) & 0x0FFFFFFF
        spidr_time = p & 0xFFFF
        toa_coarse = (data >> 14) & 0x3FFF
        ftoa = data & 0xF
        tot[i] = ((data >> 4) & 0x3FF) * 25

        # Combine to 34-bit timestamp
        raw_toa = (((spidr_time << 14) + toa_coarse) << 4) - ftoa

        # Extend timestamp
        min_ts_truncated = min_ts & bit_mask_34
        delta_t = (raw_toa - min_ts_truncated) & bit_mask_34
        if delta_t >= half_range_34:
            delta_signed = np.int64(delta_t) - np.int64(bit_mask_34 + 1)
        else:
            delta_signed = np.int64(delta_t)
        toa[i] = np.uint64(np.int64(min_ts) + delta_signed)

    return x, y, toa, tot


# =============================================================================
# Trigger Extraction
# =============================================================================

# TDC subheader constants
TDC1_RISING = 0xF
TDC1_FALLING = 0xA
TDC2_RISING = 0xE
TDC2_FALLING = 0xB


@njit(cache=True)
def extract_triggers(
    packets: np.ndarray,
    subheaders: np.ndarray,
    min_timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract trigger data from TDC packets.

    Parameters
    ----------
    packets : np.ndarray[uint64]
        TDC packets
    subheaders : np.ndarray[uint8]
        Subheader values
    min_timestamps : np.ndarray[uint64]
        Min timestamp for each packet

    Returns
    -------
    toa : np.ndarray[uint64]
        Time of arrival (raw units)
    tdc_id : np.ndarray[uint8]
        TDC identifier (1 or 2)
    edge : np.ndarray[uint8]
        Edge type (0=rising, 1=falling)
    """
    n = len(packets)
    toa = np.empty(n, dtype=np.uint64)
    tdc_id = np.empty(n, dtype=np.uint8)
    edge = np.empty(n, dtype=np.uint8)

    bit_mask_36 = np.uint64((1 << 36) - 1)
    half_range_36 = np.uint64(1 << 35)

    for i in range(n):
        p = packets[i]
        sh = subheaders[i]
        min_ts = min_timestamps[i]

        # Coarse time (35 bits at 3.125ns) -> convert to 1.5625ns units
        coarse_time = ((p >> 9) & 0x7FFFFFFFF) * 2
        tmp_fine = (p >> 5) & 0xF

        # Extend timestamp
        min_ts_truncated = min_ts & bit_mask_36
        delta_t = (coarse_time - min_ts_truncated) & bit_mask_36
        if delta_t >= half_range_36:
            delta_signed = np.int64(delta_t) - np.int64(bit_mask_36 + 1)
        else:
            delta_signed = np.int64(delta_t)
        tdc_extended = np.uint64(np.int64(min_ts) + delta_signed)

        # Combine with fine time
        toa[i] = tdc_extended * 6 + tmp_fine - 1

        # Classify by TDC and edge
        if sh == TDC1_RISING:
            tdc_id[i] = 1
            edge[i] = 0
        elif sh == TDC1_FALLING:
            tdc_id[i] = 1
            edge[i] = 1
        elif sh == TDC2_RISING:
            tdc_id[i] = 2
            edge[i] = 0
        elif sh == TDC2_FALLING:
            tdc_id[i] = 2
            edge[i] = 1
        else:
            tdc_id[i] = 0
            edge[i] = 0

    return toa, tdc_id, edge


# =============================================================================
# Correlation
# =============================================================================

@njit(cache=True)
def correlate_pixels(
    pixel_toa: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    pixel_tot: np.ndarray,
    trigger_times: np.ndarray,
    event_window_min: float,
    event_window_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Correlate pixels to triggers using binary search.

    Parameters
    ----------
    pixel_toa : np.ndarray[float64]
        Pixel times of arrival
    pixel_x, pixel_y : np.ndarray[uint16]
        Pixel coordinates
    pixel_tot : np.ndarray[uint32]
        Time over threshold
    trigger_times : np.ndarray[float64]
        Sorted trigger times
    event_window_min, event_window_max : float
        Time window for valid events (seconds)

    Returns
    -------
    t_trigger : np.ndarray[float64]
        Absolute trigger time for each event (seconds)
    x, y : np.ndarray[uint16]
        Pixel coordinates
    tof : np.ndarray[float64]
        Time of flight
    tot : np.ndarray[uint32]
        Time over threshold
    n_valid : int
        Number of valid events
    """
    n_pixels = len(pixel_toa)
    n_triggers = len(trigger_times)

    # Pre-allocate output arrays
    out_t_trigger = np.empty(n_pixels, dtype=np.float64)
    out_x = np.empty(n_pixels, dtype=np.uint16)
    out_y = np.empty(n_pixels, dtype=np.uint16)
    out_tof = np.empty(n_pixels, dtype=np.float64)
    out_tot = np.empty(n_pixels, dtype=np.uint32)

    n_valid = 0

    for i in range(n_pixels):
        toa = pixel_toa[i]

        # Binary search for rightmost trigger <= toa
        lo = 0
        hi = n_triggers
        while lo < hi:
            mid = (lo + hi) >> 1
            if trigger_times[mid] <= toa:
                lo = mid + 1
            else:
                hi = mid
        event_idx = lo - 1

        # Validity check
        if event_idx < 0 or event_idx >= n_triggers - 1:
            continue

        # Calculate ToF
        tof = toa - trigger_times[event_idx]

        # Window filter
        if tof < event_window_min or tof > event_window_max:
            continue

        # Store valid event
        out_t_trigger[n_valid] = trigger_times[event_idx]
        out_x[n_valid] = pixel_x[i]
        out_y[n_valid] = pixel_y[i]
        out_tof[n_valid] = tof
        out_tot[n_valid] = pixel_tot[i]
        n_valid += 1

    # Trim to actual count
    return (
        out_t_trigger[:n_valid],
        out_x[:n_valid],
        out_y[:n_valid],
        out_tof[:n_valid],
        out_tot[:n_valid],
        n_valid
    )


# =============================================================================
# Greedy Centroiding
# =============================================================================

@njit(cache=True)
def centroid_hits(
    x: np.ndarray,
    y: np.ndarray,
    toa: np.ndarray,
    tot: np.ndarray,
    eps_space: int,
    eps_time: float,
    b_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Greedy linear-time centroiding of raw pixel hits.

    Processes hits sorted by ToA and maintains a lookback buffer of active
    clusters. Each hit is merged into the nearest spatial neighbour within
    eps_time, or starts a new cluster.  The representative position of a
    cluster is the hit with the maximum ToT; the cluster ToA is the minimum
    ToA across all member hits.

    Parameters
    ----------
    x, y : np.ndarray[uint16]
        Pixel coordinates (unsorted; sorted internally by ToA).
    toa : np.ndarray[float64]
        Time of arrival in seconds.
    tot : np.ndarray[uint32]
        Time over threshold.
    eps_space : int
        Maximum Manhattan distance (pixels) for two hits to belong to the
        same cluster.
    eps_time : float
        Maximum ToA spread (seconds) within a cluster.
    b_size : int
        Lookback buffer depth — maximum number of simultaneously open
        clusters.

    Returns
    -------
    out_x, out_y : np.ndarray[uint16]
        Centroid coordinates (position of max-ToT hit).
    out_toa : np.ndarray[float64]
        Centroid ToA (minimum ToA in cluster), seconds.
    out_tot : np.ndarray[uint32]
        Maximum ToT in cluster.
    n_out : int
        Number of centroids produced.
    """
    n = len(x)
    if n == 0:
        return (
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.uint32),
            0,
        )

    # Sort by ToA
    sort_idx = np.argsort(toa)

    # Cluster buffer (int32 positions to avoid uint16 underflow in abs-diff)
    buf_x = np.empty(b_size, dtype=np.int32)
    buf_y = np.empty(b_size, dtype=np.int32)
    buf_toa_min = np.empty(b_size, dtype=np.float64)
    buf_tot_max = np.empty(b_size, dtype=np.uint32)
    buf_active = np.zeros(b_size, dtype=np.bool_)

    # Output arrays (worst case: every hit is its own cluster)
    out_x = np.empty(n, dtype=np.uint16)
    out_y = np.empty(n, dtype=np.uint16)
    out_toa = np.empty(n, dtype=np.float64)
    out_tot = np.empty(n, dtype=np.uint32)
    n_out = 0
    n_active = 0

    for idx in range(n):
        i = sort_idx[idx]
        xi = np.int32(x[i])
        yi = np.int32(y[i])
        toai = toa[i]
        toti = tot[i]

        # Flush expired clusters (temporal gap > eps_time)
        for k in range(b_size):
            if buf_active[k] and toai - buf_toa_min[k] > eps_time:
                out_x[n_out] = np.uint16(buf_x[k])
                out_y[n_out] = np.uint16(buf_y[k])
                out_toa[n_out] = buf_toa_min[k]
                out_tot[n_out] = buf_tot_max[k]
                n_out += 1
                buf_active[k] = False
                n_active -= 1

        # Find closest active cluster within spatial threshold
        best_k = -1
        best_dist = eps_space + 1
        for k in range(b_size):
            if not buf_active[k]:
                continue
            dist = abs(xi - buf_x[k]) + abs(yi - buf_y[k])
            if dist <= eps_space and dist < best_dist:
                best_dist = dist
                best_k = k

        if best_k >= 0:
            # Merge: update max-ToT position; toa_min stays (sorted order)
            if toti > buf_tot_max[best_k]:
                buf_tot_max[best_k] = toti
                buf_x[best_k] = xi
                buf_y[best_k] = yi
        else:
            # Find a free buffer slot
            free_k = -1
            for k in range(b_size):
                if not buf_active[k]:
                    free_k = k
                    break

            if free_k < 0:
                # Buffer full: evict and flush the oldest cluster
                oldest_k = 0
                oldest_toa = np.float64(1e300)
                for k in range(b_size):
                    if buf_active[k] and buf_toa_min[k] < oldest_toa:
                        oldest_k = k
                        oldest_toa = buf_toa_min[k]
                out_x[n_out] = np.uint16(buf_x[oldest_k])
                out_y[n_out] = np.uint16(buf_y[oldest_k])
                out_toa[n_out] = buf_toa_min[oldest_k]
                out_tot[n_out] = buf_tot_max[oldest_k]
                n_out += 1
                buf_active[oldest_k] = False
                n_active -= 1
                free_k = oldest_k

            # Open a new cluster
            buf_x[free_k] = xi
            buf_y[free_k] = yi
            buf_toa_min[free_k] = toai
            buf_tot_max[free_k] = toti
            buf_active[free_k] = True
            n_active += 1

    # Flush all remaining active clusters
    for k in range(b_size):
        if buf_active[k]:
            out_x[n_out] = np.uint16(buf_x[k])
            out_y[n_out] = np.uint16(buf_y[k])
            out_toa[n_out] = buf_toa_min[k]
            out_tot[n_out] = buf_tot_max[k]
            n_out += 1

    return out_x[:n_out], out_y[:n_out], out_toa[:n_out], out_tot[:n_out], n_out


@njit(cache=True, parallel=True)
def correlate_pixels_parallel(
    pixel_toa: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    pixel_tot: np.ndarray,
    trigger_times: np.ndarray,
    event_window_min: float,
    event_window_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Parallel version of correlate_pixels for large datasets.

    Same parameters and return values as correlate_pixels.
    """
    n_pixels = len(pixel_toa)
    n_triggers = len(trigger_times)

    # First pass: compute validity in parallel
    event_indices = np.empty(n_pixels, dtype=np.int64)
    tof_values = np.empty(n_pixels, dtype=np.float64)
    valid_mask = np.zeros(n_pixels, dtype=np.bool_)

    for i in prange(n_pixels):
        toa = pixel_toa[i]

        # Binary search
        lo = 0
        hi = n_triggers
        while lo < hi:
            mid = (lo + hi) >> 1
            if trigger_times[mid] <= toa:
                lo = mid + 1
            else:
                hi = mid
        event_idx = lo - 1
        event_indices[i] = event_idx

        if event_idx >= 0 and event_idx < n_triggers - 1:
            tof = toa - trigger_times[event_idx]
            tof_values[i] = tof
            if tof >= event_window_min and tof <= event_window_max:
                valid_mask[i] = True

    # Count valid
    n_valid = 0
    for i in range(n_pixels):
        if valid_mask[i]:
            n_valid += 1

    # Compact results
    out_t_trigger = np.empty(n_valid, dtype=np.float64)
    out_x = np.empty(n_valid, dtype=np.uint16)
    out_y = np.empty(n_valid, dtype=np.uint16)
    out_tof = np.empty(n_valid, dtype=np.float64)
    out_tot = np.empty(n_valid, dtype=np.uint32)

    j = 0
    for i in range(n_pixels):
        if valid_mask[i]:
            out_t_trigger[j] = trigger_times[event_indices[i]]
            out_x[j] = pixel_x[i]
            out_y[j] = pixel_y[i]
            out_tof[j] = tof_values[i]
            out_tot[j] = pixel_tot[i]
            j += 1

    return out_t_trigger, out_x, out_y, out_tof, out_tot, n_valid
