from typing import Tuple, Optional, Dict, List
import numpy as np


def find_last_pattern(view, pattern=b"TPX3") -> int:
    """
    Find the last occurrence of ``pattern`` in ``view``.

    Args:
        view: bytes or memoryview of the data
        pattern: byte sequence to search for

    Returns:
        int: Index of the last occurrence of ``pattern``, or -1 if not found.
    """
    if isinstance(view, memoryview):
        view = bytes(view)
    return view.rfind(pattern)


# TPX3 packet format constants
_TPX3_SIGNATURE  = np.uint32(0x33585054)  # "TPX3" in little-endian
_HEADER_MASK     = np.uint64(0xF000000000000000)
_SUBHEADER_MASK  = np.uint64(0x0F00000000000000)
_TDC_HEADER      = np.uint64(0x6)


def find_nth_trigger_boundary(data: bytes, n: int, rising_subheader: int) -> int:
    """
    Walk the TPX3 chunk structure and find the byte offset of the TPX3 chunk
    that contains the (n+1)-th rising-edge trigger matching ``rising_subheader``.

    Flushing *up to* this offset gives a chunk that contains exactly n rising
    edges (and all pixel data up to just before the next trigger), so each
    chunk sent to a worker starts with a trigger and ends cleanly before the
    next one.

    Parameters
    ----------
    data : bytes
        Raw data buffer (may contain multiple TPX3 chunks).
    n : int
        Number of rising-edge triggers to include in the flushed chunk.
        The flush boundary is placed at the start of the (n+1)-th trigger's
        TPX3 chunk.
    rising_subheader : int
        Subheader value that identifies the desired rising edge:
        0xF = TDC1 rising, 0xE = TDC2 rising.

    Returns
    -------
    int
        Byte offset of the TPX3 chunk containing the (n+1)-th rising edge,
        i.e. the flush point.  Returns -1 if fewer than (n+1) matching
        triggers are found (caller should fall back to size/time flushing).
    """
    if not data:
        return -1

    rs = np.uint64(rising_subheader)
    words = np.frombuffer(data, dtype=np.uint64)
    n_words = len(words)

    trigger_count = 0
    word_offset = 0

    while word_offset < n_words:
        chunk_start_byte = word_offset * 8

        # Validate TPX3 chunk header
        header_word = words[word_offset]
        if (header_word & np.uint64(0xFFFFFFFF)) != np.uint64(_TPX3_SIGNATURE):
            break

        chunk_size_bytes = int((header_word >> np.uint64(48)) & np.uint64(0xFFFF))
        if chunk_size_bytes == 0 or chunk_size_bytes % 8 != 0:
            break

        chunk_size_words = chunk_size_bytes // 8
        next_word_offset = word_offset + 1 + chunk_size_words

        if next_word_offset > n_words:
            break  # Incomplete chunk — stop here

        # Data packets start one word after the header, end 4 words before the
        # chunk end (channel word + 2 heartbeat words + 1 reserved word).
        n_data = chunk_size_words - 4
        if n_data > 0:
            data_words = words[word_offset + 2 : word_offset + 2 + n_data]

            headers    = (data_words & _HEADER_MASK) >> np.uint64(60)
            subheaders = (data_words & _SUBHEADER_MASK) >> np.uint64(56)

            n_triggers_in_chunk = int(np.count_nonzero(
                (headers == _TDC_HEADER) & (subheaders == rs)
            ))

            if trigger_count + n_triggers_in_chunk >= n + 1:
                # The (n+1)-th trigger is inside this chunk.
                # If this is NOT the very first chunk (trigger_count > 0 or
                # we've already accumulated some triggers), flush everything
                # up to the START of this chunk so the worker gets n clean shots.
                if trigger_count >= 1 or chunk_start_byte > 0:
                    return chunk_start_byte
                # Edge case: the (n+1)-th trigger is in the first chunk we've
                # seen — we can't cut before it, so carry on accumulating.

            trigger_count += n_triggers_in_chunk

        word_offset = next_word_offset

    return -1  # Fewer than (n+1) rising edges found