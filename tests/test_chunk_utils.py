"""
Tests for SERVAL/core/chunk_utils.py — find_last_pattern and
find_nth_trigger_boundary.
"""

import struct
import numpy as np
import pytest

from SERVAL.core.chunk_utils import find_last_pattern, find_nth_trigger_boundary

# ---------------------------------------------------------------------------
# TPX3 binary helpers
# ---------------------------------------------------------------------------

_TPX3_SIG       = 0x33585054          # "TPX3" little-endian
_TDC1_RISING_SH = 0xF                 # subheader for TDC1 rising edge
_TDC1_FALLING_SH = 0xA

def _tdc_packet(subheader: int) -> int:
    """Minimal 64-bit TDC packet with the given subheader."""
    return (0x6 << 60) | (subheader << 56)

def _pixel_packet() -> int:
    return 0xA000000000000000

def _make_chunk(data_packets: list[int]) -> bytes:
    """
    Build a minimal valid TPX3 chunk.

    Layout (all uint64):
      [0]  header: lower32 = TPX3_SIG, bits48-63 = chunk_size_bytes
      [1]  channel info word (0)
      [2..2+n-1]  data packets
      [-3..-1]  footer (heartbeat + reserved, all 0)
    """
    n_data = len(data_packets)
    # chunk_size_words = 1 (channel) + n_data + 3 (footer)
    csw   = 1 + n_data + 3
    csb   = csw * 8
    header = (csb << 48) | _TPX3_SIG

    words  = [header, 0] + data_packets + [0, 0, 0]
    return struct.pack(f"<{len(words)}Q", *words)


# ---------------------------------------------------------------------------
# find_last_pattern
# ---------------------------------------------------------------------------

class TestFindLastPattern:
    def test_found_at_end(self):
        data = b"garbage" + b"TPX3"
        assert find_last_pattern(data) == 7

    def test_found_in_middle(self):
        # b"TPX3" (4) + b"stuff" (5) = offset 9 for the second "TPX3"
        data = b"TPX3" + b"stuff" + b"TPX3" + b"end"
        assert find_last_pattern(data) == 9

    def test_not_found(self):
        assert find_last_pattern(b"no pattern here") == -1

    def test_empty_buffer(self):
        assert find_last_pattern(b"") == -1

    def test_memoryview_input(self):
        data = bytearray(b"abcTPX3xyz")
        assert find_last_pattern(memoryview(data)) == 3

    def test_custom_pattern(self):
        data = b"hello world hello"
        assert find_last_pattern(data, pattern=b"hello") == 12

    def test_single_occurrence_at_zero(self):
        data = b"TPX3" + b"\x00" * 20
        assert find_last_pattern(data) == 0


# ---------------------------------------------------------------------------
# find_nth_trigger_boundary
# ---------------------------------------------------------------------------

class TestFindNthTriggerBoundary:
    def _stream(self, chunks: list[bytes]) -> bytes:
        return b"".join(chunks)

    def test_returns_minus_one_on_empty(self):
        assert find_nth_trigger_boundary(b"", n=1, rising_subheader=_TDC1_RISING_SH) == -1

    def test_returns_minus_one_when_not_enough_triggers(self):
        # One chunk with 1 rising trigger — asking for n=2 should return -1
        chunk = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        result = find_nth_trigger_boundary(chunk, n=2, rising_subheader=_TDC1_RISING_SH)
        assert result == -1

    def test_single_trigger_chunk_n1_returns_minus_one(self):
        # Only 1 trigger total, n=1: the (n+1)=2nd trigger doesn't exist
        chunk = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        result = find_nth_trigger_boundary(chunk, n=1, rising_subheader=_TDC1_RISING_SH)
        assert result == -1

    def test_two_chunks_n1_returns_second_chunk_offset(self):
        c1 = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        c2 = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        stream = c1 + c2
        result = find_nth_trigger_boundary(stream, n=1, rising_subheader=_TDC1_RISING_SH)
        assert result == len(c1)

    def test_three_chunks_n2_returns_third_chunk_offset(self):
        c1 = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        c2 = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        c3 = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        stream = c1 + c2 + c3
        result = find_nth_trigger_boundary(stream, n=2, rising_subheader=_TDC1_RISING_SH)
        assert result == len(c1) + len(c2)

    def test_falling_edges_ignored(self):
        # 2 falling + 1 rising: with n=1, only 1 rising exists → -1
        c1 = _make_chunk([_tdc_packet(_TDC1_FALLING_SH)])
        c2 = _make_chunk([_tdc_packet(_TDC1_FALLING_SH)])
        c3 = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        stream = c1 + c2 + c3
        result = find_nth_trigger_boundary(stream, n=1, rising_subheader=_TDC1_RISING_SH)
        assert result == -1

    def test_pixel_packets_are_ignored(self):
        c1 = _make_chunk([_pixel_packet(), _tdc_packet(_TDC1_RISING_SH)])
        c2 = _make_chunk([_pixel_packet(), _tdc_packet(_TDC1_RISING_SH)])
        stream = c1 + c2
        result = find_nth_trigger_boundary(stream, n=1, rising_subheader=_TDC1_RISING_SH)
        assert result == len(c1)

    def test_multiple_triggers_in_one_chunk(self):
        # 3 rising triggers in one chunk, 1 in the next: n=3 → second chunk
        c1 = _make_chunk([
            _tdc_packet(_TDC1_RISING_SH),
            _tdc_packet(_TDC1_RISING_SH),
            _tdc_packet(_TDC1_RISING_SH),
        ])
        c2 = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        stream = c1 + c2
        result = find_nth_trigger_boundary(stream, n=3, rising_subheader=_TDC1_RISING_SH)
        assert result == len(c1)

    def test_chunk_without_triggers_is_skipped(self):
        c_pixels = _make_chunk([_pixel_packet(), _pixel_packet()])
        c_trig   = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        c_trig2  = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        stream = c_pixels + c_trig + c_trig2
        result = find_nth_trigger_boundary(stream, n=1, rising_subheader=_TDC1_RISING_SH)
        assert result == len(c_pixels) + len(c_trig)

    def test_empty_data_chunk(self):
        # A chunk with no data packets (only footer words)
        c_empty = _make_chunk([])
        c_trig  = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        c_trig2 = _make_chunk([_tdc_packet(_TDC1_RISING_SH)])
        stream  = c_empty + c_trig + c_trig2
        result  = find_nth_trigger_boundary(stream, n=1, rising_subheader=_TDC1_RISING_SH)
        assert result == len(c_empty) + len(c_trig)
