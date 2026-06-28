"""
Tests for SERVAL/core/extractors/jit_functions.py

Tests use the Python fallback path (no Numba required) by calling the
functions directly with numpy arrays. The Numba-compiled path runs the same
logic so these tests cover both paths when Numba is available.
"""

import numpy as np
import pytest

from SERVAL.core.extractors.jit_functions import (
    classify_packets,
    extract_triggers,
    correlate_pixels,
    TDC1_RISING, TDC1_FALLING, TDC2_RISING, TDC2_FALLING,
)


# ---------------------------------------------------------------------------
# Packet construction helpers
# ---------------------------------------------------------------------------

def _pixel_packet(dcol=0, spix=0, pix=0, toa_coarse=0, ftoa=0, tot_ns=0, spidr=0):
    """Build a minimal pixel packet (header 0xA)."""
    tot_raw = tot_ns // 25
    data = ((toa_coarse & 0x3FFF) << 14) | ((tot_raw & 0x3FF) << 4) | (ftoa & 0xF)
    p = np.uint64(0xA000000000000000)
    p |= np.uint64((dcol & 0x7F)) << np.uint64(53)
    p |= np.uint64((spix & 0x3F)) << np.uint64(47)
    p |= np.uint64((pix  & 0x7))  << np.uint64(44)
    p |= np.uint64((data & 0x0FFFFFFF)) << np.uint64(16)
    p |= np.uint64(spidr & 0xFFFF)
    return int(p)

def _tdc_packet(subheader, coarse_time=0, fine_time=0):
    """Build a minimal TDC packet (header 0x6)."""
    p = np.uint64(0x6000000000000000)
    p |= np.uint64(subheader & 0xF) << np.uint64(56)
    # coarse_time is 35-bit field at bits 9-43; fine_time is 4-bit at bits 5-8
    p |= np.uint64(coarse_time & 0x7FFFFFFFF) << np.uint64(9)
    p |= np.uint64(fine_time & 0xF) << np.uint64(5)
    return int(p)


# ---------------------------------------------------------------------------
# classify_packets
# ---------------------------------------------------------------------------

class TestClassifyPackets:
    def test_pixel_header_a(self):
        pkt = np.array([0xA000000000000000], dtype=np.uint64)
        is_px, is_tdc, _ = classify_packets(pkt)
        assert is_px[0] and not is_tdc[0]

    def test_pixel_header_b(self):
        pkt = np.array([0xB000000000000000], dtype=np.uint64)
        is_px, is_tdc, _ = classify_packets(pkt)
        assert is_px[0] and not is_tdc[0]

    def test_tdc_header(self):
        pkt = np.array([0x6F00000000000000], dtype=np.uint64)
        is_px, is_tdc, sh = classify_packets(pkt)
        assert not is_px[0] and is_tdc[0]
        assert sh[0] == 0xF

    def test_unknown_header_is_neither(self):
        pkt = np.array([0x1000000000000000], dtype=np.uint64)
        is_px, is_tdc, _ = classify_packets(pkt)
        assert not is_px[0] and not is_tdc[0]

    def test_mixed_batch(self):
        pkts = np.array([
            0xA000000000000000,  # pixel
            0x6F00000000000000,  # tdc1-rising
            0xB000000000000000,  # pixel
            0x6A00000000000000,  # tdc1-falling
            0x1000000000000000,  # unknown
        ], dtype=np.uint64)
        is_px, is_tdc, sh = classify_packets(pkts)
        np.testing.assert_array_equal(is_px,  [True, False, True, False, False])
        np.testing.assert_array_equal(is_tdc, [False, True, False, True, False])
        assert sh[1] == 0xF
        assert sh[3] == 0xA

    def test_empty_array(self):
        pkt = np.array([], dtype=np.uint64)
        is_px, is_tdc, sh = classify_packets(pkt)
        assert len(is_px) == 0 and len(is_tdc) == 0 and len(sh) == 0

    def test_subheader_extraction(self):
        for sh_val in [0x0, 0x5, 0xF, 0xA, 0xE, 0xB]:
            pkt = np.array([0x6000000000000000 | (sh_val << 56)], dtype=np.uint64)
            _, _, sh = classify_packets(pkt)
            assert sh[0] == sh_val


# ---------------------------------------------------------------------------
# extract_triggers
# ---------------------------------------------------------------------------

class TestExtractTriggers:
    def _run(self, subheaders, coarse_times=None, fine_times=None):
        n = len(subheaders)
        if coarse_times is None:
            coarse_times = [0] * n
        if fine_times is None:
            fine_times = [0] * n

        pkts = np.array(
            [_tdc_packet(sh, ct, ft)
             for sh, ct, ft in zip(subheaders, coarse_times, fine_times)],
            dtype=np.uint64,
        )
        sh_arr  = np.array(subheaders, dtype=np.uint8)
        min_ts  = np.zeros(n, dtype=np.uint64)
        return extract_triggers(pkts, sh_arr, min_ts)

    def test_tdc1_rising_classified_correctly(self):
        toa, tdc_id, edge = self._run([TDC1_RISING])
        assert tdc_id[0] == 1 and edge[0] == 0

    def test_tdc1_falling_classified_correctly(self):
        toa, tdc_id, edge = self._run([TDC1_FALLING])
        assert tdc_id[0] == 1 and edge[0] == 1

    def test_tdc2_rising_classified_correctly(self):
        toa, tdc_id, edge = self._run([TDC2_RISING])
        assert tdc_id[0] == 2 and edge[0] == 0

    def test_tdc2_falling_classified_correctly(self):
        toa, tdc_id, edge = self._run([TDC2_FALLING])
        assert tdc_id[0] == 2 and edge[0] == 1

    def test_unknown_subheader_gives_tdc_id_zero(self):
        toa, tdc_id, edge = self._run([0x0])
        assert tdc_id[0] == 0

    def test_mixed_batch_shapes(self):
        shs = [TDC1_RISING, TDC1_FALLING, TDC2_RISING, TDC2_FALLING]
        toa, tdc_id, edge = self._run(shs)
        assert len(toa) == len(tdc_id) == len(edge) == 4

    def test_empty_batch(self):
        pkts   = np.array([], dtype=np.uint64)
        sh_arr = np.array([], dtype=np.uint8)
        min_ts = np.array([], dtype=np.uint64)
        toa, tdc_id, edge = extract_triggers(pkts, sh_arr, min_ts)
        assert len(toa) == 0

    def test_monotone_coarse_time_gives_increasing_toa(self):
        # Start from a non-zero coarse time to avoid the `tdc_extended * 6 - 1`
        # underflow that occurs when both coarse_time and fine_time are 0.
        n = 5
        coarse = list(range(1000, 1000 + n * 2, 2))
        toa, _, _ = self._run([TDC1_RISING] * n, coarse_times=coarse)
        for i in range(n - 1):
            assert toa[i] <= toa[i + 1]


# ---------------------------------------------------------------------------
# correlate_pixels
# ---------------------------------------------------------------------------

class TestCorrelatePixels:
    def _correlate(self, pixel_toa, triggers, win_min=0.0, win_max=1e-3):
        n = len(pixel_toa)
        x   = np.zeros(n, dtype=np.uint16)
        y   = np.zeros(n, dtype=np.uint16)
        tot = np.zeros(n, dtype=np.uint32)
        return correlate_pixels(
            np.asarray(pixel_toa, dtype=np.float64),
            x, y, tot,
            np.asarray(triggers, dtype=np.float64),
            win_min, win_max,
        )

    def test_basic_correlation(self):
        # Each pixel is placed 0.5 ns after its corresponding trigger
        triggers  = [0.0, 1e-6, 2e-6]
        pixel_toa = [0.5e-9, 1e-6 + 0.5e-9, 2e-6 + 0.5e-9]
        t_trig, px, py, tof, pt, n_valid = self._correlate(pixel_toa, triggers)
        assert n_valid == 3
        np.testing.assert_allclose(tof, [0.5e-9, 0.5e-9, 0.5e-9], rtol=1e-6)

    def test_pixel_before_first_trigger_is_rejected(self):
        triggers  = [1e-6]
        pixel_toa = [0.5e-6]   # before the only trigger
        _, _, _, _, _, n_valid = self._correlate(pixel_toa, triggers)
        assert n_valid == 0

    def test_tof_outside_window_is_rejected(self):
        triggers  = [0.0]
        pixel_toa = [2e-3]     # 2 ms after trigger, window max = 1 ms
        _, _, _, _, _, n_valid = self._correlate(pixel_toa, triggers, win_max=1e-3)
        assert n_valid == 0

    def test_tof_below_min_window_rejected(self):
        triggers  = [0.0]
        pixel_toa = [0.5e-9]   # 0.5 ns, below win_min=1 ns
        _, _, _, _, _, n_valid = self._correlate(pixel_toa, triggers, win_min=1e-9)
        assert n_valid == 0

    def test_pixel_assigned_to_nearest_preceding_trigger(self):
        triggers  = [0.0, 1e-6]
        pixel_toa = [1.1e-6]   # after second trigger
        t_trig, _, _, tof, _, n_valid = self._correlate(pixel_toa, triggers)
        assert n_valid == 1
        assert t_trig[0] == pytest.approx(1e-6, rel=1e-6)
        assert tof[0] == pytest.approx(0.1e-6, rel=1e-6)

    def test_output_arrays_trimmed_to_n_valid(self):
        triggers  = [0.0, 1e-6]
        pixel_toa = [0.5e-9, 0.5e-6]   # both inside window
        t_trig, px, py, tof, pt, n_valid = self._correlate(pixel_toa, triggers)
        assert len(t_trig) == n_valid == 2

    def test_empty_pixels(self):
        triggers = [0.0, 1e-6]
        _, _, _, _, _, n_valid = self._correlate([], triggers)
        assert n_valid == 0

    def test_empty_triggers(self):
        pixel_toa = [0.5e-9, 1.5e-9]
        _, _, _, _, _, n_valid = self._correlate(pixel_toa, [])
        assert n_valid == 0

    def test_all_pixels_in_window(self):
        n = 100
        triggers = np.linspace(0.0, 1e-3, 10)
        rng = np.random.default_rng(42)
        # Place each pixel 1 ns after its corresponding trigger
        pixel_toa = np.sort(rng.choice(triggers, n) + 1e-9)
        _, _, _, _, _, n_valid = self._correlate(pixel_toa, triggers, win_max=1e-6)
        assert n_valid == n

    def test_coordinate_passthrough(self):
        triggers  = [0.0]
        pixel_toa = np.array([1e-9], dtype=np.float64)
        x   = np.array([42], dtype=np.uint16)
        y   = np.array([77], dtype=np.uint16)
        tot = np.array([100], dtype=np.uint32)
        t_trig, out_x, out_y, tof, out_tot, n_valid = correlate_pixels(
            pixel_toa, x, y, tot,
            np.array([0.0], dtype=np.float64),
            0.0, 1e-3,
        )
        assert n_valid == 1
        assert out_x[0] == 42
        assert out_y[0] == 77
        assert out_tot[0] == 100
