"""The browser's two downsampling paths, for the preprocessing question."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from resample_compare import decimate, filter_response, resample_native  # noqa: E402


def test_decimate_matches_the_javascript_on_an_integer_ratio():
    """mic.ts:110 averages input[floor(i*r) : floor((i+1)*r)]. At r=3, out[0] is the
    mean of samples 0,1,2 -- checked by hand against the JS."""
    got = decimate(np.arange(9, dtype=np.float32), 3.0)
    assert got.tolist() == pytest.approx([1.0, 4.0, 7.0])


def test_decimate_handles_the_non_integer_ratio_windows():
    """44.1 kHz -> 16 kHz is ratio 2.75, so floor boundaries alternate between 2- and
    3-sample windows. This is the harsher case and the one a 44.1 kHz device would hit,
    so the port must reproduce it exactly.

    Boundaries: floor(0*2.75)=0, floor(1*2.75)=2, floor(2*2.75)=5, floor(3*2.75)=8.
    So out[0]=mean(x[0:2]), out[1]=mean(x[2:5]), out[2]=mean(x[5:8])."""
    got = decimate(np.arange(11, dtype=np.float32), 2.75)
    assert len(got) == 4  # floor(11 / 2.75) == 4
    assert got[0] == pytest.approx(np.mean([0, 1]))
    assert got[1] == pytest.approx(np.mean([2, 3, 4]))
    assert got[2] == pytest.approx(np.mean([5, 6, 7]))


def test_decimate_output_length_matches_the_javascript():
    """`new Float32Array(Math.floor(input.length / ratio))`."""
    for n, r in ((100, 3.0), (100, 2.75), (7, 3.0), (0, 3.0)):
        assert len(decimate(np.zeros(n, dtype=np.float32), r)) == int(n // r)


def test_decimate_is_identity_at_ratio_one():
    """mic.ts skips decimate entirely when ratio === 1; the port must agree that there
    is nothing to do, or the comparison would blame it for a difference it never made."""
    x = np.linspace(-1, 1, 64, dtype=np.float32)
    assert decimate(x, 1.0) == pytest.approx(x)


def test_box_average_leaks_above_its_passband():
    """A rectangular window is a sinc in frequency: its first sidelobe sits near
    -13 dB, so energy well above the new Nyquist is attenuated but NOT removed. This
    is the whole reason the resampler choice can matter for fricatives."""
    _, mag_db = filter_response(3.0)
    assert mag_db[len(mag_db) // 2 :].max() > -25.0


def test_box_average_passes_energy_that_must_be_removed():
    """THE defect, stated as the thing it actually costs.

    A 10 kHz tone at 48 kHz is above the 8 kHz destination Nyquist, so a correct
    resampler must remove it -- otherwise it folds to |10000 - 16000| = 6 kHz and
    lands in the middle of the fricative band as noise that was never uttered.

    The 3-tap box average passes it at H = sin(3*pi*f)/(3*sin(pi*f)) with
    f = 10000/48000 = 0.2083, i.e. 0.506 -- half of it survives, and aliases.
    """
    src, dst = 48000, 16000
    t = np.arange(src) / src
    tone = np.sin(2 * np.pi * 10000 * t).astype(np.float32)

    box = decimate(tone, src / dst)
    poly = resample_native(tone, src, dst)

    box_level = float(np.abs(box).mean())
    poly_level = float(np.abs(poly).mean())
    assert box_level > 0.2, "the box average should pass roughly half of it"
    assert poly_level < 0.01, "a real anti-alias filter should remove it"
    assert box_level > 20 * poly_level


def test_box_average_aliases_that_energy_into_the_fricative_band():
    """Not just 'it survives' but 'it lands where the fricatives are'. 10 kHz at
    48 kHz folds to 6 kHz at 16 kHz, inside the 4-8 kHz band."""
    from probe_stream import band_energy

    src, dst = 48000, 16000
    t = np.arange(src) / src
    tone = np.sin(2 * np.pi * 10000 * t).astype(np.float32)

    box = decimate(tone, src / dst)
    freqs = np.fft.rfftfreq(box.size, 1.0 / dst)
    peak_hz = float(freqs[int(np.abs(np.fft.rfft(box)).argmax())])
    assert abs(peak_hz - 6000) < 50, "the survivor must appear at the fold frequency"
    assert band_energy(box, dst) > 100 * band_energy(
        resample_native(tone, src, dst), dst
    )


def test_the_two_filters_agree_well_below_nyquist():
    """The box average is not simply bad everywhere -- at 1 kHz both paths agree
    closely. Reporting it as uniformly destructive would overstate the case."""
    src, dst = 48000, 16000
    t = np.arange(src) / src
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    box = decimate(tone, src / dst)
    poly = resample_native(tone, src, dst)
    n = min(box.size, poly.size)
    assert float(np.corrcoef(box[:n], poly[:n])[0, 1]) > 0.99


def test_resample_native_hits_the_requested_rate():
    src, dst = 44100, 16000
    out = resample_native(np.zeros(src, dtype=np.float32), src, dst)
    assert abs(out.size - dst) <= 1
