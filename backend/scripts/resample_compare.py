"""Compare the browser's two downsampling paths, for the preprocessing question.

`mic.ts` asks for `new AudioContext({ sampleRate: 16000 })` and only falls back to its
own `decimate()` when the browser ignores the hint (`ratio !== 1`). Chrome honours it, so
that branch is DEAD CODE today -- it is the Firefox/Safari path. Dropping the hint, i.e.
`new AudioContext()`, therefore does not tweak the resampler: it moves EVERY browser onto
the fallback.

That matters because `decimate` is a box average, and a rectangular window is a sinc in
frequency with roughly -13 dB first sidelobes. Energy above the new Nyquist is attenuated
but not removed, so it folds back into the speech band -- and the unvoiced fricatives
(س ش ص ث) live in exactly the 4-8 kHz range that folds. The box average also droops
badly INSIDE its own passband, which costs high-frequency detail even where nothing
aliases.

44.1 kHz is the harsher case: ratio 2.75, so `Math.floor` boundaries alternate between
2- and 3-sample windows and the effective filter changes shape frame to frame.
"""

from __future__ import annotations

import numpy as np


def decimate(input: np.ndarray, ratio: float) -> np.ndarray:
    """Faithful port of frontend/src/lib/mic.ts:110.

    Deliberately not vectorised: it must reproduce the JS's `Math.floor` window
    boundaries exactly, including the uneven windows a non-integer ratio produces.
    A tidier implementation that averaged fixed-width blocks would be a different
    filter, and the whole point here is to measure THIS one.
    """
    out = np.zeros(int(np.floor(input.size / ratio)), dtype=np.float32)
    for i in range(out.size):
        start = int(np.floor(i * ratio))
        end = min(input.size, int(np.floor((i + 1) * ratio)))
        out[i] = float(np.sum(input[start:end]) / (end - start)) if end > start else 0.0
    return out


def resample_native(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """The reference a browser's native resampler approximates: a polyphase FIR with a
    real anti-alias filter."""
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(src_sr), int(dst_sr))
    return resample_poly(
        audio.astype(np.float64), dst_sr // g, src_sr // g
    ).astype(np.float32)


def filter_response(ratio: float, n: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude response of the box average, peak-normalised, in dB.

    Returns ``(freq, mag_db)`` where freq is in cycles/sample of the INPUT rate, so
    0.5 is the input Nyquist and ``0.5 / ratio`` is the destination Nyquist -- the
    point past which anything surviving will fold back into the speech band.
    """
    taps = np.ones(max(1, int(round(ratio))), dtype=np.float64)
    taps /= taps.sum()
    spectrum = np.fft.rfft(taps, n)
    mag = np.abs(spectrum)
    mag_db = 20 * np.log10(np.maximum(mag / mag.max(), 1e-12))
    return np.fft.rfftfreq(n), mag_db


def compare(path: str, dst_sr: int = 16000) -> dict:
    """Both downsampling paths on one native-rate file, with fricative-band energies.

    ``band_ratio`` below 1 means the box average LOST high-frequency energy relative to
    a correct resampler -- which is the direction that costs unvoiced fricatives.
    """
    import librosa

    from probe_stream import band_energy

    native, src_sr = librosa.load(path, sr=None, mono=True)
    native = native.astype(np.float32)
    ratio = src_sr / dst_sr
    box = decimate(native, ratio)
    poly = resample_native(native, int(src_sr), dst_sr)
    n = min(box.size, poly.size)
    box, poly = box[:n], poly[:n]

    box_band = band_energy(box, dst_sr)
    poly_band = band_energy(poly, dst_sr)
    return {
        "path": path,
        "src_sr": int(src_sr),
        "ratio": round(float(ratio), 4),
        "n_samples": int(n),
        "box_band_4k_8k": box_band,
        "poly_band_4k_8k": poly_band,
        "band_ratio": float(box_band / poly_band) if poly_band else float("nan"),
        "rms_difference": float(np.sqrt(((box - poly) ** 2).mean())),
        "correlation": float(np.corrcoef(box, poly)[0, 1]),
        "box": box,
        "poly": poly,
    }
