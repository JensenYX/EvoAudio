from __future__ import annotations

import os
from typing import Any

import numpy as np

from .clip import SAMPLE_RATE, Clip


def _librosa():
    os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/evoaudio-numba/{os.getpid()}")
    import librosa

    return librosa


def segment_wave(clip: Clip, start: float | None = None, end: float | None = None) -> np.ndarray:
    s = 0 if start is None else int(max(0.0, start) * SAMPLE_RATE)
    e = clip.wave.size if end is None else int(min(clip.duration, end) * SAMPLE_RATE)
    return clip.wave[s:e]


def duration(clip: Clip) -> float:
    return clip.duration


def rms_dbfs(wave: np.ndarray) -> float:
    if wave.size == 0:
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(wave.astype(np.float64) ** 2) + 1e-12) + 1e-12))


def active_span(wave: np.ndarray, threshold_db: float = -40.0) -> tuple[float, float]:
    frame = int(SAMPLE_RATE * 0.02)
    if wave.size < frame * 2:
        return (0.0, wave.size / SAMPLE_RATE)
    frames = wave[: wave.size // frame * frame].reshape(-1, frame)
    db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1) + 1e-12) + 1e-09)
    active = np.where(db > threshold_db)[0]
    if active.size == 0:
        return (0.0, wave.size / SAMPLE_RATE)
    return (active[0] * frame / SAMPLE_RATE, min(wave.size, (active[-1] + 1) * frame) / SAMPLE_RATE)


def speech_rate(wave: np.ndarray, units: int) -> float:
    s, e = active_span(wave)
    return units / max(0.2, e - s)


def f0_track(wave: np.ndarray, fmin: float = 60.0, fmax: float = 500.0) -> np.ndarray:
    librosa = _librosa()
    if wave.size < 2048:
        return np.array([], dtype=np.float32)
    f0, _, _ = librosa.pyin(
        wave.astype(np.float32), fmin=fmin, fmax=fmax, sr=SAMPLE_RATE, frame_length=1024, hop_length=256
    )
    return f0[np.isfinite(f0)]


def f0_median(wave: np.ndarray) -> float | None:
    voiced = f0_track(wave)
    if voiced.size < 8:
        return None
    return float(np.median(voiced))


def f0_contour(wave: np.ndarray) -> dict[str, Any]:
    voiced = f0_track(wave)
    out: dict[str, Any] = {"voiced_frames": int(voiced.size)}
    if voiced.size < 12:
        out["shape"] = None
        return out
    med = float(np.median(voiced))
    k = max(4, voiced.size * 3 // 10)
    head, tail = (voiced[:k], voiced[-k:])
    out["tail_ratio"] = round(float(np.median(tail) / max(1e-06, med)), 4)
    out["head_ratio"] = round(float(np.median(head) / max(1e-06, med)), 4)
    half = voiced.size // 2
    x1 = np.arange(half)
    x2 = np.arange(voiced.size - half)
    s1 = float(np.polyfit(x1, voiced[:half], 1)[0]) if half >= 3 else 0.0
    s2 = float(np.polyfit(x2, voiced[half:], 1)[0]) if voiced.size - half >= 3 else 0.0
    frames_per_sec = SAMPLE_RATE / 256
    out["slope_first_st_per_s"] = (
        round(12 * np.log2(max(1e-06, med + s1 * frames_per_sec) / med), 3) if med > 0 else 0.0
    )
    out["slope_second_st_per_s"] = (
        round(12 * np.log2(max(1e-06, med + s2 * frames_per_sec) / med), 3) if med > 0 else 0.0
    )
    peak_pos = float(np.argmax(voiced)) / voiced.size
    trough_pos = float(np.argmin(voiced)) / voiced.size
    out["peak_pos"] = round(peak_pos, 3)
    out["trough_pos"] = round(trough_pos, 3)
    tr, hr = (out["tail_ratio"], out["head_ratio"])
    sf, ss = (out["slope_first_st_per_s"], out["slope_second_st_per_s"])
    if sf > 0.8 and ss < -0.8:
        shape = "rise_fall"
    elif sf < -0.8 and ss > 0.8:
        shape = "fall_rise"
    elif tr >= 1.06 and ss > 0.3:
        shape = "rising"
    elif tr <= 0.94 and ss < -0.3:
        shape = "falling"
    elif abs(tr - 1) < 0.05 and abs(hr - 1) < 0.05:
        shape = "flat"
    else:
        shape = "rising" if tr > 1.03 else "falling" if tr < 0.97 else "flat"
    out["shape"] = shape
    return out


def onset_count(wave: np.ndarray, *, frame_sec: float = 0.02, min_gap_sec: float = 0.25) -> int:
    frame = max(1, int(SAMPLE_RATE * frame_sec))
    n = wave.size // frame
    if n < 3:
        return 1 if np.max(np.abs(wave)) > 0.001 else 0
    frames = wave[: n * frame].reshape(n, frame)
    energy = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-09)
    peak = float(np.max(energy))
    if peak < -50:
        return 0
    active = energy > peak - 18.0
    onsets, last_on, in_burst, off_run = (0, -(10**9), False, 0)
    gap_frames = int(min_gap_sec / frame_sec)
    for index, flag in enumerate(active):
        if flag:
            if not in_burst and index - last_on >= gap_frames:
                onsets += 1
                last_on = index
            in_burst = True
            off_run = 0
        else:
            off_run += 1
            if off_run >= max(2, gap_frames // 2):
                in_burst = False
    return onsets


def silence_gaps(
    wave: np.ndarray, *, threshold_db: float = -45.0, min_gap: float = 0.25
) -> list[tuple[float, float]]:
    frame = int(SAMPLE_RATE * 0.01)
    n = wave.size // frame
    if n < 3:
        return []
    frames = wave[: n * frame].reshape(n, frame)
    db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1) + 1e-12) + 1e-09)
    silent = db < threshold_db
    gaps: list[tuple[float, float]] = []
    start = None
    for i, flag in enumerate(silent):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if (i - start) * frame / SAMPLE_RATE >= min_gap and start > 0:
                gaps.append((start * frame / SAMPLE_RATE, i * frame / SAMPLE_RATE))
            start = None
    return gaps


def tempo_bpm(wave: np.ndarray) -> float | None:
    librosa = _librosa()
    if wave.size < SAMPLE_RATE * 2:
        return None
    try:
        tempo = librosa.feature.tempo(y=wave.astype(np.float32), sr=SAMPLE_RATE, aggregate=None)
        tempo = np.asarray(tempo).ravel()
        return float(np.median(tempo)) if tempo.size else None
    except Exception:
        try:
            t, _ = librosa.beat.beat_track(y=wave.astype(np.float32), sr=SAMPLE_RATE)
            return float(np.asarray(t).ravel()[0])
        except Exception:
            return None


def spectral_centroid(wave: np.ndarray) -> float:
    librosa = _librosa()
    c = librosa.feature.spectral_centroid(y=wave.astype(np.float32), sr=SAMPLE_RATE)
    return float(np.median(c)) if c.size else 0.0
