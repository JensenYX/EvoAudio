from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from .clip import SAMPLE_RATE, Clip

REF_RMS_DBFS = -22.0
PEAK_LIMIT = 0.89


def _librosa():
    os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/evoaudio-numba/{os.getpid()}")
    import librosa

    return librosa


def rms_dbfs(wave: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(wave.astype(np.float64) ** 2) + 1e-12) + 1e-12))


def fade(clip: Clip, ms: float = 12.0) -> Clip:
    wave = clip.wave.copy()
    n = min(int(SAMPLE_RATE * ms / 1000), wave.size // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        wave[:n] *= ramp
        wave[-n:] *= ramp[::-1]
    out = clip.copy(wave)
    return out


def trim(clip: Clip, start: float = 0.0, end: float | None = None) -> Clip:
    s = int(max(0.0, start) * SAMPLE_RATE)
    e = clip.wave.size if end is None else int(end * SAMPLE_RATE)
    out = clip.copy(np.ascontiguousarray(clip.wave[s:e]))
    out.segments = []
    out.provenance.append({"transform": "trim", "start": start, "end": end})
    return out


def trim_silence(clip: Clip, threshold_db: float = -40.0, pad_ms: float = 30.0) -> Clip:
    wave = clip.wave
    frame = int(SAMPLE_RATE * 0.02)
    if wave.size < frame * 3:
        return clip.copy()
    frames = wave[: wave.size // frame * frame].reshape(-1, frame)
    db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1) + 1e-12) + 1e-09)
    active = np.where(db > threshold_db)[0]
    if active.size == 0:
        return clip.copy()
    pad = int(SAMPLE_RATE * pad_ms / 1000)
    s = max(0, int(active[0] * frame) - pad)
    e = min(wave.size, int((active[-1] + 1) * frame) + pad)
    out = clip.copy(np.ascontiguousarray(wave[s:e]))
    out.segments = []
    out.provenance.append(
        {"transform": "trim_silence", "cut_start": s / SAMPLE_RATE, "cut_end": (wave.size - e) / SAMPLE_RATE}
    )
    return out


def gain_db(clip: Clip, db: float) -> Clip:
    out = clip.copy(clip.wave * np.float32(10 ** (db / 20)))
    out.provenance.append({"transform": "gain_db", "db": db})
    return out


def normalize(clip: Clip, target_dbfs: float = REF_RMS_DBFS) -> Clip:
    current = rms_dbfs(clip.wave)
    if current < -80:
        return clip.copy()
    return gain_db(clip, target_dbfs - current)


def peak_limit(clip: Clip, limit: float = PEAK_LIMIT) -> Clip:
    peak = float(np.max(np.abs(clip.wave)) + 1e-09)
    if peak <= limit:
        return clip.copy()
    out = clip.copy(clip.wave * np.float32(limit / peak))
    out.provenance.append({"transform": "peak_limit", "scale": limit / peak})
    return out


def time_stretch(clip: Clip, factor: float) -> Clip:
    if abs(factor - 1.0) < 0.001:
        return clip.copy()
    librosa = _librosa()
    wave = librosa.effects.time_stretch(clip.wave.astype(np.float32), rate=float(factor)).astype(np.float32)
    out = clip.copy(wave)
    out.segments = [dict(s, start=s["start"] / factor, end=s["end"] / factor) for s in clip.segments]
    out.provenance.append({"transform": "time_stretch", "factor": factor})
    return out


def pitch_shift(clip: Clip, semitones: float) -> Clip:
    if abs(semitones) < 0.001:
        return clip.copy()
    librosa = _librosa()
    wave = librosa.effects.pitch_shift(
        clip.wave.astype(np.float32), sr=SAMPLE_RATE, n_steps=float(semitones)
    ).astype(np.float32)
    out = clip.copy(wave)
    out.provenance.append({"transform": "pitch_shift", "semitones": semitones})
    return out


def ramp_gain(clip: Clip, start_db: float, end_db: float) -> Clip:
    ramp = np.linspace(start_db, end_db, clip.wave.size, dtype=np.float32)
    out = clip.copy(clip.wave * (10 ** (ramp / 20)).astype(np.float32))
    out.provenance.append({"transform": "ramp_gain", "start_db": start_db, "end_db": end_db})
    return out


def insert_silence(clip: Clip, at_sec: float, seconds: float) -> Clip:
    k = int(max(0.0, min(at_sec, clip.duration)) * SAMPLE_RATE)
    gap = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    out = clip.copy(np.concatenate([clip.wave[:k], gap, clip.wave[k:]]))
    out.segments = [
        dict(
            s,
            start=s["start"] + (seconds if s["start"] >= at_sec else 0.0),
            end=s["end"] + (seconds if s["end"] > at_sec else 0.0),
        )
        for s in clip.segments
    ]
    out.provenance.append({"transform": "insert_silence", "at": at_sec, "seconds": seconds})
    return out


def concat(
    parts: Sequence[Clip],
    gap: float = 0.5,
    *,
    gaps: Sequence[float] | None = None,
    roles: Sequence[str] | None = None,
    lead: float = 0.15,
    tail: float = 0.25,
) -> Clip:
    pieces: list[np.ndarray] = [np.zeros(int(lead * SAMPLE_RATE), dtype=np.float32)]
    segments: list[dict[str, Any]] = []
    cursor = lead
    prov: list[dict[str, Any]] = []
    for i, part in enumerate(parts):
        wave = fade(part).wave
        pieces.append(wave)
        seg = {
            "index": i,
            "start": round(cursor, 4),
            "end": round(cursor + wave.size / SAMPLE_RATE, 4),
            "role": roles[i] if roles else "part",
            "truth": dict(part.truth),
            "duration": wave.size / SAMPLE_RATE,
        }
        segments.append(seg)
        prov.append({"part": i, "provenance": part.provenance})
        cursor += wave.size / SAMPLE_RATE
        g = (gaps[i] if gaps and i < len(gaps) else gap) if i < len(parts) - 1 else tail
        pieces.append(np.zeros(int(max(0.0, g) * SAMPLE_RATE), dtype=np.float32))
        cursor += max(0.0, g)
    out = Clip(
        wave=np.concatenate(pieces).astype(np.float32),
        provenance=[{"transform": "concat", "parts": prov, "gap": gap}],
        segments=segments,
    )
    return out


def loop_to(clip: Clip, seconds: float) -> Clip:
    need = int(seconds * SAMPLE_RATE)
    wave = clip.wave
    if wave.size >= need:
        out = clip.copy(np.ascontiguousarray(wave[:need]))
    else:
        reps = need // max(1, wave.size) + 1
        out = clip.copy(np.concatenate([fade(clip).wave] * reps)[:need])
    out.provenance.append({"transform": "loop_to", "seconds": seconds})
    return out


def mix(foreground: Clip, background: Clip, *, snr_db: float, offset: float = 0.0, loop: bool = True) -> Clip:
    fg = foreground.wave
    bg = background.wave
    total = max(fg.size, int(offset * SAMPLE_RATE) + bg.size) if not loop else fg.size
    if loop:
        bg = loop_to(background, total / SAMPLE_RATE).wave
    canvas = np.zeros(total, dtype=np.float32)
    s = int(offset * SAMPLE_RATE)
    e = min(total, s + bg.size)
    bg_rms = rms_dbfs(bg[: e - s])
    fg_rms = rms_dbfs(fg)
    target_bg = fg_rms - snr_db
    scale = 10 ** ((target_bg - bg_rms) / 20) if bg_rms > -80 else 0.0
    canvas[s:e] += bg[: e - s] * np.float32(scale)
    canvas[: fg.size] += fg
    out = Clip(
        wave=canvas,
        provenance=[
            {
                "transform": "mix",
                "snr_db": snr_db,
                "offset": offset,
                "foreground": foreground.provenance,
                "background": background.provenance,
            }
        ],
        truth=dict(foreground.truth),
        segments=[dict(s_) for s_ in foreground.segments],
    )
    out.segments.append(
        {
            "index": len(out.segments),
            "start": round(s / SAMPLE_RATE, 4),
            "end": round(e / SAMPLE_RATE, 4),
            "role": "background",
            "truth": dict(background.truth),
            "duration": (e - s) / SAMPLE_RATE,
        }
    )
    return out


def overlay(events: Sequence[tuple[Clip, float]], *, length: float | None = None, tail: float = 0.35) -> Clip:
    end = max((onset + c.duration for c, onset in events), default=1.0) + tail
    total = max(end, length or 0.0)
    canvas = np.zeros(int(total * SAMPLE_RATE) + 1, dtype=np.float32)
    segments = []
    prov = []
    for i, (clip, onset) in enumerate(events):
        wave = fade(clip).wave
        s = int(onset * SAMPLE_RATE)
        e = min(canvas.size, s + wave.size)
        canvas[s:e] += wave[: e - s]
        segments.append(
            {
                "index": i,
                "start": round(onset, 4),
                "end": round(e / SAMPLE_RATE, 4),
                "role": "event",
                "truth": dict(clip.truth),
                "duration": (e - s) / SAMPLE_RATE,
                "rms_dbfs": rms_dbfs(wave[: e - s]),
            }
        )
        prov.append({"event": i, "onset": onset, "provenance": clip.provenance})
    return Clip(wave=canvas, provenance=[{"transform": "overlay", "events": prov}], segments=segments)


def add_background(clip: Clip, background: Clip, *, snr_db: float) -> Clip:
    return mix(clip, background, snr_db=snr_db, offset=0.0, loop=True)
