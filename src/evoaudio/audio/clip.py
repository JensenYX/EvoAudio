from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


@dataclass
class Clip:
    wave: np.ndarray
    sr: int = SAMPLE_RATE
    provenance: list[dict[str, Any]] = field(default_factory=list)
    truth: dict[str, Any] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.wave.size) / self.sr

    def copy(self, wave: np.ndarray | None = None) -> "Clip":
        return Clip(
            wave=self.wave.copy() if wave is None else wave,
            sr=self.sr,
            provenance=[dict(p) for p in self.provenance],
            truth=dict(self.truth),
            segments=[dict(s) for s in self.segments],
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}.wav")
        wave = np.clip(self.wave.astype(np.float32), -1.0, 1.0)
        sf.write(tmp, wave, self.sr, subtype="PCM_16")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        offset: float = 0.0,
        max_dur: float | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> "Clip":
        wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if sr != SAMPLE_RATE:
            from scipy.signal import resample_poly

            g = gcd(SAMPLE_RATE, sr)
            wave = resample_poly(wave, SAMPLE_RATE // g, sr // g).astype(np.float32)
        if offset > 0:
            wave = wave[int(offset * SAMPLE_RATE) :]
        if max_dur is not None and max_dur > 0:
            wave = wave[: int(max_dur * SAMPLE_RATE)]
        wave = np.ascontiguousarray(wave, dtype=np.float32)
        if not np.isfinite(wave).all():
            raise ValueError(f"non-finite samples in {path}")
        prov = {"source": "file", "path": str(path), "offset": offset, "max_dur": max_dur}
        if provenance:
            prov.update(provenance)
        return cls(wave=wave, provenance=[prov])

    @classmethod
    def silence(cls, seconds: float) -> "Clip":
        return cls(
            wave=np.zeros(max(1, int(seconds * SAMPLE_RATE)), dtype=np.float32),
            provenance=[{"source": "silence", "seconds": seconds}],
        )
