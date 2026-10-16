from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n")


def write_jsonl(path: str | Path, rows) -> None:
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, default=str, allow_nan=False) + "\n" for row in rows),
    )


def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def model_identity(model: str) -> str:
    path = Path(model)
    if not path.is_dir():
        return digest(model)
    files = sorted(path.glob("*.safetensors"))
    return digest(
        [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]
        + [("config", file_digest(path / "config.json"))]
    )


def log(message: str) -> None:
    print(f"[EvoAudio] {message}", flush=True)
