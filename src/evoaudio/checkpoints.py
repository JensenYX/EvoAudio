from __future__ import annotations

from pathlib import Path

from evoaudio.io import read_json


def checkpoint_complete(path: Path, *, optimizer: bool = False) -> bool:
    index = path / "model.safetensors.index.json"
    if not (path / "config.json").is_file() or not (path / "trainer_state.json").is_file():
        return False
    if index.exists():
        try:
            shards = set(read_json(index)["weight_map"].values())
        except (ValueError, KeyError):
            return False
        if not shards or not all(
            (path / name).is_file() and (path / name).stat().st_size > 0 for name in shards
        ):
            return False
    elif not (path / "model.safetensors").is_file():
        return False
    if optimizer:
        latest = path / "latest"
        if not latest.is_file():
            return False
        step = path / latest.read_text().strip()
        if (
            not step.is_dir()
            or not list(step.glob("*optim_states.pt"))
            or not list(step.glob("*model_states.pt"))
        ):
            return False
    return True
