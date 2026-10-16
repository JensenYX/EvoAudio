from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class Proposer(Options):
    mode: str = "llm"
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout: float = 120


class Resources(Options):
    hosts: list[str] = Field(default_factory=lambda: ["localhost"])
    devices: list[int] = Field(default_factory=lambda: list(range(8)))
    master_port: int = 29510
    tts_python: str = ".venv-tts/bin/python"
    train_python: str = sys.executable
    worker_timeout: int = 14400


class Data(Options):
    librispeech: str | None = None
    fsd50k: str | None = None
    audioset: str | None = None
    manifest: str | None = None
    soundfont: str = "/usr/share/sounds/sf2/TimGM6mb.sf2"
    fluidsynth: str = "fluidsynth"
    speech_per_split: int = Field(default=96, ge=4)
    music_per_split: int = Field(default=48, ge=4)
    heldout_size: int = Field(default=2500, ge=4)


class Curriculum(Options):
    candidates: int = Field(default=3000, ge=4)
    train_size: int = Field(default=1024, ge=2)
    max_generation_batches: int = Field(default=8, ge=1)
    max_skill_share: float = Field(default=0.06, gt=0, le=1)
    retention: float = Field(default=0.12, ge=0, lt=1)
    target_success: float = Field(default=0.5, gt=0, lt=1)
    difficulty_step: float = Field(default=0.25, gt=0, le=1)
    easy_anchor_share: float = Field(default=0.2, ge=0, le=1)
    pilot_per_skill: int = Field(default=8, ge=2)


class Training(Options):
    rounds: int = Field(default=13, ge=1)
    steps: int = Field(default=64, ge=2)
    save_steps: int = Field(default=32, ge=1)
    prompts_per_step: int = Field(default=16, ge=1)
    group_size: int = Field(default=16, ge=2)
    learning_rate: float = Field(default=1e-6, gt=0)
    temperature: float = Field(default=0.9, gt=0)
    beta: float = Field(default=0.02, ge=0)
    max_completion_length: int = Field(default=128, ge=16)
    max_length: int = Field(default=8192, ge=512)
    deepspeed: str = "zero2"
    attention: str = "flash_attn"


class Config(Options):
    output_dir: str = "runs/evoaudio"
    seed: int = 17
    model: str = "Qwen/Qwen2.5-Omni-7B"
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    whisper_model: str = "openai/whisper-large-v3"
    tasks: list[str] = Field(default_factory=list)
    proposer: Proposer = Field(default_factory=Proposer)
    resources: Resources = Field(default_factory=Resources)
    data: Data = Field(default_factory=Data)
    curriculum: Curriculum = Field(default_factory=Curriculum)
    training: Training = Field(default_factory=Training)

    @model_validator(mode="after")
    def validate_settings(self):
        from evoaudio.tasks.catalog import TASKS

        if not self.tasks:
            self.tasks = list(TASKS)
        unknown = set(self.tasks) - set(TASKS)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("Task names must be unique")
        if self.curriculum.max_skill_share * len(self.tasks) < 1 - 1e-9:
            raise ValueError("max_skill_share is too small for the number of enabled tasks")
        if self.proposer.mode not in {"llm", "local"}:
            raise ValueError("proposer.mode must be llm or local")
        endpoint = urlparse(self.proposer.base_url)
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.netloc
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("proposer.base_url must be an HTTP(S) endpoint without embedded credentials")
        if not self.resources.devices or len(set(self.resources.devices)) != len(self.resources.devices):
            raise ValueError("resources.devices must contain distinct GPU indices")
        if not self.resources.hosts:
            raise ValueError("resources.hosts must not be empty")
        if any(d < 0 for d in self.resources.devices):
            raise ValueError("GPU indices must be nonnegative")
        if len(set(self.resources.hosts)) != len(self.resources.hosts):
            raise ValueError("resources.hosts must be unique")
        if len(self.resources.hosts) > 1 and "localhost" in self.resources.hosts:
            raise ValueError("Multi-node hosts must have addresses reachable from every node")
        count = self.curriculum.train_size
        if int(count * self.curriculum.max_skill_share) * len(self.tasks) < count:
            raise ValueError("train_size and max_skill_share cannot produce integer task quotas")
        if self.training.steps % self.training.save_steps:
            raise ValueError("training.steps must be divisible by save_steps")
        return self

    @property
    def root(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    for field in ("model", "tts_model", "whisper_model", "output_dir"):
        value = os.environ.get(f"EVOAUDIO_{field.upper()}")
        if value:
            data[field] = value
    for field in ("tts_python", "train_python"):
        value = os.environ.get(f"EVOAUDIO_{field.upper()}")
        if value:
            data.setdefault("resources", {})[field] = value
    cfg = Config.model_validate(data)
    for field in ("tts_python", "train_python"):
        value = getattr(cfg.resources, field)
        if "/" in value:
            setattr(cfg.resources, field, str(Path(value).expanduser().absolute()))
    return cfg
