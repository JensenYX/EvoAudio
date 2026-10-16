from __future__ import annotations

import json
import time
from pathlib import Path

from evoaudio.checkpoints import checkpoint_complete
from evoaudio.io import digest, file_digest, model_identity, read_json, read_jsonl, write_json, write_jsonl
from evoaudio.resources import launch_training

TRAINABLE = r"^(thinker\.(audio_tower|model|lm_head)\.)"


def build_command(cfg, parent: str, dataset: Path, output: Path) -> list[str]:
    t = cfg.training
    world = len(cfg.resources.hosts) * len(cfg.resources.devices)
    batch = t.prompts_per_step * t.group_size
    if batch % world:
        raise ValueError(f"prompts_per_step × group_size ({batch}) must be divisible by GPU count ({world})")
    accumulation = batch // world
    plugin = Path(__file__).with_name("reward_plugin.py")
    args = {
        "rlhf_type": "grpo",
        "model": parent,
        "ref_model": parent,
        "model_type": "qwen2_5_omni",
        "use_hf": "true",
        "dataset": str(dataset),
        "output_dir": str(output),
        "add_version": "false",
        "tuner_type": "full",
        "torch_dtype": "bfloat16",
        "learning_rate": t.learning_rate,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0,
        "max_steps": t.steps,
        "save_steps": t.save_steps,
        "save_total_limit": 2,
        "save_only_model": "false",
        "resume_only_model": "false",
        "num_generations": t.group_size,
        "generation_batch_size": batch,
        "steps_per_generation": accumulation,
        "gradient_accumulation_steps": accumulation,
        "per_device_train_batch_size": 1,
        "num_iterations": 1,
        "temperature": t.temperature,
        "top_p": 0.95,
        "top_k": -1,
        "beta": t.beta,
        "epsilon": 0.2,
        "max_completion_length": t.max_completion_length,
        "max_length": t.max_length,
        "truncation_strategy": "delete",
        "dataset_shuffle": "true",
        "split_dataset_ratio": 0,
        "strict": "true",
        "remove_unused_columns": "false",
        "dataset_num_proc": 1,
        "dataloader_num_workers": 0,
        "gradient_checkpointing": "true",
        "gradient_checkpointing_kwargs": json.dumps({"use_reentrant": False}),
        "freeze_parameters_regex": ".*",
        "trainable_parameters_regex": TRAINABLE,
        "external_plugins": str(plugin),
        "reward_funcs": "evoaudio_answer",
        "reward_weights": 1.0,
        "deepspeed": t.deepspeed,
        "attn_impl": t.attention,
        "report_to": "none",
        "logging_steps": 1,
        "log_completions": "true",
        "seed": cfg.seed,
        "ddp_timeout": 1800,
    }
    command = [cfg.resources.train_python, "-m", "swift.cli.main", "rlhf"]
    for key, value in args.items():
        command.extend([f"--{key}", str(value)])
    return command


def train(cfg, parent: str, dataset: Path, root: Path, *, resume: bool = False, dry_run: bool = False):
    output = root / "checkpoints"
    command = build_command(cfg, parent, dataset, output)
    identity = digest(
        {
            "parent": model_identity(parent),
            "data": file_digest(dataset),
            "training": cfg.training.model_dump(exclude={"rounds"}),
            "world": len(cfg.resources.hosts) * len(cfg.resources.devices),
        }
    )
    receipt = root / "identity.json"
    if receipt.exists() and read_json(receipt)["identity"] != identity:
        raise ValueError("Training inputs changed. Use a new output directory.")
    checkpoints = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    completed = root / "completed.json"
    if completed.exists():
        result = read_json(completed)
        if all(checkpoint_complete(Path(c)) for c in result["checkpoints"]):
            return result
        raise FileNotFoundError("A completed training checkpoint is missing")
    if checkpoints:
        if not resume:
            raise ValueError("Training checkpoints already exist. Run evolve --resume to continue.")
        valid = [p for p in checkpoints if checkpoint_complete(p, optimizer=True)]
        if not valid:
            raise ValueError("No complete checkpoint is available for resuming this training run")
        last = valid[-1]
        for incomplete in checkpoints:
            if not checkpoint_complete(incomplete, optimizer=True):
                if not dry_run:
                    incomplete.rename(incomplete.with_name(f"incomplete-{incomplete.name}-{time.time_ns()}"))
        checkpoints = valid
        if int(last.name.split("-")[-1]) < cfg.training.steps:
            command += ["--resume_from_checkpoint", str(last)]
            if not dry_run:
                step = int(read_json(last / "trainer_state.json")["global_step"])
                for path in (root / "rewards").glob("rank*.jsonl"):
                    write_jsonl(path, [r for r in read_jsonl(path) if r["step"] < step])
    plan = {"command": command, "identity": identity}
    if dry_run:
        return plan
    write_json(receipt, {"identity": identity})
    write_json(root / "launch.json", plan)
    if not checkpoints or int(checkpoints[-1].name.split("-")[-1]) < cfg.training.steps:
        launch_training(cfg, command, root / "logs")
    checkpoints = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    required = [cfg.training.steps - cfg.training.save_steps, cfg.training.steps]
    required = [s for s in required if s > 0]
    for step in required:
        cp = output / f"checkpoint-{step}"
        if not checkpoint_complete(cp, optimizer=True):
            raise RuntimeError(f"Training did not produce a complete checkpoint at step {step}")
    result = {"checkpoints": [str(p) for p in checkpoints], "steps": cfg.training.steps}
    write_json(completed, result)
    return result
