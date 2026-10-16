from __future__ import annotations

import fcntl
from collections import defaultdict
from pathlib import Path

from evoaudio.answers import training_row
from evoaudio.assets import prepare_assets
from evoaudio.curriculum import plan_curriculum, select_items
from evoaudio.io import digest, file_digest, log, read_json, read_jsonl, write_json, write_jsonl
from evoaudio.solver import evaluate
from evoaudio.tasks.generate import generate
from evoaudio.train import train


def prepare(cfg) -> Path:
    prepare_assets(cfg)
    return generate(
        cfg, cfg.root / "heldout", count=cfg.data.heldout_size, split="heldout", seed=cfg.seed + 100000
    )


def assert_disjoint(training: list[dict], heldout: list[dict]):
    for field in ("template_id", "generation_seed"):
        if {r[field] for r in training} & {r[field] for r in heldout}:
            raise ValueError(f"Training and heldout share {field}")
    a = {s for r in training for s in r["source_ids"]}
    b = {s for r in heldout for s in r["source_ids"]}
    if a & b:
        raise ValueError("Training and heldout share source recordings")


def mixed_rates(root: Path) -> dict:
    groups = defaultdict(list)
    for path in sorted(root.glob("rank*.jsonl")):
        for row in read_jsonl(path):
            groups[(row["step"], row["item_id"], row["skill"])].append(row["reward"])
    by_skill = defaultdict(list)
    for (_, _, skill), values in groups.items():
        by_skill[skill].append(len(set(values)) > 1)
    return {skill: sum(values) / len(values) for skill, values in by_skill.items()}


def choose_candidate(parent_accuracy: float, candidates: list[dict]) -> tuple[dict, bool]:
    if not candidates:
        raise ValueError("No evaluated candidates are available")
    best = max(candidates, key=lambda r: (r["accuracy"], -r["step"]))
    return best, best["accuracy"] > parent_accuracy


def evolve(cfg, *, resume: bool = False):
    cfg.root.mkdir(parents=True, exist_ok=True)
    with (cfg.root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("Another evolution process is using this output_dir") from e
        return _evolve(cfg, resume=resume)


def _evolve(cfg, *, resume: bool):
    heldout_path = prepare(cfg)
    heldout = read_jsonl(heldout_path)
    snapshot = cfg.model_dump()
    snapshot["training"].pop("rounds")
    identity = digest({"config": snapshot, "heldout": file_digest(heldout_path)})
    state_path = cfg.root / "state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["identity"] != identity:
            raise ValueError("Run settings changed. Use a new output_dir.")
        if not resume:
            raise ValueError("A run already exists. Use evolve --resume.")
    else:
        state = {
            "identity": identity,
            "parent": cfg.model,
            "completed_rounds": 0,
            "profiles": [],
            "mixed": {},
            "plan": None,
        }
        write_json(state_path, state)
    for round_index in range(state["completed_rounds"] + 1, cfg.training.rounds + 1):
        root = cfg.root / f"round-{round_index:03d}"
        root.mkdir(parents=True, exist_ok=True)
        parent = state["parent"]
        base = evaluate(cfg, parent, heldout_path, root / "parent")
        profile = base["per_skill"]
        if not state["profiles"]:
            state["profiles"] = [profile]
            write_json(state_path, state)
        plan_path = root / "plan.json"
        if plan_path.exists():
            plan = read_json(plan_path)
        else:
            initial = plan_curriculum(cfg, profile, state["profiles"], state["mixed"], state["plan"])
            pilot_plan = {task: {**initial[task], "quota": 1 / len(cfg.tasks)} for task in cfg.tasks}
            pilot_items = generate(
                cfg,
                root / "pilot",
                count=len(cfg.tasks) * cfg.curriculum.pilot_per_skill,
                seed=cfg.seed + round_index * 10000,
                plan=pilot_plan,
            )
            pilot = evaluate(
                cfg, parent, pilot_items, root / "pilot" / "responses", samples=4, temperature=0.9
            )
            plan = {
                task: {
                    **initial[task],
                    "difficulty": min(
                        0.95,
                        max(
                            0.05,
                            initial[task]["difficulty"]
                            + cfg.curriculum.difficulty_step
                            * (pilot["per_skill"].get(task, 0.5) - cfg.curriculum.target_success),
                        ),
                    ),
                }
                for task in cfg.tasks
            }
            write_json(plan_path, plan)
        pack_path = root / "pack.jsonl"
        if not pack_path.exists():
            all_items, calibration, question_only = [], {}, {}
            selected = []
            for batch in range(cfg.curriculum.max_generation_batches):
                data_root = root / f"candidates-{batch:02d}"
                items_path = generate(
                    cfg,
                    data_root,
                    count=cfg.curriculum.candidates,
                    seed=cfg.seed + round_index * 10000 + 100 + batch,
                    plan=plan,
                )
                items = read_jsonl(items_path)
                assert_disjoint(items, heldout)
                without_audio = evaluate(
                    cfg, parent, items_path, data_root / "question_only", question_only=True
                )
                survivors = [r for r in items if without_audio["per_item"][r["item_id"]]["correct"] == 0]
                survivor_path = data_root / "survivors.jsonl"
                write_jsonl(survivor_path, survivors)
                if survivors:
                    calibrated = evaluate(
                        cfg, parent, survivor_path, data_root / "calibration", samples=4, temperature=0.9
                    )
                    calibration.update(calibrated["per_item"])
                question_only.update(without_audio["per_item"])
                all_items += survivors
                selected = select_items(cfg, all_items, calibration, question_only, plan, profile)
                if selected:
                    break
            if not selected:
                raise RuntimeError(
                    "Not enough verified questions with mixed parent answers. "
                    "Increase candidates or generation batches."
                )
            assert_disjoint(selected, heldout)
            write_jsonl(root / "selected.jsonl", selected)
            write_jsonl(pack_path, [training_row(r) for r in selected])
        result = train(cfg, parent, pack_path, root / "train", resume=resume)
        candidates = []
        for checkpoint in result["checkpoints"]:
            summary = evaluate(cfg, checkpoint, heldout_path, root / Path(checkpoint).name)
            candidates.append(
                {
                    "checkpoint": checkpoint,
                    "step": int(Path(checkpoint).name.split("-")[-1]),
                    "accuracy": summary["accuracy"],
                    "per_skill": summary["per_skill"],
                }
            )
        best, promoted = choose_candidate(base["accuracy"], candidates)
        decision = {
            "round": round_index,
            "parent": parent,
            "parent_accuracy": base["accuracy"],
            "candidates": candidates,
            "promoted": promoted,
            "next_parent": best["checkpoint"] if promoted else parent,
        }
        write_json(root / "decision.json", decision)
        state.update(
            parent=decision["next_parent"],
            completed_rounds=round_index,
            mixed=mixed_rates(root / "train" / "rewards"),
            plan=plan,
        )
        state["profiles"].append(best["per_skill"] if promoted else profile)
        next_plan = plan_curriculum(
            cfg, state["profiles"][-1], state["profiles"], state["mixed"], state["plan"]
        )
        write_json(root / "next_curriculum.json", next_plan)
        write_json(state_path, state)
        log(
            f"Round {round_index}: {'promoted' if promoted else 'retained parent'}; "
            f"heldout {base['accuracy']:.3f} → {best['accuracy']:.3f}"
        )
    return state
