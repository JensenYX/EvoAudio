from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evoaudio.config import load_config
from evoaudio.io import read_json, write_json, write_jsonl


def benchmark_items(name: str, root: Path, output: Path, limit: int | None):
    from evoaudio.benchmarks import create_adapter

    adapter = create_adapter(name, root, extraction_dir=output / "audio")
    rows, excluded = [], 0
    for example in adapter.iter_examples("test"):
        category = example.category.casefold()
        if (
            name == "mmau_pro" and category in {"spatial_audio", "open", "instruction following"}
        ) or not example.options:
            excluded += 1
            continue
        if example.answer not in {o["id"] for o in example.options}:
            excluded += 1
            continue
        rows.append(
            {
                "item_id": f"{name}:{example.example_id}",
                "skill": example.category,
                "audio": [str(p) for p in example.audio_paths],
                "question": example.question,
                "options": example.options,
                "answer": example.answer,
            }
        )
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise ValueError("No benchmark examples with official multiple-choice answers were found")
    path = output / "items.jsonl"
    write_jsonl(path, rows)
    write_json(output / "dataset.json", {"benchmark": name, "scored": len(rows), "excluded": excluded})
    return path


def parser():
    p = argparse.ArgumentParser(
        prog="evoaudio", description="Recursive self-improvement for audio understanding"
    )
    sub = p.add_subparsers(dest="command", required=True)
    tasks = sub.add_parser("tasks", help="List task generators and source requirements")
    tasks.add_argument("--json", action="store_true")
    for name, help_text in [
        ("prepare", "Prepare source recordings and the fixed heldout set"),
        ("generate", "Generate verified audio questions"),
        ("evolve", "Train and evaluate successive solver generations"),
        ("evaluate", "Evaluate a solver on heldout questions or a benchmark"),
    ]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, default=Path("configs/quickstart.yaml"))
        if name == "prepare":
            command.add_argument(
                "--assets-only", action="store_true", help="Prepare sources without rendering heldout items"
            )
        if name == "generate":
            command.add_argument("--count", type=int, default=20)
            command.add_argument("--output", type=Path)
            command.add_argument("--seed", type=int)
        if name == "evolve":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--rounds", type=int)
        if name == "evaluate":
            command.add_argument(
                "--model", help="Model ID or checkpoint directory; defaults to the current solver"
            )
            source = command.add_mutually_exclusive_group()
            source.add_argument("--items", type=Path)
            source.add_argument("--benchmark", choices=["mmsu", "mmau_pro", "mmar"])
            command.add_argument("--data-root", type=Path)
            command.add_argument("--output", type=Path)
            command.add_argument("--limit", type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "tasks":
            from evoaudio.tasks.catalog import TASKS

            if args.json:
                print(json.dumps({k: vars(v) for k, v in TASKS.items()}, indent=2))
            else:
                for task, info in TASKS.items():
                    print(f"{task:25} {info.domain:7} {info.description}")
            return
        cfg = load_config(args.config)
        if args.command == "prepare":
            from evoaudio.assets import prepare_assets
            from evoaudio.evolution import prepare

            print(prepare_assets(cfg) if args.assets_only else prepare(cfg))
        elif args.command == "generate":
            from evoaudio.assets import prepare_assets
            from evoaudio.tasks.generate import generate

            if args.count < 1:
                raise ValueError("count must be positive")
            prepare_assets(cfg)
            path = generate(
                cfg,
                args.output or cfg.root / "samples",
                count=args.count,
                seed=args.seed if args.seed is not None else cfg.seed + 1,
            )
            print(path)
        elif args.command == "evolve":
            from evoaudio.evolution import evolve

            if args.rounds is not None:
                if args.rounds < 1:
                    raise ValueError("rounds must be positive")
                cfg.training.rounds = args.rounds
            result = evolve(cfg, resume=args.resume)
            print(json.dumps({"rounds": result["completed_rounds"], "model": result["parent"]}, indent=2))
        else:
            from evoaudio.solver import evaluate

            state = cfg.root / "state.json"
            model = args.model or (read_json(state)["parent"] if state.exists() else cfg.model)
            output = (args.output or cfg.root / "evaluation" / (args.benchmark or "heldout")).resolve()
            items = args.items or cfg.root / "heldout" / "items.jsonl"
            if args.benchmark:
                if not args.data_root:
                    raise ValueError("--data-root is required for benchmark evaluation")
                items = benchmark_items(args.benchmark, args.data_root, output / "dataset", args.limit)
            print(json.dumps(evaluate(cfg, model, items, output), indent=2))
    except (ValueError, RuntimeError, FileNotFoundError, TimeoutError) as error:
        print(f"EvoAudio: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
