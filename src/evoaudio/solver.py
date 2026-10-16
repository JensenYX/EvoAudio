from __future__ import annotations

import argparse
import gc
import os
from collections import defaultdict
from pathlib import Path

from evoaudio.answers import parse_answer, prompt, reward
from evoaudio.io import (
    digest,
    file_digest,
    log,
    model_identity,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from evoaudio.resources import run_worker


def infer(
    model: str,
    items: Path,
    output: Path,
    *,
    samples: int = 1,
    temperature: float = 0,
    question_only: bool = False,
    max_tokens: int = 128,
    seed: int = 17,
    attention: str = "flash_attn",
):
    identity = digest(
        {
            "model": model_identity(model),
            "items": file_digest(items),
            "samples": samples,
            "temperature": temperature,
            "question_only": question_only,
            "max_tokens": max_tokens,
            "seed": seed,
            "attention": attention,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "identity.json"
    if receipt.exists() and read_json(receipt)["identity"] != identity:
        raise ValueError(f"Inference inputs changed at {output}")
    write_json(receipt, {"identity": identity})
    rows = read_jsonl(items)
    ids = {r["item_id"] for r in rows}
    if len(ids) != len(rows):
        raise ValueError("Inference input item IDs must be unique")
    predictions = output / "predictions.jsonl"
    results = read_jsonl(predictions) if predictions.exists() else []
    done = {(r["item_id"], r["sample_index"]) for r in results}
    todo = [(r, i) for r in rows for i in range(samples) if (r["item_id"], i) not in done]
    if todo:
        os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
        import torch
        from swift import InferRequest, RequestConfig, TransformersEngine

        engine = TransformersEngine(
            model,
            device_map="cuda:0",
            attn_impl=attention,
            max_batch_size=1,
            torch_dtype=torch.bfloat16,
            use_hf=True,
        )
        for index, (item, sample_index) in enumerate(todo):
            item_seed = int(digest([seed, item["item_id"], sample_index])[:8], 16)
            torch.manual_seed(item_seed)
            torch.cuda.manual_seed_all(item_seed)
            request = InferRequest(
                messages=[{"role": "user", "content": prompt(item, question_only)}],
                audios=[] if question_only else item["audio"],
            )
            config = RequestConfig(
                max_tokens=max_tokens, temperature=temperature, top_p=0.95 if temperature else 1.0
            )
            response = engine.infer([request], config, use_tqdm=False)[0]
            raw = response.choices[0].message.content
            prediction = parse_answer(raw, item["options"])
            results.append(
                {
                    "item_id": item["item_id"],
                    "skill": item["skill"],
                    "sample_index": sample_index,
                    "raw": raw,
                    "prediction": prediction,
                    "correct": prediction == item["answer"],
                    "reward": reward(raw, item["answer"], item["options"]),
                    "gold": item["answer"],
                }
            )
            write_jsonl(predictions, results)
            if (index + 1) % 10 == 0:
                log(f"Inference {index + 1}/{len(todo)}")
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    by_item, by_skill = defaultdict(list), defaultdict(list)
    for row in results:
        by_item[row["item_id"]].append(row)
        by_skill[row["skill"]].append(row)
    summary = {
        "samples": len(results),
        "accuracy": sum(r["correct"] for r in results) / max(1, len(results)),
        "valid_rate": sum(r["prediction"] is not None for r in results) / max(1, len(results)),
        "per_skill": {key: sum(r["correct"] for r in rs) / len(rs) for key, rs in by_skill.items()},
        "per_item": {
            key: {
                "correct": sum(r["correct"] for r in rs),
                "samples": len(rs),
                "mixed": len({r["reward"] for r in rs}) > 1,
            }
            for key, rs in by_item.items()
        },
    }
    write_json(output / "summary.json", summary)
    log(
        f"Scored {len(results)} responses; accuracy={summary['accuracy']:.3f}, "
        f"valid={summary['valid_rate']:.3f}"
    )
    return summary


def evaluate(cfg, model: str, items: Path, output: Path, *, samples=1, temperature=0.0, question_only=False):
    args = [
        "--model",
        model,
        "--items",
        str(items.resolve()),
        "--output",
        str(output.resolve()),
        "--samples",
        str(samples),
        "--temperature",
        str(temperature),
        "--seed",
        str(cfg.seed),
        "--max-tokens",
        str(cfg.training.max_completion_length),
        "--attention",
        cfg.training.attention,
    ]
    if question_only:
        args.append("--question-only")
    run_worker(cfg, "evoaudio.solver", args, output / "inference.log")
    return read_json(output / "summary.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--items", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--question-only", action="store_true")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--attention", default="flash_attn")
    args = p.parse_args()
    infer(
        args.model,
        args.items,
        args.output,
        samples=args.samples,
        temperature=args.temperature,
        seed=args.seed,
        question_only=args.question_only,
        max_tokens=args.max_tokens,
        attention=args.attention,
    )


if __name__ == "__main__":
    main()
