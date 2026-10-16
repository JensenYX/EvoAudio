from __future__ import annotations

import os
import time

import httpx
import numpy as np

from evoaudio.tasks.catalog import TASKS


def priorities(cfg, profile: dict, history: list[dict], mixed: dict) -> dict:
    local = {
        t: {
            "priority": max(0.1, 1.0 - profile.get(t, 0.5)),
            "direction": "harder" if profile.get(t, 0.5) > 0.6 else "easier",
        }
        for t in cfg.tasks
    }
    if cfg.proposer.mode == "local":
        return local
    key = os.environ.get(cfg.proposer.api_key_env)
    if not key:
        raise ValueError(f"Set {cfg.proposer.api_key_env} before using proposer.mode=llm")
    evidence = {
        "skills": [
            {
                "id": t,
                "description": TASKS[t].description,
                "accuracy": profile.get(t),
                "mixed_rate": mixed.get(t),
            }
            for t in cfg.tasks
        ],
        "recent_profiles": history[-5:],
    }
    system = (
        "Plan an audio learning curriculum from aggregate skill measurements. Return one JSON object keyed "
        "by every skill ID. Each value must contain priority (a positive number) and direction "
        "(harder, easier, or hold). Prioritize improvable weaknesses and preserve mastered skills."
    )
    import json

    error = ""
    for attempt in range(3):
        try:
            with httpx.Client(timeout=cfg.proposer.timeout) as client:
                response = client.post(
                    cfg.proposer.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": cfg.proposer.model,
                        "temperature": 0.2,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(evidence)},
                        ],
                    },
                )
                if response.status_code != 200:
                    error = f"HTTP {response.status_code}"
                    if response.status_code in {401, 403}:
                        break
                    time.sleep(2**attempt)
                    continue
                result = json.loads(response.json()["choices"][0]["message"]["content"])
                for task in cfg.tasks:
                    item = result[task]
                    weight = float(item["priority"])
                    if (
                        not np.isfinite(weight)
                        or weight <= 0
                        or item["direction"] not in {"harder", "easier", "hold"}
                    ):
                        raise ValueError("Invalid curriculum response")
                return result
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            error = "invalid or unavailable proposer response"
            time.sleep(2**attempt)
    raise RuntimeError(f"Curriculum proposer failed: {error}")


def capped_distribution(weights: dict[str, float], cap: float, lower: dict[str, float] | None = None) -> dict:
    lower = lower or {}
    result = {t: float(lower.get(t, 0.0)) for t in weights}
    if any(v > cap + 1e-9 for v in result.values()) or sum(result.values()) > 1 + 1e-9:
        raise ValueError("Curriculum quota constraints are infeasible")
    remaining = 1.0 - sum(result.values())
    active = [t for t in weights if result[t] < cap - 1e-10]
    for _ in range(len(weights) + 2):
        if remaining < 1e-9:
            break
        if not active:
            raise ValueError("Not enough task capacity to allocate the curriculum")
        total = sum(max(0.0, weights[t]) for t in active)
        allocation = {
            t: remaining * (max(0.0, weights[t]) / total if total > 0 else 1 / len(active)) for t in active
        }
        filled = []
        for task in active:
            amount = min(cap - result[task], allocation[task])
            result[task] += amount
            remaining -= amount
            if result[task] >= cap - 1e-10:
                filled.append(task)
        active = [t for t in active if t not in filled]
    return result


def plan_curriculum(
    cfg,
    profile: dict,
    history: list[dict],
    mixed: dict,
    previous: dict | None = None,
    pilot: dict | None = None,
) -> dict:
    proposed = priorities(cfg, profile, history, mixed)
    weights = {}
    for task in cfg.tasks:
        progress = [h.get(task, profile.get(task, 0.5)) for h in history[-4:]]
        stalled = len(progress) >= 3 and max(progress) - min(progress) < 0.005
        transfer = 1.0
        if stalled and len(history) >= 4:
            target = np.diff([h.get(task, 0.0) for h in history[-4:]])
            others = [
                np.diff([np.mean([h.get(t, 0.0) for t in cfg.tasks if t != task]) for h in history[-4:]])
            ]
            correlation = 0.0
            if np.std(target) > 0 and np.std(others[0]) > 0:
                correlation = float(np.corrcoef(target, others[0])[0, 1])
            if correlation < 0.1:
                transfer = 0.5
        weights[task] = float(proposed[task]["priority"]) * float(mixed.get(task, 1.0)) * transfer
    mastered = sorted(cfg.tasks, key=lambda t: (-profile.get(t, 0.0), t))[: max(1, len(cfg.tasks) // 4)]
    lower = {t: cfg.curriculum.retention / len(mastered) for t in mastered}
    quotas = capped_distribution(weights, cfg.curriculum.max_skill_share, lower)
    result = {}
    for task in cfg.tasks:
        d = (previous or {}).get(task, {}).get("difficulty", 0.5)
        measured = (pilot or {}).get(task)
        if measured is not None:
            d += cfg.curriculum.difficulty_step * (measured - cfg.curriculum.target_success)
        elif previous:
            d += {"harder": 0.08, "easier": -0.08, "hold": 0}[proposed[task]["direction"]]
        result[task] = {
            "quota": quotas[task],
            "difficulty": min(0.95, max(0.05, d)),
            "priority": proposed[task]["priority"],
            "retention": task in mastered,
        }
    return result


def select_items(
    cfg, items: list[dict], calibration: dict, question_only: dict, plan: dict, profile: dict
) -> list[dict]:
    from collections import Counter, defaultdict

    candidates = [
        r
        for r in items
        if r["partition"] == "train"
        and calibration.get(r["item_id"], {}).get("samples") == 4
        and 0 < calibration.get(r["item_id"], {}).get("correct", 0) < 4
        and question_only.get(r["item_id"], {}).get("correct", 1) == 0
    ]
    count = cfg.curriculum.train_size
    cap = max(1, int(np.floor(count * cfg.curriculum.max_skill_share)))
    by_skill = defaultdict(list)
    for row in sorted(candidates, key=lambda r: r["item_id"], reverse=True):
        by_skill[row["skill"]].append(row)
    selected, per_skill = [], Counter()
    retention_needed = int(np.ceil(cfg.curriculum.retention * count))
    retention = 0
    while len(selected) < count:
        available = [t for t in cfg.tasks if by_skill[t] and per_skill[t] < cap]
        if retention < retention_needed:
            available = [t for t in available if plan[t].get("retention")]
        if not available:
            break
        task = max(available, key=lambda t: (count * plan[t]["quota"] - per_skill[t], t))
        selected.append(by_skill[task].pop())
        per_skill[task] += 1
        retention += bool(plan[task].get("retention"))
    if len(selected) < count:
        return []
    if retention < retention_needed:
        return []
    return selected
