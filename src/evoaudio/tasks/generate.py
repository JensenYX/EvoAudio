from __future__ import annotations

import itertools
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

from evoaudio.audio import Clip
from evoaudio.audio import measure as M
from evoaudio.audio import transforms as T
from evoaudio.io import digest, log, read_json, read_jsonl, write_json, write_jsonl
from evoaudio.tasks.catalog import TASKS, question


class Rejected(ValueError):
    pass


@lru_cache(maxsize=64)
def load_source(path: str) -> Clip:
    return T.normalize(T.trim_silence(Clip.load(path)))


def source_clip(row: dict) -> Clip:
    clip = load_source(row["path"]).copy()
    clip.provenance = [{"source_id": row["source_id"], "file_digest": row["file_digest"]}]
    clip.truth = {k: row[k] for k in ("text", "speaker", "label", "bpm", "notes") if k in row}
    return clip


def metric(clip: Clip, kind: str, units: int = 1) -> float:
    if kind == "pitch":
        value = M.f0_median(clip.wave)
    elif kind == "volume":
        value = M.rms_dbfs(clip.wave)
    elif kind == "speed":
        value = M.speech_rate(clip.wave, units)
    elif kind == "tempo":
        value = M.tempo_bpm(clip.wave)
    elif kind == "duration":
        value = clip.duration
    else:
        raise ValueError(kind)
    if value is None or not math.isfinite(value):
        raise Rejected(f"No measurable {kind}")
    return float(value)


def measured_parts(clip: Clip) -> list[Clip]:
    return [
        Clip(M.segment_wave(clip, s["start"], s["end"]))
        for s in clip.segments
        if s.get("role") != "background"
    ]


def quantize(clip: Clip) -> Clip:
    clip = T.peak_limit(clip)
    return clip.copy((np.floor(np.clip(clip.wave, -1, 1 - 1 / 32768) * 32768) / 32768).astype(np.float32))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0 or abs(len(a) - len(b)) > 4:
        return 0.0
    x, y = a[:n].astype(np.float64), b[:n].astype(np.float64)
    return float(np.dot(x, y) / max(1e-12, np.linalg.norm(x) * np.linalg.norm(y)))


def ordered(values, order, kind):
    arranged = [values[i] for i in order]
    if kind == "volume":
        ok = all(b - a >= 2 for a, b in zip(arranged, arranged[1:]))
    else:
        ok = all(b / max(a, 1e-9) >= 1.06 for a, b in zip(arranged, arranged[1:]))
    if not ok:
        raise Rejected(f"Measured {kind} order disagrees with construction")


def transform(clip: Clip, attr: str, value: float) -> Clip:
    return {
        "pitch": T.pitch_shift,
        "speed": T.time_stretch,
        "tempo": T.time_stretch,
        "duration": T.time_stretch,
        "volume": T.gain_db,
    }[attr](clip, value)


def build_item(
    task: str, sources: list[dict], split: str, seed: int, difficulty: float, item_id: str, output: Path
) -> dict:
    rng = random.Random(seed)
    spec = TASKS[task]
    pool = [
        r
        for r in sources
        if r["split"] == split and r["kind"] == ("speech" if spec.source == "intonation" else spec.source)
    ]
    if task == "sentence_intonation":
        pool = [r for r in pool if r.get("contour")]
    else:
        pool = [r for r in pool if not r.get("contour") and not r.get("pause_pair")]
    if not pool:
        raise Rejected(f"No source recordings for {task}")
    selected = [rng.choice(pool)]
    base = source_clip(selected[0])
    if base.duration > 9:
        raise Rejected("Source excerpt is too long")
    checks = []
    truth = {}
    options = []
    d = min(1.0, max(0.0, difficulty))
    ratio, semitones, db = 1.65 - 0.45 * d, 6.5 - 4.0 * d, 11 - 8 * d
    if task.endswith("pattern3"):
        attr = task.split("_")[0]
        levels = ["low", "medium", "high"]
        rng.shuffle(levels)
        control = (
            {"low": -semitones, "medium": 0, "high": semitones}
            if attr == "pitch"
            else {"low": -db, "medium": 0, "high": db}
            if attr == "volume"
            else {"low": 1 / ratio, "medium": 1, "high": ratio}
        )
        clip = quantize(T.concat([transform(base, attr, control[level]) for level in levels], gap=0.45))
        values = [
            metric(part, attr, len(base.truth.get("text", "").split()) or 1) for part in measured_parts(clip)
        ]
        order = sorted(range(3), key=lambda i: ["low", "medium", "high"].index(levels[i]))
        ordered(values, order, attr)
        gold = "-".join(levels)
        choices = [
            "-".join(p) for p in itertools.permutations(["low", "medium", "high"]) if "-".join(p) != gold
        ]
        options = [gold, *rng.sample(choices, 3)]
        q, template = question(
            "pattern",
            split,
            rng,
            attribute={"pitch": "voice pitch", "volume": "loudness", "speed": "speaking rate"}[attr],
        )
        truth = {"levels": levels, "controls": control}
        checks = [{"metric": attr, "values": values, "order": order, "passed": True}]
    elif task == "sentence_intonation":
        clip = quantize(base)
        contour = M.f0_contour(clip.wave)
        gold = selected[0]["contour"]
        if contour.get("shape") != gold:
            raise Rejected("Rendered intonation disagrees with the source contour")
        options = ["rising", "falling", "rise_fall", "fall_rise"]
        q, template = question("intonation", split, rng)
        truth = {"contour": gold}
        checks = [{"metric": "intonation", **contour, "passed": True}]
    elif task == "pause_after_word":
        prefixes = [r for r in sources if r["split"] == split and r.get("pause_role") == "prefix"]
        if not prefixes:
            raise Rejected("No verified pause phrases")
        prefix = rng.choice(prefixes)
        suffix = next(
            (
                r
                for r in sources
                if r.get("pause_pair") == prefix["pause_pair"] and r.get("pause_role") == "suffix"
            ),
            None,
        )
        if suffix is None:
            raise Rejected("Pause suffix is missing")
        selected = [prefix, suffix]
        gap = 1.4 - 0.6 * d
        clip = quantize(T.concat([source_clip(prefix), source_clip(suffix)], gap=gap))
        gaps = M.silence_gaps(clip.wave, min_gap=0.3)
        target = (clip.segments[0]["end"] + clip.segments[1]["start"]) / 2
        if not gaps:
            raise Rejected("Inserted pause is not measurable")
        longest = max(gaps, key=lambda x: x[1] - x[0])
        if not longest[0] <= target <= longest[1]:
            raise Rejected("A different pause is longer")
        gold = prefix["text"].rstrip(".,?!").split()[-1]
        words = list(dict.fromkeys((prefix["text"] + " " + suffix["text"]).replace(".", "").split()))
        distractors = [w for w in words if w.casefold() != gold.casefold()]
        options = [gold, *rng.sample(distractors, 3)]
        q, template = question("pause", split, rng)
        truth = {"after_word": gold, "gap_seconds": gap}
        checks = [{"metric": "silence", "gaps": gaps, "target_time": target, "passed": True}]
    elif task in {
        "speech_pair_compare",
        "sound_louder",
        "sound_duration",
        "music_segment_louder",
        "music_pitch_compare",
        "music_speed_compare",
        "music_tempo_change",
    }:
        attr = (
            rng.choice(["pitch", "speed", "volume", "duration"])
            if task == "speech_pair_compare"
            else "volume"
            if task in {"sound_louder", "music_segment_louder"}
            else "duration"
            if task == "sound_duration"
            else "pitch"
            if task == "music_pitch_compare"
            else "tempo"
        )
        sign = rng.choice([-1, 1])
        control = (
            sign * (semitones if attr == "pitch" else db) if attr in {"pitch", "volume"} else ratio**sign
        )
        second = transform(base, attr, control)
        clip = quantize(T.concat([base, second], gap=0.45))
        parts = measured_parts(clip)
        values = [metric(p, attr, len(base.truth.get("text", "").split()) or 1) for p in parts]
        expected_sign = -sign if attr == "duration" else sign
        order = [0, 1] if expected_sign > 0 else [1, 0]
        ordered(values, order, attr)
        words = {
            "pitch": ("lower", "higher", "the same pitch"),
            "speed": ("slower", "faster", "the same speed"),
            "tempo": ("slower", "faster", "the same tempo"),
            "volume": ("quieter", "louder", "the same loudness"),
            "duration": ("shorter", "longer", "the same duration"),
        }[attr]
        gold = words[int(expected_sign > 0)]
        options = [*words, "only one segment is present"]
        q, template = question("pair", split, rng, source=spec.domain, attribute=attr)
        truth = {"attribute": attr, "transform": control, "direction": expected_sign}
        checks = [{"metric": attr, "values": values, "passed": True}]
    elif task == "sound_count":
        if M.onset_count(base.wave) != 1:
            raise Rejected("Source contains more than one event")
        count = rng.randint(2, 6)
        clip = quantize(T.concat([base] * count, gap=0.75 - 0.4 * d))
        measured = M.onset_count(clip.wave)
        if measured != count:
            raise Rejected("Rendered event count disagrees with placement")
        gold = str(count)
        options = [gold, *rng.sample([str(x) for x in range(1, 8) if x != count], 3)]
        q, template = question("count", split, rng, label=selected[0]["label"])
        truth = {"count": count, "label": selected[0]["label"]}
        checks = [{"metric": "onsets", "value": measured, "passed": True}]
    elif task == "sound_order":
        others = [r for r in pool if r.get("label") != selected[0].get("label")]
        if not others:
            raise Rejected("Two different sound labels are required")
        selected.append(rng.choice(others))
        parts = [base, source_clip(selected[1])]
        clip = quantize(T.concat(parts, gap=0.65 - 0.3 * d))
        measured = measured_parts(clip)
        scores = [
            similarity(T.fade(original).wave, got.wave) for original, got in zip(parts, measured, strict=True)
        ]
        if min(scores) < 0.98:
            raise Rejected("Rendered event sequence does not match its sources")
        gold = selected[0]["label"]
        options = [gold, selected[1]["label"], "both sounds begin together", "neither sound is audible"]
        q, template = question("order", split, rng)
        truth = {"labels": [r["label"] for r in selected]}
        checks = [{"metric": "source_alignment", "correlations": scores, "passed": True}]
    elif task == "music_dynamics":
        variant = rng.choice(["crescendo", "decrescendo", "steady"])
        slope = {"crescendo": 1, "decrescendo": -1, "steady": 0}[variant]
        clip = quantize(T.ramp_gain(base, -slope * db, slope * db))
        n = len(clip.wave) // 3
        change = M.rms_dbfs(clip.wave[-n:]) - M.rms_dbfs(clip.wave[:n])
        if (slope and change * slope < 3) or (not slope and abs(change) > 1.5):
            raise Rejected("Measured dynamics disagree with the gain envelope")
        gold = {
            "crescendo": "gradually louder",
            "decrescendo": "gradually quieter",
            "steady": "steady loudness",
        }[variant]
        options = [
            "gradually louder",
            "gradually quieter",
            "steady loudness",
            "a long silent break in the middle",
        ]
        q, template = question("dynamics", split, rng)
        truth = {"gain_start_db": -slope * db, "gain_end_db": slope * db}
        checks = [{"metric": "loudness_change_db", "value": change, "passed": True}]
    elif task == "music_repeat":
        others = [r for r in pool if r["source_id"] != selected[0]["source_id"]]
        if not others:
            raise Rejected("Two music sources are required")
        selected.append(rng.choice(others))
        a, b = (
            T.trim(base, 0, min(3, base.duration)),
            T.trim(source_clip(selected[1]), 0, min(3, base.duration)),
        )
        pattern = rng.choice(["AAB", "ABA", "ABB"])
        clip = quantize(T.concat([a if c == "A" else b for c in pattern], gap=0.4))
        parts = measured_parts(clip)
        scores = {(i, j): similarity(parts[i].wave, parts[j].wave) for i, j in [(0, 1), (0, 2), (1, 2)]}
        repeats = [pair for pair, score in scores.items() if score > 0.995]
        want = [(i, j) for i, j in scores if pattern[i] == pattern[j]]
        if repeats != want:
            raise Rejected("Repeated-passage measurement is ambiguous")
        gold = {"AAB": "first and second", "ABA": "first and third", "ABB": "second and third"}[pattern]
        options = ["first and second", "first and third", "second and third", "all three are different"]
        q, template = question("repeat", split, rng)
        truth = {"pattern": pattern}
        checks = [
            {
                "metric": "waveform_similarity",
                "values": {f"{i},{j}": v for (i, j), v in scores.items()},
                "passed": True,
            }
        ]
    elif task == "compound_speech":
        texts = {r["text"]: r for r in pool if r.get("text")}
        if len(texts) < 4:
            raise Rejected("Four distinct verified speech contents are required")
        selected = rng.sample(list(texts.values()), 4)
        levels = [-db, 0, db]
        rng.shuffle(levels)
        parts = [T.gain_db(source_clip(r), value) for r, value in zip(selected[:3], levels, strict=True)]
        clip = quantize(T.concat(parts, gap=0.45))
        values = [M.rms_dbfs(p.wave) for p in measured_parts(clip)]
        order = sorted(range(3), key=lambda i: levels[i])
        ordered(values, order, "volume")
        target = order[-1]
        gold = selected[target]["text"]
        options = [r["text"] for r in selected]
        q, template = question("compound", split, rng, attribute="loudest")
        truth = {"target": target, "texts": options, "gains_db": levels}
        checks = [{"metric": "loudness_db", "values": values, "passed": True}]
    else:
        raise ValueError(task)
    if clip.duration > 32 or clip.duration < 0.5:
        raise Rejected("Rendered duration is outside the task range")
    if len(options) != 4 or len({x.casefold().strip() for x in options}) != 4:
        raise Rejected("Answer options are not distinct")
    rng.shuffle(options)
    answer = "ABCD"[options.index(gold)]
    path = output / "audio" / f"{item_id}.wav"
    clip.save(path)
    return {
        "item_id": item_id,
        "skill": task,
        "domain": spec.domain,
        "audio": [str(path.resolve())],
        "question": q,
        "options": [{"id": "ABCD"[i], "text": text} for i, text in enumerate(options)],
        "answer": answer,
        "difficulty": round(d, 4),
        "partition": split,
        "template_id": template,
        "source_ids": sorted({r["source_id"] for r in selected}),
        "generation_seed": seed,
        "duration_sec": clip.duration,
        "truth": truth,
        "checks": checks,
        "provenance": clip.provenance,
        "segments": clip.segments,
    }


def generate(
    cfg, output: Path, *, count: int, split: str = "train", seed: int | None = None, plan: dict | None = None
) -> Path:
    output = output.resolve()
    seed = cfg.seed if seed is None else seed
    manifest = cfg.root / "assets" / "manifest.jsonl"
    sources = read_jsonl(manifest)
    identity = digest(
        {
            "source": file_digest_for(manifest),
            "seed": seed,
            "count": count,
            "split": split,
            "tasks": cfg.tasks,
            "plan": plan,
        }
    )
    receipt, items_path = output / "generation.json", output / "items.jsonl"
    if receipt.exists():
        if read_json(receipt)["identity"] != identity:
            raise ValueError(f"Generation settings changed for {output}")
        if items_path.exists() and all(Path(p).is_file() for r in read_jsonl(items_path) for p in r["audio"]):
            return items_path
    output.mkdir(parents=True, exist_ok=True)
    plan = plan or {task: {"quota": 1 / len(cfg.tasks), "difficulty": 0.5} for task in cfg.tasks}
    raw = {task: count * float(plan[task]["quota"]) for task in cfg.tasks}
    quotas = {task: int(value) for task, value in raw.items()}
    for task in sorted(quotas, key=lambda t: (-(raw[t] - quotas[t]), t))[: count - sum(quotas.values())]:
        quotas[task] += 1
    rows, rejected = [], {}
    for task, quota in quotas.items():
        accepted = 0
        for attempt in range(max(60, quota * 30)):
            if accepted >= quota:
                break
            item_seed = int(digest([seed, split, task, attempt])[:15], 16)
            item_id = f"{split}-{task}-{digest([seed, task, attempt])[:16]}"
            difficulty = float(plan[task]["difficulty"])
            if split == "heldout":
                difficulty = [0.2, 0.5, 0.8][accepted % 3]
            elif (item_seed % 1000) / 1000 < cfg.curriculum.easy_anchor_share:
                difficulty = max(0.0, difficulty - 0.3)
            try:
                item = build_item(task, sources, split, item_seed, difficulty, item_id, output)
            except Rejected as e:
                rejected[str(e)] = rejected.get(str(e), 0) + 1
                continue
            rows.append(item)
            accepted += 1
        if accepted < quota:
            write_json(output / "rejections.json", rejected)
            raise RuntimeError(
                f"Generated {accepted}/{quota} verified items for {task}. Read {output / 'rejections.json'}"
            )
        log(f"Generated {accepted} verified {task} items")
    write_jsonl(items_path, rows)
    write_json(receipt, {"identity": identity, "count": len(rows), "partition": split, "rejected": rejected})
    return items_path


def file_digest_for(path):
    from evoaudio.io import file_digest

    return file_digest(path)
