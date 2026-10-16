from __future__ import annotations

import csv
import hashlib
import io
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from evoaudio.audio import SAMPLE_RATE, Clip
from evoaudio.audio import transforms as T
from evoaudio.io import digest, file_digest, log, read_json, read_jsonl, write_json, write_jsonl
from evoaudio.resources import run_worker


def split_for(source: str, seed: int) -> str:
    return "heldout" if int(digest([source, seed])[:8], 16) % 5 == 0 else "train"


def speech_requests(cfg) -> list[dict]:
    names = ["Maya", "Daniel", "Sophie", "Thomas", "Alice", "Henry", "Clara", "James"]
    objects = [
        "a blue notebook",
        "a red umbrella",
        "a small camera",
        "a wooden box",
        "a green jacket",
        "a yellow folder",
        "a silver watch",
        "a black suitcase",
    ]
    places = ["the station", "the library", "the garden", "the kitchen", "the museum", "the office"]
    train_templates = [
        "{name} left {object} near {place} on Tuesday morning.",
        "Before visiting {place}, {name} packed {object} for the journey.",
        "{name} found {object} while walking past {place} yesterday.",
    ]
    heldout_templates = [
        "On Friday afternoon, {object} was delivered to {name} at {place}.",
        "Someone at {place} asked {name} to bring {object} tomorrow.",
    ]
    result = []
    for split, templates in [("train", train_templates), ("heldout", heldout_templates)]:
        rng = random.Random(int(digest([cfg.seed, split, "speech"])[:15], 16))
        used = set()
        for i in range(cfg.data.speech_per_split):
            for _ in range(100):
                text = rng.choice(templates).format(
                    name=rng.choice(names), object=rng.choice(objects), place=rng.choice(places)
                )
                if text not in used:
                    break
            used.add(text)
            result.append(
                {
                    "request_id": f"{split}-speech-{i:04d}",
                    "text": text,
                    "speaker": ["Ryan", "Aiden", "Vivian", "Serena"][i % 4],
                    "split": split,
                    "seed": rng.randrange(2**31),
                    "template_id": f"speech:{split}:{i % len(templates)}",
                }
            )
        if "sentence_intonation" in cfg.tasks:
            styles = {
                "rising": "Read the whole sentence with a clear rising pitch at the end, as a question.",
                "falling": "Read with a clear falling pitch at the end, as a calm declarative statement.",
                "rise_fall": (
                    "Make the voice rise through the first half, then clearly fall through the second half."
                ),
                "fall_rise": (
                    "Let the pitch fall through the first half, then clearly rise through the second half."
                ),
            }
            for kind, instruction in styles.items():
                for i in range(4):
                    text = rng.choice(templates).format(
                        name=rng.choice(names), object=rng.choice(objects), place=rng.choice(places)
                    )
                    result.append(
                        {
                            "request_id": f"{split}-intonation-{kind}-{i}",
                            "text": text,
                            "speaker": "Ryan",
                            "split": split,
                            "seed": rng.randrange(2**31),
                            "contour": kind,
                            "instruct": instruction,
                            "template_id": f"speech:{split}:intonation",
                        }
                    )
        if "pause_after_word" in cfg.tasks:
            for i in range(max(8, cfg.data.speech_per_split // 4)):
                name, obj, place = rng.choice(names), rng.choice(objects), rng.choice(places)
                phrases = (
                    [f"{name} carried {obj}", f"while walking towards {place} in the morning"]
                    if split == "train"
                    else [f"After the visit to {place}", f"{name} returned home with {obj} before dinner"]
                )
                pair = f"{split}-pause-{i}"
                for role, text in zip(("prefix", "suffix"), phrases, strict=True):
                    result.append(
                        {
                            "request_id": f"{pair}-{role}",
                            "text": text,
                            "speaker": "Ryan",
                            "split": split,
                            "seed": rng.randrange(2**31),
                            "pause_pair": pair,
                            "pause_role": role,
                            "template_id": f"pause-source:{split}",
                        }
                    )
    return result


def create_music(cfg, root: Path) -> list[dict]:
    import mido

    binary = shutil.which(cfg.data.fluidsynth)
    if not binary:
        raise FileNotFoundError("FluidSynth is required. Install fluidsynth or set data.fluidsynth.")
    font = Path(cfg.data.soundfont).expanduser()
    if not font.is_file():
        raise FileNotFoundError(f"Soundfont not found: {font}. Set data.soundfont to a .sf2 file.")
    rows = []
    for split in ("train", "heldout"):
        for i in range(cfg.data.music_per_split):
            rng = random.Random(int(digest([cfg.seed, split, "midi", i])[:15], 16))
            bpm = rng.choice([90, 100, 110, 120, 130, 140])
            notes = [rng.choice([48, 50, 52, 55, 57, 60]) for _ in range(12)]
            program = rng.choice([0, 4, 10, 11, 12])
            key = f"{split}-music-{i:04d}"
            midi_path, audio_path = root / f"{key}.mid", root / f"{key}.wav"
            if not audio_path.exists():
                midi = mido.MidiFile(ticks_per_beat=480)
                track = mido.MidiTrack()
                midi.tracks.append(track)
                track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
                track.append(mido.Message("program_change", channel=0, program=program))
                for pitch in notes:
                    track.append(mido.Message("note_on", note=pitch, velocity=85, time=0))
                    track.append(mido.Message("note_off", note=pitch, velocity=0, time=360))
                    track.append(mido.MetaMessage("marker", text="beat", time=120))
                root.mkdir(parents=True, exist_ok=True)
                midi.save(midi_path)
                subprocess.run(
                    [
                        binary,
                        "-ni",
                        "-F",
                        str(audio_path),
                        "-r",
                        str(SAMPLE_RATE),
                        "-o",
                        "synth.reverb.active=0",
                        "-o",
                        "synth.chorus.active=0",
                        str(font),
                        str(midi_path),
                    ],
                    check=True,
                    capture_output=True,
                )
                clip = T.trim_silence(Clip.load(audio_path), threshold_db=-45)
                T.peak_limit(T.normalize(clip)).save(audio_path)
            rows.append(
                {
                    "kind": "music",
                    "path": str(audio_path.resolve()),
                    "split": split,
                    "source_id": f"midi:{digest([split, notes, bpm, program])}",
                    "bpm": bpm,
                    "notes": notes,
                    "program": program,
                    "monophonic": True,
                }
            )
    return rows


def create_sounds(cfg, root: Path) -> list[dict]:
    rows = []
    labels = ["low-pitched tone", "high-pitched tone", "noise burst", "click"]
    for split in ("train", "heldout"):
        for i in range(24):
            kind = i % 4
            rng = np.random.default_rng(int(digest([cfg.seed, split, "sound", i])[:15], 16))
            seconds = float(rng.uniform(0.25, 0.55)) if kind == 3 else float(rng.uniform(0.5, 0.9))
            n = int(seconds * SAMPLE_RATE)
            t = np.arange(n) / SAMPLE_RATE
            if kind < 2:
                freq = rng.uniform(170, 230) if kind == 0 else rng.uniform(1600, 2200)
                wave = np.sin(2 * np.pi * freq * t)
            else:
                wave = rng.normal(0, 0.25, n)
                if kind == 3:
                    wave *= np.exp(-t * 45)
            wave = (wave * np.sin(np.linspace(0, np.pi, n)) ** 0.2 * 0.2).astype(np.float32)
            path = root / f"{split}-sound-{i:03d}.wav"
            T.peak_limit(T.normalize(Clip(wave))).save(path)
            rows.append(
                {
                    "kind": "sound",
                    "path": str(path.resolve()),
                    "split": split,
                    "label": labels[kind],
                    "source_id": f"procedural:{digest([cfg.seed, split, i])}",
                }
            )
    return rows


def _decode_audio(value, root: Path, cache: Path) -> Path:
    if isinstance(value, str):
        path = Path(value)
        return path if path.is_absolute() else root / path
    if isinstance(value, dict) and value.get("bytes"):
        payload = value["bytes"]
        import hashlib

        key = hashlib.sha256(payload).hexdigest()
        path = cache / f"{key}.wav"
        if not path.exists():
            wave, rate = sf.read(io.BytesIO(payload), dtype="float32")
            cache.mkdir(parents=True, exist_ok=True)
            sf.write(path, wave, rate)
        return path
    if isinstance(value, dict) and value.get("path"):
        return _decode_audio(value["path"], root, cache)
    raise ValueError("An audio path or embedded audio is required")


def _tables(root: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for path in sorted(root.rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            yield from batch.to_pylist()
    for path in sorted(root.rglob("*.arrow")):
        with pa.memory_map(str(path)) as f:
            try:
                reader = pa.ipc.open_stream(f)
            except pa.ArrowInvalid:
                f.seek(0)
                reader = pa.ipc.open_file(f)
                for i in range(reader.num_record_batches):
                    yield from reader.get_batch(i).to_pylist()
            else:
                for batch in reader:
                    yield from batch.to_pylist()


def import_recordings(cfg) -> list[dict]:
    rows = []
    cache = cfg.root / "assets" / "decoded"
    if cfg.data.manifest:
        path = Path(cfg.data.manifest).expanduser().resolve()
        for row in read_jsonl(path):
            row = dict(row)
            row["path"] = str(_decode_audio(row["path"], path.parent, cache).resolve())
            row.setdefault("source_id", f"file:{file_digest(row['path'])}")
            row.setdefault("split", split_for(row["source_id"], cfg.seed))
            rows.append(row)
    if cfg.data.librispeech:
        root = Path(cfg.data.librispeech).expanduser().resolve()
        for path in sorted(root.rglob("*.trans.txt")):
            for line in path.read_text().splitlines():
                ident, text = line.split(" ", 1)
                audio = path.parent / f"{ident}.flac"
                if audio.exists():
                    rows.append(
                        {
                            "kind": "speech",
                            "path": str(audio),
                            "text": text,
                            "speaker": ident.split("-")[0],
                            "source_id": f"librispeech:{ident}",
                            "split": split_for(f"libri-speaker:{ident.split('-')[0]}", cfg.seed),
                        }
                    )
        if not any(r["source_id"].startswith("librispeech:") for r in rows):
            for row in _tables(root):
                ident = str(row.get("id", ""))
                rows.append(
                    {
                        "kind": "speech",
                        "path": str(_decode_audio(row["audio"], root, cache)),
                        "text": row["text"],
                        "speaker": str(row["speaker_id"]),
                        "source_id": f"librispeech:{ident}",
                        "split": split_for(f"libri-speaker:{row['speaker_id']}", cfg.seed),
                    }
                )
    if cfg.data.fsd50k:
        root = Path(cfg.data.fsd50k).expanduser().resolve()
        label_csv = next(iter(sorted(root.rglob("dev.csv"))), None)
        if label_csv is None:
            raise FileNotFoundError(f"FSD50K dev.csv is missing under {root}")
        for row in csv.DictReader(label_csv.open()):
            if "," in row["labels"]:
                continue
            audio = root / "FSD50K.dev_audio" / f"{row['fname']}.wav"
            if audio.exists():
                rows.append(
                    {
                        "kind": "sound",
                        "path": str(audio),
                        "label": row["labels"].replace("_", " ").lower(),
                        "source_id": f"fsd50k:{row['fname']}",
                        "split": split_for(f"fsd50k:{row['fname']}", cfg.seed),
                    }
                )
    if cfg.data.audioset:
        root = Path(cfg.data.audioset).expanduser().resolve()
        manifest = root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"AudioSet requires {manifest} with path, label, kind and source_id fields"
            )
        for row in read_jsonl(manifest):
            row = dict(row)
            row["path"] = str(_decode_audio(row["path"], root, cache).resolve())
            row.setdefault("split", split_for(row["source_id"], cfg.seed))
            rows.append(row)
    return rows


def prepare_assets(cfg):
    from evoaudio.tasks.catalog import TASKS

    root = cfg.root / "assets"
    root.mkdir(parents=True, exist_ok=True)
    identity = digest(
        {
            "data": cfg.data.model_dump(),
            "tasks": cfg.tasks,
            "seed": cfg.seed,
            "tts": cfg.tts_model,
            "whisper": cfg.whisper_model,
        }
    )
    manifest = root / "manifest.jsonl"
    receipt = root / "prepared.json"
    inputs = root / "inputs.json"
    if inputs.exists() and read_json(inputs)["identity"] != identity:
        raise ValueError("Asset settings changed. Choose a new output_dir.")
    if receipt.exists() and read_json(receipt)["identity"] != identity:
        raise ValueError("Asset settings changed. Choose a new output_dir.")
    write_json(inputs, {"identity": identity})
    if receipt.exists():
        if manifest.exists():
            cached = read_jsonl(manifest)
            if all(Path(r["path"]).is_file() for r in cached):
                for row in cached:
                    stat = Path(row["path"]).stat()
                    if row.get("size") != stat.st_size or row.get("mtime_ns") != stat.st_mtime_ns:
                        if file_digest(row["path"]) != row["file_digest"]:
                            raise ValueError("A prepared source recording changed. Choose a new output_dir.")
                return manifest
    rows = import_recordings(cfg)
    kinds = {TASKS[t].source for t in cfg.tasks}
    if kinds & {"speech", "intonation"}:
        requests = root / "speech_requests.jsonl"
        write_jsonl(requests, speech_requests(cfg))
        output = root / "speech.jsonl"
        run_worker(
            cfg,
            "evoaudio.tts",
            [
                "--requests",
                str(requests),
                "--output",
                str(output),
                "--model",
                cfg.tts_model,
                "--whisper",
                cfg.whisper_model,
            ],
            root / "tts.log",
            python=cfg.resources.tts_python,
        )
        rows += read_jsonl(output)
    if "music" in kinds:
        rows += create_music(cfg, root / "music")
    if "sound" in kinds:
        rows += create_sounds(cfg, root / "sounds")
    seen = {}
    valid = []
    for row in rows:
        if row.get("split") not in {"train", "heldout"}:
            raise ValueError("Source split must be train or heldout")
        path = Path(row["path"])
        clip = Clip.load(path)
        if not 0.15 <= clip.duration <= 30:
            continue
        pcm_id = hashlib.sha256(clip.wave.astype("<f4").tobytes()).hexdigest()
        if pcm_id in seen and seen[pcm_id] != row["split"]:
            raise ValueError("The same recording occurs in both source partitions")
        if row["source_id"] in seen and seen[row["source_id"]] != row["split"]:
            raise ValueError("The same source ID occurs in both partitions")
        seen[pcm_id] = row["split"]
        seen[row["source_id"]] = row["split"]
        stat = path.stat()
        valid.append(
            {
                **row,
                "duration": clip.duration,
                "file_digest": file_digest(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    for split in ("train", "heldout"):
        for kind in kinds - {"intonation"}:
            if len([r for r in valid if r["split"] == split and r["kind"] == kind]) < 4:
                raise RuntimeError(f"Not enough verified {kind} sources for {split}")
    write_jsonl(manifest, valid)
    write_json(receipt, {"identity": identity, "sources": len(valid)})
    log(f"Prepared {len(valid)} source recordings")
    return manifest
