from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from evoaudio.audio import SAMPLE_RATE, Clip
from evoaudio.audio.measure import f0_contour
from evoaudio.io import digest, log, read_jsonl, write_jsonl


def transcript_match(expected: str, actual: str) -> float:
    ref = re.findall(r"[a-z']+", expected.casefold())
    hyp = re.findall(r"[a-z']+", actual.casefold())
    row = [0] * (len(hyp) + 1)
    for word in ref:
        previous = 0
        for j, other in enumerate(hyp, 1):
            old = row[j]
            row[j] = previous + 1 if word == other else max(row[j], row[j - 1])
            previous = old
    return row[-1] / max(len(ref), len(hyp), 1)


def synthesize(requests: Path, output: Path, model_id: str, whisper_id: str) -> None:
    import torch
    from qwen_tts import Qwen3TTSModel
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    rows = read_jsonl(requests)
    existing = [r for r in read_jsonl(output) if Path(r["path"]).is_file()] if output.exists() else []
    done = {r["request_id"] for r in existing}
    pending = [r for r in rows if r["request_id"] not in done]
    if not pending:
        return
    model = Qwen3TTSModel.from_pretrained(
        model_id, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(whisper_id)
    asr = (
        AutoModelForSpeechSeq2Seq.from_pretrained(whisper_id, dtype=torch.float16, low_cpu_mem_usage=True)
        .to("cuda:0")
        .eval()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for i, req in enumerate(pending):
        accepted = None
        for attempt in range(3):
            torch.manual_seed(req["seed"] + attempt)
            waves, sr = model.generate_custom_voice(
                text=req["text"],
                language="English",
                speaker=req["speaker"],
                instruct=req.get("instruct", "Speak clearly at a natural pace."),
                non_streaming_mode=True,
                max_new_tokens=1024,
            )
            wave = np.asarray(waves[0], dtype=np.float32)
            if sr != SAMPLE_RATE:
                import librosa

                wave = librosa.resample(wave, orig_sr=sr, target_sr=SAMPLE_RATE)
            if not np.isfinite(wave).all() or not 0.5 <= len(wave) / SAMPLE_RATE <= 20:
                continue
            encoded = processor(
                wave, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True
            )
            with torch.inference_mode():
                ids = asr.generate(
                    encoded.input_features.to("cuda:0", dtype=torch.float16),
                    attention_mask=encoded.attention_mask.to("cuda:0"),
                    language="english",
                    task="transcribe",
                    max_new_tokens=128,
                )
            transcript = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            score = transcript_match(req["text"], transcript)
            if score < 0.9:
                continue
            contour = f0_contour(wave) if req.get("contour") else None
            if req.get("contour") and (contour or {}).get("shape") != req["contour"]:
                continue
            path = output.parent / "speech" / f"{req['request_id']}.wav"
            Clip(wave).save(path)
            accepted = {
                **req,
                "path": str(path.resolve()),
                "kind": "speech",
                "transcript": transcript,
                "asr_match": score,
                "source_id": f"tts:{digest(req['text'])}",
                "contour_check": contour,
            }
            break
        if accepted:
            existing.append(accepted)
            write_jsonl(output, existing)
        log(f"TTS {i + 1}/{len(pending)}: {'accepted' if accepted else 'discarded'}")
    if not existing:
        raise RuntimeError("No synthesized utterances passed speech verification")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--whisper", required=True)
    args = p.parse_args()
    synthesize(args.requests, args.output, args.model, args.whisper)


if __name__ == "__main__":
    main()
