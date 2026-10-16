from __future__ import annotations

import json
import re


def option_id(answer, options: list[dict]) -> str | None:
    if not isinstance(answer, str):
        return None
    raw = answer.strip()
    by_id = {o["id"].casefold(): o["id"] for o in options}
    if raw.casefold() in by_id:
        return by_id[raw.casefold()]
    match = re.fullmatch(r"([A-Za-z])[.):]\s*(.*)", raw, re.DOTALL)
    if match and match[1].casefold() in by_id:
        chosen = next(o for o in options if o["id"].casefold() == match[1].casefold())
        if not match[2] or normalize(match[2]) == normalize(chosen["text"]):
            return chosen["id"]
    hits = [o["id"] for o in options if normalize(raw) == normalize(o["text"])]
    return hits[0] if len(hits) == 1 else None


def normalize(text: str) -> str:
    return " ".join(text.casefold().strip(" .").split())


def parse_answer(raw: str, options: list[dict]) -> str | None:
    text = str(raw).strip()
    fence = re.fullmatch(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if fence:
        text = fence[1]

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        obj = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"answer"}:
        return None
    return option_id(obj["answer"], options)


def reward(raw: str, gold: str, options: list[dict]) -> float:
    prediction = parse_answer(raw, options)
    return -1.0 if prediction is None else float(prediction == gold)


def prompt(item: dict, question_only: bool = False) -> str:
    audio = "" if question_only else "<audio>\n" * len(item["audio"])
    choices = "\n".join(f"{o['id']}. {o['text']}" for o in item["options"])
    return (
        f"{audio}{item['question']}\n{choices}\n"
        'Answer with one JSON object containing only the option letter, for example {"answer": "A"}.'
    )


def training_row(item: dict) -> dict:
    return {
        "messages": [{"role": "user", "content": prompt(item)}],
        "audios": item["audio"],
        "answer_option_id": item["answer"],
        "solution": item["answer"],
        "options_json": json.dumps(item["options"], ensure_ascii=False),
        "item_id": item["item_id"],
        "skill": item["skill"],
    }
