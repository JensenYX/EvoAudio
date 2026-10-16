import json
import os
from pathlib import Path

from swift.rewards import ORM, orms

from evoaudio.answers import reward


class AudioAnswerReward(ORM):
    def __call__(self, completions, answer_option_id, options_json, **kwargs):
        rewards = [
            reward(text, gold, json.loads(options))
            for text, gold, options in zip(completions, answer_option_id, options_json, strict=True)
        ]
        output = os.environ.get("EVOAUDIO_REWARD_LOG")
        if output:
            path = Path(output) / f"rank{os.environ.get('RANK', '0')}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            step = getattr(kwargs.get("trainer_state"), "global_step", 0)
            ids, skills = kwargs.get("item_id", []), kwargs.get("skill", [])
            with path.open("a") as f:
                for i, value in enumerate(rewards):
                    f.write(
                        json.dumps({"step": step, "item_id": ids[i], "skill": skills[i], "reward": value})
                        + "\n"
                    )
        return rewards


orms["evoaudio_answer"] = AudioAnswerReward
