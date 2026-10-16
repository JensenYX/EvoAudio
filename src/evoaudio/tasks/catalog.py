from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    domain: str
    description: str
    source: str


TASKS = {
    "speed_pattern3": Task("speech", "Order the speaking rates of three utterances", "speech"),
    "pitch_pattern3": Task("speech", "Order the voice pitches of three utterances", "speech"),
    "volume_pattern3": Task("speech", "Order the loudness of three utterances", "speech"),
    "sentence_intonation": Task("speech", "Identify a measured intonation contour", "intonation"),
    "pause_after_word": Task("speech", "Locate an inserted pause after a spoken word", "speech"),
    "speech_pair_compare": Task("speech", "Compare voice pitch, rate, loudness or duration", "speech"),
    "sound_count": Task("sound", "Count separated sound events", "sound"),
    "sound_order": Task("sound", "Identify the order of two labeled events", "sound"),
    "sound_louder": Task("sound", "Compare the loudness of two sound clips", "sound"),
    "sound_duration": Task("sound", "Compare the duration of two sound clips", "sound"),
    "music_tempo_change": Task("music", "Identify a tempo change within a musical clip", "music"),
    "music_segment_louder": Task("music", "Identify the louder musical segment", "music"),
    "music_dynamics": Task("music", "Identify a measured crescendo or decrescendo", "music"),
    "music_pitch_compare": Task("music", "Compare the pitch of two musical excerpts", "music"),
    "music_speed_compare": Task("music", "Compare the tempo of two musical excerpts", "music"),
    "music_repeat": Task("music", "Identify a repeated musical passage", "music"),
    "compound_speech": Task(
        "speech", "Retrieve words from the voice with a specified acoustic attribute", "speech"
    ),
}


TEMPLATES = {
    "pattern": (
        "Three utterances are played in sequence. "
        "Which pattern describes their {attribute}, from first to third?",
        "Listen to the three spoken segments. How are their {attribute} ordered?",
        "Choose the sequence of {attribute} levels that matches the first, second, and third utterances.",
        "What is the correct low, medium, and high ordering of {attribute} across this recording?",
    ),
    "pair": (
        "How does the second {source} compare with the first in {attribute}?",
        "Listen to the two {source} excerpts. The second one has which {attribute} relative to the first?",
        "Compare the {attribute} of the first and second {source} segments. Select the relationship.",
        "Which option correctly describes the {attribute} change from the first {source} to the second?",
    ),
    "intonation": (
        "Which overall voice-pitch contour is heard in this sentence?",
        "How does the speaker's intonation move over the sentence?",
        "Select the intonation pattern that matches this utterance.",
        "Describe the rise and fall of the voice in this spoken sentence.",
    ),
    "pause": (
        "Which word is immediately followed by the longest pause?",
        "After which word does the speaker make the longest silence?",
        "Identify the word just before the most noticeable pause in this recording.",
        "Listen for the longest break in speech. Which word comes immediately before it?",
    ),
    "count": (
        "How many separate occurrences of the {label} can be heard?",
        "Count the distinct {label} events in this recording.",
        "How many times does the {label} occur in the audio?",
        "What is the number of audible {label} events?",
    ),
    "order": (
        "Which sound is heard first?",
        "What is the first sound event in the recording?",
        "Listen to the sequence. Select the event that occurs before the other event.",
        "Which of these events opens the sound sequence?",
    ),
    "dynamics": (
        "How does the music's loudness change through the recording?",
        "Which dynamic pattern describes this musical passage?",
        "Listen to the beginning and end. What happens to the loudness of the music?",
        "Does the musical passage get louder, quieter, or stay at the same level?",
    ),
    "repeat": (
        "Which two of the three musical segments repeat the same passage?",
        "Listen to the three music clips in order. Which pair is identical?",
        "Identify the repeated musical passage by its two positions.",
        "Of the first, second, and third musical segments, which two repeat?",
    ),
    "compound": (
        "What words are spoken in the utterance with the {attribute} voice?",
        "Select the words spoken in the {attribute} of the three utterances.",
        "Listen to the three voices. Which sentence belongs to the {attribute} one?",
        "Identify the spoken content of the segment with the {attribute} voice.",
    ),
}


def question(kind: str, split: str, rng, **values) -> tuple[str, str]:
    indices = (0, 1) if split == "train" else (2, 3)
    index = rng.choice(indices)
    return TEMPLATES[kind][index].format(**values), f"{kind}:{index}"
