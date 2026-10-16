from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

_OPTION_PREFIX = re.compile("^\\s*[\\[(]?([A-Za-z0-9]+)[\\])\\.:-]\\s*(.+?)\\s*$")
_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}


class BenchmarkExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    benchmark: str
    example_id: str
    audio_path: Path
    audio_paths: list[Path] = Field(default_factory=list)
    question: str
    options: list[dict[str, str]] = Field(default_factory=list)
    answer: str | None = None
    category: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def populate_audio_paths(self) -> BenchmarkExample:
        if not self.audio_paths:
            self.audio_paths = [self.audio_path]
        elif self.audio_paths[0] != self.audio_path:
            raise ValueError("audio_path must be the first entry in audio_paths")
        return self


@runtime_checkable
class BenchmarkAdapter(Protocol):
    name: str
    root: Path

    def validate_root(self) -> None: ...

    def iter_examples(self, split: str) -> Iterator[BenchmarkExample]: ...


def _option_id(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else str(index + 1)


def _split_option(value: str, fallback_id: str) -> tuple[str, str]:
    match = _OPTION_PREFIX.match(value)
    if match and fallback_id and (match.group(1).casefold() == fallback_id.casefold()):
        return (match.group(1).upper(), match.group(2).strip())
    return (fallback_id, value.strip())


def _normalize_options(value: Any) -> list[dict[str, str]]:
    if isinstance(value, Mapping):
        mapping_result: list[dict[str, str]] = []
        for key, text in value.items():
            option_id = str(key)
            _, normalized_text = _split_option(str(text), option_id)
            mapping_result.append({"id": option_id, "text": normalized_text})
        return mapping_result
    if isinstance(value, Sequence) and (not isinstance(value, str | bytes | bytearray)):
        result: list[dict[str, str]] = []
        for index, option in enumerate(value):
            fallback_id = _option_id(index)
            if isinstance(option, Mapping):
                option_id = str(option.get("id", fallback_id))
                text = str(option.get("text", option.get("value", "")))
                _, text = _split_option(text, option_id)
            else:
                option_id, text = _split_option(str(option), fallback_id)
            result.append({"id": option_id, "text": text})
        return result
    return []


def _normalized_answer_text(value: str) -> str:
    _, text = _split_option(value, "")
    return " ".join(text.casefold().split())


def normalize_option_answer(value: Any, options: list[dict[str, str]]) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if not options:
        return _normalized_answer_text(raw)
    by_id = {option["id"].casefold(): option["id"] for option in options}
    if raw.casefold() in by_id:
        return by_id[raw.casefold()]
    prefix = _OPTION_PREFIX.match(raw)
    if prefix and prefix.group(1).casefold() in by_id:
        return by_id[prefix.group(1).casefold()]
    normalized = _normalized_answer_text(raw)
    matches = [option["id"] for option in options if _normalized_answer_text(option["text"]) == normalized]
    return matches[0] if len(matches) == 1 else None


def _normalize_category(value: Any, *, default: str = "unknown") -> str:
    if isinstance(value, list | tuple):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " / ".join(parts) if parts else default
    if value is None:
        return default
    normalized = str(value).strip()
    return normalized or default


def _official_fields(row: Mapping[str, Any], *, exclude: set[str]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items() if key not in exclude}


def _require_pyarrow() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Native benchmark Arrow/Parquet data requires pyarrow; "
            "install `evoaudio[eval]` or `pyarrow` in the evaluation environment"
        ) from error
    return (pa, ipc, pq)


def _iter_native_rows(paths: Sequence[Path]) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    pa, ipc, pq = _require_pyarrow()
    for path in paths:
        row_offset = 0
        if path.suffix == ".parquet":
            batches = pq.ParquetFile(path).iter_batches(batch_size=16)
            resource = None
        else:
            resource = pa.memory_map(str(path), "r")
            batches = ipc.open_stream(resource)
        try:
            for batch in batches:
                for batch_index in range(batch.num_rows):
                    row = batch.slice(batch_index, 1).to_pylist()[0]
                    if not isinstance(row, dict):
                        raise ValueError(f"Native benchmark row is not an object: {path}:{row_offset}")
                    yield (path, row_offset, row)
                    row_offset += 1
        finally:
            if resource is not None:
                resource.close()


def _one_snapshot(paths: Sequence[Path]) -> list[Path]:
    grouped: dict[Path, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.parent].append(path)
    if not grouped:
        return []
    _, selected = max(grouped.items(), key=lambda item: (len(item[1]), str(item[0])))
    return sorted(selected)


class JsonlBenchmarkAdapter:
    name = "generic"
    manifest_candidates: tuple[str, ...] = ("test.jsonl", "metadata.jsonl")

    def __init__(self, root: Path, *, extraction_dir: Path | None = None) -> None:
        self.root = root.expanduser().resolve()
        selected_cache = extraction_dir or self.root / ".evoaudio_cache" / self.name
        self.extraction_dir = selected_cache.expanduser().resolve()

    def _jsonl_manifests(self) -> list[Path]:
        return sorted(self.root.rglob("*.jsonl")) if self.root.is_dir() else []

    def _native_sources(self, split: str) -> list[Path]:
        del split
        return []

    def validate_root(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.name} root does not exist: {self.root}")
        native = any((self._native_sources(split) for split in ("test", "train", "validation")))
        if not self._jsonl_manifests() and (not native):
            raise FileNotFoundError(
                f"No supported JSONL, Arrow, or Parquet data found for {self.name} under {self.root}"
            )

    def _manifest(self, split: str) -> Path | None:
        candidates = (f"{split}.jsonl", *self.manifest_candidates) if split == "test" else (f"{split}.jsonl",)
        manifests = self._jsonl_manifests()
        for name in candidates:
            direct = self.root / name
            if direct.is_file():
                return direct
            nested = next((path for path in manifests if path.name == name), None)
            if nested is not None:
                return nested
        return None

    def _resolve_audio_paths(self, value: Any, *, source: Path, base_dir: Path | None = None) -> list[Path]:
        values = value if isinstance(value, list | tuple) else [value]
        if not values or not all((isinstance(item, str) for item in values)):
            raise ValueError(f"Missing audio path at {source}")
        relative_base = (base_dir or self.root).resolve()
        if not relative_base.is_relative_to(self.root):
            raise ValueError(f"Benchmark media base escapes configured root: {relative_base}")
        result: list[Path] = []
        for item in values:
            audio_path = Path(item)
            if not audio_path.is_absolute():
                audio_path = (relative_base / audio_path).resolve()
            else:
                audio_path = audio_path.resolve()
            if not audio_path.is_relative_to(self.root):
                raise ValueError(f"Benchmark audio escapes configured root: {audio_path}")
            if not audio_path.is_file():
                raise FileNotFoundError(f"Benchmark audio does not exist: {audio_path}")
            result.append(audio_path)
        return result

    def _extract_audio(self, value: Any, *, source: Path) -> Path:
        if not isinstance(value, Mapping):
            raise ValueError(f"Missing embedded audio at {source}")
        payload = value.get("bytes")
        if not isinstance(payload, bytes | bytearray | memoryview):
            path_value = value.get("path")
            return self._resolve_audio_paths(path_value, source=source)[0]
        data = bytes(payload)
        digest = hashlib.sha256(data).hexdigest()
        source_suffix = Path(str(value.get("path") or "")).suffix.casefold()
        suffix = source_suffix if source_suffix in _AUDIO_SUFFIXES else self._audio_suffix(data)
        destination = (self.extraction_dir / f"{digest}{suffix}").resolve()
        if not destination.is_relative_to(self.extraction_dir):
            raise ValueError(f"Extracted benchmark audio escapes cache root: {destination}")
        if destination.is_file():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    @staticmethod
    def _audio_suffix(data: bytes) -> str:
        if data.startswith(b"RIFF"):
            return ".wav"
        if data.startswith(b"fLaC"):
            return ".flac"
        if data.startswith((b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
            return ".mp3"
        if data.startswith(b"OggS"):
            return ".ogg"
        return ".audio"

    def _jsonl_example(
        self, row: Mapping[str, Any], *, manifest: Path, line_number: int, split: str
    ) -> BenchmarkExample:
        audio_value = row.get("audio_paths", row.get("audio_path", row.get("audio", row.get("path"))))
        audio_paths = self._resolve_audio_paths(audio_value, source=Path(f"{manifest}:{line_number}"))
        options = _normalize_options(row.get("options", row.get("choices", [])))
        raw_answer = row.get("answer")
        known_fields = {
            "answer",
            "audio",
            "audio_path",
            "audio_paths",
            "category",
            "choices",
            "id",
            "options",
            "path",
            "query",
            "question",
            "question_id",
            "task",
        }
        metadata = {
            "source_manifest": str(manifest),
            "source_split": split,
            "native_format": "jsonl",
            "official_answer": raw_answer,
            "official_fields": _official_fields(row, exclude=known_fields),
        }
        mapped_answer = normalize_option_answer(raw_answer, options)
        answer = (
            mapped_answer
            if mapped_answer is not None
            else str(raw_answer)
            if raw_answer is not None
            else None
        )
        return BenchmarkExample(
            benchmark=self.name,
            example_id=str(row.get("id", row.get("question_id", line_number - 1))),
            audio_path=audio_paths[0],
            audio_paths=audio_paths,
            question=str(row.get("question", row.get("query", ""))),
            options=options,
            answer=answer,
            category=_normalize_category(row.get("category", row.get("task"))),
            metadata=metadata,
        )

    def _native_example(
        self, row: Mapping[str, Any], *, source: Path, row_index: int, split: str
    ) -> BenchmarkExample:
        raise ValueError(f"{self.name} does not define a native reader for {source}:{row_index}")

    def iter_examples(self, split: str = "test") -> Iterator[BenchmarkExample]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.name} root does not exist: {self.root}")
        manifest = self._manifest(split)
        if manifest is not None:
            with manifest.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if line.strip():
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError(f"Benchmark row is not an object: {manifest}:{line_number}")
                        yield self._jsonl_example(
                            row, manifest=manifest, line_number=line_number, split=split
                        )
            return
        sources = self._native_sources(split)
        if not sources:
            raise FileNotFoundError(
                f"No {split!r} manifest or native Arrow/Parquet data found for {self.name} under {self.root}"
            )
        for source, row_index, row in _iter_native_rows(sources):
            yield self._native_example(row, source=source, row_index=row_index, split=split)


class MMSUAdapter(JsonlBenchmarkAdapter):
    name = "mmsu"

    def _native_sources(self, split: str) -> list[Path]:
        paths = sorted(self.root.rglob(f"mmsu-{split}-*.arrow"))
        if not paths and split == "test":
            paths = sorted(self.root.rglob("mmsu-train-*.arrow"))
        return _one_snapshot(paths)

    def _native_example(
        self, row: Mapping[str, Any], *, source: Path, row_index: int, split: str
    ) -> BenchmarkExample:
        options = _normalize_options(
            [row.get(f"choice_{letter.casefold()}", "") for letter in ("A", "B", "C", "D")]
        )
        raw_answer = row.get("answer_gt")
        audio_path = self._extract_audio(row.get("audio"), source=source)
        return BenchmarkExample(
            benchmark=self.name,
            example_id=str(row.get("id", f"{source.stem}:{row_index}")),
            audio_path=audio_path,
            audio_paths=[audio_path],
            question=str(row.get("question", "")),
            options=options,
            answer=normalize_option_answer(raw_answer, options),
            category=_normalize_category(row.get("category", row.get("task_name"))),
            metadata={
                "source_manifest": str(source),
                "source_split": "train" if "-train-" in source.name else split,
                "requested_split": split,
                "native_format": "huggingface_arrow",
                "official_answer": raw_answer,
                "official_fields": _official_fields(row, exclude={"audio"}),
            },
        )


class MMAUProAdapter(JsonlBenchmarkAdapter):
    name = "mmau_pro"

    def _native_sources(self, split: str) -> list[Path]:
        parquet = sorted(self.root.rglob(f"{split}.parquet"))
        if parquet:
            return _one_snapshot(parquet)
        arrow = sorted(self.root.rglob(f"mmau-pro-{split}*.arrow"))
        return _one_snapshot(arrow)

    def _native_example(
        self, row: Mapping[str, Any], *, source: Path, row_index: int, split: str
    ) -> BenchmarkExample:
        options = _normalize_options(row.get("choices", []))
        raw_answer = row.get("answer")
        audio_paths = self._resolve_audio_paths(row.get("audio_path"), source=source)
        return BenchmarkExample(
            benchmark=self.name,
            example_id=str(row.get("id", f"{source.stem}:{row_index}")),
            audio_path=audio_paths[0],
            audio_paths=audio_paths,
            question=str(row.get("question", "")),
            options=options,
            answer=normalize_option_answer(raw_answer, options),
            category=_normalize_category(row.get("category")),
            metadata={
                "source_manifest": str(source),
                "source_split": split,
                "native_format": "parquet" if source.suffix == ".parquet" else "huggingface_arrow",
                "official_answer": raw_answer,
                "official_fields": _official_fields(row, exclude=set()),
            },
        )


class MMARAdapter(JsonlBenchmarkAdapter):
    name = "mmar"
    manifest_candidates = ("MMAR-meta.jsonl", "test.jsonl", "metadata.jsonl")

    def _official_json_manifest(self) -> Path | None:
        direct = self.root / "MMAR-meta.json"
        if direct.is_file():
            return direct
        return next(iter(sorted(self.root.rglob("MMAR-meta.json"))), None)

    def validate_root(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.name} root does not exist: {self.root}")
        if self._manifest("test") is None and self._official_json_manifest() is None:
            raise FileNotFoundError(
                f"No MMAR-meta.jsonl, MMAR-meta.json, or test.jsonl data found for MMAR under {self.root}"
            )

    def _official_example(
        self, row: Mapping[str, Any], *, manifest: Path, row_index: int, split: str, native_format: str
    ) -> BenchmarkExample:
        options = _normalize_options(row.get("choices", []))
        raw_answer = row.get("answer")
        audio_paths = self._resolve_audio_paths(
            row.get("audio_path"), source=manifest, base_dir=manifest.parent
        )
        mapped_answer = normalize_option_answer(raw_answer, options)
        answer = (
            mapped_answer
            if mapped_answer is not None
            else str(raw_answer)
            if raw_answer is not None
            else None
        )
        return BenchmarkExample(
            benchmark=self.name,
            example_id=str(row.get("id", f"{manifest.stem}:{row_index}")),
            audio_path=audio_paths[0],
            audio_paths=audio_paths,
            question=str(row.get("question", "")),
            options=options,
            answer=answer,
            category=_normalize_category(row.get("category")),
            metadata={
                "source_manifest": str(manifest),
                "source_split": "test",
                "requested_split": split,
                "native_format": native_format,
                "official_answer": raw_answer,
                "modality": row.get("modality"),
                "sub_category": row.get("sub-category"),
                "language": row.get("language"),
                "source": row.get("source"),
                "url": row.get("url"),
                "timestamp": row.get("timestamp"),
                "official_fields": _official_fields(row, exclude={"audio_path"}),
            },
        )

    def _jsonl_example(
        self, row: Mapping[str, Any], *, manifest: Path, line_number: int, split: str
    ) -> BenchmarkExample:
        if manifest.name == "MMAR-meta.jsonl":
            return self._official_example(
                row, manifest=manifest, row_index=line_number - 1, split=split, native_format="mmar_jsonl"
            )
        return super()._jsonl_example(row, manifest=manifest, line_number=line_number, split=split)

    def iter_examples(self, split: str = "test") -> Iterator[BenchmarkExample]:
        manifest = self._manifest(split)
        if manifest is not None:
            yield from super().iter_examples(split)
            return
        if split != "test":
            raise FileNotFoundError(
                f"MMAR has only an official 'test' split; no {split!r} manifest exists under {self.root}"
            )
        official_json = self._official_json_manifest()
        if official_json is None:
            raise FileNotFoundError(f"No 'test' manifest or official MMAR metadata found under {self.root}")
        with official_json.open(encoding="utf-8") as stream:
            rows = json.load(stream)
        if not isinstance(rows, list):
            raise ValueError(f"Official MMAR metadata is not a JSON array: {official_json}")
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Official MMAR row is not an object: {official_json}:{row_index}")
            yield self._official_example(
                row, manifest=official_json, row_index=row_index, split=split, native_format="mmar_json"
            )


def create_adapter(name: str, root: Path, *, extraction_dir: Path | None = None) -> JsonlBenchmarkAdapter:
    try:
        adapter_type = ADAPTERS[name.casefold()]
    except KeyError as error:
        raise ValueError(f"Unsupported benchmark: {name}") from error
    return adapter_type(root, extraction_dir=extraction_dir)


ADAPTERS = {"mmsu": MMSUAdapter, "mmau_pro": MMAUProAdapter, "mmar": MMARAdapter}
