from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class RawDocument:
    internal_id: int
    external_id: str
    text: str


class CorpusReader:
    """Streams JSONL documents from disk.
    """

    ID_FIELDS = ("id", "ID", "doc_id", "document_id", "entity_id", "wikidata_id")
    TEXT_FIELDS = (
        "title",
        "name",
        "text",
        "description",
        "descriptive_text",
        "abstract",
        "keywords",
        "categories",
        "aliases",
    )

    def __init__(self, corpus_path: str) -> None:
        self.corpus_path = corpus_path

    def __iter__(self) -> Iterator[RawDocument]:
        with open(self.corpus_path, "rt", encoding="utf-8", errors="replace") as file:
            for internal_id, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Skip bad lines.
                    continue
                external_id = self._extract_external_id(obj, internal_id)
                text = self._extract_text(obj)
                yield RawDocument(
                    internal_id=internal_id, external_id=external_id, text=text
                )

    @classmethod
    def _extract_external_id(cls, obj: dict[str, Any], fallback: int) -> str:
        for field in cls.ID_FIELDS:
            value = obj.get(field)
            if value is not None:
                return str(value)
        return str(fallback)

    @classmethod
    def _flatten_value(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return " ".join(cls._flatten_value(item) for item in value)
        if isinstance(value, dict):
            return " ".join(cls._flatten_value(item) for item in value.values())
        return str(value)

    @classmethod
    def _extract_text(cls, obj: dict[str, Any]) -> str:
        parts: list[str] = []
        for field in cls.TEXT_FIELDS:
            if field in obj:
                parts.append(cls._flatten_value(obj[field]))

        if not parts:
            parts = [cls._flatten_value(value) for value in obj.values()]
        return " ".join(part for part in parts if part)
