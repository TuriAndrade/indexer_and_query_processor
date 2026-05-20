from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class LexiconEntry:
    document_frequency: int
    collection_frequency: int
    offset: int
    nbytes: int


def load_lexicon(path: str) -> dict[str, LexiconEntry]:
    lexicon: dict[str, LexiconEntry] = {}
    with open(path, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            term, df, cf, offset, nbytes = line.split("\t")
            lexicon[term] = LexiconEntry(
                document_frequency=int(df),
                collection_frequency=int(cf),
                offset=int(offset),
                nbytes=int(nbytes),
            )
    return lexicon


def iter_lexicon(path: str) -> Iterator[tuple[str, LexiconEntry]]:
    with open(path, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            term, df, cf, offset, nbytes = line.split("\t")
            yield term, LexiconEntry(int(df), int(cf), int(offset), int(nbytes))
