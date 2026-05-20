from __future__ import annotations

import os
import struct
from collections.abc import Mapping

from .postings import encode_postings

_RECORD_HEADER = struct.Struct("<II")  # term_nbytes, postings_nbytes


class PartialIndexWriter:
    """Writes sorted term -> encoded postings records to a binary file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.file = open(path, "wb")

    def write_record(self, term: str, encoded_postings: bytes) -> None:
        term_bytes = term.encode("utf-8")
        self.file.write(_RECORD_HEADER.pack(len(term_bytes), len(encoded_postings)))
        self.file.write(term_bytes)
        self.file.write(encoded_postings)

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "PartialIndexWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_partial_index(
    term_to_postings: Mapping[str, Mapping[int, int]],
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with PartialIndexWriter(output_path) as writer:
        for term in sorted(term_to_postings):
            postings = sorted(term_to_postings[term].items())
            encoded = encode_postings(postings)
            writer.write_record(term, encoded)


class PartialIndexReader:
    """Sequential reader for partial index records."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.file = open(path, "rb")

    def read_next(self) -> tuple[str, bytes] | None:
        header = self.file.read(_RECORD_HEADER.size)
        if not header:
            return None
        if len(header) != _RECORD_HEADER.size:
            raise ValueError(f"corrupted partial index header in {self.path}")
        term_nbytes, postings_nbytes = _RECORD_HEADER.unpack(header)
        term_bytes = self.file.read(term_nbytes)
        postings_bytes = self.file.read(postings_nbytes)
        if len(term_bytes) != term_nbytes or len(postings_bytes) != postings_nbytes:
            raise ValueError(f"corrupted partial index record in {self.path}")
        return term_bytes.decode("utf-8"), postings_bytes

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "PartialIndexReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
