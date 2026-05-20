from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass

_MAGIC = b"PA2DOC1\0"
_HEADER = struct.Struct("<8sQQd")  # magic, n_docs, total_tokens, avg_doc_len
_UINT32 = struct.Struct("<I")
_UINT64 = struct.Struct("<Q")


@dataclass(frozen=True)
class DocumentCollectionStats:
    number_of_documents: int
    total_tokens: int
    average_document_length: float


class DocumentIndexBuilder:
    """Streams document metadat to a temporary file and finalizes a binary index.
    """

    def __init__(self, temp_path: str) -> None:
        self.temp_path = temp_path
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        self.temp_file = open(temp_path, "wt", encoding="utf-8")
        self.number_of_documents = 0
        self.total_tokens = 0

    def add_document(
        self, internal_docid: int, external_id: str, document_length: int
    ) -> None:
        # external_id may contain whitespace, so escape tabs/newlines minimally.
        safe_id = external_id.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        self.temp_file.write(f"{internal_docid}\t{document_length}\t{safe_id}\n")
        self.number_of_documents += 1
        self.total_tokens += document_length

    def close(self) -> None:
        self.temp_file.close()

    def finalize(
        self, output_path: str, remove_temp: bool = True
    ) -> DocumentCollectionStats:
        self.close()
        n_docs = self.number_of_documents
        total_tokens = self.total_tokens
        avg_doc_len = (total_tokens / n_docs) if n_docs else 0.0

        with open(output_path, "wb") as out:
            out.write(_HEADER.pack(_MAGIC, n_docs, total_tokens, avg_doc_len))

            # Fixed-width document lengths.
            with open(self.temp_path, "rt", encoding="utf-8") as temp:
                for line in temp:
                    _, length, _ = line.rstrip("\n").split("\t", 2)
                    out.write(_UINT32.pack(int(length)))

            # External ID byte offsets. Offsets are relative to the beginning of
            # the ID byte blob.
            current_offset = 0
            with open(self.temp_path, "rt", encoding="utf-8") as temp:
                for line in temp:
                    _, _, external_id = line.rstrip("\n").split("\t", 2)
                    out.write(_UINT64.pack(current_offset))
                    current_offset += len(external_id.encode("utf-8"))
                out.write(_UINT64.pack(current_offset))

            # External ID byte blob.
            with open(self.temp_path, "rt", encoding="utf-8") as temp:
                for line in temp:
                    _, _, external_id = line.rstrip("\n").split("\t", 2)
                    out.write(external_id.encode("utf-8"))

        if remove_temp:
            os.remove(self.temp_path)

        return DocumentCollectionStats(
            number_of_documents=n_docs,
            total_tokens=total_tokens,
            average_document_length=avg_doc_len,
        )


class DocumentIndexReader:
    """Random-access reader for documents.bin."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.file = open(path, "rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        magic, n_docs, total_tokens, avg_doc_len = _HEADER.unpack_from(self.mm, 0)
        if magic != _MAGIC:
            raise ValueError(f"invalid document index magic in {path}")
        self.number_of_documents = int(n_docs)
        self.total_tokens = int(total_tokens)
        self.average_document_length = float(avg_doc_len)
        self.lengths_offset = _HEADER.size
        self.offsets_offset = (
            self.lengths_offset + self.number_of_documents * _UINT32.size
        )
        self.ids_offset = (
            self.offsets_offset + (self.number_of_documents + 1) * _UINT64.size
        )

    def get_length(self, internal_docid: int) -> int:
        self._check_docid(internal_docid)
        offset = self.lengths_offset + internal_docid * _UINT32.size
        return int(_UINT32.unpack_from(self.mm, offset)[0])

    def get_external_id(self, internal_docid: int) -> str:
        self._check_docid(internal_docid)
        start = _UINT64.unpack_from(
            self.mm, self.offsets_offset + internal_docid * _UINT64.size
        )[0]
        end = _UINT64.unpack_from(
            self.mm, self.offsets_offset + (internal_docid + 1) * _UINT64.size
        )[0]
        raw = self.mm[self.ids_offset + start : self.ids_offset + end]
        return raw.decode("utf-8")

    def _check_docid(self, internal_docid: int) -> None:
        if internal_docid < 0 or internal_docid >= self.number_of_documents:
            raise IndexError(f"invalid internal_docid: {internal_docid}")

    def close(self) -> None:
        self.mm.close()
        self.file.close()

    def __enter__(self) -> "DocumentIndexReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
