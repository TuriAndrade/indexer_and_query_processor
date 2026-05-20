from __future__ import annotations

import os

from .document_index import DocumentIndexReader
from .lexicon import LexiconEntry, load_lexicon
from .postings import PostingList, decode_postings


class DiskIndexReader:
    """Reads the final index directory produced by indexer.py."""

    POSTINGS_FILENAME = "postings.bin"
    LEXICON_FILENAME = "lexicon.tsv"
    DOCUMENTS_FILENAME = "documents.bin"

    def __init__(self, index_dir: str) -> None:
        self.index_dir = index_dir
        self.lexicon_path = os.path.join(index_dir, self.LEXICON_FILENAME)
        self.postings_path = os.path.join(index_dir, self.POSTINGS_FILENAME)
        self.documents_path = os.path.join(index_dir, self.DOCUMENTS_FILENAME)
        self.lexicon: dict[str, LexiconEntry] = load_lexicon(self.lexicon_path)
        self.postings_file = open(self.postings_path, "rb")
        self.documents = DocumentIndexReader(self.documents_path)

    def get_entry(self, term: str) -> LexiconEntry | None:
        return self.lexicon.get(term)

    def read_postings(self, term: str) -> PostingList | None:
        entry = self.lexicon.get(term)
        if entry is None:
            return None
        self.postings_file.seek(entry.offset)
        data = self.postings_file.read(entry.nbytes)
        return decode_postings(data)

    def close(self) -> None:
        self.postings_file.close()
        self.documents.close()

    def __enter__(self) -> "DiskIndexReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
