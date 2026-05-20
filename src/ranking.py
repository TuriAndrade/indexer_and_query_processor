from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .lexicon import LexiconEntry


class RankerName(str, Enum):
    TFIDF = "TFIDF"
    BM25 = "BM25"


@dataclass(frozen=True)
class CollectionInfo:
    number_of_documents: int
    average_document_length: float


class Scorer:
    """TFIDF and BM25 scoring functions."""

    def __init__(
        self, collection_info: CollectionInfo, k1: float = 1.2, b: float = 0.75
    ) -> None:
        self.collection_info = collection_info
        self.k1 = k1
        self.b = b

    def score(
        self,
        ranker_name: RankerName,
        entries: list[LexiconEntry],
        document_term_frequencies: list[int],
        query_term_frequencies: list[int],
        document_length: int,
    ) -> float:
        if ranker_name == RankerName.TFIDF:
            return self.score_tfidf(
                entries, document_term_frequencies, query_term_frequencies
            )
        if ranker_name == RankerName.BM25:
            return self.score_bm25(
                entries,
                document_term_frequencies,
                query_term_frequencies,
                document_length,
            )
        raise ValueError(f"unsupported ranker: {ranker_name}")

    def score_tfidf(
        self,
        entries: list[LexiconEntry],
        document_term_frequencies: list[int],
        query_term_frequencies: list[int],
    ) -> float:
        n_docs = max(1, self.collection_info.number_of_documents)
        total = 0.0
        for entry, tf, qtf in zip(
            entries, document_term_frequencies, query_term_frequencies
        ):
            if tf <= 0:
                continue
            doc_weight = 1.0 + math.log(tf)
            query_weight = 1.0 + math.log(max(1, qtf))
            idf = math.log((n_docs + 1.0) / (entry.document_frequency + 1.0)) + 1.0
            total += query_weight * doc_weight * idf
        return total

    def score_bm25(
        self,
        entries: list[LexiconEntry],
        document_term_frequencies: list[int],
        query_term_frequencies: list[int],
        document_length: int,
    ) -> float:
        n_docs = max(1, self.collection_info.number_of_documents)
        avgdl = max(1e-9, self.collection_info.average_document_length)
        total = 0.0
        for entry, tf, qtf in zip(
            entries, document_term_frequencies, query_term_frequencies
        ):
            if tf <= 0:
                continue
            df = entry.document_frequency
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            norm = self.k1 * (1.0 - self.b + self.b * (document_length / avgdl))
            tf_component = (tf * (self.k1 + 1.0)) / (tf + norm)
            total += qtf * idf * tf_component
        return total
