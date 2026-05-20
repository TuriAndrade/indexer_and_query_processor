#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass

from src.daat import conjunctive_daat
from src.index_reader import DiskIndexReader
from src.lexicon import LexiconEntry
from src.preprocessing import TextPreprocessor
from src.ranking import CollectionInfo, RankerName, Scorer


@dataclass(frozen=True)
class QueryTermData:
    term: str
    entry: LexiconEntry
    query_tf: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query processor."
    )

    parser.add_argument(
        "-i",
        "--index",
        required=True,
        help="Directory containing the index files.",
    )

    parser.add_argument(
        "-q",
        "--queries",
        required=True,
        help="Path to file containing one query per line.",
    )

    parser.add_argument(
        "-r",
        "--ranker",
        required=True,
        choices=[RankerName.TFIDF.value, RankerName.BM25.value],
        help="Ranking function to use: TFIDF or BM25.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable query processing logs.",
    )

    return parser.parse_args()


def read_queries(path: str) -> list[str]:
    with open(path, "rt", encoding="utf-8", errors="replace") as file:
        return [line.rstrip("\n") for line in file if line.strip()]


def process_query(
    query: str,
    index: DiskIndexReader,
    preprocessor: TextPreprocessor,
    scorer: Scorer,
    ranker_name: RankerName,
    top_k: int = 10,
) -> tuple[dict[str, object], int]:
    tokens = preprocessor.preprocess(query)

    query_counts = Counter(tokens)

    if not query_counts:
        return {"Query": query, "Results": []}, 0

    query_terms: list[QueryTermData] = []

    for term, query_tf in query_counts.items():
        entry = index.get_entry(term)

        if entry is None:
            return {"Query": query, "Results": []}, 0

        query_terms.append(
            QueryTermData(
                term=term,
                entry=entry,
                query_tf=int(query_tf),
            )
        )

    # Conjunctive DAAT is fastest when the shortest posting lists are first.
    query_terms.sort(key=lambda item: item.entry.document_frequency)

    posting_lists = []

    for item in query_terms:
        postings = index.read_postings(item.term)

        if postings is None or len(postings) == 0:
            return {"Query": query, "Results": []}, 0

        posting_lists.append(postings)

    entries = [item.entry for item in query_terms]
    query_tfs = [item.query_tf for item in query_terms]

    heap: list[tuple[float, int]] = []

    matched_documents = 0

    for internal_docid, document_tfs in conjunctive_daat(posting_lists):
        matched_documents += 1

        document_length = index.documents.get_length(internal_docid)

        score = scorer.score(
            ranker_name=ranker_name,
            entries=entries,
            document_term_frequencies=document_tfs,
            query_term_frequencies=query_tfs,
            document_length=document_length,
        )

        if score <= 0.0:
            continue

        item = (
            float(score),
            int(internal_docid),
        )

        if len(heap) < top_k:
            heapq.heappush(heap, item)

        else:
            # Keep deterministic tie-breaking.
            if item[0] > heap[0][0] or (item[0] == heap[0][0] and item[1] < heap[0][1]):
                heapq.heapreplace(heap, item)

    ranked = sorted(
        heap,
        key=lambda pair: (-pair[0], pair[1]),
    )

    results = [
        {
            "ID": index.documents.get_external_id(internal_docid),
            "Score": round(score, 6),
        }
        for score, internal_docid in ranked
    ]

    return {
        "Query": query,
        "Results": results,
    }, matched_documents


def main() -> None:
    args = parse_args()

    ranker_name = RankerName(args.ranker)

    queries = read_queries(args.queries)

    preprocessor = TextPreprocessor()

    startup_start = time.time()

    try:
        with DiskIndexReader(args.index) as index:
            if args.verbose:
                print(
                    json.dumps(
                        {
                            "phase": "startup",
                            "terms_in_lexicon": len(index.lexicon),
                            "documents": (index.documents.number_of_documents),
                            "elapsed_seconds": round(
                                time.time() - startup_start,
                                4,
                            ),
                        }
                    ),
                    flush=True,
                )

            scorer = Scorer(
                CollectionInfo(
                    number_of_documents=(index.documents.number_of_documents),
                    average_document_length=(index.documents.average_document_length),
                )
            )

            for query in queries:
                query_start = time.time()

                output, matched_documents = process_query(
                    query,
                    index,
                    preprocessor,
                    scorer,
                    ranker_name,
                )

                if args.verbose:
                    print(
                        json.dumps(
                            {
                                "phase": "query",
                                "query": query,
                                "matched_documents": (matched_documents),
                                "elapsed_seconds": round(
                                    time.time() - query_start,
                                    4,
                                ),
                                "ranker": (ranker_name.value),
                            }
                        ),
                        flush=True,
                    )

                print(
                    json.dumps(
                        output,
                        ensure_ascii=False,
                    )
                )

    except Exception as exc:
        print(
            f"processor.py failed: {exc}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
