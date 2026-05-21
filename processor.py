#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import statistics
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
        help="Print query characterization statistics to stderr.",
    )

    return parser.parse_args()


def read_queries(path: str) -> list[str]:
    with open(path, "rt", encoding="utf-8", errors="replace") as file:
        return [line.rstrip("\n") for line in file if line.strip()]


def percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None

    index = int(p * (len(sorted_values) - 1))
    return sorted_values[index]


def score_distribution(scores: list[float]) -> dict[str, float | None]:
    if not scores:
        return {
            "min_score": None,
            "mean_score": None,
            "median_score": None,
            "p90_score": None,
            "p95_score": None,
            "max_score": None,
        }

    sorted_scores = sorted(scores)

    return {
        "min_score": min(scores),
        "mean_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "p90_score": percentile(sorted_scores, 0.90),
        "p95_score": percentile(sorted_scores, 0.95),
        "max_score": max(scores),
    }


def empty_query_stats(
    query: str,
    tokens: list[str],
    ranker_name: RankerName,
) -> dict[str, object]:
    return {
        "query": query,
        "ranker": ranker_name.value,
        "query_tokens": tokens,
        "matched_documents": 0,
        "scored_documents": 0,
        "returned_results": 0,
        **score_distribution([]),
    }


def process_query(
    query: str,
    index: DiskIndexReader,
    preprocessor: TextPreprocessor,
    scorer: Scorer,
    ranker_name: RankerName,
    top_k: int = 10,
) -> tuple[dict[str, object], dict[str, object]]:
    tokens = preprocessor.preprocess(query)
    query_counts = Counter(tokens)

    if not query_counts:
        return {"Query": query, "Results": []}, empty_query_stats(
            query,
            tokens,
            ranker_name,
        )

    query_terms: list[QueryTermData] = []

    for term, query_tf in query_counts.items():
        entry = index.get_entry(term)

        if entry is None:
            return {"Query": query, "Results": []}, empty_query_stats(
                query,
                tokens,
                ranker_name,
            )

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
            return {"Query": query, "Results": []}, empty_query_stats(
                query,
                tokens,
                ranker_name,
            )

        posting_lists.append(postings)

    entries = [item.entry for item in query_terms]
    query_tfs = [item.query_tf for item in query_terms]

    heap: list[tuple[float, int]] = []
    scores: list[float] = []
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

        score = float(score)
        scores.append(score)

        if score <= 0.0:
            continue

        item = (
            score,
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

    query_stats = {
        "query": query,
        "ranker": ranker_name.value,
        "query_tokens": tokens,
        "matched_documents": matched_documents,
        "scored_documents": len(scores),
        "returned_results": len(results),
        **score_distribution(scores),
    }

    return {
        "Query": query,
        "Results": results,
    }, query_stats


def build_summary(
    query_stats: list[dict[str, object]],
    ranker_name: RankerName,
    elapsed_seconds: float,
) -> dict[str, object]:
    total_matched_documents = sum(
        int(stats["matched_documents"])
        for stats in query_stats
    )
    total_scored_documents = sum(
        int(stats["scored_documents"])
        for stats in query_stats
    )
    total_returned_results = sum(
        int(stats["returned_results"])
        for stats in query_stats
    )

    return {
        "phase": "summary",
        "ranker": ranker_name.value,
        "queries_processed": len(query_stats),
        "total_matched_documents": total_matched_documents,
        "total_scored_documents": total_scored_documents,
        "total_returned_results": total_returned_results,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "queries": query_stats,
    }


def main() -> None:
    args = parse_args()

    ranker_name = RankerName(args.ranker)
    queries = read_queries(args.queries)
    preprocessor = TextPreprocessor()

    startup_start = time.time()
    total_start = time.time()

    try:
        with DiskIndexReader(args.index) as index:
            if args.verbose:
                print(
                    json.dumps(
                        {
                            "phase": "startup",
                            "terms_in_lexicon": len(index.lexicon),
                            "documents": index.documents.number_of_documents,
                            "elapsed_seconds": round(
                                time.time() - startup_start,
                                4,
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

            scorer = Scorer(
                CollectionInfo(
                    number_of_documents=index.documents.number_of_documents,
                    average_document_length=index.documents.average_document_length,
                )
            )

            all_query_stats: list[dict[str, object]] = []

            for query in queries:
                query_start = time.time()

                output, query_stats = process_query(
                    query,
                    index,
                    preprocessor,
                    scorer,
                    ranker_name,
                )

                query_stats["elapsed_seconds"] = round(
                    time.time() - query_start,
                    4,
                )
                all_query_stats.append(query_stats)

                print(
                    json.dumps(
                        output,
                        ensure_ascii=False,
                    )
                )

            if args.verbose:
                summary = build_summary(
                    all_query_stats,
                    ranker_name,
                    elapsed_seconds=time.time() - total_start,
                )

                print(
                    json.dumps(summary, ensure_ascii=False),
                    file=sys.stderr,
                    flush=True,
                )

    except Exception as exc:
        print(
            f"processor.py failed: {exc}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
