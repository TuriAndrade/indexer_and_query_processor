#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

from src.corpus import CorpusReader, RawDocument
from src.document_index import DocumentIndexBuilder
from src.memory import MemoryMonitor
from src.merge import IndexMerger
from src.partial_index import write_partial_index
from src.preprocessing import TextPreprocessor

POSTINGS_FILENAME = "postings.bin"
LEXICON_FILENAME = "lexicon.tsv"
DOCUMENTS_FILENAME = "documents.bin"
PARTIALS_DIRNAME = "partials"
MERGE_WORK_DIRNAME = "merge_work"
DOC_TEMP_FILENAME = "documents.tmp.tsv"
DOC_LOG_INTERVAL = 100000


@dataclass
class ProcessedDocument:
    internal_id: int
    external_id: str
    document_length: int
    term_counts: Counter[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index builder."
    )
    parser.add_argument(
        "-m", "--memory", type=int, required=True, help="Memory budget in megabytes."
    )
    parser.add_argument(
        "-c", "--corpus", required=True, help="Path to JSONL corpus file."
    )
    parser.add_argument(
        "-i",
        "--index",
        required=True,
        help="Directory where the final index files are written.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=f"Enable progress logging every {DOC_LOG_INTERVAL:,} documents.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of preprocessing worker threads. Default: 1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Number of documents submitted to workers at once.",
    )
    parser.add_argument(
        "--flush-ratio",
        type=float,
        default=0.70,
        help="Flush partial index when RSS reaches this fraction of the memory budget.",
    )
    parser.add_argument(
        "--max-open-files",
        type=int,
        default=64,
        help="Maximum number of partial files opened simultaneously during merge.",
    )
    return parser.parse_args()


def process_document(
    raw_document: RawDocument,
    verbose: bool = False,
) -> ProcessedDocument:
    # Each worker creates its own preprocessor lazily. This avoids sharing the
    # SnowballStemmer instance across threads.
    if not hasattr(process_document, "preprocessor"):
        process_document.preprocessor = TextPreprocessor(
            verbose=verbose
        )  # type: ignore[attr-defined]

    preprocessor: TextPreprocessor = process_document.preprocessor  # type: ignore[attr-defined]

    tokens = preprocessor.preprocess(raw_document.text)

    return ProcessedDocument(
        internal_id=raw_document.internal_id,
        external_id=raw_document.external_id,
        document_length=len(tokens),
        term_counts=Counter(tokens),
    )


def iter_batches(
    items: Iterable[RawDocument], batch_size: int
) -> Iterable[list[RawDocument]]:
    batch: list[RawDocument] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare_index_directory(index_dir: str) -> tuple[str, str]:
    os.makedirs(index_dir, exist_ok=True)
    partials_dir = os.path.join(index_dir, PARTIALS_DIRNAME)
    merge_work_dir = os.path.join(index_dir, MERGE_WORK_DIRNAME)

    for path in [partials_dir, merge_work_dir]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    # Remove stale final files from previous runs.
    for filename in [
        POSTINGS_FILENAME,
        LEXICON_FILENAME,
        DOCUMENTS_FILENAME,
        DOC_TEMP_FILENAME,
    ]:
        path = os.path.join(index_dir, filename)
        if os.path.exists(path):
            os.remove(path)

    return partials_dir, merge_work_dir


def flush_partial_index(
    term_to_postings: dict[str, dict[int, int]],
    partials_dir: str,
    partial_index_number: int,
) -> str | None:
    if not term_to_postings:
        return None

    output_path = os.path.join(
        partials_dir,
        f"partial_{partial_index_number:06d}.part",
    )

    write_partial_index(term_to_postings, output_path)

    term_to_postings.clear()
    gc.collect()

    return output_path


def directory_size_bytes(path: str) -> int:
    total = 0

    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            total += os.path.getsize(file_path)

    return total


def build_index(args: argparse.Namespace) -> dict[str, float | int]:
    start_time = time.time()

    corpus_size_bytes = os.path.getsize(args.corpus)
    bytes_processed = 0

    partials_dir, merge_work_dir = prepare_index_directory(args.index)

    memory_monitor = MemoryMonitor(
        args.memory,
        flush_ratio=args.flush_ratio,
    )

    document_builder = DocumentIndexBuilder(os.path.join(args.index, DOC_TEMP_FILENAME))

    term_to_postings: dict[str, dict[int, int]] = defaultdict(dict)

    partial_paths: list[str] = []
    partial_index_number = 0

    corpus_reader = CorpusReader(args.corpus)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        for batch in iter_batches(corpus_reader, args.batch_size):
            bytes_processed += sum(len(doc.text.encode("utf-8")) for doc in batch)

            worker_fn = partial(process_document, verbose=args.verbose)

            for processed in executor.map(worker_fn, batch):
                document_builder.add_document(
                    processed.internal_id,
                    processed.external_id,
                    processed.document_length,
                )

                for term, term_frequency in processed.term_counts.items():
                    term_to_postings[term][processed.internal_id] = int(term_frequency)

                if (
                    args.verbose
                    and processed.internal_id > 0
                    and processed.internal_id % DOC_LOG_INTERVAL == 0
                ):
                    elapsed = time.time() - start_time

                    rss_mb = memory_monitor.rss_megabytes()

                    docs_per_second = processed.internal_id / max(elapsed, 1e-9)

                    progress_pct = 100.0 * bytes_processed / max(corpus_size_bytes, 1)

                    print(
                        json.dumps(
                            {
                                "phase": "indexing",
                                "documents_processed": processed.internal_id,
                                "progress_percent": round(progress_pct, 2),
                                "elapsed_seconds": round(elapsed, 2),
                                "docs_per_second": round(docs_per_second, 2),
                                "rss_mb": round(rss_mb, 2),
                                "partial_indexes": partial_index_number,
                            }
                        ),
                        flush=True,
                    )

            if memory_monitor.should_flush():
                if args.verbose:
                    print(
                        json.dumps(
                            {
                                "phase": "flush",
                                "partial_index": partial_index_number,
                                "rss_mb": round(
                                    memory_monitor.rss_megabytes(),
                                    2,
                                ),
                                "terms_in_memory": len(term_to_postings),
                            }
                        ),
                        flush=True,
                    )

                partial_path = flush_partial_index(
                    term_to_postings,
                    partials_dir,
                    partial_index_number,
                )

                if partial_path is not None:
                    partial_paths.append(partial_path)
                    partial_index_number += 1

    partial_path = flush_partial_index(
        term_to_postings,
        partials_dir,
        partial_index_number,
    )

    if partial_path is not None:
        partial_paths.append(partial_path)

    documents_path = os.path.join(
        args.index,
        DOCUMENTS_FILENAME,
    )

    document_stats = document_builder.finalize(documents_path)

    postings_path = os.path.join(
        args.index,
        POSTINGS_FILENAME,
    )

    lexicon_path = os.path.join(
        args.index,
        LEXICON_FILENAME,
    )

    if partial_paths:
        merger = IndexMerger(max_open_files=args.max_open_files)

        if args.verbose:
            print(
                json.dumps(
                    {
                        "phase": "merge_start",
                        "partial_files": len(partial_paths),
                    }
                ),
                flush=True,
            )

        merge_stats = merger.merge_to_final(
            partial_paths,
            postings_path,
            lexicon_path,
            merge_work_dir,
        )

    else:
        open(postings_path, "wb").close()
        open(lexicon_path, "wt", encoding="utf-8").close()

        from src.merge import MergeStats

        merge_stats = MergeStats(
            number_of_terms=0,
            total_postings=0,
        )

    # Remove temporary partial/merge files after final index is created.
    shutil.rmtree(partials_dir, ignore_errors=True)
    shutil.rmtree(merge_work_dir, ignore_errors=True)

    elapsed_time = time.time() - start_time

    final_index_size = sum(
        os.path.getsize(os.path.join(args.index, filename))
        for filename in [
            POSTINGS_FILENAME,
            LEXICON_FILENAME,
            DOCUMENTS_FILENAME,
        ]
    )

    average_list_size = (
        merge_stats.total_postings / merge_stats.number_of_terms
        if merge_stats.number_of_terms
        else 0.0
    )

    # Keep stdout aligned with the assignment's required four fields.
    return {
        "Index Size": final_index_size / (1024 * 1024),
        "Elapsed Time": elapsed_time,
        "Number of Lists": merge_stats.number_of_terms,
        "Average List Size": average_list_size,
    }


def main() -> None:
    args = parse_args()

    try:
        stats = build_index(args)

    except Exception as exc:
        print(f"indexer.py failed: {exc}", file=sys.stderr)
        raise

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
