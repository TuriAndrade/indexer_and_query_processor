#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
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
BYTES_PER_MEBIBYTE = 1024 * 1024

_THREAD_LOCAL = threading.local()


@dataclass
class ProcessedDocument:
    internal_id: int
    external_id: str
    document_length: int
    term_counts: Counter[str]


@dataclass
class IndexPaths:
    root_dir: str
    partials_dir: str
    merge_work_dir: str
    documents_temp_path: str
    documents_path: str
    postings_path: str
    lexicon_path: str


@dataclass
class IndexingCounters:
    documents_processed: int = 0
    tokens_processed: int = 0
    bytes_processed: int = 0

    @property
    def average_document_length(self) -> float:
        if self.documents_processed == 0:
            return 0.0
        return self.tokens_processed / self.documents_processed


@dataclass
class LexiconDistributionStats:
    min_list_size: int
    median_list_size: int
    p90_list_size: int
    p95_list_size: int
    p99_list_size: int
    max_list_size: int

    @classmethod
    def empty(cls) -> "LexiconDistributionStats":
        return cls(
            min_list_size=0,
            median_list_size=0,
            p90_list_size=0,
            p95_list_size=0,
            p99_list_size=0,
            max_list_size=0,
        )

    def to_json_fields(self) -> dict[str, int]:
        return {
            "Min List Size": self.min_list_size,
            "Median List Size": self.median_list_size,
            "P90 List Size": self.p90_list_size,
            "P95 List Size": self.p95_list_size,
            "P99 List Size": self.p99_list_size,
            "Max List Size": self.max_list_size,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index builder.")

    parser.add_argument(
        "-m",
        "--memory",
        type=int,
        required=True,
        help="Memory budget in megabytes.",
    )
    parser.add_argument(
        "-c",
        "--corpus",
        required=True,
        help="Path to JSONL corpus file.",
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


def print_json_event(event: dict[str, object]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def get_thread_local_preprocessor(verbose: bool) -> TextPreprocessor:
    preprocessor = getattr(_THREAD_LOCAL, "preprocessor", None)

    if preprocessor is None:
        preprocessor = TextPreprocessor(verbose=verbose)
        _THREAD_LOCAL.preprocessor = preprocessor

    return preprocessor


def process_document(
    raw_document: RawDocument,
    verbose: bool = False,
) -> ProcessedDocument:
    preprocessor = get_thread_local_preprocessor(verbose)
    tokens = preprocessor.preprocess(raw_document.text)

    return ProcessedDocument(
        internal_id=raw_document.internal_id,
        external_id=raw_document.external_id,
        document_length=len(tokens),
        term_counts=Counter(tokens),
    )


def iter_batches(
    items: Iterable[RawDocument],
    batch_size: int,
) -> Iterable[list[RawDocument]]:
    batch: list[RawDocument] = []

    for item in items:
        batch.append(item)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def make_index_paths(index_dir: str) -> IndexPaths:
    return IndexPaths(
        root_dir=index_dir,
        partials_dir=os.path.join(index_dir, PARTIALS_DIRNAME),
        merge_work_dir=os.path.join(index_dir, MERGE_WORK_DIRNAME),
        documents_temp_path=os.path.join(index_dir, DOC_TEMP_FILENAME),
        documents_path=os.path.join(index_dir, DOCUMENTS_FILENAME),
        postings_path=os.path.join(index_dir, POSTINGS_FILENAME),
        lexicon_path=os.path.join(index_dir, LEXICON_FILENAME),
    )


def reset_directory(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def remove_file_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def prepare_index_directory(index_dir: str) -> IndexPaths:
    os.makedirs(index_dir, exist_ok=True)
    paths = make_index_paths(index_dir)

    reset_directory(paths.partials_dir)
    reset_directory(paths.merge_work_dir)

    for path in [
        paths.postings_path,
        paths.lexicon_path,
        paths.documents_path,
        paths.documents_temp_path,
    ]:
        remove_file_if_exists(path)

    return paths


def cleanup_work_directories(paths: IndexPaths) -> None:
    shutil.rmtree(paths.partials_dir, ignore_errors=True)
    shutil.rmtree(paths.merge_work_dir, ignore_errors=True)


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


def maybe_flush_partial_index(
    *,
    term_to_postings: dict[str, dict[int, int]],
    partial_paths: list[str],
    paths: IndexPaths,
    partial_index_number: int,
) -> int:
    partial_path = flush_partial_index(
        term_to_postings,
        paths.partials_dir,
        partial_index_number,
    )

    if partial_path is None:
        return partial_index_number

    partial_paths.append(partial_path)
    return partial_index_number + 1


def add_to_in_memory_index(
    term_to_postings: dict[str, dict[int, int]],
    processed: ProcessedDocument,
) -> None:
    for term, term_frequency in processed.term_counts.items():
        term_to_postings[term][processed.internal_id] = int(term_frequency)


def update_counters_from_batch(
    counters: IndexingCounters,
    batch: list[RawDocument],
) -> None:
    counters.bytes_processed += sum(len(doc.text.encode("utf-8")) for doc in batch)


def update_counters_from_processed_document(
    counters: IndexingCounters,
    processed: ProcessedDocument,
) -> None:
    counters.documents_processed += 1
    counters.tokens_processed += processed.document_length


def should_log_progress(verbose: bool, counters: IndexingCounters) -> bool:
    return (
        verbose
        and counters.documents_processed > 0
        and counters.documents_processed % DOC_LOG_INTERVAL == 0
    )


def log_indexing_progress(
    *,
    counters: IndexingCounters,
    corpus_size_bytes: int,
    memory_monitor: MemoryMonitor,
    partial_index_number: int,
    start_time: float,
) -> None:
    elapsed = time.time() - start_time
    progress_pct = 100.0 * counters.bytes_processed / max(corpus_size_bytes, 1)
    docs_per_second = counters.documents_processed / max(elapsed, 1e-9)

    print_json_event(
        {
            "phase": "indexing",
            "documents_processed": counters.documents_processed,
            "progress_percent": round(progress_pct, 2),
            "elapsed_seconds": round(elapsed, 2),
            "docs_per_second": round(docs_per_second, 2),
            "rss_mb": round(memory_monitor.rss_megabytes(), 2),
            "partial_indexes": partial_index_number,
        }
    )


def log_flush(
    *,
    partial_index_number: int,
    memory_monitor: MemoryMonitor,
    term_to_postings: dict[str, dict[int, int]],
) -> None:
    print_json_event(
        {
            "phase": "flush",
            "partial_index": partial_index_number,
            "rss_mb": round(memory_monitor.rss_megabytes(), 2),
            "terms_in_memory": len(term_to_postings),
        }
    )


def log_merge_start(partial_paths: list[str]) -> None:
    print_json_event(
        {
            "phase": "merge_start",
            "partial_files": len(partial_paths),
        }
    )


def create_empty_final_index_files(paths: IndexPaths):
    open(paths.postings_path, "wb").close()
    open(paths.lexicon_path, "wt", encoding="utf-8").close()

    from src.merge import MergeStats

    return MergeStats(
        number_of_terms=0,
        total_postings=0,
    )


def merge_partial_indexes(
    *,
    partial_paths: list[str],
    paths: IndexPaths,
    max_open_files: int,
    verbose: bool,
):
    if not partial_paths:
        return create_empty_final_index_files(paths)

    if verbose:
        log_merge_start(partial_paths)

    merger = IndexMerger(max_open_files=max_open_files)

    return merger.merge_to_final(
        partial_paths,
        paths.postings_path,
        paths.lexicon_path,
        paths.merge_work_dir,
    )


def final_index_size_bytes(paths: IndexPaths) -> int:
    return sum(
        os.path.getsize(path)
        for path in [
            paths.postings_path,
            paths.lexicon_path,
            paths.documents_path,
        ]
    )


def parse_document_frequency_from_lexicon_line(line: str) -> int | None:
    parts = line.rstrip("\n").split("\t")

    if len(parts) < 2:
        return None

    try:
        return int(parts[1])
    except ValueError:
        return None


def percentile_from_sorted_values(values: list[int], percentile: float) -> int:
    if not values:
        return 0

    index = int(percentile * (len(values) - 1))
    return values[index]


def compute_lexicon_distribution_stats(lexicon_path: str) -> LexiconDistributionStats:
    list_sizes: list[int] = []

    with open(lexicon_path, "rt", encoding="utf-8") as lexicon_file:
        for line in lexicon_file:
            if not line.strip():
                continue

            document_frequency = parse_document_frequency_from_lexicon_line(line)
            if document_frequency is not None:
                list_sizes.append(document_frequency)

    if not list_sizes:
        return LexiconDistributionStats.empty()

    list_sizes.sort()

    return LexiconDistributionStats(
        min_list_size=list_sizes[0],
        median_list_size=percentile_from_sorted_values(list_sizes, 0.50),
        p90_list_size=percentile_from_sorted_values(list_sizes, 0.90),
        p95_list_size=percentile_from_sorted_values(list_sizes, 0.95),
        p99_list_size=percentile_from_sorted_values(list_sizes, 0.99),
        max_list_size=list_sizes[-1],
    )


def build_final_stats(
    *,
    elapsed_time: float,
    index_size_bytes: int,
    counters: IndexingCounters,
    number_of_terms: int,
    total_postings: int,
    lexicon_distribution_stats: LexiconDistributionStats,
) -> dict[str, float | int]:
    average_list_size = total_postings / number_of_terms if number_of_terms else 0.0
    documents_per_second = counters.documents_processed / max(elapsed_time, 1e-9)

    stats: dict[str, float | int] = {
        "Index Size": index_size_bytes / BYTES_PER_MEBIBYTE,
        "Elapsed Time": elapsed_time,
        "Number of Documents": counters.documents_processed,
        "Number of Tokens": counters.tokens_processed,
        "Average Document Length": counters.average_document_length,
        "Number of Lists": number_of_terms,
        "Total Postings": total_postings,
        "Average List Size": average_list_size,
        "Documents per Second": documents_per_second,
    }

    stats.update(lexicon_distribution_stats.to_json_fields())
    return stats


def index_batch(
    *,
    batch: list[RawDocument],
    executor: ThreadPoolExecutor,
    document_builder: DocumentIndexBuilder,
    term_to_postings: dict[str, dict[int, int]],
    counters: IndexingCounters,
    corpus_size_bytes: int,
    memory_monitor: MemoryMonitor,
    partial_index_number: int,
    start_time: float,
    verbose: bool,
) -> None:
    update_counters_from_batch(counters, batch)
    worker_fn = partial(process_document, verbose=verbose)

    for processed in executor.map(worker_fn, batch):
        document_builder.add_document(
            processed.internal_id,
            processed.external_id,
            processed.document_length,
        )

        add_to_in_memory_index(term_to_postings, processed)
        update_counters_from_processed_document(counters, processed)

        if should_log_progress(verbose, counters):
            log_indexing_progress(
                counters=counters,
                corpus_size_bytes=corpus_size_bytes,
                memory_monitor=memory_monitor,
                partial_index_number=partial_index_number,
                start_time=start_time,
            )


def build_index(args: argparse.Namespace) -> dict[str, float | int]:
    start_time = time.time()
    corpus_size_bytes = os.path.getsize(args.corpus)

    paths = prepare_index_directory(args.index)
    counters = IndexingCounters()

    memory_monitor = MemoryMonitor(
        args.memory,
        flush_ratio=args.flush_ratio,
    )

    document_builder = DocumentIndexBuilder(paths.documents_temp_path)
    term_to_postings: dict[str, dict[int, int]] = defaultdict(dict)
    partial_paths: list[str] = []
    partial_index_number = 0

    corpus_reader = CorpusReader(args.corpus)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        for batch in iter_batches(corpus_reader, args.batch_size):
            index_batch(
                batch=batch,
                executor=executor,
                document_builder=document_builder,
                term_to_postings=term_to_postings,
                counters=counters,
                corpus_size_bytes=corpus_size_bytes,
                memory_monitor=memory_monitor,
                partial_index_number=partial_index_number,
                start_time=start_time,
                verbose=args.verbose,
            )

            if memory_monitor.should_flush():
                if args.verbose:
                    log_flush(
                        partial_index_number=partial_index_number,
                        memory_monitor=memory_monitor,
                        term_to_postings=term_to_postings,
                    )

                partial_index_number = maybe_flush_partial_index(
                    term_to_postings=term_to_postings,
                    partial_paths=partial_paths,
                    paths=paths,
                    partial_index_number=partial_index_number,
                )

    maybe_flush_partial_index(
        term_to_postings=term_to_postings,
        partial_paths=partial_paths,
        paths=paths,
        partial_index_number=partial_index_number,
    )

    document_builder.finalize(paths.documents_path)

    merge_stats = merge_partial_indexes(
        partial_paths=partial_paths,
        paths=paths,
        max_open_files=args.max_open_files,
        verbose=args.verbose,
    )

    cleanup_work_directories(paths)

    elapsed_time = time.time() - start_time
    index_size_bytes = final_index_size_bytes(paths)
    lexicon_distribution_stats = compute_lexicon_distribution_stats(paths.lexicon_path)

    return build_final_stats(
        elapsed_time=elapsed_time,
        index_size_bytes=index_size_bytes,
        counters=counters,
        number_of_terms=merge_stats.number_of_terms,
        total_postings=merge_stats.total_postings,
        lexicon_distribution_stats=lexicon_distribution_stats,
    )


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
