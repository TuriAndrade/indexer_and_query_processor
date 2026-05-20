from __future__ import annotations

import heapq
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .partial_index import PartialIndexReader, PartialIndexWriter
from .postings import (
    encode_postings,
    merge_encoded_posting_chunks,
    posting_list_to_pairs,
)


@dataclass(frozen=True)
class MergeStats:
    number_of_terms: int
    total_postings: int


class IndexMerger:
    """External merge for partial indexes.
    """

    def __init__(self, max_open_files: int = 64) -> None:
        self.max_open_files = max(2, max_open_files)

    def merge_to_final(
        self,
        partial_paths: list[str],
        postings_path: str,
        lexicon_path: str,
        work_dir: str,
    ) -> MergeStats:
        os.makedirs(os.path.dirname(postings_path), exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)

        current_paths = list(partial_paths)
        round_number = 0
        while len(current_paths) > self.max_open_files:
            next_paths: list[str] = []
            for group_id, start in enumerate(
                range(0, len(current_paths), self.max_open_files)
            ):
                group = current_paths[start : start + self.max_open_files]
                output_path = os.path.join(
                    work_dir, f"merge_round_{round_number}_{group_id:06d}.part"
                )
                self._merge_group_to_partial(group, output_path)
                next_paths.append(output_path)
            for path in current_paths:
                if Path(path).parent == Path(work_dir):
                    os.remove(path)
            current_paths = next_paths
            round_number += 1

        stats = self._merge_group_to_final(current_paths, postings_path, lexicon_path)
        return stats

    def _push_next(
        self,
        heap: list[tuple[str, int, bytes]],
        readers: list[PartialIndexReader],
        reader_index: int,
    ) -> None:
        record = readers[reader_index].read_next()
        if record is None:
            return
        term, encoded_postings = record
        heapq.heappush(heap, (term, reader_index, encoded_postings))

    def _open_readers(self, paths: list[str]) -> list[PartialIndexReader]:
        return [PartialIndexReader(path) for path in paths]

    def _close_readers(self, readers: list[PartialIndexReader]) -> None:
        for reader in readers:
            reader.close()

    def _merge_group_records(self, paths: list[str]):
        readers = self._open_readers(paths)
        heap: list[tuple[str, int, bytes]] = []
        try:
            for reader_index in range(len(readers)):
                self._push_next(heap, readers, reader_index)

            while heap:
                term = heap[0][0]
                chunks: list[bytes] = []
                while heap and heap[0][0] == term:
                    _, reader_index, encoded_postings = heapq.heappop(heap)
                    chunks.append(encoded_postings)
                    self._push_next(heap, readers, reader_index)
                merged = merge_encoded_posting_chunks(chunks)
                yield term, merged
        finally:
            self._close_readers(readers)

    def _merge_group_to_partial(self, paths: list[str], output_path: str) -> None:
        with PartialIndexWriter(output_path) as writer:
            for term, merged_posting_list in self._merge_group_records(paths):
                encoded = encode_postings(posting_list_to_pairs(merged_posting_list))
                writer.write_record(term, encoded)

    def _merge_group_to_final(
        self,
        paths: list[str],
        postings_path: str,
        lexicon_path: str,
    ) -> MergeStats:
        number_of_terms = 0
        total_postings = 0
        with open(postings_path, "wb") as postings_file, open(
            lexicon_path, "wt", encoding="utf-8"
        ) as lexicon_file:
            for term, merged_posting_list in self._merge_group_records(paths):
                offset = postings_file.tell()
                encoded = encode_postings(posting_list_to_pairs(merged_posting_list))
                postings_file.write(encoded)
                nbytes = len(encoded)
                document_frequency = len(merged_posting_list)
                collection_frequency = int(sum(merged_posting_list.term_frequencies))
                lexicon_file.write(
                    f"{term}\t{document_frequency}\t{collection_frequency}\t{offset}\t{nbytes}\n"
                )
                number_of_terms += 1
                total_postings += document_frequency
        return MergeStats(
            number_of_terms=number_of_terms, total_postings=total_postings
        )

    @staticmethod
    def remove_directory(path: str) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path)
