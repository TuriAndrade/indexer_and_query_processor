from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass
class PostingList:
    docids: array
    term_frequencies: array

    def __len__(self) -> int:
        return len(self.docids)


def encode_varint(number: int) -> bytes:
    """Encodes an integer >= 0 using 7-bit variable-byte encoding."""
    if number < 0:
        raise ValueError("varint cannot encode negative numbers")
    out = bytearray()
    while True:
        byte = number & 0x7F
        number >>= 7
        if number:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def iter_decode_varints(data: bytes) -> Iterator[int]:
    shift = 0
    value = 0
    for byte in data:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            yield value
            shift = 0
            value = 0
    if shift != 0:
        raise ValueError("truncated varint stream")


def encode_postings(postings: Iterable[tuple[int, int]]) -> bytes:
    """Encodes sorted (docid, tf) pairs as docid gaps and term frequencies."""
    previous_docid = 0
    out = bytearray()
    for docid, term_frequency in postings:
        if docid < previous_docid:
            raise ValueError("postings must be sorted by nondecreasing docid")
        if term_frequency <= 0:
            continue
        gap = docid - previous_docid
        out.extend(encode_varint(gap))
        out.extend(encode_varint(term_frequency))
        previous_docid = docid
    return bytes(out)


def decode_postings(data: bytes) -> PostingList:
    values = iter_decode_varints(data)
    docids = array("I")
    term_frequencies = array("I")
    previous_docid = 0
    while True:
        try:
            gap = next(values)
            term_frequency = next(values)
        except StopIteration:
            break
        docid = previous_docid + gap
        docids.append(docid)
        term_frequencies.append(term_frequency)
        previous_docid = docid
    return PostingList(docids=docids, term_frequencies=term_frequencies)


def merge_encoded_posting_chunks(chunks: list[bytes]) -> PostingList:
    """Merges sorted encoded posting chunks into a single sorted posting list."""
    import heapq

    decoded = [decode_postings(chunk) for chunk in chunks if chunk]
    heap: list[tuple[int, int, int]] = []
    for list_index, posting_list in enumerate(decoded):
        if posting_list.docids:
            heapq.heappush(heap, (posting_list.docids[0], list_index, 0))

    merged_docids = array("I")
    merged_tfs = array("I")

    last_docid: int | None = None
    accumulated_tf = 0

    while heap:
        docid, list_index, position = heapq.heappop(heap)
        tf = decoded[list_index].term_frequencies[position]

        if last_docid is None:
            last_docid = docid
            accumulated_tf = tf
        elif docid == last_docid:
            accumulated_tf += tf
        else:
            merged_docids.append(last_docid)
            merged_tfs.append(accumulated_tf)
            last_docid = docid
            accumulated_tf = tf

        next_position = position + 1
        if next_position < len(decoded[list_index].docids):
            next_docid = decoded[list_index].docids[next_position]
            heapq.heappush(heap, (next_docid, list_index, next_position))

    if last_docid is not None:
        merged_docids.append(last_docid)
        merged_tfs.append(accumulated_tf)

    return PostingList(docids=merged_docids, term_frequencies=merged_tfs)


def posting_list_to_pairs(posting_list: PostingList) -> Iterator[tuple[int, int]]:
    for docid, term_frequency in zip(
        posting_list.docids, posting_list.term_frequencies
    ):
        yield int(docid), int(term_frequency)
