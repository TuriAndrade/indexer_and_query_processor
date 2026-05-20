from __future__ import annotations

from bisect import bisect_left
from typing import Iterator

from .postings import PostingList


def conjunctive_daat(
    posting_lists: list[PostingList],
) -> Iterator[tuple[int, list[int]]]:
    """Conjunctive document-at-a-time mathcing.
    """
    if not posting_lists:
        return

    positions = [0] * len(posting_lists)

    while True:
        for i, posting_list in enumerate(posting_lists):
            if positions[i] >= len(posting_list.docids):
                return

        current_docids = [
            posting_lists[i].docids[positions[i]] for i in range(len(posting_lists))
        ]
        target_docid = max(current_docids)

        if all(docid == target_docid for docid in current_docids):
            term_frequencies = [
                int(posting_lists[i].term_frequencies[positions[i]])
                for i in range(len(posting_lists))
            ]
            yield int(target_docid), term_frequencies
            for i in range(len(posting_lists)):
                positions[i] += 1
        else:
            for i, docid in enumerate(current_docids):
                if docid < target_docid:
                    positions[i] = bisect_left(
                        posting_lists[i].docids, target_docid, positions[i]
                    )
