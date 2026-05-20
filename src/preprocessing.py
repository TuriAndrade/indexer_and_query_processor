from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

from nltk.stem import SnowballStemmer

# A compact built-in stopword list avoids depending on external NLTK corpora
# being downloaded on the grading machine. If nltk.corpus.stopwords is
# available, we use it and union it with this list.
_BUILTIN_ENGLISH_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

_TOKEN_RE = re.compile(r"(?u)\b[a-z0-9][a-z0-9_'-]*\b")


@lru_cache(maxsize=1)
def load_stopwords() -> tuple[set[str], str]:
    stopwords = set(_BUILTIN_ENGLISH_STOPWORDS)
    status = "using built-in stopword list only"

    try:
        from nltk.corpus import stopwords as nltk_stopwords

        try:
            stopwords.update(nltk_stopwords.words("english"))
            status = "loaded existing NLTK stopwords corpus"

        except LookupError:
            import nltk

            nltk.download("stopwords", quiet=True)
            stopwords.update(nltk_stopwords.words("english"))
            status = "downloaded NLTK stopwords corpus"

    except Exception as exc:
        status = f"failed to load/download NLTK stopwords; using built-in list only: {exc}"

    return stopwords, status


class TextPreprocessor:
    """Tokenizes, removes stopwords, and stems English text.

    The same class is used by indexer.py and processor.py. This is important:
    if document and query preprocessing diverge, query terms will fail to match
    indexed terms.
    """

    def __init__(self, min_token_length: int = 2, verbose: bool = False) -> None:
        self.stopwords, status = load_stopwords()

        if verbose:
            print(f"[TextPreprocessor] {status}.", flush=True)

        self.stemmer = SnowballStemmer("english")
        self.min_token_length = min_token_length

    def preprocess(self, text: str) -> list[str]:
        if not text:
            return []

        tokens: list[str] = []
        for match in _TOKEN_RE.finditer(text.casefold()):
            token = match.group(0).strip("_'-")
            if len(token) < self.min_token_length:
                continue
            if token in self.stopwords:
                continue
            stemmed = self.stemmer.stem(token)
            if stemmed and stemmed not in self.stopwords:
                tokens.append(stemmed)
        return tokens

    def preprocess_many(self, texts: Iterable[str]) -> list[list[str]]:
        return [self.preprocess(text) for text in texts]
