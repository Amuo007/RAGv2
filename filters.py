import re

from classifier import _STOPWORDS


def generic_relevance_filter(query: str, chunks: list, threshold: float) -> list:
    """Return chunks that pass the relevance threshold or have sufficient title overlap."""
    query_words = set(re.findall(r'\w+', query.lower())) - _STOPWORDS
    def _title_match(title):
        tw = set(re.findall(r'\w+', title.lower())) - _STOPWORDS
        if not query_words or not tw:
            return False
        return len(query_words & tw) / len(tw) >= 0.5
    return [c for c in chunks if c[2] >= threshold or _title_match(c[0])]
