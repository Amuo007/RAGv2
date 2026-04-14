import re

_STOPWORDS = {
    'the','a','an','is','are','was','were','did','do','does','when','where',
    'who','what','how','why','which','in','on','at','to','for','of',
    'and','or','but','it','he','she','they','we','you','his','her',
    'their','its','be','been','have','has','had','this','that','these',
    'those','will','would','could','should','about','from','with',
    # instruction/query words — keep out of BM25 match expressions
    'tell','me','give','show','find','explain','describe','summarize',
    'provide','list','help','please','let','know','want','need','like',
    'get','make','go','say','just','very','also','can','no','not','my',
    'our','into','after','over','some','other','if','then','as','so',
}

# ── Query normalisation maps ──────────────────────────────────────────────────
_ARABIC_TO_ROMAN = {
    "1":"I","2":"II","3":"III","4":"IV","5":"V","6":"VI","7":"VII",
    "8":"VIII","9":"IX","10":"X","11":"XI","12":"XII","13":"XIII",
    "14":"XIV","15":"XV","16":"XVI","17":"XVII","18":"XVIII","19":"XIX","20":"XX",
}
_ORDINAL_TO_ROMAN = {
    "first":"I","second":"II","third":"III","fourth":"IV","fifth":"V",
    "sixth":"VI","seventh":"VII","eighth":"VIII","ninth":"IX","tenth":"X",
}


# ── Query normalisation ───────────────────────────────────────────────────────
def normalize_query(query: str) -> str:
    for word, roman in _ORDINAL_TO_ROMAN.items():
        query = re.sub(rf'\bthe\s+{word}\b', roman, query, flags=re.IGNORECASE)
    for word, roman in _ORDINAL_TO_ROMAN.items():
        query = re.sub(rf'\b{word}\b', roman, query, flags=re.IGNORECASE)
    def _replace_num(m):
        roman = _ARABIC_TO_ROMAN.get(m.group(2))
        return f"{m.group(1)} {roman}" if roman else m.group(0)
    query = re.sub(r'([A-Z][a-zA-Z]+)\s+(\d{1,2})\b', _replace_num, query)
    return query
