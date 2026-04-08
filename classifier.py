import re

_STOPWORDS = {
    'the','a','an','is','was','were','did','do','does','when','where',
    'who','what','how','why','which','in','on','at','to','for','of',
    'and','or','but','it','he','she','they','we','you','his','her',
    'their','its','be','been','have','has','had','this','that','these',
    'those','will','would','could','should','about','from','with',
    # instruction/query words that appear in natural-language questions
    # but never in article content — keep these out of BM25 match expressions
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


# ── Query classifier ──────────────────────────────────────────────────────────
def classify_query(query: str) -> tuple:
    _NOT_NAMES = {
        'who', 'what', 'when', 'where', 'why', 'how', 'which', 'is', 'was',
        'were', 'did', 'do', 'does', 'the', 'a', 'an', 'in', 'on', 'at',
        'i', 'tell', 'me', 'about', 'and', 'or', 'of', 'for', 'to', 'by',
        'with', 'from', 'that', 'this', 'it', 'he', 'she', 'they', 'we',
        'his', 'her', 'their', 'its', 'my', 'your',
    }
    _PERSON_TRIGGERS = {
        'born', 'died', 'death', 'birth', 'age', 'biography', 'biographer',
        'nationality', 'wife', 'husband', 'married', 'children', 'father',
        'mother', 'brother', 'sister', 'son', 'daughter', 'family',
        'inventor', 'wrote', 'painted', 'discovered', 'awarded',
        'murdered', 'killed', 'executed', 'reign', 'ruled', 'elected',
        'graduated', 'childhood', 'parents', 'spouse', 'portrait',
    }
    _NOT_PERSONS = {
        'amazon', 'google', 'apple', 'microsoft', 'facebook', 'meta',
        'netflix', 'twitter', 'youtube', 'instagram', 'tiktok', 'spotify',
        'uber', 'airbnb', 'tesla', 'spacex', 'nvidia', 'intel', 'amd',
        'ibm', 'oracle', 'samsung', 'sony', 'nintendo', 'adobe', 'salesforce',
        'paypal', 'ebay', 'alibaba', 'walmart', 'target', 'ikea', 'nike',
        'adidas', 'coca', 'pepsi', 'mcdonalds', 'starbucks',
        'america', 'europe', 'asia', 'africa', 'australia', 'canada',
        'china', 'india', 'russia', 'japan', 'germany', 'france',
        'britain', 'england', 'scotland', 'ireland', 'italy', 'spain',
        'mexico', 'brazil', 'pakistan', 'ukraine', 'israel', 'iran',
        'london', 'paris', 'berlin', 'tokyo', 'beijing', 'moscow',
        'washington', 'york', 'angeles', 'chicago', 'toronto', 'sydney',
        'dubai', 'delhi', 'mumbai', 'lahore', 'karachi',
        'university', 'college', 'institute', 'corporation', 'company',
        'organization', 'government', 'department', 'ministry', 'agency',
        'wikipedia', 'reddit', 'discord', 'github', 'linkedin',
    }
    _ORG_SUFFIXES = {
        'inc', 'inc.', 'corp', 'corp.', 'ltd', 'ltd.', 'llc', 'llc.',
        'plc', 'plc.', 'co', 'co.', 'company', 'corporation', 'incorporated',
        'limited', 'group', 'holdings', 'enterprises', 'ventures', 'partners',
        'associates', 'foundation', 'institute', 'organization', 'agency',
        'department', 'ministry', 'bureau', 'committee', 'council',
        'university', 'college', 'school', 'academy', 'hospital', 'clinic',
    }

    words = query.split()
    query_lower = query.lower()

    if any(w.lower().rstrip('.') in _ORG_SUFFIXES for w in words):
        return ("generic", "")

    def _is_name_word(w: str) -> bool:
        return (
            bool(w) and w[0].isupper()
            and w.lower() not in _NOT_NAMES
            and w.lower() not in _NOT_PERSONS
            and bool(re.match(r'^[A-Za-z]', w))
        )

    name_words = [w for w in words if _is_name_word(w)]
    has_person_trigger = any(w in query_lower for w in _PERSON_TRIGGERS)

    if len(name_words) >= 2:
        return ("person", "")
    if len(name_words) == 1 and has_person_trigger:
        return ("person", "")
    if has_person_trigger:
        return ("person", "")
    return ("generic", "")
