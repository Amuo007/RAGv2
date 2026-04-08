import re

from classifier import _STOPWORDS

# ── Rule-based title filter ───────────────────────────────────────────────────
_SUFFIX_PATTERN = re.compile(
    r'\b(i{1,3}|iv|vi{0,3}|ix|xi{0,2}|jr|sr|1st|2nd|3rd|\d{1,2}th)\b',
    re.IGNORECASE
)

def _normalize_to_roman(text: str) -> str:
    text = text.lower()
    ordinal_words = {
        "first": "i", "second": "ii", "third": "iii", "fourth": "iv",
        "fifth": "v", "sixth": "vi", "seventh": "vii", "eighth": "viii",
        "ninth": "ix", "tenth": "x", "eleventh": "xi", "twelfth": "xii",
    }
    for word, roman in ordinal_words.items():
        text = re.sub(rf'\bthe\s+{word}\b', roman, text)
        text = re.sub(rf'\b{word}\b', roman, text)
    ordinal_nums = {
        "1st": "i", "2nd": "ii", "3rd": "iii", "4th": "iv",
        "5th": "v", "6th": "vi", "7th": "vii", "8th": "viii",
        "9th": "ix", "10th": "x",
    }
    for num, roman in ordinal_nums.items():
        text = re.sub(rf'\b{num}\b', roman, text)
    arabic_to_roman = {
        "1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v",
        "6": "vi", "7": "vii", "8": "viii", "9": "ix", "10": "x",
        "11": "xi", "12": "xii",
    }
    def _replace_bare_num(m):
        return arabic_to_roman.get(m.group(1), m.group(1))
    text = re.sub(r'\b(\d{1,2})\b', _replace_bare_num, text)
    text = re.sub(r'\bjr\.\b', 'jr', text)
    text = re.sub(r'\bsr\.\b', 'sr', text)
    return text.strip()


def _extract_name_from_query(query: str) -> str:
    noise = r'\b(' + '|'.join([
        'who','what','when','where','why','how','which','whose',
        'the','a','an','in','on','at','to','for','of','by',
        'with','from','into','onto','upon','within','without',
        'through','during','before','after','above','below',
        'between','among','around','along','across','behind',
        'beyond','near','over','under','up','down','off','out',
        'and','or','but','nor','so','yet','both','either',
        'neither','not','also',
        'i','me','my','we','us','our','you','your',
        'he','him','his','she','her','hers',
        'they','them','their','it','its',
        'is','are','was','were','be','been','being',
        'do','does','did','done','have','has','had',
        'will','would','could','should','shall','may','might',
        'can','must','need','dare','ought',
        'tell','told','say','said','give','gave','get','got',
        'make','made','know','knew','think','thought',
        'want','wanted','like','liked','let','lets',
        'show','showed','find','found','look','looked',
        'go','went','come','came','take','took',
        'explain','describe','summarize','summary','elaborate',
        'detail','details','detailed','brief','briefly','quick',
        'quickly','provide','list','write','help','please',
        'about','regarding','concerning','related','information',
        'info','facts','data','overview','introduction','intro',
        'life','story','history','biography','bio','profile',
        'background','career','early','later','death','birth',
        'childhood','youth','education','legacy','impact','work',
        'works','achievement','achievements','contribution','contributions',
        'timeline','events','journey','rise','fall',
        'famous','known','well','great','important','notable',
        'significant','popular','legendary','historic','historical',
        'real','true','full','complete','whole','entire',
        'general','main','major','key','top','best',
        'interesting','good','big','long','short','old','new',
        'born','died','age','year','years','date','time',
        'century','decade','era','period','today','yesterday',
        'now','then','still','ever','never','always','recently',
        'currently','originally','finally',
        'wife','husband','married','marriage','children','child',
        'father','mother','son','daughter','brother','sister',
        'family','parents','parent','spouse','partner',
        'that','this','these','those','there','here',
        'such','some','any','all','each','every','more','most',
        'much','many','few','other','another','same','different',
        'than','as','if','then','just','only','even','very',
        'quite','really','actually','basically','specifically',
        'especially','particularly','generally','usually','often',
    ]) + r')\b'
    name = re.sub(noise, ' ', query, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _has_suffix(text: str) -> bool:
    return bool(_SUFFIX_PATTERN.search(text))


def rule_based_title_filter(query: str, chunks: list) -> list:
    raw_name    = _extract_name_from_query(query)
    norm_name   = _normalize_to_roman(raw_name)
    query_words = set(re.findall(r'\w+', norm_name)) - _STOPWORDS
    query_has_suffix = _has_suffix(norm_name)

    filtered = []
    for chunk in chunks:
        title      = chunk[0]
        norm_title = _normalize_to_roman(title)
        title_has_suffix = _has_suffix(norm_title)

        if query_has_suffix and title_has_suffix:
            query_suffix = " ".join(m.group() for m in _SUFFIX_PATTERN.finditer(norm_name))
            title_suffix = " ".join(m.group() for m in _SUFFIX_PATTERN.finditer(norm_title))
            if query_suffix != title_suffix:
                continue
        if query_has_suffix and not title_has_suffix:
            continue

        title_words = set(re.findall(r'\w+', norm_title)) - _STOPWORDS
        if not query_words:
            filtered.append(chunk)
            continue

        overlap = query_words & title_words
        score   = len(overlap) / len(title_words) if title_words else 0.0
        if score >= 0.5:
            filtered.append(chunk)

    return filtered if filtered else None


def generic_relevance_filter(query: str, chunks: list, threshold: float) -> list:
    """Return chunks that pass the relevance threshold or have sufficient title overlap."""
    query_words = set(re.findall(r'\w+', query.lower())) - _STOPWORDS
    def _title_match(title):
        tw = set(re.findall(r'\w+', title.lower())) - _STOPWORDS
        if not query_words or not tw:
            return False
        return len(query_words & tw) / len(query_words) >= 0.5
    return [c for c in chunks if c[2] >= threshold or _title_match(c[0])]
