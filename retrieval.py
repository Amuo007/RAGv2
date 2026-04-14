import re
import json
import sqlite3
import requests
import numpy as np
import faiss

from classifier import normalize_query, _STOPWORDS

# ── Config ───────────────────────────────────────────────────────────────────
with open("config.json") as f:
    _cfg = json.load(f)

OLLAMA_URL          = _cfg["ollama_url"]
EMBED_MODEL         = _cfg["embed_model"]
TOP_K               = _cfg["top_k"]
FAISS_K             = _cfg["faiss_k"]
BM25_K              = _cfg["bm25_k"]
RRF_K               = _cfg["rrf_k"]
FAISS_FILE          = _cfg["faiss_file"]
LOOKUP_DB_FILE      = _cfg["cache_file"]   # SQLite lookup DB built by build_index.py
FTS_DB_FILE         = _cfg["fts_db_file"]
RELEVANCE_THRESHOLD = _cfg.get("relevance_threshold", 0.03)

# ── Load pre-built indexes ────────────────────────────────────────────────────
print("Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_FILE)
if hasattr(faiss_index, 'nprobe'):
    faiss_index.nprobe = 64
if isinstance(faiss_index, faiss.IndexIVF):
    faiss_index.make_direct_map()
print(f"  {faiss_index.ntotal} vectors")

# Load only titles at startup (tiny compared to texts) so _article_chunks and
# title-boost scoring work without keeping all chunk text in RAM.
print("Loading titles from lookup DB...")
_lookup_conn = sqlite3.connect(LOOKUP_DB_FILE, check_same_thread=False)
_titles_map: dict[int, str] = {
    row[0]: row[1]
    for row in _lookup_conn.execute("SELECT id, title FROM chunks_lookup").fetchall()
}
print(f"  {len(_titles_map)} chunks")

# Build article → chunk indices lookup for /article command
_article_chunks: dict = {}
for _i, _t in _titles_map.items():
    _article_chunks.setdefault(_t, []).append(_i)
print(f"  {len(_article_chunks)} unique articles indexed")


def _fetch_chunks(indices: list[int]) -> dict[int, tuple[str, str]]:
    """Return {id: (title, chunk_text)} for the given FAISS indices."""
    if not indices:
        return {}
    placeholders = ",".join("?" * len(indices))
    rows = _lookup_conn.execute(
        f"SELECT id, title, chunk_text FROM chunks_lookup WHERE id IN ({placeholders})",
        indices,
    ).fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


# ── Core retrieval ────────────────────────────────────────────────────────────
def embed_query(text: str) -> np.ndarray:
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text, "truncate": True},
        timeout=30,
    )
    vec = np.array(r.json()["embeddings"][0], dtype=np.float32)
    return vec / np.clip(np.linalg.norm(vec), 1e-9, None)


def _bm25_search(query: str, top_k: int):
    words = re.findall(r'\w+', query)
    if not words:
        return []
    match_expr = " OR ".join(
        f'"{w}"' for w in words
        if len(w) >= 2 and w.lower() not in _STOPWORDS
    )
    if not match_expr:
        return []
    try:
        fts = sqlite3.connect(FTS_DB_FILE)
        rows = fts.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (match_expr, top_k),
        ).fetchall()
        fts.close()
        return [(int(row[0]) - 1, rank) for rank, row in enumerate(rows)]
    except Exception as e:
        print(f"BM25 error: {e}")
        return []


def _title_boost(query_words: set, title: str) -> float:
    if not query_words:
        return 0.0
    title_words = set(re.findall(r'\w+', title.lower()))
    return len(query_words & title_words) / len(query_words)


def retrieve_for_article(qvec: np.ndarray, article_title: str, top_k: int = TOP_K):
    """Similarity search restricted to chunks of a specific article."""
    indices = _article_chunks.get(article_title, [])
    if not indices:
        return []
    scored = []
    for i in indices:
        try:
            vec = np.array(faiss_index.reconstruct(i), dtype=np.float32)
            sim = float(np.dot(vec, qvec))
        except Exception:
            sim = 0.0
        scored.append((i, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [i for i, _ in scored[:top_k]]
    chunks = _fetch_chunks(top_ids)
    return [(chunks[i][0], chunks[i][1], s, s, i) for i, s in scored[:top_k] if i in chunks]


# Answer-pattern phrases that indicate definitional / classification content
_ANSWER_PATTERNS = re.compile(
    r'\b(is a|is an|stimulant|depressant|hallucinogen|psychedelic|narcotic|analgesic'
    r'|drug|compound|substance|chemical|medication|pharmaceutical|receptor|agonist|antagonist)\b',
    re.IGNORECASE,
)

# BM25 weight multiplier — makes BM25 rank contributions twice as heavy as FAISS
_BM25_WEIGHT = 2.0


def retrieve(qvec: np.ndarray, query: str, top_k: int = TOP_K):
    norm_query = normalize_query(query)
    vec_scores, vec_indices = faiss_index.search(qvec.reshape(1, -1), FAISS_K)
    vec_ranks     = {int(idx): rank for rank, idx in enumerate(vec_indices[0]) if idx >= 0}
    cosine_scores = {int(idx): float(vec_scores[0][rank]) for rank, idx in enumerate(vec_indices[0]) if idx >= 0}

    bm25_ranks = {idx: rank for idx, rank in _bm25_search(norm_query, BM25_K)}
    if norm_query != query:
        for idx, rank in _bm25_search(query, BM25_K):
            if idx not in bm25_ranks:
                bm25_ranks[idx] = rank

    all_indices = set(vec_ranks) | set(bm25_ranks)
    rrf_scores = {}
    for idx in all_indices:
        score = 0.0
        if idx in vec_ranks:
            score += 1.0 / (RRF_K + vec_ranks[idx])
        if idx in bm25_ranks:
            score += _BM25_WEIGHT / (RRF_K + bm25_ranks[idx])
        rrf_scores[idx] = score

    q_words = set(re.findall(r'\w+', norm_query.lower())) - _STOPWORDS
    top_ids_pre = [i for i, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]]
    chunks_pre = _fetch_chunks(top_ids_pre)

    # Fix 2: boost chunks that contain answer-pattern phrases (definitional content)
    final = {}
    for idx, score in rrf_scores.items():
        if idx not in chunks_pre:
            continue
        boost = _title_boost(q_words, _titles_map.get(idx, "")) * 0.15
        chunk_text = chunks_pre[idx][1]
        if _ANSWER_PATTERNS.search(chunk_text):
            boost += 0.10
        final[idx] = score + boost

    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
    top_ids = [i for i, _ in ranked[:top_k]]
    chunks = _fetch_chunks(top_ids)
    return [(chunks[i][0], chunks[i][1], rrf_scores[i], cosine_scores.get(i), i) for i in top_ids if i in chunks]
