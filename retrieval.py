import re
import json
import pickle
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
CACHE_FILE          = _cfg["cache_file"]
FTS_DB_FILE         = _cfg["fts_db_file"]
RELEVANCE_THRESHOLD = _cfg.get("relevance_threshold", 0.03)

# ── Load pre-built indexes ────────────────────────────────────────────────────
print("Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_FILE)
if hasattr(faiss_index, 'nprobe'):
    faiss_index.nprobe = 64
print(f"  {faiss_index.ntotal} vectors")

print("Loading chunks cache...")
with open(CACHE_FILE, "rb") as f:
    _cache = pickle.load(f)
titles = _cache["titles"]
texts  = _cache["texts"]
print(f"  {len(titles)} chunks")


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
    match_expr = " ".join(
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
            score += 1.0 / (RRF_K + bm25_ranks[idx])
        rrf_scores[idx] = score

    q_words = set(re.findall(r'\w+', norm_query.lower())) - _STOPWORDS
    final = {
        idx: score + _title_boost(q_words, titles[idx]) * 0.15
        for idx, score in rrf_scores.items()
    }

    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
    return [(titles[i], texts[i], rrf_scores[i], cosine_scores.get(i)) for i, _ in ranked[:top_k]]
