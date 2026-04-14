# RAGv2 — Wikipedia Retrieval-Augmented Generation

A fully offline RAG system built on Wikipedia, running on a Raspberry Pi 5.  
Uses hybrid FAISS + BM25 retrieval with a local LLM (Gemma 3) via Ollama.

---

## Stack

| Component | Technology |
|-----------|-----------|
| Embedding | `embeddinggemma` via Ollama |
| LLM | `gemma3:1b` via Ollama |
| Vector search | FAISS IVFFlat (inner product) |
| Keyword search | SQLite FTS5 (Porter stemmer) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Backend | FastAPI + SQLite |
| Knowledge base | Wikipedia EN 100k articles (ZIM archive) |

---

## Project Structure

```
RAGv2/
├── server.py           # FastAPI server entry point
├── retrieval.py        # FAISS + BM25 + RRF hybrid retrieval
├── llm.py              # Ollama LLM + embedding helpers
├── classifier.py       # Query normalisation + stopwords
├── filters.py          # Relevance threshold filter
├── database.py         # Chat history SQLite
├── schemas.py          # Pydantic request models
├── state.py            # Session ID
├── config.json         # All configuration
├── evaluate_rag.py     # End-to-end evaluation script
├── golden_dataset.json # 300 Q&A pairs with ground-truth chunk IDs
├── routes/
│   ├── chat.py         # POST /chat — main RAG endpoint
│   ├── chats.py        # Chat CRUD
│   └── misc.py         # Health / misc routes
├── faiss.index         # Pre-built FAISS vector index
├── chunks_lookup.db    # SQLite: id → title, chunk_text
└── fts_index.db        # SQLite FTS5 BM25 keyword index
```

---

## Quick Start

### 1. Start Ollama
```bash
ollama serve
ollama pull gemma3:1b
ollama pull embeddinggemma
```

### 2. Start the server
```bash
cd RAGv2
python server.py
```

### 3. Open the UI
```
http://localhost:8000
```

---

## Pipelines

### `/search` — Full Hybrid RAG

```
User: "/search <question>"
           │
           ▼
    embed_query()
    POST Ollama /api/embed
    → float32 vector (L2 normalised)
           │
           ▼
    ┌──────────────────────────────────┐
    │         retrieve()               │
    │                                  │
    │  FAISS search   BM25 search      │
    │  top 60 chunks  top 60 chunks    │
    │  (semantic)     (keyword/FTS5)   │
    │       │               │          │
    │       └──────┬────────┘          │
    │              ▼                   │
    │     RRF Fusion scoring           │
    │     FAISS weight: 1.0×           │
    │     BM25  weight: 2.0×           │
    │              │                   │
    │              ▼                   │
    │     Title boost     +0.15        │
    │     Answer pattern  +0.10        │
    │              │                   │
    │              ▼                   │
    │     Top 3 chunks selected        │
    └──────────────┬───────────────────┘
                   │
                   ▼
    generic_relevance_filter()
    drop chunks below threshold (0.01)
                   │
           ┌───────┴───────┐
        chunks ok       no chunks
           │               │
           ▼               ▼
    build_rag_prompt()   raw query
    inject context         │
           └───────┬───────┘
                   ▼
    stream_generate()
    POST Ollama /api/generate
    model: gemma3:1b
    streams tokens → client
    appends [STATS] at end
```

---

### `/article` — Article-Scoped Search

```
User: "/article <title> | <question>"
           │
           ▼
    Parse: split on " | "
    article_title = "<title>"
    article_query = "<question>"
           │
           ▼
    embed_query(article_query)
    POST Ollama /api/embed
           │
           ▼
    retrieve_for_article()
    ┌──────────────────────────────┐
    │ lookup all chunk ids for     │
    │ this article only            │
    │          │                   │
    │          ▼                   │
    │ cosine similarity per chunk  │
    │ (no BM25, no RRF)            │
    │          │                   │
    │          ▼                   │
    │ top 3 by similarity          │
    └──────────┬───────────────────┘
               │
               ▼
    build_rag_prompt()
    stream_generate()
    streams tokens → client
```

**Key difference:**

| | `/search` | `/article` |
|---|---|---|
| Search space | Entire index | One article only |
| Method | FAISS + BM25 + RRF | Cosine similarity |
| Relevance filter | Yes | No |
| Accuracy (eval) | 69% hit rate | ~92% hit rate |

---

## Evaluation

Requires server running. Set `"test": true` in `config.json` so the server includes `db_id` in sources for accurate chunk matching.

```bash
# Install RAM sampler dependency
pip install psutil

# search pipeline — full dataset
python evaluate_rag.py --pipeline search --out report_search.json

# article pipeline — full dataset
python evaluate_rag.py --pipeline article --out report_article.json

# quick test — first 5 questions only
python evaluate_rag.py --pipeline search --limit 5 --out test.json
```

### Output JSON structure

```json
{
  "results": [
    {
      "id": "q0001",
      "article": "...",
      "question": "...",
      "pipeline_command": "/search ...",
      "expected_answer": "...",
      "llm_response": "...",
      "expected_chunk_ids": [5],
      "retrieved_chunk_ids": [2, 4, 5],
      "chunks_hit": true,
      "any_chunk_hit": true,
      "sources": [
        {"title": "...", "score": 0.05, "text": "...", "db_id": 5}
      ],
      "latency_s": 1.91,
      "llm_stats": {
        "input_tokens":   245,
        "output_tokens":  18,
        "total_tokens":   263,
        "tokens_per_sec": 20.22,
        "ttft_ms":        312,
        "gen_ms":         890,
        "embed_ms":       45,
        "retrieve_ms":    120,
        "total_ms":       1400
      },
      "ram": {
        "ram_samples_mb": [1820.3, 1821.1],
        "ram_min_mb":     1820.3,
        "ram_max_mb":     1821.1,
        "ram_avg_mb":     1820.7
      }
    }
  ],
  "evaluation": {
    "pipeline":         "search",
    "total_questions":  300,
    "full_hits":        207,
    "full_hits_pct":    69.0,
    "partial_hits":     11,
    "partial_hits_pct": 3.7,
    "misses":           82,
    "misses_pct":       27.3,
    "avg_latency_s":    0.80,
    "missed_ids":       ["q0018", "q0019", "..."]
  }
}
```

### Metrics explained

| Field | Meaning |
|-------|---------|
| `chunks_hit` | ALL expected chunk IDs were retrieved |
| `any_chunk_hit` | AT LEAST ONE expected chunk was retrieved |
| `ttft_ms` | Time to first token from LLM |
| `gen_ms` | Pure LLM generation time |
| `tokens_per_sec` | LLM generation speed (output tokens / gen time) |
| `embed_ms` | Time to embed the query |
| `retrieve_ms` | Time to run FAISS + BM25 retrieval |
| `total_ms` | Full server-side time from request to last token |
| `ram_max_mb` | Peak system RAM during this cycle (sampled every 4s) |

---

## Configuration

`config.json`:

```json
{
  "ollama_url":               "http://localhost:11434",
  "embed_model":              "embeddinggemma",
  "llm_model":                "gemma3:1b",
  "top_k":                    3,
  "faiss_k":                  60,
  "bm25_k":                   60,
  "rrf_k":                    60,
  "chunk_max_chars":          800,
  "relevance_threshold":      0.01,
  "test":                     false
}
```

| Key | Description |
|-----|-------------|
| `top_k` | Number of chunks returned to LLM |
| `faiss_k` | Candidates fetched from FAISS before RRF |
| `bm25_k` | Candidates fetched from BM25 before RRF |
| `rrf_k` | RRF smoothing constant |
| `relevance_threshold` | Minimum RRF score to keep a chunk |
| `test` | Set `true` to include `db_id` in API sources response |

---

## Results

| Pipeline | Full Hit | Partial | Miss |
|----------|----------|---------|------|
| `/search` | 69.0% | 3.7% | 27.3% |
| `/article` | ~92% | — | — |

Evaluated on 300 questions from the golden dataset.
