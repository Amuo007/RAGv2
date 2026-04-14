"""
RAG Evaluation Script — hits the real running server via HTTP.
Writes each result to disk immediately after each LLM call (low RAM usage).

Usage:
    # First start the server:
    #   cd RAGv2 && python server.py

    python evaluate_rag.py --pipeline search
    python evaluate_rag.py --pipeline article
    python evaluate_rag.py --pipeline search --limit 20 --out report.json --server http://localhost:8000
"""

import argparse
import json
import sys
import time

import requests

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pipeline", choices=["search", "article"], default="search",
                    help="'search' → /search <question>  |  'article' → /article <title> | <question>")
parser.add_argument("--server",  default="http://localhost:8000", help="Server base URL")
parser.add_argument("--limit",   type=int, default=None, help="Only evaluate first N questions")
parser.add_argument("--out",     default="rag_eval_report.json", help="Output JSON filename")
parser.add_argument("--dataset", default="golden_dataset.json", help="Path to golden dataset")
parser.add_argument("--top-k",   type=int, default=None, help="Override top_k for retrieval")
args = parser.parse_args()

BASE = args.server.rstrip("/")

# ── Helpers ───────────────────────────────────────────────────────────────────
def create_chat() -> str:
    r = requests.post(f"{BASE}/chats", json={}, timeout=15)
    r.raise_for_status()
    return r.json()["id"]

def send_message(chat_id: str, message: str) -> tuple[str, list[dict]]:
    payload = {"chat_id": chat_id, "message": message}
    if args.top_k:
        payload["top_k"] = args.top_k

    r = requests.post(f"{BASE}/chat", json=payload, stream=True, timeout=120)
    if not r.ok:
        raise Exception(f"{r.status_code} - {r.text[:300]}")

    full_text = ""
    sources   = []
    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
        if not chunk:
            continue
        if "[STATS]" in chunk:
            text_part, _, stats_part = chunk.partition("[STATS]")
            full_text += text_part
            try:
                stats   = json.loads(stats_part)
                sources = stats.get("sources", [])
            except Exception:
                pass
        else:
            full_text += chunk

    return full_text.strip(), sources

def delete_chat(chat_id: str):
    requests.delete(f"{BASE}/chats/{chat_id}", timeout=10)

# ── Check server is up ────────────────────────────────────────────────────────
try:
    requests.get(f"{BASE}/chats", timeout=5).raise_for_status()
except Exception as e:
    print(f"ERROR: Cannot reach server at {BASE}\n  {e}")
    print("Start it first:  cd RAGv2 && python server.py")
    sys.exit(1)

# ── Load dataset ──────────────────────────────────────────────────────────────
with open(args.dataset) as f:
    dataset = json.load(f)

if args.limit:
    dataset = dataset[: args.limit]

total = len(dataset)
print(f"Pipeline : {args.pipeline}")
print(f"Server   : {BASE}")
print(f"Questions: {total}")
print(f"Output   : {args.out}\n")

# ── Running counters (no results list kept in RAM) ────────────────────────────
n_full_hits = 0
n_partial   = 0
n_misses    = 0
total_lat   = 0.0
missed_ids  = []

# ── Open file and write opening bracket ──────────────────────────────────────
out_f = open(args.out, "w", encoding="utf-8")
out_f.write('{\n  "results": [\n')

# ── Run evaluation ────────────────────────────────────────────────────────────
for idx, item in enumerate(dataset, 1):
    qid      = item.get("id", f"q{idx:04d}")
    question = item["question"]
    expected = item["answer"]
    article  = item.get("article", "").strip('"').strip()

    exp_chunks = item.get("relevant_chunks", [])
    exp_ids    = set(c["db_id"] for c in exp_chunks)
    text_to_id = {c["chunk_text"].strip(): c["db_id"] for c in exp_chunks}

    if args.pipeline == "search":
        command = f"/search {question}"
    else:
        command = f"/article {article} | {question}"

    print(f"[{idx}/{total}] {qid}: {question[:65]}...")
    print(f"  → {command[:80]}")

    chat_id = create_chat()
    t0 = time.time()

    try:
        response, sources = send_message(chat_id, command)
    except Exception as e:
        print(f"  ERROR: {e}")
        response, sources = f"ERROR: {e}", []

    latency = round(time.time() - t0, 2)

    # Match retrieved chunk ids
    retrieved_ids = []
    for src in sources:
        if "db_id" in src:
            retrieved_ids.append(src["db_id"])
        else:
            src_text = src.get("text", "").strip()
            if src_text in text_to_id:
                retrieved_ids.append(text_to_id[src_text])

    hit_ids    = exp_ids & set(retrieved_ids)
    chunks_hit = len(hit_ids) == len(exp_ids) and len(exp_ids) > 0
    any_hit    = len(hit_ids) > 0

    # Update counters
    total_lat += latency
    if chunks_hit:
        n_full_hits += 1
    elif any_hit:
        n_partial += 1
    else:
        n_misses += 1
        missed_ids.append(qid)

    hit_label = "YES" if chunks_hit else ("PARTIAL" if any_hit else "NO")
    print(f"  Chunk hit: {hit_label}  |  Latency: {latency}s")

    # Build result object and write immediately to disk
    result = {
        "id":                  qid,
        "article":             item.get("article", ""),
        "question":            question,
        "pipeline_command":    command,
        "expected_answer":     expected,
        "llm_response":        response,
        "expected_chunk_ids":  sorted(exp_ids),
        "retrieved_chunk_ids": sorted(retrieved_ids),
        "chunks_hit":          chunks_hit,
        "any_chunk_hit":       any_hit,
        "sources":             sources,
        "latency_s":           latency,
    }

    # Indent each result by 4 spaces; add comma between items
    prefix = "    " if idx == 1 else ",\n    "
    out_f.write(prefix + json.dumps(result, ensure_ascii=False, indent=2)
                .replace("\n", "\n    "))
    out_f.flush()

    # Delete from DB only after data is safely written to disk
    delete_chat(chat_id)

# ── Write evaluation summary and close ───────────────────────────────────────
avg_lat = round(total_lat / total, 2) if total else 0.0

evaluation = {
    "pipeline":         args.pipeline,
    "total_questions":  total,
    "full_hits":        n_full_hits,
    "full_hits_pct":    round(n_full_hits / total * 100, 1),
    "partial_hits":     n_partial,
    "partial_hits_pct": round(n_partial / total * 100, 1),
    "misses":           n_misses,
    "misses_pct":       round(n_misses / total * 100, 1),
    "avg_latency_s":    avg_lat,
    "missed_ids":       missed_ids,
}

out_f.write('\n  ],\n  "evaluation": ')
out_f.write(json.dumps(evaluation, indent=2).replace("\n", "\n  "))
out_f.write("\n}\n")
out_f.close()

print("\n" + "=" * 60)
print(f"Report saved : {args.out}")
print(f"Total        : {total}")
print(f"Full hits    : {n_full_hits}  ({evaluation['full_hits_pct']}%)")
print(f"Partial hits : {n_partial}  ({evaluation['partial_hits_pct']}%)")
print(f"Misses       : {n_misses}  ({evaluation['misses_pct']}%)")
print(f"Avg latency  : {avg_lat:.2f}s")
print("=" * 60)
