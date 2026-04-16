"""
Reads the first 10 results from medqa_search_full.json, looks up chunk text
from chunks_lookup.db for each db_id, and writes a new enriched JSON file.

Usage:
    python enrich_chunks.py
    python enrich_chunks.py --input medqa_search_full.json --output enriched.json --limit 10
"""

import argparse
import json
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="medqa_search_full.json")
parser.add_argument("--output", default="medqa_enriched.json")
parser.add_argument("--db",     default="chunks_lookup.db")
parser.add_argument("--limit",  type=int, default=10)
args = parser.parse_args()

# ── Load chunk text from DB ───────────────────────────────────────────────────
conn = sqlite3.connect(args.db)

def fetch_chunk_text(db_id: int) -> str:
    if db_id is None:
        return ""
    row = conn.execute(
        "SELECT chunk_text FROM chunks_lookup WHERE id = ?", (db_id,)
    ).fetchone()
    return row[0] if row else ""

# ── Load source JSON ──────────────────────────────────────────────────────────
with open(args.input) as f:
    data = json.load(f)

results = data["results"][: args.limit]

# ── Enrich each result ────────────────────────────────────────────────────────
for result in results:
    for chunk in result.get("chunks_used", []):
        chunk["chunk_text"] = fetch_chunk_text(chunk.get("db_id"))

conn.close()

# ── Write output ──────────────────────────────────────────────────────────────
output = {
    "source_file": args.input,
    "total_shown": len(results),
    "results": results,
}

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Done — {len(results)} results written to {args.output}")
