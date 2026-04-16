"""
RAG Evaluation Script — hits the real running server via HTTP.
Writes each result to disk immediately after each LLM call (low RAM usage).
Samples RAM every 4s per cycle via a background thread (Pi 5 friendly).

Pipelines:
  search  — full hybrid RAG (/search command via server)
  article — article-scoped RAG (/article command via server)
  raw     — bare LLM with no retrieval, talks directly to Ollama

Dataset modes:
  default  — golden_dataset.json (chunk-hit scoring)
  --medqa  — HuggingFace bigbio/med_qa (A/B/C/D letter scoring)

Usage:
    python evaluate_rag.py --pipeline search  --out report_search.json
    python evaluate_rag.py --pipeline article --out report_article.json
    python evaluate_rag.py --pipeline raw     --out report_raw.json
    python evaluate_rag.py --pipeline search  --limit 5 --out test.json
    python evaluate_rag.py --medqa --pipeline search  --limit 50 --out medqa_search.json
    python evaluate_rag.py --medqa --pipeline raw     --limit 50 --out medqa_raw.json
"""

import argparse
import json
import re
import sys
import time
import threading

import requests

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    print("⚠ psutil not found — RAM tracking disabled. Install with: pip install psutil")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pipeline", choices=["search", "article", "raw"], default="search",
                    help="search | article | raw (no retrieval, direct Ollama)")
parser.add_argument("--server",  default="http://localhost:8000", help="Server base URL (search/article only)")
parser.add_argument("--limit",   type=int, default=None, help="Only evaluate first N questions")
parser.add_argument("--out",     default="rag_eval_report.json", help="Output JSON filename")
parser.add_argument("--dataset", default="golden_dataset.json", help="Path to golden dataset (ignored when --medqa is set)")
parser.add_argument("--top-k",   type=int, default=None, help="Override top_k for retrieval")
parser.add_argument("--medqa",   action="store_true", help="Use HuggingFace bigbio/med_qa instead of golden_dataset.json")
parser.add_argument("--medqa-split", default="test", help="med_qa split to use: test | validation | train (default: test)")
args = parser.parse_args()

BASE = args.server.rstrip("/")

# ── Load config (for raw pipeline Ollama settings) ────────────────────────────
with open("config.json") as _f:
    _cfg = _json.load(_f) if False else json.load(open("config.json"))

OLLAMA_URL = _cfg.get("ollama_url", "http://localhost:11434")
LLM_MODEL  = _cfg.get("llm_model",  "gemma3:1b")

# ── RAM sampler ───────────────────────────────────────────────────────────────
class RamSampler:
    """
    Samples system RAM every 4 seconds in a background thread.
    Fresh instance per cycle — no state bleeds between questions.
    """
    INTERVAL = 4

    def __init__(self):
        self._samples: list[float] = []
        self._stop   = threading.Event()
        self._thread = None

    def start(self):
        self._samples.clear()
        self._stop.clear()
        if _PSUTIL:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stop.wait(timeout=self.INTERVAL):
            try:
                mem = psutil.virtual_memory()
                self._samples.append(round(mem.used / 1024 / 1024, 1))
            except Exception:
                pass

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=6)
        if not self._samples:
            return {"ram_samples_mb": [], "ram_min_mb": None,
                    "ram_max_mb": None, "ram_avg_mb": None}
        return {
            "ram_samples_mb": self._samples,
            "ram_min_mb":     round(min(self._samples), 1),
            "ram_max_mb":     round(max(self._samples), 1),
            "ram_avg_mb":     round(sum(self._samples) / len(self._samples), 1),
        }


# ── med_qa loader ─────────────────────────────────────────────────────────────
def load_medqa(split: str = "test") -> list[dict]:
    """
    Load bigbio/med_qa from HuggingFace and normalise into a flat list of dicts:
      { id, question, options: {A:.., B:.., C:.., D:..}, answer_letter, answer_text }
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading GBaker/MedQA-USMLE-4-options ({split} split) from HuggingFace...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)

    items = []
    for i, row in enumerate(ds):
        # options is already a dict: {"A": "...", "B": "...", "C": "...", "D": "..."}
        opts = row["options"]
        answer_letter = row.get("answer_idx", "").strip().upper()
        answer_text   = opts.get(answer_letter, row.get("answer", "")).strip()

        if not answer_letter or answer_letter not in opts:
            continue

        items.append({
            "id":            row.get("id", f"medqa_{i}"),
            "question":      row["question"].strip(),
            "options":       opts,
            "answer_letter": answer_letter,
            "answer_text":   answer_text,
        })

    print(f"Loaded {len(items)} med_qa questions from '{split}' split.")
    return items


def format_medqa_prompt(item: dict) -> str:
    """Format a med_qa item as an MCQ prompt the LLM can answer with a single letter."""
    lines = [item["question"], ""]
    for key in sorted(item["options"]):
        lines.append(f"{key}) {item['options'][key]}")
    lines.append("")
    lines.append("Answer with only the letter A, B, C, or D.")
    return "\n".join(lines)


def extract_letter(response: str) -> str | None:
    """
    Pull the first A/B/C/D answer letter out of the LLM response.
    Handles formats like: 'A', 'A)', 'A.', 'The answer is A', '(A)', etc.
    Returns None if no letter found.
    """
    # Prefer explicit "answer is X" patterns first
    m = re.search(r'\banswer\s+is\s+([A-D])\b', response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Standalone letter at start of response
    m = re.match(r'^\s*([A-D])[)\.\s]', response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Any standalone letter
    m = re.search(r'\b([A-D])\b', response)
    if m:
        return m.group(1).upper()
    return None


# ── Server helpers (search / article) ────────────────────────────────────────
def create_chat() -> str:
    r = requests.post(f"{BASE}/chats", json={}, timeout=15)
    r.raise_for_status()
    return r.json()["id"]

def send_message(chat_id: str, message: str) -> tuple[str, list[dict], dict]:
    """Returns (response_text, sources, llm_stats)"""
    payload = {"chat_id": chat_id, "message": message}
    if args.top_k:
        payload["top_k"] = args.top_k

    r = requests.post(f"{BASE}/chat", json=payload, stream=True, timeout=120)
    if not r.ok:
        raise Exception(f"{r.status_code} - {r.text[:300]}")

    full_text = ""
    sources   = []
    llm_stats = {}

    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
        if not chunk:
            continue
        if "[STATS]" in chunk:
            text_part, _, stats_part = chunk.partition("[STATS]")
            full_text += text_part
            try:
                stats      = json.loads(stats_part)
                sources    = stats.get("sources", [])
                gen_tokens = stats.get("gen_tokens", 0)
                gen_ms     = stats.get("gen_ms", 0)
                tok_per_s  = round(gen_tokens / (gen_ms / 1000), 2) if gen_ms > 0 else 0
                llm_stats  = {
                    "input_tokens":   stats.get("prompt_tokens", 0),
                    "output_tokens":  gen_tokens,
                    "total_tokens":   stats.get("prompt_tokens", 0) + gen_tokens,
                    "tokens_per_sec": tok_per_s,
                    "ttft_ms":        stats.get("ttft_ms", 0),
                    "gen_ms":         gen_ms,
                    "embed_ms":       stats.get("embed_ms", 0),
                    "retrieve_ms":    stats.get("retrieve_ms", 0),
                    "total_ms":       stats.get("total_ms", 0),
                }
            except Exception:
                pass
        else:
            full_text += chunk

    return full_text.strip(), sources, llm_stats

def delete_chat(chat_id: str):
    requests.delete(f"{BASE}/chats/{chat_id}", timeout=10)


# ── Raw LLM helper (no retrieval, direct Ollama) ──────────────────────────────
def raw_llm_call(question: str) -> tuple[str, dict]:
    """Send question straight to Ollama — no context, no retrieval."""
    t_start  = time.time()
    t_first  = None
    response = ""
    prompt_tokens = 0
    gen_tokens    = 0

    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": question, "stream": True},
        stream=True,
        timeout=120,
    )
    r.raise_for_status()

    for line in r.iter_lines():
        if not line:
            continue
        data  = json.loads(line)
        token = data.get("response", "")
        if token and t_first is None:
            t_first = time.time()
        response += token
        if data.get("done"):
            prompt_tokens = data.get("prompt_eval_count", 0)
            gen_tokens    = data.get("eval_count", 0)

    total_ms = round((time.time() - t_start) * 1000)
    ttft_ms  = round((t_first - t_start) * 1000) if t_first else 0
    gen_ms   = round((time.time() - (t_first or t_start)) * 1000)
    tok_per_s = round(gen_tokens / (gen_ms / 1000), 2) if gen_ms > 0 else 0

    llm_stats = {
        "input_tokens":   prompt_tokens,
        "output_tokens":  gen_tokens,
        "total_tokens":   prompt_tokens + gen_tokens,
        "tokens_per_sec": tok_per_s,
        "ttft_ms":        ttft_ms,
        "gen_ms":         gen_ms,
        "embed_ms":       0,
        "retrieve_ms":    0,
        "total_ms":       total_ms,
    }
    return response.strip(), llm_stats


# ── Keyword correctness scoring (used for raw pipeline) ──────────────────────
_STOPWORDS = {
    'the','a','an','is','are','was','were','of','and','or','in','on',
    'at','to','for','it','its','by','as','with','from','that','this',
    'be','been','has','have','had','their','which','who','what','how',
}

def keyword_score(response: str, expected: str) -> tuple[bool, float]:
    """
    Checks how many key words from the expected answer appear in the response.
    Returns (correct: bool, score: float 0-1).
    Threshold: 0.5 — at least half the key words must match.
    """
    exp_words = set(re.findall(r'\b\w+\b', expected.lower())) - _STOPWORDS
    res_words = set(re.findall(r'\b\w+\b', response.lower()))
    if not exp_words:
        return False, 0.0
    matches = exp_words & res_words
    score   = round(len(matches) / len(exp_words), 3)
    return score >= 0.5, score


# ── Check server (only needed for search / article) ───────────────────────────
if args.pipeline in ("search", "article"):
    try:
        requests.get(f"{BASE}/chats", timeout=5).raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach server at {BASE}\n  {e}")
        print("Start it first:  cd RAGv2 && python server.py")
        sys.exit(1)
else:
    # raw — just check Ollama is reachable
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama at {OLLAMA_URL}\n  {e}")
        print("Start it first:  ollama serve")
        sys.exit(1)

# ── Load dataset ──────────────────────────────────────────────────────────────
if args.medqa:
    dataset     = load_medqa(args.medqa_split)
    MEDQA_MODE  = True
else:
    with open(args.dataset) as f:
        dataset = json.load(f)
    MEDQA_MODE  = False

if args.limit:
    dataset = dataset[: args.limit]

total = len(dataset)
print(f"Pipeline : {args.pipeline}")
print(f"Dataset  : {'med_qa (' + args.medqa_split + ')' if MEDQA_MODE else args.dataset}")
print(f"Model    : {LLM_MODEL}")
if args.pipeline != "raw":
    print(f"Server   : {BASE}")
print(f"Questions: {total}")
print(f"RAM track: {'yes (every 4s)' if _PSUTIL else 'no (install psutil)'}")
print(f"Output   : {args.out}\n")

# ── Running counters ──────────────────────────────────────────────────────────
n_correct   = 0
n_incorrect = 0
n_partial   = 0   # golden dataset only
total_lat   = 0.0
missed_ids  = []

# ── Open output file ──────────────────────────────────────────────────────────
out_f = open(args.out, "w", encoding="utf-8")
out_f.write('{\n  "results": [\n')

# ── Run evaluation ────────────────────────────────────────────────────────────
for idx, item in enumerate(dataset, 1):
    qid = item.get("id", f"q{idx:04d}")

    if MEDQA_MODE:
        # ── med_qa item ───────────────────────────────────────────────────────
        prompt  = format_medqa_prompt(item)
        command = f"/search {prompt}" if args.pipeline == "search" else prompt

        print(f"[{idx}/{total}] {qid}: {item['question'][:65]}...")

        ram = RamSampler()
        ram.start()
        t0 = time.time()

        if args.pipeline in ("search", "article"):
            chat_id = create_chat()
            try:
                response, sources, llm_stats = send_message(chat_id, command)
            except Exception as e:
                print(f"  ERROR: {e}")
                response, sources, llm_stats = f"ERROR: {e}", [], {}
            delete_chat(chat_id)
        else:
            sources  = []
            llm_stats = {}
            try:
                response, llm_stats = raw_llm_call(prompt)
            except Exception as e:
                print(f"  ERROR: {e}")
                response, llm_stats = f"ERROR: {e}", {}

        latency   = round(time.time() - t0, 2)
        ram_stats = ram.stop()
        total_lat += latency

        extracted = extract_letter(response)
        correct   = extracted == item["answer_letter"]

        if correct:
            n_correct += 1
        else:
            n_incorrect += 1
            missed_ids.append(qid)

        chunks_used = [{"title": s.get("title",""), "score": s.get("score", 0), "db_id": s.get("db_id")}
                       for s in sources]

        print(f"  Correct: {'YES' if correct else 'NO'}  |  "
              f"Expected: {item['answer_letter']}  |  Got: {extracted}  |  "
              f"Latency: {latency}s  |  "
              f"Chunks used: {len(chunks_used)}  |  "
              f"RAM peak: {ram_stats.get('ram_max_mb','?')}MB")

        result = {
            "id":                  qid,
            "question":            item["question"],
            "options":             item["options"],
            "correct_answer":      item["answer_letter"],
            "correct_answer_text": item["answer_text"],
            "llm_response":        response,
            "extracted_letter":    extracted,
            "correct":             correct,
            "chunks_used":         chunks_used,
            "latency_s":           latency,
            "llm_stats":           llm_stats,
            "ram":                 ram_stats,
        }

    else:
        # ── golden dataset item ───────────────────────────────────────────────
        question = item["question"]
        expected = item["answer"]
        article  = item.get("article", "").strip('"').strip()

        exp_chunks = item.get("relevant_chunks", [])
        exp_ids    = set(c["db_id"] for c in exp_chunks)
        text_to_id = {c["chunk_text"].strip(): c["db_id"] for c in exp_chunks}

        if args.pipeline == "search":
            command = f"/search {question}"
        elif args.pipeline == "article":
            command = f"/article {article} | {question}"
        else:
            command = question

        print(f"[{idx}/{total}] {qid}: {question[:65]}...")

        ram = RamSampler()
        ram.start()
        t0 = time.time()

        if args.pipeline in ("search", "article"):
            chat_id = create_chat()
            try:
                response, sources, llm_stats = send_message(chat_id, command)
            except Exception as e:
                print(f"  ERROR: {e}")
                response, sources, llm_stats = f"ERROR: {e}", [], {}

            latency   = round(time.time() - t0, 2)
            ram_stats = ram.stop()

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
            kw_correct, kw_score = keyword_score(response, expected)

            total_lat += latency
            if chunks_hit:
                n_correct += 1
            elif any_hit:
                n_partial += 1
            else:
                n_incorrect += 1
                missed_ids.append(qid)

            hit_label = "YES" if chunks_hit else ("PARTIAL" if any_hit else "NO")
            print(f"  Chunk hit: {hit_label}  |  KW score: {kw_score}  |  Latency: {latency}s  |  "
                  f"Tokens in/out: {llm_stats.get('input_tokens',0)}/{llm_stats.get('output_tokens',0)}  |  "
                  f"Tok/s: {llm_stats.get('tokens_per_sec',0)}  |  "
                  f"TTFT: {llm_stats.get('ttft_ms',0)}ms  |  "
                  f"RAM peak: {ram_stats.get('ram_max_mb','?')}MB")

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
                "keyword_correct":     kw_correct,
                "keyword_score":       kw_score,
                "sources":             sources,
                "latency_s":           latency,
                "llm_stats":           llm_stats,
                "ram":                 ram_stats,
            }
            delete_chat(chat_id)

        else:
            try:
                response, llm_stats = raw_llm_call(question)
            except Exception as e:
                print(f"  ERROR: {e}")
                response, llm_stats = f"ERROR: {e}", {}

            latency   = round(time.time() - t0, 2)
            ram_stats = ram.stop()
            total_lat += latency

            print(f"  Latency: {latency}s  |  "
                  f"Tokens in/out: {llm_stats.get('input_tokens',0)}/{llm_stats.get('output_tokens',0)}  |  "
                  f"Tok/s: {llm_stats.get('tokens_per_sec',0)}  |  "
                  f"TTFT: {llm_stats.get('ttft_ms',0)}ms  |  "
                  f"RAM peak: {ram_stats.get('ram_max_mb','?')}MB")

            result = {
                "id":              qid,
                "question":        question,
                "expected_answer": expected,
                "llm_response":    response,
                "latency_s":       latency,
                "llm_stats":       llm_stats,
                "ram":             ram_stats,
            }

    out_f.write(("    " if idx == 1 else ",\n    ") +
                json.dumps(result, ensure_ascii=False, indent=2).replace("\n", "\n    "))
    out_f.flush()

# ── Write evaluation summary ──────────────────────────────────────────────────
avg_lat = round(total_lat / total, 2) if total else 0.0

if MEDQA_MODE:
    evaluation = {
        "dataset":         "GBaker/MedQA-USMLE-4-options",
        "split":           args.medqa_split,
        "pipeline":        args.pipeline,
        "model":           LLM_MODEL,
        "total_questions": total,
        "correct":         n_correct,
        "correct_pct":     round(n_correct / total * 100, 1),
        "incorrect":       n_incorrect,
        "incorrect_pct":   round(n_incorrect / total * 100, 1),
        "avg_latency_s":   avg_lat,
        "missed_ids":      missed_ids,
    }
elif args.pipeline == "raw":
    evaluation = {
        "pipeline":        "raw",
        "model":           LLM_MODEL,
        "total_questions": total,
        "avg_latency_s":   avg_lat,
    }
else:
    evaluation = {
        "pipeline":         args.pipeline,
        "total_questions":  total,
        "full_hits":        n_correct,
        "full_hits_pct":    round(n_correct / total * 100, 1),
        "partial_hits":     n_partial,
        "partial_hits_pct": round(n_partial / total * 100, 1),
        "misses":           n_incorrect,
        "misses_pct":       round(n_incorrect / total * 100, 1),
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
if MEDQA_MODE:
    print(f"Correct      : {n_correct}  ({evaluation['correct_pct']}%)")
    print(f"Incorrect    : {n_incorrect}  ({evaluation['incorrect_pct']}%)")
elif args.pipeline != "raw":
    print(f"Full hits    : {n_correct}  ({evaluation['full_hits_pct']}%)")
    print(f"Partial hits : {n_partial}  ({evaluation['partial_hits_pct']}%)")
    print(f"Misses       : {n_incorrect}  ({evaluation['misses_pct']}%)")
print(f"Avg latency  : {avg_lat:.2f}s")
print("=" * 60)
