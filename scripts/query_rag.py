import json
import os
import re
import shutil
import time
from typing import Any
import numpy as np
import faiss
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from scripts.cost_tracker import track_cost

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Index loading ─────────────────────────────────────────────────────────────

def load_index_and_chunks():
    if not os.path.exists("data/my_index.faiss") or not os.path.exists("data/chunks.json"):
        raise FileNotFoundError("Index or chunks file not found. Run index_data.py first.")

    index = faiss.read_index("data/my_index.faiss")
    with open("data/chunks.json", "r") as f:
        chunks = json.load(f)
    return index, chunks


def build_bm25(chunks):
    """Build a BM25 index from the chunk texts (done once at startup, free)."""
    tokenized = [re.findall(r"\w+", c["text"].lower()) for c in chunks]
    return BM25Okapi(tokenized)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def get_embeddings(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    track_cost(response, is_embedding=True)
    return [item.embedding for item in response.data]


def _faiss_retrieve(query, index, chunks, k, query_vec=None):
    """Vector search via FAISS. Accepts a pre-computed query_vec to avoid re-embedding."""
    if query_vec is None:
        query_vec = np.array(get_embeddings([query])).astype("float32")
    else:
        query_vec = np.array(query_vec).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k)
    ranked = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx == -1:
            continue
        ranked.append((int(idx), float(score), rank))
    return ranked                          # list of (chunk_idx, score, rank)


def _bm25_retrieve(query, bm25, k):
    """Keyword search via BM25."""
    tokens = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(tokens)
    top_k = np.argsort(scores)[::-1][:k]
    return [(int(idx), float(scores[idx]), rank)
            for rank, idx in enumerate(top_k)]


def _reciprocal_rank_fusion(faiss_ranked, bm25_ranked, k=60):
    """
    Merge two ranked lists using Reciprocal Rank Fusion (RRF).
    RRF score = Σ 1 / (k + rank)   — higher is better.
    """
    rrf_scores = {}
    for idx, _score, rank in faiss_ranked:
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for idx, _score, rank in bm25_ranked:
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def _rerank_chunks_llm(query: str, chunks: list, top_n: int) -> list:
    """
    Re-rank retrieved chunks using GPT (listwise approach).

    Sends all candidates to GPT with numbered labels; GPT returns a
    comma-separated list of indices ordered most → least relevant.
    Costs ~1–2k tokens on gpt-4o-mini per query.
    Falls back to original order if the API call fails.
    """
    numbered_passages = "\n\n".join(
        f"[{i}] {chunk['text'][:400]}"
        for i, chunk in enumerate(chunks)
    )
    messages: list[Any] = [
        {
            "role": "system",
            "content": (
                "You are a passage relevance ranker. "
                "Given a question and a list of numbered passages, output only a "
                "comma-separated list of passage indices ordered from most to least "
                "relevant. Output nothing else — no explanation, no labels."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Passages:\n{numbered_passages}\n\n"
                f"Return the {top_n} most relevant passage indices, "
                f"comma-separated, most relevant first."
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=200,
        )
        track_cost(response)
        raw = response.choices[0].message.content or ""
    except (RateLimitError, APIError) as e:
        print(f"LLM re-rank failed ({e}); returning original order.")
        return chunks[:top_n]

    # Parse "2, 0, 5, 1, 3" → [2, 0, 5, 1, 3]
    indices: list[int] = []
    for part in re.split(r"[,\s]+", raw.strip()):
        try:
            idx = int(part)
            if 0 <= idx < len(chunks) and idx not in indices:
                indices.append(idx)
        except ValueError:
            continue

    # If GPT returned fewer than top_n, pad with remaining chunks in original order
    if len(indices) < top_n:
        for i in range(len(chunks)):
            if i not in indices:
                indices.append(i)
            if len(indices) == top_n:
                break

    reranked = []
    for idx in indices[:top_n]:
        reranked.append(dict(chunks[idx]))  # score stays as FAISS cosine similarity

    return reranked


def retrieve(query, index, chunks, bm25=None, k=8, rerank=True, query_vec=None,
             return_candidates=False):
    """
    Hybrid retrieval: FAISS (semantic) + BM25 (keyword) merged with RRF.
    Falls back to FAISS-only if bm25 is not provided.

    If rerank=True, fetches 2*k candidates first, then re-ranks with GPT
    (listwise) and returns the top k.

    query_vec: optional pre-computed embedding (numpy array) to skip re-embedding.

    return_candidates: if True, returns (results, candidates) where candidates is
    the full pre-rerank pool (up to 2*k chunks).  Useful for PDF page discovery
    when the reranked top-k collapses to very few unique pages.
    """
    fetch_k = k * 2 if rerank else k

    faiss_ranked = _faiss_retrieve(query, index, chunks, fetch_k, query_vec=query_vec)

    if bm25 is not None:
        bm25_ranked = _bm25_retrieve(query, bm25, fetch_k)
        merged = _reciprocal_rank_fusion(faiss_ranked, bm25_ranked)
        top_indices = [idx for idx, _ in merged[:fetch_k]]
    else:
        top_indices = [idx for idx, _, _ in faiss_ranked]

    results = []
    seen = set()
    for idx in top_indices:
        if idx in seen:
            continue
        seen.add(idx)
        result = dict(chunks[idx])
        result["score"] = next(
            (s for i, s, _ in faiss_ranked if i == idx), 0.0
        )
        results.append(result)

    candidates = results[:]   # pre-rerank pool (up to fetch_k entries)

    if rerank and results:
        results = _rerank_chunks_llm(query, results, top_n=k)

    if return_candidates:
        return results, candidates
    return results

# ── Dynamic Prompt Selection ────────────────────────────────────────────────

def detect_question_type(query: str) -> str:
    """
    Classifies the question into one of five types so the right
    system prompt can be selected for generation.

    Types: yes_no | factual | comparison | listing | open_ended
    """
    q = query.lower().strip()

    # Yes/No questions
    yes_no_starters = ("is ", "are ", "can ", "does ", "do ", "should ", "was ", "were ",
                       "has ", "have ", "will ", "would ", "must ", "did ")
    if q.startswith(yes_no_starters) or q.startswith(("is it", "are there")):
        return "yes_no"

    # Comparison questions
    comparison_keywords = ("compare", "difference", "differ", "vs", "versus",
                           "contrast", "similar", "both", "between")
    if any(kw in q for kw in comparison_keywords):
        return "comparison"

    # Listing questions  
    listing_keywords = ("what are the", "list", "enumerate", "name the",
                        "which are", "what are all", "characteristics",
                        "requirements", "steps", "functions", "principles")
    if any(kw in q for kw in listing_keywords):
        return "listing"

    # Factual / specific number or definition questions
    factual_keywords = ("how much", "how many", "how quickly", "how long",
                        "when", "who", "define", "definition", "what is",
                        "what does", "how do you define", "exactly")
    if any(kw in q for kw in factual_keywords):
        return "factual"

    return "open_ended"


# Prompt templates per question type
_PROMPTS = {
    "yes_no": (
        "You are a precise regulatory assistant. "
        "Start your answer with YES or NO on the very first word. "
        "Then give a single sentence explaining why, citing the source and page. "
        "Only say 'I don't have enough information to answer this' if the context "
        "contains absolutely no relevant information."
    ),
    "factual": (
        "You are a precise regulatory assistant. "
        "Give the exact figure, date, or definition asked for — do not paraphrase it away. "
        "Always cite the source document and page number. "
        "Keep your answer to 1–2 sentences. "
        "Only say 'I don't have enough information to answer this' if the context "
        "contains absolutely no relevant information."
    ),
    "comparison": (
        "You are a helpful regulatory assistant. "
        "Structure your answer clearly: first describe what Framework/Law A says, "
        "then what Framework/Law B says, then give a one-sentence synthesis. "
        "Cite sources and page numbers for each point. "
        "Only say 'I don't have enough information to answer this' if the context "
        "contains absolutely no relevant information."
    ),
    "listing": (
        "You are a precise regulatory assistant. "
        "Return the complete list — do not omit any items mentioned in the context. "
        "Format as a numbered or bullet list. "
        "After the list, add one sentence citing the source and page. "
        "Only say 'I don't have enough information to answer this' if the context "
        "contains absolutely no relevant information."
    ),
    "open_ended": (
        "You are a helpful regulatory assistant. Answer the question using "
        "the provided context passages. The answer may be spread across "
        "multiple passages — synthesize them into a complete answer. "
        "Always cite which source and page your answer comes from. "
        "Only say 'I don't have enough information to answer this' if the context "
        "contains absolutely no relevant information. "
        "Keep your answer concise: 2–3 sentences maximum."
    ),
}

# ── Generation ────────────────────────────────────────────────────────────────

def generate_answer(query, retrieved_chunks, max_retries=3):
    context = "\n\n".join([
        f"[Source: {chunk['source']} p.{chunk['page']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])

    # ── Dynamic prompt: pick system instructions based on question type ──
    q_type = detect_question_type(query)
    system_prompt = _PROMPTS[q_type]
    print(f"[Dynamic Prompt] Question type detected: '{q_type}'")

    messages: list[Any] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                max_tokens=600
            )
            track_cost(response)
            return response.choices[0].message.content or ""
        except RateLimitError:
            print(f"Rate limited — retrying in {2 ** attempt}s...")
            time.sleep(2 ** attempt)
        except APIError as e:
            print(f"API error: {e} — retrying in 1s...")
            time.sleep(1)

    return "Sorry, I encountered an error generating the answer. Please try again."


# ── Side-by-side comparison ───────────────────────────────────────────────────

def _parse_setting(s: str) -> dict:
    """
    Parse a setting string like 'k=5' or 'rerank=false' into a dict.
    Supported keys: k (int), rerank (bool).
    """
    key, _, val = s.partition("=")
    key = key.strip().lower()
    val = val.strip().lower()
    if key == "k":
        return {"k": int(val)}
    if key == "rerank":
        return {"rerank": val not in ("false", "0", "no")}
    raise ValueError(f"Unknown setting '{key}'. Supported: k, rerank")


def _wrap(text: str, width: int) -> list[str]:
    """Hard-wrap text to `width` characters."""
    lines = []
    for raw_line in text.splitlines():
        while len(raw_line) > width:
            lines.append(raw_line[:width])
            raw_line = raw_line[width:]
        lines.append(raw_line)
    return lines


def _side_by_side(cols: list[list[str]], col_width: int) -> None:
    """Print N columns of text side by side."""
    sep = " | "
    max_rows = max(len(c) for c in cols)
    for i in range(max_rows):
        parts = []
        for j, col in enumerate(cols):
            cell = col[i] if i < len(col) else ""
            # last column: no padding needed
            parts.append(f"{cell:<{col_width}}" if j < len(cols) - 1 else cell)
        print(sep.join(parts))


def run_comparison(query: str, index, chunks: list, bm25, settings_list: list[dict]) -> None:
    """
    Run `retrieve` + `generate_answer` for each settings dict and print
    all results side by side.

    Example settings_list:
        [{"k": 3}, {"k": 5}]
        [{"k": 8, "rerank": False}, {"k": 8, "rerank": True}]
    """
    term_width = shutil.get_terminal_size(fallback=(120, 40)).columns
    n = len(settings_list)
    # Each column gets an equal share minus separators (3 chars each)
    col_width = max(30, (term_width - 3 * (n - 1)) // n)

    label_lines: list[list[str]] = []
    chunk_lines: list[list[str]] = []
    answer_lines: list[list[str]] = []

    for cfg in settings_list:
        k = cfg.get("k", 8)
        rerank = cfg.get("rerank", True)
        label = f"k={k}  rerank={rerank}"

        print(f"  Running [{label}] ...", flush=True)
        results = retrieve(query, index, chunks, bm25=bm25, k=k, rerank=rerank)
        answer = generate_answer(query, results)

        # ── label column ─────────────────────────────────────────────────────
        header = ["─" * col_width, label, "─" * col_width]
        label_lines.append(header)

        # ── chunks column ────────────────────────────────────────────────────
        col: list[str] = []
        for rank, r in enumerate(results, 1):
            col += _wrap(f"#{rank} {r['source']} p.{r['page']} [{r['score']:.3f}]", col_width)
            col += _wrap(f"    {r['text'][:120]}...", col_width)
            col.append("")
        chunk_lines.append(col)

        # ── answer column ─────────────────────────────────────────────────────
        answer_lines.append(_wrap(answer, col_width))

    print(f"\nQuery: {query}\n")

    print("=== RETRIEVED CHUNKS ===")
    _side_by_side(label_lines, col_width)
    print()
    _side_by_side(chunk_lines, col_width)

    print("\n=== ANSWERS ===")
    _side_by_side(label_lines, col_width)
    print()
    _side_by_side(answer_lines, col_width)
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import sys

    args = sys.argv[1:]

    # --compare mode: python query_rag.py "question" --compare k=3 k=8
    if "--compare" in args:
        compare_idx = args.index("--compare")
        query = args[0] if args[0] != "--compare" else None
        setting_strs = args[compare_idx + 1:]

        if not query or not setting_strs:
            print('Usage: python scripts/query_rag.py "question" --compare k=3 k=8')
            print('       python scripts/query_rag.py "question" --compare rerank=false rerank=true')
            return

        try:
            settings_list = [_parse_setting(s) for s in setting_strs]
        except ValueError as e:
            print(f"Error: {e}")
            return

        try:
            index, chunks = load_index_and_chunks()
        except Exception as e:
            print(f"Error: {e}")
            return

        bm25 = build_bm25(chunks)
        run_comparison(query, index, chunks, bm25, settings_list)
        return

    # ── normal single-query mode ──────────────────────────────────────────────
    if not args:
        print('Usage: python scripts/query_rag.py "your question here"')
        print('       python scripts/query_rag.py "question" --compare k=3 k=8')
        return

    query = args[0]

    try:
        index, chunks = load_index_and_chunks()
    except Exception as e:
        print(f"Error: {e}")
        return

    bm25 = build_bm25(chunks)

    print(f"\nQuery: {query}")
    print("Retrieving (hybrid FAISS + BM25 → LLM re-rank)...")
    results = retrieve(query, index, chunks, bm25=bm25, k=8, rerank=True)

    print("Generating answer...")
    answer = generate_answer(query, results)

    print("\n--- ANSWER ---")
    print(answer)
    print("\n--- SOURCES ---")
    for r in results:
        print(f"- {r['source']} (Page {r['page']}) [Score: {r['score']:.3f}]")


if __name__ == "__main__":
    main()
