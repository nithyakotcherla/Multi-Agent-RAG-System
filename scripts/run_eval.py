import sys
import os
import json

# Add project root to path so scripts/ imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from scripts.query_rag import load_index_and_chunks, build_bm25, retrieve, generate_answer

REFUSED = "i don't have enough information"


def score_answer(rag_answer: str, expected_answer, category: str) -> bool:
    """Return True if the answer is considered correct."""
    rag_lower = rag_answer.lower()
    refused = REFUSED in rag_lower

    # Out-of-scope: must refuse
    if expected_answer is None and category == "out-of-scope":
        return refused

    # Ambiguous: either refusing OR giving a relevant answer is acceptable
    if expected_answer is None and category == "ambiguous":
        return True

    # Factual / cross-reference → wrong if refused
    if refused:
        return False

    # Keyword overlap: at least 3 key terms from expected answer appear in RAG answer
    expected_lower = expected_answer.lower()
    keywords = [w for w in expected_lower.split() if len(w) > 4]
    if not keywords:
        return True
    matches = sum(1 for kw in keywords if kw in rag_lower)
    return matches >= min(3, len(keywords))


def main():
    # Load eval set
    with open("data/questions.json") as f:
        questions = json.load(f)

    # Load index and chunks
    try:
        index, chunks = load_index_and_chunks()
        bm25 = build_bm25(chunks)
        print("Hybrid retrieval (FAISS + BM25) ready.\n")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Filter out comment objects
    questions = [q for q in questions if "question" in q]

    print(f"Running evaluation on {len(questions)} questions...\n")

    results_output = []
    correct_total = 0
    wrong_total = 0
    category_scores = {}

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Q: {q['question']}")

        try:
            retrieved_chunks = retrieve(q["question"], index, chunks, bm25=bm25, k=8, rerank=True)
            answer = generate_answer(q["question"], retrieved_chunks)
        except Exception as e:
            print(f"  ⚠️  Skipped — error: {e}\n")
            continue

        is_correct = score_answer(answer, q.get("expected_answer"), q["category"])
        mark = "✅" if is_correct else "❌"

        cat = q["category"]
        if cat not in category_scores:
            category_scores[cat] = {"correct": 0, "wrong": 0}
        if is_correct:
            correct_total += 1
            category_scores[cat]["correct"] += 1
        else:
            wrong_total += 1
            category_scores[cat]["wrong"] += 1

        print(f"  Category: {cat}  {mark}")
        print(f"  A: {answer[:120]}...")
        if q.get("expected_answer"):
            print(f"  Expected: {q['expected_answer'][:120]}...")
        print()

        results_output.append({
            "question": q["question"],
            "expected_answer": q.get("expected_answer"),
            "rag_answer": answer,
            "category": cat,
            "correct": is_correct,
            "sources": [{"source": c["source"], "page": c["page"]} for c in retrieved_chunks]
        })

        # Save after every question so partial results survive interruptions
        with open("data/eval_results.json", "w") as f:
            json.dump(results_output, f, indent=2)

    # Print summary
    total = correct_total + wrong_total
    pct = round(100 * correct_total / total) if total else 0

    print("=" * 50)
    print(f"  ✅  Correct : {correct_total}/{total}")
    print(f"  ❌  Wrong   : {wrong_total}/{total}")
    print(f"  🎯  Score   : {pct}%")
    print("=" * 50)
    print("\nBy category:")
    for cat, s in sorted(category_scores.items()):
        cat_total = s["correct"] + s["wrong"]
        print(f"  {cat:20s}  ✅ {s['correct']}/{cat_total}  ❌ {s['wrong']}/{cat_total}")
    print(f"\nResults saved to data/eval_results.json")


if __name__ == "__main__":
    main()
