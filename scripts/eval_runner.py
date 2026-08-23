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
    rag_lower = rag_answer.lower()
    refused = REFUSED in rag_lower

    if expected_answer is None and category == "out-of-scope":
        return refused

    if expected_answer is None and category == "ambiguous":
        return True

    if refused:
        return False

    if not expected_answer:
        return True

    expected_lower = expected_answer.lower()
    keywords = [w for w in expected_lower.split() if len(w) > 4]
    if not keywords:
        return True
    
    matches = sum(1 for kw in keywords if kw in rag_lower)
    return matches >= min(3, len(keywords))

def check_retrieval(expected_source, expected_pages, retrieved_chunks) -> bool:
    if not expected_source or not expected_pages:
        return True # Not applicable (e.g. out of scope)
    
    if not isinstance(expected_pages, list):
        expected_pages = [expected_pages]
        
    for chunk in retrieved_chunks:
        if chunk.get("source") == expected_source and chunk.get("page") in expected_pages:
            return True
            
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Eval Runner")
    parser.add_argument(
        "--questions",
        default="data/questions.json",
        help="Path to questions JSON file (default: data/questions.json)"
    )
    parser.add_argument(
        "--output",
        default="data/eval_results.json",
        help="Path to save eval results (default: data/eval_results.json)"
    )
    args = parser.parse_args()

    print(f"Loading questions from: {args.questions}")
    with open(args.questions) as f:
        questions = json.load(f)


    try:
        index, chunks = load_index_and_chunks()
        bm25 = build_bm25(chunks)
        print("Hybrid retrieval (FAISS + BM25) ready.\n")
    except Exception as e:
        print(f"Error: {e}")
        return

    questions = [q for q in questions if "question" in q]

    print(f"Running evaluation on {len(questions)} questions...\n")

    results_output = []
    correct_total = 0
    wrong_total = 0
    retrieval_hits = 0
    retrieval_misses = 0
    
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
        
        # Check if it was a retrieval hit
        is_retrieval_hit = check_retrieval(q.get("source"), q.get("page"), retrieved_chunks)
        if q.get("source") and q.get("page"):
            if is_retrieval_hit:
                retrieval_hits += 1
            else:
                retrieval_misses += 1

        mark = "✅" if is_correct else "❌"
        hit_mark = "✅" if is_retrieval_hit else "❌"

        cat = q["category"]
        if cat not in category_scores:
            category_scores[cat] = {"correct": 0, "wrong": 0, "rhits": 0, "rmisses": 0}
            
        if is_correct:
            correct_total += 1
            category_scores[cat]["correct"] += 1
        else:
            wrong_total += 1
            category_scores[cat]["wrong"] += 1
            
        if q.get("source") and q.get("page"):
            if is_retrieval_hit:
                category_scores[cat]["rhits"] += 1
            else:
                category_scores[cat]["rmisses"] += 1

        print(f"  Category: {cat}  Correct: {mark}  Retrieval Hit: {hit_mark}")
        print(f"  A: {answer[:120]}...")
        if q.get("expected_answer"):
            print(f"  Expected: {q['expected_answer'][:120]}...")
        if not is_correct:
            reason = "Generation error (retrieval was ok)" if is_retrieval_hit else "Retrieval miss"
            print(f"  Fail Reason: {reason}")
            
        print()

        results_output.append({
            "question": q["question"],
            "expected_answer": q.get("expected_answer"),
            "rag_answer": answer,
            "category": cat,
            "correct": is_correct,
            "retrieval_hit": is_retrieval_hit,
            "sources": [{"source": c["source"], "page": c["page"]} for c in retrieved_chunks],
            "fail_reason": None if is_correct else ("Generation error" if is_retrieval_hit else "Retrieval miss")
        })

        with open(args.output, "w") as f:
            json.dump(results_output, f, indent=2)

    total = correct_total + wrong_total
    pct = round(100 * correct_total / total) if total else 0
    rtotal = retrieval_hits + retrieval_misses
    rpct = round(100 * retrieval_hits / rtotal) if rtotal else 0

    print("=" * 50)
    print("=== SCORECARD ===")
    print(f"  🎯 Generation Correct Score: {pct}% ({correct_total}/{total})")
    print(f"  🔍 Retrieval Hit Rate:       {rpct}% ({retrieval_hits}/{rtotal})")
    print("=" * 50)
    print("\nBy category:")
    for cat, s in sorted(category_scores.items()):
        cat_total = s["correct"] + s["wrong"]
        rcat_total = s["rhits"] + s["rmisses"]
        cat_rpct = round(100*s["rhits"]/rcat_total) if rcat_total else "N/A "
        print(f"  {cat:20s}  Ans: ✅ {s['correct']}/{cat_total}  | Retr: ✅ {s['rhits']}/{rcat_total} ({cat_rpct}%)")
        
    print("\nFailed Questions Breakdown:")
    for r in results_output:
        if not r["correct"]:
            print(f" - [{r['category']}] {r['question']}\n   Reason: {r['fail_reason']}")

    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
