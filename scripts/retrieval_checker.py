import sys
import os
import json

# Add project root to path so scripts/ imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from scripts.query_rag import load_index_and_chunks, build_bm25, retrieve

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/retrieval_checker.py \"your question string\"")
        return

    query = sys.argv[1]

    # Load questions to find expected source/page
    with open("data/questions.json") as f:
        questions = json.load(f)

    expected_pages = []
    expected_sources = []
    for q in questions:
        if q.get("question") == query:
            src = q.get("source")
            pm = q.get("page")
            if src:
                expected_sources.append(src)
            if pm:
                if isinstance(pm, list):
                    expected_pages.extend(pm)
                else:
                    expected_pages.append(pm)

    try:
        index, chunks = load_index_and_chunks()
        bm25 = build_bm25(chunks)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    print(f"\nEvaluating query: \"{query}\"")
    k = 8
    # Run retrieval
    results = retrieve(query, index, chunks, bm25=bm25, k=k, rerank=True)

    print(f"\n--- Top {k} Retrieved Chunks ---")
    hit = False
    for i, r in enumerate(results):
        score = r.get("score", 0.0)
        source = r.get("source", "Unknown")
        page = r.get("page", 0)
        text = r.get("text", "").replace('\n', ' ')
        
        is_expected = False
        if expected_sources and expected_pages:
            if source in expected_sources and page in expected_pages:
                is_expected = True
                hit = True

        mark = "⭐ (Expected)" if is_expected else ""
        print(f"{i+1}. [Score: {score:.4f}] {source} p.{page} {mark}")
        print(f"   {text[:200]}...\n")

    print("\n--- Retrieval Evaluation ---")
    if expected_sources and expected_pages:
        if hit:
            print("✅ Expected chunk(s) retrieved.")
        else:
            print("❌ Expected chunk(s) missed.")
            print(f"   Expected sources: {expected_sources}, Expected pages: {expected_pages}")
    else:
        print("ℹ️ No expected chunks defined for this question in data/questions.json (or question not found).")
    print()

if __name__ == "__main__":
    main()
