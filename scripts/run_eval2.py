import sys
import os
import json

# Add project root to path so scripts/ imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from scripts.query_rag import load_index_and_chunks, build_bm25, retrieve, generate_answer

REFUSED = "i don't have enough information"

# Define 15 new evaluation questions
NEW_QUESTIONS = [
    {
        "question": "What is the 'Right to be Forgotten' under the GDPR?",
        "expected_answer": "The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or her without undue delay.",
        "category": "factual"
    },
    {
        "question": "Under the GDPR, can I process personal data just because I have a legitimate interest?",
        "expected_answer": "Yes, if processing is necessary for the purposes of the legitimate interests pursued by the controller or by a third party, except where such interests are overridden by the interests or fundamental rights and freedoms of the data subject.",
        "category": "factual"
    },
    {
        "question": "Are deepfakes regulated under the EU AI Act?",
        "expected_answer": "Yes, users of an AI system that generates or manipulates image, audio or video content that appreciably resembles existing persons ('deep fake') shall disclose that the content has been artificially generated or manipulated.",
        "category": "factual"
    },
    {
        "question": "What are the rules regarding the AI literacy obligation in the EU AI Act?",
        "expected_answer": "Providers and deployers of AI systems shall take measures to ensure, to their best extent, a sufficient level of AI literacy of their staff and other persons dealing with the operation and use of AI systems.",
        "category": "factual"
    },
    {
        "question": "What is the 'MAP' function in the NIST AI RMF?",
        "expected_answer": "The MAP function establishes the context to frame risks related to an AI system. The AI lifecycle is described and risks are identified and documented.",
        "category": "factual"
    },
    {
        "question": "How heavily do the EU AI Act and GDPR rely on third-party audits?",
        "expected_answer": "The EU AI Act mandates third-party conformity assessments for certain high-risk AI systems. GDPR occasionally encourages independent audits or certifications but relies more heavily on internal Data Protection Impact Assessments (DPIAs) and supervisory authorities.",
        "category": "cross-reference"
    },
    {
        "question": "Does either the GDPR or the EU AI Act restrict automated decision making?",
        "expected_answer": "Yes, GDPR Article 22 regulates decisions based solely on automated processing. The EU AI Act specifically regulates AI systems used to make these kinds of automated decisions, classifying many of them as high-risk.",
        "category": "cross-reference"
    },
    {
        "question": "How do the GDPR and NIST approach the concept of fairness?",
        "expected_answer": "GDPR mandates that personal data be processed lawfully, fairly and in a transparent manner. NIST AI RMF treats fairness as a trustworthiness characteristic primarily concerned with managing harmful bias and preventing discriminatory outcomes.",
        "category": "cross-reference"
    },
    {
        "question": "Do the NIST AI framework and the EU AI Act both prohibit certain AI systems?",
        "expected_answer": "No. The EU AI Act explicitly prohibits certain AI practices (like social scoring). The NIST AI framework is purely voluntary and does not ban any systems; it only provides guidelines for managing risk.",
        "category": "cross-reference"
    },
    {
        "question": "Can anonymized data be regulated by both GDPR and the EU AI Act?",
        "expected_answer": "GDPR does not apply to truly anonymized data. However, the EU AI Act regulates the AI systems themselves, meaning an AI system trained on or processing strictly anonymized data is still regulated under the AI Act based on its risk classification.",
        "category": "cross-reference"
    },
    {
        "question": "Can you tell me how to bake a chocolate cake?",
        "expected_answer": None,
        "category": "out-of-scope"
    },
    {
        "question": "What are the core components of the HIPAA privacy rule in the USA?",
        "expected_answer": None,
        "category": "out-of-scope"
    },
    {
        "question": "Write me a short python script to download a YouTube video.",
        "expected_answer": None,
        "category": "out-of-scope"
    },
    {
        "question": "What happens if someone complains?",
        "expected_answer": None,
        "category": "ambiguous"
    },
    {
        "question": "Who needs to read the documentation?",
        "expected_answer": None,
        "category": "ambiguous"
    }
]


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
    # We will use the hardcoded new questions
    questions = NEW_QUESTIONS

    # Load index and chunks
    try:
        index, chunks = load_index_and_chunks()
        bm25 = build_bm25(chunks)
        print("Hybrid retrieval (FAISS + BM25) ready.\n")
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Running new evaluation on {len(questions)} questions...\n")

    results_output = []
    correct_total = 0
    wrong_total = 0
    category_scores = {}

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Q: {q['question']}")

        retrieved_chunks = retrieve(q["question"], index, chunks, bm25=bm25, k=8)
        answer = generate_answer(q["question"], retrieved_chunks)

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

    # Save results to a new target
    with open("data/eval_results2.json", "w") as f:
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
    print(f"\nResults saved to data/eval_results2.json")


if __name__ == "__main__":
    main()
