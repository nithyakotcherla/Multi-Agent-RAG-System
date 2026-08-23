# RAG Regulators — Failure Log

Every failure found during eval runs: what broke, why, what fixed it.  
Run `python scripts/retrieval_checker.py "<question>"` to diagnose any new one.

---
## Pipeline changes applied (pre-eval_results2.json)

### top-k increased: 5 → 8

| Bug | Root cause | Fix |
|---|---|---|
| Cross-reference questions only retrieved one of the two required pages | With k=5 the second relevant chunk (from a different section or document) was consistently outside the top results, so the LLM had an incomplete picture | Increased `DEFAULT_TOP_K` from 5 to 8 in `config.py`; both pages now regularly appear in the context window for cross-document queries |

### Keyword-only search returned no results

| Bug | Root cause | Fix |
|---|---|---|
| Exact-term queries (article numbers, acronyms, short phrases) returned "no results found" with the original keyword retriever | Pure keyword search relied on exact string matching in the chunk index; short or technical terms that didn't appear verbatim in chunk text caused a complete miss with an empty result set | Replaced keyword retriever with **BM25** (`rank_bm25` library); BM25 handles partial matches and term frequency weighting, eliminating empty-result failures while still rewarding exact term presence |

### System prompt loosened

| Change | Reason |
|---|---|
| Removed strict "only answer if you find a direct quote" constraint from SYSTEM_PROMPT | The original tight prompt caused the LLM to refuse questions where the relevant information was paraphrased across multiple chunks rather than appearing as a verbatim sentence; loosening the instruction to "answer based on the context provided, even if the exact phrasing differs" reduced false refusals on cross-reference and stress-test questions without introducing hallucinations |

---


## Eval run history

| Report | Score | Questions file | Notes |
|---|---|---|---|
| `eval_results.json` | 20/20 (100%) | `questions.json` | First full passing run; 5 retrieval misses that still scored correct |
| `eval_results2.json` | 44/48 (92%) | `questions2.json` | 4 failures: 2 retrieval misses, 1 generation error, 1 scorer bug on unanswerable stress-test |

---

## Active failures (eval_results2.json — 44/48)

| Question | Category | Retrieval OK? | Problem | Root cause |
|---|---|:---:|---|---|
| "What are the trustworthiness characteristics that the NIST AI RMF describes?" | factual | ❌ | LLM listed only 2 of 7 characteristics (Secure/Resilient + Privacy-Enhanced) | p.8 has the full list but ranked below p.44, p.5, p.17 in both FAISS and BM25; wrong chunks flooded the context |
| "What are all the specific situations in which real-time remote biometric identification in public spaces is permitted for law enforcement under the EU AI Act?" | stress-test (table_list) | ❌ | LLM gave an expanded, partially wrong list instead of the three narrow situations | Retriever pulled Article text from p.52 (enforcement rules) instead of Recital 33 on p.9 which describes the three permitted scenarios in narrative form; the article text on p.52 lists sub-cases of each scenario making the LLM over-count |
| "Can you tell me what the four main functions in the NIST AI RMF are?" | factual | ✅ | Retrieval correct (p.25), but answer flagged as FAIL | LLM prefixed its answer with "YES." before listing the functions; keyword scorer in `run_eval2.py` matched against "YES." and failed the minimum-3-keyword check |
| "How does the NIST AI RMF define acceptable risk thresholds in numerical terms, and what percentage of AI incidents require escalation?" | stress-test (on_topic_unanswerable) | ✅ | System correctly refused ("I don't have enough information"), but scored as FAIL | `score_answer()` only handles `expected_answer is None` for `"out-of-scope"` and `"ambiguous"` categories; for `"stress-test"` it falls through to `if refused: return False` — refusals are penalised regardless of intent |

---

## Retrieval misses that still passed (correct answer despite wrong chunk)

These are fragile — the answer was correct by accident. Track them as risk items.

| Question | Expected page | Pages actually retrieved | Risk |
|---|---|---|---|
| "How much can a company be fined under the EU AI Act for using a prohibited AI practice?" | p.116 | p.115, p.117, p.117 | p.115 has a near-identical sentence; correct today but may regress on index rebuild |
| "What exactly counts as valid consent from a user under the GDPR?" | p.34 | p.37, p.37, p.6 | Consent definition is split across p.6 (recital) and p.37 (article); p.34 was never retrieved |
| "Do the EU AI Act and the NIST AI RMF say similar things about keeping humans in the loop?" | EU AI ACT p.61 | p.8, p.41, p.25 | Article 14 (p.61) never retrieved; answer assembled from weaker recital passages |
| "How important is transparency in both the EU AI Act and the NIST AI RMF?" | NIST p.21–22 | p.20, p.8, p.47 | Core transparency definition page not retrieved; answer was constructed from adjacent pages |
| "What information do you need to give someone when collecting their personal data?" | GDPR p.40 | p.7, p.12, p.42 | Answer correct but cited the indirect-collection article (p.42) rather than the primary direct-collection article |
| "What data about individuals is NOT considered personal data under GDPR?" | GDPR p.33 | p.5, p.11, p.38 | Anonymous data definition on p.5 was retrieved but p.33 (core definition) was not |
| "A company runs an AI-based credit scoring system … does the EU AI Act also impose additional requirements?" | GDPR p.47 + EU AI ACT p.53 | EU AI ACT p.16, p.127, p.54 | GDPR Article 22 (p.47) on automated decision-making never retrieved; cross-regulation hop incomplete |
| "What's the deal with companies that collect a ton of data on people — do they have to hire someone specific?" | GDPR p.55 | p.18, p.11, p.29 | DPO article (p.55) not retrieved; answer assembled from recital on p.18 |
| "What is GDPR, when does it apply, and what rights does it give individuals?" | GDPR p.33–34 | p.1, p.14, p.12 | Definitional pages not retrieved; rights pages were retrieved so answer was still correct |

---

## Resolved failures (eval_results.json — 20/20)

All 20 questions in the original `questions.json` passed. No failures to document.

Notable retrieval misses that happened to still pass (same fragile patterns as above, now also tracked in the table above for `questions2.json`):

| Question | Miss | Notes |
|---|---|---|
| EU AI Act fine (p.116) | Retrieved p.115 | Near-identical sentence on adjacent page saved the answer |
| NIST trustworthiness (p.8) | Retrieved p.44 first | Correct answer assembled from secondary pages |
| GDPR consent (p.34) | Retrieved p.37 | Recital + article together gave sufficient signal |
| Human oversight cross-ref | EU AI ACT p.61 never retrieved | Answer built from p.8 recital; passed keyword check |
| Transparency cross-ref | NIST p.21–22 not retrieved | p.47 and p.20 gave enough overlap to pass |

---

## Pipeline changes applied (pre-eval_results2.json)

### top-k increased: 5 → 8

| Bug | Root cause | Fix |
|---|---|---|
| Cross-reference questions only retrieved one of the two required pages | With k=5 the second relevant chunk (from a different section or document) was consistently outside the top results, so the LLM had an incomplete picture | Increased `DEFAULT_TOP_K` from 5 to 8 in `config.py`; both pages now regularly appear in the context window for cross-document queries |

### Keyword-only search returned no results

| Bug | Root cause | Fix |
|---|---|---|
| Exact-term queries (article numbers, acronyms, short phrases) returned "no results found" with the original keyword retriever | Pure keyword search relied on exact string matching in the chunk index; short or technical terms that didn't appear verbatim in chunk text caused a complete miss with an empty result set | Replaced keyword retriever with **BM25** (`rank_bm25` library); BM25 handles partial matches and term frequency weighting, eliminating empty-result failures while still rewarding exact term presence |

### System prompt loosened

| Change | Reason |
|---|---|
| Removed strict "only answer if you find a direct quote" constraint from SYSTEM_PROMPT | The original tight prompt caused the LLM to refuse questions where the relevant information was paraphrased across multiple chunks rather than appearing as a verbatim sentence; loosening the instruction to "answer based on the context provided, even if the exact phrasing differs" reduced false refusals on cross-reference and stress-test questions without introducing hallucinations |

---

## Known issues / pending fixes

| Issue | Severity | Status |
|---|---|---|
| `score_answer()` penalises correct refusals for `stress-test` category when `expected_answer is None` | Medium | Open — fix: add `if expected_answer is None and "unanswerable" in sub_type: return refused` check |
| NIST trustworthiness list (p.8) consistently ranked below p.44/p.47 | Medium | Open — p.8 chunk may be too short; consider chunk overlap increase or manual boost for this page |
| Biometric ID permitted situations: p.52 outscores p.9 for list-style queries | Medium | Open — Recital text on p.9 uses narrative style while article on p.52 uses numbered lists; BM25 favours the numbered list |
| Generation error on yes/no prefix ("YES. The four…") | Low | Open — add prompt instruction: "Do not start your answer with YES or NO; begin directly with the information" |
