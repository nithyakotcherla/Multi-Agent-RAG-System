import os
from openai import AsyncOpenAI
import sys

# Ensure scripts can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.query_rag import load_index_and_chunks, build_bm25, retrieve, generate_answer

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def agent_1_internal_researcher(query: str, index=None, chunks=None, bm25=None, k: int = 8):
    """Agent 1: Queries the existing RAG corpus for internal knowledge."""
    try:
        if index is None or chunks is None or bm25 is None:
            index, chunks = load_index_and_chunks()
            bm25 = build_bm25(chunks)

        retrieved_chunks = retrieve(query, index, chunks, bm25=bm25, k=k, rerank=True)
        # Using the synchronous generate_answer from our existing script for simplicity,
        # but we could rewrite it asynchronously if needed.
        answer = generate_answer(query, retrieved_chunks)
        return answer, retrieved_chunks
    except Exception as e:
        return f"Error in internal research: {e}", []

async def agent_2_external_fact_checker(query: str, internal_answer: str, mcp_client) -> str:
    """Agent 2: Uses the MCP server to verify/update the internal answer with live web data."""
    # LLM decides what to search for
    search_prompt = f"Original question: '{query}'\nInternal answer: '{internal_answer}'\nWhat search query should we use to fact-check this with live web data? Reply ONLY with the search query string."
    
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": search_prompt}]
    )
    search_query = response.choices[0].message.content.strip() if response.choices[0].message.content else query
    
    # Use FastMCP client tool
    try:
        print(f"      [External Fact-Checker] Executing DDGS Search for: '{search_query}'")
        search_results = await mcp_client.call_tool("search_web", {"query": search_query, "max_results": 3})
    except Exception as e:
        return f"External Fact-Check couldn't execute search: {e}"
        
    # Analyze the search results
    analyze_prompt = f"""
Original Query: {query}
Internal Answer: {internal_answer}

Web Search Results:
{search_results}

Fact-check the Internal Answer using the Web Search Results. Summarize any new, contradicting, or supporting information found on the live web.
"""
    response2 = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": analyze_prompt}]
    )
    return response2.choices[0].message.content or "No external data verified."

async def agent_3_synthesizer(query: str, internal_answer: str, external_verification: str) -> str:
    """Agent 3: Merges both internal and external outputs into a final synthesized response."""
    prompt = f"""
You are the Synthesizer Agent. Your job is to merge internal corpus knowledge with external web fact-checks.

User Query: {query}

[Internal Corpus Findings]
{internal_answer}

[External Live Web Findings]
{external_verification}

Synthesize these into a cohesive final response. 
Make sure to clearly delineate what our internal corpus says vs. what the live external search reveals (e.g. "According to our corpus, X. However, a live search indicates Y.").
"""
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a precise and helpful synthesis agent."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content or "Synthesis failed."
