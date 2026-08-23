import asyncio
import os
import sys

# Ensure scripts can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.agents import agent_1_internal_researcher, agent_2_external_fact_checker, agent_3_synthesizer

# Safely import FastMCP client - handle case where FastMCP might not be installed
try:
    from fastmcp import Client
except ImportError:
    print("Warning: fastmcp package not installed. The external web fact-checker will not work.")
    Client = None

async def main():
    print("=== Multi-Agent RAG with MCP Web Search ===")
    query = input("\nEnter your question: ")
    
    print("\n[Agent 1: Internal Researcher] Searching local corpus...")
    internal_answer = await agent_1_internal_researcher(query)
    
    if Client is None:
        print("\n[Client Error] FastMCP client is missing. Here's what we found internally:\n")
        print(internal_answer)
        return

    print("\n[Agent 2: External Fact-Checker] Connecting to MCP Server for web search...")
    try:
        # Assuming MCP server is running on localhost:8000 via `fastmcp run app/mcp_server.py:mcp --transport http --port 8000`
        mcp_client = Client("http://localhost:8000/mcp")
        async with mcp_client:
            external_answer = await agent_2_external_fact_checker(query, internal_answer, mcp_client)
            
            print("\n[Agent 3: Synthesizer] Merging outputs...")
            final_response = await agent_3_synthesizer(query, internal_answer, external_answer)
            
            print(f"\n================================\n===== DRAFT FINAL RESPONSE =====\n================================\n")
            print(final_response)
            print(f"\n================================================================")
            
            # Human in the loop step
            decision = input("\nAction: (A)pprove and save, (R)eject, or (U)pdate vector DB with new facts? [A/R/U]: ").strip().lower()
            
            if decision == 'a':
                filename = input("Enter filename to save (e.g. data/final_report.md): ")
                res = await mcp_client.call_tool("create_markdown_report", {"filename": filename, "content": final_response})
                print(f"[MCP Server]: {res}")
                
            elif decision == 'u':
                print("[Publishing] Updating active database...")
                res = await mcp_client.call_tool("add_to_database", {"text": external_answer, "source": "Live Web Search MCP"})
                print(f"[MCP Server]: {res}")
                
                filename = input("Enter filename to save the report to (e.g. data/final_report.md): ")
                res = await mcp_client.call_tool("create_markdown_report", {"filename": filename, "content": final_response})
                print(f"[MCP Server]: {res}")
                
            else:
                print("Draft rejected. Exiting without saving.")
                
    except Exception as e:
        print(f"\nMCP Server connection or execution error: {e}")
        print("Please ensure you have started the MCP server in a separate terminal via:\n")
        print("  fastmcp run app/mcp_server.py:mcp --transport http --port 8000\n")
        
        print("For now, here is the internal answer our Researcher found:\n")
        print(internal_answer)

if __name__ == "__main__":
    asyncio.run(main())
