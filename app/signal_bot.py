"""
Native Python wrapper for signal-cli for RAG Regulators.
No Docker or signal-cli-rest-api needed!

Usage:
    python app/signal_bot.py
"""

import os
import sys
import json
import time
import logging
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from scripts.query_rag import load_index_and_chunks, build_bm25, retrieve, generate_answer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load RAG index once at startup
logging.info("Loading FAISS index...")
INDEX, CHUNKS = load_index_and_chunks()
BM25 = build_bm25(CHUNKS)
logging.info(f"✅ Loaded {len(CHUNKS)} chunks. RAG ready!")

PHONE_NUMBER = os.environ.get("SIGNAL_BOT_NUMBER", "")


def send_signal_message(recipient: str, text: str):
    """Sends a message via signal-cli."""
    logging.info(f"Sending reply to {recipient}...")
    try:
        subprocess.run(
            ["signal-cli", "-u", PHONE_NUMBER, "send", "-m", text, recipient],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        logging.error(f"Failed to send message: {e}")


import asyncio
from fastmcp import Client
from agents import agent_1_internal_researcher, agent_2_external_fact_checker, agent_3_synthesizer

MCP_URL = "http://localhost:8001/mcp"
_mcp_proc = None

def start_mcp_server():
    """Start MCP server directly without importing app.py (prevents loading FAISS twice)"""
    global _mcp_proc
    mcp_script = os.path.join(PROJECT_ROOT, "app", "mcp_server.py")
    fastmcp_bin = os.path.join(os.path.dirname(sys.executable), "fastmcp")
    if not os.path.exists(fastmcp_bin): fastmcp_bin = "fastmcp"
    
    cmd = [fastmcp_bin, "run", mcp_script + ":mcp", "--transport", "http", "--host", "127.0.0.1", "--port", "8001"]
    
    _mcp_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(2) # Give it a moment to boot

import atexit
def _cleanup():
    if _mcp_proc and _mcp_proc.poll() is None:
        _mcp_proc.terminate()
atexit.register(_cleanup)

# Start the MCP server
logging.info("Starting up MCP Server for web search...")
start_mcp_server()

MCP_URL = "http://localhost:8001/mcp"

# Keep track of agentic mode per user (default to False for speed)
USER_PREFS = {}

async def process_query(query: str, source_number: str) -> str:
    """Runs the AGENTIC RAG pipeline (Optional Web Search)."""
    # Check if this user wants agentic mode (default off)
    use_agentic = USER_PREFS.get(source_number, False)
    
    try:
        logging.info(f"1️⃣ Running internal researcher (Agentic: {use_agentic})...")
        internal_ans, retrieved = await agent_1_internal_researcher(query, index=INDEX, chunks=CHUNKS, bm25=BM25)
        
        sources = "\n".join(
            f"• {r['source']} — p.{r['page']} (score: {r['score']:.2f})"
            for r in retrieved[:3]
        )
        sources_str = f"\n\n📚 Internal Sources:\n{sources}"
        
        if not use_agentic:
            return f"{internal_ans}{sources_str}\n\n*(Tip: Reply '/agentic on' to enable web search!)*"
            
        try:
            logging.info("2️⃣ Connecting to Web Agent / MCP...")
            mcp_client = Client(MCP_URL)
            async with mcp_client:
                external_answer = await agent_2_external_fact_checker(query, internal_ans, mcp_client)
                
                logging.info("3️⃣ Synthesizing final response...")
                final_response = await agent_3_synthesizer(query, internal_ans, external_answer)
                
                return f"{final_response}{sources_str}"
        except Exception as e:
            logging.error(f"Web agent failed: {e}")
            return f"⚠️ *Web Agent Offline. Falling back to internal corpus:*\n\n{internal_ans}{sources_str}"
            
    except Exception as e:
        logging.error(f"Error during agentic RAG: {e}")
        return f"❌ Error processing your question: {e}"


def check_for_messages():
    """Polls signal-cli for new messages."""
    cmd = ["signal-cli", "-o", "json", "-u", PHONE_NUMBER, "receive"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if not result.stdout.strip():
        if result.stderr.strip() and "unrecognized" in result.stderr:
            logging.error(f"Signal-cli error: {result.stderr.strip()}")
        return 
        
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
             continue
        try:
            payload = json.loads(line)
            envelope = payload.get("envelope", {})
            source = envelope.get("sourceNumber") or envelope.get("source")
            data = envelope.get("dataMessage") or envelope.get("syncMessage", {}).get("sentMessage", {})
            
            message_text = data.get("message", "").strip()
            
            if source and message_text:
                logging.info(f"Received query from {source}: {message_text}")
                
                # Check for settings commands
                if message_text.lower() == "/agentic on":
                    USER_PREFS[source] = True
                    send_signal_message(source, "🟢 Agentic Mode ENABLED. Future questions will be verified against live web search.")
                    continue
                elif message_text.lower() == "/agentic off":
                    USER_PREFS[source] = False
                    send_signal_message(source, "🔴 Agentic Mode DISABLED. Future questions will only use the internal PDF database.")
                    continue
                
                use_agentic = USER_PREFS.get(source, False)
                if use_agentic:
                    send_signal_message(source, "🔍 Agents are researching your question...")
                else:
                    send_signal_message(source, "🔍 Searching internal database...")
                
                # Process RAG and reply
                reply = asyncio.run(process_query(message_text, source))
                send_signal_message(source, reply)
                
        except Exception as e:
            logging.error(f"Failed to parse incoming message payload: {e}")
    """Polls signal-cli for new messages."""
    # Global flags like -o json MUST come before the subcommand (receive)
    cmd = ["signal-cli", "-o", "json", "-u", PHONE_NUMBER, "receive"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if not result.stdout.strip():
        # If there's an error like unrecognized arguments, it prints to stderr
        if result.stderr.strip() and "unrecognized" in result.stderr:
            logging.error(f"Signal-cli error: {result.stderr.strip()}")
        return # No new messages
        
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
             continue
        try:
            payload = json.loads(line)
            envelope = payload.get("envelope", {})
            source = envelope.get("sourceNumber") or envelope.get("source")
            data = envelope.get("dataMessage") or envelope.get("syncMessage", {}).get("sentMessage", {})
            
            message_text = data.get("message")
            
            if source and message_text:
                logging.info(f"Received query from {source}: {message_text}")
                
                # Send immediate confirmation
                send_signal_message(source, "🔍 Searching regulatory corpus...")
                
                # Process RAG and reply
                reply = process_query(message_text.strip())
                send_signal_message(source, reply)
                
        except Exception as e:
            logging.error(f"Failed to parse incoming message payload: {e}")


if __name__ == "__main__":
    logging.info(f"🤖 RAG Regulators Native Signal Bot listening on {PHONE_NUMBER}...")
    logging.info("Send a message to this number on Signal to test it! Press Ctrl+C to stop.")
    
    # Simple polling loop — no daemons or REST APIs required!
    while True:
        try:
            check_for_messages()
            time.sleep(2) # check every 2 seconds
        except KeyboardInterrupt:
            logging.info("Shutting down bot...")
            break
        except Exception as e:
            logging.error(f"Loop error: {e}")
            time.sleep(5)
