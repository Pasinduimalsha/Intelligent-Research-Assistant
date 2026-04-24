import asyncio
import httpx
import json
from httpx_sse import aconnect_sse
from config.applicationConfig import ApplicationConfig

async def run_research_client():
    config = ApplicationConfig()
    url = f"{config.backend_url}/research/stream"
    payload = {
        "query": "Compare the data ingestion workflow defined in our internal Vector DB guide with the latest 2024 performance benchmarks for Qdrant and suggest three optimizations based on recent web findings",
        "thread_id": "hybrid-session-001"
    }

    print("="*60)
    print(f"🚀 SENDING RESEARCH QUERY: '{payload['query']}'")
    print("="*60)
    print("Waiting for AI response stream...\n")

    async with httpx.AsyncClient() as client:
        try:
            # Connect to our FastAPI Server-Sent Events endpoint
            async with aconnect_sse(client, "POST", url, json=payload, timeout=None) as event_source:
                async for sse in event_source.aiter_sse():
                    try:
                        data = json.loads(sse.data)
                        
                        # The AI is streaming text (like ChatGPT)
                        if data.get("status") == "generating":
                            print(data.get("text", ""), end="", flush=True)
                            
                        # A LangGraph Agent Node finished its work
                        elif data.get("status") == "node_completed":
                            print(f"\n\n[✓] Agent Node Completed: {data.get('node')}")
                            
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print("\n❌ Connection Error: Ensure the backend is running!")
            print("Run 'python main.py' in a separate terminal.")
            print(f"Details: {e}")

if __name__ == "__main__":
    asyncio.run(run_research_client())
