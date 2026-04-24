from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import json
import asyncio

from pydantic import BaseModel

class ResearchRequest(BaseModel):
    query: str
    thread_id: str = "default_thread"

router = APIRouter(prefix="/research", tags=["research"])

@router.post("/stream")
async def stream_research_graph(request: Request, payload: ResearchRequest):
    # 1. Grab the fully initialized orchestrator from the app state
    orchestrator = request.app.state.orchestrator
    
    # 2. Setup the Initial State
    initial_state = {
        "query": payload.query,
        "research_plan": "",
        "sources": [],
        "notes": [],
        "draft": "",
        "revision_count": 0,
        "is_complete": False
    }
    
    # 3. Setup Config for memory/thread tracking
    config = {"configurable": {"thread_id": payload.thread_id}}
    
    # 4. Stream the events
    async def event_stream():
        try:
            # Yield events from the orchestrator
            async for event in orchestrator.run_stream(initial_state, config):
                
                # Filter for LLM streaming chunks (Writer node, Planner node, etc.)
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        yield f"data: {json.dumps({'status': 'generating', 'text': chunk})}\n\n"
                        
                # Catch when specific nodes start or finish
                elif event["event"] == "on_chain_start" and "name" in event:
                    node_name = event["name"]
                    valid_nodes = ["InputGuard", "Normalizer", "Router", "Retriever", "Planner", "Search", "LocalSearch", "Scraper", "Writer", "OutputGuard", "Reviewer"]
                    if node_name in valid_nodes:
                        yield f"data: {json.dumps({'status': 'node_started', 'node': node_name})}\n\n"

                elif event["event"] == "on_chain_end" and "name" in event:
                    node_name = event["name"]
                    valid_nodes = ["InputGuard", "Normalizer", "Router", "Retriever", "Planner", "Search", "LocalSearch", "Scraper", "Writer", "OutputGuard", "Reviewer"]
                    if node_name in valid_nodes:
                        yield f"data: {json.dumps({'status': 'node_completed', 'node': node_name})}\n\n"
                        
        except asyncio.CancelledError:
            print("Client disconnected.")
            
    # 5. Return as a Server-Sent Event stream
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )

@router.post("/run")
async def run_research_graph(request: Request, payload: ResearchRequest):
    """Non-streaming endpoint that waits for the graph to complete and returns the final draft."""
    orchestrator = request.app.state.orchestrator
    
    initial_state = {
        "query": payload.query,
        "research_plan": "",
        "sources": [],
        "notes": [],
        "draft": "",
        "revision_count": 0,
        "is_complete": False
    }
    
    config = {"configurable": {"thread_id": payload.thread_id}}
    
    # Run synchronously to completion
    final_state = await orchestrator.run_sync(initial_state, config)
    
    return {
        "status": "success",
        "thread_id": payload.thread_id,
        "final_draft": final_state.get("draft", ""),
        "sources": final_state.get("sources", []),
        "revisions": final_state.get("revision_count", 0)
    }
