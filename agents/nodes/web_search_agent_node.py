from typing import Dict, Any
import json
from langchain_core.runnables import RunnableConfig
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agents.states.research_state import ResearchState

class WebSearchAgent:
    """Agent responsible for gathering info from the web using MCP."""
    
    def __init__(self):
        # We could inject MCP server params here if they were configurable
        pass

    async def _call_mcp_web_search(self, query: str, max_results: int = 3) -> list:
        server_params = StdioServerParameters(
            command="python",
            args=["mcp/web_mcp.py"]
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("web_search", arguments={"query": query, "max_results": max_results})
                try:
                    return json.loads(result.content[0].text)
                except Exception as e:
                    print(f"Error parsing MCP response: {e}")
                    return []

    async def __call__(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n--- WEB SEARCH AGENT (via MCP) ---")
        search_query = state["query"]
        
        if state.get("revision_count", 0) > 0:
            search_query += " latest news details"
            
        print(f"Executing web search via MCP: '{search_query}'")
        results = await self._call_mcp_web_search(search_query, max_results=3)
        
        new_sources = []
        new_notes = []
        
        for r in results:
            if isinstance(r, dict) and 'href' in r:
                new_sources.append(r['href'])
                new_notes.append(f"Source: {r['href']}\nTitle: {r.get('title')}\nSnippet: {r.get('body')}")
                
        return {"sources": new_sources, "notes": new_notes}
