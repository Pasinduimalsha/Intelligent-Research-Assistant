from typing import Dict, Any
import json
from langchain_core.runnables import RunnableConfig
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agents.states.research_state import ResearchState

class FileSearchAgent:
    """Agent responsible for gathering info from local files using MCP."""
    
    def __init__(self):
        pass

    async def _call_mcp_read_file(self, file_path: str) -> str:
        server_params = StdioServerParameters(
            command="python",
            args=["mcp/file_mcp.py"]
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("read_file", arguments={"path": file_path})
                try:
                    return result.content[0].text
                except Exception as e:
                    print(f"Error parsing file MCP response: {e}")
                    return f"Error: {e}"

    async def __call__(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n" + "="*50)
        print("--- FILE SEARCH AGENT ---")
        print("="*50)
        query = state["query"]
        print("Reading local file context via MCP...")
        
        # We will attempt to read the README.md to get context about the project.
        # In a more advanced implementation, the planner node could pass specific
        # file names into the state for this node to read.
        file_path = "README.md"
        
        content = await self._call_mcp_read_file(file_path)
        
        new_notes = [f"Local Context from {file_path}:\n{content[:500]}..."] # Truncated for brevity
        
        return {"notes": new_notes}
