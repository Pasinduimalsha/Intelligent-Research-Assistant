import asyncio
from contextlib import AsyncExitStack
from typing import List, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

class MCPClient:
    """
    Manages persistent connections to multiple local MCP servers.
    Provides a unified interface to load all available tools.
    """
    
    def __init__(self, server_scripts: List[str]):
        """
        :param server_scripts: List of script paths (e.g., ['mcp/web_mcp.py', 'mcp/file_mcp.py'])
        """
        self.server_scripts = server_scripts
        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []

    async def start(self):
        """Starts subprocesses and initializes sessions for all registered MCP servers."""
        for script_path in self.server_scripts:
            server_params = StdioServerParameters(
                command="python",
                args=[script_path]
            )
            
            # Enter the stdio_client context
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
            
            # Enter the ClientSession context
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            
            # Initialize the session
            await session.initialize()
            
            self.sessions.append(session)
            print(f"✅ Successfully connected to MCP server: {script_path}")

    async def get_all_tools(self) -> List[Any]:
        """
        Fetches tools from all active MCP sessions and returns a combined list of LangChain tools.
        """
        all_tools = []
        for session in self.sessions:
            tools = await load_mcp_tools(session)
            all_tools.extend(tools)
        return all_tools

    async def stop(self):
        """Cleanly shuts down all managed MCP server subprocesses."""
        await self.exit_stack.aclose()
        self.sessions.clear()
        print("🛑 Shut down all MCP server connections.")
