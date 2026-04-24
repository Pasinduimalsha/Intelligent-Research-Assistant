import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastmcp import FastMCP

from config.applicationConfig import ApplicationConfig
from agents.orchestrators.research_orchestrator import ResearchOrchestrator
from controllers.langgraph_router import router

# ==========================================
# 1. FastMCP Server Setup
# ==========================================
mcp = FastMCP("Research Assistant MCP")

@mcp.tool()
def greet(name: str) -> str:
    """A sample greeting tool exposed via FastMCP."""
    return f"Hello from the Intelligent Research Assistant, {name}!"

# ==========================================
# 2. FastAPI Lifespan & Dependency Injection
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Configuration
    config = ApplicationConfig()
    
    # Instantiate the Orchestrator
    orchestrator = ResearchOrchestrator(config)
    
    # Compile and load the graph
    await orchestrator.initialize()
    
    # Store the orchestrator in the FastAPI app state for routers to access
    app.state.orchestrator = orchestrator
    
    yield
    
    # Cleanup logic (if any) goes here

# ==========================================
# 3. FastAPI Application Initialization
# ==========================================
app = FastAPI(
    title="Intelligent Research Assistant API",
    description="API for running the LangGraph research workflows and FastMCP tools.",
    version="1.0.0",
    lifespan=lifespan
)

# Attach the LangGraph REST router
app.include_router(router)

# Mount the FastMCP server using SSE transport
app.mount("/mcp", mcp.http_app(transport='sse'))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
