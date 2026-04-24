from langchain_openai import ChatOpenAI
from config.applicationConfig import ApplicationConfig
from agents.graphs.research_graph import create_graph

class ResearchOrchestrator:
    """Manages the lifecycle and execution of the LangGraph application."""
    
    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=self.config.generation_model,
            api_key=self.config.openai_api_key,
            temperature=0
        )
        self.app = None

    async def initialize(self):
        """Compile the graph and store it."""
        print("Initializing Research Assistant Orchestrator...")
        self.app = await create_graph(llm=self.llm, app_config=self.config)
        print("Graph compiled successfully.")
        return self.app
        
    async def run_stream(self, initial_state: dict, config: dict):
        """Yield events from the graph using astream_events."""
        if not self.app:
            raise ValueError("Orchestrator not initialized. Call initialize() first.")
        
        # We use astream_events with version v2 to get detailed progress
        async for event in self.app.astream_events(initial_state, config=config, version="v2"):
            yield event

    async def run_sync(self, initial_state: dict, config: dict):
        """Run the graph asynchronously and wait for the final result."""
        if not self.app:
            raise ValueError("Orchestrator not initialized. Call initialize() first.")
        
        # ainvoke runs the graph to completion and returns the final state
        return await self.app.ainvoke(initial_state, config=config)
