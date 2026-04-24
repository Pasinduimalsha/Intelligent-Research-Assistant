from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from agents.states.research_state import ResearchState

class ScraperAgent:
    """Agent responsible for scraping full text from URLs."""
    
    def __init__(self):
        pass

    async def __call__(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n" + "="*50)
        print("--- WEB SCRAPER AGENT ---")
        print("="*50)
        # Simulating scraping from the first search result
        # In a full implementation, this node would use BeautifulSoup to extract full text.
        # For now, it simply passes through, relying on the search snippets for notes.
        return {}
