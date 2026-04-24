import re
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from agents.states.research_state import ResearchState

class QueryNormalizerAgent:
    """Agent responsible for basic normalization: removing emojis and trailing spaces."""
    
    def __init__(self, llm=None):
        # LLM no longer needed for basic normalization
        pass

    async def __call__(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n" + "="*50)
        print("--- QUERY NORMALIZER AGENT ---")
        print("="*50)
        raw_query = state["query"]
        
        # 1. Trim trailing spaces
        normalized_query = raw_query.strip()
        
        # 2. Remove emojis using regex
        # This regex covers most emoji ranges
        normalized_query = re.sub(r'[^\x00-\x7F]+', '', normalized_query)
        
        # 3. Clean up any resulting double spaces
        normalized_query = re.sub(r'\s+', ' ', normalized_query)
        
        print(f"Normalized Query: {normalized_query}")
        
        return {
            "query": normalized_query,
            "original_query": raw_query
        }
