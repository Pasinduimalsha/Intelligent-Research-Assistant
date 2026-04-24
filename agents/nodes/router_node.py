from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agents.states.research_state import ResearchState

class RouterAgent:
    """Agent responsible for deciding which data sources are needed."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(
            schema={
                "title": "RoutingDecision",
                "type": "object",
                "properties": {
                    "needed_sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["internal", "web", "local"]},
                        "description": "List of sources needed: 'internal' (RAG), 'web' (DuckDuckGo), 'local' (Files)."
                    },
                    "reasoning": {"type": "string", "description": "Short explanation for the choice."}
                },
                "required": ["needed_sources", "reasoning"]
            }
        )

    async def __call__(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n" + "="*50)
        print("--- ROUTER AGENT ---")
        print("="*50)
        
        # Use followup_query if we are in a re-routing loop
        active_query = state.get("followup_query") or state["query"]
        if state.get("needs_reroute"):
            print(f"Loop-back detected. Analyzing sub-query: {active_query}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intelligent research router. Analyze the query and decide which information sources are absolutely necessary to answer it accurately. \n\nSources:\n- 'internal': Your vector database (use for company specific info, internal guides, or RAG-indexed datasets).\n- 'web': Public internet search.\n- 'local': Local file system search.\n\nOnly select sources that are relevant. If the query is generic, stick to 'web'. If it asks about internal docs, use 'internal'."),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm
        result = await chain.ainvoke({"query": active_query})
        
        needed = result.get("needed_sources", ["web"])
        print(f"Decision: {needed} (Reason: {result.get('reasoning')})")
        
        return {
            "needed_sources": needed,
            "needs_reroute": False,  # Reset the flag after decision
            "followup_query": "",    # Clear the sub-query
            "reroute_count": state.get("reroute_count", 0) + (1 if state.get("needs_reroute") else 0)
        }
