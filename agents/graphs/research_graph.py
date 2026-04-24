from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from config.applicationConfig import ApplicationConfig
from agents.states.research_state import ResearchState
from agents.nodes.planner_agent_node import PlannerAgent
from agents.nodes.web_search_agent_node import WebSearchAgent
from agents.nodes.file_search_agent_node import FileSearchAgent
from agents.nodes.scraper_agent_node import ScraperAgent
from agents.nodes.response_gen_agent_node import ResponseGenAgent
from agents.nodes.reviewer_agent_node import ReviewerAgent

async def create_graph(
    llm: ChatOpenAI,
    app_config: ApplicationConfig
):
    """
    Create LangGraph for the research assistant.
    
    Args:
        llm: LLM provider for generation and reasoning.
        app_config: Application configuration.
    """
    graph = StateGraph(ResearchState)
    
    # 1. Initialize nodes with injected dependencies
    planner_node_instance = PlannerAgent(llm)
    search_node_instance = WebSearchAgent()
    local_search_node_instance = FileSearchAgent()
    scraper_node_instance = ScraperAgent()
    writer_node_instance = ResponseGenAgent(llm)
    reviewer_node_instance = ReviewerAgent(llm)
    
    # 2. Add nodes
    graph.add_node("Planner", planner_node_instance)
    graph.add_node("Search", search_node_instance)
    graph.add_node("LocalSearch", local_search_node_instance)
    graph.add_node("Scraper", scraper_node_instance)
    graph.add_node("Writer", writer_node_instance)
    graph.add_node("Reviewer", reviewer_node_instance)
    
    # 3. Set entry point
    graph.set_entry_point("Planner")
    
    # 4. Standard Edges (Routing through BOTH MCPs)
    graph.add_edge("Planner", "Search")
    graph.add_edge("Search", "LocalSearch")
    graph.add_edge("LocalSearch", "Scraper")
    graph.add_edge("Scraper", "Writer")
    graph.add_edge("Writer", "Reviewer")
    
    # 5. Conditional routing logic
    def route_after_review(state: ResearchState) -> Literal["Search", "__end__"]:
        """Conditional router based on the reviewer's evaluation."""
        if state.get("is_complete", False) or state.get("revision_count", 0) >= 3:
            print("\n=> Routing to END (Research Complete or Max Revisions Reached)")
            return END
        else:
            print("\n=> Routing back to SEARCH (Research Incomplete)")
            return "Search"

    # 6. Add conditional edges
    graph.add_conditional_edges(
        "Reviewer",
        route_after_review,
        {
            "Search": "Search",
            END: END
        }
    )
    
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    return app
