from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from config.applicationConfig import ApplicationConfig
from agents.states.research_state import ResearchState
from agents.nodes.input_guardrail_node import InputGuardrailAgent
from agents.nodes.query_normalizer_node import QueryNormalizerAgent
from agents.nodes.planner_agent_node import PlannerAgent
from agents.nodes.web_search_agent_node import WebSearchAgent
from agents.nodes.file_search_agent_node import FileSearchAgent
from agents.nodes.scraper_agent_node import ScraperAgent
from agents.nodes.response_gen_agent_node import ResponseGenAgent
from agents.nodes.output_guardrail_node import OutputGuardrailAgent
from agents.nodes.reviewer_agent_node import ReviewerAgent

async def create_graph(
    llm: ChatOpenAI,
    app_config: ApplicationConfig
):
    """
    Create LangGraph for the research assistant.
    """
    graph = StateGraph(ResearchState)
    
    # 1. Initialize nodes with injected dependencies
    input_guard_node_instance = InputGuardrailAgent(llm)
    normalizer_node_instance = QueryNormalizerAgent(llm)
    planner_node_instance = PlannerAgent(llm)
    search_node_instance = WebSearchAgent()
    local_search_node_instance = FileSearchAgent()
    scraper_node_instance = ScraperAgent()
    writer_node_instance = ResponseGenAgent(llm)
    output_guard_node_instance = OutputGuardrailAgent(llm)
    reviewer_node_instance = ReviewerAgent(llm)
    
    # 2. Add nodes
    graph.add_node("InputGuard", input_guard_node_instance)
    graph.add_node("Normalizer", normalizer_node_instance)
    graph.add_node("Planner", planner_node_instance)
    graph.add_node("Search", search_node_instance)
    graph.add_node("LocalSearch", local_search_node_instance)
    graph.add_node("Scraper", scraper_node_instance)
    graph.add_node("Writer", writer_node_instance)
    graph.add_node("OutputGuard", output_guard_node_instance)
    graph.add_node("Reviewer", reviewer_node_instance)
    
    # 3. Set entry point
    graph.set_entry_point("InputGuard")
    
    # 4. Conditional routing after InputGuard
    def route_after_input(state: ResearchState) -> Literal["Normalizer", "__end__"]:
        if not state.get("is_safe_input", True):
            return END
        return "Normalizer"
        
    graph.add_conditional_edges(
        "InputGuard",
        route_after_input,
        {
            "Normalizer": "Normalizer",
            END: END
        }
    )
    
    # 5. Normalizer to Planner Edge
    graph.add_edge("Normalizer", "Planner")
    
    # 6. Standard Edges
    graph.add_edge("Planner", "Search")
    graph.add_edge("Search", "LocalSearch")
    graph.add_edge("LocalSearch", "Scraper")
    graph.add_edge("Scraper", "Writer")
    graph.add_edge("Writer", "OutputGuard")
    
    # 6. Conditional routing after OutputGuard
    def route_after_output(state: ResearchState) -> Literal["Reviewer", "__end__"]:
        if not state.get("is_safe_output", True):
            return END
        return "Reviewer"
        
    graph.add_conditional_edges(
        "OutputGuard",
        route_after_output,
        {
            "Reviewer": "Reviewer",
            END: END
        }
    )
    
    # 7. Conditional routing after Reviewer
    def route_after_review(state: ResearchState) -> Literal["Search", "__end__"]:
        if state.get("is_complete", False) or state.get("revision_count", 0) >= 3:
            print("\n=> Routing to END (Research Complete or Max Revisions Reached)")
            return END
        else:
            print("\n=> Routing back to SEARCH (Research Incomplete)")
            return "Search"

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
