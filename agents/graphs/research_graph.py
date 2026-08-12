from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from config.applicationConfig import ApplicationConfig
from agents.states.research_state import ResearchState
from agents.nodes.input_guardrail_node import InputGuardrailAgent
from agents.nodes.query_normalizer_node import QueryNormalizerAgent
from agents.nodes.planner_agent_node import PlannerAgent
from agents.nodes.tool_execution_agent_node import ToolExecutionAgent
from agents.nodes.scraper_agent_node import ScraperAgent
from agents.nodes.response_gen_agent_node import ResponseGenAgent
from agents.nodes.output_guardrail_node import OutputGuardrailAgent
from agents.nodes.reviewer_agent_node import ReviewerAgent
from agents.nodes.retriever_agent_node import RetrieverAgent
from services.rag.rag_service import RAGService
from services.rag.qdrant_service import QdrantService
from services.rag.embedding_service import EmbeddingService
from providers.reranker.openai_reranker import OpenAIReranker

 
from agents.nodes.router_node import RouterAgent

async def create_graph(
    llm: ChatOpenAI,
    app_config: ApplicationConfig,
    mcp_tools: list
):
    """
    Create LangGraph for the research assistant.
    """
    graph = StateGraph(ResearchState)
    
    # 1. Initialize RAG Infrastructure
    embedding_service = EmbeddingService(app_config)
    qdrant_service = QdrantService(app_config)
    reranker_provider = OpenAIReranker(app_config)
    rag_service = RAGService(
        qdrant_service=qdrant_service,
        embedding_service=embedding_service,
        collection_name=app_config.qdrant_collection_name,
        top_k=app_config.rag_top_k,
        rerank_top_n=app_config.rerank_top_n,
        reranker=reranker_provider
    )
   
    
    # 2. Initialize nodes with injected dependencies
    input_guard_node_instance = InputGuardrailAgent(llm)
    normalizer_node_instance = QueryNormalizerAgent(llm)
    router_node_instance = RouterAgent(llm)
    retriever_node_instance = RetrieverAgent(rag_service)
    planner_node_instance = PlannerAgent(llm)
    tool_execution_node_instance = ToolExecutionAgent(llm, mcp_tools)
    scraper_node_instance = ScraperAgent()
    writer_node_instance = ResponseGenAgent(llm)
    output_guard_node_instance = OutputGuardrailAgent(llm)
    reviewer_node_instance = ReviewerAgent(llm)
    
    # 3. Add nodes
    graph.add_node("InputGuard", input_guard_node_instance)
    graph.add_node("Normalizer", normalizer_node_instance)
    graph.add_node("Router", router_node_instance)
    graph.add_node("Retriever", retriever_node_instance)
    graph.add_node("Planner", planner_node_instance)
    graph.add_node("ToolExecution", tool_execution_node_instance)
    graph.add_node("Scraper", scraper_node_instance)
    graph.add_node("Writer", writer_node_instance)
    graph.add_node("OutputGuard", output_guard_node_instance)
    graph.add_node("Reviewer", reviewer_node_instance)
    
    # 4. Set entry point
    graph.set_entry_point("InputGuard")
    
    # --- ROUTING FUNCTIONS ---

    def route_after_input(state: ResearchState) -> Literal["Normalizer", "__end__"]:
        if not state.get("is_safe_input", True):
            return END
        return "Normalizer"
        
    def route_after_router(state: ResearchState) -> Literal["Retriever", "Planner"]:
        needed = state.get("needed_sources", [])
        if "internal" in needed:
            return "Retriever"
        return "Planner"

    def route_after_planner(state: ResearchState) -> Literal["ToolExecution", "Scraper"]:
        needed = state.get("needed_sources", [])
        if "web" in needed or "local" in needed:
            return "ToolExecution"
        return "Scraper"

    def route_after_tool_execution(state: ResearchState) -> Literal["Router", "Scraper"]:
        if state.get("needs_reroute") and state.get("reroute_count", 0) < 3:
            print("\n=> RE-ROUTING: Tool Execution discovered sub-query.")
            return "Router"
        return "Scraper"

    def route_after_output(state: ResearchState) -> Literal["Reviewer", "__end__"]:
        if not state.get("is_safe_output", True):
            return END
        return "Reviewer"
        
    def route_after_review(state: ResearchState) -> Literal["Planner", "__end__"]:
        if state.get("is_complete", False) or state.get("revision_count", 0) >= 3:
            return END
        return "Planner"

    # --- ADD EDGES ---

    graph.add_conditional_edges("InputGuard", route_after_input, {"Normalizer": "Normalizer", END: END})
    graph.add_edge("Normalizer", "Router")
    
    graph.add_conditional_edges(
        "Router", 
        route_after_router, 
        {"Retriever": "Retriever", "Planner": "Planner"}
    )
    
    graph.add_edge("Retriever", "Planner")
    
    graph.add_conditional_edges(
        "Planner",
        route_after_planner,
        {"ToolExecution": "ToolExecution", "Scraper": "Scraper"}
    )
    
    graph.add_conditional_edges(
        "ToolExecution",
        route_after_tool_execution,
        {"Router": "Router", "Scraper": "Scraper"}
    )
    
    graph.add_edge("Scraper", "Writer")
    graph.add_edge("Writer", "OutputGuard")
    
    graph.add_conditional_edges("OutputGuard", route_after_output, {"Reviewer": "Reviewer", END: END})
    graph.add_conditional_edges("Reviewer", route_after_review, {"Planner": "Planner", END: END})
    
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    return app
