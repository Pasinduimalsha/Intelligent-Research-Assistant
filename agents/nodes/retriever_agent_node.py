from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from agents.states.research_state import ResearchState
from services.rag.rag_service import RAGService
from util.logger import Logger

logger = Logger().get_logger("agents.nodes.retriever")

class RetrieverAgent:
    """Agent responsible for retrieving context from RAG service."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    async def __call__(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n" + "="*50)
        print("--- RETRIEVER AGENT (RAG) ---")
        print("="*50)
        query = state["query"]
        
        try:
            # Retrieve documents
            docs = await self.rag_service.retrieve_context(query)
            
            # Extract source URLs or info if available in payload
            sources = []
            for doc in docs:
                source = doc.get("payload", {}).get("source_url") or doc.get("payload", {}).get("source")
                if source:
                    sources.append(source)
            
            # Format context for the next steps
            context_text = "\n".join([
                f"- {doc.get('payload', {}).get('description') or doc.get('payload', {}).get('content')}" 
                for doc in docs
            ])
            
            augmented_query = f"Context from Knowledge Base:\n{context_text}\n\nUser Query: {query}" if context_text else query
            
            return {
                "context_documents": docs,
                "sources": sources,
                "query": augmented_query
            }
        except Exception as e:
            logger.error(f"Error in RetrieverAgent: {str(e)}")
            return {
                "context_documents": [],
                "sources": []
            }
