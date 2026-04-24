import asyncio
from typing import List, Dict, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from interfaces.reranker_provider import IRerankerProvider
from config.applicationConfig import ApplicationConfig
from util.logger import Logger

logger = Logger().get_logger("reranker.providers.openai")

class OpenAIReranker(IRerankerProvider):
    """
    OpenAI-based reranker using LLM scoring.
    """

    def __init__(self, config: ApplicationConfig):
        self.llm = ChatOpenAI(
            model=config.reranker_model,
            openai_api_key=config.openai_api_key,
            temperature=0
        )
        self.config = config

    def _extract_text_content(self, doc: Dict) -> str:
        payload = doc.get("payload", {})
        return payload.get("description") or payload.get("content") or str(payload)

    async def _score_document(self, query: str, doc_text: str) -> float:
        """Score a single document using the LLM."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert search relevance evaluator. Score the relevance of the following document to the user's query on a scale of 0 to 1, where 1 is highly relevant and 0 is not relevant at all. Output ONLY the numerical score."),
            ("user", "Query: {query}\n\nDocument: {doc_text}")
        ])
        
        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({"query": query, "doc_text": doc_text})
            score_str = response.content.strip()
            return float(score_str)
        except Exception as e:
            logger.error(f"Error scoring document: {e}")
            return 0.0

    def rerank(self, query: str, documents: List[Dict], top_n: int) -> List[Dict]:
        """Synchronous rerank (runs async in thread pool)."""
        return asyncio.run(self.rerank_async(query, documents, top_n))

    async def rerank_async(
        self,
        query: str,
        documents: List[Dict],
        top_n: int,
        *,
        thread_id: Optional[str] = None,
    ) -> List[Dict]:
        if not documents:
            return []

        logger.info(f"Reranking {len(documents)} documents using OpenAI...")
        
        # Scoring documents in parallel
        tasks = [
            self._score_document(query, self._extract_text_content(doc))
            for doc in documents
        ]
        scores = await asyncio.gather(*tasks)
        
        scored_docs = []
        for i, score in enumerate(scores):
            doc = documents[i].copy()
            doc["score"] = score
            scored_docs.append(doc)
            
        # Sort by score
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_n]
