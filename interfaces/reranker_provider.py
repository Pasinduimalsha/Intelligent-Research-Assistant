"""
Interface for reranker providers.
This interface abstracts different reranker providers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import asyncio


class IRerankerProvider(ABC):
    """Abstract interface for reranker providers"""
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_n: int
    ) -> List[Dict]:
        """
        Rerank documents based on query relevance.
        Args:
            query: The search query string
            documents: List of documents with structure:
                {
                    "id": ...,
                    "score": ...,
                    "payload": {...}
                }
            top_n: Number of top documents to return after reranking

        Returns:
            List of reranked documents with updated relevance scores.
            Documents should maintain the same structure as input but with
            updated scores reflecting reranking.
        """
        pass

    async def rerank_async(
        self,
        query: str,
        documents: List[Dict],
        top_n: int,
        *,
        thread_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Async rerank with optional stop check (e.g. before HTTP).
        Default implementation runs sync rerank in thread pool.
        Override to check thread_id (stop requested) before long operations.
        """
        return await asyncio.to_thread(self.rerank, query, documents, top_n)
