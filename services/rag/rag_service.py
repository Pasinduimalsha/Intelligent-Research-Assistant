from typing import List, Dict
from services.rag.qdrant_service import QdrantService
from services.rag.embedding_service import EmbeddingService
from interfaces.reranker_provider import IRerankerProvider
from util.logger import Logger

logger = Logger().get_logger("services.rag.rag_service")

class RAGService:
    def __init__(
        self, 
        qdrant_service: QdrantService, 
        embedding_service: EmbeddingService, 
        collection_name: str,
        top_k: int = 10,
        rerank_top_n: int = 5,
        reranker: IRerankerProvider = None
    ):
        self.qdrant = qdrant_service
        self.embeddings = embedding_service
        self.collection_name = collection_name
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.reranker = reranker

    async def retrieve_context(self, query: str) -> List[Dict]:
        print(f"\n[RAG SERVICE] Processing query: {query}")
        
        # 1. Generate Query Vector
        query_vector = await self.embeddings.generate(query)
        
        # 2. Vector Search
        # Using search method which is standard in qdrant-client
        try:
            results = await self.qdrant.search(self.collection_name, query_vector, top_k=self.top_k)
            print(f"[RAG SERVICE] Retrieved {len(results)} raw chunks from Qdrant.")
        except Exception as e:
            print(f"[RAG SERVICE] Search Error: {e}")
            # Fallback/Debug: check client attributes
            print(f"[DEBUG] QdrantClient attributes: {dir(self.qdrant.client)}")
            raise e
        
        if not results:
            return []
            
        # 3. Optional Rerank
        if self.reranker:
            print(f"[RAG SERVICE] Reranking top {len(results)} chunks using {self.reranker.__class__.__name__}...")
            results = await self.reranker.rerank_async(query, results, top_n=self.rerank_top_n)
            print(f"[RAG SERVICE] Reranking complete. Kept top {len(results)} chunks.")
            
        # Log chunk contents for visibility
        for i, doc in enumerate(results):
            content = doc.get("payload", {}).get("content", "")[:100] + "..."
            print(f"  > Chunk {i+1}: {content}")
            
        return results
