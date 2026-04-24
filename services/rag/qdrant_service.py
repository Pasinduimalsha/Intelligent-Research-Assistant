from qdrant_client import QdrantClient
from config.applicationConfig import ApplicationConfig
from util.logger import Logger
from qdrant_client.http import models

logger = Logger().get_logger("services.rag.qdrant")

class QdrantService:
    def __init__(self, config: ApplicationConfig):
        # QdrantClient can handle None values and will default to localhost:6333
        self.client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key
        )
        
        if not config.qdrant_url:
            logger.info("Qdrant URL not provided")

    async def search(self, collection_name: str, query_vector: list[float], top_k: int = 5):
        # Using query_points which is the modern unified API in qdrant-client
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k
            ).points
        except AttributeError:
            # Fallback to search if query_points is not available (older versions)
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k
            )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]

    async def upsert(self, collection_name: str, points: list):
        """Upsert points into a collection."""
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    def ensure_collection(self, collection_name: str, vector_size: int):
        """Ensure a collection exists with the correct vector size."""
        collections = self.client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            logger.info(f"Creating collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
