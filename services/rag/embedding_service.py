from langchain_openai import OpenAIEmbeddings
from config.applicationConfig import ApplicationConfig

class EmbeddingService:
    def __init__(self, config: ApplicationConfig):
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key
        )

    async def generate_embedding(self, text: str):
        """Generates a single vector for a query."""
        return await self.embeddings.aembed_query(text)

    async def generate_embeddings_batch(self, texts: list[str]):
        """Generates multiple vectors for data ingestion."""
        return await self.embeddings.aembed_documents(texts)
    
    # Keeping the old names for compatibility if needed, but updating to match user's request
    async def generate(self, text: str):
        return await self.generate_embedding(text)
    
    async def generate_batch(self, texts: list[str]):
        return await self.generate_embeddings_batch(texts)
