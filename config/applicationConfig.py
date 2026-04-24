import sys
import logging
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure default Python logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config")

class ApplicationConfig(BaseSettings):
    """
    Centralized configuration management for the Research Assistant using Pydantic Settings.
    Loads all environment variables from .env file or environment.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )
    
    # === Logging Configuration ===
    log_level: str = Field(default="INFO", description="Logging level")
    backend_url: str = Field(default="", description="Backend API URL")
    
    # === API Keys ===
    openai_api_key: str = Field(description="OpenAI API key")
    
    # === Generation Provider Configuration ===
    generation_provider_type: str = Field(
        default="openai",
        description="Provider for generation: 'openai'"
    )
    generation_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for code generation"
    )
    
    # === RAG Configuration ===
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant vector DB URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    qdrant_collection_name: str = Field(default="research_assistant", description="Qdrant collection name")
    rag_top_k: int = Field(default=10, description="Number of documents to retrieve from vector DB")
    rerank_top_n: int = Field(default=5, description="Number of documents to keep after reranking")
    embedding_dimension: int = Field(default=1536, description="Dimension of the vector embeddings")
    
    # === Embedding Configuration ===
    embedding_provider_type: str = Field(default="openai", description="Embedding provider type")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")

    # === Reranker Configuration ===
    reranker_provider_type: str = Field(default="openai", description="Reranker provider type")
    reranker_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use for reranking"
    )
    
    def model_post_init(self, __context) -> None:
        """Validate required fields after model initialization."""
        self._validate_required_fields()
    
    def _validate_required_fields(self) -> None:
        """Validate that all required environment variables are present."""
        required_fields = {
            "openai_api_key": self.openai_api_key,
        }
        
        missing_fields = [field for field, value in required_fields.items() if not value]
        
        if missing_fields:
            error_msg = (
                "\n" + "="*70 + "\n"
                "❌ CONFIGURATION ERROR: Missing Required Environment Variables\n"
                "="*70 + "\n\n"
                "The following required environment variables are missing:\n"
            )
            for field in missing_fields:
                error_msg += f"  • {field.upper()}\n"
            error_msg += (
                "\nPlease set these variables in your .env file or environment.\n"
                "="*70 + "\n"
            )
            logger.error(error_msg)
            sys.exit(1)
