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
