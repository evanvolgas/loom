"""Configuration management for Loom."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class GlobalConfig(BaseModel):
    """Global runtime configuration."""

    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    google_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY")
    )
    groq_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )

    # Paths
    pipelines_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("LOOM_PIPELINES_DIR", "pipelines"))
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("LOOM_DATA_DIR", "data"))
    )
    prompts_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("LOOM_PROMPTS_DIR", "prompts"))
    )

    # Circuit Breaker Defaults
    circuit_breaker_enabled: bool = Field(
        default_factory=lambda: os.getenv("LOOM_CIRCUIT_BREAKER_ENABLED", "true").lower()
        == "true"
    )
    circuit_breaker_failure_threshold: int = Field(
        default_factory=lambda: int(
            os.getenv("LOOM_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
        )
    )
    circuit_breaker_timeout: int = Field(
        default_factory=lambda: int(os.getenv("LOOM_CIRCUIT_BREAKER_TIMEOUT", "60"))
    )

    # Retry Defaults
    retry_max_attempts: int = Field(
        default_factory=lambda: int(os.getenv("LOOM_RETRY_MAX_ATTEMPTS", "3"))
    )
    retry_initial_delay: float = Field(
        default_factory=lambda: float(os.getenv("LOOM_RETRY_INITIAL_DELAY", "1.0"))
    )

    # Logging
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOOM_LOG_LEVEL", "INFO")
    )
    log_format: str = Field(
        default_factory=lambda: os.getenv(
            "LOOM_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )

    # Performance
    max_concurrent_records: int = Field(
        default_factory=lambda: int(os.getenv("LOOM_MAX_CONCURRENT_RECORDS", "10"))
    )


# Global configuration instance
config = GlobalConfig()


def get_config() -> GlobalConfig:
    """Get global configuration instance."""
    return config


def reload_config() -> GlobalConfig:
    """Reload configuration from environment."""
    load_dotenv(override=True)
    global config
    config = GlobalConfig()
    return config
