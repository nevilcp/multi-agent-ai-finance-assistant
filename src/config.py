"""Configuration management."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class GoogleAIConfig:
    """Google AI Studio configuration."""
    api_key: str
    flash_model: str = "gemini-3-flash-preview"
    flash_lite_model: str = "gemini-3.1-flash-lite-preview"
    flash_rpm: int = 5
    flash_rpd: int = 20
    flash_lite_rpm: int = 15
    flash_lite_rpd: int = 500
    tpm: int = 250_000


@dataclass(frozen=True)
class AppConfig:
    """Application configuration."""
    google: GoogleAIConfig
    finnhub_key: str
    log_level: str


def load_config() -> AppConfig:
    """Load and validate configuration from environment."""
    google_key = os.getenv("GOOGLE_API_KEY")
    finnhub_key = os.getenv("FINNHUB_API_KEY")

    if not google_key:
        raise ValueError("GOOGLE_API_KEY is required")
    if not finnhub_key:
        raise ValueError("FINNHUB_API_KEY is required")

    return AppConfig(
        google=GoogleAIConfig(api_key=google_key),
        finnhub_key=finnhub_key,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


config = load_config()
