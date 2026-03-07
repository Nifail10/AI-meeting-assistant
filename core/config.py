"""
core/config.py — Application settings loaded from .env

Uses pydantic-settings to parse environment variables with sensible
defaults.  Import the pre-built ``settings`` singleton anywhere:

    from core.config import settings
"""

from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the meeting-assistant pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ── Whisper model ────────────────────────────────────────────
    whisper_model: str = "base.en"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    # ── Audio capture ────────────────────────────────────────────
    audio_sample_rate: int = 16000
    audio_chunk_seconds: int = 5
    audio_input_device: Optional[int] = None


# Singleton — imported by every other module
settings = Settings()
