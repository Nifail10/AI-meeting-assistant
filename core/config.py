"""
core/config.py — Application settings loaded from .env

Uses pydantic-settings to parse environment variables with sensible
defaults.  Import the pre-built ``settings`` singleton anywhere:

    from core.config import settings
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


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

    # ── Validators ───────────────────────────────────────────────

    @field_validator("whisper_compute_type")
    @classmethod
    def validate_compute_type(cls, v: str, info) -> str:  # noqa: N805
        """Check device/compute-type compatibility at startup."""
        device = info.data.get("whisper_device", "cuda")

        if device == "cuda" and v == "int8":
            logger.warning(
                "[CONFIG] compute_type='int8' on CUDA is valid but suboptimal "
                "— consider 'float16' for better GPU performance."
            )
        if device == "cpu" and v == "float16":
            raise ValueError(
                "[CONFIG] compute_type='float16' is not supported on CPU. "
                "Use 'int8' or 'float32' instead."
            )
        return v

    # ── Helpers ──────────────────────────────────────────────────

    def log_active_config(self) -> None:
        """Log every setting at INFO level for startup diagnostics."""
        logger.info("[CONFIG] whisper_model      = %s", self.whisper_model)
        logger.info("[CONFIG] whisper_device      = %s", self.whisper_device)
        logger.info("[CONFIG] whisper_compute_type= %s", self.whisper_compute_type)
        logger.info("[CONFIG] audio_sample_rate   = %d", self.audio_sample_rate)
        logger.info("[CONFIG] audio_chunk_seconds = %d", self.audio_chunk_seconds)
        logger.info(
            "[CONFIG] audio_input_device  = %s",
            self.audio_input_device if self.audio_input_device is not None else "default",
        )


# Singleton — imported by every other module
settings = Settings()
settings.log_active_config()
