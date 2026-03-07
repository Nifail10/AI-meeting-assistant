"""
core/transcriber.py — faster-whisper speech-to-text engine

Loads a CTranslate2-optimised Whisper model and exposes a simple
``transcribe(audio_np)`` method that returns the recognised text.
"""

from __future__ import annotations

import logging

import numpy as np
from faster_whisper import WhisperModel

from core.config import settings

logger = logging.getLogger(__name__)


class Transcriber:
    """Wraps a faster-whisper ``WhisperModel`` for single-call transcription.

    The model is loaded once at construction time and reused for every
    subsequent call.
    """

    def __init__(self) -> None:
        logger.info(
            "[STT] Loading Whisper model  (model=%s, device=%s, compute=%s) …",
            settings.whisper_model,
            settings.whisper_device,
            settings.whisper_compute_type,
        )
        try:
            self._model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
            logger.info("[STT] Whisper model loaded.")
        except Exception as exc:
            logger.error(
                "[STT] Failed to load Whisper model '%s' on device '%s': %s",
                settings.whisper_model,
                settings.whisper_device,
                exc,
            )
            raise

    def transcribe(self, audio_np: np.ndarray) -> str:
        """Transcribe a float32 audio array (16 kHz, mono) to text.

        Parameters
        ----------
        audio_np:
            1-D NumPy array, dtype float32, sample rate 16 000.

        Returns
        -------
        str
            Joined segment texts, or an empty string if no speech was
            detected.
        """
        try:
            logger.debug(
                "[STT] Transcribing chunk  shape=%s  dtype=%s",
                audio_np.shape,
                audio_np.dtype,
            )

            segments, _info = self._model.transcribe(
                audio_np,
                beam_size=5,
                vad_filter=True,
                language="en",
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text
        except Exception as exc:
            logger.error("[STT] Transcription failed: %s", exc)
            return ""
