"""
core/audio_capture.py — Microphone input → audio chunk queue

Opens a sounddevice InputStream at 16 kHz / mono / float32, accumulates
incoming frames in a NumPy buffer, and enqueues completed chunks (each
``chunk_seconds`` long) into a thread-safe ``queue.Queue``.
"""

from __future__ import annotations

import queue
import logging

import numpy as np
import sounddevice as sd

from core.config import settings

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures microphone audio and pushes fixed-length chunks to a queue.

    Parameters
    ----------
    audio_queue:
        Thread-safe queue that receives ``np.ndarray`` chunks
        (float32, shape ``(N,)``, sample rate 16 000).
    """

    def __init__(self, audio_queue: queue.Queue[np.ndarray]) -> None:
        self._queue = audio_queue
        self._sample_rate: int = settings.audio_sample_rate
        self._chunk_samples: int = settings.audio_sample_rate * settings.audio_chunk_seconds
        self._device: int | None = settings.audio_input_device

        # Accumulation buffer — grows until it reaches _chunk_samples
        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)

        self._stream: sd.InputStream | None = None

    # ── Public API ───────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone stream and begin capturing audio."""
        logger.info(
            "Starting audio capture  (device=%s, rate=%d, chunk=%ds)",
            self._device if self._device is not None else "default",
            self._sample_rate,
            settings.audio_chunk_seconds,
        )
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the microphone stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio capture stopped.")

    # ── Internal ─────────────────────────────────────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,       # noqa: ARG002
        time_info: object,  # noqa: ARG002
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice on every audio block (runs on a C thread)."""
        if status:
            logger.warning("Audio callback status: %s", status)

        # indata shape: (frames, 1) — flatten to 1-D
        self._buffer = np.append(self._buffer, indata[:, 0])

        # Flush complete chunks into the queue
        while self._buffer.shape[0] >= self._chunk_samples:
            chunk = self._buffer[: self._chunk_samples].copy()
            self._buffer = self._buffer[self._chunk_samples :]
            self._queue.put(chunk)
