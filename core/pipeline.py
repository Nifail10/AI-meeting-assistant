"""
core/pipeline.py — Producer-consumer wiring for the transcription pipeline

Connects :class:`AudioCapture` (producer) to :class:`Transcriber` (consumer)
via a thread-safe queue.  External code registers callbacks through
:meth:`Pipeline.register_callback` — this is the **only** extension point
needed for Stage 1 and is designed for future modules (LLM, DB, UI) to
subscribe to transcripts without touching audio or STT logic.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

import numpy as np

from core.audio_capture import AudioCapture
from core.config import settings  # noqa: F401 – available for future use
from core.transcriber import Transcriber

logger = logging.getLogger(__name__)


class Pipeline:
    """Wires mic capture → transcription → callbacks.

    Usage::

        pipe = Pipeline()
        pipe.register_callback(lambda text: print(text))
        pipe.start()       # blocks until stop() is called
    """

    def __init__(self) -> None:
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._capture = AudioCapture(self._audio_queue)
        self._transcriber = Transcriber()

        self._callbacks: list[Callable[[str], None]] = []
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

    # ── Public API ───────────────────────────────────────────────

    def register_callback(self, fn: Callable[[str], None]) -> None:
        """Register a function to be called with each transcript string.

        This is the extension point for future modules (summariser,
        database writer, UI, etc.).
        """
        self._callbacks.append(fn)

    def start(self) -> None:
        """Start audio capture and the transcription worker thread."""
        self._stop_event.clear()

        # Start the consumer first so it's ready when audio arrives
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="transcription-worker",
            daemon=True,
        )
        self._worker.start()

        # Start producing audio chunks
        self._capture.start()
        logger.info("Pipeline started.")

    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self._capture.stop()
        self._stop_event.set()

        # Unblock the worker if it's sitting on queue.get()
        self._audio_queue.put(np.empty(0, dtype=np.float32))

        if self._worker is not None:
            self._worker.join(timeout=5)
            self._worker = None

        logger.info("Pipeline stopped.")

    # ── Internal ─────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Continuously pull audio chunks, transcribe, and dispatch."""
        logger.info("Transcription worker started.")
        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Ignore the sentinel empty array used during shutdown
            if chunk.shape[0] == 0:
                continue

            text = self._transcriber.transcribe(chunk)
            if text:
                for cb in self._callbacks:
                    try:
                        cb(text)
                    except Exception:
                        logger.exception("Error in transcript callback")

        logger.info("Transcription worker exiting.")
