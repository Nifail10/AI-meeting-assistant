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
from datetime import datetime
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
        pipe.register_callback(lambda entry: print(entry["text"]))
        pipe.start()       # blocks until stop() is called
    """

    def __init__(self) -> None:
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._capture = AudioCapture(self._audio_queue)
        self._transcriber = Transcriber()

        self._callbacks: list[Callable[[dict], None]] = []
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        # Accumulated transcript entries for the session
        self._transcript_buffer: list[dict] = []

    # ── Public API ───────────────────────────────────────────────

    def register_callback(self, fn: Callable[[dict], None]) -> None:
        """Register a function to be called with each transcript entry.

        Each entry is a dict with keys ``"text"`` (str) and
        ``"timestamp"`` (ISO 8601 UTC str).  This is the extension point
        for future modules (summariser, database writer, UI, etc.).
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
        logger.info("[PIPELINE] Pipeline started.")

    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self._capture.stop()
        self._stop_event.set()

        if self._worker is not None:
            self._worker.join(timeout=5)
            self._worker = None

        logger.info("[PIPELINE] Pipeline stopped.")

    def get_transcript(self) -> list[dict]:
        """Return a shallow copy of the accumulated transcript buffer.

        Each entry is ``{"text": str, "timestamp": str}``.  This is the
        primary interface Stage 2 processors will use to read the
        session transcript so far.
        """
        return list(self._transcript_buffer)

    # ── Internal ─────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Continuously pull audio chunks, transcribe, and dispatch."""
        logger.info("[PIPELINE] Transcription worker started.")
        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Ignore the sentinel empty array used during shutdown
            if chunk.shape[0] == 0:
                continue

            logger.debug(
                "[PIPELINE] Pulled chunk from queue at %s",
                datetime.utcnow().isoformat() + "Z",
            )

            text = self._transcriber.transcribe(chunk)
            if text:
                entry: dict = {
                    "text": text,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                self._transcript_buffer.append(entry)

                for cb in self._callbacks:
                    try:
                        cb(entry)
                    except Exception:
                        logger.exception(
                            "[PIPELINE] Error in transcript callback '%s'",
                            getattr(cb, "__name__", repr(cb)),
                        )

        logger.info("[PIPELINE] Transcription worker exiting.")
