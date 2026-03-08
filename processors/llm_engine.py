"""
processors/llm_engine.py — Shared LLM inference engine

Loads a GGUF model via llama-cpp-python and exposes two interfaces:
  infer(prompt)        — blocking inference, for end-of-session processors
  submit(prompt, cb)   — non-blocking, queues work to a background thread
                         so pipeline callbacks never stall transcription

The engine is instantiated once in main.py and injected into processors.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LLMEngine:
    """Wraps a llama-cpp-python Llama instance with sync and async inference."""

    def __init__(self, model_path: str) -> None:
        """Load the GGUF model and start the async inference worker."""
        logger.info("[LLM] Loading model from %s …", model_path)
        try:
            self._llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=-1,
                verbose=False,
            )
            logger.info("[LLM] Model loaded successfully.")
        except Exception as exc:
            logger.error("[LLM] Failed to load model: %s", exc)
            raise

        self._infer_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._async_worker,
            name="llm-worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.debug("[LLM] Async worker started.")

    def infer(self, prompt: str) -> str:
        """Run synchronous inference. Blocks until the model responds."""
        try:
            response = self._llm(
                prompt,
                max_tokens=512,
                stop=["</s>", "\n\n\n"],
                echo=False,
            )
            return response["choices"][0]["text"].strip()
        except Exception as exc:
            logger.error("[LLM] Inference error: %s", exc)
            return ""

    def submit(self, prompt: str, callback: Callable[[str], None]) -> None:
        """Queue a prompt for async inference. Returns immediately."""
        self._infer_queue.put((prompt, callback))

    def shutdown(self) -> None:
        """Signal the async worker to stop and wait for it to exit."""
        logger.info("[LLM] Shutting down engine…")
        self._stop_event.set()
        self._infer_queue.put((None, None))
        self._worker_thread.join(timeout=10)
        logger.info("[LLM] Engine shut down.")

    def _async_worker(self) -> None:
        """Background thread: drain the inference queue until stopped."""
        logger.info("[LLM] Async worker running.")
        while not self._stop_event.is_set():
            try:
                item = self._infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            prompt, callback = item
            if prompt is None:
                break
            result = self.infer(prompt)
            if result and callback is not None:
                try:
                    callback(result)
                except Exception as exc:
                    logger.error("[LLM] Callback error: %s", exc)
        logger.info("[LLM] Async worker exiting.")
