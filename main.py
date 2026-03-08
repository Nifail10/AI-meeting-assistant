"""
main.py — Entry point for the AI Meeting Assistant (Stage 1)

Starts the real-time transcription pipeline and prints every recognised
transcript to the terminal.  Press Ctrl+C to stop.
"""

from __future__ import annotations

import logging
import sys
import time

# Configure logging BEFORE any project imports so that config.log_active_config()
# output is visible on first import.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

from core.pipeline import Pipeline  # noqa: E402
from processors.llm_engine import LLMEngine  # noqa: E402
from processors.keypoint_processor import KeypointProcessor  # noqa: E402
from processors.summary_processor import SummaryProcessor  # noqa: E402
from processors.question_processor import QuestionProcessor  # noqa: E402

# Path to the Mistral GGUF model file.
# Download: mistral-7b-instruct-v0.2.Q4_K_M.gguf from TheBloke on HuggingFace
# Place in the models/ directory and update this path if needed.
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"


def on_transcript(entry: dict) -> None:
    """Simple callback that prints each transcript entry to stdout."""
    logger.info("[TRANSCRIPT] %s — %s", entry["timestamp"], entry["text"])


def main() -> None:
    # ── Stage 2: LLM engine and processors ──────────────────────────
    engine    = LLMEngine(MODEL_PATH)
    keypoint  = KeypointProcessor(engine)
    summary   = SummaryProcessor(engine)
    questions = QuestionProcessor(engine)

    # ── Stage 1: pipeline (EXISTING — do not change) ────────────────
    pipeline = Pipeline()
    pipeline.register_callback(on_transcript)

    # ── Stage 2: register real-time processor ───────────────────────
    keypoint.register(pipeline)

    logger.info("AI Meeting Assistant — Stage 1 + 2")
    logger.info("Listening… (press Ctrl+C to stop)")

    pipeline.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Stopping…")
        pipeline.stop()

        # ── Stage 2: end-of-session processing ──────────────────────
        transcript = pipeline.get_transcript()
        summary.on_session_end(transcript)
        questions.on_session_end(transcript)
        engine.shutdown()

        logger.info("Session ended.")
        sys.exit(0)


if __name__ == "__main__":
    main()
