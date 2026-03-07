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


def on_transcript(entry: dict) -> None:
    """Simple callback that prints each transcript entry to stdout."""
    logger.info("[TRANSCRIPT] %s — %s", entry["timestamp"], entry["text"])


def main() -> None:
    pipeline = Pipeline()
    pipeline.register_callback(on_transcript)

    logger.info("AI Meeting Assistant — Stage 1")
    logger.info("Listening… (press Ctrl+C to stop)")

    pipeline.start()

    try:
        # Keep the main thread alive; all work happens in daemon threads
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Stopping…")
        pipeline.stop()
        logger.info("Session ended.")
        sys.exit(0)


if __name__ == "__main__":
    main()
