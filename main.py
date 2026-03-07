"""
main.py — Entry point for the AI Meeting Assistant (Stage 1)

Starts the real-time transcription pipeline and prints every recognised
transcript to the terminal.  Press Ctrl+C to stop.
"""

from __future__ import annotations

import logging
import sys

from core.pipeline import Pipeline


def on_transcript(text: str) -> None:
    """Simple callback that prints each transcript to stdout."""
    print(f"[TRANSCRIPT] {text}")


def main() -> None:
    # Basic logging so internal messages are visible
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = Pipeline()
    pipeline.register_callback(on_transcript)

    print("🎙  AI Meeting Assistant — Stage 1")
    print("    Listening… (press Ctrl+C to stop)\n")

    pipeline.start()

    try:
        # Keep the main thread alive; all work happens in daemon threads
        while True:
            pass
    except KeyboardInterrupt:
        print("\n⏹  Stopping…")
        pipeline.stop()
        print("Session ended.")
        sys.exit(0)


if __name__ == "__main__":
    main()
