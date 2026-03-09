"""
main.py — Entry point for the AI Meeting Assistant (Stage 1 + 2 + 3 + 3.5)

Starts the real-time transcription pipeline and prints every recognised
transcript to the terminal. Press Ctrl+C to stop.
"""

from __future__ import annotations

import logging
import sys
import os
import time

# Ensure project root is on Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging BEFORE any project imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Project imports
from core.pipeline import Pipeline
from processors.llm_engine import LLMEngine
from processors.keypoint_processor import KeypointProcessor
from processors.summary_processor import SummaryProcessor
from processors.question_processor import QuestionProcessor
from datetime import datetime, timezone
from storage.meeting_models import MeetingRecord, Keypoint, TranscriptEntry, Segment
from storage.meeting_store import MeetingStore
from processors.topic_segmenter import TopicSegmenter


# Path to the Mistral GGUF model file
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

meeting_store = MeetingStore()


def on_transcript(entry: dict) -> None:
    """Print transcript entries to the terminal."""
    logger.info("[TRANSCRIPT] %s — %s", entry["timestamp"], entry["text"])


def main() -> None:
    # ── Stage 2: initialise LLM engine and processors ──────────
    engine = LLMEngine(MODEL_PATH)
    keypoint = KeypointProcessor(engine)
    summary = SummaryProcessor(engine)
    questions = QuestionProcessor(engine)
    segmenter = TopicSegmenter(engine)

    # ── Stage 1: pipeline setup ────────────────────────────────
    pipeline = Pipeline()
    pipeline.register_callback(on_transcript)

    # ── Stage 2: register real-time processor ──────────────────
    keypoint.register(pipeline)

    logger.info("AI Meeting Assistant — Stage 1 + 2 + 3 + 3.5")
    logger.info("Listening… (press Ctrl+C to stop)")

    session_start = datetime.now(timezone.utc)
    pipeline.start()

    try:
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Stopping…")
        pipeline.stop()

        try:
            # ── Stage 2: run end-of-session processors ─────────────
            transcript = pipeline.get_transcript()
            summary_text = summary.on_session_end(transcript)
            questions_text = questions.on_session_end(transcript)
            segments = segmenter.segment(transcript)
            engine.shutdown()

            # ── Stage 3: persist meeting record ─────────────────────────
            session_end = datetime.now(timezone.utc)
            duration = (session_end - session_start).total_seconds()

            record = MeetingRecord(
                started_at=session_start.isoformat(),
                ended_at=session_end.isoformat(),
                duration_secs=round(duration, 2),
                summary=summary_text,
                questions=questions_text,
                segments=segments,
                keypoints=[
                    Keypoint(
                        timestamp=kp["timestamp"],
                        category=kp["category"],
                        text=kp["text"],
                    )
                    for kp in keypoint.get_keypoints()
                ],
                transcript=[
                    TranscriptEntry(
                        timestamp=entry["timestamp"],
                        text=entry["text"],
                    )
                    for entry in transcript
                ],
            )
            meeting_store.save_meeting(record)

        except KeyboardInterrupt:
            logger.warning("Interrupted during saving — shutting down.")
            engine.shutdown()

        logger.info("Session ended.")
        sys.exit(0)


if __name__ == "__main__":
    main()