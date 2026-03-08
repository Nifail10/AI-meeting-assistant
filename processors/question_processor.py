"""
processors/question_processor.py — End-of-session clarifying question generator

Called once after pipeline.stop(). Generates 3–5 targeted questions
about unclear decisions, missing owners, or unresolved commitments.
"""

from __future__ import annotations

import logging

from processors.llm_engine import LLMEngine

logger = logging.getLogger(__name__)

QUESTIONS_PROMPT = """\
You are reviewing a meeting transcript to find gaps and ambiguities.
Generate between 3 and 5 clarifying questions that need to be answered after this meeting.
Focus on: missing owners for tasks, deadlines that were not set, decisions that were not finalized,
or commitments that were made vaguely.
Number each question. Be specific — reference the actual content of the transcript.

Transcript:
{transcript}

Clarifying Questions:
"""


class QuestionProcessor:

    def __init__(self, engine: LLMEngine) -> None:
        self._engine = engine

    def on_session_end(self, transcript: list[dict]) -> None:
        """Generate and print clarifying questions. Blocks until complete."""
        if not transcript:
            logger.warning("[QUESTIONS] Transcript is empty — skipping.")
            return

        transcript_text = "\n".join(
            f"[{e['timestamp']}] {e['text']}" for e in transcript
        )
        if len(transcript_text) > 3000:
            transcript_text = (
                "[transcript truncated to fit context]\n"
                + transcript_text[-3000:]
            )

        prompt = QUESTIONS_PROMPT.format(transcript=transcript_text)
        logger.info("[QUESTIONS] Generating clarifying questions…")
        result = self._engine.infer(prompt)

        if result:
            print("\n" + "─" * 60)
            print("[QUESTIONS]")
            print(result)
            print("─" * 60 + "\n")
        else:
            logger.warning("[QUESTIONS] Model returned an empty response.")
