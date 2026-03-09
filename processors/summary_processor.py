"""
processors/summary_processor.py — End-of-session meeting summariser

Called once after pipeline.stop(). Reads the full session transcript
and generates a concise structured summary using blocking LLM inference.
"""

from __future__ import annotations

import logging

from processors.llm_engine import LLMEngine

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """\
You are summarizing a meeting. Write exactly 4 to 6 bullet points.
Each bullet must describe a specific topic discussed, decision made, or outcome agreed.
Be direct and factual. Do not use filler phrases like "the team discussed".

Transcript:
{transcript}

Summary (bullet points only):
"""


class SummaryProcessor:

    def __init__(self, engine: LLMEngine) -> None:
        self._engine = engine

    def on_session_end(self, transcript: list[dict]) -> str:
        """Generate and print a meeting summary. Blocks until complete."""
        if not transcript:
            logger.warning("[SUMMARY] Transcript is empty — skipping summary.")
            return ""

        transcript_text = "\n".join(
            f"[{e['timestamp']}] {e['text']}" for e in transcript
        )
        if len(transcript_text) > 3000:
            transcript_text = (
                "[transcript truncated to fit context]\n"
                + transcript_text[-3000:]
            )

        prompt = SUMMARY_PROMPT.format(transcript=transcript_text)
        logger.info("[SUMMARY] Generating meeting summary…")
        result = self._engine.infer(prompt)

        if result:
            print("\n" + "─" * 60)
            print("[SUMMARY]")
            print(result)
            print("─" * 60 + "\n")
            return result
        else:
            logger.warning("[SUMMARY] Model returned an empty response.")
            return ""
