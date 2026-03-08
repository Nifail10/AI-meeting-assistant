"""
processors/keypoint_processor.py — Real-time transcript key point detector

Subscribes to the pipeline callback and classifies each transcript
sentence into one of: ACTION, DECISION, RISK, DEADLINE, NONE.
Uses non-blocking submission to LLMEngine so the pipeline is never stalled.
NONE results are silently discarded.
"""

from __future__ import annotations

import logging

from processors.llm_engine import LLMEngine

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = """\
You are analyzing a single sentence from a meeting transcript.
Classify it into exactly one of these categories:
  ACTION   - a task or to-do assigned to someone
  DECISION - a choice or conclusion that was reached
  RISK     - a problem, concern, or blocker raised
  DEADLINE - a date, time constraint, or schedule mentioned
  NONE     - ordinary conversation with no meeting significance

Rules:
- Respond with the category name, a colon, then one short explanation.
- Do not add any other text before or after.
- Example: ACTION: John will send the report by Friday.

Sentence: "{sentence}"
"""


class KeypointProcessor:

    def __init__(self, engine: LLMEngine) -> None:
        self._engine = engine

    def register(self, pipeline) -> None:
        """Attach to the pipeline. Call this before pipeline.start()."""
        pipeline.register_callback(self._on_transcript)
        logger.info("[KEYPOINT] Processor registered with pipeline.")

    def _on_transcript(self, entry: dict) -> None:
        """Pipeline callback — must return quickly, never block."""
        text: str = entry["text"].strip()
        timestamp: str = entry["timestamp"]
        if not text:
            return
        prompt = CLASSIFY_PROMPT.format(sentence=text)
        self._engine.submit(
            prompt,
            callback=lambda result: self._on_result(result, timestamp),
        )

    def _on_result(self, result: str, timestamp: str) -> None:
        """Called by the LLM async worker when classification is ready."""
        if not result:
            return
        category = result.split(":")[0].strip().upper()
        if category != "NONE":
            logger.info("[KEYPOINT] %s | %s", timestamp, result)
