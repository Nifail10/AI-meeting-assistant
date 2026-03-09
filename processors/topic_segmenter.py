"""
processors/topic_segmenter.py — LLM-based meeting topic segmenter

Analyzes a completed meeting transcript and divides it into topic
segments using a sliding window approach. Each window pair is evaluated
by the LLM to detect topic boundaries.

Algorithm:
  1. Divide transcript into windows of `window_size` entries each.
  2. For each consecutive window pair, ask the LLM if the topic changed.
  3. If topic changed → close current segment, start a new one.
  4. If no change → merge window into current segment.
  5. Returns a list of Segment objects with titles and entry ranges.
"""

from __future__ import annotations

import logging

from processors.llm_engine import LLMEngine
from storage.meeting_models import Segment, TranscriptEntry

logger = logging.getLogger(__name__)

SEGMENT_PROMPT = """\
You are analyzing a meeting transcript to detect topic changes.

Previous window:
{prev_window}

New window:
{curr_window}

Does the new window introduce a clearly different topic from the previous window?

Rules:
- If the topic is clearly different: respond with "NEW: <3 to 5 word topic title>"
- If the topic is the same or continues: respond with "CONTINUE"
- Do NOT add any explanation, punctuation, or extra text.
- Examples of valid responses:
    NEW: API implementation discussion
    NEW: Deployment timeline planning
    NEW: Budget approval decision
    CONTINUE

Your response:
"""


class TopicSegmenter:

    def __init__(self, engine: LLMEngine, window_size: int = 4) -> None:
        """
        Parameters
        ----------
        engine:
            Shared LLMEngine instance from main.py.
        window_size:
            Number of transcript entries per evaluation window.
            Default 4 — balances context vs inference speed.
        """
        self._engine = engine
        self._window_size = window_size
        logger.info(
            "[SEGMENTER] Initialized with window_size=%d", window_size
        )

    def segment(self, transcript: list[dict]) -> list[Segment]:
        """
        Divide a transcript into topic segments.

        Parameters
        ----------
        transcript:
            List of dicts from pipeline.get_transcript().
            Each dict: {"text": str, "timestamp": str}

        Returns
        -------
        list[Segment]
            Ordered list of topic segments. Returns a single segment
            containing the full transcript if it is too short to segment.
        """
        logger.info(
            "[SEGMENTER] Starting segmentation of %d entries…",
            len(transcript),
        )

        if len(transcript) == 0:
            logger.warning("[SEGMENTER] Empty transcript — no segments.")
            return []

        entries = [
            TranscriptEntry(timestamp=e["timestamp"], text=e["text"])
            for e in transcript
        ]

        if len(entries) <= self._window_size:
            logger.info(
                "[SEGMENTER] Transcript too short to segment — returning single segment."
            )
            single = self._finalize_segment(entries, "Full Meeting")
            return [single]

        windows = [
            entries[i : i + self._window_size]
            for i in range(0, len(entries), self._window_size)
        ]

        segments: list[Segment] = []
        current_entries: list[TranscriptEntry] = list(windows[0])
        current_title: str = "Opening"

        for i in range(1, len(windows)):
            prev_text = self._build_window_text(windows[i - 1])
            curr_text = self._build_window_text(windows[i])

            changed, new_title = self._detect_topic_change(
                prev_text, curr_text
            )

            if changed:
                segment = self._finalize_segment(
                    current_entries, current_title
                )
                segments.append(segment)
                logger.info(
                    "[SEGMENTER] New segment: '%s' (%d entries)",
                    segment.title,
                    segment.entry_count,
                )
                current_entries = list(windows[i])
                current_title = new_title
            else:
                current_entries.extend(windows[i])

        last_segment = self._finalize_segment(
            current_entries, current_title
        )
        segments.append(last_segment)
        logger.info(
            "[SEGMENTER] Final segment: '%s' (%d entries)",
            last_segment.title,
            last_segment.entry_count,
        )

        logger.info(
            "[SEGMENTER] Segmentation complete — %d segment(s).",
            len(segments),
        )
        return segments

    def _build_window_text(self, entries: list[TranscriptEntry]) -> str:
        """Format a list of TranscriptEntry objects as readable text."""
        return "\n".join(f"[{e.timestamp}] {e.text}" for e in entries)

    def _detect_topic_change(
        self, prev_text: str, curr_text: str
    ) -> tuple[bool, str]:
        """
        Ask the LLM if the current window introduces a new topic.

        Returns
        -------
        tuple[bool, str]
            (True, "topic title") if topic changed
            (False, "")           if topic continues
        """
        prompt = SEGMENT_PROMPT.format(
            prev_window=prev_text,
            curr_window=curr_text,
        )

        result = self._engine.infer(prompt)
        result = result.strip()

        if not result:
            logger.warning(
                "[SEGMENTER] Empty LLM response — assuming CONTINUE."
            )
            return False, ""

        result_upper = result.upper()

        if result_upper.startswith("NEW:"):
            title = result[4:].strip()
            if not title:
                title = "New Topic"
            logger.debug("[SEGMENTER] Topic change detected: '%s'", title)
            return True, title

        if result_upper == "CONTINUE" or result_upper.startswith("CONTINUE"):
            return False, ""

        logger.warning(
            "[SEGMENTER] Unexpected LLM response: '%s' — assuming CONTINUE.",
            result,
        )
        return False, ""

    def _finalize_segment(
        self,
        entries: list[TranscriptEntry],
        title: str,
    ) -> Segment:
        """Build a Segment from a list of accumulated TranscriptEntry items."""
        return Segment(
            title=title,
            start_timestamp=entries[0].timestamp,
            end_timestamp=entries[-1].timestamp,
            entries=entries,
            entry_count=len(entries),
        )
