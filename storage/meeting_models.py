"""
storage/meeting_models.py — Pydantic data models for meeting persistence

Defines the structured data models used to store, load, and search
meeting records. Designed to be extended in Stage 4 with embeddings,
tags, and UI metadata without breaking existing records.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
import uuid


class TranscriptEntry(BaseModel):
    """A single transcribed speech segment."""
    timestamp: str  # ISO 8601 UTC string
    text: str       # transcribed speech text


class Keypoint(BaseModel):
    """A classified key point extracted from the transcript."""
    timestamp: str  # ISO 8601 UTC string when it was said
    category: str   # one of: ACTION, DECISION, RISK, DEADLINE
    text: str       # full LLM classification result string


class MeetingRecord(BaseModel):
    """Complete record of a single meeting session."""
    meeting_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str       # ISO 8601 UTC, when recording started
    ended_at: str         # ISO 8601 UTC, when recording stopped
    duration_secs: float  # total session duration in seconds
    summary: str = ""     # LLM-generated summary, empty if none
    questions: str = ""   # LLM-generated questions, empty if none
    keypoints: list[Keypoint] = Field(default_factory=list)
    transcript: list[TranscriptEntry] = Field(default_factory=list)

    # Reserved for future stages — do not use yet
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def to_filename(self) -> str:
        """Return a filesystem-safe filename based on started_at timestamp."""
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
        formatted = dt.strftime("%Y-%m-%d_%H-%M")
        return f"{formatted}_{self.meeting_id[:8]}.json"
