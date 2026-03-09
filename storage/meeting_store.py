"""
storage/meeting_store.py — JSON-based meeting persistence service

Saves, loads, lists, and searches meeting records stored as JSON files
in the meetings/ directory. Designed to be used by main.py at session
end and by the CLI tool for retrospective access.
Future stages can extend this class or swap the backend without
changing the interface.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from storage.meeting_models import MeetingRecord

logger = logging.getLogger(__name__)

MEETINGS_DIR = Path("meetings")


class MeetingStore:

    def __init__(self) -> None:
        """Create the meetings/ directory if it does not exist."""
        MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("[STORAGE] Meeting store initialized at %s", MEETINGS_DIR.resolve())

    def save_meeting(self, record: MeetingRecord) -> Path:
        """Serialize a MeetingRecord to JSON and write it to disk.

        Returns the Path of the saved file.
        """
        filename = record.to_filename()
        filepath = MEETINGS_DIR / filename
        filepath.write_text(
            json.dumps(record.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("[STORAGE] Meeting saved → %s", filepath)
        return filepath

    def list_meetings(self) -> list[dict]:
        """Return a list of all saved meetings as summary dicts.

        Each dict contains: meeting_id, started_at, ended_at,
        duration_secs, filename, keypoint_count, transcript_count.
        Sorted by started_at descending (newest first).
        """
        results: list[dict] = []
        for path in MEETINGS_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                results.append({
                    "meeting_id": data["meeting_id"],
                    "started_at": data["started_at"],
                    "ended_at": data["ended_at"],
                    "duration_secs": data["duration_secs"],
                    "filename": path.name,
                    "keypoint_count": len(data.get("keypoints", [])),
                    "transcript_count": len(data.get("transcript", [])),
                })
            except Exception:
                logger.warning("[STORAGE] Skipping corrupted file: %s", path)

        results.sort(key=lambda r: r["started_at"], reverse=True)
        logger.info("[STORAGE] Found %d meeting(s).", len(results))
        return results

    def load_meeting(self, meeting_id: str) -> MeetingRecord | None:
        """Load and return a MeetingRecord by meeting_id prefix or full id.

        Searches all JSON files for a matching meeting_id (prefix match
        supported — e.g. "a1b2c3d4" matches "a1b2c3d4-xxxx-...").
        Returns None if not found.
        """
        for path in MEETINGS_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data["meeting_id"].startswith(meeting_id):
                    return MeetingRecord(**data)
            except Exception:
                continue
        logger.warning("[STORAGE] Meeting '%s' not found.", meeting_id)
        return None

    def search_meetings(self, keyword: str) -> list[MeetingRecord]:
        """Search all meeting records for a keyword.

        Searches inside: summary, questions, transcript text,
        and keypoint text. Case-insensitive.
        Returns list of matching MeetingRecord objects.
        """
        keyword_lower = keyword.lower()
        results: list[MeetingRecord] = []
        for path in MEETINGS_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                search_text = " ".join([
                    data.get("summary", ""),
                    data.get("questions", ""),
                    " ".join(e["text"] for e in data.get("transcript", [])),
                    " ".join(kp["text"] for kp in data.get("keypoints", [])),
                ])
                if keyword_lower in search_text.lower():
                    results.append(MeetingRecord(**data))
            except Exception:
                logger.warning("[STORAGE] Skipping corrupted file: %s", path)

        logger.info("[STORAGE] Search '%s' → %d result(s).", keyword, len(results))
        return results
