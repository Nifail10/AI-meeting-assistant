"""
meetings_cli.py — Command-line interface for browsing saved meetings

Usage:
  python meetings_cli.py list
  python meetings_cli.py show <meeting_id>
  python meetings_cli.py search <keyword>
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from storage.meeting_store import MeetingStore


def cmd_list() -> None:
    """List all saved meetings in a formatted table."""
    store = MeetingStore()
    meetings = store.list_meetings()
    if not meetings:
        print("No meetings found.")
        return

    print(f"\n{'#':<4} {'ID':<10} {'Started':<22} {'Duration':<10} {'Keypoints':<12} {'Transcript'}")
    print("─" * 75)
    for i, m in enumerate(meetings, 1):
        mid = m["meeting_id"][:8]
        started = m["started_at"][:19].replace("T", " ") + " UTC"
        duration = f"{m['duration_secs']}s"
        kps = f"{m['keypoint_count']} keypoints"
        lines = f"{m['transcript_count']} lines"
        print(f"{i:<4} {mid:<10} {started:<22} {duration:<10} {kps:<12} {lines}")
    print()


def cmd_show(meeting_id: str) -> None:
    """Show a full formatted report for a meeting."""
    store = MeetingStore()
    record = store.load_meeting(meeting_id)
    if not record:
        print(f"Meeting '{meeting_id}' not found.")
        sys.exit(1)

    print()
    print("══════════════════════════════════════")
    print("MEETING REPORT")
    print("══════════════════════════════════════")
    print(f"ID:        {record.meeting_id}")
    print(f"Started:   {record.started_at}")
    print(f"Ended:     {record.ended_at}")
    print(f"Duration:  {record.duration_secs}s")

    print()
    print("── SUMMARY ───────────────────────────")
    print(record.summary or "(none)")

    print()
    print("── KEY POINTS ────────────────────────")
    if record.keypoints:
        for kp in record.keypoints:
            print(f"[{kp.category}] {kp.timestamp} — {kp.text}")
    else:
        print("(none)")

    print()
    print("── QUESTIONS ─────────────────────────")
    print(record.questions or "(none)")

    print()
    print(f"── TRANSCRIPT ({len(record.transcript)} lines) ──────────")
    for entry in record.transcript:
        print(f"[{entry.timestamp}] {entry.text}")
    print()


def cmd_search(keyword: str) -> None:
    """Search meetings for a keyword and display results."""
    store = MeetingStore()
    results = store.search_meetings(keyword)
    if not results:
        print(f"No meetings found containing '{keyword}'.")
        return

    print(f"Found {len(results)} meeting(s) containing '{keyword}':\n")
    for r in results:
        mid = r.meeting_id[:8]
        preview = (r.summary[:100] + "…") if len(r.summary) > 100 else (r.summary or "(no summary)")
        print(f"  {mid} | {r.started_at} | {r.duration_secs}s")
        print(f"    Preview: {preview}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python meetings_cli.py list")
        print("  python meetings_cli.py show <meeting_id>")
        print("  python meetings_cli.py search <keyword>")
        sys.exit(1)

    command = sys.argv[1].lower()
    if command == "list":
        cmd_list()
    elif command == "show":
        if len(sys.argv) < 3:
            print("Usage: python meetings_cli.py show <meeting_id>")
            sys.exit(1)
        cmd_show(sys.argv[2])
    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python meetings_cli.py search <keyword>")
            sys.exit(1)
        cmd_search(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
