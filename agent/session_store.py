"""Persistence helpers for conversation contents across CLI resumes."""
from __future__ import annotations

import json
from pathlib import Path

from google.genai import types


def session_path(strategy_path: Path) -> Path:
    return strategy_path.with_suffix(".session.json")


def load_session_contents(strategy_path: Path) -> list[types.Content] | None:
    path = session_path(strategy_path)
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    contents = payload.get("contents", [])
    return [types.Content.model_validate(item) for item in contents]


def save_session_contents(strategy_path: Path, contents: list[types.Content]) -> None:
    path = session_path(strategy_path)
    payload = {
        "contents": [content.model_dump(exclude_none=True) for content in contents],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def last_model_text(contents: list[types.Content]) -> str:
    for content in reversed(contents):
        if content.role != "model":
            continue
        texts = [part.text for part in content.parts if part.text]
        if texts:
            return "\n".join(texts).strip()
    return ""


def prune_session_contents(
    contents: list[types.Content],
    keep_recent_messages: int = 8,
) -> list[types.Content]:
    if keep_recent_messages <= 0:
        return []
    if len(contents) <= keep_recent_messages:
        return list(contents)
    return list(contents[-keep_recent_messages:])
