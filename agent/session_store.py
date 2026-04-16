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
    keep_recent_turns: int = 4,
) -> list[types.Content]:
    if keep_recent_turns <= 0:
        return []
    turns = _split_into_turns(contents)
    if len(turns) <= keep_recent_turns:
        return list(contents)

    retained_turns = turns[-keep_recent_turns:]
    pruned: list[types.Content] = []
    for turn in retained_turns:
        pruned.extend(turn)
    return pruned


def _split_into_turns(contents: list[types.Content]) -> list[list[types.Content]]:
    turns: list[list[types.Content]] = []
    current: list[types.Content] = []

    for content in contents:
        if _is_user_text_message(content) and current:
            turns.append(current)
            current = [content]
        else:
            current.append(content)

    if current:
        turns.append(current)

    return turns


def _is_user_text_message(content: types.Content) -> bool:
    if content.role != "user":
        return False
    return any(part.text and part.text.strip() for part in content.parts)
