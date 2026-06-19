from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path, limit: int = 0) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if limit and idx > limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 16].rstrip() + "\n[TRUNCATED]"
