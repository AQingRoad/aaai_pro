from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any


DETAIL_KEYS = (
    "Artist",
    "Composer",
    "Label",
    "Manufacturer",
    "Original Release Date",
    "Run time",
    "Number of discs",
    "Format",
)


def compact(text: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " [TRUNCATED]"


def as_text_list(value: Any, limit: int = 8) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            text = compact(item, 500)
            if text:
                out.append(text)
            if len(out) >= limit:
                break
        return out
    return [compact(value, 500)]


def category_label(category: str) -> str:
    return category.replace("_", " ").replace("And", "and")


def parse_details(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_item_map(item_info: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    item_map: dict[int, dict[str, Any]] = {}
    for item in item_info:
        item_id = item.get("item_id")
        if item_id is not None:
            item_map[int(item_id)] = dict(item)
    return item_map


def build_item_text(item: Mapping[str, Any] | None, title: str, max_chars: int) -> str:
    if not item:
        return compact(title, max_chars)

    parts: list[str] = []
    for key in ("title", "main_category", "store"):
        value = compact(item.get(key), 300)
        if value:
            parts.append(value)
    categories = " > ".join(as_text_list(item.get("categories"), limit=6))
    if categories:
        parts.append(f"Categories: {categories}")
    features = "; ".join(as_text_list(item.get("features"), limit=8))
    if features:
        parts.append(f"Features: {features}")
    description = " ".join(as_text_list(item.get("description"), limit=2))
    if description:
        parts.append(f"Description: {description}")
    if not parts:
        parts.append(title)
    return compact(" ".join(parts), max_chars)


def _item_stats(item: Mapping[str, Any]) -> str:
    avg = item.get("average_rating")
    count = item.get("rating_number")
    if avg in (None, "", 0) and count in (None, "", 0):
        return ""
    pieces = []
    if avg not in (None, ""):
        pieces.append(f"avg_rating={avg}")
    if count not in (None, ""):
        pieces.append(f"rating_count={count}")
    return ", ".join(pieces)


def build_history_item_metadata(item: Mapping[str, Any] | None, max_chars: int) -> str:
    if not item:
        return ""

    parts: list[str] = []
    store = compact(item.get("store"), 180)
    if store:
        parts.append(f"Store/artist/format: {store}")

    categories = " > ".join(as_text_list(item.get("categories"), limit=4))
    if categories:
        parts.append(f"Categories: {categories}")

    features = "; ".join(as_text_list(item.get("features"), limit=3))
    if features:
        parts.append(f"Features: {features}")

    description = " ".join(as_text_list(item.get("description"), limit=1))
    if description:
        parts.append(f"Description: {description}")

    details = parse_details(item.get("details"))
    detail_parts = []
    for key in DETAIL_KEYS:
        value = compact(details.get(key), 120)
        if value:
            detail_parts.append(f"{key}={value}")
    if detail_parts:
        parts.append("Details: " + "; ".join(detail_parts[:5]))

    stats = _item_stats(item)
    if stats:
        parts.append(f"Catalog stats: {stats}")

    return compact("; ".join(parts), max_chars)


def history_text(
    category: str,
    titles: list[str],
    ratings: list[float],
    max_history_items: int,
    item_ids: list[int] | None = None,
    item_map: Mapping[int, Mapping[str, Any]] | None = None,
    metadata_mode: str = "none",
    max_item_chars: int = 320,
) -> str:
    if max_history_items > 0:
        titles = titles[-max_history_items:]
        ratings = ratings[-max_history_items:]
        if item_ids is not None:
            item_ids = item_ids[-max_history_items:]

    if metadata_mode == "none":
        entries = []
        for title, rating in zip(titles, ratings):
            title = compact(title, 240)
            if title:
                entries.append(f"{title} ({float(rating):g} stars)")

        history = "; ".join(entries)
        return (
            f"This user's Amazon {category_label(category)} interaction history over time is listed below. "
            f"{history}."
        )

    entries = []
    for pos, (title, rating) in enumerate(zip(titles, ratings), start=1):
        title = compact(title, 240)
        if not title:
            continue
        entry = f"{pos}. {title} ({float(rating):g} stars)"
        if metadata_mode != "none" and item_ids is not None and item_map is not None and pos - 1 < len(item_ids):
            metadata = build_history_item_metadata(item_map.get(int(item_ids[pos - 1])), max_item_chars)
            if metadata:
                entry = f"{entry}; {metadata}"
        entries.append(entry)

    history = "\n".join(entries)
    return (
        f"This user's Amazon {category_label(category)} interaction history over time is listed below.\n"
        f"{history}"
    ).strip()
