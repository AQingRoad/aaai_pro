#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXAMPLES = ROOT / "data/rrec_amazon/CDs_and_Vinyl/examples.jsonl"
DEFAULT_ITEM_INFO = ROOT / "github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "ablantion/datas/processed_datas/cds_query_ablation"

ABLATIONS = [
    ("title_only", ()),
    ("title_store", ("store",)),
    ("title_store_categories", ("store", "categories")),
    ("title_store_categories_features", ("store", "categories", "features")),
    (
        "title_store_categories_features_description",
        ("store", "categories", "features", "description"),
    ),
    (
        "full_compact_no_all_ratings",
        ("store", "categories", "features", "description", "details"),
    ),
]

ASIN_REF_RE = re.compile(r"(?:,\s*)?\b(?:Amazon\s+)?ASIN\s*[:#]?\s*[A-Z0-9]{10}\b", re.I)
RAW_ASIN_RE = re.compile(r"\bB0[0-9A-Z]{8}\b")


def compact(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " [TRUNCATED]"


def clean_asin(text: str) -> str:
    text = ASIN_REF_RE.sub("", text)
    text = RAW_ASIN_RE.sub("", text)
    text = re.sub(r"\s+([.;,])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def as_list(value: Any, limit: int, max_chars: int = 500) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [compact(value, max_chars)] if value.strip() else []
    if isinstance(value, list):
        out = []
        for item in value:
            text = compact(item, max_chars)
            if text:
                out.append(text)
            if len(out) >= limit:
                break
        return out
    return [compact(value, max_chars)]


def read_jsonl(path: Path, max_rows: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def item_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["item_id"]): row for row in rows if row.get("item_id") is not None}


def get_history_ids(example: dict[str, Any]) -> list[int]:
    raw = example.get("history_item_ids")
    if raw is None:
        raw = example.get("history_item_id", [])
    return [int(x) for x in raw]


def get_target_id(example: dict[str, Any]) -> int:
    if example.get("target_item_id") is not None:
        return int(example["target_item_id"])
    return int(example["item_id"])


def get_target_title(example: dict[str, Any], item: dict[str, Any] | None) -> str:
    title = example.get("target_item_title") or example.get("item_title")
    if title:
        return compact(title, 0)
    return compact(item.get("title") if item else "", 0)


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


def selected_details(item: dict[str, Any]) -> str:
    details = parse_details(item.get("details"))
    out = []

    for key in ("Artist", "Composer"):
        value = compact(details.get(key), 180)
        if value:
            out.append((key, value))

    label = compact(details.get("Label"), 180) or compact(details.get("Manufacturer"), 180)
    if label:
        out.append(("Label", label))

    for key in ("Original Release Date", "Genre", "Style"):
        value = compact(details.get(key), 180)
        if value:
            out.append((key, value))

    store = compact(item.get("store"), 300).lower()
    if "format:" not in store:
        fmt = compact(details.get("Media Format"), 180) or compact(details.get("Format"), 180)
        if fmt:
            out.append(("Format", fmt))

    runtime = compact(details.get("Run time"), 180) or compact(details.get("Runtime"), 180)
    if runtime:
        out.append(("Run time", runtime))

    discs = compact(details.get("Number of discs"), 60) or compact(details.get("Number Of Discs"), 60)
    if discs:
        out.append(("Number of discs", discs))

    language = compact(details.get("Language"), 120)
    if language:
        out.append(("Language", language))

    return "; ".join(f"{key}={value}" for key, value in out)


def target_text(item: dict[str, Any] | None, title: str) -> str:
    if not item:
        return compact(title, 0)

    parts = []
    for key in ("title", "main_category", "store"):
        value = compact(item.get(key), 0)
        if value:
            parts.append(value)

    categories = " > ".join(as_list(item.get("categories"), limit=6, max_chars=0))
    if categories:
        parts.append(f"Categories: {categories}")

    features = "; ".join(as_list(item.get("features"), limit=8, max_chars=0))
    if features:
        parts.append(f"Features: {features}")

    description = " ".join(as_list(item.get("description"), limit=2, max_chars=0))
    if description:
        parts.append(f"Description: {description}")

    return " ".join(parts) if parts else compact(title, 0)


def metadata_text(item: dict[str, Any] | None, fields: tuple[str, ...], max_item_chars: int) -> str:
    if not item or not fields:
        return ""

    parts = []
    if "store" in fields:
        store = compact(item.get("store"), max_item_chars)
        if store:
            parts.append(f"Store/artist/format: {store}")

    if "categories" in fields:
        categories = " > ".join(as_list(item.get("categories"), limit=4, max_chars=max_item_chars))
        if categories:
            parts.append(f"Categories: {categories}")

    if "features" in fields:
        features = "; ".join(as_list(item.get("features"), limit=3, max_chars=max_item_chars))
        if features:
            parts.append(f"Features: {features}")

    if "description" in fields:
        description = " ".join(as_list(item.get("description"), limit=1, max_chars=max_item_chars))
        if description:
            parts.append(f"Description: {description}")

    if "details" in fields:
        details = selected_details(item)
        if details:
            parts.append(f"Details: {details}")

    return "; ".join(parts)


def build_query(
    example: dict[str, Any],
    items: dict[int, dict[str, Any]],
    fields: tuple[str, ...],
    max_item_chars: int,
) -> str:
    history_ids = get_history_ids(example)
    history_titles = [str(x) for x in example.get("history_item_title", [])]
    lines = ["This user's Amazon CDs and Vinyl interaction history over time is listed below."]

    for pos, item_id in enumerate(history_ids, start=1):
        item = items.get(item_id)
        fallback_title = history_titles[pos - 1] if pos - 1 < len(history_titles) else f"item_{item_id}"
        title = compact(item.get("title") if item else fallback_title, max_item_chars)
        entry = f"{pos}. {title}" if title else f"{pos}. [missing title]"
        meta = metadata_text(item, fields, max_item_chars)
        if meta:
            entry = f"{entry}; {meta}"
        lines.append(entry)

    return clean_asin("\n".join(lines))


def output_row(
    example: dict[str, Any],
    items: dict[int, dict[str, Any]],
    query: str,
    ablation_name: str,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    target_id = get_target_id(example)
    target_item = items.get(target_id)
    target_title = get_target_title(example, target_item)
    positive = str(example.get("target_item_text") or "").strip()
    if not positive:
        positive = target_text(target_item, target_title)
    if not positive:
        raise ValueError(f"Missing positive text for target_id={target_id}")

    history_ids = get_history_ids(example)

    return {
        "query": query,
        "positive": positive,
        "category": "CDs_and_Vinyl",
        "split": example.get("split", "train"),
        "user_id": example.get("user_id", ""),
        "interaction_id": example.get("interaction_id", ""),
        "target_item_id": target_id,
        "target_item_title": target_title,
        "history_item_ids": history_ids,
        "history_item_count": example.get("history_item_count", len(history_ids)),
        "ablation_name": ablation_name,
        "query_fields": ["title", *fields],
    }


def build_one(
    examples: list[dict[str, Any]],
    items: dict[int, dict[str, Any]],
    ablation_name: str,
    fields: tuple[str, ...],
    max_item_chars: int,
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        query = build_query(example, items, fields, max_item_chars)
        rows.append(output_row(example, items, query, ablation_name, fields))
    return rows


def selected_ablations(raw: str) -> list[tuple[str, tuple[str, ...]]]:
    if raw == "all":
        return ABLATIONS
    wanted = {name.strip() for name in raw.split(",") if name.strip()}
    known = dict(ABLATIONS)
    missing = sorted(wanted - set(known))
    if missing:
        raise ValueError(f"Unknown ablation name(s): {', '.join(missing)}")
    selected = [(name, fields) for name, fields in ABLATIONS if name in wanted]
    if not selected:
        raise ValueError("No ablations selected")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CDs query-side metadata ablation datasets.")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--item-info", type=Path, default=DEFAULT_ITEM_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="cds_query")
    parser.add_argument("--ablations", default="all")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-item-chars", type=int, default=0)
    args = parser.parse_args()

    examples = read_jsonl(args.examples, max_rows=args.max_rows)
    items = item_map(read_jsonl(args.item_info))
    runs = selected_ablations(args.ablations)

    for name, fields in runs:
        rows = build_one(examples, items, name, fields, args.max_item_chars)
        output = args.output_dir / f"{args.output_prefix}_{name}.jsonl"
        write_jsonl(output, rows)
        print(json.dumps({"ablation": name, "rows": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
