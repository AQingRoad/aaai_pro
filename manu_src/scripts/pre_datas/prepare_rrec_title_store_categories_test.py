#!/usr/bin/env python3
"""把已审计的 RRec title_store_categories test 转成 manu_src 标准 pair。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rubric_cot_pipeline.item_metadata import strip_asin_references  # noqa: E402


RAW_ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b", re.IGNORECASE)
TIME_FIELD_RE = re.compile(r"^\d+\.\s*Time:\s", re.MULTILINE)
RATING_FIELD_RE = re.compile(r";\s*Rating:\s*[1-5](?:\.\d)?\s+star", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def text_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [text for item in value if (text := clean(item))]
    text = clean(value)
    return [text] if text else []


def rebuild_query(history_ids: list[int], items: dict[int, dict]) -> str:
    """从完整 item_info 重建严格 title_store_categories history。"""
    lines = ["This user's Amazon CDs and Vinyl interaction history over time is listed below."]
    for position, item_id in enumerate(history_ids, 1):
        if item_id not in items:
            raise ValueError(f"history item_id={item_id} 不在 item_info 中")
        item = items[item_id]
        title = clean(item.get("title")) or f"item_{item_id}"
        parts = [title]
        if store := clean(item.get("store")):
            parts.append(f"Store/artist/format: {store}")
        if categories := " > ".join(text_list(item.get("categories"))):
            parts.append(f"Categories: {categories}")
        lines.append(f"{position}. {strip_asin_references('; '.join(parts))}")
    return strip_asin_references("\n".join(lines)).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--item-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    with args.item_info.open(encoding="utf-8") as file:
        items = {
            int(row["item_id"]): row
            for row in (json.loads(line) for line in file if line.strip())
        }
    rows = []
    with args.input.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            source = json.loads(line)
            positive = str(source.get("target_item_text") or "").strip()
            example_id = str(source.get("example_id") or "").strip()
            if not positive or not example_id:
                raise ValueError(f"第 {line_number} 行缺少 target_item_text 或 example_id")
            if source.get("split") != "test":
                raise ValueError(f"第 {line_number} 行 split 不是 test")
            history_ids = [int(item_id) for item_id in source.get("history_item_ids", [])]
            query = rebuild_query(history_ids, items)
            if (
                source.get("query_fields") != ["title", "store", "categories"]
                or TIME_FIELD_RE.search(query)
                or RATING_FIELD_RE.search(query)
                or "; Description:" in query
                or "; Details:" in query
            ):
                raise ValueError(f"第 {line_number} 行 query 超出 title_store_categories 口径")
            if RAW_ASIN_RE.search(query) or "[TRUNCATED]" in query or "[TRUNCATED]" in positive:
                raise ValueError(f"第 {line_number} 行包含 ASIN 或截断标记")

            rows.append(
                {
                    "example_id": example_id,
                    "query": query,
                    "positive": positive,
                    "category": "CDs_and_Vinyl",
                    "split": "test",
                    "user_id": source.get("user_id", ""),
                    "interaction_id": source.get("interaction_id", ""),
                    "target_item_id": int(source["target_item_id"]),
                    "target_item_title": str(source.get("target_item_title") or "").strip(),
                    "history_item_ids": history_ids,
                    "history_item_count": len(history_ids),
                    "ablation_name": "title_store_categories",
                    "query_fields": ["title", "store", "categories"],
                }
            )

    if len(rows) != 1341 or len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("test 应包含 1341 个唯一 example_id")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)

    audit = {
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "item_info": str(args.item_info.resolve()),
        "item_info_sha256": sha256(args.item_info),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "rows": len(rows),
        "query_fields": ["title", "store", "categories"],
        "query_rebuilt_from_item_info": True,
        "positive_in_prompt": False,
        "target_in_history_count": sum(
            row["target_item_id"] in set(row["history_item_ids"]) for row in rows
        ),
        "raw_asin_query_count": sum(bool(RAW_ASIN_RE.search(row["query"])) for row in rows),
        "truncated_query_count": sum("[TRUNCATED]" in row["query"] for row in rows),
        "truncated_positive_count": sum("[TRUNCATED]" in row["positive"] for row in rows),
        "seed": 42,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
