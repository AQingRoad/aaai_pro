#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.item_metadata import build_item_text, compact, strip_asin_references


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "ablantion/datas/processed_datas/rrec_meta/CDs_and_Vinyl/title_store_categories"
RAW_ASIN_RE = re.compile(r"\bB0[0-9A-Z]{8}\b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_native_dataset(path: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    from datasets import load_from_disk

    ds = load_from_disk(str(path))
    splits = {split: [dict(row) for row in ds[split]] for split in ("train", "valid", "test")}
    item_info = [dict(row) for row in ds["item_info"]]
    return splits, item_info


def save_rrec_dataset(path: Path, native_rows: dict[str, list[dict[str, Any]]], item_info: list[dict[str, Any]]) -> None:
    from datasets import Dataset, DatasetDict

    path.mkdir(parents=True, exist_ok=True)
    DatasetDict(
        {
            "train": Dataset.from_list(native_rows["train"]),
            "valid": Dataset.from_list(native_rows["valid"]),
            "test": Dataset.from_list(native_rows["test"]),
            "item_info": Dataset.from_list(item_info),
        }
    ).save_to_disk(str(path))
    make_saved_dataset_legacy_compatible(path)


def make_saved_dataset_legacy_compatible(path: Path) -> None:
    for info_path in path.glob("*/dataset_info.json"):
        text = info_path.read_text(encoding="utf-8")
        info_path.write_text(text.replace('"_type": "List"', '"_type": "Sequence"'), encoding="utf-8")
    for arrow_path in path.glob("*/*.arrow"):
        make_arrow_metadata_legacy_compatible(arrow_path)


def make_arrow_metadata_legacy_compatible(path: Path) -> None:
    import pyarrow.ipc as arrow_ipc

    with path.open("rb") as f:
        reader = arrow_ipc.RecordBatchStreamReader(f)
        schema = reader.schema
        batches = list(reader)

    metadata = dict(schema.metadata or {})
    hf_metadata = metadata.get(b"huggingface")
    if not hf_metadata or b'"_type": "List"' not in hf_metadata:
        return

    metadata[b"huggingface"] = hf_metadata.replace(b'"_type": "List"', b'"_type": "Sequence"')
    tmp_path = path.with_suffix(".arrow.tmp")
    with tmp_path.open("wb") as f:
        with arrow_ipc.RecordBatchStreamWriter(f, schema.with_metadata(metadata)) as writer:
            for batch in batches:
                writer.write_batch(batch)
    tmp_path.replace(path)


def item_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["item_id"]): row for row in rows if row.get("item_id") is not None}


def text_list(value: Any, limit: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [compact(value, 500)] if value.strip() else []
    if isinstance(value, list):
        return [compact(x, 500) for x in value[:limit] if compact(x, 500)]
    return [compact(value, 500)]


def history_ids(row: dict[str, Any]) -> list[int]:
    raw = row.get("history_item_ids", row.get("history_item_id", []))
    return [int(x) for x in raw]


def target_id(row: dict[str, Any]) -> int:
    return int(row.get("target_item_id", row.get("item_id")))


def target_title(row: dict[str, Any], item: dict[str, Any] | None) -> str:
    title = row.get("target_item_title") or row.get("item_title")
    if title:
        return str(title)
    return compact((item or {}).get("title"), 300)


def history_item_text(item_id: int, items: dict[int, dict[str, Any]], fallback_title: str = "") -> str:
    item = items.get(item_id)
    title = compact((item or {}).get("title"), 240) if item else compact(fallback_title, 240)
    parts = [title or "[missing title]"]

    store = compact((item or {}).get("store"), 180)
    if store:
        parts.append(f"Store/artist/format: {store}")

    categories = " > ".join(text_list((item or {}).get("categories"), limit=4))
    if categories:
        parts.append(f"Categories: {categories}")

    return strip_asin_references("; ".join(parts))


def user_history(row: dict[str, Any], items: dict[int, dict[str, Any]]) -> str:
    lines = ["This user's Amazon CDs and Vinyl interaction history over time is listed below."]
    fallback_titles = row.get("history_item_title", [])
    for pos, item_id in enumerate(history_ids(row), start=1):
        fallback = fallback_titles[pos - 1] if pos - 1 < len(fallback_titles) else ""
        lines.append(f"{pos}. {history_item_text(item_id, items, fallback)}")
    return strip_asin_references("\n".join(lines)).strip()


def native_lookup(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(row["interaction_id"]), str(row["user_id"])): row for row in rows}


def assert_train_alignment(example: dict[str, Any], native: dict[str, Any]) -> None:
    checks = (
        ("user_id", "user_id"),
        ("interaction_id", "interaction_id"),
        ("target_item_id", "item_id"),
        ("target_item_asin", "item_asin"),
    )
    for left, right in checks:
        if example.get(left) != native.get(right):
            raise ValueError(
                f"Train alignment mismatch at interaction_id={example.get('interaction_id')}: "
                f"{left}={example.get(left)!r}, {right}={native.get(right)!r}"
            )
    if [int(x) for x in example.get("history_item_ids", [])] != [int(x) for x in native.get("history_item_id", [])]:
        raise ValueError(f"History item id mismatch at interaction_id={example.get('interaction_id')}")


def examples_row_from_native(
    native: dict[str, Any],
    split: str,
    items: dict[int, dict[str, Any]],
    source_example: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tid = target_id(native)
    item = items.get(tid)
    title = target_title(native, item)
    query = user_history(native, items)

    if source_example is not None:
        assert_train_alignment(source_example, native)
        out = dict(source_example)
    else:
        out = {
            "example_id": f"CDs_and_Vinyl:{split}:{native.get('interaction_id')}:{native.get('user_id')}",
            "dataset": "rrec-amazon-2023",
            "category": "CDs_and_Vinyl",
            "split": split,
            "user_id": native.get("user_id", ""),
            "interaction_id": native.get("interaction_id", ""),
            "target_item_id": tid,
            "target_item_asin": native.get("item_asin", ""),
            "target_item_title": title,
            "target_item_text": build_item_text(item, title, 0),
            "target_rating": native.get("rating"),
            "history_item_ids": history_ids(native),
            "history_item_asins": native.get("item_asins", []),
            "history_item_count": len(history_ids(native)),
        }

    out["history_rating"] = list(native.get("history_rating", []))
    out["history_timestamp"] = list(native.get("history_timestamp", []))
    out["timestamp"] = native.get("timestamp")
    out["user_history"] = query
    out["query"] = query
    out["query_fields"] = ["title", "store", "categories"]
    out["query_metadata_mode"] = "title_store_categories"
    out["rrec_native_source"] = "data/rrec_amazon/CDs_and_Vinyl_0_2022-10-2023-10"
    return out


def native_row_with_meta(row: dict[str, Any], items: dict[int, dict[str, Any]], keep_audit_cols: bool) -> dict[str, Any]:
    out = dict(row)
    original_titles = list(row.get("history_item_title", []))
    out["history_item_title"] = [
        history_item_text(item_id, items, fallback)
        for item_id, fallback in zip(history_ids(row), original_titles)
    ]
    if keep_audit_cols:
        out["history_item_title_original"] = original_titles
        out["query_fields"] = ["title", "store", "categories"]
        out["query_metadata_mode"] = "title_store_categories"
    return out


def audit_examples(
    source_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    split: str,
    source_is_train_examples: bool,
) -> dict[str, Any]:
    target_diffs = 0
    for src, out in zip(source_rows, output_rows):
        if source_is_train_examples:
            pairs = (
                ("target_item_id", "target_item_id"),
                ("target_item_title", "target_item_title"),
                ("target_item_text", "target_item_text"),
                ("target_rating", "target_rating"),
            )
        else:
            pairs = (
                ("item_id", "target_item_id"),
                ("item_title", "target_item_title"),
                ("item_asin", "target_item_asin"),
                ("rating", "target_rating"),
            )
        if any(src.get(src_key) != out.get(out_key) for src_key, out_key in pairs):
            target_diffs += 1

    queries = [str(row.get("user_history", "")) for row in output_rows]
    return {
        "split": split,
        "rows": len(output_rows),
        "target_diff_vs_source": target_diffs,
        "empty_query": sum(1 for q in queries if not q.strip()),
        "raw_asin_in_query": sum(1 for q in queries if RAW_ASIN_RE.search(q)),
        "query_contains_store": sum(1 for q in queries if "Store/artist/format:" in q),
        "query_contains_categories": sum(1 for q in queries if "Categories:" in q),
        "missing_history_timestamp": sum(1 for row in output_rows if not row.get("history_timestamp")),
        "missing_timestamp": sum(1 for row in output_rows if row.get("timestamp") is None),
        "missing_history_rating": sum(1 for row in output_rows if not row.get("history_rating")),
    }


def audit_native(source_rows: list[dict[str, Any]], output_rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    target_diffs = 0
    for src, out in zip(source_rows, output_rows):
        for key in ("item_id", "item_title", "item_asin", "rating", "timestamp"):
            if src.get(key) != out.get(key):
                target_diffs += 1
                break
    titles = [title for row in output_rows for title in row.get("history_item_title", [])]
    return {
        "split": split,
        "rows": len(output_rows),
        "target_diff_vs_source_native": target_diffs,
        "raw_asin_in_history_item_title": sum(1 for title in titles if RAW_ASIN_RE.search(str(title))),
        "history_title_contains_store": sum(1 for title in titles if "Store/artist/format:" in str(title)),
        "history_title_contains_categories": sum(1 for title in titles if "Categories:" in str(title)),
        "missing_history_timestamp": sum(1 for row in output_rows if not row.get("history_timestamp")),
        "missing_timestamp": sum(1 for row in output_rows if row.get("timestamp") is None),
        "missing_history_rating": sum(1 for row in output_rows if not row.get("history_rating")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CDs RRec-meta title+store+categories data.")
    parser.add_argument("--train-examples", type=Path, default=ROOT / "data/rrec_amazon/CDs_and_Vinyl/examples.jsonl")
    parser.add_argument(
        "--native-dataset-dir",
        type=Path,
        default=ROOT / "data/rrec_amazon/CDs_and_Vinyl_0_2022-10-2023-10",
    )
    parser.add_argument("--item-info", type=Path, default=ROOT / "github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    items = item_map(read_jsonl(args.item_info))
    train_examples = read_jsonl(args.train_examples)
    native_src, native_item_info = load_native_dataset(args.native_dataset_dir)

    train_native_by_key = native_lookup(native_src["train"])
    examples = {}
    examples["train"] = [
        examples_row_from_native(
            train_native_by_key[(int(row["interaction_id"]), str(row["user_id"]))],
            "train",
            items,
            source_example=row,
        )
        for row in train_examples
    ]
    for split in ("valid", "test"):
        examples[split] = [examples_row_from_native(row, split, items) for row in native_src[split]]

    native_jsonl = {
        split: [native_row_with_meta(row, items, keep_audit_cols=True) for row in rows]
        for split, rows in native_src.items()
    }
    rrec_dataset_rows = {
        split: [native_row_with_meta(row, items, keep_audit_cols=False) for row in rows]
        for split, rows in native_src.items()
    }

    for split, rows in examples.items():
        write_jsonl(args.output_dir / "examples" / f"{split}.jsonl", rows)
    for split, rows in native_jsonl.items():
        write_jsonl(args.output_dir / "native_jsonl" / f"{split}.jsonl", rows)
    write_jsonl(args.output_dir / "native_jsonl" / "item_info.jsonl", native_item_info)
    save_rrec_dataset(args.output_dir / "rrec_dataset", rrec_dataset_rows, native_item_info)

    audit = {
        "query_side": "title_store_categories",
        "target_side": "train examples keep original target_item_text; valid/test examples rebuild target_item_text from item_info; native rows keep original RRec target fields.",
        "timestamp_source": str(args.native_dataset_dir),
        "inputs": {
            "train_examples": str(args.train_examples),
            "native_dataset_dir": str(args.native_dataset_dir),
            "item_info_for_query_and_target_text": str(args.item_info),
        },
        "outputs": {
            "examples": str(args.output_dir / "examples"),
            "native_jsonl": str(args.output_dir / "native_jsonl"),
            "rrec_dataset": str(args.output_dir / "rrec_dataset"),
        },
        "examples_audit": [
            audit_examples(train_examples, examples["train"], "train", source_is_train_examples=True),
            audit_examples(native_src["valid"], examples["valid"], "valid", source_is_train_examples=False),
            audit_examples(native_src["test"], examples["test"], "test", source_is_train_examples=False),
        ],
        "native_audit": [
            audit_native(native_src["train"], native_jsonl["train"], "train"),
            audit_native(native_src["valid"], native_jsonl["valid"], "valid"),
            audit_native(native_src["test"], native_jsonl["test"], "test"),
        ],
        "rrec_dataset_counts": {
            "train": len(rrec_dataset_rows["train"]),
            "valid": len(rrec_dataset_rows["valid"]),
            "test": len(rrec_dataset_rows["test"]),
            "item_info": len(native_item_info),
        },
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
