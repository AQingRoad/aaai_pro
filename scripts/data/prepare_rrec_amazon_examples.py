#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.item_metadata import build_item_map, build_item_summary_map, build_item_text, compact, history_text
from rubric_cot_pipeline.io import read_jsonl
from rubric_cot_pipeline.io import write_jsonl


def split_names(raw: str) -> list[str]:
    if raw == "all":
        return ["train", "valid", "test"]
    return [raw]


def read_arrow_rows(split_dir: Path) -> list[dict]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required for the Arrow fallback when datasets.load_from_disk cannot read the dataset."
        ) from exc

    rows: list[dict] = []
    arrow_files = sorted(split_dir.glob("*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {split_dir}")
    for arrow_path in arrow_files:
        with pa.memory_map(str(arrow_path), "r") as source:
            try:
                reader = ipc.open_stream(source)
            except pa.ArrowInvalid:
                source.seek(0)
                reader = ipc.open_file(source)
            rows.extend(reader.read_all().to_pylist())
    return rows


def load_dataset_or_arrow(dataset_dir: Path):
    try:
        from datasets import load_from_disk

        return load_from_disk(str(dataset_dir)), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def build_output_row(
    row: dict,
    *,
    args: argparse.Namespace,
    split: str,
    item_map: dict[int, dict],
    summary_map: dict[int, str],
    fallback_interaction_id: int,
) -> dict | None:
    history_item_ids = [int(x) for x in (row.get("history_item_id") or row.get("history_item_ids") or [])]
    titles = [compact(x, 300) for x in row.get("history_item_title", [])]
    ratings = [float(x) for x in row.get("history_rating", [])]
    if len(titles) < args.min_history or float(row.get("rating", row.get("target_rating", 0.0)) or 0.0) < args.min_rating:
        return None
    if len(ratings) != len(titles):
        raise ValueError(
            f"history_rating length mismatch in {args.category}:{split}: "
            f"user_id={row.get('user_id')} interaction_id={row.get('interaction_id')} "
            f"titles={len(titles)} ratings={len(ratings)}."
        )

    item_id = int(row.get("item_id", row.get("target_item_id")))
    target_title = compact(row.get("item_title", row.get("target_item_title", "")), 300)
    item_text = build_item_text(item_map.get(item_id), target_title, args.max_target_chars)
    interaction_id = int(row.get("interaction_id", fallback_interaction_id))
    user_id = str(row["user_id"])
    example_id = f"{args.category}:{split}:{interaction_id}:{user_id}"

    max_history_items = args.max_history_items
    history_slice = slice(-max_history_items, None) if max_history_items > 0 else slice(None)
    out = {
        "example_id": example_id,
        "dataset": "rrec-amazon-2023",
        "category": args.category,
        "split": split,
        "user_id": user_id,
        "interaction_id": interaction_id,
        "target_item_id": item_id,
        "target_item_asin": row.get("item_asin", row.get("target_item_asin", "")),
        "target_item_title": target_title,
        "target_item_text": item_text,
        "target_rating": float(row.get("rating", row.get("target_rating", 0.0)) or 0.0),
        "history_item_ids": history_item_ids[history_slice],
        "history_item_asins": list(row.get("item_asins", row.get("history_item_asins", [])))[history_slice],
        "history_item_count": min(len(titles), max_history_items) if max_history_items > 0 else len(titles),
        "user_history": history_text(
            args.category,
            titles,
            ratings,
            max_history_items,
            item_ids=history_item_ids,
            item_map=item_map,
            metadata_mode=args.history_metadata_mode,
            max_item_chars=args.history_max_item_chars,
            summary_map=summary_map,
            include_ratings=args.history_include_ratings,
            include_catalog_stats=args.history_include_catalog_stats,
        ),
    }
    if args.strip_rating_fields:
        for key in ("target_rating", "history_rating", "history_ratings", "rating"):
            out.pop(key, None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/autodl-tmp/rec/RRec_official/data")
    parser.add_argument("--category", required=True)
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--examples-jsonl", default="")
    parser.add_argument("--item-info", default="")
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="train")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-target-chars",
        type=int,
        default=0,
        help="Maximum target_item_text characters; 0 keeps the full target text.",
    )
    parser.add_argument(
        "--history-metadata-mode",
        choices=["none", "compact", "summary"],
        default=os.getenv("HISTORY_METADATA_MODE", "none"),
    )
    parser.add_argument("--history-max-item-chars", type=int, default=int(os.getenv("HISTORY_MAX_ITEM_CHARS", "320")))
    parser.add_argument("--item-summary", default=os.getenv("ITEM_METADATA_SUMMARY", ""))
    parser.add_argument("--history-include-ratings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history-include-catalog-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strip-rating-fields", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(args.data_root) / f"{args.category}_0_2022-10-2023-10"
    output = args.output or f"data/rrec_amazon/{args.category}/examples.jsonl"
    summary_map = build_item_summary_map(read_jsonl(args.item_summary)) if args.item_summary else {}

    rows = []
    skipped = 0
    source = ""
    if dataset_dir.exists():
        ds, dataset_load_error = load_dataset_or_arrow(dataset_dir)
        source = str(dataset_dir)
        if ds is not None:
            item_map = build_item_map(ds["item_info"])
            for split in split_names(args.split):
                split_rows = ds[split]
                if args.shuffle:
                    split_rows = split_rows.shuffle(seed=args.seed)
                if args.max_examples > 0:
                    split_rows = split_rows.select(range(min(args.max_examples, len(split_rows))))

                for row in split_rows:
                    out = build_output_row(
                        dict(row),
                        args=args,
                        split=split,
                        item_map=item_map,
                        summary_map=summary_map,
                        fallback_interaction_id=len(rows),
                    )
                    if out is None:
                        skipped += 1
                        continue
                    rows.append(out)
        else:
            item_map = build_item_map(read_arrow_rows(dataset_dir / "item_info"))
            source = f"{dataset_dir} (pyarrow fallback after {dataset_load_error})"
            for split in split_names(args.split):
                split_rows = read_arrow_rows(dataset_dir / split)
                if args.shuffle:
                    random.Random(args.seed).shuffle(split_rows)
                if args.max_examples > 0:
                    split_rows = split_rows[: args.max_examples]

                for row in split_rows:
                    out = build_output_row(
                        row,
                        args=args,
                        split=split,
                        item_map=item_map,
                        summary_map=summary_map,
                        fallback_interaction_id=len(rows),
                    )
                    if out is None:
                        skipped += 1
                        continue
                    rows.append(out)
    else:
        if args.split == "all" and not args.examples_jsonl:
            raise FileNotFoundError(f"RRec dataset directory does not exist and JSONL fallback needs a single split: {dataset_dir}")
        examples_path = Path(args.examples_jsonl) if args.examples_jsonl else Path("github_artifacts") / args.category / "rrec_eval" / f"{args.split}.jsonl"
        item_info_path = Path(args.item_info) if args.item_info else Path("github_artifacts") / args.category / "rrec_eval" / "item_info.jsonl"
        if not examples_path.exists() or not item_info_path.exists():
            raise FileNotFoundError(
                f"RRec dataset directory does not exist: {dataset_dir}; "
                f"JSONL fallback also missing examples={examples_path} or item_info={item_info_path}"
            )
        item_map = build_item_map(read_jsonl(item_info_path))
        jsonl_rows = list(read_jsonl(examples_path))
        if args.shuffle:
            random.Random(args.seed).shuffle(jsonl_rows)
        if args.max_examples > 0:
            jsonl_rows = jsonl_rows[: args.max_examples]
        source = str(examples_path)
        split = args.split
        for row in jsonl_rows:
            out = build_output_row(
                row,
                args=args,
                split=split,
                item_map=item_map,
                summary_map=summary_map,
                fallback_interaction_id=len(rows),
            )
            if out is None:
                skipped += 1
                continue
            rows.append(out)

    count = write_jsonl(output, rows)
    stats = {
        "category": args.category,
        "split": args.split,
        "dataset_dir": str(dataset_dir),
        "source": source,
        "output": output,
        "written": count,
        "skipped": skipped,
        "history_metadata_mode": args.history_metadata_mode,
        "history_max_item_chars": args.history_max_item_chars,
        "history_include_ratings": args.history_include_ratings,
        "history_include_catalog_stats": args.history_include_catalog_stats,
        "strip_rating_fields": args.strip_rating_fields,
        "item_summary": args.item_summary,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
