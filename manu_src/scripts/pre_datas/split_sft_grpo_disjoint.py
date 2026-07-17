#!/usr/bin/env python3
"""Deterministically reserve disjoint SFT and GRPO examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-input", type=Path, required=True)
    parser.add_argument("--grpo-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sft-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-rows", type=int, default=10722)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {line_number} is invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def index_by_example_id(
    rows: list[dict[str, Any]], path: Path
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, 1):
        example_id = str(row.get("example_id") or "").strip()
        if not example_id:
            raise ValueError(f"{path} line {line_number} has an empty example_id")
        if example_id in indexed:
            raise ValueError(f"{path} contains duplicate example_id: {example_id}")
        indexed[example_id] = row
    return indexed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: set[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_ids(path: Path, ids: set[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("Project seed must remain 42")
    if not 0.0 < args.sft_fraction < 1.0:
        raise ValueError("--sft-fraction must be in (0, 1)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_dir}"
        )

    sft_source_rows = read_jsonl(args.sft_input)
    grpo_source_rows = read_jsonl(args.grpo_source)
    if len(sft_source_rows) != args.expected_rows:
        raise ValueError(
            f"SFT source has {len(sft_source_rows)} rows; expected {args.expected_rows}"
        )
    if len(grpo_source_rows) != args.expected_rows:
        raise ValueError(
            f"GRPO source has {len(grpo_source_rows)} rows; expected {args.expected_rows}"
        )

    sft_source_by_id = index_by_example_id(sft_source_rows, args.sft_input)
    grpo_source_by_id = index_by_example_id(grpo_source_rows, args.grpo_source)
    if sft_source_by_id.keys() != grpo_source_by_id.keys():
        only_sft = sorted(sft_source_by_id.keys() - grpo_source_by_id.keys())[:5]
        only_grpo = sorted(grpo_source_by_id.keys() - sft_source_by_id.keys())[:5]
        raise ValueError(
            "SFT and GRPO sources have different example_id sets: "
            f"only_sft={only_sft}, only_grpo={only_grpo}"
        )

    all_ids = sorted(sft_source_by_id)
    sft_size = math.floor(len(all_ids) * args.sft_fraction)
    sft_ids = set(random.Random(args.seed).sample(all_ids, sft_size))
    grpo_ids = set(all_ids) - sft_ids

    # Preserve the original file order inside each output. DataLoader shuffle remains
    # controlled independently by seed=42 during training.
    sft_rows = [
        row
        for row in sft_source_rows
        if str(row["example_id"]) in sft_ids
    ]
    grpo_rows = [
        row
        for row in grpo_source_rows
        if str(row["example_id"]) in grpo_ids
    ]

    if len(sft_rows) != sft_size or len(grpo_rows) != len(all_ids) - sft_size:
        raise AssertionError("Split row counts do not match the selected ID sets")
    if sft_ids & grpo_ids or sft_ids | grpo_ids != set(all_ids):
        raise AssertionError("SFT/GRPO split is not an exact partition")

    serialized_sft = "\n".join(
        json.dumps(row.get("messages"), ensure_ascii=False) for row in sft_rows
    )
    forbidden_message_fields = {
        field
        for field in ("target_item_id", "target_item_title", "positive")
        if f'"{field}"' in serialized_sft
    }
    if forbidden_message_fields:
        raise ValueError(
            f"SFT messages contain target fields: {sorted(forbidden_message_fields)}"
        )

    truncated_positive = sum(
        "[TRUNCATED]" in str(row.get("positive") or "") for row in grpo_rows
    )
    if truncated_positive:
        raise ValueError(
            f"GRPO source contains {truncated_positive} truncated positive fields"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_output = args.output_dir / f"sft_train20_seed42_n{len(sft_rows)}.jsonl"
    grpo_output = args.output_dir / f"grpo_train80_seed42_n{len(grpo_rows)}.jsonl"
    sft_ids_output = args.output_dir / "sft_example_ids.txt"
    grpo_ids_output = args.output_dir / "grpo_example_ids.txt"
    manifest_output = args.output_dir / "split_manifest.json"

    write_jsonl(sft_output, sft_rows)
    write_jsonl(grpo_output, grpo_rows)
    write_ids(sft_ids_output, sft_ids)
    write_ids(grpo_ids_output, grpo_ids)

    id_to_user = {
        example_id: str(row.get("user_id") or "").strip()
        for example_id, row in grpo_source_by_id.items()
    }
    sft_users = {id_to_user[x] for x in sft_ids if id_to_user[x]}
    grpo_users = {id_to_user[x] for x in grpo_ids if id_to_user[x]}
    manifest = {
        "split_unit": "example_id",
        "selection": "random sample over sorted example_id values",
        "seed": args.seed,
        "requested_sft_fraction": args.sft_fraction,
        "source_rows": len(all_ids),
        "sft_rows": len(sft_rows),
        "grpo_rows": len(grpo_rows),
        "actual_sft_fraction": len(sft_rows) / len(all_ids),
        "sft_grpo_example_overlap": len(sft_ids & grpo_ids),
        "partition_missing_examples": len(set(all_ids) - (sft_ids | grpo_ids)),
        "sft_users": len(sft_users),
        "grpo_users": len(grpo_users),
        "sft_grpo_user_overlap": len(sft_users & grpo_users),
        "sft_input": str(args.sft_input),
        "sft_input_sha256": file_sha256(args.sft_input),
        "grpo_source": str(args.grpo_source),
        "grpo_source_sha256": file_sha256(args.grpo_source),
        "sft_output": str(sft_output),
        "grpo_output": str(grpo_output),
        "sft_ids_sha256": ids_sha256(sft_ids),
        "grpo_ids_sha256": ids_sha256(grpo_ids),
        "target_fields_in_sft_messages": False,
        "grpo_truncated_positive_rows": truncated_positive,
    }
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
