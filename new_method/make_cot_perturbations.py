#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_method.core import (
    cot_from_row,
    extract_answer,
    file_sha256,
    history_from_row,
    read_jsonl,
    stable_row_key,
    write_jsonl,
)


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")
SUPPORTED_VARIANTS = (
    "original",
    "think_plus_answer",
    "answer_only",
    "reverse_sentences",
    "repeat_tail",
)


def iter_candidates(path: str) -> Iterator[dict[str, Any]]:
    for source_row in read_jsonl(path):
        candidates = source_row.get("candidates")
        if isinstance(candidates, list):
            base = {key: value for key, value in source_row.items() if key != "candidates"}
            for candidate_index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    continue
                row = {**base, **candidate}
                row.setdefault("candidate_index", candidate_index)
                yield row
        else:
            yield source_row


def original_text(row: dict[str, Any]) -> str:
    return str(
        row.get("cot")
        or row.get("completion")
        or row.get("reference_cot")
        or row.get("selected_cot")
        or ""
    ).strip()


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def variant_texts(row: dict[str, Any], variants: tuple[str, ...]) -> list[tuple[str, str]]:
    think, _ = cot_from_row(row)
    answer = str(row.get("answer") or "").strip() or extract_answer(original_text(row))
    sentences = split_sentences(think)
    outputs: list[tuple[str, str]] = []
    for variant in variants:
        text = ""
        if variant == "original":
            text = think
        elif variant == "think_plus_answer":
            text = "\n".join(part for part in (think, answer) if part)
        elif variant == "answer_only":
            text = answer
        elif variant == "reverse_sentences":
            text = " ".join(reversed(sentences))
        elif variant == "repeat_tail":
            text = " ".join(sentences + ([sentences[-1], sentences[-1]] if sentences else []))
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        text = text.strip()
        if text:
            outputs.append((variant, text))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create target-blind deterministic CoT variants for paired gain scoring."
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variants",
        default=",".join(SUPPORTED_VARIANTS),
        help="Comma-separated variants.",
    )
    args = parser.parse_args()

    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    unknown = sorted(set(variants) - set(SUPPORTED_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    output_rows: list[dict[str, Any]] = []
    source_rows = 0
    for input_path in args.input:
        for row in iter_candidates(input_path):
            source_rows += 1
            history = history_from_row(row)
            if not history:
                continue
            source_id = str(
                row.get("candidate_id")
                or row.get("cot_candidate_id")
                or row.get("candidate_index")
                or row.get("cot_candidate_index")
                or "0"
            )
            for variant, cot_think in variant_texts(row, variants):
                if "[TRUNCATED]" in history or "[TRUNCATED]" in cot_think:
                    raise ValueError(
                        f"Truncation marker in example_id={stable_row_key(row)} variant={variant}"
                    )
                output_rows.append(
                    {
                        **row,
                        "example_id": stable_row_key(row),
                        "user_history": history,
                        "cot_think": cot_think,
                        "cot_source": str(row.get("cot_source") or Path(input_path).stem),
                        "candidate_id": f"{source_id}:{variant}",
                        "perturbation_type": variant,
                        "has_tags": True,
                        "format_ok": True,
                        "perturbation_target_blind": True,
                        "source_file": str(Path(input_path).resolve()),
                    }
                )

    count = write_jsonl(args.output, output_rows)
    metadata = {
        "inputs": [str(Path(path).resolve()) for path in args.input],
        "input_sha256": {str(Path(path).resolve()): file_sha256(path) for path in args.input},
        "output": str(Path(args.output).resolve()),
        "source_rows": source_rows,
        "written_rows": count,
        "variants": variants,
        "target_fields_read_by_transform": [],
    }
    meta_path = Path(args.output).with_suffix(Path(args.output).suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
