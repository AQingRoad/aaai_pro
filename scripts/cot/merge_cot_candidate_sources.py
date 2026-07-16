#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.cot.generate_cot_candidate_lists import aggregate_output


RAW_ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b", re.IGNORECASE)
REQUIRED_TAGS = ("<think>", "</think>", "<answer>", "</answer>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly merge target-free GLM and Codex candidate files and aggregate a split."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-split", required=True)
    args = parser.parse_args()

    input_rows = read_jsonl(args.input)
    input_ids = [str(row.get("example_id") or "") for row in input_rows]
    if not all(input_ids):
        raise ValueError("generation input contains an empty example_id")
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("generation input contains duplicate example_id")
    wrong_split = sum(str(row.get("split") or "") != args.expected_split for row in input_rows)
    if wrong_split:
        raise ValueError(f"generation input has {wrong_split} rows outside {args.expected_split}")

    expected_ids = set(input_ids)
    candidate_map: dict[str, dict[int, dict[str, Any]]] = {}
    source_by_id: dict[str, str] = {}
    source_rows: dict[str, int] = {}
    generator_models: Counter[str] = Counter()
    generation_modes: Counter[str] = Counter()
    failures: Counter[str] = Counter()

    for source in args.candidate_source:
        rows = read_jsonl(source)
        source_rows[str(source.resolve())] = len(rows)
        for row in rows:
            example_id = str(row.get("example_id") or "")
            try:
                candidate_index = int(row.get("candidate_index"))
            except (TypeError, ValueError):
                failures["invalid_candidate_index"] += 1
                continue
            if example_id not in expected_ids:
                failures["candidate_id_outside_input"] += 1
                continue
            if candidate_index != 0:
                failures["nonzero_candidate_index"] += 1
                continue
            if example_id in source_by_id:
                failures["duplicate_candidate_across_sources"] += 1
                continue
            cot = str(row.get("cot") or "")
            if not all(tag in cot for tag in REQUIRED_TAGS):
                failures["missing_tagged_cot"] += 1
            if not str(row.get("think") or "").strip() or not str(row.get("answer") or "").strip():
                failures["empty_think_or_answer"] += 1
            if RAW_ASIN_RE.search(cot):
                failures["raw_asin_in_cot"] += 1
            if "[TRUNCATED]" in cot:
                failures["truncated_marker_in_cot"] += 1
            model = str(row.get("generator_model") or "")
            mode = str(row.get("generation_mode") or "")
            if not model or not mode:
                failures["missing_generator_provenance"] += 1
            if mode == "codex_cli_fallback":
                meta = row.get("generation_api_meta") or {}
                if meta.get("target_fields_exposed") is not False:
                    failures["codex_target_fields_exposed_not_false"] += 1
                if meta.get("api_prompt_input_fields") != ["slot", "category", "user_history"]:
                    failures["codex_prompt_fields_mismatch"] += 1
            source_by_id[example_id] = str(source.resolve())
            candidate_map[example_id] = {0: row}
            generator_models[model] += 1
            generation_modes[mode] += 1

    missing_ids = expected_ids - set(candidate_map)
    if missing_ids:
        failures["missing_candidate"] += len(missing_ids)
    if len(candidate_map) != len(input_rows):
        failures["candidate_count_mismatch"] += 1

    if failures:
        report = {
            "input": str(args.input.resolve()),
            "candidate_sources": [str(path.resolve()) for path in args.candidate_source],
            "expected_split": args.expected_split,
            "input_rows": len(input_rows),
            "candidate_rows": len(candidate_map),
            "source_rows": source_rows,
            "generator_models": dict(generator_models),
            "generation_modes": dict(generation_modes),
            "failures": dict(failures),
            "complete": False,
        }
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    written = aggregate_output(input_rows, candidate_map, args.output, num_candidates=1)
    report = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "candidate_sources": [str(path.resolve()) for path in args.candidate_source],
        "candidate_source_sha256": {
            str(path.resolve()): sha256(path) for path in args.candidate_source
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "expected_split": args.expected_split,
        "input_rows": len(input_rows),
        "candidate_rows": len(candidate_map),
        "output_rows": written,
        "source_rows": source_rows,
        "generator_models": dict(generator_models),
        "generation_modes": dict(generation_modes),
        "target_fields_sent_to_codex": False,
        "failures": {},
        "complete": written == len(input_rows),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if written != len(input_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
