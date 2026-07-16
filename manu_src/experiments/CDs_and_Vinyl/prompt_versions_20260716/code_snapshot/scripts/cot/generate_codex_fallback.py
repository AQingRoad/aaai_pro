#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from scripts.cot.generate_cot_candidate_lists import (
    output_word_count,
    split_api_output,
    validate_answer_constraints,
    validate_reasoning_constraints,
)
from rubric_cot_pipeline.prompts import COT_SYSTEM, build_history_analysis_prompt


RAW_ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b", re.IGNORECASE)
REQUIRED_INPUT_FIELDS = ("category", "example_id", "user_history")
FORBIDDEN_MODEL_INPUT_FIELDS = (
    "interaction_id",
    "positive",
    "target_item_id",
    "target_item_text",
    "target_item_title",
    "user_id",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [field for field in REQUIRED_INPUT_FIELDS if not row.get(field)]
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {missing}")
            rows.append(row)
    return rows


def read_candidate_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row.get("candidate_index", 0)) != 0:
                    continue
                example_id = str(row.get("example_id") or "")
                if example_id:
                    ids.add(example_id)
    return ids


def safe_model_record(row: dict[str, Any], slot: int) -> dict[str, Any]:
    record = {
        "slot": slot,
        "system_prompt": COT_SYSTEM,
        "user_prompt": build_history_analysis_prompt(
            str(row["user_history"]),
            category=str(row["category"]),
            output_format="tagged",
            rating_context="no_rating",
        ),
    }
    if any(field in record for field in FORBIDDEN_MODEL_INPUT_FIELDS):
        raise AssertionError("target or identity field entered Codex model input")
    return record


def validate_text(analysis: str, answer: str) -> None:
    if not analysis.strip() or not answer.strip():
        raise ValueError("empty analysis or answer")
    validate_reasoning_constraints(analysis, rating_context="no_rating")
    validate_answer_constraints(answer, rating_context="no_rating")
    if output_word_count(analysis, answer) > 1024:
        raise ValueError("analysis plus answer exceeds 1024 words")
    combined = f"{analysis}\n{answer}"
    if RAW_ASIN_RE.search(combined):
        raise ValueError("raw ASIN in Codex output")
    if "[TRUNCATED]" in combined:
        raise ValueError("truncated marker in Codex output")
    if any(tag in combined.lower() for tag in ("<think>", "</think>", "<answer>", "</answer>")):
        raise ValueError("Codex structured fields must not contain wrapper tags")


def build_candidate(
    row: dict[str, Any],
    analysis: str,
    answer: str,
    *,
    model: str,
    reasoning_effort: str,
    batch_size: int,
    batch_seconds: float,
    prompt_chars: int,
) -> dict[str, Any]:
    analysis = analysis.strip()
    answer = answer.strip()
    validate_text(analysis, answer)
    example_id = str(row["example_id"])
    return {
        "example_id": example_id,
        "candidate_id": f"{example_id}-0",
        "candidate_index": 0,
        "temperature": 0.0,
        "think": analysis,
        "answer": answer,
        "cot": f"<think>\n{analysis}\n</think>\n<answer>\n{answer}\n</answer>",
        "generator_model": f"codex-{model}",
        "generation_mode": "codex_cli_fallback",
        "generation_timing": {
            "batch_total_seconds": round(batch_seconds, 6),
            "batch_size": batch_size,
        },
        "generation_api_meta": {
            "codex_model": model,
            "codex_reasoning_effort": reasoning_effort,
            "codex_structured_output": True,
            "api_output_format": "tagged",
            "api_rating_context": "no_rating",
            "api_prompt_chars": prompt_chars,
            "api_prompt_input_fields": ["slot", "category", "user_history"],
            "codex_transport_fields": ["slot", "system_prompt", "user_prompt"],
            "codex_prompt_adapter_version": "exact_glm_prompt_transport_v2",
            "target_fields_exposed": False,
            "api_reasoning_word_count": output_word_count(analysis),
            "api_answer_word_count": output_word_count(answer),
            "api_output_word_count": output_word_count(analysis, answer),
        },
    }


class CodexBatchGenerator:
    def __init__(self, args: argparse.Namespace, prompt_template: str) -> None:
        self.args = args
        self.prompt_template = prompt_template
        self.print_lock = threading.Lock()

    def log(self, message: str) -> None:
        with self.print_lock:
            print(message, flush=True)

    def invoke(self, rows: list[dict[str, Any]], attempt: int) -> list[dict[str, Any]]:
        records = [safe_model_record(row, slot) for slot, row in enumerate(rows)]
        prompt = self.prompt_template.format(
            records_json=json.dumps(records, ensure_ascii=False, indent=2)
        )
        with tempfile.TemporaryDirectory(prefix="codex-cot-") as temp_dir:
            output_path = Path(temp_dir) / "final.json"
            command = [
                self.args.codex_bin,
                "exec",
                "--ignore-user-config",
                "--ephemeral",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--cd",
                temp_dir,
                "--model",
                self.args.model,
                "--config",
                f'model_reasoning_effort="{self.args.reasoning_effort}"',
                "--output-schema",
                str(self.args.schema.resolve()),
                "--output-last-message",
                str(output_path),
                "-",
            ]
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=self.args.timeout,
                env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
                check=False,
            )
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                tail = "\n".join(completed.stderr.splitlines()[-8:])
                raise RuntimeError(
                    f"Codex exit={completed.returncode} attempt={attempt}; stderr tail: {tail}"
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("Codex output has no results array")
        by_slot: dict[int, dict[str, Any]] = {}
        for result in results:
            slot = int(result["slot"])
            if slot in by_slot:
                raise ValueError(f"duplicate Codex slot {slot}")
            by_slot[slot] = result
        expected_slots = set(range(len(rows)))
        if set(by_slot) != expected_slots:
            raise ValueError(
                f"Codex slots mismatch expected={sorted(expected_slots)} actual={sorted(by_slot)}"
            )
        candidates = []
        for slot, row in enumerate(rows):
            result = by_slot[slot]
            analysis, answer, _ = split_api_output(
                str(result["cot"]),
                "",
                max_output_words=1024,
                rating_context="no_rating",
                require_content_tags=True,
                require_literal_tags=True,
            )
            candidates.append(
                build_candidate(
                    row,
                    analysis,
                    answer,
                    model=self.args.model,
                    reasoning_effort=self.args.reasoning_effort,
                    batch_size=len(rows),
                    batch_seconds=elapsed,
                    prompt_chars=len(prompt),
                )
            )
        return candidates

    def generate_with_recovery(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        last_error = ""
        for attempt in range(1, self.args.max_retries + 1):
            try:
                candidates = self.invoke(rows, attempt)
                self.log(
                    f"codex_batch_ok rows={len(rows)} attempt={attempt} "
                    f"first={rows[0]['example_id']}"
                )
                return candidates, []
            except Exception as exc:  # noqa: BLE001 - each batch must be recoverable
                last_error = f"{type(exc).__name__}: {exc}"
                self.log(
                    f"codex_batch_retry rows={len(rows)} attempt={attempt}/{self.args.max_retries} "
                    f"first={rows[0]['example_id']} error={last_error}"
                )
        if len(rows) > 1:
            middle = len(rows) // 2
            left_candidates, left_failures = self.generate_with_recovery(rows[:middle])
            right_candidates, right_failures = self.generate_with_recovery(rows[middle:])
            return left_candidates + right_candidates, left_failures + right_failures
        return [], [{"example_id": rows[0]["example_id"], "error": last_error}]


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill missing target-free CoT candidates with Codex CLI."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--existing-candidates", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures-output", type=Path)
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=Path("experiments/prompts/codex_target_free_cot_fallback_transport_v2.txt"),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("experiments/prompts/codex_target_free_cot_fallback_transport_v2.schema.json"),
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high"), default="low")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--expected-split", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed != 42:
        raise ValueError("seed must remain 42")
    if args.batch_size < 1 or args.max_workers < 1 or args.max_retries < 1:
        raise ValueError("batch size, workers, and retries must be positive")
    input_rows = read_jsonl(args.input)
    if args.expected_split:
        wrong = sum(str(row.get("split")) != args.expected_split for row in input_rows)
        if wrong:
            raise ValueError(f"wrong split rows: {wrong}")

    existing_paths = [*args.existing_candidates]
    if args.output.exists():
        existing_paths.append(args.output)
    existing_ids = read_candidate_ids(existing_paths)
    pending = [row for row in input_rows if str(row["example_id"]) not in existing_ids]
    if args.max_examples > 0:
        pending = pending[: args.max_examples]
    print(
        f"input_rows={len(input_rows)} existing_candidates={len(existing_ids)} "
        f"pending={len(pending)} batch_size={args.batch_size} workers={args.max_workers} "
        f"model={args.model} seed={args.seed}",
        flush=True,
    )
    if not pending:
        print("no pending Codex fallback rows", flush=True)
        return

    prompt_template = args.prompt_template.read_text(encoding="utf-8")
    generator = CodexBatchGenerator(args, prompt_template)
    all_candidates: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    batches = chunks(pending, args.batch_size)
    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {
            pool.submit(generator.generate_with_recovery, batch): batch for batch in batches
        }
        completed_rows = 0
        for future in futures.as_completed(future_map):
            candidates, failures = future.result()
            all_candidates.extend(candidates)
            all_failures.extend(failures)
            completed_rows += len(candidates) + len(failures)
            print(
                f"codex_progress={completed_rows}/{len(pending)} "
                f"success={len(all_candidates)} failures={len(all_failures)}",
                flush=True,
            )

    existing_output_rows: list[dict[str, Any]] = []
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            existing_output_rows = [json.loads(line) for line in handle if line.strip()]
    input_order = {str(row["example_id"]): index for index, row in enumerate(input_rows)}
    merged = {str(row["example_id"]): row for row in existing_output_rows}
    merged.update({str(row["example_id"]): row for row in all_candidates})
    ordered = sorted(merged.values(), key=lambda row: input_order[str(row["example_id"])])
    write_jsonl(args.output, ordered)

    failures_output = args.failures_output or args.output.with_suffix(".failures.jsonl")
    write_jsonl(failures_output, all_failures)
    print(
        f"done output_rows={len(ordered)} new_success={len(all_candidates)} "
        f"failures={len(all_failures)} output={args.output}",
        flush=True,
    )
    if all_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
