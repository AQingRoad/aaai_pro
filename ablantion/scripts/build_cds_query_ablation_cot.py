#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.embeddings import append_recommendation_reasoning
from rubric_cot_pipeline.item_metadata import strip_asin_references


RAW_ASIN_RE = re.compile(r"\bB0[0-9A-Z]{8}\b")
TRUNCATION_MARKER = "[TRUNCATED]"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def compact(text: Any, max_chars: int) -> str:
    text = normalize_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + f" {TRUNCATION_MARKER}"


def row_key(row: dict[str, Any], fallback_split: str) -> tuple[str, str, str, str]:
    target_id = row.get("target_item_id", row.get("item_id", ""))
    def text(value: Any, default: str = "") -> str:
        return default if value is None else str(value)

    return (
        text(row.get("split"), fallback_split),
        text(row.get("interaction_id")),
        text(row.get("user_id")),
        text(target_id),
    )


def candidate_text(candidate: dict[str, Any], mode: str) -> str:
    think = str(candidate.get("think") or "").strip()
    answer = str(candidate.get("answer") or "").strip()
    cot = str(candidate.get("cot") or "").strip()
    if mode == "answer":
        return answer or cot or think
    if mode == "think":
        return think or cot or answer
    if mode == "tagged":
        if think and answer:
            return f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"
        return cot or answer or think
    if mode == "full":
        return cot or candidate_text(candidate, "tagged")
    raise ValueError(f"Unsupported cot text mode: {mode}")


def selected_candidate(row: dict[str, Any], candidate_index: int) -> dict[str, Any] | None:
    candidates = row.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            try:
                index = int(candidate.get("candidate_index", -1))
            except (TypeError, ValueError):
                index = -1
            if index == candidate_index:
                return candidate
        return None
    if any(key in row for key in ("think", "answer", "cot")):
        return row
    return None


def cot_lookup(rows: list[dict[str, Any]], fallback_split: str) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    out = {}
    duplicates = 0
    for row in rows:
        key = row_key(row, fallback_split)
        if key in out:
            duplicates += 1
        out[key] = row
    if duplicates:
        raise ValueError(f"CoT source has duplicate keys: {duplicates}")
    return out


def load_item_map(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    items = {}
    for row in read_jsonl(path):
        if row.get("item_id") is None:
            continue
        items[int(row["item_id"])] = row
    return items


def as_text_list(value: Any, limit: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = normalize_text(value)
        return [text] if text else []
    if isinstance(value, list):
        out = []
        for item in value:
            text = normalize_text(item)
            if text:
                out.append(text)
            if len(out) >= limit:
                break
        return out
    text = normalize_text(value)
    return [text] if text else []


def get_history_item_ids(row: dict[str, Any]) -> list[int]:
    raw = row.get("history_item_ids") or row.get("history_item_id") or []
    return [int(item_id) for item_id in raw]


def history_item_text(item_id: int, item_map: dict[int, dict[str, Any]]) -> str:
    item = item_map.get(item_id, {})
    title = normalize_text(item.get("title")) or f"item_{item_id}"
    parts = [title]

    store = normalize_text(item.get("store"))
    if store:
        parts.append(f"Store/artist/format: {store}")

    categories = " > ".join(as_text_list(item.get("categories"), limit=4))
    if categories:
        parts.append(f"Categories: {categories}")

    return strip_asin_references("; ".join(parts))


def rebuild_base_query(row: dict[str, Any], item_map: dict[int, dict[str, Any]]) -> str:
    lines = ["This user's Amazon CDs and Vinyl interaction history over time is listed below."]
    for pos, item_id in enumerate(get_history_item_ids(row), start=1):
        lines.append(f"{pos}. {history_item_text(item_id, item_map)}")
    return strip_asin_references("\n".join(lines)).strip()


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows = read_jsonl(args.base_jsonl)
    cot_rows = read_jsonl(args.cot_jsonl)
    cot_by_key = cot_lookup(cot_rows, args.split)
    item_map = load_item_map(args.item_info)
    if args.rebuild_base_query_from_item_info and not item_map:
        raise ValueError("--item-info is required when rebuilding base query")

    rows: list[dict[str, Any]] = []
    missing_cot: list[tuple[str, str, str, str]] = []
    empty_base_query = 0
    empty_positive = 0

    for base in base_rows:
        key = row_key(base, args.split)
        if args.rebuild_base_query_from_item_info:
            base_query = rebuild_base_query(base, item_map)
        else:
            base_query = str(base.get("query") or base.get("user_history") or "").strip()
        positive = str(base.get("positive") or base.get("target_item_text") or "").strip()
        if not base_query:
            empty_base_query += 1
            continue
        if not positive:
            empty_positive += 1
            continue

        cot_row = cot_by_key.get(key)
        candidate = selected_candidate(cot_row, args.candidate_index) if cot_row else None
        cot = compact(candidate_text(candidate, args.cot_text_mode), args.max_cot_chars) if candidate else ""
        if not cot:
            missing_cot.append(key)
            if args.require_cot:
                continue

        out = dict(base)
        out.pop("user_history", None)
        out["query"] = append_recommendation_reasoning(base_query, cot)
        out["base_query"] = base_query
        out["query_type"] = f"{base.get('ablation_name', 'query_ablation')}_plus_cot_{args.cot_text_mode}"
        out["cot_text_mode"] = args.cot_text_mode
        out["cot_candidate_index"] = args.candidate_index
        out["cot_source_example_id"] = cot_row.get("example_id") if cot_row else ""
        out["cot"] = cot
        if candidate:
            out["cot_candidate_id"] = candidate.get("candidate_id", "")
            out["cot_generator_model"] = candidate.get("generator_model", "")
        rows.append(out)

    queries = [str(row.get("query") or "") for row in rows]
    positives = [str(row.get("positive") or "") for row in rows]
    audit = {
        "base_jsonl": str(args.base_jsonl),
        "cot_jsonl": str(args.cot_jsonl),
        "item_info": str(args.item_info) if args.item_info else "",
        "output": str(args.output),
        "split": args.split,
        "rebuild_base_query_from_item_info": args.rebuild_base_query_from_item_info,
        "query_fields": ["title", "store", "categories"],
        "source_base_rows": len(base_rows),
        "source_cot_rows": len(cot_rows),
        "written": len(rows),
        "candidate_index": args.candidate_index,
        "cot_text_mode": args.cot_text_mode,
        "max_cot_chars": args.max_cot_chars,
        "require_cot": args.require_cot,
        "empty_base_query": empty_base_query,
        "empty_positive": empty_positive,
        "missing_cot": len(missing_cot),
        "raw_asin_in_query": sum(1 for query in queries if RAW_ASIN_RE.search(query)),
        "truncated_query": sum(1 for query in queries if TRUNCATION_MARKER in query),
        "truncated_positive": sum(1 for positive in positives if TRUNCATION_MARKER in positive),
        "query_contains_cot": sum(1 for query in queries if "Recommendation reasoning:" in query),
        "query_contains_store": sum(1 for query in queries if "Store/artist/format:" in query),
        "query_contains_categories": sum(1 for query in queries if "Categories:" in query),
    }
    if missing_cot and args.require_cot:
        audit["missing_cot_examples"] = ["|".join(key) for key in missing_cot[:10]]
        raise ValueError(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["raw_asin_in_query"] and args.fail_on_raw_asin:
        raise ValueError(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["truncated_positive"] and args.fail_on_truncated_positive:
        raise ValueError(json.dumps(audit, ensure_ascii=False, indent=2))
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Append existing CoT to CDs query-ablation JSONL rows.")
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--cot-jsonl", type=Path, required=True)
    parser.add_argument("--item-info", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--split", default="")
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--cot-text-mode", choices=["answer", "think", "tagged", "full"], default="tagged")
    parser.add_argument("--max-cot-chars", type=int, default=0)
    parser.add_argument("--rebuild-base-query-from-item-info", action="store_true")
    parser.add_argument("--require-cot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-raw-asin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-truncated-positive", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows, audit = build_rows(args)
    write_jsonl(args.output, rows)
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
