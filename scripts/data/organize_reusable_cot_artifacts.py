#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.refresh_target_item_text_from_item_info import refreshed_rows
from rubric_cot_pipeline.io import read_jsonl, write_jsonl
from rubric_cot_pipeline.item_metadata import build_item_map


TRUNCATION_MARKER = "[TRUNCATED]"


DEFAULT_ARTIFACTS = [
    {
        "name": "cds_glm47_meta_compact_one_train",
        "category": "CDs_and_Vinyl",
        "split": "train",
        "source": "outputs/rrec_amazon/CDs_and_Vinyl/cot/api/cot_candidate_lists_glm47_meta_compact_one_train_raw.jsonl",
        "item_info": "github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl",
        "canonical": "CDs_and_Vinyl/cds_glm47_meta_compact_one_train_full_target.jsonl",
        "embedder": "CDs_and_Vinyl/phase0_embedder_cds_glm47_meta_compact_one_tagged_cot_only_full_target.jsonl",
        "build_embedder": True,
        "notes": "Original GLM-4.7 train CoT with ratings in user_history.",
    },
    {
        "name": "cds_glm47_meta_compact_no_all_ratings_observed_train",
        "category": "CDs_and_Vinyl",
        "split": "train",
        "source": "outputs/rrec_amazon/CDs_and_Vinyl/cot/api/cot_candidate_lists_glm47_meta_compact_no_all_ratings_observed_one_train_raw.jsonl",
        "item_info": "github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl",
        "canonical": "CDs_and_Vinyl/cds_glm47_meta_compact_no_all_ratings_observed_one_train_full_target.jsonl",
        "embedder": "CDs_and_Vinyl/phase0_embedder_cds_glm47_meta_compact_no_all_ratings_observed_tagged_cot_only_full_target.jsonl",
        "build_embedder": True,
        "notes": "GLM-4.7 train CoT with ratings and catalog stats removed from history.",
    },
    {
        "name": "cds_glm47_meta_compact_no_trunc_one_test",
        "category": "CDs_and_Vinyl",
        "split": "test",
        "source": "outputs/rrec_amazon/CDs_and_Vinyl/cot/api/cot_candidate_lists_glm47_meta_compact_no_trunc_one_test_raw.jsonl",
        "item_info": "github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl",
        "canonical": "CDs_and_Vinyl/cds_glm47_meta_compact_no_trunc_one_test_full_target.jsonl",
        "embedder": "",
        "build_embedder": False,
        "notes": "Latest GLM-4.7 test CoT for history_plus_cot evaluation.",
    },
]


def count_jsonl(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": 0,
        "candidate_rows": 0,
        "candidate_items": 0,
        "cot_rows": 0,
        "missing_cot_rows": 0,
        "target_truncated": 0,
        "positive_truncated": 0,
        "categories": {},
        "splits": {},
    }
    for row in read_jsonl(path):
        stats["rows"] += 1
        category = str(row.get("category") or "")
        split = str(row.get("split") or "")
        if category:
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
        if split:
            stats["splits"][split] = stats["splits"].get(split, 0) + 1
        if TRUNCATION_MARKER in str(row.get("target_item_text") or ""):
            stats["target_truncated"] += 1
        if TRUNCATION_MARKER in str(row.get("positive") or ""):
            stats["positive_truncated"] += 1

        candidates = row.get("candidates")
        if isinstance(candidates, list):
            stats["candidate_rows"] += 1
            stats["candidate_items"] += len(candidates)
            has_cot = any(str(c.get("cot") or c.get("think") or c.get("answer") or "").strip() for c in candidates)
        else:
            has_cot = bool(str(row.get("cot") or row.get("think") or row.get("answer") or "").strip())
        if has_cot:
            stats["cot_rows"] += 1
        else:
            stats["missing_cot_rows"] += 1
    return stats


def refresh_to_canonical(source: Path, output: Path, item_info: Path, max_target_chars: int) -> dict[str, int]:
    item_map = build_item_map(read_jsonl(item_info))
    stats = {
        "rows": 0,
        "updated": 0,
        "positive_updated": 0,
        "target_truncated_before": 0,
        "target_truncated_after": 0,
        "positive_truncated_before": 0,
        "positive_truncated_after": 0,
        "missing_target_id": 0,
        "missing_item_info": 0,
        "empty_rebuilt_text": 0,
    }
    write_jsonl(output, refreshed_rows(source, item_map, max_target_chars, stats))
    return stats


def build_embedder_dataset(
    *,
    candidate_lists: Path,
    item_info: Path,
    output: Path,
    cot_text_mode: str,
    max_cot_chars: int,
    max_item_chars: int,
    include_history: bool,
    include_cot: bool,
) -> None:
    cmd = [
        sys.executable,
        "scripts/data/make_cot_embedder_dataset.py",
        "--candidate-lists",
        str(candidate_lists),
        "--item-info",
        str(item_info),
        "--output",
        str(output),
        "--cot-text-mode",
        cot_text_mode,
        "--max-cot-chars",
        str(max_cot_chars),
        "--max-item-chars",
        str(max_item_chars),
        "--negative-sampling",
        "none",
        "--num-negatives",
        "0",
        "--include-cot" if include_cot else "--no-include-cot",
        "--include-history" if include_history else "--no-include-history",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize existing CoT artifacts for reuse without calling generation APIs again.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-root", default="outputs/rrec_amazon/reusable_cot")
    parser.add_argument("--max-target-chars", type=int, default=0)
    parser.add_argument("--cot-text-mode", choices=["answer", "think", "tagged", "full"], default="tagged")
    parser.add_argument("--max-cot-chars", type=int, default=0)
    parser.add_argument("--max-item-chars", type=int, default=0)
    parser.add_argument("--include-history", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-cot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-build-embedder", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "description": "Reusable CoT artifacts: old generated CoT is preserved; target_item_text/positive are refreshed from item_info.",
        "root": str(root),
        "output_root": str(output_root),
        "max_target_chars": args.max_target_chars,
        "cot_text_mode": args.cot_text_mode,
        "max_cot_chars": args.max_cot_chars,
        "max_item_chars": args.max_item_chars,
        "include_history": args.include_history,
        "include_cot": args.include_cot,
        "artifacts": [],
    }

    for artifact in DEFAULT_ARTIFACTS:
        source = root / artifact["source"]
        item_info = root / artifact["item_info"]
        canonical = output_root / artifact["canonical"]
        canonical.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            raise FileNotFoundError(f"Missing source: {source}")
        if not item_info.exists():
            raise FileNotFoundError(f"Missing item_info: {item_info}")

        refresh_stats = refresh_to_canonical(source, canonical, item_info, args.max_target_chars)
        canonical_stats = count_jsonl(canonical)
        entry: dict[str, Any] = {
            **artifact,
            "source": str(source.relative_to(root)),
            "item_info": str(item_info.relative_to(root)),
            "canonical": str(canonical.relative_to(root)),
            "refresh_stats": refresh_stats,
            "canonical_stats": canonical_stats,
        }

        if artifact["build_embedder"] and not args.no_build_embedder:
            embedder = output_root / artifact["embedder"]
            embedder.parent.mkdir(parents=True, exist_ok=True)
            build_embedder_dataset(
                candidate_lists=canonical,
                item_info=item_info,
                output=embedder,
                cot_text_mode=args.cot_text_mode,
                max_cot_chars=args.max_cot_chars,
                max_item_chars=args.max_item_chars,
                include_history=args.include_history,
                include_cot=args.include_cot,
            )
            entry["embedder"] = str(embedder.relative_to(root))
            entry["embedder_stats"] = count_jsonl(embedder)
        manifest["artifacts"].append(entry)

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.relative_to(root)), "artifacts": len(manifest["artifacts"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
