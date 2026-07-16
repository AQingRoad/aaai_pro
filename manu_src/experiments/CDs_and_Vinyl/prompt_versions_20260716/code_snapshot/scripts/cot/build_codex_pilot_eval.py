#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b")
FORBIDDEN_HISTORY_MARKERS = (" stars)", "Description:", "Details:", "Catalog stats:", "[TRUNCATED]")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    previous = {int(row["index"]): row for row in read_jsonl(args.previous)}
    generated_doc = json.loads(args.generated.read_text(encoding="utf-8"))
    generated = generated_doc.get("cases", [])
    sources = {int(row["interaction_id"]): row for row in read_jsonl(args.test_source)}

    rows = []
    audit = {
        "rows": len(generated),
        "history_truncated": 0,
        "history_forbidden_marker": 0,
        "history_asin": 0,
        "target_title_in_generated_text": 0,
        "invalid_mode_payload": 0,
    }
    for case in generated:
        index = int(case["index"])
        old = previous[index]
        source = sources[int(old["interaction_id"])]
        history = str(old["base_query"]).strip()
        think = str(case.get("think") or "").strip()
        answer = str(case.get("answer") or "").strip()
        mode = str(case["mode"])

        audit["history_truncated"] += int("[TRUNCATED]" in history)
        audit["history_forbidden_marker"] += int(any(marker in history for marker in FORBIDDEN_HISTORY_MARKERS))
        audit["history_asin"] += int(bool(ASIN_RE.search(history)))
        target_title = str(source.get("target_item_title") or "").strip().casefold()
        generated_text = f"{think}\n{answer}".casefold()
        audit["target_title_in_generated_text"] += int(bool(target_title and target_title in generated_text))
        if mode == "history_only":
            audit["invalid_mode_payload"] += int(bool(think or answer))
            cot = ""
        elif mode == "cot":
            audit["invalid_mode_payload"] += int(not (think and answer))
            cot = f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"
        else:
            raise ValueError(f"Unsupported mode for index {index}: {mode}")

        rows.append(
            {
                **source,
                "user_history": history,
                "candidate_count": 1,
                "candidates": [{"candidate_index": 0, "think": think, "answer": answer, "cot": cot}],
                "pilot_index": index,
                "generator_model": "gpt-5.5",
                "generation_mode": "local_codex_target_free_concise_general_prompt",
                "codex_mode": mode,
                "codex_confidence": case["confidence"],
            }
        )

    expected = [881, 1077, 1187, 1171, 715, 820, 1168]
    actual = [int(row["pilot_index"]) for row in rows]
    if actual != expected:
        raise ValueError(f"Unexpected case order: {actual}")
    failed = {key: value for key, value in audit.items() if key != "rows" and value}
    if failed:
        raise ValueError(f"Audit failed: {failed}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
