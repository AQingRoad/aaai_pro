from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


THINK_RE = re.compile(
    r"<(?:analysis|think|thinking|thoughts)>(.*?)</(?:analysis|think|thinking|thoughts)>",
    flags=re.IGNORECASE | re.DOTALL,
)
ANSWER_RE = re.compile(
    r"<(?:answer|recommendation)>(.*?)</(?:answer|recommendation)>",
    flags=re.IGNORECASE | re.DOTALL,
)
RAW_ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b", flags=re.IGNORECASE)
TRUNCATION_MARKER = "[TRUNCATED]"
RANK_BUCKETS = ("1-20", "21-100", "101-1000", "1000+")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {source}:{line_number}")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def as_int_set(value: Any) -> set[int]:
    if value is None:
        return set()
    values = value if isinstance(value, (list, tuple, set)) else [value]
    output: set[int] = set()
    for item in values:
        try:
            output.add(int(item))
        except (TypeError, ValueError):
            continue
    return output


def stable_row_key(row: dict[str, Any]) -> str:
    direct = row.get("example_id") or row.get("source_example_id")
    if direct:
        return str(direct)
    user_id = row.get("user_id")
    interaction_id = row.get("interaction_id")
    if user_id is not None or interaction_id is not None:
        return f"{user_id}::{interaction_id}"
    history = history_from_row(row)
    return hashlib.sha256(history.encode("utf-8")).hexdigest()[:24]


def history_from_row(row: dict[str, Any]) -> str:
    return str(
        row.get("user_history")
        or row.get("history")
        or row.get("base_query")
        or row.get("source_prompt")
        or row.get("query")
        or ""
    ).strip()


def extract_think(text: Any) -> tuple[str, bool]:
    raw = str(text or "").strip()
    if not raw:
        return "", False
    match = THINK_RE.search(raw)
    if match:
        return match.group(1).strip(), True
    return raw, False


def extract_answer(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    match = ANSWER_RE.search(raw)
    return match.group(1).strip() if match else ""


def cot_from_row(row: dict[str, Any]) -> tuple[str, bool]:
    explicit_think = str(row.get("cot_think") or row.get("think") or "").strip()
    if explicit_think:
        return explicit_think, bool(row.get("has_tags", True))
    raw = (
        row.get("cot")
        or row.get("completion")
        or row.get("reference_cot")
        or row.get("selected_cot")
        or row.get("answer")
        or ""
    )
    return extract_think(raw)


def append_think(history: str, think: str) -> str:
    history = str(history or "").strip()
    think = str(think or "").strip()
    if not think:
        return history
    return f"{history}\n\nRecommendation reasoning:\n{think}"


def ndcg_at_rank(rank: int, k: int = 20) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def delta_log_rank(baseline_rank: int, cot_rank: int) -> float:
    if baseline_rank <= 0 or cot_rank <= 0:
        raise ValueError("Ranks must be positive integers")
    return math.log1p(baseline_rank) - math.log1p(cot_rank)


def rank_bucket(rank: int) -> str:
    if rank <= 20:
        return "1-20"
    if rank <= 100:
        return "21-100"
    if rank <= 1000:
        return "101-1000"
    return "1000+"


def contains_truncation(value: Any) -> bool:
    return TRUNCATION_MARKER in str(value or "")


def quality_failures(row: dict[str, Any], *, max_unsupported_claims: int = 0) -> list[str]:
    failures: list[str] = []
    if row.get("format_ok") is False:
        failures.append("format_error")
    if bool(row.get("leakage") or row.get("target_leakage")):
        failures.append("target_leakage")
    unsupported = int(row.get("unsupported_claim_count") or 0)
    if unsupported > max_unsupported_claims:
        failures.append("unsupported_claim")
    if int(row.get("metadata_contradiction_count") or 0) > 0:
        failures.append("metadata_contradiction")
    if int(row.get("history_truncated_tokens") or 0) > 0:
        failures.append("history_truncated")
    if bool(row.get("raw_asin_in_cot")):
        failures.append("raw_asin_in_cot")
    for field in ("user_history", "history", "cot", "cot_think", "target_item_text", "positive"):
        if contains_truncation(row.get(field)):
            failures.append(f"{field}_truncated_marker")
    return sorted(set(failures))


def classify_gain(
    row: dict[str, Any],
    *,
    min_good_log_rank: float,
    min_good_margin: float,
    min_bad_log_rank: float,
    min_bad_margin: float,
    max_unsupported_claims: int = 0,
) -> tuple[str, list[str]]:
    failures = quality_failures(row, max_unsupported_claims=max_unsupported_claims)
    if failures:
        return "rejected", failures

    baseline_rank = int(row["baseline_rank"])
    cot_rank = int(row["cot_rank"])
    log_gain = float(row.get("delta_log_rank", delta_log_rank(baseline_rank, cot_rank)))
    margin_gain = float(row.get("delta_margin", row.get("sim_gain", 0.0)))

    if cot_rank <= baseline_rank and log_gain >= min_good_log_rank and margin_gain >= min_good_margin:
        return "good", []
    if cot_rank >= baseline_rank and log_gain <= min_bad_log_rank and margin_gain <= min_bad_margin:
        return "bad", []
    return "neutral", []


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
