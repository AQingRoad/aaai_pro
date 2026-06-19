from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

TAG_RE = re.compile(
    r"<(?:analysis|think|thinking|thoughts)>(.*?)</(?:analysis|think|thinking|thoughts)>\s*"
    r"<(?:recommendation|answer)>(.*?)</(?:recommendation|answer)>",
    re.DOTALL | re.IGNORECASE,
)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
TITLE_RE = re.compile(r"([^;]+?\(\d{4}\))\s*,\s*([1-5])\s+stars", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")

PREFERENCE_TERMS = {
    "action", "adventure", "animation", "animated", "classic", "comedy", "crime", "drama",
    "fantasy", "family", "historical", "horror", "mystery", "romance", "romantic", "sci-fi",
    "science fiction", "thriller", "war", "western", "musical", "documentary", "character",
    "plot", "theme", "tone", "dark", "humor", "satire", "nostalgia", "award", "director",
    "actor", "story", "visual", "soundtrack", "foreign", "independent", "psychological",
}

TRANSITION_TERMS = {
    "suggests", "therefore", "likely", "may enjoy", "could enjoy", "generalize", "transition",
    "extends to", "points toward", "indicates", "because", "so the user", "unseen",
}

DISCRIMINATIVE_TERMS = {
    "avoid", "less likely", "rather than", "instead of", "not", "dislike", "negative",
    "counterexample", "prefer", "over", "whereas", "distinguish", "contrast", "low-rated",
}

LEAKAGE_TERMS = {
    "target item", "held-out", "ground truth", "future interaction", "label", "test item",
    "the answer is", "will click", "will rate",
}

GENERIC_PHRASES = {
    "diverse range of interests",
    "open to different types",
    "variety of movies",
    "it is difficult to say",
    "overall the user likes good movies",
}


def extract_blocks(text: str) -> tuple[str, str, bool]:
    match = TAG_RE.search(text or "")
    if not match:
        return "", (text or "").strip(), False
    return match.group(1).strip(), match.group(2).strip(), True


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def history_titles(user_history: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for title, rating in TITLE_RE.findall(user_history or ""):
        out.append((title.strip(), int(rating)))
    return out


def _contains_title(text_norm: str, title: str) -> bool:
    title_norm = normalize(title)
    title_no_year = normalize(re.sub(r"\s*\(\d{4}\)\s*$", "", title))
    return bool(title_norm and title_norm in text_norm) or bool(title_no_year and title_no_year in text_norm)


def _score_from_hits(hits: int, thresholds: tuple[int, int, int, int]) -> int:
    score = 1
    for t in thresholds:
        if hits >= t:
            score += 1
    return min(score, 5)


def _term_hits(text_norm: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term in text_norm)


def rule_score(user_history: str, cot: str) -> dict[str, Any]:
    think, answer, has_tags = extract_blocks(cot)
    combined = f"{think}\n{answer}".strip() or cot
    low = normalize(combined)
    titles = history_titles(user_history)
    positive_titles = [t for t, r in titles if r >= 4]
    low_titles = [t for t, r in titles if r <= 2]

    mentioned_any = sum(1 for title, _ in titles if _contains_title(low, title))
    mentioned_pos = sum(1 for title in positive_titles if _contains_title(low, title))
    mentioned_low = sum(1 for title in low_titles if _contains_title(low, title))
    grounding_hits = mentioned_pos + min(mentioned_any, 2)

    preference_hits = _term_hits(low, PREFERENCE_TERMS)
    transition_hits = _term_hits(low, TRANSITION_TERMS)
    discriminative_hits = _term_hits(low, DISCRIMINATIVE_TERMS) + min(mentioned_low, 2)
    leakage_hits = _term_hits(low, LEAKAGE_TERMS)
    generic_hits = _term_hits(low, GENERIC_PHRASES)

    wc = word_count(combined)
    if 70 <= wc <= 220:
        conciseness = 5
    elif 45 <= wc < 70 or 221 <= wc <= 280:
        conciseness = 4
    elif 25 <= wc < 45 or 281 <= wc <= 360:
        conciseness = 3
    elif wc > 0:
        conciseness = 2
    else:
        conciseness = 1

    preference_grounding = _score_from_hits(grounding_hits, (1, 2, 4, 6))
    taste_specificity = _score_from_hits(preference_hits, (2, 4, 7, 10))
    transitional_reasoning = _score_from_hits(transition_hits, (1, 2, 4, 6))
    discriminative_framing = _score_from_hits(discriminative_hits, (1, 2, 4, 6))

    if not has_tags:
        conciseness = max(1, conciseness - 1)
    if leakage_hits:
        preference_grounding = max(1, preference_grounding - leakage_hits)
        discriminative_framing = max(1, discriminative_framing - leakage_hits)
    if generic_hits:
        taste_specificity = max(1, taste_specificity - generic_hits)

    dims = {
        "preference_grounding": preference_grounding,
        "taste_specificity": taste_specificity,
        "transitional_reasoning": transitional_reasoning,
        "discriminative_framing": discriminative_framing,
        "conciseness": conciseness,
    }
    total = sum(dims.values())
    return {
        **dims,
        "total": total,
        "score_norm": total / 25.0,
        "has_tags": has_tags,
        "word_count": wc,
        "mentioned_history_items": mentioned_any,
        "mentioned_positive_items": mentioned_pos,
        "mentioned_low_items": mentioned_low,
        "leakage_hits": leakage_hits,
        "generic_hits": generic_hits,
        "comment": "rule-based proxy score",
    }


def parse_judge_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidates = [text]
    match = JSON_RE.search(text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def coerce_dimension(value: Any) -> int | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    if 0.0 <= score < 1.0:
        score *= 5.0
    return max(1, min(5, int(round(score))))


def normalize_judge_score(obj: dict[str, Any]) -> dict[str, Any] | None:
    dims: dict[str, int] = {}
    for key in [
        "preference_grounding",
        "taste_specificity",
        "transitional_reasoning",
        "discriminative_framing",
        "conciseness",
    ]:
        val = coerce_dimension(obj.get(key))
        if val is None:
            return None
        dims[key] = val
    total = sum(dims.values())
    return {
        **dims,
        "total": total,
        "score_norm": total / 25.0,
        "comment": str(obj.get("comment", ""))[:500],
    }


def hashed_cosine(left: str, right: str) -> float:
    left_counts = Counter(WORD_RE.findall(normalize(left)))
    right_counts = Counter(WORD_RE.findall(normalize(right)))
    if not left_counts or not right_counts:
        return 0.0
    dot = sum(v * right_counts.get(k, 0) for k, v in left_counts.items())
    left_norm = math.sqrt(sum(v * v for v in left_counts.values()))
    right_norm = math.sqrt(sum(v * v for v in right_counts.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
