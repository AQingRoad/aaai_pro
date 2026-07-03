from __future__ import annotations

import re

REASONING_TAG = "think"
API_REASONING_TAG = "analysis"
ANSWER_TAG = "answer"
LEGACY_REASONING_TAGS = ("analysis", "thinking", "thoughts")
LEGACY_ANSWER_TAGS = ("recommendation",)

COT_SYSTEM = (
    "You are a recommendation master capable of deeply analyzing user "
    "preferences based on their historical interactions."
)

COT_OUTPUT_INSTRUCTION = """\
Analyze the historical interactions and predict a feature description for the
next {category_singular} the user is likely to prefer or interact with.

Use concise internal reasoning covering history-grounded evidence, taste signals,
preference transition, and discriminative cues.

For the visible final response, output only the transferable feature description.
Do not include explanations, evidence, or user/history phrasing.
"""

TAGGED_COT_OUTPUT_INSTRUCTION = """\
Analyze the historical interactions and predict a feature description for the
next {category_singular} the user is likely to prefer or interact with.

Output only:
<analysis>
...
</analysis>
<answer>
...
</answer>

Format rules:
- Use exact raw tags <analysis> and <answer>; do not use <think>, <reasoning>,
  <thoughts>, markdown, JSON, or any other wrapper.
- No text outside the two blocks; the first non-whitespace character must be "<".
- The combined length of <analysis> and <answer> must not exceed 1024 words.

<analysis> rules:
- Be concise; cover history evidence, taste signals, preference transition, and
  discriminative prefer/avoid cues.
- Use step-by-step analysis: compare positive evidence, weak or negative
  evidence, preference transition, and the final feature direction.
- Ground features in metadata: category, creator/brand/source, format/type/platform,
  description, attributes/details, rating or feedback.
- Do not infer unsupported creator, brand, region, era, platform, label/publisher,
  series/franchise, scene/community, or domain-specific subcategory from title tokens.
- Do not include task restatement, prompt analysis, candidate brainstorming,
  format discussion, or filler such as "let's", "maybe", "wait", "the prompt asks".

<answer> rules:
- Output one 20-45 word neutral noun phrase or item-profile sentence describing
  transferable item features only.
- Do not include non-feature text: explanations, evidence, user/history phrasing,
  examples, parentheses, recommendation/advice/action wording, item IDs, exact
  titles, or named entities copied from metadata.
- Do not use external appraisal, popularity, review, award, sales, rating, or
  catalog-stat signals. Describe intrinsic content, form, style, format, use-case,
  or mood features instead.
- Do not start with "The user is likely to be interested in".
- Describe only supported high-level category, subcategory/style,
  format/type/platform, use-case, content/design/production, and
  mood/tone/audience features.
- If history has one positive item, generalize to category or feature level;
  do not predict an exact creator, brand, series/franchise, title, or item.
- When evidence is sparse or dominated by one item, do not copy every specific
  attribute of that item into <answer>. Prefer a broader transferable feature
  description that covers adjacent styles, formats, or use-cases supported by
  metadata.
- Low-rated items indicate disliked item-specific features, not rejection of the
  whole high-level category. Do not exclude a broad category from one low rating.
- If all history is negative or low-rated, identify only metadata-supported
  disliked features in <analysis>; in <answer>, keep the description conservative,
  leave those specific features out, and do not introduce unsupported category
  or domain jumps.
"""

NO_RATING_COT_OUTPUT_INSTRUCTION = """\
Analyze the observed historical interactions and predict a feature description
for the next {category_singular} the user is likely to interact with.

The history contains observed interactions without user rating scores. Treat
each item only as observed behavior, not as positive or negative feedback.

Use concise internal reasoning covering observed interaction evidence,
content/style/format cues, preference transition, and discriminative cues.

For the visible final response, output only the transferable feature description.
Do not include explanations, evidence, user/history phrasing, rating language,
review language, popularity language, or catalog-stat language.
"""

TAGGED_NO_RATING_COT_OUTPUT_INSTRUCTION = """\
Analyze the observed historical interactions and predict a feature description
for the next {category_singular} the user is likely to interact with.

The history contains observed interactions without user rating scores. Treat
each item only as observed behavior, not as positive or negative feedback.

Output only:
<analysis>
...
</analysis>
<answer>
...
</answer>

Format rules:
- Use exact raw tags <analysis> and <answer>; do not use <think>, <reasoning>,
  <thoughts>, markdown, JSON, or any other wrapper.
- No text outside the two blocks; the first non-whitespace character must be "<".
- The combined length of <analysis> and <answer> must not exceed 1024 words.

<analysis> rules:
- Be concise; cover observed interaction evidence, content/style/format cues,
  preference transition, and discriminative cues.
- Do not describe any interaction as positive, negative, high-rated, low-rated,
  liked, disliked, or feedback-based.
- Ground features in metadata: category, creator/brand/source,
  format/type/platform, description, and attributes/details.
- Do not mention rating scores, star ratings, review, acclaim, award, popularity,
  catalog stats, avg_rating, or rating_count.
- Do not infer unsupported creator, brand, region, era, platform, label/publisher,
  series/franchise, scene/community, or domain-specific subcategory from title tokens.
- Do not include task restatement, prompt analysis, candidate brainstorming,
  format discussion, or filler such as "let's", "maybe", "wait", "the prompt asks".

<answer> rules:
- Output one concise, complete neutral item-profile sentence describing
  transferable item features only; do not return a bare category label,
  title-like fragment, or keyword list.
- Combine supported category/subcategory, format/type, content/style,
  production/mood, or use-case cues when the history provides them.
- Do not include non-feature text: explanations, evidence, user/history phrasing,
  examples, parentheses, recommendation/advice/action wording, item IDs, exact
  titles, or named entities copied from metadata.
- Do not use external appraisal, popularity, review, award, sales, rating, or
  catalog-stat signals. Describe intrinsic content, form, style, format, use-case,
  or mood features instead.
- Do not start with "The user is likely to be interested in".
- Describe only supported high-level category, subcategory/style,
  format/type/platform, use-case, content/design/production, and
  mood/tone/audience features.
- If history has one observed item, generalize to category or feature level;
  do not predict an exact creator, brand, series/franchise, title, or item.
- When evidence is sparse or dominated by one item, do not copy every specific
  attribute of that item into <answer>. Prefer a broader transferable feature
  description that covers adjacent styles, formats, or use-cases supported by
  metadata.
"""


def normalize_rating_context(value: str = "rating") -> str:
    normalized = (value or "rating").strip().lower().replace("-", "_")
    if normalized in {"rating", "ratings", "with_rating", "with_ratings"}:
        return "rating"
    if normalized in {"none", "no_rating", "no_ratings", "observed", "observed_only"}:
        return "no_rating"
    raise ValueError(f"Unsupported rating context: {value}")


def normalize_cot_tags(text: str) -> str:
    out = text or ""
    for tag in LEGACY_REASONING_TAGS:
        out = re.sub(rf"<\s*{tag}\s*>", f"<{REASONING_TAG}>", out, flags=re.IGNORECASE)
        out = re.sub(rf"<\s*/\s*{tag}\s*>", f"</{REASONING_TAG}>", out, flags=re.IGNORECASE)
    for tag in LEGACY_ANSWER_TAGS:
        out = re.sub(rf"<\s*{tag}\s*>", f"<{ANSWER_TAG}>", out, flags=re.IGNORECASE)
        out = re.sub(rf"<\s*/\s*{tag}\s*>", f"</{ANSWER_TAG}>", out, flags=re.IGNORECASE)
    out = re.sub(r"</?tool_call>", "", out, flags=re.IGNORECASE)
    return out.strip()


CATEGORY_LABELS = {
    "Video_Games": ("video games", "video game"),
    "CDs_and_Vinyl": ("CDs and vinyl", "CD or vinyl"),
    "Musical_Instruments": ("musical instruments", "musical instrument"),
}


def category_names(category: str = "") -> tuple[str, str]:
    if category in CATEGORY_LABELS:
        return CATEGORY_LABELS[category]
    if category:
        readable = category.replace("_", " ")
        return readable, readable.rstrip("s") or "item"
    return "items", "item"


def build_history_analysis_prompt(
    user_history: str,
    category: str = "",
    output_format: str = "answer_only",
    rating_context: str = "rating",
) -> str:
    category_plural, category_singular = category_names(category)
    rating_context = normalize_rating_context(rating_context)
    if output_format == "tagged":
        if rating_context == "no_rating":
            instruction = TAGGED_NO_RATING_COT_OUTPUT_INSTRUCTION
        else:
            instruction = TAGGED_COT_OUTPUT_INSTRUCTION
    elif output_format == "answer_only":
        if rating_context == "no_rating":
            instruction = NO_RATING_COT_OUTPUT_INSTRUCTION
        else:
            instruction = COT_OUTPUT_INSTRUCTION
    else:
        raise ValueError(f"Unsupported generation output format: {output_format}")
    return (
        "The following are the items the user has interacted with:\n"
        "<Historical Interactions>\n"
        f"{user_history.strip()}\n"
        "</Historical Interactions>\n\n"
        f"Category: {category_plural}.\n\n"
        f"{instruction.format(category_singular=category_singular).strip()}"
    )


JUDGE_SYSTEM = (
    "You score recommendation reasoning. Do not reason, explain, or comment. "
    "Return one compact JSON object only."
)

JUDGE_USER_TEMPLATE = """\
Score the candidate on five dimensions. Each value must be an integer from 1 to 5.

User history:
{user_history}

Candidate reasoning:
{cot}

{target_block}
Rules:
- Score reasoning quality and consistency with user history.
- Held-out target is only for leakage or severe contradiction checks; do not score by target similarity.
- Penalize direct target leakage if the candidate names the held-out title/name, creator/brand, item ID/ASIN, or exact item identity.
- Dimension meanings: preference_grounding=evidence from history; taste_specificity=concrete subcategories/styles/attributes/features; transitional_reasoning=clear shift from history to next interest; discriminative_framing=prefer/avoid boundary; conciseness=compact and information-dense.
- Output JSON only. No prose, markdown, chain-of-thought, explanation, or comment.

Return one JSON object only:
{{
  "preference_grounding": 1,
  "taste_specificity": 1,
  "transitional_reasoning": 1,
  "discriminative_framing": 1,
  "conciseness": 1
}}
"""


def build_generation_messages(
    user_history: str,
    category: str = "",
    output_format: str = "answer_only",
    rating_context: str = "rating",
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": COT_SYSTEM},
        {
            "role": "user",
            "content": build_history_analysis_prompt(
                user_history,
                category,
                output_format,
                rating_context,
            ),
        },
    ]


def build_user_prompt(user_history: str, category: str = "") -> str:
    return build_history_analysis_prompt(user_history, category)


def build_judge_messages(user_history: str, cot: str, target_item: str = "") -> list[dict[str, str]]:
    target_block = ""
    if target_item:
        target_block = (
            "Held-out positive item for evaluation only. Penalize direct leakage if "
            "the candidate appears to know it:\n"
            f"{target_item}\n\n"
        )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                user_history=user_history.strip(),
                cot=cot.strip(),
                target_block=target_block,
            ),
        },
    ]
