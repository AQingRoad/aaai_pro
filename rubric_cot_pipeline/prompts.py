from __future__ import annotations

import re

REASONING_TAG = "think"
ANSWER_TAG = "answer"
LEGACY_REASONING_TAGS = ("analysis", "thinking", "thoughts")
LEGACY_ANSWER_TAGS = ("recommendation",)

COT_SYSTEM = (
    "You are a recommendation master capable of deeply analyzing user "
    "preferences based on their historical interactions."
)

COT_OUTPUT_INSTRUCTION = """\
Please infer the user's explicit and latent preferences internally based on their historical behavior,
and predict the next {category_singular} the user is likely to be interested in purchasing.

Use concise internal reasoning that integrates these four aspects: history-grounded evidence, specific taste signals, preference transition, discriminative cues.

For the visible final response, output only the feature description of the next likely interested item.
Do not include any explanation, evidence, or phrases like "Based on the user's history".
"""


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


def build_history_analysis_prompt(user_history: str, category: str = "") -> str:
    category_plural, category_singular = category_names(category)
    return (
        "The following are the items the user has interacted with:\n"
        "<Historical Interactions>\n"
        f"{user_history.strip()}\n"
        "</Historical Interactions>\n\n"
        f"Category: {category_plural}.\n\n"
        f"{COT_OUTPUT_INSTRUCTION.format(category_singular=category_singular).strip()}"
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
- Penalize direct target leakage if the candidate names the held-out title, artist, ASIN, or exact item identity.
- Dimension meanings: preference_grounding=evidence from history; taste_specificity=concrete genres/styles/features; transitional_reasoning=clear shift from history to next interest; discriminative_framing=prefer/avoid boundary; conciseness=compact and information-dense.
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


def build_generation_messages(user_history: str, category: str = "") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": COT_SYSTEM},
        {"role": "user", "content": build_history_analysis_prompt(user_history, category)},
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
