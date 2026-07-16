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

TARGET_AWARE_SYSTEM = (
    "You are an expert in generating chain-of-thought reasoning for recommendation systems. "
    "You will receive a user's interaction history and information about the item with which the user actually "
    "interacts next. Generate a clear, history-grounded reasoning process that teaches a model to extract item "
    "preference features from interaction history. The next actual interaction item may only guide the selection "
    "of historical evidence. Every judgment and attribute in the analysis and answer must be independently "
    "derivable from the interaction history."
)

TARGET_AWARE_INSTRUCTION = """\
Use the information below to generate a recommendation-feature chain-of-thought example.

Tasks:
1. Identify categories, formats, styles, and content attributes in the history that
   are relevant to the subsequent interest direction.
2. Distinguish primary preferences from incidental features.
3. Derive broad and transferable item features from historical evidence.
4. Output the analysis and the final item features.

Information boundaries:
- Information about the next actual interaction item may only guide how historical
  evidence is selected and organized. It cannot serve as evidence for any judgment.
- Every judgment and attribute in the output must stand independently on the interaction
  history. Removing the next actual interaction item information must leave the reasoning
  conclusions unchanged.
- You may directly extract historical facts or reasonably generalize from one or more
  historical interactions. A generalization must remain within what the evidence supports.
- An attribute may be output even when it matches the next actual interaction item, provided
  that the analysis contains a complete and independent evidence chain from the history.
- Do not directly copy, rewrite, translate, summarize, or combine information from the next
  actual interaction item when that information lacks historical support.
- Do not use external knowledge to interpret names, entities, or descriptions, and do not
  add information absent from the input.
- Calibrate statement strength to the amount, consistency, and feedback of the evidence.
  Use broader wording when evidence is sparse, conflicting, or only indirect.
- When several attributes are individually supported, also check whether their combination
  identifies a specific item too narrowly. Reduce the number or specificity of attributes
  when necessary.
- Follow the historical evidence when it conflicts with the next actual interaction item.
- Do not mention information sources, the evidence-selection process, training labels,
  hidden information, or generation rules in the output.

<analysis> requirements:
- Use 4 to 6 sequential steps. Each step must make one clear judgment.
- Cite the history item numbers that support each judgment.
- Start from historical facts, then infer preferences and derive item features.
- You may infer broad attributes that do not appear verbatim in the history, but you must
  explain the supporting evidence.
- Calibrate the strength of each statement to the amount and consistency of evidence.
- Silently check that every attribute can be independently derived from the history. Do not
  describe this checking process in the output.

<answer> requirements:
- Write one concise, neutral, and transferable item-feature description.
- Retain only the attributes with the strongest historical support that are relevant to
  the subsequent interest direction.
- Describe only the item. Do not mention the user, history, training task, or recommendation action.
- Do not include titles, people, brands, series, identifiers, or proprietary descriptions
  that could directly identify a specific item.
- Do not include attributes available only from the next actual interaction item.
- Use broader wording when evidence is insufficient, and avoid stacking excessive detail.

Output only:
<analysis>
Step 1: ...
Step 2: ...
Step 3: ...
Step 4: ...
</analysis>
<answer>
One item-feature description supported by historical evidence.
</answer>

Do not output any text outside <analysis> and <answer>.
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


JUDGE_PROMPT_VERSION = "rubric_v2_ordinal_anchors"


JUDGE_SYSTEM = (
    "You are a strict evaluator for recommendation reasoning quality. "
    "Evaluate silently and return one compact JSON object only."
)

JUDGE_USER_TEMPLATE = """\
Score the candidate recommendation reasoning using only the provided user history and candidate reasoning.
Each value must be an integer from 1 to 5.

User history:
{user_history}

Candidate reasoning:
{cot}

{target_block}
General rules:
- Judge whether the reasoning extracts evidence from the history and turns it into a next-item preference.
- Do not reward similarity to any hidden answer, held-out positive, label, item ID, or future interaction.
- Penalize invented facts, direct leakage wording, target identity guesses, and claims unsupported by the history.
- Score the reasoning content, not grammar polish.
{target_rules}

Rubric anchors:

preference_grounding:
1 = ignores or contradicts the user history.
2 = mentions the history only in broad terms.
3 = uses one or two concrete history signals such as genre, format, creator, era, or rating pattern.
4 = uses several concrete history signals and separates strong evidence from weak evidence.
5 = synthesizes repeated signals across the history and accounts for exceptions or low-rated items when present.

taste_specificity:
1 = generic preference such as "music" or "popular items".
2 = broad category only, with little item-level detail.
3 = names concrete subcategories, styles, formats, eras, or product attributes.
4 = combines multiple specific attributes into a clear item profile.
5 = gives a tight profile with subgenre/style, format, era, edition, creator, or usage constraints grounded in the history.

transitional_reasoning:
1 = gives no path from history to next-item interest.
2 = jumps to a recommendation profile with little support.
3 = gives a plausible continuation from observed items.
4 = explains how concrete history signals imply the next-item attributes.
5 = handles mixed signals, chronology, repeated patterns, or negative evidence before forming the next-item profile.

discriminative_framing:
1 = gives no boundary between suitable and unsuitable items.
2 = gives a vague boundary such as "similar style".
3 = states at least one positive or negative boundary.
4 = clearly states preferred and avoided attributes using history evidence.
5 = defines a narrow inclusion/exclusion profile that would filter candidates in retrieval.

conciseness:
1 = empty, off-format, or mostly irrelevant.
2 = too short to justify the score or too long with repeated claims.
3 = understandable but contains avoidable repetition or loose wording.
4 = compact and covers the main evidence and next-item profile.
5 = dense, specific, and complete without repeated claims.

Output JSON only. No prose, markdown, chain-of-thought, explanation, or comment.

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


def build_target_aware_messages(
    user_history: str,
    target_item_title: str,
    target_item_text: str,
    category: str = "",
) -> list[dict[str, str]]:
    category_plural, _ = category_names(category)
    clean_history = re.sub(
        r"^This user's Amazon CDs and Vinyl interaction history over time is listed below\.",
        "Observed Amazon CDs and Vinyl interactions over time are listed below.",
        user_history.strip(),
        flags=re.IGNORECASE,
    )
    target_parts = []
    if target_item_title.strip():
        target_parts.append(f"Title: {target_item_title.strip()}")
    if target_item_text.strip():
        target_parts.append(f"Item text: {target_item_text.strip()}")
    target_block = "\n".join(target_parts)
    if not target_block:
        raise ValueError("Target-aware generation requires target title or target item text")
    user_prompt = (
        "<Historical Interactions>\n"
        f"{clean_history}\n"
        "</Historical Interactions>\n\n"
        "<Next Actual Interaction Item Information>\n"
        f"{target_block}\n"
        "</Next Actual Interaction Item Information>\n\n"
        f"Category: {category_plural}.\n\n"
        f"{TARGET_AWARE_INSTRUCTION.strip()}"
    )
    return [
        {"role": "system", "content": TARGET_AWARE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]


def build_user_prompt(user_history: str, category: str = "") -> str:
    return build_history_analysis_prompt(user_history, category)


def build_judge_messages(user_history: str, cot: str, target_item: str = "") -> list[dict[str, str]]:
    target_block = ""
    target_rules = ""
    if target_item:
        target_block = (
            "Held-out positive item for leakage check only. Do not score by target similarity:\n"
            f"{target_item}\n\n"
        )
        target_rules = (
            "- If the candidate names the held-out title, creator, item ID/ASIN, or exact item identity, "
            "set preference_grounding and discriminative_framing to at most 2.\n"
        )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                user_history=user_history.strip(),
                cot=cot.strip(),
                target_block=target_block,
                target_rules=target_rules,
            ),
        },
    ]
