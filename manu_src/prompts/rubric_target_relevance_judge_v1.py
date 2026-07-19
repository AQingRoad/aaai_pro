#!/usr/bin/env python3
"""Target-Relevance Rubric Judge 的中英文提示词，运行时默认使用英文。"""

from __future__ import annotations


SYSTEM_PROMPT_ZH = (
    "你是一名严格的推荐推理质量评估专家。"
    "请在内部完成评估，只返回一个紧凑的 JSON 对象。"
)


USER_PROMPT_TEMPLATE_ZH = """\
请根据给定的用户历史、候选推荐推理和用户随后实际交互的目标物品，对候选推荐推理进行评分。

每个评分维度必须使用 1 到 5 之间的整数。

<用户历史>
{{用户历史}}
</用户历史>

<候选推荐推理>
{{候选推荐推理}}
</候选推荐推理>

<后续目标物品>
{{目标物品信息}}
</后续目标物品>

评分要求：

1. 判断候选推理是否从用户历史中提取具体证据，并将其归纳为清晰的下一物品偏好特征。
2. 判断候选推导出的物品特征是否覆盖目标物品所代表的真实后续兴趣方向。
3. 同时考虑语义一致性。候选推理与目标物品使用不同表述，但表达相近的类别、风格、格式或内容方向时，也应视为匹配。
4. 对违背历史证据、添加缺少依据的属性，或偏离真实后续兴趣方向的推理降低评分。
5. 评分聚焦推理内容、偏好特征的具体程度和推导连贯性，不根据语法或表达风格额外加分。

请按照以下五个维度评分。

preference_grounding：

1 = 忽略用户历史，或者推理结论与历史证据矛盾。
2 = 只笼统提及用户历史，没有使用具体的历史交互信号。
3 = 使用一到两个具体的历史信号，例如类别、风格、格式、创作者、年代或评分模式。
4 = 综合多个具体的历史信号，并区分较强证据与较弱证据。
5 = 综合历史中重复出现的偏好信号，同时处理例外、兴趣变化、低评分物品或相互冲突的证据。

taste_specificity：

1 = 只给出“音乐”“电影”“热门物品”等缺少区分度的泛化偏好。
2 = 只描述宽泛类别，几乎没有物品层面的具体特征。
3 = 给出具体的子类别、风格、格式、年代或产品属性。
4 = 将多个有历史依据的具体属性组合成清晰的下一物品特征描述，并覆盖目标物品所体现的主要兴趣方向。
5 = 形成紧凑且具体的物品特征画像，既有充分的历史依据，也准确覆盖目标物品所体现的后续兴趣方向。

transitional_reasoning：

1 = 没有说明如何从用户历史推导下一物品兴趣。
2 = 直接跳到下一物品特征，缺少历史证据与结论之间的联系。
3 = 给出从已观察交互到下一物品兴趣的合理延续方向。
4 = 清楚说明具体历史信号如何支持下一物品可能具有的属性，并且推导方向与目标物品基本一致。
5 = 合理处理混合兴趣、时间变化、重复模式或评分差异，形成与目标物品所代表的后续兴趣高度一致的推导方向。

discriminative_framing：

1 = 没有提供区分匹配物品与不匹配物品的特征。
2 = 只给出“相似风格”“相关内容”等模糊边界。
3 = 至少给出一个明确的匹配特征或排除特征。
4 = 根据历史证据明确描述偏好属性和应避免的属性，能够缩小符合后续兴趣方向的候选范围。
5 = 给出具体且有历史依据的纳入与排除边界，能够在检索过程中区分目标兴趣方向与相似但不匹配的物品。

conciseness：

1 = 内容为空、格式错误，或大部分内容与推荐推理无关。
2 = 内容过短，无法支撑结论；或者内容过长并包含大量重复判断。
3 = 内容可以理解，但存在可以删除的重复、松散表达或无关内容。
4 = 表达紧凑，覆盖主要历史证据和下一物品特征。
5 = 信息密度高，描述具体且完整，没有重复判断或无关内容。

只输出 JSON。不要输出分析过程、解释、Markdown、代码块或附加说明。

严格使用以下格式：

{
  "preference_grounding": 1,
  "taste_specificity": 1,
  "transitional_reasoning": 1,
  "discriminative_framing": 1,
  "conciseness": 1
}
"""


SYSTEM_PROMPT_EN = (
    "You are a strict evaluator of recommendation reasoning quality. "
    "Evaluate internally and return only one compact JSON object."
)


USER_PROMPT_TEMPLATE_EN = """\
Score the candidate recommendation reasoning using the given user history, candidate reasoning, and the item with which the user actually interacted next.

Every scoring dimension must be an integer from 1 to 5.

<User History>
{{USER_HISTORY}}
</User History>

<Candidate Recommendation Reasoning>
{{CANDIDATE_REASONING}}
</Candidate Recommendation Reasoning>

<Actual Next Interaction Item>
{{TARGET_ITEM}}
</Actual Next Interaction Item>

Scoring requirements:

1. Determine whether the candidate extracts concrete evidence from the user history and turns it into a clear next-item preference profile.
2. Determine whether the inferred item features capture the actual subsequent interest direction represented by the target item.
3. Account for semantic consistency. Treat descriptions as matching when they express similar categories, styles, formats, or content directions even if the candidate and target use different wording.
4. Lower the score when the reasoning contradicts the history, introduces unsupported attributes, or points away from the actual subsequent interest direction.
5. Score the reasoning content, preference specificity, and inferential coherence. Do not give extra credit for grammar or writing style.

Score the candidate on the following five dimensions.

preference_grounding:

1 = Ignores the user history or reaches conclusions that contradict the historical evidence.
2 = Refers to the history only in broad terms and uses no concrete interaction signals.
3 = Uses one or two concrete historical signals, such as category, style, format, creator, era, or rating pattern.
4 = Integrates several concrete historical signals and distinguishes stronger evidence from weaker evidence.
5 = Synthesizes repeated preference signals while accounting for exceptions, interest shifts, low-rated items, or conflicting evidence.

taste_specificity:

1 = Gives only generic preferences such as "music," "movies," or "popular items."
2 = Describes only a broad category with almost no item-level features.
3 = Identifies concrete subcategories, styles, formats, eras, or product attributes.
4 = Combines multiple history-supported attributes into a clear next-item profile that covers the main interest direction reflected by the target item.
5 = Produces a compact and specific item profile that is well grounded in the history and accurately covers the subsequent interest direction reflected by the target item.

transitional_reasoning:

1 = Provides no explanation of how the next-item interest follows from the user history.
2 = Jumps directly to next-item features without connecting historical evidence to the conclusion.
3 = Gives a plausible continuation from the observed interactions to a next-item interest direction.
4 = Clearly explains how concrete historical signals support the inferred item attributes, and the resulting direction is broadly consistent with the target item.
5 = Handles mixed interests, temporal changes, repeated patterns, or rating differences and derives a direction highly consistent with the subsequent interest represented by the target item.

discriminative_framing:

1 = Provides no features that distinguish matching items from non-matching items.
2 = Gives only vague boundaries such as "similar style" or "related content."
3 = States at least one clear matching feature or exclusion feature.
4 = Uses historical evidence to state preferred and avoided attributes clearly enough to narrow the candidate set toward the subsequent interest direction.
5 = Gives specific, history-grounded inclusion and exclusion boundaries that distinguish the target interest direction from similar but non-matching items during retrieval.

conciseness:

1 = Is empty, malformed, or mostly irrelevant to recommendation reasoning.
2 = Is too short to support its conclusion, or too long and highly repetitive.
3 = Is understandable but contains removable repetition, loose wording, or irrelevant content.
4 = Is compact and covers the main historical evidence and next-item features.
5 = Is dense, specific, and complete without repeated judgments or irrelevant content.

Output JSON only. Do not output analysis, explanations, Markdown, code fences, or additional text.

Use exactly the following format:

{
  "preference_grounding": 1,
  "taste_specificity": 1,
  "transitional_reasoning": 1,
  "discriminative_framing": 1,
  "conciseness": 1
}
"""


# Backward-compatible aliases point to the runtime default: English.
SYSTEM_PROMPT = SYSTEM_PROMPT_EN
USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE_EN


RUBRIC_DIMENSIONS = (
    "preference_grounding",
    "taste_specificity",
    "transitional_reasoning",
    "discriminative_framing",
    "conciseness",
)


def build_judge_messages(
    user_history: str,
    candidate_reasoning: str,
    target_item: str,
    language: str = "en",
) -> list[dict[str, str]]:
    """构造 Chat Completions 消息；默认使用英文提示词。"""
    normalized_language = str(language or "en").strip().lower()
    if normalized_language in {"en", "eng", "english"}:
        system_prompt = SYSTEM_PROMPT_EN
        user_prompt = USER_PROMPT_TEMPLATE_EN
        values = {
            "{{USER_HISTORY}}": str(user_history or "").strip(),
            "{{CANDIDATE_REASONING}}": str(candidate_reasoning or "").strip(),
            "{{TARGET_ITEM}}": str(target_item or "").strip(),
        }
    elif normalized_language in {"zh", "cn", "chinese"}:
        system_prompt = SYSTEM_PROMPT_ZH
        user_prompt = USER_PROMPT_TEMPLATE_ZH
        values = {
            "{{用户历史}}": str(user_history or "").strip(),
            "{{候选推荐推理}}": str(candidate_reasoning or "").strip(),
            "{{目标物品信息}}": str(target_item or "").strip(),
        }
    else:
        raise ValueError(f"不支持的提示词语言: {language}")
    for placeholder, value in values.items():
        user_prompt = user_prompt.replace(placeholder, value)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


__all__ = [
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "SYSTEM_PROMPT_EN",
    "USER_PROMPT_TEMPLATE_EN",
    "SYSTEM_PROMPT_ZH",
    "USER_PROMPT_TEMPLATE_ZH",
    "RUBRIC_DIMENSIONS",
    "build_judge_messages",
]
