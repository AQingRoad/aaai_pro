#!/usr/bin/env python3
"""manu_src 后续 CoT 数据生成、训练和推理共用的提示词。"""

from __future__ import annotations


GENERAL_RECOMMENDATION_COT_SYSTEM_EN = (
    "You are a recommendation reasoning expert skilled at understanding user preferences "
    "from historical interactions and inferring the features of items they may be interested in next."
)


GENERAL_RECOMMENDATION_COT_USER_EN = """\
Analyze the historical interactions below and infer the key features that the next {item_type} the user may be interested in or interact with should have.
The reasoning will be appended to the user history for next-item retrieval, so retain concrete, retrievable, and discriminative information.

<Historical Interactions>
{user_history}
</Historical Interactions>

Follow these requirements:

1. Extract preference signals directly reflected in the historical interactions and infer latent preferences supported by the history.
2. Analyze how the user's interests may continue, change, or develop, and form a coherent reasoning path from historical preferences to next-item features.
3. Ground every judgment in the historical interactions and avoid adding attributes, relationships, or external information that lack historical support.
4. Identify the key features the next item may have and retain cues that distinguish matching items from non-matching items.
5. Keep the reasoning concise. Do not repeat the full history, list unrelated interest directions, or guess a specific item title.
6. The combined content inside <analysis> and <answer> must not exceed 512 words.

Output strictly in the following format:

<analysis>
Briefly analyze the relevant historical evidence and user preferences, determine how the interests may continue or change, and infer the key features of the next item.
</analysis>
<answer>
Use one concise and transferable feature description to summarize the key features that the user's next item of interest should have.
</answer>
"""


GENERAL_RECOMMENDATION_COT_SYSTEM_ZH = (
    "你是一名推荐系统推理专家，擅长根据用户的历史交互理解其偏好，"
    "并推导用户下一步可能感兴趣的物品特征。"
)


GENERAL_RECOMMENDATION_COT_USER_ZH = """\
请分析以下历史交互，推断用户下一个可能感兴趣或发生交互的{item_type}应具备哪些关键特征。
推理结果将与用户历史拼接并用于下一物品检索，因此需要保留具体、可检索且具有区分度的信息。

<历史交互>
{user_history}
</历史交互>

请遵循以下要求：

1. 从历史交互中提取直接呈现的偏好信号，并推断有历史依据的潜在偏好。
2. 分析用户兴趣的延续、变化和可能的发展方向，建立从历史偏好到下一物品特征的连贯推理。
3. 所有判断都应以历史交互为依据，避免补充缺少历史支持的属性、关系或外部信息。
4. 提炼下一物品可能具有的关键特征，并保留有助于区分匹配物品与不匹配物品的信息。
5. 保持推理简洁，不复述完整历史，不罗列互不相关的兴趣方向，不猜测具体物品标题。
6. <analysis> 与 <answer> 中的内容合计不得超过 512 words。

严格按以下格式输出：

<analysis>
简洁分析相关历史证据和用户偏好，判断兴趣的延续或变化，并推导下一物品可能具有的关键特征。
</analysis>
<answer>
用一句简洁、可迁移的特征描述，概括用户下一个可能感兴趣的物品应具备的关键特征。
</answer>
"""


TARGET_AWARE_RECOMMENDATION_COT_SYSTEM_EN = (
    "You are a recommendation reasoning expert. You will be given a user's historical interactions "
    "and the target item the user actually interacted with next."
)


TARGET_AWARE_RECOMMENDATION_COT_USER_EN = """\
Generate recommendation reasoning that explains how the key features of a possible next item can be derived from the user's historical interactions.
The target item may only help identify historical signals relevant to the user's subsequent interest.

<Historical Interactions>
{user_history}
</Historical Interactions>

<Subsequent Target Item>
{target_item_info}
</Subsequent Target Item>

Follow these requirements:

1. Extract preference signals related to the subsequent interest direction from the historical interactions and infer latent preferences supported by the history.
2. Every judgment and feature in <analysis> and <answer> must be independently supported by the historical interactions. Information from the target item that lacks historical support must not enter the output.
3. The preference judgments and item features in the reasoning must remain valid if the target item information is removed.
4. When the target item conflicts with the historical evidence, follow the historical interactions.
5. Do not mention the target item, ground-truth answer, training label, or generation rules in the output.
6. Keep the reasoning concise. Do not repeat the full history or guess a specific item title.
7. The combined content inside <analysis> and <answer> must not exceed 512 words.

Output strictly in the following format:

<analysis>
Analyze the historical evidence and user preferences related to the subsequent interest direction, and infer next-item features that are supported by the history.
</analysis>
<answer>
Use one concise and transferable feature description to summarize the next-item features supported by the historical evidence.
</answer>
"""


TARGET_AWARE_RECOMMENDATION_COT_SYSTEM_ZH = (
    "你是一名推荐系统推理专家。你将看到用户的历史交互，以及用户随后实际交互的目标物品。"
)


TARGET_AWARE_RECOMMENDATION_COT_USER_ZH = """\
请生成一段推荐推理，说明如何从历史交互中推导下一物品可能具有的关键特征。
目标物品只能帮助识别与后续兴趣相关的历史信号。

<历史交互>
{user_history}
</历史交互>

<后续目标物品>
{target_item_info}
</后续目标物品>

请遵循以下要求：

1. 从历史交互中提取与后续兴趣方向相关的偏好信号，并推断有历史依据的潜在偏好。
2. <analysis> 和 <answer> 中的所有判断与特征都必须由历史交互独立支撑。目标物品中缺少历史依据的信息不得进入输出。
3. 删除目标物品信息后，推理中的偏好判断和物品特征仍应成立。
4. 当目标物品与历史证据不一致时，以历史交互为准。
5. 不在输出中提及目标物品、真实答案、训练标签或生成规则。
6. 保持推理简洁，不复述完整历史，不猜测具体物品标题。
7. <analysis> 与 <answer> 中的内容合计不得超过 512 words。

严格按以下格式输出：

<analysis>
分析与后续兴趣方向相关的历史证据和用户偏好，并推导具有历史依据的下一物品关键特征。
</analysis>
<answer>
用一句简洁、可迁移的特征描述，概括历史证据支持的下一物品关键特征。
</answer>
"""


SYSTEM_PROMPTS = {
    "en": GENERAL_RECOMMENDATION_COT_SYSTEM_EN,
    "zh": GENERAL_RECOMMENDATION_COT_SYSTEM_ZH,
}

USER_PROMPTS = {
    "en": GENERAL_RECOMMENDATION_COT_USER_EN,
    "zh": GENERAL_RECOMMENDATION_COT_USER_ZH,
}

TARGET_AWARE_SYSTEM_PROMPTS = {
    "en": TARGET_AWARE_RECOMMENDATION_COT_SYSTEM_EN,
    "zh": TARGET_AWARE_RECOMMENDATION_COT_SYSTEM_ZH,
}

TARGET_AWARE_USER_PROMPTS = {
    "en": TARGET_AWARE_RECOMMENDATION_COT_USER_EN,
    "zh": TARGET_AWARE_RECOMMENDATION_COT_USER_ZH,
}

# 未指定语言时使用英文版本。
GENERAL_RECOMMENDATION_COT_SYSTEM = GENERAL_RECOMMENDATION_COT_SYSTEM_EN
GENERAL_RECOMMENDATION_COT_USER = GENERAL_RECOMMENDATION_COT_USER_EN
TARGET_AWARE_RECOMMENDATION_COT_SYSTEM = TARGET_AWARE_RECOMMENDATION_COT_SYSTEM_EN
TARGET_AWARE_RECOMMENDATION_COT_USER = TARGET_AWARE_RECOMMENDATION_COT_USER_EN


def normalize_language(language: str = "en") -> str:
    """把常见语言名称归一化为 en 或 zh。"""
    normalized = (language or "en").strip().lower().replace("_", "-")
    if normalized in {"en", "english", "en-us", "en-gb"}:
        return "en"
    if normalized in {"zh", "chinese", "zh-cn", "zh-hans"}:
        return "zh"
    raise ValueError(f"不支持的提示词语言: {language}")


def build_general_recommendation_cot_prompt(
    user_history: str,
    item_type: str,
    language: str = "en",
) -> str:
    """填入历史和物品类型，默认返回英文用户提示词。"""
    history = user_history.strip()
    item = item_type.strip()
    if not history:
        raise ValueError("user_history 不能为空")
    if not item:
        raise ValueError("item_type 不能为空")
    template = USER_PROMPTS[normalize_language(language)]
    return template.format(
        user_history=history,
        item_type=item,
    )


def build_general_recommendation_cot_messages(
    user_history: str,
    item_type: str,
    language: str = "en",
) -> list[dict[str, str]]:
    """返回 chat-completions messages，默认使用英文提示词。"""
    normalized_language = normalize_language(language)
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[normalized_language]},
        {
            "role": "user",
            "content": build_general_recommendation_cot_prompt(
                user_history,
                item_type,
                language=normalized_language,
            ),
        },
    ]


def build_target_aware_recommendation_cot_prompt(
    user_history: str,
    target_item_info: str,
    language: str = "en",
) -> str:
    """填入历史和目标物品信息，默认返回英文 target-aware 用户提示词。"""
    history = user_history.strip()
    target = target_item_info.strip()
    if not history:
        raise ValueError("user_history 不能为空")
    if not target:
        raise ValueError("target_item_info 不能为空")
    template = TARGET_AWARE_USER_PROMPTS[normalize_language(language)]
    return template.format(
        user_history=history,
        target_item_info=target,
    )


def build_target_aware_recommendation_cot_messages(
    user_history: str,
    target_item_info: str,
    language: str = "en",
) -> list[dict[str, str]]:
    """返回 target-aware chat-completions messages，默认使用英文提示词。"""
    normalized_language = normalize_language(language)
    return [
        {
            "role": "system",
            "content": TARGET_AWARE_SYSTEM_PROMPTS[normalized_language],
        },
        {
            "role": "user",
            "content": build_target_aware_recommendation_cot_prompt(
                user_history,
                target_item_info,
                language=normalized_language,
            ),
        },
    ]
