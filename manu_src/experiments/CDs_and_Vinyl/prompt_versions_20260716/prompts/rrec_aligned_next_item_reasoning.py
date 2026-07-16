"""基于 RRec 的下一物品推理提示词；英文版本与中文对照放在同一文件。"""

from __future__ import annotations


PROMPT_NAME = "rrec_aligned_next_item_reasoning"
PROMPT_VERSION = "1.3"
OUTPUT_FORMAT = "rrec_freeform_answer"
USES_POSITIVE = False

# RRec 在该标记出现时停止生成，并把标记所在的末尾隐藏状态用作用户表示。
EMB_TOKEN = "<answer>"
EMB_END_TOKEN = "</answer>"
STOP_SEQUENCE = EMB_TOKEN

# 与 external/RRec/prompters/prompts.py 保持相同的数据集名称映射。
CATEGORY_NAMES = {
    "Video_Games": ("video games", "video game"),
    "CDs_and_Vinyl": ("CDs and vinyl", "CD or vinyl"),
    "Musical_Instruments": ("musical instruments", "musical instrument"),
}

CATEGORY_NAMES_ZH = {
    "Video_Games": ("电子游戏", "电子游戏"),
    "CDs_and_Vinyl": ("CD 和黑胶唱片", "CD 或黑胶唱片"),
    "Musical_Instruments": ("乐器", "乐器"),
}


# 在 RRec 原提示词上明确要求先输出分析，避免模型直接给出答案。
USER_PROMPT_TEMPLATE_EN = """Analyze in depth and finally recommend next {category_singular} I might purchase inside {emb_token} and {emb_end_token}.
Before {emb_token}, write a concise analysis grounded in the history. Avoid generic praise. Your response must contain non-empty analysis text before {emb_token}; an answer-only response is invalid.
For example, {emb_token}a product{emb_end_token}.

Below is my historical {category} purchases and ratings (out of 5):"""


# 中文版本只用于人工审阅或中文输入实验，不替代英文对齐实验。
USER_PROMPT_TEMPLATE_ZH = """深入分析，并最终在 {emb_token} 和 {emb_end_token} 之间推荐我接下来可能购买的{category_singular}。
在 {emb_token} 之前，先根据历史进行简要分析；避免泛泛夸赞。{emb_token} 之前必须包含非空分析文本，只输出答案视为格式错误。
例如，{emb_token}一个物品{emb_end_token}。

以下是我过去购买{category}的记录和评分（满分 5 分）："""


PROMPTS = {
    "en": (USER_PROMPT_TEMPLATE_EN, CATEGORY_NAMES),
    "zh": (USER_PROMPT_TEMPLATE_ZH, CATEGORY_NAMES_ZH),
}


def build_prompt(
    query: str,
    category: str = "CDs_and_Vinyl",
    language: str = "en",
) -> str:
    """将 RRec 提示词与现成 query 拼接，不重构或截断历史内容。"""
    query = query.strip()
    if not query:
        raise ValueError("query 不能为空")
    if language not in PROMPTS:
        raise ValueError(f"不支持的提示词语言：{language}；可选值为 en、zh")

    template, category_names = PROMPTS[language]
    if category not in category_names:
        choices = "、".join(category_names)
        raise ValueError(f"不支持的数据集类别：{category}；可选值为 {choices}")

    category_plural, category_singular = category_names[category]
    instruction = template.format(
        category=category_plural,
        category_singular=category_singular,
        emb_token=EMB_TOKEN,
        emb_end_token=EMB_END_TOKEN,
    )
    return f"{instruction}\n{query}"


def build_messages(
    query: str,
    positive: str | None = None,
    language: str = "en",
    category: str = "CDs_and_Vinyl",
) -> list[dict[str, str]]:
    """返回 RRec 对齐的单条 user message；positive 仅为兼容旧调用接口，不会进入提示词。"""
    del positive
    return [
        {
            "role": "user",
            "content": build_prompt(query=query, category=category, language=language),
        }
    ]
