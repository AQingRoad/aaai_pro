"""中英双语 QUERY-only 学生提示词，用于下一物品检索 CoT 的 SFT。"""

# 一个文件只维护一套逻辑提示词；英文用于当前英文数据，中文用于人工审阅。
PROMPT_NAME = "query_only_cot_student"
PROMPT_VERSION = "1.0"


SYSTEM_PROMPT_EN = """You are a reasoning model for next-item retrieval. Given QUERY only, infer one next-item profile grounded in the interaction history. Every judgment and output attribute must be reproducible from QUERY alone.

Output exactly one <think> block followed by one <answer> block, with no text outside them. In <think>, follow four steps: evidence inventory, signal weighting and clustering, pattern selection, and retrieval projection. Weigh repetition, recency, field agreement, and explicit feedback. In <answer>, write one concise item-profile sentence containing only attributes selected in the retrieval projection. Keep both blocks concise and use no outside knowledge."""


USER_PROMPT_TEMPLATE_EN = """<QUERY_DATA>
{query}
</QUERY_DATA>"""


SYSTEM_PROMPT_ZH = """你是面向下一物品检索的推理模型。请仅根据 QUERY 中的交互历史推断一个下一物品特征方向。每个判断和输出属性都必须能由 QUERY 独立复现。

严格输出一个 <think> 块和紧随其后的一个 <answer> 块，标签外禁止输出文本。<think> 依次完成证据清单、信号加权与聚类、模式选择和检索投影，并结合重复、近期程度、字段一致性和明确反馈判断信号强弱。<answer> 使用一句简洁的物品特征描述，只保留检索投影选中的属性。两部分均需简洁，禁止使用外部知识。"""


USER_PROMPT_TEMPLATE_ZH = """<QUERY_DATA>
{query}
</QUERY_DATA>"""


PROMPTS = {
    "en": (SYSTEM_PROMPT_EN, USER_PROMPT_TEMPLATE_EN),
    "zh": (SYSTEM_PROMPT_ZH, USER_PROMPT_TEMPLATE_ZH),
}


def build_messages(query: str, language: str = "en") -> list[dict[str, str]]:
    """将 QUERY 填入学生提示词，返回训练和推理共用的 messages。"""
    query = query.strip()
    if not query:
        raise ValueError("query 不能为空")
    if language not in PROMPTS:
        raise ValueError(f"不支持的提示词语言：{language}；可选值为 en、zh")
    system_prompt, user_template = PROMPTS[language]
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_template.format(query=query).strip()},
    ]
