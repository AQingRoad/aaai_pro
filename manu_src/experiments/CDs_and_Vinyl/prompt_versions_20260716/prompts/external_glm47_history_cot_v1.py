"""Archived GLM-4.7 prompts used by the legacy external-CoT experiment.

Training CoT used ``TAGGED_NO_RATING_USER_INSTRUCTION``. The historical test
CoT source used ``TAGGED_RATING_USER_INSTRUCTION``. The API normalized
``<analysis>`` to ``<think>`` before the tagged text entered embedding data.
"""

# 文件说明：
# 该文件保存旧版 external-CoT 实验调用 GLM-4.7 时使用的提示词。
# 训练集 CoT 使用 TAGGED_NO_RATING_USER_INSTRUCTION。
# 历史测试集 CoT 使用 TAGGED_RATING_USER_INSTRUCTION。
# API 返回结果进入 embedding 数据前，会把 <analysis> 标签统一改成 <think> 标签。

# 系统提示词：
# 将模型设定为推荐分析角色，要求模型深入分析用户的历史交互与偏好。
SYSTEM_PROMPT = (
    "You are a recommendation master capable of deeply analyzing user "
    "preferences based on their historical interactions."
)

# 无评分历史版本完整说明：
#
# 任务：
# 分析已经观察到的历史交互，预测用户下一次可能交互的 CD 或黑胶唱片特征描述。
# 历史记录不含用户评分。每条记录只表示用户与该物品发生过交互，不能把它解释为
# 正向反馈或负向反馈。
#
# 输出结构：
# 只输出以下两个原始标签块，并保持 <analysis> 在前、<answer> 在后：
# <analysis>...</analysis>
# <answer>...</answer>
#
# 格式规则：
# 1. 必须使用 <analysis> 和 <answer> 这两个原始标签。
# 2. 禁止改用 <think>、<reasoning>、<thoughts>、Markdown、JSON 或其它包装格式。
# 3. 两个标签块之外不能出现任何文本。
# 4. 去掉开头空白后，输出的第一个字符必须是“<”。
# 5. <analysis> 与 <answer> 的总长度不得超过 1024 个英文单词。
#
# <analysis> 规则：
# 1. 分析保持简洁，覆盖历史交互证据、内容/风格/载体线索、偏好变化和区分性线索。
# 2. 禁止把任何交互描述成正向、负向、高评分、低评分、喜欢、不喜欢或反馈信号。
# 3. 特征必须来自 metadata，允许使用品类、创作者/品牌/来源、格式/类型/平台、
#    描述和属性/详情。
# 4. 禁止提及评分数值、星级、评论、赞誉、奖项、流行度、目录统计、avg_rating
#    或 rating_count。
# 5. 禁止根据标题词元推断 metadata 没有支持的创作者、品牌、地区、年代、平台、
#    厂牌/出版商、系列/IP、场景/社群或领域细分类别。
# 6. 禁止复述任务、分析提示词、枚举候选、讨论输出格式，也不能使用“let's”、
#    “maybe”、“wait”、“the prompt asks”等填充语。
#
# <answer> 规则：
# 1. 只输出一句简洁、完整且语气中性的物品画像，描述可迁移的物品特征。
# 2. 禁止只输出宽泛品类名、类似标题的片段或关键词列表。
# 3. 历史证据充分时，可组合品类/细分类别、格式/类型、内容/风格、制作/氛围
#    或使用场景线索。
# 4. 禁止加入解释、证据、用户或历史措辞、示例、括号、推荐/建议/行动措辞、
#    物品 ID、精确标题或从 metadata 复制的命名实体。
# 5. 禁止使用外部评价、流行度、评论、奖项、销量、评分或目录统计信号。
# 6. 答案应描述物品内在的内容、形态、风格、格式、使用场景或氛围。
# 7. 禁止用“The user is likely to be interested in”作为开头。
# 8. 只描述证据支持的上位品类、细分类别/风格、格式/类型/平台、使用场景、
#    内容/设计/制作以及氛围/语气/受众特征。
# 9. 历史只有一个已观察物品时，需要概括到品类或特征层级，不能预测精确的
#    创作者、品牌、系列/IP、标题或具体物品。
# 10. 历史证据稀疏或主要由单个物品构成时，不能把该物品的每项具体属性复制到
#     <answer>；应概括 metadata 支持的相邻风格、格式或使用场景。
TAGGED_NO_RATING_USER_INSTRUCTION = """\
Analyze the observed historical interactions and predict a feature description for the
next {category_singular} the user is likely to interact with.

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

# 有评分历史版本完整说明：
#
# 任务：
# 分析包含评分或反馈的历史交互，预测用户下一次可能偏好或交互的 CD 或黑胶唱片
# 特征描述。
#
# 输出结构：
# 只输出 <analysis>...</analysis> 和 <answer>...</answer> 两个原始标签块。
#
# 格式规则：
# 1. 必须使用 <analysis> 和 <answer> 标签，禁止使用 <think>、<reasoning>、
#    <thoughts>、Markdown、JSON 或其它包装格式。
# 2. 标签块之外不能出现任何文本，去掉开头空白后的首字符必须是“<”。
# 3. <analysis> 与 <answer> 的总长度不得超过 1024 个英文单词。
#
# <analysis> 规则：
# 1. 分析保持简洁，覆盖历史证据、品味信号、偏好变化以及偏好/回避的区分线索。
# 2. 按步骤比较正向证据、弱信号或负向证据、偏好变化，再确定最终特征方向。
# 3. 特征必须落到 metadata 中的品类、创作者/品牌/来源、格式/类型/平台、描述、
#    属性/详情以及评分或反馈。
# 4. 禁止根据标题词元推断 metadata 没有支持的创作者、品牌、地区、年代、平台、
#    厂牌/出版商、系列/IP、场景/社群或领域细分类别。
# 5. 禁止复述任务、分析提示词、枚举候选、讨论输出格式，也不能使用“let's”、
#    “maybe”、“wait”、“the prompt asks”等填充语。
#
# <answer> 规则：
# 1. 输出一个 20 至 45 个英文单词的中性名词短语或物品画像句，只描述可迁移特征。
# 2. 禁止加入解释、证据、用户或历史措辞、示例、括号、推荐/建议/行动措辞、
#    物品 ID、精确标题或从 metadata 复制的命名实体。
# 3. 禁止使用外部评价、流行度、评论、奖项、销量、评分或目录统计信号。
# 4. 答案应描述物品内在的内容、形态、风格、格式、使用场景或氛围。
# 5. 禁止用“The user is likely to be interested in”作为开头。
# 6. 只描述证据支持的上位品类、细分类别/风格、格式/类型/平台、使用场景、
#    内容/设计/制作以及氛围/语气/受众特征。
# 7. 历史只有一个正向物品时，需要概括到品类或特征层级，不能预测精确的
#    创作者、品牌、系列/IP、标题或具体物品。
# 8. 历史证据稀疏或主要由单个物品构成时，不能把该物品的每项具体属性复制到
#    <answer>；应概括 metadata 支持的相邻风格、格式或使用场景。
# 9. 低评分表示用户回避该物品的具体特征，不能据此判定用户排斥整个上位品类。
# 10. 全部历史均为负向或低评分时，<analysis> 只能识别 metadata 支持的厌恶特征；
#     <answer> 需要采用保守描述，排除这些具体特征，禁止引入无证据的新类别或
#     跨领域偏好。
TAGGED_RATING_USER_INSTRUCTION = """\
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


def build_user_prompt(user_history: str, *, no_rating: bool) -> str:
    # no_rating=True 时选择无评分模板，历史交互只按“观察行为”处理。
    # no_rating=False 时选择有评分模板，允许模型使用评分或反馈区分偏好方向。
    instruction = (
        TAGGED_NO_RATING_USER_INSTRUCTION
        if no_rating
        else TAGGED_RATING_USER_INSTRUCTION
    )
    # 构造最终用户提示词：
    # 1. 用 <Historical Interactions> 标签包裹去除首尾空白后的历史文本。
    # 2. 将实验品类固定为 CDs and vinyl。
    # 3. 将模板中的 {category_singular} 替换为 CD or vinyl。
    # 4. strip() 只清理格式化后指令模板两端的空白，不改动模板正文。
    return (
        "The following are the items the user has interacted with:\n"
        "<Historical Interactions>\n"
        f"{user_history.strip()}\n"
        "</Historical Interactions>\n\n"
        "Category: CDs and vinyl.\n\n"
        f"{instruction.format(category_singular='CD or vinyl').strip()}"
    )


def build_messages(user_history: str, *, no_rating: bool) -> list[dict[str, str]]:
    # 按聊天补全接口的消息格式返回两条消息：
    # system 消息负责限定推荐分析角色；user 消息携带历史记录、品类和输出规则。
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(user_history, no_rating=no_rating)},
    ]
