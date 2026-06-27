# Pipeline 实现状态梳理

更新时间：2026-06-15

这份文档用于核对当前推荐 CoT pipeline 的实现状态。重点说明：目前代码已经把主流程机械打通，但最近的运行主要是小样本 smoke run，尚未形成论文级最终实验结果。

## 1. 总体流程

当前实现基本对应 `paper/notes/method.md` 中的训练流程：

1. Phase 0：训练或加载推荐 embedder。
2. Phase 1：构造 CoT 数据，包括候选 CoT 生成、API/rule judge 打分、CoT gain 计算和筛选。
3. Phase 2：把筛选后的 CoT 转成 SFT 数据，用 LoRA 微调 reasoner。
4. Phase 3：构造 GRPO 数据，用 rubric-gated reward 做强化学习。
5. Evaluation：比较 `user_history` baseline 检索与 `user_history + CoT` 增强检索。

当前代码路径已经可以端到端跑通；但大部分产物是 smoke artifacts，只验证 pipeline 机制。

## 2. 数据集与 Benchmark

当前 benchmark 对齐 R2ec/RRec 的 Amazon 设置。

服务器上的 RRec 数据根目录：

```bash
/root/autodl-tmp/rec/RRec_official/data
```

已接入的 Amazon 类别：

```text
Musical_Instruments
Video_Games
CDs_and_Vinyl
```

每个类别的数据目录格式：

```text
<category>_0_2022-10-2023-10
```

当前 Phase 1 脚本默认参数：

```bash
CATEGORY=Musical_Instruments
SPLIT=train
MAX_EXAMPLES=1000
NUM_CANDIDATES=4
MAX_HISTORY_ITEMS=20
```

## 3. 当前模型配置

### 3.1 Reasoner

method 目标模型：

```text
Qwen3-4B
```

当前 Qwen3-4B SFT smoke 使用的服务器路径：

```bash
/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B
```

当前 `scripts/pipelines/run_phase1_full_rrec_amazon.sh` 默认用于候选 CoT 生成的模型也是：

```bash
/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B
```

如需显式指定，也应传入同一路径：

```bash
MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B
```

### 3.2 Embedder

默认 embedder：

```bash
/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B
```

相关配置：

```bash
QWEN3_EMBEDDING_MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B
GAIN_EMBEDDER_MODE=qwen3_embedding
GRPO_BASELINE_EMBEDDER_MODE=qwen3_embedding
RUBRIC_GAIN_EMBEDDER_MODE=qwen3_embedding
```

### 3.3 API Judge

默认 API judge 已改为快手 TokenVerse，通过本地 OpenAI-compatible tunnel 访问，不再默认使用智谱。

```bash
RUBRIC_JUDGE_API_PROVIDER=openai_compatible
RUBRIC_JUDGE_API_BASE_URL=http://127.0.0.1:18080/v1
RUBRIC_JUDGE_API_MODEL=glm-5-1
API_WORKERS=1
```

这要求本地 proxy/tunnel 已经启动。如果没有 tunnel，API judge 会失败，除非显式改成 mock 或其他可用服务。

## 4. Prompt 与 CoT 标签格式

当前 pipeline 把完整结构化 CoT 视为 method 中的变量 `r`。

Reasoner 被训练为输出严格格式：

```text
<think>
基于用户历史的推荐推理过程
</think>
<answer>
用于下游检索的浓缩偏好画像
</answer>
```

当前策略：

1. SFT 训练完整 `<think> + <answer>` assistant message。
2. Rubric scoring 会解析 `think` 和 `answer`，并将二者合并后评价。
3. CoT gain 使用 `user_history + full completion`。
4. GRPO reward 使用 `user_history + full completion`。
5. Evaluation 使用 `user_history + full generated completion`。
6. 不做 `think-only` 或 `answer-only` 消融。

标签清洗在以下脚本中实现：

```text
scripts/datasets/make_sft_dataset.py
```

它会移除嵌套的 `<think>` / `<answer>` 标签，并强制每条 SFT 样本只有一组合法标签。

## 5. 文本拼接与格式统一

用户历史文本由以下函数统一构造：

```text
scripts/data/prepare_rrec_amazon_examples.py::history_text
```

该函数复用于：

1. Phase 1 examples。
2. Phase 0 embedder 训练样本。
3. 检索与评测 query。

Qwen3-Embedding 的 query instruction 包装集中在：

```text
rubric_cot_pipeline/embeddings.py::format_qwen3_query
```

当前 query instruction：

```text
Given a user's past item interactions and optional recommendation reasoning, retrieve items the user is likely to prefer next.
```

CoT/reasoning 拼接集中在：

```text
rubric_cot_pipeline/embeddings.py::append_recommendation_reasoning
```

拼接格式：

```text
{user_history}

Recommendation reasoning:
{full_structured_cot}
```

当前 gain、reward、proxy evaluation、reasoner evaluation 都使用这个公共函数，避免不同阶段 prompt 拼接不一致。

## 6. Phase 0：Embedder 训练

已实现脚本：

```text
scripts/data/make_phase0_embedder_dataset.py
scripts/embedding/train_phase0_embedder.py
```

Phase 0 数据行结构：

```json
{
  "query": "user history text",
  "positive": "target item text",
  "category": "...",
  "split": "train",
  "user_id": "...",
  "target_item_id": 0,
  "target_item_title": "...",
  "target_rating": 5.0
}
```

训练目标：

```text
InfoNCE with in-batch negatives
query = format_qwen3_query(user_history)
document = target item text
loss = cross_entropy(query_emb @ doc_emb.T / temperature, diagonal_labels)
```

服务器 smoke 输出：

```text
/root/autodl-tmp/rec/aaai_pro/data/rrec_amazon/phase0_embedder_tiny.jsonl
/root/autodl-tmp/rec/aaai_pro/checkpoints/phase0_qwen3_embedding_rrec_tiny/checkpoint-1
```

smoke 统计：

```text
phase0 rows: 48
training steps: 1
loss: 0.461817
batch_acc: 1.0
```

注意：这个 checkpoint 只是验证代码路径，不是有效的预训练 embedder。

## 7. Phase 1：CoT 数据构造 Pipeline

这一阶段的目标是从 RRec/Amazon 训练样本构造两类训练数据：

1. SFT 数据：高质量 `<think>...</think><answer>...</answer>` CoT 监督数据。
2. GRPO 数据：只包含 prompt、target item 和 baseline similarity，供 RL 阶段在线生成 completion 并打 reward。

主 orchestration 脚本：

```text
scripts/pipelines/run_phase1_full_rrec_amazon.sh
```

默认关键参数：

```bash
MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B
QWEN3_EMBEDDING_MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B
JUDGE_MODE=api
API_PROVIDER=openai_compatible
API_BASE_URL=http://127.0.0.1:18080/v1
API_MODEL=glm-5-1
API_WORKERS=1
GAIN_EMBEDDER_MODE=qwen3_embedding
GRPO_BASELINE_EMBEDDER_MODE=qwen3_embedding
```

因此，当前数据构造阶段默认是：

```text
Qwen3-4B 生成候选 CoT
快手 TokenVerse glm-5-1 做 API Judge
Qwen3-Embedding-0.6B 计算 CoT Gain 和 GRPO baseline
```

### 7.0 总体数据流

```text
RRec HuggingFace dataset
  -> phase1_examples.jsonl
  -> cot_candidates.jsonl
  -> cot_judged.jsonl
  -> cot_scored.jsonl
  -> filtered_high_quality_cot.jsonl + rejected_cot.jsonl
  -> sft.jsonl

phase1_examples.jsonl
  -> grpo.jsonl
```

其中 `sft.jsonl` 用于 Phase 2 SFT，`grpo.jsonl` 用于 Phase 3 GRPO。

### 7.1 准备 RRec examples

脚本：

```text
scripts/data/prepare_rrec_amazon_examples.py
```

输出：

```text
data/rrec_amazon/<CATEGORY>/phase1_examples.jsonl
```

输入来自 RRec 官方预处理后的 HuggingFace dataset：

```text
/root/autodl-tmp/rec/RRec_official/data/<CATEGORY>_0_2022-10-2023-10
```

每条 `phase1_examples.jsonl` 的关键字段：

```json
{
  "example_id": "Musical_Instruments:train:<interaction_id>:<user_id>",
  "dataset": "rrec-amazon-2023",
  "category": "Musical_Instruments",
  "split": "train",
  "user_id": "...",
  "target_item_id": 123,
  "target_item_title": "...",
  "target_item_text": "...",
  "target_rating": 5.0,
  "history_item_ids": [1, 2, 3],
  "history_item_count": 20,
  "user_history": "This user's Amazon ... interaction history over time is listed below. ..."
}
```

这里的 `user_history` 由统一函数构造：

```text
scripts/data/prepare_rrec_amazon_examples.py::history_text
```

注意：候选 CoT 生成时只使用 `user_history`，不把 `target_item_text` 暴露给 generator。`target_item_text` 只用于 judge / gain / reward / evaluation。

### 7.2 生成候选 CoT

脚本：

```text
scripts/cot/generate_cot_candidates.py
```

输出：

```text
outputs/rrec_amazon/<CATEGORY>/cot_candidates.jsonl
```

默认生成模型：

```text
Qwen3-4B
```

生成 prompt 来自：

```text
rubric_cot_pipeline/prompts.py::build_generation_messages
```

要求模型输出结构化 CoT：

```text
<think>
基于用户历史的推荐推理
</think>
<answer>
浓缩偏好画像
</answer>
```

候选生成默认每个用户生成：

```bash
NUM_CANDIDATES=4
temperatures=0.6,0.8,1.0
```

关键输出字段：

```json
{
  "example_id": "...",
  "user_id": "...",
  "user_history": "...",
  "target_item_title": "...",
  "target_item_text": "...",
  "candidate_id": 0,
  "temperature": 0.6,
  "generator_model": "/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B",
  "cot": "<think>...</think><answer>...</answer>"
}
```

### 7.3 Judge CoT 质量

脚本：

```text
scripts/cot/judge_cot_quality.py
```

默认 provider：

```text
openai_compatible -> http://127.0.0.1:18080/v1 -> glm-5-1
```

也就是通过本地 reverse tunnel 调快手 TokenVerse 的 `glm-5-1`。

Judge prompt 使用 5 个推荐专用 rubric 维度：

1. Preference Grounding：是否基于用户历史。
2. Taste Specificity：是否有具体偏好维度。
3. Transitional Reasoning：是否有从已知到未知的迁移推理。
4. Discriminative Framing：是否能区分应选/应避免特征。
5. Conciseness：是否简洁、有信息密度。

API judge 返回每个维度 1-5 分。代码会归一化出：

```text
rubric_total = 五个维度总分，范围 5-25
rubric_score_norm = rubric_total / 25
```

输出：

```text
outputs/rrec_amazon/<CATEGORY>/cot_judged.jsonl
```

关键新增字段：

```json
{
  "preference_grounding": 4,
  "taste_specificity": 4,
  "transitional_reasoning": 3,
  "discriminative_framing": 4,
  "conciseness": 5,
  "rubric_total": 20,
  "rubric_score_norm": 0.8,
  "judge_mode_used": "api_openai_compatible",
  "judge_raw": "..."
}
```

如果 API 失败，`judge_cot_quality.py` 会 fallback 到本地 rules 并在 `judge_mode_used` 中标记 fallback。

### 7.4 计算 CoT Gain

脚本：

```text
scripts/selection/compute_cot_gain.py
```

gain 公式：

```text
cot_gain = sim(user_history + full_cot, target_item) - sim(user_history, target_item)
```

其中：

```text
full_cot = 完整 <think> + <answer>
user_history + full_cot = append_recommendation_reasoning(user_history, full_cot)
sim = Qwen3-Embedding cosine similarity
```

当前默认用 Qwen3-Embedding：

```bash
GAIN_EMBEDDER_MODE=qwen3_embedding
```

Qwen3-Embedding query 会统一包成：

```text
Instruct: Given a user's past item interactions and optional recommendation reasoning, retrieve items the user is likely to prefer next.
Query: ...
```

document/item 侧不加 instruction。

输出：

```text
outputs/rrec_amazon/<CATEGORY>/cot_scored.jsonl
```

关键新增字段：

```json
{
  "baseline_sim": 0.61,
  "cot_sim": 0.65,
  "cot_gain": 0.04,
  "embedder_mode": "qwen3_embedding"
}
```

### 7.5 筛选高质量 CoT

脚本：

```text
scripts/selection/select_filtered_cot.py
```

当前默认筛选策略：

```text
top-k = 1
min_rubric = 0.5
min_gain = 0.0
fallback_when_empty = true
```

选择分数：

```text
selection_score = rubric_score_norm * max(cot_gain, 0)
```

筛选过程：

1. 按 `example_id` 分组，每个用户/样本有多条候选 CoT。
2. 按 `selection_score`、`rubric_score_norm`、`cot_gain` 降序排序。
3. 保留满足 `rubric_score_norm >= min_rubric` 且 `cot_gain >= min_gain` 的 top-k。
4. 如果某个样本没有候选通过，且 `fallback_when_empty=true`，则保留该组排序最高的一条，并标记 `fallback_selected=true`。

输出：

```text
outputs/rrec_amazon/<CATEGORY>/filtered_high_quality_cot.jsonl
outputs/rrec_amazon/<CATEGORY>/rejected_cot.jsonl
```

`filtered_high_quality_cot.jsonl` 用于构造 SFT 数据；`rejected_cot.jsonl` 保留 rejected candidates，后续可用于分析或 DPO。

### 7.6 转换为 SFT 数据

脚本：

```text
scripts/datasets/make_sft_dataset.py
```

输出：

```text
outputs/rrec_amazon/<CATEGORY>/sft.jsonl
```

SFT 数据格式是 ms-swift chat messages：

```json
{
  "user_id": "...",
  "candidate_id": 0,
  "rubric_total": 20,
  "cot_gain": 0.04,
  "selection_score": 0.032,
  "messages": [
    {"role": "system", "content": "...recommendation reasoning model..."},
    {"role": "user", "content": "<user_history> + 输出格式要求"},
    {"role": "assistant", "content": "<think>...</think><answer>...</answer>"}
  ]
}
```

`make_sft_dataset.py` 会清洗 assistant 内容：

1. 如果已有合法 `<think>/<answer>`，去掉内部嵌套标签。
2. 如果没有合法标签，则把原始 CoT 包进 `<think>`，并用已有 answer 或默认 answer 填充 `<answer>`。
3. 最终每条 SFT 样本只保留一组 `<think>` 和一组 `<answer>`。

### 7.7 转换为 GRPO 数据

脚本：

```text
scripts/datasets/make_grpo_dataset.py
```

输出：

```text
outputs/rrec_amazon/<CATEGORY>/grpo.jsonl
```

GRPO 数据不包含 teacher CoT。它只提供 prompt、用户历史、target item 和可选 baseline similarity；训练时由当前 policy 在线生成 completion。

关键字段：

```json
{
  "user_id": "...",
  "dataset": "rrec-amazon-2023",
  "source_prompt": "...",
  "user_history": "...",
  "target_item_title": "...",
  "target_item_text": "...",
  "baseline_sim": 0.61,
  "baseline_embedder_mode": "qwen3_embedding",
  "messages": [
    {"role": "system", "content": "...recommendation reasoning model..."},
    {"role": "user", "content": "<user_history> + 输出格式要求"}
  ]
}
```

当前默认：

```bash
GRPO_BASELINE_EMBEDDER_MODE=qwen3_embedding
```

所以 `make_grpo_dataset.py` 会预计算：

```text
baseline_sim = sim(user_history, target_item)
```

如果 GRPO reward 发现数据中没有 baseline，或 baseline 模式不是当前 reward 模式，会在线重新计算 baseline。

### 7.8 数据构造的最终产物

每个 category 完整跑完后，核心产物如下：

```text
data/rrec_amazon/<CATEGORY>/phase1_examples.jsonl
outputs/rrec_amazon/<CATEGORY>/cot_candidates.jsonl
outputs/rrec_amazon/<CATEGORY>/cot_judged.jsonl
outputs/rrec_amazon/<CATEGORY>/cot_scored.jsonl
outputs/rrec_amazon/<CATEGORY>/filtered_high_quality_cot.jsonl
outputs/rrec_amazon/<CATEGORY>/rejected_cot.jsonl
outputs/rrec_amazon/<CATEGORY>/sft.jsonl
outputs/rrec_amazon/<CATEGORY>/grpo.jsonl
```

其中真正进入训练的是：

```text
sft.jsonl  -> Phase 2 SFT
grpo.jsonl -> Phase 3 GRPO
```

## 8. Phase 2：SFT

SFT shell 脚本：

```text
scripts/train/run_sft_qwen3_4b.sh
```

该脚本默认使用 Qwen3-4B。

Qwen3-4B smoke 中使用的 LoRA 参数与 method 对齐：

```bash
TRAIN_TYPE=lora
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05  # ms-swift 默认值
target_modules=all-linear
```

ms-swift 在 Qwen3-4B 上实际报告的 LoRA target modules：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

smoke checkpoint：

```text
/root/autodl-tmp/rec/aaai_pro/checkpoints/rrec_amazon_qwen3_4b_sft_phase0_tiny_clean_rank64/v0-20260615-165144/checkpoint-1
```

smoke 统计：

```text
steps: 1
loss: 0.35146287
token_acc: 0.94252874
trainable params: 132.1206M / 4154.5887M
```

## 9. Phase 3：GRPO

GRPO shell 脚本：

```text
scripts/train/run_grpo_qwen3_4b.sh
```

Reward plugin：

```text
scripts/train/rubric_gated_reward.py
```

已注册 reward：

```text
rubric_format
rubric_quality
rubric_gated_gain
```

当前 reward 逻辑：

1. `rubric_format`：检查 completion 是否包含非空 `<think>` 和 `<answer>`。
2. `rubric_quality`：调用可配置的 rubric scorer，得到 completion 质量分。
3. `rubric_gated_gain`：用同一个质量分门控推荐增益。

RL 阶段当前支持三种 rubric scorer：

```bash
RUBRIC_REWARD_SCORER=api        # 默认，高质量 LLM-as-Judge
RUBRIC_REWARD_SCORER=rules      # 方案 B，本地规则打分，适合离线 smoke
RUBRIC_REWARD_SCORER=classifier # 方案 A，轻量 rubric classifier 接口预留
```

默认 RL rubric scorer 已设置为 API：

```bash
RUBRIC_REWARD_SCORER=api
RUBRIC_REWARD_API_PROVIDER=openai_compatible
RUBRIC_REWARD_API_BASE_URL=http://127.0.0.1:18080/v1
RUBRIC_REWARD_API_MODEL=glm-5-1
RUBRIC_REWARD_API_FALLBACK=rules
```

也就是说，正常情况下 GRPO 在线 reward 会通过快手 TokenVerse `glm-5-1` 给 completion 打 rubric 质量分；如果 API 临时失败，默认 fallback 到本地 rules，避免训练直接崩掉。若要严格禁止 fallback，可以设置：

```bash
RUBRIC_REWARD_API_FALLBACK=none
```

在 qwen3 embedding gain 模式下：

```text
cot_sim = sim(user_history + full_completion, target_item)
baseline_sim = sim(user_history, target_item)
gain = cot_sim - baseline_sim
reward = quality * gain if quality >= threshold and gain > 0 else penalty
```

重要 caveat：`classifier` 模式目前只是接口预留，需要后续先用 Phase 1 的 `(CoT, Rubric Score)` 数据训练轻量 classifier，并设置 `RUBRIC_CLASSIFIER_CHECKPOINT` 后再启用。

## 10. Evaluation

已实现评测脚本：

```text
scripts/eval/evaluate_rrec_fullset_proxy.py
scripts/eval/evaluate_reasoner_fullset_proxy.py
scripts/eval/evaluate_proxy_retrieval.py
```

Reasoner evaluation 流程：

1. 构造 `user_history`。
2. 用 reasoner 生成结构化 CoT。
3. 构造增强 query：

```text
reasoner_query = append_recommendation_reasoning(user_history, generated_full_cot)
```

4. 比较以下两种 query 下 target item 的 rank：

```text
baseline_query = user_history
reasoner_query = user_history + full_cot
```

5. 指标：

```text
HR@5, HR@10, HR@20
NDCG@5, NDCG@10, NDCG@20
```

## 11. 当前服务器 Smoke 产物

服务器项目目录：

```text
/root/autodl-tmp/rec/aaai_pro
```

服务器 venv：

```text
/root/autodl-tmp/rec/ms-swift-312-cu124-venv
```

最近 smoke 产物：

```text
data/rrec_amazon/phase0_embedder_tiny.jsonl
checkpoints/phase0_qwen3_embedding_rrec_tiny/checkpoint-1
outputs/rrec_amazon/Musical_Instruments_qwen3_4btiny/cot_scored_phase0_tiny.jsonl
outputs/rrec_amazon/Musical_Instruments_qwen3_4btiny/grpo_phase0_tiny.jsonl
outputs/rrec_amazon/Musical_Instruments_qwen3_4btiny/sft_phase0_tiny_clean.jsonl
checkpoints/rrec_amazon_qwen3_4b_sft_phase0_tiny_clean_rank64/v0-20260615-165144/checkpoint-1
```

最近 smoke 文件行数：

```text
phase0_embedder_tiny.jsonl: 48
cot_scored_phase0_tiny.jsonl: 4
grpo_phase0_tiny.jsonl: 2
sft_phase0_tiny_clean.jsonl: 4
```

## 12. 当前决策与待核对 Gap

当前已确定决策：

1. 使用完整 `<think> + <answer>` 作为 CoT `r`。
2. 不做 `think-only` 或 `answer-only` 消融。
3. Phase 1 和 GRPO 默认都使用快手 TokenVerse `glm-5-1`，通过本地 tunnel 调用。
4. Qwen3-Embedding 用于 gain 和 GRPO baseline similarity。
5. LoRA rank 对齐 method：`rank=64`。

论文级实验前仍需处理的 gap：

1. Phase 0 smoke checkpoint 不是有效预训练 embedder，只验证代码路径。
2. 需要在快手 API tunnel 正常可用的情况下重新跑足够规模的 Phase 1 数据构造。
3. 当前 pipeline 默认 LLM 已统一为 Qwen3-4B；后续新增脚本也需要保持该默认。
4. GRPO 已支持 `api/rules/classifier` scorer 路由，但轻量 rubric classifier 训练与推理 checkpoint 尚未实现。
5. 当前输出都是 smoke artifacts，不是最终实验结果。
