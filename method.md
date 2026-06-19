# What Makes Reasoning Useful for Recommendation?
# A Systematic Study of CoT Quality in Reasoning-Enhanced Recommendation

---

## 1. Problem: The Black Box of Reasoning in Recommendation

近期工作（R2ec, Rec-R1, LangPTune）将 LLM reasoning 引入推荐系统并取得了显著提升。然而，一个核心问题始终未被回答：

> **什么样的 CoT 对推荐真正有用？**

现有方法把 reasoning 当作黑箱——R2ec 用端到端 RL 联合优化推理和推荐，Rec-R1 用 NDCG reward 间接激励推理，但没有工作系统性地分析：

1. **CoT 的哪些特征**与推荐增益正相关？
2. **什么样的推理模式**是有害的？
3. 如何**显式地构造和筛选**对推荐有用的 CoT？

我们的工作正是回答这些问题。我们提出一个完整的研究框架：

- **分析工具**：定义推荐场景下 CoT 质量的多维 Rubric
- **数据构造**：Reject Sampling + LLM-as-Judge pipeline，构造高质量推荐 CoT
- **训练方法**：Rubric-Gated RL，将 CoT 质量评价与推荐增益信号融合
- **实证发现**：系统性分析 CoT 质量各维度与推荐效果的关系

---

## 2. Recommendation-Specific CoT Quality Rubric

通用的 CoT 评价（逻辑性、连贯性、正确性）不足以衡量推荐场景的 CoT 质量。我们提出面向推荐的 **5 维 Rubric**：

### 2.1 五个维度

| 维度 | 定义 | 为什么对推荐重要 | 评分 (1-5) |
|------|------|-----------------|-----------|
| **Preference Grounding** | CoT 是否基于用户的实际行为历史进行推理，而非泛泛而谈 | 不grounded的推理（"用户可能喜欢热门商品"）无法提供个性化信号 | 1=完全无依据, 5=每个推断都有行为证据 |
| **Taste Specificity** | CoT 是否挖掘出具体、细粒度的用户偏好维度 | "用户喜欢音乐" 对 embedding 没有区分度；"用户偏好 90 年代 grunge rock 的原始录音" 才有 | 1=泛泛而谈, 5=高度具体 |
| **Transitional Reasoning** | CoT 是否建立了从已知偏好到未知兴趣的推理链 | 推荐的核心是泛化——从已购推断未购；纯总结历史无法帮助发现新物品 | 1=纯历史总结, 5=有明确的兴趣迁移推理 |
| **Discriminative Framing** | CoT 是否帮助区分 target item 与相似但不匹配的 items | embedding 检索靠的是区分度；CoT 如果不能拉开正负样本距离就没用 | 1=无区分信号, 5=明确指出应选/应避免的特征 |
| **Conciseness** | CoT 是否简洁无冗余 | embedding 模型有上下文限制；冗余信息稀释有效信号 | 1=大量废话, 5=每句都有信息量 |

### 2.2 为什么是这 5 个维度？

这些维度并非凭空设计，而是从推荐系统的基本原理推导出来的：

- **Preference Grounding** 和 **Taste Specificity** → 保证 CoT 包含**有效的用户建模信号**（信号的质量和粒度）
- **Transitional Reasoning** → 保证 CoT 超越简单复述，提供**泛化能力**（这是推理的核心价值）
- **Discriminative Framing** → 保证 CoT 的信号对 **embedding 检索有用**（alignment with downstream task）
- **Conciseness** → 保证信号**不被稀释**（信噪比）

### 2.3 Rubric 的作用

这套 Rubric 在我们的方法中承担三重角色：

1. **分析工具**：用于系统性分析不同 CoT 的质量特征与推荐效果的关系
2. **数据筛选标准**：LLM-as-Judge 按此 Rubric 打分，筛选高质量 CoT 做 SFT
3. **RL Reward 成分**：在 RL 阶段作为 CoT 质量 reward 的基础

---

## 3. Method

### 3.1 架构：Reasoner + Frozen Embedder

```
              ┌────────────────────┐
  用户行为     │   Reasoner (LLM)    │
  历史 u  ───►│  e.g., Qwen3-4B    │──► reasoning text r
              │  (可训练)           │
              └────────────────────┘
                        │
                        │ text concat: [u; r]
                        ▼
              ┌────────────────────┐
              │  Embedder (frozen) │
              │  e.g., Qwen3-Emb  │──► user embedding h_u
              └────────────────────┘
                        │
                        │ inner product
                        ▼
              ┌────────────────────┐
              │ Item Embedding Table│
              │  h_v for all v ∈ V │──► Top-K recommendation
              └────────────────────┘
```

解耦架构的选择不是为了对标 R2ec，而是服务于我们的研究目标：

- **Frozen Embedder 提供稳定的评测基准**：CoT 的推荐增益可以被精确测量
- **Reasoner 和 Embedder 职责正交**：一个负责理解和推理，一个负责编码和检索
- **支持 CoT 质量的因果分析**：可以做 ablation（有 CoT vs 无 CoT vs 不同质量 CoT）

### 3.2 训练流程

#### Phase 0: Embedder 预训练

在推荐数据上训练 embedding 模型（标准 InfoNCE），训练完成后冻结。

```
L_embed = -log exp(h_u · h_v+ / τ) / Σ_v' exp(h_u · h_v' / τ)
```

#### Phase 1: CoT 数据构造 — Reject Sampling + LLM-as-Judge

这是我们方法的核心环节之一。目标是构造一批"对推荐有用的高质量 CoT"用于 SFT 冷启动。

```
Pipeline:
                                                   ┌─────────┐
For each training user u:                          │ Rubric  │
                                                   │ Score   │
  Step 1: 采样 N 条 CoT                             │ (Judge) │
  ┌──────────────┐                                  └────┬────┘
  │ Strong LLM    │──► {r_1, r_2, ..., r_N}              │
  │ (DeepSeek-R1  │    (不同 temperature/prompt)          │
  │  / Qwen3-235B)│                                      │
  └──────────────┘                                       │
                                                         ▼
  Step 2: 双重评价                              ┌────────────────┐
  ┌──────────────────┐                         │ 综合排序        │
  │ CoT Gain 计算     │◄── frozen embedder     │ = Rubric Score  │
  │ = sim(u+r, v+)   │                        │ × CoT Gain     │
  │ - sim(u, v+)     │                        │ (乘法门控)      │
  └──────────────────┘                         └───────┬────────┘
                                                       │
  Step 3: 筛选                                         ▼
  ┌──────────────────────────────────────────────────────┐
  │ 选择 top-k CoT → SFT 训练数据                         │
  │ 同时保留 rejected CoT → 用于对比分析和 DPO (可选)       │
  └──────────────────────────────────────────────────────┘
```

**关键设计**：我们用**乘法门控**而非简单加权来组合 Rubric Score 和 CoT Gain：

```
综合评分 = Rubric_Score × max(CoT_Gain, 0)
```

- Rubric 质量差的 CoT（即使碰巧有正增益）被抑制
- CoT Gain 为负的 CoT（即使 Rubric 评分高）直接被过滤
- 只有**既合理又有效**的 CoT 才被保留

**LLM-as-Judge 评分 Prompt 设计**：

```
你是推荐系统推理质量评估专家。请根据以下 5 个维度评估这段推理的质量。
每个维度打 1-5 分，并给出简要理由。

用户行为历史：{user_history}
推理文本：{cot}
推荐目标物品：{target_item}（仅用于评估，不告知 judge 以避免信息泄漏）

评估维度：
1. Preference Grounding (1-5): 推理是否基于用户实际行为？
2. Taste Specificity (1-5): 是否挖掘出具体的偏好维度？
3. Transitional Reasoning (1-5): 是否有从已知到未知的推理链？
4. Discriminative Framing (1-5): 是否有助于区分相似物品？
5. Conciseness (1-5): 是否简洁无冗余？
```

> **注意**：评估时是否将 target_item 暴露给 Judge 是一个需要实验验证的设计选择。暴露 target 可以更好地评估 Discriminative Framing，但可能引入信息泄漏偏差。我们将在消融实验中对比两种设定。

#### Phase 2: SFT 冷启动

用 Phase 1 筛选出的高质量 CoT 对 Reasoner 进行 SFT 训练，建立推荐推理的基本能力。

```bash
swift sft \
    --model Qwen/Qwen3-4B \
    --dataset filtered_high_quality_cot.jsonl \
    --train_type lora \
    --lora_rank 64
```

#### Phase 3: Rubric-Gated RL (GRPO)

SFT 后的 Reasoner 已具备生成推荐推理的基本能力，但可能过拟合 teacher 的风格。RL 阶段通过 Rubric-Gated Reward 进一步优化。

**Reward 设计**：

```
For each user u, ground-truth item v+, generated CoT r:

  # 推荐增益（连续值，可正可负）
  R_gain(r) = cos(embed(u + r), embed(v+)) - cos(embed(u), embed(v+))

  # Rubric 质量评分（可选：LLM Judge 或轻量 classifier）
  R_rubric(r) = Rubric_Score(r) / 25.0   # 归一化到 [0, 1]

  # Rubric-Gated Reward（门控机制）
  R(r) = R_rubric(r) × R_gain(r)     if R_rubric(r) > threshold_rubric
        = penalty                      otherwise
```

**门控机制的直觉**：

- 如果 CoT 的 Rubric 质量不达标（低于门槛），直接给负 reward → 避免 RL 找到"质量差但碰巧有效"的 shortcut
- 如果 Rubric 质量达标，reward 的大小由推荐增益决定 → 在质量合格的 CoT 中选择推荐效果最好的
- **乘法而非加法**的好处：质量和增益是**相乘关系**——我们要的是"既好又有用"的 CoT，而非"一个维度好就行"

**RL Reward 中 Rubric 评分的效率问题**：

每条 CoT 都调用 LLM Judge 打分在 RL 训练中代价过高。我们有两个解决方案：

- **方案 A：轻量 Rubric Classifier**。在 Phase 1 积累的 (CoT, Rubric Score) 数据上训练一个轻量分类器/回归器（如用 Embedder 本身 + 线性层），推理时替代 LLM Judge
- **方案 B：简化 Rubric 为规则**。将 5 个维度的核心特征转化为可自动检测的规则（如 Conciseness → 长度约束，Preference Grounding → 是否包含用户历史中的 item 关键词），避免 LLM 调用

实际训练中，方案 A 和 B 可以结合：用规则做粗筛 + 轻量模型做精评。

**GRPO 更新**：

```
A_i = (R_i - mean(R)) / std(R)   # Group Relative Advantage
L = -Σ_i Σ_t min(ratio · A_i, clip(ratio) · A_i) - β · KL(π_θ || π_ref)
```

### 3.3 与 R2ec 的对比

| 维度 | R2ec | Ours |
|------|------|------|
| **核心问题** | 如何做 reasoning-enhanced recommendation | 什么样的 reasoning 对 recommendation 有用 |
| **架构** | 统一 dual-head (lm_head + rec_head) | 解耦 Reasoner + frozen Embedder |
| **CoT 质量控制** | 无显式控制（端到端 RL 隐式优化） | 显式 Rubric 定义 + Judge 筛选 + Gated RL |
| **数据构造** | 用模型自身 rollout | Reject Sampling + LLM-as-Judge |
| **Reward** | β·softmax_sim + (1-β)·NDCG@K | Rubric_Score × CoT_Gain（门控乘法） |
| **可解释性** | 推理文本可读但不知有没有用 | 每条推理有明确的质量评分和增益量化 |
| **研究价值** | 提出方法 | 提出分析框架 + 方法 |

---

## 4. Experiments

### 4.1 数据集

与 R2ec 保持一致，使用 Amazon 公开数据集：

| 数据集 | 领域 |
|--------|------|
| Amazon CDs and Vinyl | 音乐 |
| Amazon Video Games | 游戏 |
| Amazon Musical Instruments | 乐器 |

预处理：5-core filtering, leave-one-out split, full-set evaluation。

### 4.2 Baselines

| 类别 | 方法 |
|------|------|
| 传统推荐 | SASRec, GRU4Rec, Caser |
| LLM-based 推荐 | LLaRA, BigRec, D3, SDPO, SPRec |
| 推理增强推荐 | **R2ec** (NeurIPS'25), LangPTune, Rec-R1 |
| 消融 | Ours (no reasoning), Ours (SFT only, no RL), Ours (RL w/o Rubric gating) |

### 4.3 指标

- Hit Rate @ K (H@5, H@10, H@20)
- NDCG @ K (N@5, N@10, N@20)
- Full-set evaluation

### 4.4 核心分析实验

#### RQ1: CoT 的哪些质量维度与推荐增益最相关？

- 对大量 CoT 同时计算 5 维 Rubric Score 和 CoT Gain
- 分析每个 Rubric 维度与 CoT Gain 的 **Pearson/Spearman 相关系数**
- 回归分析：哪些维度是推荐增益的最强预测因子？
- **预期发现**：Transitional Reasoning 和 Discriminative Framing 与推荐增益的相关性最高，而 Conciseness 可能呈非线性关系

#### RQ2: Reject Sampling + Judge 构造的数据是否优于直接 SFT？

| 设定 | 训练数据来源 |
|------|-------------|
| SFT-Raw | 直接用 strong LLM 生成的 CoT（不筛选） |
| SFT-Gain | 只按 CoT Gain 筛选 |
| SFT-Rubric | 只按 Rubric Score 筛选 |
| SFT-Gated | 按 Rubric × Gain 乘法门控筛选（我们的方法） |

对比不同数据筛选策略对下游推荐效果的影响。

#### RQ3: Rubric-Gated RL 是否优于单一信号 RL？

| 设定 | Reward |
|------|--------|
| RL-Gain | 纯 CoT Gain |
| RL-Rubric | 纯 Rubric Score |
| RL-Sum | α·Rubric + (1-α)·Gain（加权求和） |
| RL-Gated | Rubric × Gain（门控，我们的方法） |

对比不同 reward 组合方式的效果。

#### RQ4: CoT 质量分布如何随训练阶段变化？

- 分别在 SFT 后和 RL 后，对同一组用户生成 CoT
- 绘制 5 维 Rubric Score 的雷达图对比
- 绘制 CoT Gain 分布直方图对比
- **预期发现**：SFT 后 CoT 质量整体合格但 Gain 方差大；RL 后 Gain 分布右移且方差缩小

#### RQ5: 不同类型的用户/物品需要什么样的推理？

- 按用户行为丰富度分组（低/中/高活跃度）
- 按物品受欢迎程度分组（长尾/中腰/头部）
- 分析不同组中高增益 CoT 的 Rubric 维度分布差异
- **预期发现**：低活跃用户更依赖 Transitional Reasoning（需要推断偏好），高活跃用户更依赖 Taste Specificity（行为已丰富，需要精准画像）

#### RQ6: Selective Reasoning 策略

- 不是所有用户都需要推理
- 用 Rubric Classifier 预测 CoT Gain，只对预期增益 > δ 的用户启用推理
- Pareto 曲线：推理调用比例 vs 推荐效果

---

## 5. Implementation

### 5.1 模型

| 组件 | 模型 | 参数量 |
|------|------|--------|
| Reasoner | Qwen3-4B | 4B |
| Embedder | Qwen3-Embedding-0.6B | 0.6B |
| Judge (Phase 1) | DeepSeek-R1 / Qwen3-235B-Instruct (API) | - |
| Rubric Classifier (Phase 3) | Qwen3-Embedding-0.6B + Linear | 0.6B |

### 5.2 训练配置

```bash
# Phase 0: Embedder 训练 (已完成)
# checkpoint: output/qwen3_emb_recsys_full_v3_0601_sample_0_6B/v1-20260608-211028/checkpoint-534

# Phase 1: CoT 数据构造
python scripts/generate_cot_candidates.py \
    --model deepseek-r1 \
    --data amazon_train.jsonl \
    --num_candidates 16 \
    --temperatures 0.6,0.8,1.0

python scripts/judge_cot_quality.py \
    --model qwen3-235b \
    --rubric recommendation_rubric.yaml \
    --candidates cot_candidates.jsonl \
    --embedder_checkpoint <embedder_path> \
    --output filtered_cot.jsonl

# Phase 2: SFT
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen3-4B \
    --dataset filtered_cot.jsonl \
    --train_type lora --lora_rank 64 \
    --deepspeed zero2

# Phase 3: Rubric-Gated RL
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model <sft_checkpoint> \
    --dataset grpo_train.jsonl \
    --num_generations 8 \
    --external_plugins rubric_gated_reward.py
```

### 5.3 Reward 实现

```python
class RubricGatedORM(ORM):
    """Rubric-Gated Reward: CoT 质量门控 × 推荐增益"""

    def __init__(self, embedder, rubric_classifier, threshold=0.5, penalty=-0.5):
        self.embedder = embedder          # frozen
        self.rubric_clf = rubric_classifier  # 轻量 rubric 评分器
        self.threshold = threshold
        self.penalty = penalty

    def __call__(self, completions, metas):
        rewards = []
        for completion, meta in zip(completions, metas):
            # Rubric 质量评分
            rubric_score = self.rubric_clf.score(
                user_history=meta['user_context'],
                cot=completion
            )  # 归一化到 [0, 1]

            if rubric_score < self.threshold:
                rewards.append(self.penalty)
                continue

            # CoT Gain
            h_with_cot = self.embedder.encode(meta['user_context'] + '\n' + completion)
            h_item = meta['precomputed_item_emb']
            baseline_sim = meta['precomputed_baseline_sim']

            sim_with_cot = cosine_similarity(h_with_cot, h_item)
            gain = sim_with_cot - baseline_sim

            # Rubric-Gated Reward
            reward = rubric_score * gain
            rewards.append(reward)

        return rewards
```

---

## 6. Expected Contributions

1. **推荐 CoT 质量 Rubric**：首次提出面向推荐场景的 CoT 质量多维评估体系（5 维 Rubric），为该领域的后续研究提供分析工具

2. **系统性实证分析**：首次定量分析 CoT 各质量维度与推荐增益的关系，回答"什么样的推理对推荐有用"这个核心问题

3. **数据构造方法论**：提出 Reject Sampling + LLM-as-Judge 的推荐 CoT 数据构造 pipeline，解决"高质量推荐推理数据从哪来"的问题

4. **Rubric-Gated RL**：提出将 CoT 质量评价与推荐增益通过门控机制融合的 reward 设计，避免 RL 优化中的 quality-effectiveness 失衡

5. **实践指导**：给出不同用户/物品类型所需推理特征的分析，指导推荐系统中推理模块的差异化部署

---

## 7. Related Work

### 推理增强推荐系统

| 工作 | 方法 | 与我们的关系 |
|------|------|-------------|
| **R2ec** (NeurIPS'25) | 统一 dual-head + RecPO | 核心 baseline；端到端方法但不分析 CoT 质量 |
| **Rec-R1** | RL 推理推荐 | RL for reasoning，但无质量分析 |
| **LangPTune** | LLM 偏好分析增强推荐 | 解耦架构 + SFT，无 RL 优化 |
| **LREM** | 同一 LLM 推理+embedding | SFT + RL，未探索 CoT 质量 |

### LLM-as-Judge

| 工作 | 方法 |
|------|------|
| **JudgeLM** | 训练 LLM 做评估 |
| **MT-Bench** | 多轮对话评估 with LLM judge |
| **Self-Taught Evaluators** | 迭代自训练评估器 |

### RL for Reasoning

| 工作 | 方法 |
|------|------|
| **DeepSeek-R1** | GRPO 训练推理 |
| **GRPO** | Group Relative Policy Optimization |
| **Process Reward Models** | 过程监督（Math-Shepherd, OmegaPRM） |

---

## 8. Risks & Mitigations

| 风险 | 影响 | 缓解 |
|------|------|------|
| LLM Judge 评分与实际推荐增益不相关 | 高 | 先做相关性验证；若不相关则调整 rubric 或放弃 rubric reward |
| Rubric 维度设计不完备/冗余 | 中 | 通过相关性分析迭代；用因子分析检验维度独立性 |
| Phase 1 数据构造成本过高 (N×Judge 调用) | 中 | 先小规模验证 pipeline 有效性；可用 API 降低成本 |
| RL 训练中 Rubric Classifier 不准 | 中 | 对比 Classifier vs LLM Judge 的 RL 效果；Classifier 不行就用规则 |
| Amazon 数据集规模有限 | 中 | LoRA + KL 惩罚 + early stopping |
| R2ec 统一模型全面碾压 | 低 | 核心贡献是分析框架而非性能，即使效果略低也有研究价值 |
