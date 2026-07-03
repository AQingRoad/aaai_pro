# 模型训练结果台账

用途：集中记录每次模型训练的配置、数据口径、checkpoint、评测指标和结论。每次新增实验时复制“记录模板”，不要覆盖旧结果。

## 示例：CDs_and_Vinyl Qwen3-Embedding-0.6B history+CoT

### 基本信息
| 字段 | 内容 |
|---|---|
| 实验日期 | 2026-07-02 |
| 任务 | Amazon CDs_and_Vinyl next-item retrieval |
| 模型 | Qwen3-Embedding-0.6B |
| 训练方式 | contrastive embedding training |
| 训练数据 | `outputs/rrec_amazon/CDs_and_Vinyl/cot/training/phase0_embedder_cds_glm47_meta_compact_one_tagged_cot_only_full_target.jsonl` |
| 评测数据 | `outputs/rrec_amazon/CDs_and_Vinyl/cot/api/cot_candidate_lists_glm47_meta_compact_no_trunc_one_test_raw.jsonl` |
| 评测模式 | `history_plus_cot` |
| 候选物品数 | 12001 |
| 测试样本数 | 1341 |

### 模型和训练配置
| 参数 | 值 | 说明 |
|---|---:|---|
| `EMBEDDER_MAX_LENGTH` | `4096` | query 和 positive 编码的最大 token 长度 |
| `EMBEDDER_BATCH_SIZE` | `32` | 单次前向的样本数 |
| `EMBEDDER_GRAD_ACCUM` | `8` | 梯度累积步数 |
| effective batch | `256` | `batch_size * grad_accum` |
| `EMBEDDER_EPOCHS` | `5` | 训练轮数 |
| `EMBEDDER_LR` | `3e-6` | 学习率 |
| `EMBEDDER_ATTN_IMPLEMENTATION` | `flash_attention_2` | attention 实现 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | PyTorch CUDA allocator 配置 |

### 数据口径
| 字段 | 内容 |
|---|---|
| query 构造 | `user_history + tagged CoT` |
| CoT 格式 | `<think>...</think><answer>...</answer>` |
| history metadata | compact |
| target text 截断 | 不截断，`max_target_chars=0` |
| item text 截断 | 不截断，`max_item_chars=0` |
| 显式负样本 | `0`，只使用 batch 内负样本 |
| 泄漏检查 | 未发现 target 字段级泄漏；通用标题 overlap 需要在报告中单独说明 |

### 最佳 checkpoint 指标
| checkpoint | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-168` | 0.076808 | 0.128262 | 0.170022 | 0.039669 | 0.056301 | 0.066935 | 2044.2051 | 726 |

### 结论
- 当前最佳 checkpoint 为 `checkpoint-168`，主指标 `NDCG@20=0.066935`。
- 与 plain user history-only best `NDCG@20=0.030393` 相比，history metadata 和 CoT 提供了主要检索信号。
- 后续需要加入显式 hard negatives 或扩大负样本池，再比较 history-only、history+CoT 和 no-all-ratings 口径。

## 记录模板

### 基本信息
| 字段 | 内容 |
|---|---|
| 实验日期 |  |
| 任务 |  |
| 模型 |  |
| 训练方式 |  |
| 训练数据 |  |
| 评测数据 |  |
| 评测模式 |  |
| 候选物品数 |  |
| 测试样本数 |  |

### 模型和训练配置
| 参数 | 值 | 说明 |
|---|---:|---|
| `MODEL_PATH` |  | 模型或 checkpoint 路径 |
| `MAX_LENGTH` |  | tokenizer 最大长度 |
| `BATCH_SIZE` |  | 单卡 batch size |
| `GRAD_ACCUM` |  | 梯度累积步数 |
| effective batch |  | 实际有效 batch |
| `EPOCHS` |  | 训练轮数 |
| `LEARNING_RATE` |  | 学习率 |
| `ATTN_IMPLEMENTATION` |  | attention 实现 |
| `CUDA_ALLOC_CONF` |  | CUDA allocator 配置 |

### 数据口径
| 字段 | 内容 |
|---|---|
| query 构造 |  |
| prompt/CoT 格式 |  |
| metadata 字段 |  |
| target text 截断 |  |
| history 截断 |  |
| 负样本策略 |  |
| seen item mask |  |
| 泄漏检查 |  |

### checkpoint 指标
| checkpoint | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |

### 结论
- 最佳 checkpoint：
- 主要提升或退化：
- 与关键 baseline 的差异：
- 下一步动作：
