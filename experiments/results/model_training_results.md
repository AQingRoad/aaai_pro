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
| checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-168` | 0.076808 | 0.039669 | 0.128262 | 0.056301 | 0.170022 | 0.066935 | 2044.2051 | 726 |

### 结论
- 当前最佳 checkpoint 为 `checkpoint-168`，主指标 `NDCG@20=0.066935`。
- 与 plain user history-only best `NDCG@20=0.030393` 相比，history metadata 和 CoT 提供了主要检索信号。
- 后续需要加入显式 hard negatives 或扩大负样本池，再比较 history-only、history+CoT 和 no-all-ratings 口径。

## CDs_and_Vinyl Qwen3-Embedding-0.6B plain history+rating

### 基本信息
| 字段 | 内容 |
|---|---|
| 实验日期 | 2026-07-03 |
| 任务 | Amazon CDs_and_Vinyl next-item retrieval |
| 模型 | Qwen3-Embedding-0.6B |
| 训练方式 | contrastive embedding training |
| 训练数据 | `outputs/rrec_amazon/CDs_and_Vinyl/embedding/phase0_embedder_cds_plain_user_history_rating_only.jsonl` |
| 评测数据 | `data/rrec_amazon/CDs_and_Vinyl/examples_plain_user_history_one_test.jsonl` |
| 评测模式 | `plain_user_history` |
| 候选物品数 | 12001 |
| 测试样本数 | 1341 |
| checkpoint root | `/home/user/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_0p6b_cds_plain_history_rating_only_len4096_batch128_epoch5` |
| eval dir | `/home/user/aaai_pro/outputs/rrec_amazon/eval/CDs_and_Vinyl/qwen3_embedding_0p6b_cds_plain_history_rating_only_len4096_batch128_epoch5_test_plain_user_history` |

### 模型和训练配置
| 参数 | 值 | 说明 |
|---|---:|---|
| `MODEL_PATH` | `/home/user/models_hf/Qwen3-Embedding-0.6B` | base embedding 模型 |
| `MAX_LENGTH` | `4096` | query 和 positive 编码最大 token 长度 |
| `BATCH_SIZE` | `128` | 单卡 batch size |
| `GRAD_ACCUM` | `1` | 梯度累积步数 |
| effective batch | `128` | `batch_size * grad_accum` |
| `EPOCHS` | `5` | 训练轮数 |
| `LEARNING_RATE` | `3e-6` | 学习率 |
| `ATTN_IMPLEMENTATION` | `flash_attention_2` | attention 实现 |
| `CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA allocator 配置 |

### 数据口径
| 字段 | 内容 |
|---|---|
| query 构造 | history item title + rating |
| prompt/CoT 格式 | 无 CoT |
| metadata 字段 | 无 item metadata |
| rating/catalog stats | 包含 history rating，不包含 catalog stats |
| target text 截断 | 不截断 |
| history 截断 | plain title 和 rating，token 长度远低于 4096 |
| 负样本策略 | 无显式负样本，只使用 batch 内负样本 |
| seen item mask | 评测脚本按 RRec examples 中 history 过滤已见物品 |
| 泄漏检查 | query 不拼 target、不拼 CoT；该组评测发生在删除 pad item 前，候选数为 12001 |

### checkpoint 指标
| checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | 0.029083 | 0.016136 | 0.057420 | 0.025109 | 0.078300 | 0.030431 | 3659.7315 | 2744 |
| `checkpoint-166` | 0.028337 | 0.015902 | 0.061148 | 0.026527 | 0.082774 | 0.032019 | 3501.1462 | 2566 |
| `checkpoint-249` | 0.028337 | 0.015967 | 0.059657 | 0.026155 | 0.084265 | 0.032422 | 3458.4489 | 2469 |
| `checkpoint-332` | 0.028337 | 0.015888 | 0.061894 | 0.026754 | 0.083520 | 0.032275 | 3449.8658 | 2467 |
| `checkpoint-415` | 0.029083 | 0.016353 | 0.061148 | 0.026788 | 0.083520 | 0.032503 | 3451.0932 | 2441 |

### 结论
- 最佳 checkpoint 为 `checkpoint-415`，主指标 `NDCG@20=0.032503`，`HR@20=0.083520`。
- 该结果明显低于 metadata history-only 结果，说明只保留标题和 rating 会丢失主要检索信号。
- 该组评测候选数为 `12001`，metadata no-all-ratings 最新评测候选数为 `12000`，跨组对比需要标明 item_info 口径。

## CDs_and_Vinyl Qwen3-Embedding-0.6B no-all-ratings metadata history-only

### 基本信息
| 字段 | 内容 |
|---|---|
| 实验日期 | 2026-07-03 |
| 任务 | Amazon CDs_and_Vinyl next-item retrieval |
| 模型 | Qwen3-Embedding-0.6B |
| 训练方式 | contrastive embedding training |
| 训练数据 | `outputs/rrec_amazon/CDs_and_Vinyl/embedding/phase0_embedder_cds_meta_compact_no_all_ratings_history_only.jsonl` |
| 评测数据 | `data/rrec_amazon/CDs_and_Vinyl/examples_meta_compact_no_all_ratings_test.jsonl` |
| 评测模式 | `user_history` |
| 候选物品数 | 12000 |
| 测试样本数 | 1341 |
| checkpoint root | `/home/user/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_0p6b_cds_meta_compact_no_all_ratings_history_only_len4096_batch128_accum1_epoch5` |
| eval dir | `/home/user/aaai_pro/outputs/rrec_amazon/eval/CDs_and_Vinyl/qwen3_embedding_0p6b_cds_meta_compact_no_all_ratings_history_only_len4096_batch128_accum1_epoch5_test_user_history` |

### 模型和训练配置
| 参数 | 值 | 说明 |
|---|---:|---|
| `MODEL_PATH` | `/home/user/models_hf/Qwen3-Embedding-0.6B` | base embedding 模型 |
| `MAX_LENGTH` | `4096` | query 和 positive 编码最大 token 长度 |
| `BATCH_SIZE` | `128` | 单卡 batch size |
| `GRAD_ACCUM` | `1` | 梯度累积步数 |
| effective batch | `128` | `batch_size * grad_accum` |
| `EPOCHS` | `5` | 训练轮数 |
| `LEARNING_RATE` | `3e-6` | 学习率 |
| `ATTN_IMPLEMENTATION` | `flash_attention_2` | attention 实现 |
| `CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA allocator 配置 |

### 数据口径
| 字段 | 内容 |
|---|---|
| query 构造 | compact metadata user history |
| prompt/CoT 格式 | 无 CoT |
| metadata 字段 | store、artist/format、categories、features、description、details |
| rating/catalog stats | 不包含 rating，不包含 catalog stats |
| target text 截断 | 不截断 |
| history 截断 | item metadata 字段按构造脚本口径处理；训练 token max length 为 4096 |
| 负样本策略 | 无显式负样本，只使用 batch 内负样本 |
| seen item mask | 评测脚本按 RRec examples 中 history 过滤已见物品 |
| 泄漏检查 | query 不拼 target、不拼 CoT；A100 评测候选 item_info 已删除 pad item |

### checkpoint 指标
| checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | 0.071588 | 0.036370 | 0.121551 | 0.052729 | 0.170022 | 0.065174 | 2203.3117 | 974 |
| `checkpoint-166` | 0.076808 | 0.039079 | 0.123043 | 0.054156 | 0.178225 | 0.068194 | 2083.2401 | 808 |
| `checkpoint-249` | 0.078300 | 0.039914 | 0.126025 | 0.055456 | 0.181954 | 0.069664 | 2043.5623 | 795 |
| `checkpoint-332` | 0.079045 | 0.039885 | 0.124534 | 0.054706 | 0.181954 | 0.069321 | 2038.7122 | 797 |
| `checkpoint-415` | 0.079791 | 0.040499 | 0.125280 | 0.055276 | 0.179717 | 0.069153 | 2036.8949 | 786 |

### 结论
- 最佳 checkpoint 为 `checkpoint-249`，主指标 `NDCG@20=0.069664`，`HR@20=0.181954`。
- `checkpoint-249` 之后 mean rank 继续下降，但 `NDCG@20` 没有继续上升，后续选 checkpoint 按 `NDCG@20` 使用 `checkpoint-249`。
- 与旧 no-all-ratings metadata history-only best `NDCG@20=0.064838` 相比，本次 A100 batch 128 训练结果提高 `0.004826`。该比较受候选集合从 12001 改为 12000 影响，报告中需要标明评测口径。

## CDs_and_Vinyl Qwen3-Embedding-0.6B no-all-ratings metadata history+CoT

### 基本信息
| 字段 | 内容 |
|---|---|
| 实验日期 | 2026-07-04 |
| 任务 | Amazon CDs_and_Vinyl next-item retrieval |
| 模型 | Qwen3-Embedding-0.6B |
| 训练方式 | contrastive embedding training |
| 训练数据 | `outputs/rrec_amazon/CDs_and_Vinyl/cot/training/phase0_embedder_cds_meta_compact_no_all_ratings_history_plus_tagged_cot_manual_filled_full_target.jsonl` |
| 评测数据 | `outputs/rrec_amazon/CDs_and_Vinyl/cot/api/cds_glm47_meta_compact_no_trunc_one_test_full_target.jsonl` |
| 评测模式 | `history_plus_cot` |
| 候选物品数 | 12000 |
| 测试样本数 | 1341 |
| checkpoint root | `/home/user/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_0p6b_cds_meta_compact_no_all_ratings_history_plus_cot_len4096_batch128_accum1_epoch5` |
| eval dir | `/home/user/aaai_pro/outputs/rrec_amazon/eval/CDs_and_Vinyl/qwen3_embedding_0p6b_cds_meta_compact_no_all_ratings_history_plus_cot_len4096_batch128_accum1_epoch5_test_history_plus_cot` |

### 模型和训练配置
| 参数 | 值 | 说明 |
|---|---:|---|
| `MODEL_PATH` | `/home/user/models_hf/Qwen3-Embedding-0.6B` | base embedding 模型 |
| `MAX_LENGTH` | `4096` | query 和 positive 编码最大 token 长度 |
| `BATCH_SIZE` | `128` | 单卡 batch size |
| `GRAD_ACCUM` | `1` | 梯度累积步数 |
| effective batch | `128` | `batch_size * grad_accum` |
| `EPOCHS` | `5` | 训练轮数 |
| `LEARNING_RATE` | `3e-6` | 学习率 |
| `ATTN_IMPLEMENTATION` | `flash_attention_2` | attention 实现 |
| `CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA allocator 配置 |

### 数据口径
| 字段 | 内容 |
|---|---|
| query 构造 | compact metadata user history + tagged CoT |
| prompt/CoT 格式 | `<think>...</think><answer>...</answer>` |
| metadata 字段 | store、artist/format、categories、features、description、details |
| rating/catalog stats | 不包含 rating，不包含 catalog stats |
| target text 截断 | 不截断 |
| CoT 来源 | `10687` 条 API CoT + `35` 条 manual_fill CoT |
| manual_fill 约束 | 只依据 history 编写，不使用 target 字段 |
| history 截断 | item metadata 字段按构造脚本口径处理；训练 token max length 为 4096 |
| 负样本策略 | 无显式负样本，只使用 batch 内负样本 |
| seen item mask | 评测脚本按 RRec examples 中 history 过滤已见物品 |
| 泄漏检查 | query 不拼 target；manual CoT target title 命中 `0`；positive 截断 `0` |

### checkpoint 指标
| checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | 0.074571 | 0.038684 | 0.124534 | 0.054949 | 0.163311 | 0.064823 | 2141.773 | 872 |
| `checkpoint-166` | 0.079045 | 0.040727 | 0.134228 | 0.058526 | 0.175988 | 0.068991 | 2020.187 | 727 |
| `checkpoint-249` | 0.081283 | 0.041628 | 0.137957 | 0.059861 | 0.173751 | 0.068883 | 1982.127 | 684 |
| `checkpoint-332` | 0.079791 | 0.041519 | 0.137957 | 0.060309 | 0.175988 | 0.069862 | 1973.577 | 664 |
| `checkpoint-415` | 0.079791 | 0.041388 | 0.138702 | 0.060387 | 0.173751 | 0.069206 | 1975.014 | 672 |

### 结论
- 按 `N@20` 选择 `checkpoint-332`，主指标 `NDCG@20=0.069862`，`HR@20=0.175988`。
- 与 no-all-ratings metadata history-only best `checkpoint-249` 相比，`N@20` 增加 `0.000198`，相对提升 `0.284%`；`H@20` 下降 `0.005966`，相对变化 `-3.279%`。
- CoT 版本在 `N@10` 上高于 history-only best，`0.060309` 对 `0.055456`，差值 `0.004853`，相对提升 `8.751%`；top-20 命中数没有同步增加。
- 当前 CDs_and_Vinyl 结果显示，metadata history 已覆盖主要检索信号，直接拼接 tagged CoT 对 `N@20` 的增量很小。

## 横向对比：相同 checkpoint 下不同训练版本

### 对比口径
| 字段 | plain history+rating | metadata no-all-ratings history-only |
|---|---|---|
| base 模型 | Qwen3-Embedding-0.6B | Qwen3-Embedding-0.6B |
| 训练数据 | `phase0_embedder_cds_plain_user_history_rating_only.jsonl` | `phase0_embedder_cds_meta_compact_no_all_ratings_history_only.jsonl` |
| query 内容 | item title + rating | compact item metadata，不含 rating 和 catalog stats |
| batch / grad | `128 / 1` | `128 / 1` |
| epoch | `5` | `5` |
| max length | `4096` | `4096` |
| 评测样本数 | `1341` | `1341` |
| 候选 item 数 | `12001` | `12000` |

候选数不同来自 `item_info` 口径变化：plain 结果在删除 `pad_title` 前评测，metadata 结果在删除后评测。下面的对比主要用于判断训练版本趋势，正式报告需要标明这个候选集合差异。

### 主指标横向对比
| checkpoint | plain H@20 | plain N@20 | metadata H@20 | metadata N@20 | ΔH@20 | ΔN@20 |
|---|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | 0.078300 | 0.030431 | 0.170022 | 0.065174 | 0.091722 | 0.034743 |
| `checkpoint-166` | 0.082774 | 0.032019 | 0.178225 | 0.068194 | 0.095451 | 0.036175 |
| `checkpoint-249` | 0.084265 | 0.032422 | 0.181954 | 0.069664 | 0.097689 | 0.037242 |
| `checkpoint-332` | 0.083520 | 0.032275 | 0.181954 | 0.069321 | 0.098434 | 0.037046 |
| `checkpoint-415` | 0.083520 | 0.032503 | 0.179717 | 0.069153 | 0.096197 | 0.036650 |

### 全指标横向对比
| checkpoint | 训练版本 | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | plain history+rating | 0.029083 | 0.016136 | 0.057420 | 0.025109 | 0.078300 | 0.030431 |
| `checkpoint-83` | metadata no-all-ratings history-only | 0.071588 | 0.036370 | 0.121551 | 0.052729 | 0.170022 | 0.065174 |
| `checkpoint-166` | plain history+rating | 0.028337 | 0.015902 | 0.061148 | 0.026527 | 0.082774 | 0.032019 |
| `checkpoint-166` | metadata no-all-ratings history-only | 0.076808 | 0.039079 | 0.123043 | 0.054156 | 0.178225 | 0.068194 |
| `checkpoint-249` | plain history+rating | 0.028337 | 0.015967 | 0.059657 | 0.026155 | 0.084265 | 0.032422 |
| `checkpoint-249` | metadata no-all-ratings history-only | 0.078300 | 0.039914 | 0.126025 | 0.055456 | 0.181954 | 0.069664 |
| `checkpoint-332` | plain history+rating | 0.028337 | 0.015888 | 0.061894 | 0.026754 | 0.083520 | 0.032275 |
| `checkpoint-332` | metadata no-all-ratings history-only | 0.079045 | 0.039885 | 0.124534 | 0.054706 | 0.181954 | 0.069321 |
| `checkpoint-415` | plain history+rating | 0.029083 | 0.016353 | 0.061148 | 0.026788 | 0.083520 | 0.032503 |
| `checkpoint-415` | metadata no-all-ratings history-only | 0.079791 | 0.040499 | 0.125280 | 0.055276 | 0.179717 | 0.069153 |

### 横向结论
- 相同 checkpoint 下，metadata no-all-ratings history-only 的 `N@20` 比 plain history+rating 高 `0.034743` 到 `0.037242`。
- 最大 `N@20` 差值出现在 `checkpoint-249`，metadata 版本为 `0.069664`，plain 版本为 `0.032422`。
- 两个版本的训练超参相同，差异主要来自 query 输入：metadata 版本提供 store、format、categories、features、description 和 details，plain 版本只提供标题和 rating。

## 横向对比：metadata history-only 与 metadata history+CoT

### 对比口径
| 字段 | metadata no-all-ratings history-only | metadata no-all-ratings history+CoT |
|---|---|---|
| base 模型 | Qwen3-Embedding-0.6B | Qwen3-Embedding-0.6B |
| 训练数据 | `phase0_embedder_cds_meta_compact_no_all_ratings_history_only.jsonl` | `phase0_embedder_cds_meta_compact_no_all_ratings_history_plus_tagged_cot_manual_filled_full_target.jsonl` |
| query 内容 | compact item metadata history | compact item metadata history + tagged CoT |
| batch / grad | `128 / 1` | `128 / 1` |
| epoch | `5` | `5` |
| max length | `4096` | `4096` |
| 评测样本数 | `1341` | `1341` |
| 候选 item 数 | `12000` | `12000` |

### 主指标横向对比
| checkpoint | history-only H@20 | history-only N@20 | history+CoT H@20 | history+CoT N@20 | ΔH@20 | ΔN@20 |
|---|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | 0.170022 | 0.065174 | 0.163311 | 0.064823 | -0.006711 | -0.000351 |
| `checkpoint-166` | 0.178225 | 0.068194 | 0.175988 | 0.068991 | -0.002237 | 0.000797 |
| `checkpoint-249` | 0.181954 | 0.069664 | 0.173751 | 0.068883 | -0.008203 | -0.000781 |
| `checkpoint-332` | 0.181954 | 0.069321 | 0.175988 | 0.069862 | -0.005966 | 0.000541 |
| `checkpoint-415` | 0.179717 | 0.069153 | 0.173751 | 0.069206 | -0.005966 | 0.000053 |

### 全指标横向对比
| checkpoint | 训练版本 | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | metadata history-only | 0.071588 | 0.036370 | 0.121551 | 0.052729 | 0.170022 | 0.065174 |
| `checkpoint-83` | metadata history+CoT | 0.074571 | 0.038684 | 0.124534 | 0.054949 | 0.163311 | 0.064823 |
| `checkpoint-166` | metadata history-only | 0.076808 | 0.039079 | 0.123043 | 0.054156 | 0.178225 | 0.068194 |
| `checkpoint-166` | metadata history+CoT | 0.079045 | 0.040727 | 0.134228 | 0.058526 | 0.175988 | 0.068991 |
| `checkpoint-249` | metadata history-only | 0.078300 | 0.039914 | 0.126025 | 0.055456 | 0.181954 | 0.069664 |
| `checkpoint-249` | metadata history+CoT | 0.081283 | 0.041628 | 0.137957 | 0.059861 | 0.173751 | 0.068883 |
| `checkpoint-332` | metadata history-only | 0.079045 | 0.039885 | 0.124534 | 0.054706 | 0.181954 | 0.069321 |
| `checkpoint-332` | metadata history+CoT | 0.079791 | 0.041519 | 0.137957 | 0.060309 | 0.175988 | 0.069862 |
| `checkpoint-415` | metadata history-only | 0.079791 | 0.040499 | 0.125280 | 0.055276 | 0.179717 | 0.069153 |
| `checkpoint-415` | metadata history+CoT | 0.079791 | 0.041388 | 0.138702 | 0.060387 | 0.173751 | 0.069206 |

### best checkpoint 相对变化
基线使用 metadata history-only best `checkpoint-249`，CoT 使用 metadata history+CoT best `checkpoint-332`。

| 指标 | history-only best | history+CoT best | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| H@5 | 0.078300 | 0.079791 | 0.001491 | 1.904% |
| N@5 | 0.039914 | 0.041519 | 0.001605 | 4.021% |
| H@10 | 0.126025 | 0.137957 | 0.011932 | 9.468% |
| N@10 | 0.055456 | 0.060309 | 0.004853 | 8.751% |
| H@20 | 0.181954 | 0.175988 | -0.005966 | -3.279% |
| N@20 | 0.069664 | 0.069862 | 0.000198 | 0.284% |

### 横向结论
- history+CoT best `N@20=0.069862`，history-only best `N@20=0.069664`，差值 `0.000198`。
- history+CoT 的 `H@20` 在所有相同 checkpoint 上都低于 history-only，差值范围为 `0.002237` 到 `0.008203`。
- history+CoT 在 `H@10` 和 `N@10` 上更高，`checkpoint-415` 的 `N@10=0.060387`，history-only best `N@10=0.055456`，差值 `0.004931`。
- 当前 CoT 拼接方式主要改变前 10 位排序，未增加 top-20 命中数；下一步应检查 CoT 文本长度占比和 query 中 metadata 的截断位置。

## CDs_and_Vinyl query-side metadata 增量消融

### 基本信息
| 字段 | 内容 |
|---|---|
| 实验日期 | 2026-07-04 |
| 任务 | Amazon CDs_and_Vinyl next-item retrieval |
| 模型 | Qwen3-Embedding-0.6B |
| 训练方式 | contrastive embedding training |
| 训练数据 | `ablantion/datas/processed_datas/cds_query_ablation/cds_query_*.jsonl` |
| 评测数据 | `ablantion/datas/processed_datas/cds_query_ablation_test/cds_test_query_*.jsonl` |
| 评测模式 | `user_history`，直接读取测试 JSONL 内的 query |
| 候选物品数 | 12000 |
| 测试样本数 | 1341 |
| checkpoint root | `/home/user/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/cds_query_ablation` |
| eval dir | `/home/user/aaai_pro/outputs/rrec_amazon/eval/CDs_and_Vinyl/cds_query_ablation` |
| 运行队列 | `cds_query_ablation_all_20260704` |
| 代码 commit | `30d890fbc8193821db77bc726634e8fffc394939` |

### 模型和训练配置
| 参数 | 值 | 说明 |
|---|---:|---|
| `MODEL_PATH` | `/home/user/models_hf/Qwen3-Embedding-0.6B` | base embedding 模型 |
| `MAX_LENGTH` | `4096` | query 和 positive 编码最大 token 长度 |
| `BATCH_SIZE` | `128` | 单卡 batch size |
| `GRAD_ACCUM` | `1` | 梯度累积步数 |
| effective batch | `128` | `batch_size * grad_accum` |
| `EPOCHS` | `5` | 训练轮数 |
| `LEARNING_RATE` | `6e-6` | 学习率 |
| `ATTN_IMPLEMENTATION` | `flash_attention_2` | attention 实现 |
| `CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA allocator 配置 |

### 数据口径
| 字段 | 内容 |
|---|---|
| 消融变量 | 只改 query 侧 metadata 字段 |
| target/positive | 所有消融共享同一 target text，positive 与源 test target 文本对齐 |
| query 增量顺序 | title -> store -> categories -> features -> description -> details/full compact |
| rating/catalog stats | 不包含 history rating，不包含 catalog stats |
| prompt/CoT 格式 | 无 CoT |
| target text 截断 | 不截断 |
| history item 数 | 使用构造文件内的 `history_item_ids`，训练 token max length 为 4096 |
| 负样本策略 | 无显式负样本，只使用 batch 内负样本 |
| seen item mask | 评测脚本按 RRec examples 中 history 过滤已见物品 |
| 运行状态 | 6 组训练和评测均写入 `.train_done` / `.eval_done`；日志未检出 OOM、Traceback、NaN 或 RuntimeError |

### best checkpoint 汇总
| query 侧字段 | best checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| title only | `checkpoint-166` | 0.034303 | 0.018716 | 0.058911 | 0.026706 | 0.082028 | 0.032589 |
| title + store | `checkpoint-332` | 0.089485 | 0.047916 | 0.137211 | 0.063383 | 0.197614 | 0.078753 |
| title + store + categories | `checkpoint-332` | 0.093960 | 0.048947 | 0.154362 | 0.068490 | 0.199851 | 0.079958 |
| title + store + categories + features | `checkpoint-415` | 0.092468 | 0.047946 | 0.155108 | 0.068323 | 0.197614 | 0.079139 |
| title + store + categories + features + description | `checkpoint-415` | 0.093214 | 0.048199 | 0.150634 | 0.066781 | 0.202088 | 0.079763 |
| full compact no-all-ratings | `checkpoint-332` | 0.087994 | 0.044393 | 0.148397 | 0.063948 | 0.197614 | 0.076399 |

### title only checkpoint 指标
| checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | 0.029828 | 0.015696 | 0.056674 | 0.024210 | 0.084265 | 0.031305 | 3266.7375 | 2191 |
| `checkpoint-166` | 0.034303 | 0.018716 | 0.058911 | 0.026706 | 0.082028 | 0.032589 | 3137.9098 | 2021 |
| `checkpoint-249` | 0.035794 | 0.019103 | 0.058911 | 0.026509 | 0.079791 | 0.031906 | 3110.5384 | 2012 |
| `checkpoint-332` | 0.037286 | 0.019824 | 0.058166 | 0.026465 | 0.081283 | 0.032368 | 3102.9657 | 1999 |
| `checkpoint-415` | 0.035794 | 0.019143 | 0.057420 | 0.026092 | 0.082774 | 0.032553 | 3103.8971 | 2009 |

### title + store checkpoint 指标
| checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | 0.087994 | 0.048234 | 0.141685 | 0.065398 | 0.184937 | 0.076313 | 2128.1141 | 750 |
| `checkpoint-166` | 0.087994 | 0.047707 | 0.137957 | 0.063649 | 0.196868 | 0.078565 | 1985.6890 | 639 |
| `checkpoint-249` | 0.090231 | 0.047767 | 0.139448 | 0.063584 | 0.199105 | 0.078690 | 1967.1961 | 606 |
| `checkpoint-332` | 0.089485 | 0.047916 | 0.137211 | 0.063383 | 0.197614 | 0.078753 | 1961.2260 | 616 |
| `checkpoint-415` | 0.091723 | 0.048404 | 0.140940 | 0.064177 | 0.198359 | 0.078748 | 1961.1394 | 611 |

### title + store + categories checkpoint 指标
| checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | 0.086503 | 0.045685 | 0.140940 | 0.063398 | 0.192394 | 0.076497 | 1960.4676 | 615 |
| `checkpoint-166` | 0.087248 | 0.046216 | 0.148397 | 0.066133 | 0.193885 | 0.077669 | 1791.3177 | 490 |
| `checkpoint-249` | 0.093214 | 0.048464 | 0.152871 | 0.067887 | 0.198359 | 0.079404 | 1773.7301 | 486 |
| `checkpoint-332` | 0.093960 | 0.048947 | 0.154362 | 0.068490 | 0.199851 | 0.079958 | 1769.2528 | 485 |
| `checkpoint-415` | 0.092468 | 0.047936 | 0.155108 | 0.068218 | 0.197614 | 0.078977 | 1769.8345 | 482 |

### title + store + categories + features checkpoint 指标
| checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | 0.087994 | 0.046692 | 0.143922 | 0.064743 | 0.193139 | 0.077233 | 1959.3311 | 588 |
| `checkpoint-166` | 0.090977 | 0.046645 | 0.147651 | 0.065136 | 0.196122 | 0.077399 | 1786.1402 | 480 |
| `checkpoint-249` | 0.092468 | 0.047541 | 0.155108 | 0.067828 | 0.199851 | 0.079134 | 1769.4072 | 481 |
| `checkpoint-332` | 0.094705 | 0.048504 | 0.153617 | 0.067747 | 0.197614 | 0.078982 | 1767.0783 | 482 |
| `checkpoint-415` | 0.092468 | 0.047946 | 0.155108 | 0.068323 | 0.197614 | 0.079139 | 1769.8121 | 478 |

### title + store + categories + features + description checkpoint 指标
| checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | 0.085757 | 0.043193 | 0.140940 | 0.061115 | 0.187174 | 0.073113 | 1886.9925 | 615 |
| `checkpoint-166` | 0.090977 | 0.046052 | 0.149142 | 0.064835 | 0.193139 | 0.075922 | 1708.4765 | 479 |
| `checkpoint-249` | 0.092468 | 0.046750 | 0.150634 | 0.065641 | 0.199105 | 0.077954 | 1693.7502 | 481 |
| `checkpoint-332` | 0.092468 | 0.046702 | 0.150634 | 0.065499 | 0.201342 | 0.078282 | 1689.0134 | 460 |
| `checkpoint-415` | 0.093214 | 0.048199 | 0.150634 | 0.066781 | 0.202088 | 0.079763 | 1688.6309 | 474 |

### full compact no-all-ratings checkpoint 指标
| checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | 0.085757 | 0.043268 | 0.140194 | 0.060826 | 0.190902 | 0.073799 | 1873.2632 | 611 |
| `checkpoint-166` | 0.087248 | 0.043322 | 0.143922 | 0.061640 | 0.193885 | 0.074228 | 1687.1029 | 484 |
| `checkpoint-249` | 0.088740 | 0.044263 | 0.147651 | 0.063475 | 0.197614 | 0.076147 | 1672.7591 | 464 |
| `checkpoint-332` | 0.087994 | 0.044393 | 0.148397 | 0.063948 | 0.197614 | 0.076399 | 1668.3341 | 458 |
| `checkpoint-415` | 0.088740 | 0.044341 | 0.149142 | 0.063971 | 0.195377 | 0.075649 | 1667.4146 | 457 |

### 结论
- `title + store + categories` 在 `checkpoint-332` 取得最高 `N@20=0.079958`，对应 `H@20=0.199851`。
- 从 `title only` 到 `title + store`，best `N@20` 从 `0.032589` 增加到 `0.078753`，差值 `0.046164`；CDs 的 `store` 字段主要写入 artist/format，直接补充用户历史中的同艺术家偏好。
- 从 `title + store` 到 `title + store + categories`，best `N@20` 增加 `0.001205`，`H@10` 从 `0.137211` 增加到 `0.154362`。
- 继续加入 `features`、`description`、`details/full compact` 没有超过 `title + store + categories` 的 `N@20`。`features` 在 item_info 中覆盖很低，`description/details` 增加文本长度后没有带来 top-20 排序收益。
- 该组消融只改 query 侧字段，target/positive 和候选集合固定；因此指标差异主要来自 query 中 metadata 字段改变。

### 与 RRec 图中指标对比

图中 RRec 指标按列解释为 `H@5, N@5, H@10, N@10, H@20, N@20`，数值为 `0.0513, 0.0372, 0.0647, 0.0414, 0.0818, 0.0457`。下面只比较测试指标，不比较训练成本。

| 模型或输入口径 | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | N@20 相对 RRec |
|---|---:|---:|---:|---:|---:|---:|---:|
| RRec 图中结果 | 0.051300 | 0.037200 | 0.064700 | 0.041400 | 0.081800 | 0.045700 | 0.00% |
| title only best | 0.034303 | 0.018716 | 0.058911 | 0.026706 | 0.082028 | 0.032589 | -28.69% |
| title + store best | 0.089485 | 0.047916 | 0.137211 | 0.063383 | 0.197614 | 0.078753 | 72.33% |
| title + store + categories best | 0.093960 | 0.048947 | 0.154362 | 0.068490 | 0.199851 | 0.079958 | 74.96% |
| full compact no-all-ratings best | 0.087994 | 0.044393 | 0.148397 | 0.063948 | 0.197614 | 0.076399 | 67.18% |
| metadata history-only best | 0.078300 | 0.039914 | 0.126025 | 0.055456 | 0.181954 | 0.069664 | 52.44% |
| metadata history+CoT best | 0.079791 | 0.041519 | 0.137957 | 0.060309 | 0.175988 | 0.069862 | 52.87% |

| 对比项 | H@5 差值 | N@5 差值 | H@10 差值 | N@10 差值 | H@20 差值 | N@20 差值 |
|---|---:|---:|---:|---:|---:|---:|
| title + store + categories best - RRec | 0.042660 | 0.011747 | 0.089662 | 0.027090 | 0.118051 | 0.034258 |
| title + store best - RRec | 0.038185 | 0.010716 | 0.072511 | 0.021983 | 0.115814 | 0.033053 |
| full compact no-all-ratings best - RRec | 0.036694 | 0.007193 | 0.083697 | 0.022548 | 0.115814 | 0.030699 |
| metadata history-only best - RRec | 0.027000 | 0.002714 | 0.061325 | 0.014056 | 0.100154 | 0.023964 |
| metadata history+CoT best - RRec | 0.028491 | 0.004319 | 0.073257 | 0.018909 | 0.094188 | 0.024162 |

对比结论：`title + store + categories` best 比 RRec 图中结果高 `N@20=0.034258`，相对增加 `74.96%`；`H@20` 高 `0.118051`，相对增加 `144.32%`。`title only` 的 `N@20` 低于 RRec `0.013111`，说明只用历史标题不能复现 RRec 图中指标；加入 `store` 后，artist/format 信号使 `N@20` 超过 RRec `0.033053`。

## CDs query 消融：title + store + categories seen item mask 重评

### 基本信息
| 字段 | 内容 |
|---|---|
| 实验日期 | 2026-07-04 |
| 任务 | 只重评 `title + store + categories` 普通 5 epoch checkpoint |
| checkpoint root | `/home/user/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/cds_query_ablation/title_store_categories` |
| eval dir | `/home/user/aaai_pro/outputs/rrec_amazon/eval/CDs_and_Vinyl/cds_query_ablation_masked_seen/title_store_categories` |
| 测试数据 | `/home/user/aaai_pro/ablantion/datas/processed_datas/cds_query_ablation_test/cds_test_query_title_store_categories.jsonl` |
| 候选 item | `/home/user/aaai_pro/github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl` |
| 测试样本数 | `1341` |
| 候选物品数 | `12000` |
| query 构造 | 直接读取测试 JSONL 的 `user_history` |
| seen item mask | `history_item_id` 分数置为 `-inf`，同时屏蔽 pad item；`target_in_history_count=0` |
| 本次范围 | 不重评其他 query 消融；不重评 `cds_query_ablation_lr1e-5_epoch10/title_store_categories` |

### checkpoint 指标
| checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | mean rank | median rank | ΔN@20 vs no mask |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-83` | 0.118568 | 0.082696 | 0.158837 | 0.095637 | 0.204325 | 0.107257 | 1956.7189 | 609 | 0.030760 |
| `checkpoint-166` | 0.125280 | 0.084704 | 0.167040 | 0.098158 | 0.202834 | 0.107202 | 1787.5638 | 487 | 0.029532 |
| `checkpoint-249` | 0.129754 | 0.087361 | 0.172260 | 0.101046 | 0.206562 | 0.109667 | 1769.9776 | 485 | 0.030263 |
| `checkpoint-332` | 0.128262 | 0.087555 | 0.170768 | 0.101381 | 0.210291 | 0.111288 | 1765.4899 | 481 | 0.031329 |
| `checkpoint-415` | 0.129008 | 0.087341 | 0.173005 | 0.101516 | 0.208799 | 0.110460 | 1766.0738 | 480 | 0.031483 |

### 结论
- seen item mask 后最佳 checkpoint 仍为 `checkpoint-332`，`N@20=0.111288`，`H@20=0.210291`。
- 与未 mask 的同 checkpoint 相比，`checkpoint-332` 的 `N@20` 从 `0.079958` 增加到 `0.111288`，差值 `0.031329`；`H@20` 从 `0.199851` 增加到 `0.210291`，差值 `0.010440`。
- 每条样本平均屏蔽 `4.9679` 个历史 item，等于测试集平均历史长度；`target_in_history_count=0`，所以本次变化来自移除历史已交互候选和 pad item。

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
| checkpoint | H@5 | N@5 | H@10 | N@10 | H@20 | N@20 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |

### 结论
- 最佳 checkpoint：
- 主要提升或退化：
- 与关键 baseline 的差异：
- 下一步动作：
