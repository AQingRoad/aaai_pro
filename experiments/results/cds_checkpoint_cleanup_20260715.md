# CDs_and_Vinyl checkpoint 清理记录（2026-07-15）

## 清理原则

本次只删除三类权重：已有逐 checkpoint 测试 JSON 的非保留 embedding checkpoint、仅用于启动验证的 smoke checkpoint、已有 LoRA adapter 的 merged 模型。评测 JSON、训练参数、实验报告和保留 checkpoint 不删除。尚未评测、尚未归档或可能继续训练的目录不进入本次清理。

清理前，A100 的 `/home/user/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl` 占用约 `114G`，`/home` 可用空间约 `287G`。执行时只有对齐实验的 GLM CoT 生成进程在运行，没有 embedding 训练或评测进程。

## 保留 checkpoint

| 实验目录 | 保留项 | 保留依据 |
|---|---:|---|
| `cds_query_ablation_lr1e-5_epoch10/title_store_categories` | `checkpoint-415` | 登记的该组测试最优项 |
| `cds_query_ablation_lr2e-5_epoch4/title_store_categories` | `checkpoint-83` | 当前旧版 history 检索器代表项 |
| `cds_query_ablation_lr3e-5_epoch5/title_store_categories` | `checkpoint-83` | 登记的该组测试最优项 |
| `cds_query_ablation_lr5e-5_epoch5/title_store_categories` | `checkpoint-83` | 登记的该组测试最优项 |
| `cds_query_ablation_cot_lr2e-5_epoch5/title_store_categories_no_trunc_plus_cot` | `checkpoint-83` | 旧版 external CoT/reward 链路代表项 |
| `qwen3_embedding_0p6b_cds_meta_compact_no_all_ratings_history_only_len4096_batch128_accum1_epoch5` | `checkpoint-249` | 登记的该组测试最优项 |
| `qwen3_embedding_0p6b_cds_meta_compact_no_all_ratings_history_plus_cot_len4096_batch128_accum1_epoch5` | `checkpoint-332` | 登记的该组测试最优项 |
| `qwen3_embedding_0p6b_cds_plain_history_rating_only_len4096_batch128_epoch5` | `checkpoint-415` | 登记的该组测试最优项 |

## 删除项

- `cds_query_ablation_smoke/title_only/checkpoint-1`。
- `cds_query_ablation_lr1e-5_epoch10/title_store_categories` 下除 `checkpoint-415` 外的 9 个 checkpoint。
- `cds_query_ablation_lr2e-5_epoch4/title_store_categories` 下除 `checkpoint-83` 外的 3 个 checkpoint。
- `cds_query_ablation_lr3e-5_epoch5/title_store_categories` 下除 `checkpoint-83` 外的 4 个 checkpoint。
- `cds_query_ablation_lr5e-5_epoch5/title_store_categories` 下除 `checkpoint-83` 外的 4 个 checkpoint。
- `cds_query_ablation_cot_lr2e-5_epoch5/title_store_categories_no_trunc_plus_cot` 下除 `checkpoint-83` 外的 4 个 checkpoint。
- 三组 metadata/history baseline 下除上表保留项外的 12 个 checkpoint。
- `qwen25_3b_lora_grpo_ndcg100_ckpt300_merged` 和 `qwen25_3b_lora_grpo_ndcg100_ckpt600_merged`。对应 LoRA adapter 分别保留在 dense-rank GRPO 实验的 `checkpoint-300` 和 `checkpoint-600`，每个 adapter 目录约 `458M`，均含 `adapter_config.json` 与 `adapter_model.safetensors`。

合计删除 39 个权重目录，其中 36 个为非保留 embedding checkpoint、1 个为 smoke checkpoint、2 个为可重建 merged 模型。

## 明确保留的其它内容

- 所有 `outputs/rrec_amazon/eval/CDs_and_Vinyl` 评测 JSON。
- 当前对齐实验 clean worktree 及其选中 history `checkpoint-249`。
- 本地 `prepared/CDs_and_Vinyl/qwen3_embedding_cds/checkpoint-83`，其权重哈希未找到同尺寸远端副本。
- `cds_query_ablation`、dual-view、SFT、GRPO adapter、未评测实验和当前 CoT 生成数据。

## 执行结果

- 39 个目标目录全部删除，路径残留检查为 0。
- `du -sk` 汇总删除 `55,716,332 KiB`，约 `53.1 GiB`。
- checkpoint 根目录占用从约 `114G` 降至约 `61G`。
- `/home` 可用空间从约 `287G` 增至约 `340G`。
- 上表列出的 8 个代表 checkpoint 均保留，每个约 `1.2G`。
- 删除期间，对齐实验 GLM CoT 生成进程持续运行；删除操作没有访问 clean run worktree。
