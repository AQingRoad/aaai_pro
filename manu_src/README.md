# manu_src 精简基线

本目录只保留 `qwen3emb06b_time_title_store_categories_desc256_details256_bs128_ga1_lr2e5_ep5_len4096_seed42_test_final` 实验的现有评测结果、基础数据和直接依赖代码。后续提示词实验从这套数据构建与检索口径重新开始。

## 保留结果

结果目录：

```text
eval_results/CDs_and_Vinyl/embedding/
qwen3emb06b_time_title_store_categories_desc256_details256_bs128_ga1_lr2e5_ep5_len4096_seed42_test_final/
```

当前保留 epoch 1–5 的全量 12000 候选评测。各轮 test 指标为：

| Epoch | NDCG@20 | HR@20 | MRR |
|---:|---:|---:|---:|
| 1 | 0.1245939749 | 0.2259507830 | 0.0988324335 |
| 2 | 0.1195042982 | 0.2170022371 | 0.0952078594 |
| 3 | 0.1220578934 | 0.2207307979 | 0.0973567367 |
| 4 | 0.1179300618 | 0.2170022371 | 0.0930284953 |
| 5 | 0.1172229440 | 0.2162565250 | 0.0922480013 |

共同设置：

- test 样本：1341
- seed：42
- max_length：4096
- seen-item mask：开启
- item 文本：title、main category、store/artist/format、categories、features、Description 前 256 字符、Details 前 256 字符

评测记录中的 checkpoint 位于旧 A100 路径，本地 `manu_src` 不保留模型权重。

## 保留数据

`datas/CDs_and_Vinyl/arrow_to_jsonls/` 保存 train、valid、test 和 item_info 基础 JSONL。`datas/CDs_and_Vinyl/train_datas/{train,val,test}.jsonl` 保存该 embedding 模型实际使用的无评分 processed pair，行数分别为 10722、1340、1341。

`datas/CDs_and_Vinyl/train_datas/time_title_rating_store_categories_desc256_details256_v1.0/` 保存新增评分的严格配对版本。它只在每条历史物品中加入 `Rating: x.x star`；移除评分片段后，query 与无评分版本逐字节一致，positive 和其它字段也保持一致。

## 保留代码

```text
scripts/pre_datas/arrow_to_jsonl.py
scripts/pre_datas/format_positive.py
scripts/pre_datas/build_title_store_categories.py
scripts/models/train_embedding.py
scripts/eval/evaluate_embedding_fullset.py
```

代码依赖关系：

```text
arrow_to_jsonls
  -> build_title_store_categories.py + format_positive.py
  -> train_embedding.py
  -> evaluate_embedding_fullset.py
       -> format_positive.py
       -> train_embedding.py 中的编码函数
```

## 当前输入口径

`build_title_store_categories.py` 默认构造无评分 history；传入 `--include-ratings` 后加入评分。两种版本都包含相对时间、标题、store/artist/format、类别路径、Description 前 256 字符和 Details 前 256 字符。positive 使用 `format_positive.py` 构造。

训练和该目录中的五轮评测沿用 `keep_instruction_and_most_recent_history_tokens`：query 超过 4096 tokens 时保留 embedding instruction 与最近历史 token。positive 和候选 item 禁止截断。

下一版提示词定稿前，需要先明确 prompt 使用哪些字段，并同步修改数据构建、训练 query 和评测 query，防止同名实验混用不同输入。
