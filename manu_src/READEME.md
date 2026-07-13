# manu_src 目录说明

`manu_src` 是当前实验的独立工作目录，集中存放原始数据、预处理脚本、模型训练脚本、checkpoint 和评测结果。当前已接入 `CDs_and_Vinyl` 数据集，后续新增数据集时沿用相同层级。

## 1. 目录结构

```text
manu_src/
├── datas/
│   └── CDs_and_Vinyl/
│       ├── arrow_datas/                 # Hugging Face DatasetDict 原始 Arrow 数据
│       │   ├── dataset_dict.json
│       │   ├── train/
│       │   ├── valid/
│       │   ├── test/
│       │   └── item_info/
│       ├── arrow_to_jsonls/             # Arrow 数据逐 split 转换后的 JSONL
│       │   ├── train.jsonl
│       │   ├── val.jsonl
│       │   ├── test.jsonl
│       │   └── item_info.jsonl
│       ├── train_datas/                  # 可直接交给模型训练的 query-positive pair
│       │   ├── train.jsonl
│       │   ├── val.jsonl
│       │   └── test.jsonl
│       └── output_datas/                 # 预留的数据处理结果目录
├── scripts/
│   ├── pre_datas/
│   │   ├── arrow_to_jsonl.py            # DatasetDict Arrow 转 JSONL
│   │   ├── build_title_store_categories.py
│   │   │                                  # 从交互记录和 item_info 构建训练 pair
│   │   └── format_positive.py           # 按 item_info 重建带字段标签的 positive
│   └── models/
│       └── train_embedding.py           # Qwen3 Embedding 对比学习训练入口
├── model_outputs/
│   └── CDs_and_Vinyl/
│       ├── embedding/                   # Embedding 模型 checkpoint 和训练日志
│       ├── sft/                         # SFT 模型 checkpoint 和训练日志
│       └── grpo/                        # GRPO 模型 checkpoint 和训练日志
├── eval_results/
│   └── CDs_and_Vinyl/
│       ├── embedding/                   # Embedding 检索评测结果
│       ├── sft/                         # SFT 评测结果
│       └── grpo/                        # GRPO 评测结果
└── READEME.md                           # 本文件
```

## 2. 数据流

```text
arrow_datas
    │  arrow_to_jsonl.py
    ▼
arrow_to_jsonls
    │  build_title_store_categories.py
    ▼
train_datas
    │  train_embedding.py
    ▼
model_outputs/CDs_and_Vinyl/embedding/<实验名>
    │  后续评测脚本
    ▼
eval_results/CDs_and_Vinyl/embedding/<实验名>
```

各阶段职责如下：

1. `arrow_datas` 保存 RRec 生成的 Hugging Face DatasetDict，不修改原始字段和样本顺序。
2. `arrow_to_jsonls` 将 train、valid、test 和 item_info 转成便于检查的 JSONL。源 split 名为 `valid`，输出文件名为 `val.jsonl`。
3. `train_datas` 将交互序列转成 `query`，将目标物品 metadata 转成 `positive`，并保留样本 ID、target item ID 和时间戳等审计字段。
4. `model_outputs` 按数据集、训练类型和实验名保存模型；`eval_results` 使用相同层级保存评测结果。

## 3. 当前数据规模

| 目录或文件 | 行数/大小 | 内容 |
|---|---:|---|
| `arrow_datas/` | 约 20 MB | 原始 DatasetDict Arrow 文件 |
| `arrow_to_jsonls/item_info.jsonl` | 12001 行 | 12000 个真实物品和一个 padding item |
| `arrow_to_jsonls/train.jsonl` | 10722 行 | 原始训练交互 |
| `arrow_to_jsonls/val.jsonl` | 1340 行 | 原始验证交互 |
| `arrow_to_jsonls/test.jsonl` | 1341 行 | 原始测试交互 |
| `train_datas/train.jsonl` | 10722 行 | 处理后的训练 pair |
| `train_datas/val.jsonl` | 1340 行 | 处理后的验证 pair |
| `train_datas/test.jsonl` | 1341 行 | 处理后的测试 pair |

## 4. 脚本职责

### 4.1 `arrow_to_jsonl.py`

输入包含 `dataset_dict.json` 的 `arrow_datas`，依次读取 `train`、`valid`、`test` 和 `item_info`，按原始顺序写入 JSONL。该脚本只改变存储格式，不修改字段值。

```bash
python3 manu_src/scripts/pre_datas/arrow_to_jsonl.py \
  --input-dir manu_src/datas/CDs_and_Vinyl/arrow_datas \
  --output-dir manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls
```

### 4.2 `build_title_store_categories.py`

该脚本按 `item_id` 关联 `item_info.jsonl`，为每条交互构造训练 pair。虽然文件名保留 `title_store_categories`，当前实际 query 字段为：

```text
relative_time + title + store + categories + description + details
```

每个历史物品的文本格式为：

```text
N. Time: <相对目标交互的时间差>;
   Title: <标题>;
   Store/artist/format: <店铺、艺人或格式>;
   Categories: <完整类别路径>;
   Description: <最多400字符>;
   Details: <最多400字符>
```

脚本会清理 history 中显式 `Amazon ASIN` 和裸露的十位 ASIN，并拒绝包含 `[TRUNCATED]` 标记的 query。输出样本包含 `example_id`，格式为：

```text
CDs_and_Vinyl:<split>:<interaction_id>:<user_id>
```

```bash
python3 manu_src/scripts/pre_datas/build_title_store_categories.py \
  --input-dir manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls \
  --output-dir manu_src/datas/CDs_and_Vinyl/train_datas
```

### 4.3 `format_positive.py`

该脚本根据 `target_item_id` 从 `item_info.jsonl` 重建 positive，只替换输入样本的 `positive` 字段。字段之间统一使用分号分隔，可能包含：

```text
Title
Main category
Store/artist/format
Categories
Features
Description
Details
```

当前代码将 Description 的全部段落先拼接，再保留前 400 个字符；Details 的全部键值按原始顺序拼接，再保留前 400 个字符。输入文件和输出文件不能使用同一路径。

```bash
python3 manu_src/scripts/pre_datas/format_positive.py \
  --input <待处理JSONL> \
  --item-info manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl \
  --output <新JSONL路径>
```

### 4.4 `train_embedding.py`

该脚本使用 Qwen3 Embedding 编码 query 和 positive，执行 L2 归一化后计算 multi-positive InfoNCE：

\[
s_{ij}=\frac{q_i^\top d_j}{\tau},\qquad
\mathcal{L}_i=
\log\sum_j e^{s_{ij}}
-
\log\sum_{j:y_j=y_i}e^{s_{ij}}
\]

同一 batch 内 `target_item_id` 相同的文档都算正例，避免把同一目标物品误作负例。脚本采用 Qwen3 最后一个有效 token 的 hidden state 作为 embedding，并在训练前审计 query 和 positive 的真实 token 长度。query 超过 `max_length` 时保留完整 instruction 和最近历史 token，移除最旧历史 token；positive 超限时直接停止，禁止截断。

当前版本只要求 `--train-file` 和 `--test-file`。训练过程不读取 validation，不根据测试指标选模；脚本保存每轮 checkpoint，并在最后一轮训练结束后测试一次。

## 5. 模型输出命名规范

模型产物先按数据集分类，再按训练类型分类：

```text
model_outputs/<数据集>/<embedding|sft|grpo>/<实验名>/
```

本次计划中的 embedding 实验目录为：

```text
manu_src/model_outputs/CDs_and_Vinyl/embedding/
qwen3emb06b_time_title_store_categories_desc_details_bs128_ga1_lr2e5_ep5_len4096_seed42_test_final/
```

实验名记录模型、query 口径、batch size、梯度累积、学习率、epoch、最大长度、随机种子和测试策略。计划输出结构如下：

```text
checkpoint-epoch-01/
checkpoint-epoch-02/
checkpoint-epoch-03/
checkpoint-epoch-04/
checkpoint-epoch-05/
run_config.json
train_metrics.jsonl
test_metrics.json
token_audit.json
train.log
```

本轮不按 test loss 选取 checkpoint，也不额外复制 `checkpoint-best` 或 `checkpoint-last`。第五轮 checkpoint 作为最终模型，测试集合只在第五轮训练完成后读取并评测一次。

## 6. 评测结果命名规范

评测结果与模型目录使用同一个实验名：

```text
eval_results/<数据集>/<embedding|sft|grpo>/<实验名>/
```

例如，本次 embedding 实验的评测结果应写入：

```text
manu_src/eval_results/CDs_and_Vinyl/embedding/
qwen3emb06b_time_title_store_categories_desc_details_bs128_ga1_lr2e5_ep5_len4096_seed42_test_final/
```

## 7. 当前状态

- 原始 Arrow、转换后的 JSONL 和训练 pair 已放入统一目录。
- 三个数据 split 均已生成，训练数据没有空 query 或空 positive。
- Embedding 训练脚本已具备 token 长度审计、multi-positive InfoNCE、BF16、FlashAttention 2、梯度检查点和固定随机种子 42。
- `model_outputs` 和 `eval_results` 已建立 embedding、SFT、GRPO 分类目录，目前为空。
- 当前 embedding 实验按 batch size 128、梯度累积 1、5 个 epoch 和末轮单次测试的口径启动。
