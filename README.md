# Rubric-Gated CoT 推荐训练流水线

本仓库实现 `method.md` 中的推荐推理训练流程，当前主要服务
`CDs_and_Vinyl` 数据集。代码入口已经按功能分组到 `scripts/` 子目录，
根目录下仍保留同名软链接，旧命令可以继续使用。

目录结构和脚本分类见：

- [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)
- [`scripts/README.md`](scripts/README.md)

## 流程概览

```text
构建推荐样本
-> 生成 CoT 候选
-> Rubric 打分
-> merge
-> gain / NDCG 增益计算
-> select
-> SFT / GRPO 数据集
-> SFT
-> GRPO
-> full-candidate ranking 评测
```

当前默认的 `gain` 计算使用 `NDCG@100(cot_query) - NDCG@100(history_query)`。
GRPO 在线 reward 由 `rubric_format`、`rubric_quality`、`rubric_gated_gain`
三个 reward 组成。

## 目录约定

```text
rubric_cot_pipeline/      共享 Python 模块
scripts/                  数据、CoT、筛选、训练、评测脚本
configs/                  配置模板；真实 key 文件不入库
github_artifacts/         随 GitHub 分发的小型数据和评测文件
data/                     本地原始/转换数据，忽略提交
outputs/                  本地中间结果和评测结果，忽略提交
checkpoints/              本地模型权重，忽略提交
prepared/                 本地上传或训练准备目录，忽略提交
```

不要提交 API key、模型权重、缓存目录和服务器本地输出。

## Tidal 服务器路径

当前脚本默认支持下面这套路径：

```text
项目根目录: /mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro
conda 环境: /root/miniconda3/envs/swift
LLM base:  /mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3/4B
Embedding base: /mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3_embedding/0.6B
```

进入项目：

```bash
cd /mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro
conda activate swift
git pull
```

## 构建 DeepSeek one-CoT 的 SFT / GRPO 数据

该脚本读取 GitHub 中的 one-CoT 候选和 GLM-5.2 rubric 文件，使用指定
embedding checkpoint 计算 NDCG gain，筛选 top 20% 作为 SFT，剩余样本构建
GRPO。

```bash
cd /mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro
conda activate swift

ROOT=/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro \
VENV=/root/miniconda3/envs/swift \
COT=$ROOT/github_artifacts/CDs_and_Vinyl/cot/cot_candidate_one_lists_deepseek_v4_pro_low.jsonl \
RUBRIC=$ROOT/github_artifacts/CDs_and_Vinyl/cot/cot_candidate_one_lists_deepseek_v4_pro_low.rubric_glm_5_2.jsonl \
ITEM_INFO=$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl \
EMBEDDING_MODEL=$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_cot_deepseek_v4_pro_low_one_cot_global_bsz128_xgpu/checkpoint-167 \
RUN_TAG=deepseek_one_glm52_ckpt167 \
DEVICES=0,1,2,3,4,5,6,7 \
TOP_PERCENT=0.2 \
MIN_GAIN=0 \
bash scripts/build_deepseek_one_glm52_top20_sft_grpo_tidal.sh
```

默认输出：

```text
outputs/rrec_amazon/CDs_and_Vinyl/sft_top20_deepseek_one_glm52_ckpt167.jsonl
outputs/rrec_amazon/CDs_and_Vinyl/grpo_deepseek_one_glm52_ckpt167_exclude_sft_top20.jsonl
```

## SFT 全量训练

ms-swift v4 使用 `tuner_type` 控制 full / LoRA。脚本中 `TRAIN_TYPE=full`
会传给 swift 的官方参数 `--tuner_type full`。启动后日志应显示
`Qwen3ForCausalLM`，trainable 参数量接近 4B；如果出现
`PeftModelForCausalLM`，说明仍在跑 LoRA。

8 卡训练命令：

```bash
cd /mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro
conda activate swift
git pull

SFT_DATA=/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro/outputs/rrec_amazon/CDs_and_Vinyl/sft_top20_deepseek_one_glm52_ckpt167.jsonl
N=$(wc -l < "$SFT_DATA")
SAVE_STEPS=$(( (N + 8*8 - 1) / (8*8) ))
echo "SFT rows=$N, SAVE_STEPS=$SAVE_STEPS"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MASTER_PORT=29701 \
NCCL_NET=Socket \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_NVLS_ENABLE=0 \
NCCL_MNNVL_ENABLE=0 \
NCCL_COLLNET_ENABLE=0 \
NCCL_SHM_DISABLE=0 \
NCCL_ASYNC_ERROR_HANDLING=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
ROOT=/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro \
VENV=/root/miniconda3/envs/swift \
MODEL=/mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3/4B \
MODEL_TYPE=qwen3 \
TEMPLATE=qwen3 \
DATASET=$SFT_DATA \
OUT=/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_4b_sft_cds_deepseek_one_top20_full_ep10_8gpu \
TRAIN_TYPE=full \
TUNER_TYPE=full \
MAX_LENGTH=2048 \
BATCH_SIZE=8 \
GRAD_ACCUM=1 \
LEARNING_RATE=5e-5 \
NUM_TRAIN_EPOCHS=10 \
MAX_STEPS=-1 \
SAVE_STEPS=$SAVE_STEPS \
SAVE_TOTAL_LIMIT=12 \
bash scripts/run_sft_qwen3_4b.sh
```

如果 NCCL 默认路径在机器上不稳定，保留上面的 `NCCL_NET=Socket`、
`NCCL_NVLS_ENABLE=0`、`NCCL_MNNVL_ENABLE=0` 等变量。

## GRPO 训练

GRPO 数据应与 SFT 数据按 prompt 去重。前置脚本已经使用
`--exclude-prompts-from "$SFT"` 构建 GRPO。

示例命令：

```bash
cd /mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro
conda activate swift

SFT_CKPT=/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_4b_sft_cds_deepseek_one_top20_full_ep10_8gpu/<run>/checkpoint-<step>
GRPO_DATA=/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro/outputs/rrec_amazon/CDs_and_Vinyl/grpo_deepseek_one_glm52_ckpt167_exclude_sft_top20.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MASTER_PORT=29711 \
NCCL_NET=Socket \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_NVLS_ENABLE=0 \
NCCL_MNNVL_ENABLE=0 \
NCCL_COLLNET_ENABLE=0 \
ROOT=/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro \
VENV=/root/miniconda3/envs/swift \
MODEL=$SFT_CKPT \
MODEL_TYPE=qwen3 \
TEMPLATE=qwen3 \
DATASET=$GRPO_DATA \
OUT=/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_4b_grpo_cds_deepseek_one_ndcg \
TRAIN_TYPE=full \
TUNER_TYPE=full \
MAX_LENGTH=2048 \
MAX_COMPLETION_LENGTH=2048 \
BATCH_SIZE=1 \
GRAD_ACCUM=4 \
LEARNING_RATE=1e-6 \
MAX_STEPS=250 \
SAVE_STEPS=50 \
NUM_GENERATIONS=4 \
RUBRIC_GAIN_MODE=ndcg \
RUBRIC_NDCG_ITEM_INFO=$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl \
QWEN3_EMBEDDING_MODEL=$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_cot_deepseek_v4_pro_low_one_cot_global_bsz128_xgpu/checkpoint-167 \
bash scripts/run_grpo_qwen3_4b.sh
```

`configs/glm_codeplan.env` 可保存 `BIGMODEL_API_KEY`，该文件被 `.gitignore`
忽略。

## Embedding 训练

基础 history->target embedding 训练：

```bash
cd /mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro
conda activate swift

ROOT=/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro \
VENV=/root/miniconda3/envs/swift \
BASE_EMBEDDING_MODEL=/mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3_embedding/0.6B \
EMBEDDER_DATASET=$ROOT/github_artifacts/CDs_and_Vinyl/phase0/phase0_embedder_rrec_amazon_cds_and_vinyl_train.jsonl \
EMBEDDER_OUT=$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_tidal \
EMBEDDER_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EMBEDDER_NPROC_PER_NODE=8 \
EMBEDDER_BATCH_SIZE=32 \
EMBEDDER_GRAD_ACCUM=1 \
EMBEDDER_EPOCHS=3 \
EMBEDDER_LR=6e-6 \
EMBEDDER_SAVE_STEPS=83 \
bash scripts/run_train_cds_embedding_tidal.sh
```

CoT-aware embedding 数据由 `scripts/make_cot_embedder_dataset.py` 构建，再用
`scripts/run_train_cds_cot_embedding_tidal.sh` 训练。

## vLLM 批量评测所有 checkpoint

该入口扫描 `CHECKPOINT_ROOT` 下所有 `checkpoint-*`，每个 checkpoint 生成一份
`eval.json` 和 `pred.jsonl`。如果某个 checkpoint 的结果已经存在，脚本跳过它。
结果目录名包含数据集、模型目录、embedding checkpoint、split、scorer 和 vLLM
并行配置。

```bash
cd /mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro
conda activate swift
git pull

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_NET=Socket \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_NVLS_ENABLE=0 \
NCCL_MNNVL_ENABLE=0 \
NCCL_COLLNET_ENABLE=0 \
ROOT=/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro \
VENV=/root/miniconda3/envs/swift \
CHECKPOINT_ROOT=/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_4b_sft_cds_deepseek_one_top20_full_ep10_8gpu \
MODEL_KIND=sft \
QWEN3_EMBEDDING_MODEL=/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_cot_deepseek_v4_pro_low_one_cot_global_bsz128_xgpu/checkpoint-167 \
DEVICES=0,1,2,3,4,5,6,7 \
TENSOR_PARALLEL_SIZE=8 \
MAX_EXAMPLES=0 \
GENERATION_BATCH_SIZE=64 \
MAX_PROMPT_TOKENS=2048 \
MAX_NEW_TOKENS=2048 \
VLLM_MAX_MODEL_LEN=4096 \
VLLM_MAX_NUM_SEQS=64 \
SCORER=qwen3_embedding \
KS=5,10,20 \
bash scripts/run_eval_checkpoints_vllm_tidal.sh
```

直接评测单个 checkpoint 时，把 `CHECKPOINT_ROOT` 指向
`.../checkpoint-<step>`。

## 评测协议

评测读取：

```text
github_artifacts/CDs_and_Vinyl/rrec_eval/test.jsonl
github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl
```

每个样本先用 reasoner 生成 CoT，再构造：

```text
query = user_history + generated_cot
```

然后用指定 embedding checkpoint 对 `query` 和全量 item embedding 计算余弦相似度，
得到 target item 的 rank，并汇总：

```text
HR@5 / HR@10 / HR@20
NDCG@5 / NDCG@10 / NDCG@20
```

baseline 使用不加 CoT 的 `user_history` query。

## 常见检查

确认 SFT 是否全量训练：

```text
全量: Qwen3ForCausalLM，Trainable 接近 4039M
LoRA: PeftModelForCausalLM，Trainable 约 16M
```

确认 NCCL 8 卡通信：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_NET=Socket \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_NVLS_ENABLE=0 \
NCCL_MNNVL_ENABLE=0 \
NCCL_COLLNET_ENABLE=0 \
python -m torch.distributed.run \
  --nproc_per_node 8 \
  --master_port 29731 \
  /tmp/nccl_smoke.py
```

8 卡通过时应输出：

```text
NCCL OK, reduced: 36.0
```
