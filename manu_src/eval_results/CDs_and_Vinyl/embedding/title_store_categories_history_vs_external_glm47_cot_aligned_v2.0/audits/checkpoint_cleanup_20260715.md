# Checkpoint 清理记录

2026-07-15 在 A100 清理 39 个后续无需直接复用的权重目录：36 个已有测试 JSON 的非保留 embedding checkpoint、1 个 smoke checkpoint、2 个已有 LoRA adapter 的 3B merged 模型。

清理保留以下内容：

- 每个旧 embedding 对照组的登记代表 checkpoint。
- 全部评测 JSON、训练参数和实验报告。
- dense-rank GRPO 的 LoRA `checkpoint-300` 与 `checkpoint-600`，两个 merged 模型可由这两个 adapter 重新生成。
- 当前 aligned v2.0 clean run 的 history `checkpoint-249` 和正在生成的 external GLM CoT 数据。
- 尚未评测、尚未归档或仍可能继续训练的实验目录。

A100 checkpoint 根目录从约 `114G` 降至约 `61G`，`du -sk` 汇总释放 `55,716,332 KiB`，约 `53.1 GiB`。完整路径级清理原则与保留表见 `experiments/results/cds_checkpoint_cleanup_20260715.md`。
