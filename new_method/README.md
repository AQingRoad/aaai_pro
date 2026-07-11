# GDR 第一阶段：成对检索器

本目录实现 story.md 中的第一阶段，代码与现有训练链路隔离。现阶段包含：

- core.py：CoT 提取、query 构造、排名增量和标签规则；
- make_cot_perturbations.py：不读取 target 的确定性多候选构造；
- score_paired_gain.py：使用冻结的 Qwen3 embedding 成对计算 history 与 history+think；
- build_paired_dataset.py：构造 history、good CoT、bad CoT、positive 和 hard negatives；
- audit_paired_dataset.py：训练前数据门禁；
- paired_loss.py：multi-positive InfoNCE 与 good/bad 顺序损失；
- train_paired_retriever.py：单卡 A100 三视图训练入口；
- tests/：纯函数和损失测试。

## 固定口径

1. embedding query 只拼接 think 内容；
2. history 与 history+think 使用相同 target、候选集合和 seen-item mask；
3. target、positive、CoT 和 history 不允许出现 [TRUNCATED]；
4. query 超过 query_max_length、item 超过 item_max_length 时由 scorer 拒绝，不静默截断；
5. test 不参与标签阈值、checkpoint 和训练参数选择；
6. scorer checkpoint、query instruction、item-info 哈希和数据 split 写入产物元数据。

## 执行门禁

当前只提交代码。数据打分、数据生成、评测和训练命令需要先打印完整参数并获得用户确认。
