# RRec Amazon CDs_and_Vinyl Training Artifacts

This directory contains the JSONL files needed to train the CDs_and_Vinyl pipeline without rebuilding candidate CoT data.

Files:

- `phase0/phase0_embedder_rrec_amazon_cds_and_vinyl_train.jsonl`
  - 10,722 rows.
  - Query-positive pairs for training the Qwen3-Embedding-0.6B recommendation embedder.
- `sft/sft_rrec_amazon_cds_and_vinyl_deepseek_v4_pro_cds_embedder_partial.jsonl`
  - 1,112 rows.
  - Supervised fine-tuning examples selected by rubric-gated CoT gain.
- `grpo/grpo_rrec_amazon_cds_and_vinyl_deepseek_v4_pro_cds_embedder_disjoint_full.jsonl`
  - 9,429 rows.
  - GRPO prompts disjoint from the SFT prompts, with Qwen3 embedding baseline similarities.

The embedding checkpoint is intentionally not stored here. Train it on the target server from the phase0 JSONL before SFT/GRPO.
