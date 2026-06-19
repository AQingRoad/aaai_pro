# RRec Amazon CDs_and_Vinyl Training Artifacts

This directory contains the JSONL files needed to train the CDs_and_Vinyl pipeline without rebuilding candidate CoT data.

Files:

- `phase0/phase0_embedder_rrec_amazon_cds_and_vinyl_train.jsonl`
  - 10,722 rows.
  - Query-positive pairs for training the Qwen3-Embedding-0.6B recommendation embedder.
- `cot/cot_candidate_lists_deepseek_v4_pro_low.jsonl`
  - 2,187 CoT candidate-list rows, 8,732 candidate CoTs.
- `cot/cot_candidate_lists_deepseek_v4_pro_low.rubric_deepseek_v4_pro.jsonl`
  - 8,732 candidate-level rubric scores for the CoT candidate lists.
- `sft/sft_rrec_amazon_cds_and_vinyl_deepseek_v4_pro_cds_embedder_partial.jsonl`
  - 1,112 rows.
  - Supervised fine-tuning examples selected by rubric-gated CoT gain.
- `grpo/grpo_rrec_amazon_cds_and_vinyl_deepseek_v4_pro_cds_embedder_disjoint_full.jsonl`
  - 9,429 rows.
  - GRPO prompts disjoint from the SFT prompts, with Qwen3 embedding baseline similarities.
- `rrec_eval/valid.jsonl`
  - 1,340 RRec-style validation interactions.
- `rrec_eval/test.jsonl`
  - 1,341 RRec-style test interactions.
- `rrec_eval/item_info.jsonl`
  - 12,001 full-candidate item records, including the pad item.
- `rrec_eval/MANIFEST.json`
  - Generation parameters and split counts for the validation/test artifacts.

The embedding checkpoint is intentionally not stored here. Train it on the target server from the phase0 JSONL before SFT/GRPO.

The `rrec_eval` files were generated from Amazon Reviews 2023 CDs_and_Vinyl raw ratings and metadata with the same RRec preprocessing semantics used for the train split: `K=0`, date window `2022-10` to `2023-10` with the official item-count window extension, and `window_size=20`. The generated train count is 10,722, matching the phase0 training rows.

To rebuild the SFT/GRPO JSONL files from the committed CoT candidate and rubric files on Tidal, run:

```bash
bash scripts/prepare_cds_from_cot_tidal.sh
```
