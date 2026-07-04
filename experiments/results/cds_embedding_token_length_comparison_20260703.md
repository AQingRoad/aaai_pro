# CDs_and_Vinyl Embedding Token Length Comparison

Date: 2026-07-03

## Scope

This note compares token lengths for three CDs_and_Vinyl embedding training datasets under the Qwen3-Embedding-0.6B training format.

Tokenizer:

`/home/user/models_hf/Qwen3-Embedding-0.6B`

Query counting rule:

`format_qwen3_query(query)` is applied before tokenization, so the count includes the training instruction prefix:

`Instruct: Given a user's past item interactions and optional recommendation reasoning, retrieve items the user is likely to prefer next.`

Positive counting rule:

`positive` is tokenized as raw document text, without the query instruction.

## Dataset Files

| Name | Rows | File |
|---|---:|---|
| plain base history+rating | 10721 | `outputs/rrec_amazon/CDs_and_Vinyl/embedding/phase0_embedder_cds_plain_user_history_rating_only.jsonl` |
| metadata-rich history-only | 10722 | `outputs/rrec_amazon/CDs_and_Vinyl/embedding/phase0_embedder_cds_meta_compact_no_all_ratings_history_only.jsonl` |
| metadata-rich history+CoT | 10722 | `outputs/rrec_amazon/CDs_and_Vinyl/cot/training/phase0_embedder_cds_meta_compact_no_all_ratings_history_plus_tagged_cot_manual_filled_full_target.jsonl` |

## Query Token Lengths

| Dataset | Mean | P50 | P75 | P90 | P95 | P99 | Max | >2048 | >3072 | >4096 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain base history+rating | 96.54 | 71 | 110 | 204 | 237 | 274 | 357 | 0 | 0 | 0 |
| metadata-rich history-only | 701.34 | 394.5 | 870 | 2062 | 2411 | 2819 | 3195 | 1081 | 11 | 0 |
| metadata-rich history+CoT | 878.81 | 575 | 1066 | 2267 | 2625 | 3039 | 3457 | 1254 | 85 | 0 |

## Positive Token Lengths

The three datasets use the same full target item text as `positive`, so their positive length distribution is effectively identical.

| Dataset | Mean | P50 | P75 | P90 | P95 | P99 | Max | >2048 | >3072 | >4096 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain base history+rating | 123.94 | 102 | 155 | 240 | 318 | 533 | 2427 | 1 | 0 | 0 |
| metadata-rich history-only | 123.93 | 102 | 155 | 240 | 318 | 533 | 2427 | 1 | 0 | 0 |
| metadata-rich history+CoT | 123.93 | 102 | 155 | 240 | 318 | 533 | 2427 | 1 | 0 | 0 |

## Manual CoT Fill Subset

The metadata-rich history+CoT dataset includes 35 manually filled CoT rows. For those 35 rows:

| Subset | Mean | P50 | P75 | P90 | P95 | P99 | Max | >2048 | >3072 | >4096 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| manual-fill query + instruction | 638.34 | 382 | 661 | 1550 | 2110 | 2484 | 2524 | 3 | 0 | 0 |

Manual fill audit:

`experiments/results/cds_no_all_ratings_manual_cot_fills_20260703.json`

## Conclusions

- `EMBEDDER_MAX_LENGTH=4096` does not truncate query or positive text for any of the three datasets.
- `EMBEDDER_MAX_LENGTH=3072` would truncate 11 metadata-rich history-only queries and 85 metadata-rich history+CoT queries.
- `EMBEDDER_MAX_LENGTH=2048` would truncate 1081 metadata-rich history-only queries and 1254 metadata-rich history+CoT queries.
- Adding metadata increases query length much more than adding CoT. Compared with metadata-rich history-only, the CoT version adds about 177 tokens on average and increases max query length from 3195 to 3457.
- The plain base is short because its query only contains item titles and history ratings.
