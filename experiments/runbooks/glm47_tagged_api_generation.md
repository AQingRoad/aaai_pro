# GLM-4.7 Tagged API Generation Smoke Runs

## Purpose

Use this runbook for small API generation checks such as 10-case prompt tests.
These runs inspect prompt compliance, raw API payloads, and parsed candidate
fields before launching full data construction.

## Output Location

Put smoke or case-study API outputs under:

```text
experiments/smoke/api_generation/
```

Keep formal reusable pipeline artifacts under a dataset/category-specific path:

```text
outputs/<DATASET>/<CATEGORY>/
```

Do not mix 10-case API checks with formal candidate-list outputs. Use a filename
that states the model, format, strictness, case count, and whether raw API
records are present.

Example:

```text
experiments/smoke/api_generation/cot_candidate_lists_glm47_tagged_strict_10_raw.jsonl
```

## Required API Audit Fields

For smoke runs, set:

```bash
RECORD_API_RAW=1
```

This records the request payload and raw response inside each candidate's
`generation_api_meta`:

```text
api_request_payload
api_raw_response
api_raw_content
api_raw_reasoning_content
```

The recorded request payload excludes the Authorization header and API key. It
does include model name, messages, decoding parameters, and thinking settings.

## Generic Recommended Command

```bash
MAX_EXAMPLES=10 \
MAX_WORKERS=2 \
AGGREGATE_EVERY=5 \
python3 scripts/cot/generate_cot_candidate_lists.py \
  --input data/<DATASET>/<CATEGORY>/examples.jsonl \
  --output experiments/smoke/api_generation/cot_candidate_lists_<DATASET>_<CATEGORY>_glm47_tagged_10_raw.jsonl \
  --max-examples 10 \
  --num-candidates 1 \
  --temperatures 0.6 \
  --max-workers 2 \
  --aggregate-every 5 \
  --resume \
  --api-provider glm_codeplan \
  --api-base-url https://open.bigmodel.cn/api/coding/paas/v4 \
  --api-model glm-4.7 \
  --api-thinking disabled \
  --cot-output-format tagged \
  --max-output-words 1024 \
  --record-api-raw
```

For the current RRec CDs wrapper, the equivalent smoke command is:

```bash
MAX_EXAMPLES=10 \
MAX_WORKERS=2 \
AGGREGATE_EVERY=5 \
RECORD_API_RAW=1 \
OUTPUT=experiments/smoke/api_generation/cot_candidate_lists_rrec_amazon_cds_glm47_tagged_10_raw.jsonl \
bash scripts/pipelines/run_generate_cds_glm47_low_one_cot.sh
```

Default GLM-4.7 settings in that wrapper:

```text
API_MODEL=glm-4.7
API_THINKING=disabled
COT_OUTPUT_FORMAT=tagged
MAX_OUTPUT_WORDS=1024
```

The GLM API generation path does not apply a prompt-token limit. Do not set
`MAX_PROMPT_TOKENS` for these API runs. Each candidate records prompt length
audit fields instead:

```text
generation_api_meta.api_prompt_message_count
generation_api_meta.api_prompt_chars
generation_api_meta.api_prompt_est_tokens
```

## Acceptance Checks

After the run, verify:

```text
rows == MAX_EXAMPLES
failed_candidates == 0
candidate.think is non-empty
candidate.answer is non-empty
candidate.cot contains <think> and <answer>
generation_api_meta.api_has_reasoning_content == false
generation_api_meta.api_has_content_analysis == true
generation_api_meta.api_has_content_think == true
generation_api_meta.api_has_content_answer == true
generation_api_meta.api_output_word_count <= 1024
generation_api_meta.api_request_payload exists when RECORD_API_RAW=1
generation_api_meta.api_raw_response exists when RECORD_API_RAW=1
generation_api_meta.api_content_reasoning_tag records the raw reasoning tag name
generation_api_meta.api_prompt_chars exists
generation_api_meta.api_prompt_est_tokens exists
candidate.answer is a neutral transferable item-feature profile
candidate.answer does not contain self-reference, advice/action wording, or external appraisal/stat signals
candidate.answer does not name historical titles, creators, brands, IDs, or exact item identities
```

## Generic Prompt Constraints

The tagged prompt should stay dataset-agnostic. It can use metadata fields such
as category, creator/brand/source, format/type/platform, description,
attributes/details, and rating or feedback signal.

The `<answer>` block should describe transferable item features only:

```text
high-level category
subcategory/style
format/type/platform
functional or use-case attributes
content/design/production attributes
mood/tone/audience when supported
```

Do not let `<answer>` contain non-feature text: user/history self-reference,
advice or future-action wording, external appraisal, popularity, review, award,
sales, rating, catalog-stat signals, copied entities, item IDs, or exact item
identities.

For low-rated or negative-only histories, the prompt must keep the model
conservative:

```text
Low-rated items indicate disliked item-specific features, not rejection of the whole high-level category.
Do not exclude a broad category from one low rating.
Identify only metadata-supported disliked features.
Avoid those specific features.
Do not infer a strong positive preference by taking the opposite of the disliked item.
Do not introduce unsupported category or domain jumps.
```

## Current Lesson

The prompt requests raw `<analysis>` and `<answer>` blocks because GLM-4.7
emits that reasoning tag more reliably than `<think>`. The candidate writer
stores:

```text
generation_api_meta.api_content_reasoning_tag: raw tag name, expected analysis
generation_api_meta.api_has_content_analysis: true when raw <analysis> is present
cot: normalized <think>...</think><answer>...</answer> content for downstream use
```

The default `tagged` parser normalizes raw `<analysis>` to saved `<think>`.
Set `REQUIRE_LITERAL_TAGS=1` only when testing exact raw `<analysis>/<answer>`
compliance.

In `tagged` mode, the parser requires both a reasoning block and an answer block.
A missing block raises an error and triggers retry or failure logging.

Store both split fields and the tagged field:

```text
think: tag-stripped reasoning text
answer: tag-stripped final feature description
cot: full <think>...</think><answer>...</answer> content
```
