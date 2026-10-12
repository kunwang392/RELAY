<div align="center">

<h1>
RELAY: Selective Self-Distillation with Adaptive Hand-off<br>
for Rubric-Based Reinforcement Learning
</h1>

<p>
  <a href="#"><img src="https://img.shields.io/badge/Paper-Under%20Review-orange.svg" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

</div>

---

## Paper

This is the official implementation of
**RELAY: Selective Self-Distillation with Adaptive Hand-off for Rubric-Based
Reinforcement Learning**. The paper is currently under anonymous review; the
citation will be added upon publication.

---

## Overview

<p align="center">
  <img src="docs/figures/relay_overview.png" width="96%">
</p>

Rubric-based reinforcement learning grades a response against many written
criteria but collapses them into one reward, which discards the detail the
rubric carries and gives no GRPO reward gradient when a group's responses tie.
A frozen teacher that reads the rubric as privileged context provides
token-level supervision, but the gains over GRPO plateau early.

RELAY makes both choices from the training signal itself:

- **Stage 1** combines selective distillation from a frozen privileged-context
  teacher with GRPO. Advantages and rewards select positive-advantage and
  low-scoring zero-advantage rollouts, covering about half of the response
  tokens.
- **Plateau detection** tracks the relative decrease of the distillation loss
  and retires the teacher once it flattens. Calibrated once on
  Qwen3-1.7B/Medicine and reused unchanged in the three other settings, the
  rule fires between steps 122 and 171.
- **Stage 2** continues with GRPO alone.

This repository contains RELAY's rollout selection, conditional top-64
forward-KL distillation, plateau detector, and hand-off to GRPO, together
with the ROLL runtime needed to execute them. It supports Qwen3-1.7B and
Qwen3-4B on medicine or science data supplied by the user, with a separate
final GPT-5.1 evaluation entry point. No plotting or paper-production code is
included.

Datasets, model weights, historical checkpoints, experiment logs, cloud
submission scripts, baseline implementations, and ablation launchers are not
included. `roll/datasets/` contains Python input-loading code, not dataset
files.

---

## Results

Main results under GPT-5.1, averaged over three evaluations. Medicine and
science columns use separately trained models; Aggregate is the unweighted
mean of the five benchmarks. Best per model size in **bold**.

| Model | Method | RaR-Med. | HealthBench | LLMEval-Med | RaR-Sci. | ResearchQA | Aggregate |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | Base | 34.87 | 21.91 | 54.11 | 47.79 | 43.54 | 40.44 |
| Qwen3-1.7B | + OPSD | 40.20 | 18.79 | 55.71 | 49.25 | 45.31 | 41.85 |
| Qwen3-1.7B | + GRPO | 42.15 | 23.87 | 56.28 | 52.79 | 51.11 | 45.24 |
| Qwen3-1.7B | + CriPO | 42.37 | 27.97 | 54.99 | 55.79 | **60.58** | 48.34 |
| Qwen3-1.7B | **+ RELAY** | **46.77** | **29.31** | **57.51** | **57.00** | 59.73 | **50.06** |
| Qwen3-4B | Base | 48.56 | 33.82 | 63.18 | 62.60 | 51.85 | 52.00 |
| Qwen3-4B | + OPSD | 56.13 | 29.17 | 66.21 | 63.29 | 52.89 | 53.54 |
| Qwen3-4B | + GRPO | 58.41 | 39.53 | 68.01 | 68.36 | 63.19 | 59.50 |
| Qwen3-4B | + CriPO | 59.67 | 41.93 | 67.77 | 66.17 | 63.95 | 59.90 |
| Qwen3-4B | **+ RELAY** | **64.43** | **42.63** | **69.14** | **69.80** | **65.79** | **62.36** |

<p align="center">
  <img src="docs/figures/fkl_traces.png" width="96%">
</p>

Per-token divergence during Stage 1 of RELAY across the four settings.
Vertical lines mark the plateau-rule trigger. The divergence scale differs
across settings, but the rule tests relative change and transfers without
re-tuning.

---

## Installation

Use Linux, Python 3.10, a CUDA 12 toolchain compatible with the selected PyTorch
build, and one node with **8 NVIDIA GPUs**. The supplied configuration uses TP=4,
DP=2, BF16, and separate frozen-teacher and reference scoring. GPU memory
requirements depend on the model size and batch settings.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-kernels.txt
python -m pip install --no-deps -e ./mcore_adapter -e .
python -m pip check
```

The installation target pins the main runtime packages, including vLLM 0.10.2.
Supporting Python utilities listed without versions are not a complete transitive
lock; save `python -m pip freeze > environment.lock.txt` after validating your GPU
setup. The package supports method reproduction; exact numerical results also
depend on the data, model weights, judge service and execution environment.

Package release references:
[vLLM 0.10.2](https://pypi.org/project/vllm/0.10.2/),
[Transformer Engine 2.6.0](https://pypi.org/project/transformer-engine/2.6.0/),
[FlashAttention 2.8.3](https://pypi.org/project/flash-attn/2.8.3/).

---

## Data

Benchmark data are not redistributed in this repository. Use your existing
prepared parquet files. This release does not download, convert, filter, copy,
or redistribute datasets. Choose the correct official training and validation
splits yourself; final test evaluation is separate.

The trainer expects these columns:

| Column | Meaning |
|---|---|
| `id` | Unique question ID, as a string |
| `messages` | List of `{role, content}` messages for the student |
| `prompt_str` | Question text given to the rubric judge |
| `tag` | `rubric_opsd_train` for training; `rubric_eval` or `rar_science_eval` for validation |
| `rubrics_str` | Numbered signed rubric text |
| `rubric_weights` | Numeric weights in the same order as the rubric text |
| `rubric_count` | Number of criteria |
| `solution` | Existing privileged-context column, if present; RELAY explicitly reads `rubrics_str` |
| `ground_truth` | String; may be empty when no reference answer is needed |

Rubric headers must match their weights, for example
`1. [POSITIVE REQUIREMENT, weight=+5] ...` and
`2. [NEGATIVE PENALTY, weight=-2] ...`.
The existing training path filters prompts by configured lengths during loading;
no such processing was run while preparing this code package.

---

## Judge endpoint

An OpenAI-compatible rubric judge is required. Configure its URL, served
model name, and credential through environment variables:

```bash
export RUBRIC_JUDGE_API_BASE=http://localhost:8000/v1
export RUBRIC_JUDGE_MODEL=your-judge-model
export RUBRIC_JUDGE_API_KEY=EMPTY  # replace privately if authentication is needed
```

The default judge response mode is `compact_regex`, matching the existing
vLLM judge interface. For a server supporting JSON-schema structured responses,
add `rewards.math_rule.rubric_judge_output_mode=json_schema` to the command.
No judge requests are made by the configuration preview or CPU tests.

---

## Training

```bash
# Inspect without data, GPU access, or Ray startup.
bash run.sh --model 1.7b --domain medicine --print-config

# Train on your external files.
bash run.sh --model 1.7b --domain medicine \
  --train-file /path/to/medicine_train.parquet \
  --validation-file /path/to/medicine_validation.parquet \
  --output-dir ./output/medicine_1p7b

# Use --model 4b and/or --domain science for the other settings.
bash run.sh --model 4b --domain science \
  --train-file /path/to/science_train.parquet \
  --validation-file /path/to/science_validation.parquet \
  --output-dir ./output/science_4b
```

`--domain` labels the run; it does not infer dataset paths or transform data.
All four settings share the method parameters. `--model-path` accepts local
weights. `TRAIN_FILE`, `VALIDATION_FILE`, and `PRETRAIN_MODEL` can alternatively
be supplied through the environment. An explicit `--model` overrides the
environment's model choice; `--model-path` takes priority over both.

Add `--check-runtime` to construct the runtime configuration and check visible
GPUs without starting training. Extra `key=value` arguments override the YAML.
For example, `save_steps=5` changes checkpoint frequency. The launcher rejects
excluded methods and teacher-refresh ablations.

The default runs 400 rollout iterations, indexed **0–399**, with 64 prompts and
8 responses each. Each rollout has two optimizer updates of 32 prompt groups.
Regular validation runs before iterations 0, 10, ..., 390. After all 400 rollout
iterations, the trainer transfers the final weights to the inference workers
and records an additional validation at **step 400**, without another training
update. This point evaluates `checkpoint-399`; the metrics include
`validation/checkpoint_step=399` and `validation/completed_steps=400`. Checkpoint
labels remain zero-based for resume compatibility. Validation and training both
allow **5120 response tokens**; validation reserves 12288 prompt tokens plus
5120 response tokens (17408 total).

Training, validation and the rubric teacher all retain the full signed rubric.
The reward is the sum of satisfied positive weights and triggered negative
weights, divided by total positive weight, then clipped to [0, 1]. Positive-only
training is rejected by the RELAY entry point.

The selection threshold is the running Q30 with EMA rate 0.1. Selected responses
receive unit distillation weights and coefficient 0.01. The plateau rule uses
W=20, relative decrease <0.05, patience 5, minimum step count 60, and latest trigger
label 199. A trigger takes effect on the following iteration. Stage 2 skips the
privileged teacher pass and retains GRPO's frozen-reference KL term.

### Outputs and resume

Outputs stay under the selected output directory: local JSONL metrics,
checkpoints, and logs. Rollout dumps are disabled by default. Checkpoints retain
the selection EMA and plateau history as part of the existing trainer state.

```bash
bash run.sh --model 1.7b --domain medicine \
  --train-file /path/to/medicine_train.parquet \
  --validation-file /path/to/medicine_validation.parquet \
  resume_from_checkpoint=/path/to/checkpoint-149
```

When resuming a historical checkpoint lacking plateau history, provide the
actual historical `[step, selected_token_fkl]` values through
`opsd_auto_switch_history_seed=/path/to/history.json`. No history is fabricated
or bundled. Reproducing a historical result also requires its matching data,
weights, judge service, and restored training state.

The bundled adapter exports Megatron checkpoints to Hugging Face format:

```bash
python mcore_adapter/tools/convert.py \
  --checkpoint_path /path/to/model_checkpoint_directory \
  --output_path ./exported_model --bf16
```

Use the model checkpoint subdirectory containing `mca_config.json`.

---

## Evaluation

Evaluate exported Hugging Face checkpoints separately for medicine and science.
The candidate uses non-thinking generation, temperature 0.7, top-p 0.8, top-k 20,
and a **5120-token limit for every response, including every LLMEval-Med turn**.
Three fresh rollout sets use distinct deterministic request seeds. The judge is
`gpt-5.1-2025-11-13` with reasoning effort `none` and strict JSON schemas, using
the shared rubric and LLMEval-Med prompts. The judge credential is read only from
the environment; `GPT_EVAL_API_BASE` optionally selects an OpenAI-compatible URL.

```bash
export GPT_EVAL_API_KEY=YOUR_PRIVATE_KEY
# Optional: export GPT_EVAL_API_BASE=https://your-compatible-endpoint/v1

python evaluate.py run --model-path ./exported_medicine --model-size 1.7b \
  --domain medicine --rar-medicine /path/to/medicine_test.parquet \
  --healthbench /path/to/healthbench_500.parquet \
  --llmeval-med /path/to/llmeval_med.parquet --output-dir ./eval/medicine_1p7b

python evaluate.py run --model-path ./exported_science --model-size 1.7b \
  --domain science --rar-science /path/to/science_test.parquet \
  --researchqa /path/to/researchqa.parquet --output-dir ./eval/science_1p7b

python evaluate.py aggregate ./eval/medicine_1p7b/summary.json \
  ./eval/science_1p7b/summary.json --output ./eval/aggregate_1p7b.json
```

Use `--model-size 4b` with the corresponding domain checkpoints for Qwen3-4B.
Candidate inference defaults to TP=4 and a 32768-token context; use
`--tensor-parallel-size` and `--max-model-len` to match your hardware and model.
Long histories fail explicitly instead of reducing the response budget or
silently truncating input. Add `--print-config` to `run` to inspect the protocol
without reading data, loading models or contacting a judge.

Provide prepared external test files: 2242 RaR-Medicine questions, 2292
RaR-Science questions, the fixed 500-question HealthBench subset, 776 ResearchQA
questions, and 571 LLMEval-Med conversation rows containing 667 turns. The code
checks counts and IDs; the user must supply the correct official splits and the
matching HealthBench subset. It does not download, resample or rewrite datasets.
Rubric benchmarks use `id`, `messages`, `prompt_str`, `rubrics_str`,
`rubric_weights`, and `rubric_count` as above. ResearchQA criteria must have unit
positive weights. LLMEval-Med uses `id`, `llmeval_category` (the original Chinese
category), and an ordered `llmeval_turns` list. Each turn has `question`,
`reference` (or `sanswer`), and optional text `checklist`. Later turns receive
earlier answers generated by the same candidate, without reference-answer leakage.

RaR and ResearchQA scores are clipped per question. HealthBench retains signed
question scores and clips only after averaging. LLMEval-Med averages its 1–5
scores over turns and multiplies by 20. Summaries report each repeat, its mean,
and population standard deviation across three repeats; Aggregate is the
unweighted mean of the five benchmark means from the two domain models.

Each output directory contains a protocol/input-hash manifest, response JSONL
files, judge-result JSONL files, and `summary.json`. Failed or incomplete judgments
produce a nonzero exit and no complete benchmark score. Retry only those judgments
without regenerating answers or replacing successful turn judgments:

```bash
python evaluate.py judge --run-dir ./eval/medicine_1p7b
```

Use a new output directory when evaluating another checkpoint or generating
another set of responses. Evaluation outputs and credentials are not bundled.

---

## Checks

```bash
# Uses synthetic tensors and temporary test state, not any dataset.
python -m pip install -r requirements-test.txt
python -m pytest
python scripts/check_package.py
```

---

## Structure

```
RELAY/
|-- roll/           ROLL runtime: selection, threshold EMA, plateau rule, training loop, rewards
|-- mcore_adapter/  required Megatron model/checkpoint adapter
|-- configs/        flattened RELAY configuration
|-- scripts/        package checks
|-- tests/          CPU tests
|-- docs/figures/   figures used in this README
|-- evaluate.py     final evaluation entry point
|-- train.py        training entry point
|-- run.sh          training launcher
`-- README.md
```

| File | Role |
|---|---|
| `configs/relay.yaml` | One flattened RELAY configuration |
| `roll/utils/grpo_opsd.py` | Selection, threshold EMA, schedule and plateau rule |
| `roll/pipeline/rlvr/rlvr_pipeline.py` | Training loop, teacher retirement and saved state |
| `roll/pipeline/rlvr/actor_worker.py` | GRPO plus conditional forward-KL loss |
| `roll/pipeline/rlvr/rewards/rubric_evaluator_reward_worker.py` | Rubric judge requests and rewards |
| `evaluate.py`, `roll/evaluation/final_eval.py` | Final generation, GPT scoring, retries and benchmark aggregation |
| `mcore_adapter/` | Required Megatron model/checkpoint adapter |
| `VALIDATION.json` | Checks actually executed for this release |

The shared ROLL runtime retains compatibility fields for optional features;
their excluded implementations are replaced with explicit rejection guards.
The default configuration writes metrics and checkpoints to the local filesystem.

The release has CPU and static validation; a full multi-GPU training run was
not performed while assembling it. See `VALIDATION.json` for exact coverage.

---

## Citation

The paper is under anonymous review. The citation will be added upon
publication.

---

## License

RELAY is released under the [Apache License 2.0](LICENSE). Upstream ROLL and
adapter attribution is retained in the source files.
