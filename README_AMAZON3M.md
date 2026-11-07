# 🌱 PLANT on Amazon-3M

This is the Amazon-3M companion to [README.md](README.md). The PLANT paradigm,
architecture, and 5-stage pipeline are all unchanged — only the dataset
changes. Read [README.md](README.md) first for the concepts; this file only
documents what's different when `--source_url="XURLs.AMAZON_3M"` instead of
`XURLs.MIMIC4_DEMO`.

> 💻 **Where this runs**: this document was assembled and dry-run-verified
> (CPU-only, Stage 3's data-check) on the primary dev machine, which has no
> GPU. Actual training (Stages 1, 3, 4, 5) is meant to run on a separate
> cloud GPU box. The box just clones this repository.

## 📋 Table of Contents

- [📊 Dataset: Amazon-3M](#-dataset-amazon-3m)
- [📝 Naming Conventions for Amazon-3M](#-naming-conventions-for-amazon-3m)
- [⚠️ The 2.8M-label problem (read before running anything)](#️-the-28m-label-problem-read-before-a-full-scale-run)
- [🧭 Which command track do I need?](#-which-command-track-do-i-need)
- [🚀 PLANT Pipeline](#-plant-pipeline)
  - [Stage 1: SOTA LLM Fine-tuning](#stage-1-sota-llm-fine-tuning-)
  - [Stage 2: Task-Specific Head Attachment](#stage-2-task-specific-head-attachment-)
  - [Stage 3: L2R Pretraining with MIG](#stage-3-l2r-pretraining-with-mig-)
  - [Stage 4: L2R Attention Transfer](#stage-4-l2r-attention-transfer-)
  - [Stage 5: PLANT Fine-tuning](#stage-5-plant-fine-tuning-)
- [🧪 Smoke-testing all 5 stages on a sample first (dense / non-extreme track)](#-smoke-testing-all-5-stages-on-a-sample-first)
  - [The commands](#the-commands)
- [🏗️ Full-scale candidate-generator subsystem (extreme-label-space track)](#️-full-scale-candidate-generator-subsystem-plant-full-scale-branch-in-progress)
  - [Commands: the full candidate-generator pipeline](#commands-the-full-candidate-generator-pipeline-validated-on-the-sample)
- [🔧 What had to change to port this from MIMIC-IV](#-what-had-to-change-to-port-this-from-mimic-iv)
- [📐 Precision@k / NDCG@k for Amazon-3M's label density](#-precisionk--ndcgk-for-amazon-3ms-label-density)

## 📊 **Dataset: Amazon-3M**

[Amazon-3M](https://xcubebucket.s3.us-east-2.amazonaws.com/Amazon-3M/amazon-3m.tar.gz)
(`XURLs.AMAZON_3M` in `xcube/data/external.py`) is an extreme multi-label
product classification dataset: product title + description as `text`,
`;`-delimited related-product ids as `labels`.

### 📥 Download Dataset

```python
# 🐍 Python REPL commands
from xcube.imports import *
p = untar_xxx(XURLs.AMAZON_3M)
print(f"Dataset location: {p}")
# Available files: amazon-3m.csv  (columns: text, labels, is_valid)
```

### ⚠️ One-time dataset prep: the `_full` symlink

Every stage's `--data` argument is used two ways by the pipeline code:

1. As a literal filename — `get_data()` in `finetune_llm.py` /
   `multilabel_classify.py` / `learning2rank.py` reads
   `{source}/{data}.csv` directly.
2. As a naming-convention string — `bootl2r.py`'s Stage 3 derives the
   prefix for all the MIG intelligence files by **stripping the trailing
   `_full` suffix** from `--data` (`'_'.join(data.split('_')[:-1])`), the
   same way `mimic4_icd10_full` → `mimic4_icd10`.

MIMIC's raw CSV is literally named `mimic4_icd10_full.csv`, so both uses
line up for free. Amazon's raw CSV is just `amazon-3m.csv` — no `_full`
suffix — so we symlink one into existence once per box:

```bash
python3 - <<'PY'
from xcube.imports import *
p = untar_xxx(XURLs.AMAZON_3M)
full = p / "amazon-3m_full.csv"
if not full.exists():
    full.symlink_to("amazon-3m.csv")
print(f"Ready: {full}")
PY
```

After this, `--data="amazon-3m_full"` resolves to the real CSV **and**
Stage 3 derives the correct `amazon-3m` prefix for its output files. This
step has already been done on the primary dev machine; **repeat it once on
the GPU box** after you pull the code there (the dataset itself is
downloaded separately via `untar_xxx`, it isn't part of the git sync).

### 🔍 Optional: Tokenization Verification

<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
python check_tokens.py \
    --source_url="XURLs.AMAZON_3M" \
    --data="amazon-3m_full" \
    --base_dir="../tmp" \
    --output_dir="amazon-3m_tokenization" \
    --label_delim=";"
```
</details>

This was run on the dev machine as a smoke test of this port (CPU-only,
~45s: it loads the full CSV but only tokenizes+checks a random 10-row
sample) — label consistency check **passed**. Note `check_tokens.py`'s
comparison tokenizers (`Bio_ClinicalBERT`, `PubMedBERT`, `bert-base`) are
hardcoded and biomedical — that part of the report isn't meaningful for
product text, but the underlying chunking/label-split logic it exercises
is dataset-agnostic and is what actually got verified here.

## 📝 **Naming Conventions for Amazon-3M**

Same rules as [README.md](README.md#-naming-conventions--consistency):

```
hub_model_id: {username}/{model_name}-{dataset}-{stage}
output_dir: {dataset}_{stage}_{model_shortname}
```

```bash
# Setup names for Mistral-7B on Amazon-3M
source setup_plant_names.sh "amazon-3m_full" "mistralai/Mistral-7B-Instruct-v0.3"
```

This prints (and exports) exactly:

| Stage | Hub ID | Output Dir |
| --- | --- | --- |
| 1 | `deb101/mistral-7b-amazon-3m-adapt` | `mistral7b_amazon-3m` |
| 2 | *(same as Stage 1)* | `amazon-3m_classify_mistral7b` |
| 3 | `mistralai/Mistral-7B-Instruct-v0.3` (tokenizer only) | `amazon-3m_l2rboot_mistral7b` |
| 4 | *(same as Stage 1)* | `amazon-3m_l2rtrain_mistral7b` |
| 4 (L2R output) | `deb101/mistral-7b-amazon-3m-adapt-l2r` | — |
| 5 | *(same as Stage 4 L2R output)* | `amazon-3m_plantclassify_mistral7b` |

`DATASET_NAME="amazon-3m_full"` is exported for use as `--data` in every
stage below. See [What had to change](#-what-had-to-change-to-port-this-from-mimic-iv)
for why `setup_plant_names.sh` needed a one-line fix to produce this
correctly.

---

## ⚠️ **The 2.8M-label problem (read before a full-scale run)**

Before building a sample, it's worth knowing what "extreme" means for this
particular dataset, because it changes how you should scope any run —
sample or full.

Measured directly on the downloaded `amazon-3m.csv` (2,460,406 rows):

| Quantity | Value |
| --- | --- |
| Total unique labels across the whole corpus | **2,812,281** |
| Labels appearing in ≥ 50 docs ("not rare" per the repo's threshold) | 400,974 |
| Labels appearing in ≥ 100 docs | 180,771 |
| Coverage of all label-instances by the top 50,000 most frequent labels | 27.1% |
| Median labels per document | 24 (capped at 100) |

Two consequences for this codebase specifically:

1. **`multilabel_classify.py`'s `preprocess_data()` calls
   `MultiLabelBinarizer().fit_transform(labels)` with no `classes=`
   restriction and no `sparse_output=True`.** Run that against the full
   CSV as-is and it tries to materialize a **dense** `(2.46M × 2.81M)`
   boolean array — many terabytes — before any GPU code even runs. It
   will not get that far in practice; it'll exhaust memory first.
2. **The custom classifier head (`MultilabelICDClassifier` /
   `MultilabelICDPlantClassifier`) allocates
   `nn.Parameter(torch.randn(num_labels, hidden_size))`** — i.e. one
   learned embedding per label. At `num_labels ≈ 2.8M` and
   `hidden_size=4096` (Mistral-7B), that's ~11.5B parameters for label
   embeddings alone, before the backbone.

This is architecturally different from MIMIC-IV, whose ~8–17K ICD codes
are small enough for both of the above to just work. Amazon-3M's label
space, as delivered, is not — this isn't a bug to port around, it's the
actual "extreme" in extreme multi-label classification, and this
particular classifier design (dense per-label embeddings + a plain
`fit_transform`) assumes a bounded label vocabulary.

**This needs a decision from you before a full-scale run**, not just a
sample: some form of label filtering (e.g. keep only the top-N most
frequent labels and drop the rest from each document's label list,
dropping documents left with none) is necessary for Stage 2/4/5 to even
initialize, let alone train. How aggressively to filter — and whether
that changes the task in a way that still matches what you want to show —
is a modeling decision, not something I picked unilaterally. The sample
below applies a simple top-150 filter purely to prove the *code paths*
work; it is not a proposal for the full-scale label vocabulary.

## 🧭 **Which command track do I need?**

This file (and the base [README.md](README.md)) both document two different command "tracks" for Stages 2, 4, and 5. Pick one **before** running anything, based on a single question:

> **Does your label space blow up a dense `MultiLabelBinarizer` or a `nn.Parameter(num_labels, hidden_size)` per-label embedding table?** See [the 2.8M-label problem](#️-the-28m-label-problem-read-before-a-full-scale-run) above for exactly what "blow up" means and the numbers behind it.

| | **No** — bounded label space (e.g. MIMIC-IV's ~8–17K ICD codes) | **Yes** — extreme label space (Amazon-3M's ~2.81M labels) |
| --- | --- | --- |
| What breaks otherwise | Nothing — dense Stage 2/4/5 just work | Stage 2/4/5 try to materialize a `(num_docs × num_labels)` dense array and/or a `(num_labels, hidden_size)` embedding table before any GPU code runs — it won't OOM gracefully, it'll exhaust memory before training even starts |
| Which section to run | [🧪 Smoke-testing all 5 stages on a sample first](#-smoke-testing-all-5-stages-on-a-sample-first) → ["The commands"](#the-commands) — then the same commands against the full dataset | [🏗️ Full-scale candidate-generator subsystem](#️-full-scale-candidate-generator-subsystem-plant-full-scale-branch-in-progress) → ["Commands: the full candidate-generator pipeline"](#commands-the-full-candidate-generator-pipeline-validated-on-the-sample) |
| Flags | none (`--label_space` defaults to `label`) | `--label_space=candidate --candidate_budget=... --label_embed_rank=...` on Stage 2/4/5, plus a one-time `build_label_clusters.py` run |
| **The PLANT-vs-baseline head-to-head** | **Stage 2** (`--task=multilabel-classify`, *"Baseline Comparison Only"*) **vs. Stage 5** (`--task=multilabel-plantclassify`) — same label space, same architecture; the only difference is Stage 5 initializes from Stage 4's L2R-attention-transferred checkpoint instead of Stage 1's plain adapter | **Stage 2 with `--label_space=candidate`** **vs. Stage 5 with `--label_space=candidate`** (matching `--candidate_budget`/`--label_embed_rank` on both) — same isolation logic, just inside the candidate-restricted architecture |

Amazon-3M itself is the "Yes" case (2.81M labels), so every command in ["Commands: the full candidate-generator pipeline"](#commands-the-full-candidate-generator-pipeline-validated-on-the-sample) already uses `--label_space=candidate` throughout. If you're adapting this repo for a different, non-extreme dataset, use the plain [Stage 1→5 commands](#the-commands) instead — the candidate-generator machinery (label clustering, candidate budgets, factorized embeddings) buys you nothing there and is strictly more moving parts to get wrong.

**A subtlety inside the extreme-label-space track**: the *router track* (Stage 2/4 with `--label_space=cluster`, plus `build_label_clusters.py` and `aggregate_cluster_mig.py`) is **not** part of the PLANT-vs-baseline comparison — it's a separate diagnostic answering "is the label clustering itself good enough" (via `router_coverage_at_k`/`psp_at_k`), useful if you want to sanity-check the clustering before trusting it to generate the fine track's candidate sets. Skip it entirely if you only care about the head-to-head comparison; go straight to fine-track Stage 2-candidate and Stage 5-candidate.

**Regardless of track**: always run the smoke test on a sample first (see the next section for the dense track, or the sample-scale commands inside the candidate-generator section for the extreme track) before pointing any of this at a real dataset.

## 🚀 **PLANT Pipeline**

> 📍 **Read [Which command track do I need?](#-which-command-track-do-i-need) first.** The commands below show the conceptual shape of each stage against the *full* dataset (`--data="$DATASET_NAME"`, no `--label_space` flag). For Amazon-3M specifically — an extreme-label-space dataset — don't run Stage 2/4/5 exactly as shown here; use the [candidate-generator pipeline](#️-full-scale-candidate-generator-subsystem-plant-full-scale-branch-in-progress) instead. And regardless of dataset, always [smoke-test on a sample](#-smoke-testing-all-5-stages-on-a-sample-first) before a real run.

All commands below assume you've run the `setup_plant_names.sh` command
above in the same shell (or `scripts/`, since `launches/*` are invoked
relative to `scripts/` per README.md).

### **Stage 1: SOTA LLM Fine-tuning** 🤖

#### 🆕 Fresh Training
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launchLLM \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --max_length=512 \
    --base_dir="$BASE_DIR" \
    --output_dir="$STAGE1_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" \
    --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training
```
</details>

#### ⏭️ Resume from Checkpoint
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launchLLM \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --max_length=512 \
    --base_dir="$BASE_DIR" \
    --output_dir="$STAGE1_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" \
    --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training \
    --load_from_checkpoint
```
</details>

#### 📊 Evaluation Only
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launchLLM \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --max_length=512 \
    --base_dir="$BASE_DIR" \
    --output_dir="$STAGE1_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" \
    --model_name="$BASE_MODEL_NAME"
```
</details>

---

### **Stage 2: Task-Specific Head Attachment** 🏷️
*(Baseline Comparison Only)*

#### 🆕 Fresh Training
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launchmultilabelclas \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --max_length=512 \
    --base_dir="$BASE_DIR" \
    --output_dir="$STAGE2_OUTPUT_DIR" \
    --hub_model_id="$STAGE2_HUB_ID" \
    --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-classify" \
    --do_fresh_training \
    --num_train_epochs=5 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 \
    --final_lr_scheduling=1e-6 \
    --warmup_steps=500 \
    --metric_for_best_model="precision_at_15"
```
</details>

#### ⏭️ Resume Training
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launchmultilabelclas \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --max_length=512 \
    --base_dir="$BASE_DIR" \
    --output_dir="$STAGE2_OUTPUT_DIR" \
    --hub_model_id="$STAGE2_HUB_ID" \
    --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-classify" \
    --do_fresh_training \
    --num_train_epochs=7 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 \
    --final_lr_scheduling=1e-6 \
    --warmup_steps=50 \
    --metric_for_best_model="precision_at_15" \
    --load_from_checkpoint
```
</details>

#### 📊 Evaluation Only
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launchmultilabelclas \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --max_length=512 \
    --base_dir="$BASE_DIR" \
    --output_dir="$STAGE2_OUTPUT_DIR" \
    --hub_model_id="$STAGE2_HUB_ID" \
    --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-classify" \
    --per_device_eval_batch_size=8
```
</details>

---

### **Stage 3: L2R Pretraining with MIG** 🔬

#### 🧠 MIG Bootstrap Training
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launch_l2rboot \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --base_dir="$BASE_DIR" \
    --output_dir="$STAGE3_OUTPUT_DIR" \
    --hub_model_id="$STAGE3_HUB_ID" \
    --bs=8 \
    --chnk_sz=200 \
    --max_length=4096
```
</details>

> 🔑 **Critical Note**: same as MIMIC — `--hub_model_id` specifies the
> tokenizer model, which **must match** `--model_name` in Stage 4.

> 📏 **Why `--max_length=4096` and not MIMIC's `11776`?** Stage 3 has to
> see the **whole** document (MIG needs full-document token↔label
> co-occurrence, unlike Stages 1/2/4/5 which truncate to 512). MIMIC-IV
> discharge notes are very long, hence 11776. Amazon-3M's `title /SEP/
> description` text is much shorter — measured on the actual downloaded
> `amazon-3m.csv` with the Mistral-7B tokenizer (50K-row sample):
>
> | Percentile | Token count |
> | --- | --- |
> | p50 | 214 |
> | p90 | 498 |
> | p99 | 1,363 |
> | p99.9 | 2,998 |
> | max (sample) | 17,130 |
>
> 4096 covers ~99.9% of documents in full. If you want to double-check
> coverage on the full 2.46M-row dataset before a long Stage 3 run, use
> the verification step below and inspect chunk counts — bump
> `--max_length` up if you see meaningful truncation.

#### 📊 **Generated MIG Intelligence Files**

Written to `$STAGE3_OUTPUT_DIR`, prefixed with `amazon-3m` (the
`_full`-stripped form of `$DATASET_NAME`, exactly like `mimic4_icd10` for
MIMIC):

| Component | File | Description |
| --- | --- | --- |
| 🎯 **Core Data** | `amazon-3m_p_L.pkl` | Label probability distributions |
| 🧠 **Intelligence** | `amazon-3m_tok_lbl_info.pkl` | Token-label mutual information matrix |
| 🔤 **Mappings** | `amazon-3m_tok.ft` | Token ID to string mappings |
| 🏷️ **Labels** | `amazon-3m_lbl.ft` | Label ID to string mappings |
| 📈 **Rankings** | `amazon-3m_tok_rank_per_lbl.ft` | Token importance per label |
| 📊 **Inverse Rankings** | `amazon-3m_lbl_rank_per_tok.ft` | Label relevance per token |

#### 🔍 **Verify MIG Quality**
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launch_l2r_datacheck \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --base_dir="$BASE_DIR" \
    --output_dir="$STAGE3_OUTPUT_DIR"
```
</details>

**Generated Intelligence Reports:**
- 📋 `icd10_codes_to_tokens_*.md` / `tokens_to_icd10_codes_*.md` — the
  report file names are hardcoded with an `icd10` fragment inside
  `l2r_dataload.py` regardless of dataset (cosmetic only — for Amazon-3M
  read "label" wherever it says "ICD-10 code"). Same applies to log
  messages and class names like `MultilabelICDClassifier` in
  `multilabel_classify.py` / `learning2rank.py` — purely cosmetic,
  functionally dataset-agnostic.

---

### **Stage 4: L2R Attention Transfer** 🧠

#### 🆕 Fresh Training
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launch_l2r_training \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --data_l2r_fname_prefix="amazon-3m" \
    --max_length=512 \
    --base_dir="$BASE_DIR" \
    --l2r_boot_dir="$STAGE3_OUTPUT_DIR" \
    --output_dir="$STAGE4_OUTPUT_DIR" \
    --hub_model_id="$STAGE4_HUB_ID" \
    --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training \
    --num_train_epochs=5 \
    --learning_rate=1e-4 \
    --warmup_steps=0 \
    --metric_for_best_model="ndcg@25"
```
</details>

#### ⏭️ Resume from Checkpoint
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launch_l2r_training \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --data_l2r_fname_prefix="amazon-3m" \
    --max_length=512 \
    --base_dir="$BASE_DIR" \
    --l2r_boot_dir="$STAGE3_OUTPUT_DIR" \
    --output_dir="$STAGE4_OUTPUT_DIR" \
    --hub_model_id="$STAGE4_HUB_ID" \
    --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training \
    --num_train_epochs=2 \
    --learning_rate=1e-4 \
    --warmup_steps=0 \
    --metric_for_best_model="ndcg@25" \
    --load_from_checkpoint
```
</details>

#### 📊 Evaluation Only
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launch_l2r_training \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --data_l2r_fname_prefix="amazon-3m" \
    --max_length=512 \
    --base_dir="$BASE_DIR" \
    --l2r_boot_dir="$STAGE3_OUTPUT_DIR" \
    --output_dir="$STAGE4_OUTPUT_DIR" \
    --hub_model_id="$STAGE4_HUB_ID" \
    --model_name="$BASE_MODEL_NAME"
```
</details>

> 📊 **Detailed Reports**: add `--generate_report` for
> `tokenrank_report_*.md` / `.json`.

> 🎯 **Critical Output**: the trained L2R model is saved as
> `$STAGE4_HUB_ID_L2R` (`deb101/mistral-7b-amazon-3m-adapt-l2r`) — this
> becomes the input for Stage 5.

---

### **Stage 5: PLANT Fine-tuning** 🌱

#### 🆕 Fresh Training
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launchmultilabelclas \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --max_length=512 \
    --base_dir="$BASE_DIR" \
    --output_dir="$STAGE5_OUTPUT_DIR" \
    --hub_model_id="$STAGE5_HUB_ID" \
    --task="multilabel-plantclassify" \
    --do_fresh_training \
    --num_train_epochs=5 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 \
    --final_lr_scheduling=1e-6 \
    --warmup_steps=50 \
    --metric_for_best_model="precision_at_15"
```
</details>

#### ⏭️ Resume from Checkpoint
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launchmultilabelclas \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --max_length=512 \
    --base_dir="$BASE_DIR" \
    --output_dir="$STAGE5_OUTPUT_DIR" \
    --hub_model_id="$STAGE5_HUB_ID" \
    --task="multilabel-plantclassify" \
    --do_fresh_training \
    --num_train_epochs=7 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 \
    --final_lr_scheduling=1e-6 \
    --warmup_steps=50 \
    --metric_for_best_model="precision_at_15" \
    --load_from_checkpoint
```
</details>

#### 📊 Evaluation Only
<details>
  <summary>💻 <strong>Click to expand command</strong></summary>

```bash
launches/launchmultilabelclas \
    --source_url="XURLs.AMAZON_3M" \
    --data="$DATASET_NAME" \
    --max_length=512 \
    --base_dir="$BASE_DIR" \
    --output_dir="$STAGE5_OUTPUT_DIR" \
    --hub_model_id="$STAGE5_HUB_ID" \
    --task="multilabel-plantclassify" \
    --per_device_eval_batch_size=8
```
</details>

### 🔄 Stage Flow Summary

Identical to MIMIC-IV — see
[README.md](README.md#-stage-flow-summary).

---

## 🧪 **Smoke-testing all 5 stages on a sample first**

> 📍 **This is the dense / non-extreme track** — plain `--label_space=label` (the default), no clustering or candidate budgets. Per [Which command track do I need?](#-which-command-track-do-i-need): if your dataset's label space is bounded (like MIMIC-IV's), ["The commands"](#the-commands) below is also what you run for the real (non-sample) pipeline, and it already gives you the Stage 2 (baseline) vs. Stage 5 (PLANT) comparison directly. If your label space is extreme (like Amazon-3M's — see [the 2.8M-label problem](#️-the-28m-label-problem-read-before-a-full-scale-run)), skip ahead to the [candidate-generator pipeline](#️-full-scale-candidate-generator-subsystem-plant-full-scale-branch-in-progress) instead.

### Why, and what a "sample" has to satisfy here

A sample for this pipeline isn't just "fewer rows" — it
has to bound **three** things at once, because they each drive a
different cost/failure mode:

1. **Row count** — drives per-step training time (Stages 1, 2, 4, 5).
2. **Unique label count** — drives the classifier head size (Stages 2, 4,
   5) *and* the number of full-dataset passes Stage 3's MIG bootstrap
   makes (`ceil(num_unique_labels / chnk_sz)`, `chnk_sz=200` by default —
   stay under 200 total labels and Stage 3 does a single pass).
3. **A hidden cross-stage dependency**: `bootl2r.py`'s Stage 3
   (`get_info()` → `load_mlb_classes()`) reads
   `{base_dir}/labels_partition.json`, which is only ever *written* by
   `multilabel_classify.py`'s `preprocess_data()` — i.e. by Stage 2 (or
   5). This isn't mentioned in README.md's pipeline diagram at all,
   but it's a hard runtime dependency: **Stage 3 cannot run before Stage
   2 has run at least once against the same `--base_dir`.** Running the
   stages in numeric order (1→2→3→4→5) satisfies it for free.

### Building the sample

Use `scripts/create_amazon3m_sample.py` — committed alongside the rest of
the pipeline (not a scratch one-off), so it travels with the repo and
reproduces the same sample deterministically
(fixed `--seed`, default 42) on any box:

```bash
cd ~/plant/scripts
python3 create_amazon3m_sample.py
# defaults: --source_url=XURLs.AMAZON_3M --csv_name=amazon-3m
#           --output_name=amazon-3m-sample_full --top_k_labels=150
#           --target_rows=600 --valid_frac=0.10 --seed=42
```

It does exactly this:
1. Counts global label frequency across the full `amazon-3m.csv` (one pass).
2. Keeps the top `--top_k_labels` (150) most frequent labels — comfortably
   under `chnk_sz=200`, so Stage 3 does exactly one label-chunk pass.
3. Scans the CSV, filters each doc's label list down to that top-150 set,
   drops docs left with zero labels, collects a pool of qualifying rows
   (`--pool_multiplier × --target_rows`), then randomly samples
   `--target_rows` (600) of them.
4. Re-derives `is_valid` as a clean split (`--valid_frac`, default 90/10 →
   540 train / 60 eval) rather than trusting the original column, since
   heavy label-filtering can skew it.

It downloads/locates the source dataset itself via `untar_xxx(XURLs.AMAZON_3M)`
and writes the sample into that same directory
(`~/.xcube/data/amazon-3m/amazon-3m-sample_full.csv` — same `_full`
naming convention as the full dataset; see the symlink section above), so
there's nothing to configure path-wise on a fresh box beyond having run
the dataset download step.

Result:

| | Full dataset | Sample |
| --- | --- | --- |
| Rows | 2,460,406 | 600 (540 train / 60 eval) |
| Unique labels | 2,812,281 | 95 |
| Median labels/doc | 24 | 1 |
| File | `amazon-3m.csv` | `amazon-3m-sample_full.csv` |

The median-labels/doc collapsing from 24 to 1 is expected and fine for a
*smoke test*: filtering to only the head-150 labels naturally leaves most
sampled docs with just one surviving label, since Amazon-3M's labels are
mostly near-unique "related product" ids with little overlap outside a
small frequent core. This sample proves the pipeline *runs* on real data
shaped correctly; it says nothing about model quality.

### Isolating sample runs from the real run

All sample commands below use **`--base_dir="../tmp_sample"`** instead of
`../tmp`. Since every stage derives its actual working directory as
`{base_dir}/{source_url's-second-segment}` (i.e. `.../AMAZON_3M/...`),
this keeps every artifact the sample touches — including the shared
`labels_partition.json` — in a completely separate tree from a real
`../tmp/AMAZON_3M/...` run. Nothing here can collide with or get
overwritten by the real run later.

### Avoiding Hub pollution and slow uploads

`push_to_hub=True` was hardcoded in all three training scripts'
`TrainingArguments`. For `multilabel_classify.py` specifically, the
active save path (`trainer.save_model()` on a plain `nn.Module`, since
`MultilabelICDClassifier` doesn't subclass `PreTrainedModel`) serializes
the **full** state dict — frozen 7B backbone included — before a
(correctly) size-filtered copy is written separately for the
already-dead `save_and_push_old` path. In other words: a real push here
uploads a multi-GB checkpoint, which could dominate the time budget on
its own regardless of sample size. `--skip_push_to_hub` (new flag, all
three scripts, default `False` so nothing changes unless you pass it) is
the fix — set it for every sample-run command below. Stage 1's own model
*is* just a small LoRA adapter (cheap to push either way), but the flag
is there for consistency.

### Chaining stages locally without the Hub

Stage 5 has no `--model_name` fallback — its `--hub_model_id` **must**
resolve to Stage 4's actual L2R output (that's the whole point of
attention transfer), and Stage 4/2 similarly want Stage 1's adapted
checkpoint as their backbone. Since `--skip_push_to_hub` means nothing
reaches the Hub, point `--hub_model_id` at the **local output directory**
of the previous stage instead of a Hub id string — `from_pretrained()` /
`PeftConfig.from_pretrained()` accept local paths natively, and nothing
in these particular load functions makes a Hub-only API call that would
reject one (verified by reading `load_base_model_and_tokenizer()` — it's
pure `from_pretrained()` calls throughout).

### The commands

```bash
cd ~/plant/scripts
source setup_plant_names.sh "amazon-3m-sample_full" "mistralai/Mistral-7B-Instruct-v0.3" "../tmp_sample"
# -> DATASET_NAME=amazon-3m-sample_full, BASE_DIR=../tmp_sample
# -> STAGE1_HUB_ID=deb101/mistral-7b-amazon-3m-sample-adapt, STAGE1_OUTPUT_DIR=mistral7b_amazon-3m-sample
# -> STAGE2_OUTPUT_DIR=amazon-3m-sample_classify_mistral7b
# -> STAGE3_OUTPUT_DIR=amazon-3m-sample_l2rboot_mistral7b
# -> STAGE4_OUTPUT_DIR=amazon-3m-sample_l2rtrain_mistral7b, STAGE4_HUB_ID_L2R=deb101/mistral-7b-amazon-3m-sample-adapt-l2r
# -> STAGE5_OUTPUT_DIR=amazon-3m-sample_plantclassify_mistral7b

STAGE1_LOCAL="$BASE_DIR/AMAZON_3M/$STAGE1_OUTPUT_DIR"
STAGE4_LOCAL="$BASE_DIR/AMAZON_3M/$STAGE4_OUTPUT_DIR"

# --- Stage 1 -------------------------------------------------------------
launches/launchLLM \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE1_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=1 \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 --skip_push_to_hub

# --- Stage 2 (also writes labels_partition.json, needed by Stage 3) ------
launches/launchmultilabelclas \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE2_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-classify" --do_fresh_training \
    --num_train_epochs=1 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_25" --rare_threshold=5 \
    --label_coverage_frac=1.0 --skip_push_to_hub

# --- Stage 3 (needs labels_partition.json from Stage 2 above) ------------
launches/launch_l2rboot \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" \
    --base_dir="$BASE_DIR" --output_dir="$STAGE3_OUTPUT_DIR" \
    --hub_model_id="$STAGE3_HUB_ID" --bs=8 --chnk_sz=200 --max_length=1024

# --- Stage 4 ---------------------------------------------------------------
launches/launch_l2r_training \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" \
    --data_l2r_fname_prefix="amazon-3m-sample" --max_length=512 \
    --base_dir="$BASE_DIR" --l2r_boot_dir="$STAGE3_OUTPUT_DIR" \
    --output_dir="$STAGE4_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=1 --learning_rate=1e-4 --warmup_steps=0 \
    --metric_for_best_model="ndcg@25" \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 --skip_push_to_hub

# --- Stage 5 (hub_model_id = Stage 4's local L2R output) ------------------
launches/launchmultilabelclas \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE5_OUTPUT_DIR" \
    --hub_model_id="$STAGE4_LOCAL" \
    --task="multilabel-plantclassify" --do_fresh_training \
    --num_train_epochs=1 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_25" --rare_threshold=5 \
    --label_coverage_frac=1.0 --skip_push_to_hub
```

### Time budget on an H100 80GB SXM5

Not executed end-to-end here (no GPU on this dev machine) — this is a
reasoned estimate to sanity-check against, not a measurement:

| Stage | Steps (540 train / batch 8, 1 epoch) | Est. wall time |
| --- | --- | --- |
| 1 (LoRA finetune) | ~68 | 1–3 min |
| 2 (frozen backbone + small head) | ~68 | 1–3 min |
| 3 (MIG bootstrap, 95 labels ⇒ 1 chunk pass) | n/a (batched stats, not SGD) | <2 min |
| 4 (L2R ranking head) | ~68 | 1–3 min |
| 5 (PLANT head, loads Stage 4 locally) | ~68 | 1–3 min |
| **Total compute** | | **~5–14 min** |

Plus fixed per-process overhead (Python/library imports, tokenizer/model
load from local cache) of roughly 20–60s × 5 stages ≈ 2–5 min. That
leaves comfortable headroom under 30 minutes — **provided
`mistralai/Mistral-7B-Instruct-v0.3` is already cached** on the GPU box.
The first-ever download of it (~15GB) is a one-time, bandwidth-dependent
cost that is *not* part of the pipeline's own budget; pre-warm it before
timing the run:

```bash
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3'); \
AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')"
```

If Stage 2 or 5 still looks like it's uploading multiple GB despite
`--skip_push_to_hub` (it shouldn't — that flag sets
`training_args.push_to_hub=False`, which every push code path in these
scripts checks before touching the network), Ctrl-C is safe once the logs
show training + evaluation completed: that's the part this smoke test
exists to verify, and this sample run is entirely disposable.

## 🏗️ **Full-scale candidate-generator subsystem (`plant-full-scale` branch, in progress)**

> 📍 **This is the extreme-label-space track** — see [Which command track do I need?](#-which-command-track-do-i-need). Needed for Amazon-3M (2.81M labels); not needed for bounded label spaces like MIMIC-IV's, which should use the plain [smoke-testing / "The commands"](#-smoke-testing-all-5-stages-on-a-sample-first) track instead.

Rather than permanently filtering the label space (which would silently
redefine the task to the easy head of a power-law distribution — top
50,000 labels are only 27.1% of all label-instances, so a lot of signal
lives in the tail), the chosen approach is a candidate-generator +
hard-negative-mining subsystem: score true positives plus a bounded set of
plausible-but-wrong labels per document instead of the full ~2.81M. Work so
far, in phases:

1. **Label clustering** (`scripts/build_label_clusters.py`, new) — a flat,
   single-level recursive balanced 2-means clustering of labels, adapted
   from AttentionXML's label-tree construction. Since Amazon-3M labels have
   no text of their own, each label's feature vector is reconstructed from
   its documents' `SentenceTransformer` embeddings (`labels_f =
   normalize(Y^T @ X)`). Validated on the 600-row/95-label sample: 8
   balanced clusters, sizes 11–12.
2. **Sparse `MultiLabelBinarizer`** (`multilabel_classify.py`'s
   `preprocess_data`) — `sparse_output=True` fixes the acute dense
   (num_docs × num_labels) blowup for full-corpus preprocessing.
3. **Router training** — a new `--label_space={label,cluster}` flag on both
   `multilabel_classify.py` (Stage 2) and `learning2rank.py`/`bootl2r.py`
   (Stage 3/4). In `cluster` mode, `MultilabelICDClassifier`/`LTRModel` are
   reused **completely unmodified**, just instantiated with
   `num_labels=num_clusters` — a bounded router problem the existing
   classes already handle fine at MIMIC-IV scale. Stage 3's MIG bootstrap
   (`bootl2r.py`) runs once per cluster instead of once globally, keeping
   every dense token-label tensor bounded by one cluster's label count; a
   new `scripts/aggregate_cluster_mig.py` collapses those per-cluster
   outputs (max score per token per cluster) into one combined file Stage
   4 can consume with zero code changes. Both routers validated end-to-end
   on the sample.
4. **Factorized label embeddings + candidate-restricted scoring**
   (`xcube/l2r/models/core.py`'s `LTRModel`, via
   `nbs/07_l2r.models.core.ipynb`) — the real memory blocker for the *full*
   label space isn't the forward pass (candidate-scoping already bounds
   that), it's that `torch.optim.SparseAdam` allocates **full dense**
   optimizer state regardless of `sparse=True` on the embedding
   (`torch.zeros_like(p.data)`), so a naive full-rank `(2.81M, 4096)`
   embedding table + Adam state is ~161GB — doesn't fit a single GPU no
   matter how the forward pass is restricted. Fixed via a low-rank base
   table (`nn.Embedding(num_labels, r=256)`) plus one shared
   `nn.Linear(256, 4096)` up-projection (~17GB total, comfortable
   alongside a 7B backbone). `LTRModel.forward()` gained
   `candidate_label_ids`/`candidate_mask` params; when given, the ground-truth
   lookup uses paired advanced indexing
   (`lookup_table[tok_idx, cand_idx]`) rather than
   `lookup_table[input_ids, :]` followed by a column gather — the latter
   would still materialize a full `(batch, max_len, num_labels)`
   intermediate before narrowing to the candidate budget, defeating the
   entire point.
5. **`approxNDCG_loss` shape bug, found and fixed everywhere** — while unit
   testing the new masking wrapper, found that `ptranking`'s
   `approxNDCG_loss` divides `batch_dcg` (shape `(num_rows,)`) by
   `batch_idcgs` (shape `(num_rows, 1)`, from `torch_dcg_at_k`'s
   `keepdim=True`) without squeezing the trailing dim first, so the
   division broadcasts to `(num_rows, num_rows)` — only the diagonal is
   the correct per-row nDCG; every off-diagonal entry divides one row's
   DCG by an unrelated row's IDCG, and the final mean averages over all
   `num_rows²` entries instead of the intended `num_rows`. This is not new
   code — it's the unmodified upstream `ptranking` function, called the
   same way at every `approxNDCG_loss` site in this repo (Stage 2/5's
   classifier heads, `LTRModel`), so it affected every loss value produced
   before this fix, including previously-validated runs. Fixed via
   `xcube/l2r/models/core.py`'s new `fixed_approxNDCG_loss`
   (squeezes `batch_idcgs` before dividing, avoiding the broadcast
   entirely) and `masked_approxNDCG_loss` (same fix, plus excludes
   padding rows — from a fixed-size candidate budget — from the mean).
   Both classifier heads and `LTRModel` now use `fixed_approxNDCG_loss`.
   **Loss/eval numbers from any run before this fix are not directly
   comparable to runs after it.**

6. **Factorized embeddings + candidate plumbing in Stage 2/5's classifier
   heads** (`multilabel_classify.py`'s `MultilabelICDClassifier` and
   `MultilabelICDPlantClassifier`, inside `define_model()`) — same
   `label_embed_rank` mechanism as `LTRModel`, applied to all three
   per-label parameters that scale with `num_labels` here
   (`label_embeddings`, `boost_mul`, `boost_add`). `boost_mul`/`boost_add`'s
   factorized form uses a shared projection whose *bias* recovers the
   original uniform init (`boost_mul≈1`, `boost_add≈0` for every label at
   init) while the low-rank per-label base term (initialized to zero) is
   free to diverge during training — verified via a standalone init test.
   Both `forward()`s gained `candidate_label_ids`/`candidate_mask` params.
   Masking here is different from `LTRModel`'s: these classifiers' tensors
   are `(batch, n)` — each row is one *document*, columns are its own
   candidate labels — so padding means some *columns* within a row are
   fake, not that whole rows are (unlike `LTRModel`, where each row is
   already one `(example, candidate-slot)` pair). Handled by masking the
   predicted logits to a very low score and zeroing the padding columns'
   relevance before calling `fixed_approxNDCG_loss`, rather than a
   dedicated masked-mean wrapper — verified that real candidates' relative
   ranking is unaffected by how many padding columns exist. New
   `--label_embed_rank` CLI flag on `multilabel_classify.py`; both
   checkpoint-loading paths (`load_custom_task_model_and_tokenizer`, and
   the fresh-training path in `main()`) now read/pass it consistently so a
   factorized checkpoint always reloads with the same architecture it was
   saved with.

7. **Candidate/hard-negative dataset construction** (Phase 5's last piece +
   Phase 6) — new `label_space="candidate"` mode in `preprocess_data()`,
   backed by new `build_candidate_arrays()`. For each document: true
   positive label indices (from the full vocabulary) plus same-cluster hard
   negatives — other labels sharing a cluster with at least one of the
   doc's positives — padded/truncated to a fixed `--candidate_budget`
   (default 256, matching the earlier factorization-rank decision). Same-
   cluster negatives are "free" here: they fall directly out of Phase 1's
   clustering, no separate mining step, matching Parabel/AttentionXML's
   mechanism (Phase 6 of the plan). Requires a prior "label"-space Stage 2
   run plus `scripts/build_label_clusters.py`, same dependency as `cluster`
   mode. `chunk_text()`/`process_batch()` extended to carry the new
   `candidate_label_ids`/`candidate_mask` columns through chunking
   alongside `labels` (now budget-length rather than num_labels-length in
   this mode) and `is_valid` — both are fixed-length per row, so the
   existing default `DataCollatorWithPadding` collates them with no new
   collator needed, exactly as planned. New `--candidate_budget` CLI flag.
   Verified via a standalone unit test against the real 600-row sample's
   actual label/cluster data: every positive is included (truncated
   correctly when a doc's true-label count exceeds the budget), no
   duplicate candidate ids per row, and every hard negative genuinely
   shares a cluster with one of that row's true positives.

8. **Fixed a real eval-metrics bug surfaced by the first `candidate`-mode GPU
   run**: `--label_space=candidate` trained cleanly (loss went from -0.36 to
   -0.44, stable, correct shapes throughout), but eval metrics came back
   implausibly low (`eval_f1_macro≈0.006` vs the same run's `label_embed_rank`
   dense-router baseline of `0.17`). Root cause: `compute_metrics_basic`/
   `compute_metrics_with_labels` (`multilabel_classify.py`) call
   `sklearn.f1_score(..., average="macro")` and build top-k predictions by
   column position, both of which assume **column `k` means the same label
   in every row** — true in `label`/`cluster` mode (column `k` is always
   `mlb_classes[k]`), but false in `candidate` mode, where column `k` is
   `candidate_label_ids[row, k]` — a *different* real label per row, since
   each document gets its own candidate set. Macro-averaging over that
   silently aggregates unrelated labels across rows into the same "column",
   producing a number that looks like a real (badly wrong-looking) score
   but isn't measuring anything coherent — not a model-quality problem, an
   eval-plumbing one. Fixed by new `remap_candidate_eval_arrays()`: scatters
   each row's real (positive or negative) candidate slots back into their
   true global column of a proper `(n_samples, num_labels)` array before
   handing off to the existing (now unmodified) metric functions; every
   label *not* in a row's candidate set is scored as confidently absent (a
   very low logit, 0 target) — the honest reflection of candidate-restricted
   scoring, since a two-stage system never predicts positive on a label it
   never shortlisted. Wired into `CustomTrainer.evaluate()`, which now
   detects candidate mode from the eval dataset's columns and remaps
   automatically — no changes needed at any call site. Verified via a
   standalone unit test (scatter correctness, untouched-column defaults,
   and that untouched columns correctly predict ~0 probability).

**Item 8's fix validated on the GPU box**: re-running the exact same
`--label_space=candidate` command after the fix, with no other changes,
moved `eval_f1_micro` from 0.010 → 0.178 and `eval_f1_macro` from 0.006 →
0.101, with precision/recall@k landing in a range comparable in shape to
the dense router run — confirming the previous numbers really were an
eval-plumbing artifact, not a reflection of model quality.

9. **Phase 7: router coverage@k + PSP@k eval metrics** (`multilabel_classify.py`).
   Two additions, both computed purely from dense eval-time `probs`/`labels`
   arrays (already remapped to global label columns in `candidate` mode by
   item 8, so no candidate-mode special-casing needed here):
   - **`router_coverage_at_k`** (new `compute_router_metrics()`, layered on
     top of the ordinary metrics only when `--label_space=cluster`) — a
     per-*document* all-or-nothing check: are **all** of a doc's true
     clusters within its top-k predicted clusters? A doc's true clusters are
     exactly the columns marked 1 in its cluster multi-hot target (by
     construction, cluster `c` is 1 iff it contains ≥1 of the doc's true
     labels — see `build_label_or_cluster_vectors`'s `"cluster"` branch), so
     this needs no extra per-doc fine-label bookkeeping, just
     `np.all(labels <= top_k_preds, axis=1)`. This is deliberately *not* the
     same thing as the already-existing label-averaged `recall_at_k`: a doc
     can have 90% of its true clusters in the top-k (good average recall)
     while still having one true cluster excluded — and every label inside
     that excluded cluster becomes permanently unreachable once candidate
     generation drops it. `router_coverage_at_k` catches exactly that
     failure mode, since it upper-bounds what the downstream fine stage
     could ever recover for that document. Verified via a standalone unit
     test with a constructed 3-doc/5-cluster example where one doc's top-k
     shortlist misses one true cluster entirely — `recall_at_k`-style
     partial credit would hide this, `router_coverage_at_k` correctly scores
     it as uncovered.
   - **`psp_at_k`** (new `get_inv_propensity()`/`get_psp_at_k()`, wired into
     both `compute_metrics_basic` and `compute_metrics_with_labels`, all
     label spaces) — propensity-scored precision@k (Jain et al. 2016),
     reimplemented in plain numpy against this repo's own dense eval arrays
     rather than imported, following the reference formulas in
     `~/AttentionXML/deepxml/evaluation.py::get_inv_propensity`/`get_psp`. A
     hit on a rare (high-propensity) label counts for more than a hit on a
     head label, and the score is normalized per-row against the best any
     model could do given that row's own true labels (the top-k
     highest-propensity among them) rather than a flat `k` — needed because
     on a power-law label distribution, vanilla P@k is dominated by head
     labels and says nothing about tail recall, which is the entire reason
     the candidate-generator subsystem exists. Propensity weights are fit
     from the eval split's own label frequency (not a separately-plumbed
     train split) — PSP@k only needs relative rarity, and on this pipeline's
     sample-sized smoke tests train/eval frequency is close enough; wiring
     true corpus-wide (train-split) frequency through would mean threading a
     new array through `preprocess_data()`/`main()`/`train_model()` for a
     metric that doesn't need that precision at this stage. Verified via a
     standalone unit test: a model that always retrieves a doc's true rare
     label scores meaningfully higher PSP@1 than one that always guesses the
     common label, even though both get the same raw precision@1.

   **Scope decision — Stage 4/5's `learning2rank.py` does *not* get these
   two metrics.** Its `compute_metrics()` operates on a fundamentally
   different shape: `(batch, max_len, num_labels)` token-relevance
   predictions, reshaped to one row per `(document, label)` pair for
   token-ranking NDCG — there is no single row per *document* to check
   cluster/label coverage against, the way there is in
   `multilabel_classify.py`'s classifier-head eval. Retrofitting
   `router_coverage_at_k`/`psp_at_k` there would mean first collapsing
   token-level relevance into one score per document per label (a real
   design decision, not a metric-plumbing one) — out of scope for what
   Phase 7 asked for. `learning2rank.py`'s existing NDCG/NDCG@k/precision@k
   already answer the question that shape of eval *can* answer (token-
   ranking quality per label), just not the router-shortlist-coverage
   question, which lives entirely on the classifier-head (Stage 2/5) side.

Every piece of the full-scale candidate-generator plan (Phases 1-7) now has
code in place, validated end-to-end on the GPU box (clustering, both
routers including Phase 7's new metrics, one `candidate`-mode training run
pre-fix, one post-fix confirming the corrected metrics) or via standalone
unit tests against real sample data or constructed test cases.

**A real bug surfaced by the first `--label_space=cluster` GPU run of Phase
7**: training ran fine, but crashed with `KeyError: 'candidate_label_ids'`
on the very first `evaluate()` call. Root cause: `chunk_text()`
unconditionally put `candidate_label_ids`/`candidate_mask` keys into every
chunk dict (`None` outside `candidate` mode), so
`Dataset.from_list(flattened_chunks)` (`preprocess_data()`) always inferred
a schema containing those two columns, regardless of `label_space` — even
though `set_format("torch", columns=...)` only actually includes them for
`label_space="candidate"`. `CustomTrainer.evaluate()`'s candidate-mode
detection (`"candidate_label_ids" in eval_dataset.column_names`) checks
schema presence, not the real mode, so it misfired `True` in `cluster` mode
too, then crashed trying to read a key the formatted batch never had. Fixed
at the source: `chunk_text()` now only adds those two keys to a chunk dict
when `candidate_label_ids is not None`, so a `label`/`cluster`-mode
dataset's schema never contains them and `evaluate()`'s existing
presence-check correctly discriminates all three modes with no further
change needed. Verified via a standalone test reproducing the exact
`Dataset.from_list` schema-inference behavior with and without the fix.

**Validated on the GPU box** after the fix: a fresh `--label_space=cluster`
run trained and evaluated cleanly end to end (8 clusters, sample corpus).
`eval_router_coverage_at_5=0.932` came in meaningfully below
`eval_recall_at_5=0.946` — exactly the gap the metric is designed to
surface, since coverage is the stricter all-true-clusters-in-top-k check.
`eval_psp_at_5=0.882` also landed in a sane sub-1.0 range. As expected for
only 8 clusters, `router_coverage_at_k`/`psp_at_k`/`recall_at_k` all hit the
trivial ceiling of `1.0` once `k>=8` (top-"k≥num_clusters" is just every
cluster) — the degenerate case flagged ahead of the run, not a bug.

10. **Stage 4's candidate-restricted "fine" training, wired up in
    `learning2rank.py`** — `LTRModel.forward()` already supported
    `candidate_label_ids`/`candidate_mask` (item 4 above), but the actual
    Stage 4 training driver (`learning2rank.py`) had no `--label_space` or
    candidate-dataset-construction code at all; the Stage 4 **router** only
    ever worked because `aggregate_cluster_mig.py` hands it a smaller
    pre-aggregated dataset (item 3), not because Stage 4 itself understood
    candidate restriction. Fixed by mirroring `multilabel_classify.py`'s
    mechanism into `learning2rank.py`: new `--label_space={label,candidate}`
    CLI flag, local duplicates of `load_cluster_assignments`/
    `build_candidate_arrays` (this repo's established no-cross-import
    convention), `candidate_label_ids`/`candidate_mask` threaded through
    `chunk_text`/`process_batch`/`set_format` exactly as in
    `multilabel_classify.py` (including proactively applying the item-9
    schema-inference fix in `chunk_text`, rather than reintroducing it), and
    a new `--label_embed_rank` flag threaded into `define_model()`/
    `LTRConfig` plus the checkpoint-reload path. `mlb_classes` for the
    candidate space is derived from `df_lbs` (`df_lbs.sort_values('lbl')`)
    rather than `labels_partition.json` directly, since `df_lbs`'s `lbl` ids
    are what `lookup_table`'s columns and every `candidate_label_ids` value
    actually index into — this requires a **plain** (non-cluster) Stage 3
    bootstrap run so those ids line up with the clustering, not the
    per-cluster Stage 3 output the router track uses. `CustomTrainer.evaluate()`
    needed zero changes for training/loss (candidate tensors already flow
    into `LTRModel.forward()` generically via the existing `self.model(**batch)`
    call) — only the eval-metrics bug below needed a real fix. Known,
    documented-not-fixed limitation: `--generate_report` combined with
    `--label_space=candidate` will misattribute label names in the token-rank
    report (an analogous issue to item 8, but for a diagnostic feature, not
    core metrics — a runtime warning fires if both are set together).

11. **A second eval-metrics bug, structurally identical to item 8's but in a
    different shape**, surfaced by the first GPU run of Stage 4 candidate
    mode: training and saving both worked cleanly, but
    `eval_precision@25` came back `0` with a `"No valid samples found for
    metric 'precision@25'"` warning, and `eval_ndcg@25=0.693` looked
    suspicious next to `eval_ndcg=0.980`. Root cause: `learning2rank.py`'s
    `compute_metrics()` reshapes `(batch, max_len, num_labels)` into one row
    per `(example, label-slot)` pair, then uses
    `non_zero_mask = labels.sum(dim=1) > 0` to drop rows with no relevance
    signal — a heuristic that's correct in `label`/`cluster` mode (a
    genuinely irrelevant label's relevance row really is all zero), but
    **wrong** in `candidate` mode: padding candidate slots default
    `candidate_label_ids` to `0` (a real label id, not a sentinel), so
    `LTRModel`'s `lookup_table[tok_idx, cand_idx]` gather pulls that real
    label's actual (often non-zero) relevance data into the padding slot —
    `non_zero_mask` alone can't tell a padding row from a real one, so NDCG/
    precision got computed over a mix of real candidates and
    randomly-labeled padding noise. Fixed by threading `candidate_mask`
    through as a new `row_mask` parameter on `compute_metrics()`
    (`non_zero_mask &= row_mask.reshape(-1)`, matching `LTRModel.forward()`'s
    own `masked_approxNDCG_loss` row-exclusion convention exactly), with
    `CustomTrainer.evaluate()` detecting candidate mode from the eval
    dataset's columns and passing `batch["candidate_mask"]` through — `None`
    outside candidate mode, so `label`/`cluster`-mode behavior (and every
    previously-GPU-validated router run) is provably unaffected. Verified
    via a standalone test reproducing the exact contamination (a padding
    slot with nonzero "borrowed" relevance data wrongly surviving the old
    mask, correctly excluded by the new one) and confirming the fix is a
    no-op outside candidate mode.

**Item 11's fix validated on the GPU box.** The first re-run after the fix
showed nearly-unchanged `eval_ndcg@25`/`eval_precision@25` numbers, which
looked suspicious given ~95% of candidate slots are padding — a correct fix
should move these a lot. Rather than guess, temporary diagnostics were added
(`is_candidate_mode` + `eval_dataset.column_names` at the start of
`evaluate()`, an aggregate real-vs-padding row count from `candidate_mask`,
and a per-batch before/after `non_zero_mask` row count inside
`compute_metrics()`). The next run's diagnostics settled it definitively:
`candidate_mask real-vs-padding rows across eval: 951/18944 real (5.0%)`,
and every batch's `non_zero_mask` dropped from `2048` (`batch_size=8 *
budget=256`) to within `92-128` — a ~5% survival rate, exactly matching the
real-candidate ratio. The fix **is** correctly excluding padding rows. The
apparently "unchanged" `ndcg@25`/`precision@25` across runs was a false
signal from comparing independent stochastic training runs (no fixed seed)
against each other, not a controlled before/after comparison of the same
trained model's eval code — the loss curves and final weights differ run to
run regardless of the eval-metrics fix. `eval_precision@25=0` persisting is
most likely a genuine characteristic of only 80 training steps combined
with a strict `top_k=25` + `relevance_threshold=80` token-ranking metric on
this tiny sample, not a residual bug — `eval_ndcg` (`0.97`, no `@k` cutoff)
looks healthy throughout. The verbose per-batch/one-time diagnostics were
downgraded to `logging.debug` afterward (available if needed again, silent
by default); the aggregate real-vs-padding summary line was kept at `info`
since it's a single low-noise line per eval that's useful to sanity-check
in any future candidate-mode run.

12. **A third bug, specific to Stage 5's `MultilabelICDPlantClassifier` in
    candidate mode**, surfaced immediately (first training step) when
    running Stage 5 with `--label_space=candidate` on top of Stage 4's new
    candidate-mode checkpoint: `RuntimeError: The size of tensor a (256)
    must match the size of tensor b (95) at non-singleton dimension 1`
    inside `torch.addcmul`. Root cause: `MultilabelICDPlantClassifier.forward()`
    calls its `self.base_model` (the upstream LTR model) *without* forwarding
    `candidate_label_ids`/`candidate_mask`. `LTRModel.forward()` falls back to
    scoring every one of its own `num_labels` (95) when those aren't given,
    so `outputs["logits"]` — used as `attn_weights`, then `attn_output` — came
    back 95-wide. Meanwhile the boosting step a few lines later *does* gather
    `boost_mul`/`boost_add` at the 256-wide `candidate_label_ids` (from
    `--candidate_budget=256`), so `torch.addcmul(boost_add, attn_output,
    boost_mul)` tried to combine a 256-wide and a 95-wide tensor. This is
    structurally different from `MultilabelICDClassifier` (Stage 2), which
    never has this problem: it does its own label-attention internally
    against a generic, label-count-agnostic base model, so candidate
    restriction never needs to reach into `self.base_model` at all. Stage 5's
    `base_model` *is* an LTR model that does its own per-label scoring, so
    the candidate restriction has to propagate through it too. Fixed by
    passing `candidate_label_ids`/`candidate_mask` into the
    `self.base_model(...)` call, matching what the boosting step already
    expects.

**Item 12's fix validated on the GPU box.** Stage 5 with
`--label_space=candidate --label_embed_rank=256 --candidate_budget=256`,
loading Stage 4's candidate-mode checkpoint, now trains and evaluates
cleanly end to end: `eval_f1_micro=0.272`, `eval_psp_at_5=0.744`,
`eval_precision_at_5=0.205`/`eval_recall_at_5=0.901` — all in the same
plausible range as Stage 2's earlier candidate-mode validation (item 8),
confirming the full candidate-restricted chain (Stage 3 → Stage 4 → Stage 5,
attention transfer plus independent candidate-restricted label scoring at
each stage) now works end to end on the sample.

With items 10-12 validated, **every stage of the full-scale
candidate-generator pipeline now has a working, GPU-tested command** — see
"Commands: the full candidate-generator pipeline" below for the complete
step-by-step sequence.

### Commands: the full candidate-generator pipeline (validated on the sample)

Self-contained end to end (repeats Stages 1-3 from "The commands" above,
since the router and fine tracks both need them) — skip a step if you've
already run its exact command elsewhere. The **router track** (Stage
2/3/4 in `cluster` mode) and the **fine track** (Stage 3/4/5 in `candidate`
mode) both branch off the same Phase 1 clustering artifact and can be run
independently of each other; only the fine track produces a final
classifier, the router track is a diagnostic on whether the clustering is
good enough (`router_coverage_at_k`/`psp_at_k`).

```bash
cd ~/plant/scripts
source setup_plant_names.sh "amazon-3m-sample_full" "mistralai/Mistral-7B-Instruct-v0.3" "../tmp_sample"

STAGE1_LOCAL="$BASE_DIR/AMAZON_3M/$STAGE1_OUTPUT_DIR"
SOURCE_DIR=$(python3 -c "from xcube.imports import *; print(untar_xxx(XURLs.AMAZON_3M))")

# --- Stage 1 (unchanged from the base 5-stage pipeline) -------------------
launches/launchLLM \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE1_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=1 \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 --skip_push_to_hub

# --- Stage 2, label space (unchanged -- writes labels_partition.json, -----
# a hard dependency of every step below) ------------------------------------
launches/launchmultilabelclas \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE2_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-classify" --do_fresh_training \
    --num_train_epochs=1 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_25" --rare_threshold=5 \
    --label_coverage_frac=1.0 --skip_push_to_hub

# --- Phase 1: label clustering ----------------------------------------------
# Writes cluster_assignments.json/cluster_to_labels.json under
# $BASE_DIR/AMAZON_3M. --target_cluster_size is tuned small here to get a
# handful of clusters out of only 95 sample labels -- this is still an open
# tuning question for the real ~2.81M-label corpus (see "Open decisions").
python3 build_label_clusters.py \
    --base_dir="$BASE_DIR/AMAZON_3M" \
    --csv_path="$SOURCE_DIR/$DATASET_NAME.csv" \
    --target_cluster_size=12

# ============================================================================
# Router track -- validates the clustering itself (router_coverage_at_k /
# psp_at_k) before trusting it to generate the fine track's candidate sets
# ============================================================================

# --- Stage 2 router (cluster label space) -----------------------------------
launches/launchmultilabelclas \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="${STAGE2_OUTPUT_DIR}_router" \
    --hub_model_id="$STAGE1_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-classify" --do_fresh_training \
    --num_train_epochs=1 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_5" --rare_threshold=5 \
    --label_coverage_frac=1.0 --label_space=cluster --skip_push_to_hub
# Watch for eval_router_coverage_at_k / eval_psp_at_k in the output.

# --- Stage 3, per-cluster MIG bootstrap (loops over every cluster) ---------
launches/launch_l2rboot \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" \
    --base_dir="$BASE_DIR" --output_dir="$STAGE3_OUTPUT_DIR" \
    --hub_model_id="$STAGE3_HUB_ID" --bs=8 --chnk_sz=200 --max_length=1024 \
    --label_space=cluster

# --- Bridge: combine the per-cluster outputs into one Stage-4-consumable ---
# file set (max token-relevance score per cluster) --------------------------
python3 aggregate_cluster_mig.py \
    --l2r_boot_dir="$BASE_DIR/AMAZON_3M/$STAGE3_OUTPUT_DIR" \
    --fname="amazon-3m-sample" \
    --output_prefix="amazon-3m-sample_cluster_agg" \
    --base_dir="$BASE_DIR/AMAZON_3M"

# --- Stage 4 router (plain command pointed at the aggregated data -- no ----
# --label_space flag needed: num_labels is inferred from the data's shape) -
launches/launch_l2r_training \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" \
    --data_l2r_fname_prefix="amazon-3m-sample_cluster_agg" --max_length=512 \
    --base_dir="$BASE_DIR" --l2r_boot_dir="$STAGE3_OUTPUT_DIR" \
    --output_dir="${STAGE4_OUTPUT_DIR}_router" \
    --hub_model_id="$STAGE1_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=1 --learning_rate=1e-4 --warmup_steps=0 \
    --metric_for_best_model="ndcg@25" \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 --skip_push_to_hub

# ============================================================================
# Fine track -- bounded per-document candidate scoring; this is what actually
# scales to Amazon-3M's full ~2.81M-label vocabulary
# ============================================================================

# --- Stage 3, plain (non-cluster) bootstrap ---------------------------------
# Candidate mode needs THIS run, not the per-cluster one above -- its 'lbl'
# ids must line up with labels_partition.json/cluster_assignments.json's
# global label ordering, which the per-cluster run's cluster-local ids don't
# provide. Writes into the SAME $STAGE3_OUTPUT_DIR as the router track above
# (different filenames, no collision) -- this is also the exact "Stage 3"
# command from "The commands" above, so skip it if already run.
launches/launch_l2rboot \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" \
    --base_dir="$BASE_DIR" --output_dir="$STAGE3_OUTPUT_DIR" \
    --hub_model_id="$STAGE3_HUB_ID" --bs=8 --chnk_sz=200 --max_length=1024

# --- Stage 4, candidate mode -------------------------------------------------
launches/launch_l2r_training \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" \
    --data_l2r_fname_prefix="amazon-3m-sample" --max_length=512 \
    --base_dir="$BASE_DIR" --l2r_boot_dir="$STAGE3_OUTPUT_DIR" \
    --output_dir="${STAGE4_OUTPUT_DIR}_candidate" \
    --hub_model_id="$STAGE1_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=1 --learning_rate=1e-4 --warmup_steps=0 \
    --metric_for_best_model="ndcg@25" \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 --skip_push_to_hub \
    --label_space=candidate --label_embed_rank=256 --candidate_budget=256

# --- Stage 2, candidate mode (optional) -------------------------------------
# Standalone proof that the candidate-restricted classifier-head
# architecture works on its own -- NOT required by Stage 5 below, which
# loads Stage 4's checkpoint directly, not Stage 2's.
launches/launchmultilabelclas \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="${STAGE2_OUTPUT_DIR}_candidate" \
    --hub_model_id="$STAGE1_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-classify" --do_fresh_training \
    --num_train_epochs=1 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_25" --rare_threshold=5 \
    --label_coverage_frac=1.0 --skip_push_to_hub \
    --label_space=candidate --label_embed_rank=256 --candidate_budget=256

# --- Stage 5, candidate mode -------------------------------------------------
# hub_model_id = Stage 4 candidate's LOCAL output. Only its attention weights
# get reused (see the "PLANT Attention Transfer" section), so this works
# regardless of whether Stage 4 or the optional Stage 2 above used
# label/cluster/candidate mode -- they don't need to match each other.
STAGE4_LOCAL_CANDIDATE="$BASE_DIR/AMAZON_3M/${STAGE4_OUTPUT_DIR}_candidate"
launches/launchmultilabelclas \
    --source_url="XURLs.AMAZON_3M" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="${STAGE5_OUTPUT_DIR}_candidate" \
    --hub_model_id="$STAGE4_LOCAL_CANDIDATE" \
    --task="multilabel-plantclassify" --do_fresh_training \
    --num_train_epochs=1 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_25" --rare_threshold=5 \
    --label_coverage_frac=1.0 --skip_push_to_hub \
    --label_space=candidate --label_embed_rank=256 --candidate_budget=256
```

## 🔧 **What had to change to port this from MIMIC-IV**

1. **`scripts/setup_plant_names.sh` — fixed a latent naming bug.**
   `DATASET_SHORT` was computed as `cut -d'_' -f1-2`, which assumes the
   dataset string always has ≥3 underscore-separated fields (true for
   `mimic4_icd10_full`, false for `amazon-3m_full` — only 2 fields, so the
   old logic returned the whole string, `_full` suffix included). Replaced
   with `${DATASET%_*}` (strip the trailing underscore segment), which
   matches exactly how `bootl2r.py` derives its own file-naming prefix and
   gives the correct result for both patterns. Verified no change in
   output for the existing MIMIC-IV naming (regression-checked). This
   also silently fixes `scripts/launches/launchl2rboot_amazon670k`'s
   `amazon-670k_full` case, which hit the identical bug.
2. **One-time symlink** `amazon-3m_full.csv → amazon-3m.csv` in the
   downloaded data directory, so `--data="amazon-3m_full"` satisfies both
   of `get_data()`'s literal-filename lookup and `bootl2r.py`'s
   suffix-stripped prefix derivation. See
   [above](#️-one-time-dataset-prep-the-_full-symlink).
3. Everything else — `finetune_llm.py`, `multilabel_classify.py`,
   `bootl2r.py`, `learning2rank.py`, `check_tokens.py`, all `launches/*`
   scripts — is dataset-agnostic by construction (parametrized entirely
   through `--source_url` / `--data` / `--label_delim`, defaulting to `;`
   which is also Amazon-3M's label delimiter). No code changes were
   needed there; only the command-line values above differ from the
   MIMIC-IV invocations in README.md.
4. Verified with a real (not mocked) CPU-only run of the Stage 3
   tokenization/label-consistency check against the actual downloaded
   `amazon-3m.csv` — passed. Heavier stages (1, 3 full run, 4, 5) need the
   GPU box and weren't run here.
5. Extended `scripts/finetune_llm.py` and `scripts/learning2rank.py` with
   `--num_train_epochs` / `--per_device_train_batch_size` /
   `--per_device_eval_batch_size` / `--gradient_accumulation_steps` CLI
   overrides. They were hardcoded (`epochs=3, batch=1, grad_accum=4`)
   while `scripts/multilabel_classify.py` already exposed them — a plain
   inconsistency, now fixed the same way in all three, defaults unchanged
   (so existing MIMIC invocations behave identically).
6. Added `--skip_push_to_hub` to all three training scripts (default
   `False`, no behavior change unless passed) — see
   [the smoke-test section](#-smoke-testing-all-5-stages-on-a-sample-first)
   for why.
7. Added `precision_at_25` / `recall_at_25` (+ `rare_`/`not_rare_`
   variants) as additional `--metric_for_best_model` choices in
   `multilabel_classify.py` — see
   [Precision@k for Amazon-3M](#-precisionk--ndcgk-for-amazon-3ms-label-density).
8. Fixed `--skip_push_to_hub` to also short-circuit each script's
   post-training "reload the model" step (it previously still tried to
   fetch `--hub_model_id` from the Hub whenever `--do_fresh_training` was
   set, regardless of whether the push happened — 404s the moment a push
   is skipped). Same fix, same reasoning, in all four affected call sites
   across the three training scripts.
9. Fixed a crash in `multilabel_classify.py`'s `compute_metrics_with_labels()`:
   the rare/not-rare label split uses a hardcoded `rare_threshold=50`
   (labels appearing in fewer than 50 training docs are "rare"). On a
   small sample, one whole partition can end up empty (0 columns), and
   sklearn's `f1_score`/`precision_recall_fscore_support` raise
   `ValueError: unknown is not supported` on a 0-column target instead of
   returning a sane default. Guarded all four rare/not-rare metric blocks
   to skip cleanly (default `0.0`) when a partition is empty, and made
   `rare_threshold` a `--rare_threshold` CLI arg (default `50`, unchanged)
   so sample runs can lower it to actually exercise both partitions
   instead of just avoiding the crash — `--rare_threshold=5` splits the
   600-row sample's 95 labels into 64 rare / 31 not-rare, both non-empty.
10. `multilabel_classify.py`'s `main()` calls an internal
    `sample_df_with_full_label_coverage(df, frac=0.001, ...)` *before*
    `preprocess_data()` runs, at a hardcoded `frac=0.001` call site
    (`multilabel_classify.py:4011`) — it first draws that fraction of the
    data, then greedily adds rows one at a time until every unique label
    is covered at least once. `0.001` is sized for MIMIC-IV's ~270K-row
    full dataset (≈270 rows); on the 600-row sample it rounds to ~0, so
    the greedy top-up alone decides the final size — observed shrinking
    Stage 2's actual training set to 39 rows (down from the intended 540)
    on a real run, which also explained why the rare/not-rare split
    looked different than expected and why eval metrics were noisy (3-row
    eval set). Made `frac` a `--label_coverage_frac` CLI arg (default
    `0.001`, unchanged); pass `--label_coverage_frac=1.0` to make it use
    the full split as-is (a `frac=1.0` draw already covers every label by
    construction, so the greedy top-up never triggers). Also worth
    flagging for the full-scale run: this same greedy loop, run against
    the real 2.8M-label corpus, could need to keep adding rows until it's
    touched a large fraction of the whole 2.46M-row dataset — another
    consequence of the label cardinality above, not a separate bug.
11. Removed a leftover dev-iteration block in `learning2rank.py`'s `main()`
    (Stage 4) that subsampled `df_data` to `frac = 0.001 * 0.3` (0.03%) of
    rows before every training run, marked by the original author's own
    `# quick iteration <remove later>` comments. On Amazon-3M's full
    dataset this silently trained on a few thousand rows instead of the
    full corpus; on the 600-row sample both the `is_valid`/`~is_valid`
    `.sample(frac=...)` calls rounded down to 0 rows, producing a fully
    empty `df_data`, which `preprocess_data()` turned into a zero-column
    HF `Dataset` and crashed with `Columns [...] not in the dataset`.
    Deleted the block entirely — Stage 4 now trains on the full
    `df_data` it's given, for both the sample and real MIMIC-IV runs.
12. Fixed a numerically-unstable training loss in Stage 4's `LTRModel`
    (`xcube/l2r/models/core.py`): `approxNDCG_loss` was called with
    `label_type=LABEL_TYPE.MultiLabel`, whose gain formula is `2^rel - 1`
    — sane for a handful of LETOR-style relevance grades (0-4), but the
    ground-truth relevance here comes from `quantize_l2r_data`'s 101
    quantile bins (grades 0-101), so `2^100 ≈ 2.5e30`. That single call
    produced training losses like `-8.4e15` and frequent `grad_norm: nan`
    on a real Stage 4 run — not a sample-only artifact, since 101
    quantization levels are hardcoded regardless of dataset size. Eval
    metrics looked fine because they're computed separately via
    `torchmetrics.retrieval_normalized_dcg`, which doesn't use this gain
    formula. Switched to `label_type=LABEL_TYPE.Permutation`, which uses
    linear gains (`batch_stds` as-is) — the mode `ptranking` itself
    provides for fine-grained/many-level relevance signals like this one,
    and numerically stable across the full 0-101 grade range.
13. Fixed Stage 5 failing to load Stage 4's local checkpoint as its base
    model (`../tmp_sample/.../amazon-3m-sample_l2rtrain_mistral7b` passed
    as `--hub_model_id`): `LTRModel.from_pretrained()`
    (`xcube/l2r/models/core.py`) unconditionally called `hf_hub_download`
    for `model.safetensors`, `pytorch_model.bin`, *and* `lookup_table.pt`
    — `hf_hub_download` requires a `namespace/repo_name` Hub id and raises
    `HFValidationError` on a local path, so all three lookups failed with
    `Repo id must be in the form 'repo_name' or 'namespace/repo_name'`.
    This is exactly the local-vs-Hub check `load_custom_task_model_and_tokenizer()`
    in `multilabel_classify.py` already does correctly — `LTRModel` (the
    Stage 4 output's model class) never got the same guard. Added an
    `os.path.isdir(...)` check before each of the three downloads,
    loading directly from disk when given a local directory and falling
    back to `hf_hub_download` only for genuine Hub ids.
    `xcube/l2r/models/core.py` is nbdev-generated
    (`nbs/07_l2r.models.core.ipynb`); this fix and item 12's were both
    made in the source notebook and re-exported via `nbdev_export`, so
    they survive the next notebook-driven rebuild instead of being
    silently clobbered.

## 📐 **Precision@k / NDCG@k for Amazon-3M's label density**

Your instinct that label density should inform the choice of k was
right, but it plays out differently for the two `--metric_for_best_model`
mechanisms in this codebase, because they measure different things:

- **`multilabel_classify.py` (Stages 2 & 5) — `precision_at_k` /
  `recall_at_k`**: this really is "top-k predicted labels vs. this
  document's true labels," so it directly depends on labels-per-document.
  The code only supported fixed `k ∈ {5, 8, 15}` (hardcoded in
  `train_model()`'s `k_values` default and the `--metric_for_best_model`
  `choices=[...]` list) — chosen, presumably, to match MIMIC-IV's ICD
  coding density. Measured directly on the full `amazon-3m.csv`: **median
  24 labels/doc**, capped at 100 — meaningfully higher than MIMIC's
  typical range and higher than any available k. I added `k=25` (matching
  the median almost exactly, and matching Stage 4's `ndcg@25` for a
  consistent k across the pipeline) as `precision_at_25` / `recall_at_25`
  (+ `rare_`/`not_rare_` variants) — additive, so existing MIMIC-IV
  invocations using `precision_at_15` etc. are unaffected. **Recommendation:
  use `--metric_for_best_model="precision_at_25"` for Amazon-3M's Stage 2
  and Stage 5**, in place of the MIMIC-tuned `precision_at_15` shown
  earlier in this doc.
- **`learning2rank.py` (Stage 4) — `ndcg@25` / `precision@25`**: despite
  the similar name, this is a *different* metric measuring token-ranking
  quality within one label (are the MIG-important tokens ranked near the
  top of the document, for documents of length up to `--max_length`) —
  it's governed by document length, not labels-per-document. Its `k=25`
  was already fine and needs no change for Amazon-3M; I left
  `learning2rank.py`'s metric set untouched.

One caveat: the median-24 figure is real for the **full** dataset; the
95-label smoke-test sample above collapses to ~1 label/doc by
construction (see above), so don't read anything into its metric values
— they're only there to prove the code runs, not to validate `k=25`.
