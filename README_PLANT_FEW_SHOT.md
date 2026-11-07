# 🌱 PLANT Few-Shot on MIMIC-IV

This is the **few-shot** companion to [README.md](README.md). The PLANT
paradigm, architecture, and 5-stage pipeline are all unchanged — read
[README.md](README.md) first for the concepts. What changes here is the
**dataset and the evaluation lens**: instead of running against all ~8K
MIMIC-IV ICD-10 codes (`--source_url="XURLs.MIMIC4_DEMO"`, as in
[README.md](README.md)), we carve out a purpose-built *few-shot* slice —
**MIMIC-IV-few** — by keeping only the discharge summaries that carry at
least one **rare target code** (a code seen only a handful of times in
training). Crucially, each kept note **keeps its full ICD-10 labels**, so
the task stays a genuine extreme-multi-label problem (**3,343 codes,
~17 codes/note**) — PLANT's native regime — not a stripped-down toy. We
then evaluate by **bucketing every code by its training frequency** into
**zero-shot / few-shot / head** and scoring each bucket separately, which
measures zero- and few-shot generalization directly. This file documents
how the slice is constructed and evaluated, measured directly on the
downloaded data, so a reader can reproduce it exactly.

Unlike [README_EURLEX_WIKI10.md](README_EURLEX_WIKI10.md) and
[README_AMAZON3M.md](README_AMAZON3M.md), there is **no separate
smoke-sample vs. full-scale split here**. Those files first build a
600-row/150-label sample to smoke-test the five stages fast, then point
the same commands at the full corpus. MIMIC-IV-few is deliberately built
small — **800 notes** total (see [Resulting dataset](#-resulting-dataset-mimic-iv-few)
below) — so that a single run *is* the experiment and trains quickly.
One dataset, one set of commands.

> 💻 **Where this runs**: same setup as the other companion READMEs —
> the dataset is assembled on the primary dev machine (CPU-only, this is
> just pandas over a CSV); the actual PLANT training runs on a separate
> cloud GPU box, which just clones this repo. This work has been **merged
> into `main`** — the reproduction commands below (and
> [GPU Box Setup](#️-gpu-box-setup)) target `main`, where this file lives.

## 📋 Table of Contents

- [✅ Status — what's done, what's next](#-status--whats-done-whats-next)
- [🧭 What we compare (and two meanings of "zero/few")](#-what-we-compare-and-two-meanings-of-zerofew)
- [⚙️ GPU Box Setup](#️-gpu-box-setup)
- [📊 Source Dataset: MIMIC-IV (ICD-10)](#-source-dataset-mimic-iv-icd-10)
  - [📥 Download](#-download)
  - [Measured statistics](#measured-statistics)
- [🎯 Constructing MIMIC-IV-few](#-constructing-mimic-iv-few)
  - [The definition](#the-definition)
  - [Choosing the frequency range, and the note cap](#choosing-the-frequency-range-and-the-note-cap)
  - [The note cap — honest caveat](#the-note-cap--honest-caveat)
  - [🔁 Reproducing the dataset](#-reproducing-the-dataset)
- [📦 Resulting dataset: MIMIC-IV-few](#-resulting-dataset-mimic-iv-few)
  - [Why keep the full labels (and what "few-shot" means here)](#why-keep-the-full-labels-and-what-few-shot-means-here)
  - [Files produced](#files-produced)
- [🎌 Evaluation: zero / few / head buckets](#-evaluation-zero--few--head-buckets)
- [🤖 Choosing a Base Model](#-choosing-a-base-model)
- [📝 Naming Conventions](#-naming-conventions)
- [🚀 Running All 5 Stages](#-running-all-5-stages)
  - [The commands](#the-commands)
  - [Errors we hit, and the fixes (reproducibility)](#errors-we-hit-and-the-fixes-reproducibility)
  - [Notes on the parameter choices](#notes-on-the-parameter-choices)
  - [Firing other combos](#firing-other-combos)
- [📈 Results: PLANT vs. baseline](#-results-plant-vs-baseline)
- [🧪 Training-free LLM baseline (context)](#-training-free-llm-baseline-context)
  - [Method: extract → link → score](#method-extract--link--score)
  - [The commands](#the-commands-1)
  - [Results: training-free vs. trained](#results-training-free-vs-trained)
  - [What this establishes](#what-this-establishes)
  - [Errors we hit (training-free)](#errors-we-hit-training-free)

## ✅ **Status — what's done, what's next**

| Step | State |
| --- | --- |
| Download `XURLs.MIMIC4_DEMO` | ✅ done |
| Build MIMIC-IV-few (full-label rare-code slice) | ✅ done — 800 notes, **3,343** codes, 203 target codes (train freq 3–7) selected the notes |
| Run the 5-stage PLANT pipeline (Mistral-7B, Llama-3.1-8B, Phi-3-small) | ✅ done — all 5 stages, 6 epochs (Stage 1 at 1), all three models |
| Compare Stage 2 (baseline) vs Stage 5 (PLANT), bucketed | ✅ done — **PLANT wins zero-shot / head / overall on all three**; the few-shot win holds for Mistral & Llama but not Phi-3-small — see [Results](#-results-plant-vs-baseline) |
| Training-free LLM baseline (extract → link), for context | ✅ done — Mistral-7B & Llama-3.1-8B, zero-/few-shot prompting — see [Training-free LLM baseline](#-training-free-llm-baseline-context) |

Everything below is **measured directly** — dataset stats from the files
on disk, result numbers from the actual Mistral-7B, Llama-3.1-8B, and
Phi-3-small runs. The
[Running All 5 Stages](#-running-all-5-stages) block is the exact set of
commands that produced [the results](#-results-plant-vs-baseline),
including the two failures we hit and the fixes that resolved them.

## 🧭 **What we compare (and two meanings of "zero/few")**

The question this experiment asks is narrow: **does PLANT's label-attention
head (Stage 5) actually help on rare and never-seen ICD-10 codes**, versus
alternatives, in the full-label extreme-multi-label regime? To answer it
fairly we score **three systems** on the *same* 167 held-out notes, the
*same* 3,343-code label space, and the *same* bucketed metric:

| System | Trained on MIMIC? | What it is |
| --- | --- | --- |
| **Stage 2 — baseline** | ✅ yes | The non-PLANT multi-label classifier head on the same base LLM. The "does training alone get there?" control. |
| **Stage 5 — PLANT** | ✅ yes | The full PLANT head (adds the Stage 3→4 label-attention prior). The method under test. |
| **Training-free LLM baseline** | ❌ **no** | The *same* base LLM, off-the-shelf, prompted to extract diagnosis/procedure phrases, which are then entity-linked to codes. The "how much needs training at all?" control. See [that section](#-training-free-llm-baseline-context). |

The training-free system matters because a trained classifier **structurally
cannot** rank a code it never saw in training — so a raw zero-shot number
against Stage 2 could flatter PLANT for the wrong reason. Pitting PLANT
against a strong *training-free* LLM tells us which gains are really about
PLANT and which are just "an LLM already knows this."

> ⚠️ **Two different things are both called "zero/few" in this document.**
> They are **independent axes** — keep them separate when reading the tables:
>
> | Axis | "zero" | "few" | Property of | Applies to |
> | --- | --- | --- | --- | --- |
> | **Evaluation bucket** | code seen **0×** in training | code seen **1–7×** in training | each **code** | scoring of **every** system |
> | **Prompting regime** | LLM given **0** in-context examples | LLM given **2** worked examples | the **prompt** | the **training-free** baseline only |
>
> The two compose freely. "Training-free **few-shot** on the **zero-shot**
> bucket" is a real, sensible cell: the *few-example-prompted* extractor
> scored on codes that were *never seen in training*. Bucket = which codes we
> score on; regime = how many examples the extractor's prompt carried.

## ⚙️ **GPU Box Setup**

Prerequisites first, so the rest of this file runs top-to-bottom.
Building the dataset (next section) is CPU-only — just pandas over a
CSV — but PLANT *training* runs on a GPU box; this is the from-scratch
bootstrap for that box. See [README.md's Installation & Setup](README.md#-installation--setup)
for the general xcube setup.

This setup uses the **default** base model,
**Mistral-7B-Instruct-v0.3** — the same model
[README.md](README.md#-naming-conventions--consistency) uses for
MIMIC-IV, which needs nothing beyond the standard install below. The
pipeline also supports Llama-3.1-8B and Phi-3-small — see
[Choosing a Base Model](#-choosing-a-base-model); those two need extra
per-architecture setup (HF gating, `flash_attn`, etc.), documented in the
EURLEX/Wiki readme, but nothing dataset-specific changes here.

1. **Get the code.** Fresh box:
   ```bash
   git clone git@github.com:deb-official/plant.git ~/plant
   ```
   Already have a checkout on this box? Just update it:
   ```bash
   cd ~/plant && git pull
   ```

2. **Python environment:**
   ```bash
   cd ~/plant
   pip install -e .
   ```
   This installs the `transformers==4.49.0` pinned in `settings.ini` —
   the version this pipeline was validated against (the primary dev
   machine's environment may have drifted to a newer `transformers`; a
   fresh GPU box following `settings.ini` as-is gets the tested
   combination). No Weights & Biases setup is needed — `report_to="none"`
   is hardcoded in the training scripts' `TrainingArguments`.
   > 🧪 Only for the [Training-free LLM baseline](#-training-free-llm-baseline-context)
   > (not the 5-stage pipeline): `pip install -U "jinja2>=3.1"` — the pinned
   > env ships 3.0.3, which is too old for `apply_chat_template`.

3. **Hugging Face auth** — Mistral-7B-Instruct-v0.3 requires accepting
   its terms on the model page and an authenticated token to download:
   ```bash
   huggingface-cli login   # or: export HF_TOKEN="..."
   ```

4. **Build the dataset on this box.** The `~/.xcube` data cache is
   per-box and *not* part of the git sync, so regenerate MIMIC-IV-few
   here (it also downloads `XURLs.MIMIC4_DEMO` if absent):
   ```bash
   cd ~/plant/scripts && python3 create_mimic4_few.py
   ```
   This is the same command as [Reproducing the dataset](#-reproducing-the-dataset)
   below, which explains the flags and expected output — deterministic,
   so the box gets the byte-identical 800-note `mimic4_icd10_few.csv`.

5. **`accelerate` precision config** — the pipeline's model loads request
   `bfloat16`, so set `accelerate`'s mixed-precision mode to match rather
   than leaving it at a generic default (`fp16`, in at least one observed
   case):
   ```bash
   accelerate config   # choose bf16 when prompted for mixed precision
   ```
   This lives in `~/.cache/huggingface/accelerate/default_config.yaml`
   per box (not in this repo), so it's a one-time manual step per fresh
   box, same as the HF login above. (Mistral tolerates fp32 attention, so
   this is best-practice consistency rather than a hard requirement — but
   set it to avoid surprises.)

**Disk space**: budget ~29GB for the Mistral-7B-Instruct-v0.3 Hub repo
(`from_pretrained` typically pulls only the `safetensors` shards, so
actual usage tends to run lower) plus <10MB for the few-shot dataset.
Comfortable on any GPU box with 50GB+ free. The reference companion runs
were on an **NVIDIA H100 80GB**; the few-shot dataset is far smaller, so
it is not GPU-memory-bound at this scale.

## 📊 **Source Dataset: MIMIC-IV (ICD-10)**

MIMIC-IV-few is derived entirely from the ICD-10 discharge-summary file
shipped inside `XURLs.MIMIC4_DEMO`
(`XURLs.MIMIC4_DEMO` in `xcube/data/external.py` →
`https://xcubebucket.s3.us-east-2.amazonaws.com/mimic4/mimic4_demo.tar.gz`).
Each row is a hospital discharge summary (`text`) with its assigned
ICD-10 codes as `;`-delimited `labels`.

### 📥 **Download**

```python
# 🐍 Python REPL
from xcube.data.external import untar_xxx, XURLs
p = untar_xxx(XURLs.MIMIC4_DEMO)
print(f"Dataset location: {p}")
# -> ~/.xcube/data/mimic4_demo
# Files present:
#   mimic4_icd10_full.csv   <-- used here (discharge summaries + ICD-10 codes)
#   mimic4_icd9_full.csv    <-- the ICD-9 counterpart, not used in this file
```

`mimic4_icd10_full.csv` has 15 columns; the ones that matter for the
few-shot construction:

| Column | Meaning |
| --- | --- |
| `text` | Discharge-summary free text (already lowercased/cleaned). |
| `labels` | `;`-delimited ICD-10 codes for the note (e.g. `E78.5;F02.80;G31.83`). |
| `split` | Authoritative 3-way split: `train` / `val` / `test`. |
| `is_valid` | Boolean — `True` **iff** `split == "test"` (verified on the full file). `val` rows are `is_valid=False`, i.e. folded in with `train` for the pipeline's train/valid boolean. `split` is the source of truth for the 3-way partition. |

The other columns (`note_id`, `subject_id`, `_id`, `note_type`,
`note_seq`, `charttime`, `storetime`, `icd10_proc`, `icd10_diag`,
`num_words`, `num_targets`) are carried through the filter untouched but
are not used to define the slice.

### Measured statistics

Measured directly on the downloaded `mimic4_icd10_full.csv`:

| Quantity | Value |
| --- | --- |
| Rows (discharge summaries) | 122,279 |
| Split | 89,099 train / 13,378 val / 19,802 test |
| Unique ICD-10 codes (all splits) | 7,942 |
| Unique ICD-10 codes (**train only**) | 7,939 |
| Labels/doc | median 14 (mean 15.65, max 66) |

> ℹ️ This "full" CSV is already a **curated** MIMIC-IV subset, not the
> raw code space — see [the frequency cliff](#choosing-the-frequency-range-and-the-note-cap)
> below, which is what shapes the few-shot construction.

## 🎯 **Constructing MIMIC-IV-few**

The goal is a *few-shot* label-coding benchmark: a set of ICD-10 codes
for which the model sees only a handful of training examples, plus the
notes those codes appear on. Everything is keyed off **training-set**
frequency so that "few-shot" refers to how much *training* signal exists
per target code.

### The definition

1. **Count** each ICD-10 code's frequency in the **training split**
   only (number of `train` notes whose `labels` include it).
2. **Select target ("rare") codes** by a **training-frequency range
   `[min_freq, max_freq]`** — every code whose train frequency lies in
   that range. The default range is **[3, 7]** → **203 target codes**,
   each with 3–7 training examples (genuinely few-shot, while skipping
   the near-impossible 1–2-shot codes). The range is the script's main
   knob (`--min_freq` / `--max_freq`).
3. **Filter notes**: keep a discharge summary — in **any** split
   (`train`/`val`/`test`), preserving the existing `split` column — **iff
   at least one of its `labels` is a target code**. Every note without a
   target code is dropped. For **[3, 7]** this yields **2,134 notes**.
4. **Cap to ~800 notes** for quick training: because a pure frequency
   range can't land near 800 on its own (see the
   [cliff](#choosing-the-frequency-range-and-the-note-cap) below), the
   kept notes are then **deterministically subsampled to 800**
   (`--max_notes`, split-proportional, fixed `--seed`).
5. **Keep each note's full ICD-10 labels** (`--label_scope=all`, the
   default): the target codes only *select* which notes to keep — every
   kept note retains **all** its original codes, so the label space is
   every code that co-occurs on the selected notes (**3,343**), a genuine
   extreme-multi-label task (~17 codes/note). We do *not* strip to the
   target codes, because that would collapse the task to a ~200-code
   single-label problem and throw away PLANT's multi-label wheelhouse.
   Instead, few-shot difficulty is recovered at **evaluation** time by
   bucketing every code by its training frequency — see
   [Evaluation](#-evaluation-zero--few--head-buckets). (`--label_scope=targets`
   still exists to reproduce the stripped variant; see
   [Why keep the full labels](#why-keep-the-full-labels-and-what-few-shot-means-here).)

Rare target codes are defined by *training* frequency, but the
note-keeping filter is applied across **all splits** so that `val`/`test`
notes are retained for evaluation — the standard few-shot / rare-code
benchmark setup.

### Choosing the frequency range, and the note cap

**TL;DR** — a frequency *range* can't hit ~800 notes on its own (the
source has a sharp frequency cliff), so we pick a sensible few-shot range
**[3, 7]** and separately **cap the notes at 800**. Expand for the why.

<details>
<summary>The frequency cliff, and why range + cap are separate knobs</summary>

The target was **quick training** — roughly ~800 notes. The key finding:
**no pure frequency range yields ~800 notes**, because this curated CSV
has a sharp **cliff** in its training-frequency distribution:

| Train frequency | # codes | | Range | # target codes | # notes kept |
| --- | --- | --- | --- | --- | --- |
| ≤ 6 (all) | 21 | | [1, 6] | 21 | 291 |
| 7 | 184 | | [1, 7] | 205 | 2,156 |
| 8 | 622 | | [3, 7] | 203 | 2,134 |
| 9 | 339 | | [7, 7] | 184 | 1,873 |

Frequency 7 alone adds 184 codes and ~1,865 notes **as one indivisible
band**, so the achievable note count jumps straight from ~291 (any
`hi ≤ 6`) to ~1,900 (`hi = 7`) — **800 falls squarely in that gap**, and
raising `min_freq` barely moves it (the freq 1–6 codes contribute almost
nothing). So the range alone can't hit the target.

The chosen resolution: pick a sensible few-shot range — **[3, 7]** (203
codes, 3–7 training examples each) — and then **cap the notes at 800**
via a deterministic, split-proportional subsample. The range defines
*which codes are few-shot*; the cap sets *how many notes train*. They're
independent knobs, which is why the script takes both.

> ℹ️ **The cliff is a property of the source file, not our method.**
> `mimic4_icd10_full.csv` was evidently pre-filtered to drop most
> ultra-rare codes: only **12** codes occur ≤ 5 times in training, then
> the count explodes at freq 7–8. A *raw* MIMIC-IV ICD-10 space would
> have a long tail of thousands of 1–5-occurrence codes; this curated
> file does not. If a genuinely long-tail few-shot space is wanted, this
> source won't provide it — that would need a less-filtered MIMIC-IV
> export.

</details>

### The note cap — honest caveat

**TL;DR** — the cap is a real, seeded subsample: it drops **1 of 203**
target codes entirely, and `--max_notes=0` disables it. Expand for detail.

<details>
<summary>What the subsample costs, and how to turn it off</summary>

The 800-note cap subsamples the 2,134 filtered notes down to 800,
**proportionally within each split** (so the train/val/test ratio is
preserved) and with a **fixed seed** — re-running the script produces the
byte-identical file (verified). It's a real subsample, so two honest
consequences:

- **A target code can lose all its notes.** At [3, 7], the subsample
  keeps **202 of the 203** target codes represented — one code (its
  handful of notes all dropped by the cut) no longer appears. The script
  prints this count (`Target codes still represented: 202 / 203`) so it's
  visible, not silent. Raise `--max_notes` (or `--seed`-shop) if you need
  all codes retained.
- **The cap, not the frequency range, is what sets 800.** If you want the
  full, uncapped few-shot slice instead (every note with a target code,
  no subsample), pass `--max_notes=0` — [3, 7] then gives all 2,134
  notes. The cap exists purely to make training fast.

</details>

### 🔁 **Reproducing the dataset**

The whole construction — training-frequency count, range-based
target-code selection, note filtering, and the note cap — is packaged in
[`scripts/create_mimic4_few.py`](scripts/create_mimic4_few.py). It's
self-contained and deterministic: with a fixed `--seed`, re-running it
yields the byte-identical file (verified). Run it after the
[download](#-download) above:

```bash
cd ~/plant/scripts
python3 create_mimic4_few.py
```

It downloads the source if needed (via `untar_xxx(XURLs.MIMIC4_DEMO)`),
writes the two output files listed in [Files produced](#files-produced),
and prints the counts this README reports so you can confirm an exact
match:

```
Unique codes in training split: 7939
Selected 203 target codes with train freq in [3, 7]
Notes with >=1 target code: 2134 / 122279  (label_scope=all)
Capped to 800 notes (--max_notes=800, --seed=42)
Target codes still represented: 202 / 203
train    516
test     167
val      117
Wrote 800 notes to .../mimic4_demo/mimic4_icd10_few.csv
Wrote 203 target codes to .../mimic4_demo/mimic4_icd10_few_codes.txt
```

The defaults reproduce the dataset described here. The relevant flags:

| Flag | Default | Effect |
| --- | --- | --- |
| `--min_freq` / `--max_freq` | `3` / `7` | Target ("rare") codes are those with **training** frequency in `[min_freq, max_freq]`. This is the main knob — e.g. `--min_freq=7 --max_freq=7` for the freq-7-only band. |
| `--max_notes` | `800` | Cap kept notes at ~this many via a deterministic, split-proportional subsample. `0` disables the cap (keeps every note with a target code — 2,134 at the default range). |
| `--label_scope` | `all` | `all` (default) keeps the full original labels (**3,343** codes) — the extreme-multi-label task this experiment uses. `targets` strips each note's labels to only the target codes (a ~200-code single-label variant) — see [Why keep the full labels](#why-keep-the-full-labels-and-what-few-shot-means-here). |
| `--seed` | `42` | Seed for the `--max_notes` subsample; fixed value ⇒ reproducible file. |
| `--source_url` / `--csv_name` / `--output_name` | `XURLs.MIMIC4_DEMO` / `mimic4_icd10_full` / `mimic4_icd10_few` | Source download, source basename, and output basename. |

## 📦 **Resulting dataset: MIMIC-IV-few**

Measured directly on the produced `mimic4_icd10_few.csv`:

| Quantity | MIMIC-IV (source) | **MIMIC-IV-few** |
| --- | --- | --- |
| Rows (notes) | 122,279 | **800** (capped) |
| Split (train / val / test) | 89,099 / 13,378 / 19,802 | **516 / 117 / 167** |
| Distinct labels in the CSV | 7,942 | **3,343** (full labels kept) |
| Labels/doc | median 14 (mean 15.65) | **median 17** (mean 18.58, max 57) |
| Target codes (selected the notes) | — | **203** (train freq 3–7); 202 present after the cap |
| Eval buckets: zero / few / head | — | **358 / 2,710 / 275** codes (by training freq — see [Evaluation](#-evaluation-zero--few--head-buckets)) |

The label space is the **full 3,343-code** extreme-multi-label space; the
203 target codes only selected the notes. Few-shot difficulty is measured
by [bucketing at evaluation](#-evaluation-zero--few--head-buckets) — see below.

### Why keep the full labels (and what "few-shot" means here)

By default (`--label_scope=all`) each note keeps **all** its original ICD-10
codes, so the label space is the full **3,343** codes (median **17**/note) —
a real extreme-multi-label task. The 203 target codes only *selected* the
notes; "few-shot" is then a property measured **per code at evaluation**,
not a restriction baked into the label space.

<details>
<summary>Why we keep full labels instead of stripping to the target codes</summary>

- **"Few-shot" is a per-code property, recovered at eval.** Every code in
  the 3,343-label space is bucketed by its **training frequency** into
  zero-shot (0 occurrences), few-shot (1–7), or head (≥8), and each bucket
  is scored separately (see [Evaluation](#-evaluation-zero--few--head-buckets)).
  This measures zero- and few-shot generalization directly *without*
  amputating the task.
- **Why not strip to the target codes?** An earlier variant did exactly
  that (`--label_scope=targets`): it stripped every note to its ≤2 target
  codes, giving a ~200-code space with median **1** true code/note. That
  turned an extreme-multi-label problem into a near-single-label one — and
  in that regime PLANT (Stage 5) actually **lost** to the baseline
  (Stage 2) on top-1 accuracy, because the stripping removed the dense
  label co-occurrence structure PLANT's label-attention is built to
  exploit. Keeping full labels puts the task back in PLANT's wheelhouse;
  the [results](#-results-plant-vs-baseline) bear this out.
- **`num_few_codes`** records each note's target-code count (median 1,
  max 2) — provenance for *why* each note was kept, separate from its full
  label set. The metadata columns (`icd10_diag`, `num_targets`, …) reflect
  the original coding and are left untouched — the pipeline only reads
  `text` / `labels` / `is_valid` / `split`.

</details>

### Files produced

Both written into `~/.xcube/data/mimic4_demo/` alongside the source:

| File | Size | Contents |
| --- | --- | --- |
| `mimic4_icd10_few.csv` | ~9 MB | The 800 kept notes — all 15 original columns preserved (full `labels`, 3,343-code space, `--label_scope=all`), plus `num_few_codes`. `text,labels,is_valid,split` are all intact, so it's format-compatible with the pipeline's `text,labels,is_valid` expectation. |
| `mimic4_icd10_few_codes.txt` | <1 KB | The 203 target codes, one per line, sorted — the codes whose train frequency (3–7) selected the notes. (All 203 are listed; 1 has no surviving note after the cap.) Not the label space — that's the full 3,343. |

Neither file is committed to git — they're regenerated from the download
by [`scripts/create_mimic4_few.py`](scripts/create_mimic4_few.py) (see
[Reproducing the dataset](#-reproducing-the-dataset)) and live in the
`~/.xcube` data cache, which is per-box, same as every other `untar_xxx`
download.

## 🎌 **Evaluation: zero / few / head buckets**

Because we keep the full 3,343-code label space, "few-shot performance"
is measured **per code**, by how much *training* signal that code had.
Every code is assigned to one of three buckets by its
**training-set frequency** (computed on `train` + `val`, i.e. every row
with `is_valid == False` — the same rows the model learns from), and
each bucket is scored **separately**:

| Bucket | Definition (training frequency) | # codes | True codes / eval note (median) | Headline metric |
| --- | --- | --- | --- | --- |
| **zero-shot** | **0** occurrences — never seen in training | **358** | ~2 | P@1 |
| **few-shot** | **1–7** occurrences (`< rare_threshold`) | **2,710** | ~6 | P@5 |
| **head** | **≥ 8** occurrences (`≥ rare_threshold`) | **275** | ~9 | P@8 |
| overall | all 3,343 | 3,343 | 17 | P@15 |

This is the crux of the experiment: the baseline and PLANT see the
**identical** training data and label space, and we ask *where* on the
frequency spectrum each model earns its predictions. The headline metric
per bucket is **precision@(median true codes/note)** — a threshold-free
retrieval score at the natural cutoff for that bucket, so we're not
rewarding a model for padding predictions past the number of codes a note
actually has. The pipeline also emits recall@k and F1 per bucket.

<details>
<summary>Why buckets by training frequency, why precision@median, and where the buckets come from</summary>

- **Zero-shot codes are genuinely unseen but still in the output space.**
  The label binarizer is fit on the full train+val+test frame, so all
  3,343 codes are columns the model *can* predict — but 358 of them have
  **zero** training rows. A code the baseline never saw in training it
  essentially cannot rank; a label-attention model like PLANT can, via the
  code's text/description. That contrast is the zero-shot bucket.
- **Frequency is computed train-only** (`is_valid == False`). If we
  counted eval occurrences too, a code with 0 training rows but a few test
  rows would look "few-shot" and the zero-shot bucket would leak. The
  metric code computes `bucket_freq` from the non-valid rows exactly for
  this reason.
- **Precision@median, not a fixed k.** A note carries ~17 codes overall
  but only ~2 zero-shot ones, so scoring the zero bucket at P@15 would
  divide by mostly-irrelevant slots. Using each bucket's own median
  true-count (zero≈2→P@1, few≈6→P@5, head≈9→P@8, overall≈17→P@15) keeps
  the cutoff honest per bucket.
- **The split is written by Stage 2** into `labels_partition.json`
  (`zero` / `rare` / `not_rare`, where `rare` = zero ∪ few and the head
  bucket is `not_rare`), keyed off `--rare_threshold`. Stages 3–5 read it,
  so the buckets are **identical** across the baseline and PLANT runs.

</details>

## 🤖 **Choosing a Base Model**

PLANT is model-agnostic — any SOTA causal LLM works as the Stage 1 base
(see [README.md](README.md)). This few-shot experiment **defaults to
Mistral-7B-Instruct-v0.3** (what [GPU Box Setup](#️-gpu-box-setup)
installs), but the same three models validated end-to-end in
[README_EURLEX_WIKI10.md](README_EURLEX_WIKI10.md#-choosing-a-base-model)
apply here unchanged — base-model choice is independent of the dataset:

| Model | HF id | Params | Notes |
| --- | --- | --- | --- |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | **Default.** The model [README.md](README.md) uses for MIMIC-IV; nothing needed beyond the standard install in [GPU Box Setup](#️-gpu-box-setup). |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Gated on Hugging Face — accept Meta's license on the model page and use an authenticated HF token (`huggingface-cli login` or `HF_TOKEN`) before downloading. |
| Phi-3-small-8k-instruct | `microsoft/Phi-3-small-8k-instruct` | 7B | Size-matched to the other two (unlike the more commonly-used 3.8B Phi-3-mini). Loaded via `trust_remote_code=True` (already set in the training scripts); needs `flash_attn`, `tiktoken`, and `pytest` installed on the GPU box (`pip install flash_attn tiktoken pytest`) — see [Errors we hit](#errors-we-hit-and-the-fixes-reproducibility) item 3. |

> ℹ️ Llama-3.1-8B and Phi-3-small each need extra per-architecture setup
> (HF gating for Llama; `flash_attn` plus a chain of dtype / tokenizer /
> tied-embedding-save fixes for Phi-3-small). All of it is
> **dataset-independent** and documented in full in
> [README_EURLEX_WIKI10.md](README_EURLEX_WIKI10.md#-choosing-a-base-model)
> and its porting-note subsections — nothing there changes for
> MIMIC-IV-few. To use one, swap `MODEL` and re-source
> `setup_plant_names.sh` when the pipeline runs.

## 📝 **Naming Conventions**

Same mechanism as [README.md](README.md#-naming-conventions--consistency)
and the other companion READMEs — `setup_plant_names.sh` derives every
stage's Hub id and output dir from the dataset name and base model, so
the five pipeline commands stay consistent:

```bash
source setup_plant_names.sh "<dataset>_full" "<model-hf-id>"
```

Hub ids default to the `deb101` namespace. To reproduce under your own
Hugging Face account, `export HF_USERNAME="your-hf-username"` before
sourcing (the script reads `HF_USERNAME=${HF_USERNAME:-"deb101"}`), and
every `STAGE*_HUB_ID` is generated under your account instead.

For MIMIC-IV-few the dataset argument is **`mimic4_icd10_few_full`** — the
`_full`-suffixed form, exactly parallel to README.md's
`mimic4_icd10_full`. This matters: the script sets
`DATASET_SHORT=${DATASET%_*}`, so `mimic4_icd10_few_full` → short
`mimic4_icd10_few`, whereas a bare `mimic4_icd10_few` would strip to
`mimic4_icd10` and collide with the full-MIMIC run's names.

> ⚠️ The `_full` name means the pipeline reads `mimic4_icd10_few_full.csv`,
> but [the builder](#-reproducing-the-dataset) writes `mimic4_icd10_few.csv`.
> Symlink one to the other once per box — the same `_full` convention as
> [README_EURLEX_WIKI10.md](README_EURLEX_WIKI10.md#️-one-time-dataset-prep-the-_full-symlink):
> ```bash
> ln -sf mimic4_icd10_few.csv ~/.xcube/data/mimic4_demo/mimic4_icd10_few_full.csv
> ```

`setup_plant_names.sh` derives a short, collision-free tag per model:

| Model | `MODEL_SHORT` |
| --- | --- |
| Mistral-7B-Instruct-v0.3 | `mistral7b` |
| Llama-3.1-8B-Instruct | `llama8b` |
| Phi-3-small-8k-instruct | `phi3small` |

Example — MIMIC-IV-few + the default Mistral:

```bash
source setup_plant_names.sh "mimic4_icd10_few_full" "mistralai/Mistral-7B-Instruct-v0.3"
```

prints (and exports) exactly:

| Stage | Hub ID | Output Dir |
| --- | --- | --- |
| 1 | `deb101/mistral-7b-mimic4_icd10_few-adapt` | `mistral7b_mimic4_icd10_few` |
| 2 | *(same as Stage 1)* | `mimic4_icd10_few_classify_mistral7b` |
| 3 | `mistralai/Mistral-7B-Instruct-v0.3` (tokenizer only) | `mimic4_icd10_few_l2rboot_mistral7b` |
| 4 | *(same as Stage 1)* | `mimic4_icd10_few_l2rtrain_mistral7b` |
| 4 (L2R output) | `deb101/mistral-7b-mimic4_icd10_few-adapt-l2r` | — |
| 5 | *(same as Stage 4 L2R output)* | `mimic4_icd10_few_plantclassify_mistral7b` |

Swap the model id for one of the other two in
[Choosing a Base Model](#-choosing-a-base-model) and every name follows
the same pattern — `DATASET_NAME` (`mimic4_icd10_few_full`) is exported
for use as `--data` in every stage.

## 🚀 **Running All 5 Stages**

The standard dense Stage 1→5 PLANT pipeline (see
[README.md](README.md#-plant-pipeline)), run **once** on MIMIC-IV-few —
no separate smoke sample, since the dataset is already small. The block
below is the **exact** set of commands that produced
[the results](#-results-plant-vs-baseline) on Mistral-7B, kept in
**skip-push, local-chaining** mode so you can fire combos fast without
pushing multi-GB checkpoints to the Hub.

> ✅ **Executed end-to-end on MIMIC-IV-few (Mistral-7B, H100 80GB).** All
> five stages ran to completion; Stage 5 (PLANT) beats Stage 2 (baseline)
> in every bucket ([Results](#-results-plant-vs-baseline)). The block
> reflects the parameters that actually worked — including two fixes over
> the first attempt (a Stage 4 OOM and a Stage 5 unsupported flag),
> documented in [Errors we hit](#errors-we-hit-and-the-fixes-reproducibility).

**Prerequisites** (once per box, both covered above):
1. [GPU Box Setup](#️-gpu-box-setup) done, and the dataset built
   (`create_mimic4_few.py`).
2. The `_full` symlink from [Naming Conventions](#-naming-conventions):
   ```bash
   ln -sf mimic4_icd10_few.csv ~/.xcube/data/mimic4_demo/mimic4_icd10_few_full.csv
   ```

### The commands

One generic block. Only the `MODEL` line changes to fire a different
combo; everything else is derived. Run the stages **in order 1→5** —
Stage 3 reads `labels_partition.json`, which only Stage 2 writes.

```bash
cd ~/plant/scripts

# --- auth + runtime env (all stages read HF_TOKEN even with --skip_push_to_hub) ---
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"   # from `huggingface-cli login`
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export TOKENIZERS_PARALLELISM=False
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # matters for Stage 4 (see fixes below)

# --- pick the combo (change MODEL to fire a different one) ----------------
DATASET_BASE="mimic4_icd10_few"                    # fixed for this experiment
SOURCE_URL="XURLs.MIMIC4_DEMO"                      # fixed
MODEL="mistralai/Mistral-7B-Instruct-v0.3"         # or the Llama/Phi-3 id from "Choosing a Base Model"
RARE_THRESHOLD=8                                    # few/head boundary: freq <8 = few (∪ zero), >=8 = head
EP=6                                                # epochs for the trainable stages (2/4/5)
SEL="few_recall_at_5"                               # selection metric -- see notes below

source setup_plant_names.sh "${DATASET_BASE}_full" "$MODEL" "../tmp_few"
# -> DATASET_NAME=mimic4_icd10_few_full, DATASET_SHORT=mimic4_icd10_few, BASE_DIR=../tmp_few

STAGE1_LOCAL="$BASE_DIR/${SOURCE_URL#XURLs.}/$STAGE1_OUTPUT_DIR"
STAGE4_LOCAL="$BASE_DIR/${SOURCE_URL#XURLs.}/$STAGE4_OUTPUT_DIR"

# --- Stage 1: SOTA LLM fine-tuning ---------------------------------------
launches/launchLLM \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE1_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=1 \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 --skip_push_to_hub

# --- Stage 2: baseline classifier (also writes labels_partition.json) -----
launches/launchmultilabelclas \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE2_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-classify" --do_fresh_training \
    --num_train_epochs=$EP --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="$SEL" --rare_threshold="$RARE_THRESHOLD" \
    --label_coverage_frac=1.0 --skip_push_to_hub

# --- Stage 3: L2R bootstrap with MIG (needs Stage 2's labels_partition.json)
launches/launch_l2rboot \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" \
    --base_dir="$BASE_DIR" --output_dir="$STAGE3_OUTPUT_DIR" \
    --hub_model_id="$STAGE3_HUB_ID" --bs=8 --chnk_sz=200 --max_length=11776

# --- Stage 4: L2R attention transfer -------------------------------------
# NOTE: batch=2 x grad_accum=4 (effective 8). batch=8 OOMs here -- Stage 4 is
# a FULL 7B fine-tune (~77 GiB weights+AdamW) and the 3,343-label attention
# activation tips an 80GB card over. See "Errors we hit" below.
launches/launch_l2r_training \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" \
    --data_l2r_fname_prefix="$DATASET_SHORT" --max_length=512 \
    --base_dir="$BASE_DIR" --l2r_boot_dir="$STAGE3_OUTPUT_DIR" \
    --output_dir="$STAGE4_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=$EP --learning_rate=1e-4 --warmup_steps=0 \
    --metric_for_best_model="ndcg@25" \
    --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=4 --skip_push_to_hub

# --- Stage 5: PLANT classifier (hub_model_id = Stage 4's local L2R output) -
# NOTE: batch=8, and NO --gradient_accumulation_steps -- multilabel_classify.py
# does not accept that flag (only launchLLM / launch_l2r_training do). See fixes.
launches/launchmultilabelclas \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE5_OUTPUT_DIR" \
    --hub_model_id="$STAGE4_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-plantclassify" --do_fresh_training \
    --num_train_epochs=$EP --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="$SEL" --rare_threshold="$RARE_THRESHOLD" \
    --label_coverage_frac=1.0 --skip_push_to_hub
```

> ⚠️ `setup_plant_names.sh` uses a variable literally named `DATASET`
> internally for its first positional arg — and since it's `source`d, it
> would silently clobber a caller-side `$DATASET`. The block above sidesteps
> that by naming the caller variable `DATASET_BASE`, and uses the exported
> `$DATASET_SHORT` (`mimic4_icd10_few`) for Stage 4's
> `--data_l2r_fname_prefix` rather than reconstructing it by hand — same
> collision fix as [README_EURLEX_WIKI10.md](README_EURLEX_WIKI10.md#the-commands).

> ⚠️ **Phi-3 models only**: nothing changes in the commands above — the
> LoRA `target_modules` fix is automatic based on `model.config.model_type`
> (see [Choosing a Base Model](#-choosing-a-base-model)), not a flag you
> pass. Its other setup (flash-attn, dtype/save fixes) is install-time,
> covered in the EURLEX/Wiki README.

### Errors we hit, and the fixes (reproducibility)

The command block above is already fixed. On the **first** attempt two
stages failed (items 1–2 below); both fixes are baked into the block. A
third class of failure (item 3) only appears with **Phi-3-small**, which
needs two extra Python packages. All are recorded here so the parameter
choices aren't mysterious and so anyone re-running on different hardware or
a different base model knows which knobs are load-bearing.

<details>
<summary>1 — Stage 4 CUDA OOM at <code>batch=8</code>, and the batch=2 × accum=4 fix</summary>

First attempt ran Stage 4 with `--per_device_train_batch_size=8`
(carried over from the other stages). It OOM'd on the **first training
step**:

```
CUDA out of memory. Tried to allocate 26.12 GiB.
GPU 0 has a total capacity of 94.50 GiB of which 16.61 GiB is free.
```

**Why:** Stage 4 (L2R attention transfer) is a **full** Mistral-7B
fine-tune — weights + gradients + AdamW optimizer state already sit around
~77 GiB before activations — and the label-attention activation over the
**3,343-code** space adds a ~26 GiB tensor at `batch=8`. That overflows
even an 80–94 GB card. (Stage 2's baseline classifier is lighter and ran
fine at `batch=8`, which is why only Stage 4 hit this.)

**Fix (in the block):** `--per_device_train_batch_size=2
--gradient_accumulation_steps=4` — identical effective batch of 8, but the
activation tensor is ~4× smaller — plus
`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to curb
fragmentation. Stage 4 then trained cleanly (~1h40 for 6 epochs on an
H100). If your card is smaller, drop to `batch=1 --gradient_accumulation_steps=8`.

> ⚠️ The launch wrappers print a success line and `unset` the exit code, so
> a failed training step does **not** surface as a non-zero exit. When
> chaining stages in a script, grep each stage's log for `CUDA out of
> memory` / `An error occurred during model training` rather than trusting
> `$?`, or Stage 5 will run against a Stage 4 model that was never saved.

</details>

<details>
<summary>2 — Stage 5 <code>unrecognized arguments: --gradient_accumulation_steps</code></summary>

Having added `--gradient_accumulation_steps=4` to Stage 4, the natural
next move was to add it to Stage 5 too. Stage 5 died instantly:

```
multilabel_classify.py: error: unrecognized arguments: --gradient_accumulation_steps=4
```

**Why:** `multilabel_classify.py` (Stages 2 and 5) simply has no
`--gradient_accumulation_steps` argument — only `launchLLM` (Stage 1) and
`launch_l2r_training` (Stage 4) do.

**Fix (in the block):** Stage 5 runs at `--per_device_train_batch_size=8`
with **no** grad-accum flag — matching Stage 2 exactly, which also keeps
the baseline-vs-PLANT comparison apples-to-apples. Stage 5 is not a full
fine-tune of the same shape as Stage 4, so `batch=8` fits comfortably
(~36 GiB).

</details>

<details>
<summary>3 — Phi-3-small only: missing <code>tiktoken</code> and <code>pytest</code> at Stage 1</summary>

`microsoft/Phi-3-small-8k-instruct` loads its architecture from
Microsoft's own `modeling_phi3_small.py` via `trust_remote_code=True`, and
that custom code pulls in two packages the base install doesn't have.
Stage 1 died at model/tokenizer load:

```
ImportError: This modeling file requires the following packages that were
not found in your environment: pytest. Run `pip install pytest`
```

and, separately, the tiktoken-based `Phi3SmallTokenizer` needs `tiktoken`.

**Why:** transformers' `check_imports` scans the remote modeling file's
import list and hard-fails if any are absent — Microsoft's file lists
`pytest`. And Phi-3-small tokenizes with tiktoken rather than a
SentencePiece/BPE `tokenizer.json`, so `tiktoken` must be importable.
Neither is needed for Mistral-7B or Llama-3.1-8B.

**Fix:** install both once before running the Phi-3-small combo:

```bash
pip install tiktoken pytest
```

(The other Phi-3-small quirks — fused q/k/v projection names, tied
`embed_tokens`/`lm_head` forcing `save_safetensors=False`, and the
RotaryEmbedding fp32 guard for flash-attn — are already handled in the
committed `multilabel_classify.py` / `learning2rank.py` / `finetune_llm.py`
code, so no flags change.)

</details>

### Notes on the parameter choices

- **`max_length` is MIMIC-specific.** Training stages (1/2/4/5) use `512`,
  but **Stage 3 uses `11776`** (not EURLEX's `1024`) — discharge summaries
  are far longer than Eurlex/Wiki docs, and this matches
  [README.md](README.md#-plant-pipeline)'s full-MIMIC Stage 3. Stage 3
  makes `ceil(unique_labels / chnk_sz)` passes, so with the long
  `max_length` it's the slowest CPU-bound stage; budget for it.
- **Epochs**: Stage 1 at `1`, Stages 2/4/5 at `6`. The Eurlex/Wiki runs
  found 1 epoch on 2/4/5 gave a false "PLANT loses"
  ([their note](README_EURLEX_WIKI10.md#️-stage-4-needs-enough-epochs--1-isnt-always-sufficient));
  README.md's full-MIMIC uses `5`–`7`. With only 516 training notes here,
  epochs are cheap, and at 6 both the baseline's `few_recall_at_5` and
  Stage 4's `ndcg@25` were still inching up — 6 was a reasonable stop, not
  a converged ceiling, so more epochs may widen the margins further.
- **`metric_for_best_model=few_recall_at_5`.** Model selection tracks the
  metric the experiment is *about* — recall of few-shot codes in the top 5
  — computed **identically** for Stage 2 and Stage 5, so checkpoint
  selection can't bias the comparison. It's threshold-free (unlike F1,
  which for the baseline sits near zero because it rarely crosses its
  decision threshold). The full per-bucket P@k / R@k / F1 suite is what we
  actually read for the verdict ([Results](#-results-plant-vs-baseline));
  `few_precision_at_5` or `zero_precision_at_1` are reasonable
  selection swaps if you care about a different bucket.
- **`--rare_threshold=8` is load-bearing now.** With full labels kept, it
  sets the **few/head boundary**: codes with training frequency `< 8` are
  the "rare" bucket (which, minus the zero-frequency codes, is the
  few-shot bucket), `≥ 8` are head. Stage 2 writes the resulting split to
  `labels_partition.json` and Stages 3–5 reuse it, so all stages bucket
  identically. Change it and you redefine what "few-shot" vs "head" means
  — see [Evaluation](#-evaluation-zero--few--head-buckets).

### Firing other combos

Swap the `MODEL` line for `meta-llama/Llama-3.1-8B-Instruct` or
`microsoft/Phi-3-small-8k-instruct` (see
[Choosing a Base Model](#-choosing-a-base-model)) and re-run the whole
block — every name re-derives from `setup_plant_names.sh`, and the
`../tmp_few` base dir keeps each combo's outputs separate by model-short
tag. Those two models need their extra install-time setup (HF gating;
flash-attn + fixes for Phi-3-small) first, all documented in the
EURLEX/Wiki README.

## 📈 **Results: PLANT vs. baseline**

> ✅ **All three combos done: Mistral-7B, Llama-3.1-8B, and Phi-3-small
> (below).** PLANT wins zero-shot / head / overall on all three; the
> **few-shot** win holds for Mistral and Llama but **not** Phi-3-small —
> see that subsection and the cross-model takeaway at the end.

### Mistral-7B-Instruct-v0.3

6 epochs, evaluated on the 167 test notes;
best checkpoint by `few_recall_at_5` (epoch 6 for both models).
**Stage 2** is the non-PLANT baseline classifier; **Stage 5** is PLANT
(same base LLM, plus the Stage 3→4 label-attention prior). Both see the
identical training data, label space, and buckets.

**Headline — precision@(bucket median) per bucket** (the natural cutoff
from [Evaluation](#-evaluation-zero--few--head-buckets)):

| Bucket (codes) | metric | Stage 2 (baseline) | Stage 5 (PLANT) | Lift |
| --- | --- | --- | --- | --- |
| **zero-shot** (358) | P@1 | 0.0055 | **0.3087** | ✅ **~56×** |
| **few-shot** (2,710) | P@5 | 0.0545 | **0.1160** | ✅ ~2.1× |
| **head** (275) | P@8 | 0.2766 | **0.3083** | ✅ +11% |
| **overall** (3,343) | P@15 | 0.2221 | **0.2460** | ✅ +11% |

**PLANT wins every bucket, at every cutoff.** The gains are largest
exactly where the experiment set out to probe — codes with little or no
training signal.

<details>
<summary>Full per-bucket scorecard (P@k / R@k, epoch 6)</summary>

| Bucket | Metric | Stage 2 | Stage 5 |
| --- | --- | --- | --- |
| **zero** | P@1 | 0.0055 | **0.3087** |
| | P@5 | 0.0073 | **0.1528** |
| | R@8 | 0.0174 | **0.3234** |
| **few** | P@1 | 0.0805 | **0.1544** |
| | P@5 | 0.0545 | **0.1160** |
| | R@8 | 0.0721 | **0.1296** |
| **head** | P@1 | 0.4223 | **0.5094** |
| | P@5 | 0.3184 | **0.3643** |
| | P@8 | 0.2766 | **0.3083** |
| **overall** | P@1 | 0.4289 | **0.4509** |
| | P@5 | 0.3197 | **0.3502** |
| | P@15 | 0.2221 | **0.2460** |
| | R@15 | 0.1869 | **0.2135** |

Stage 4's ranking quality climbed steadily over its 6 epochs
(`ndcg@25`: 0.530 → 0.559 → 0.572 → 0.575 → 0.579 → 0.579), and full-list
`ndcg` sat near ceiling (~0.994) — a well-trained label-attention prior is
what feeds Stage 5.

</details>

**What the numbers say:**

- **Zero-shot is the headline.** The baseline sits at the floor
  (P@1 ≈ 0.006): a code it never saw in training it essentially cannot
  surface as a top pick. PLANT reaches P@1 ≈ 0.31 by ranking codes through
  their text/attention prior rather than learned frequency — a **~56×**
  gap. This is the core PLANT thesis, shown directly.
- **Few-shot roughly doubles** across P@1 / P@5 / R@8 (e.g. P@5 0.055 →
  0.116).
- **Head codes also improve** (+11–21%) — PLANT lifts the rare buckets
  **without** sacrificing the common ones, which is the failure mode a
  rare-code-focused method would risk.
- **F1 caveat.** Thresholded F1 is near zero for the baseline
  (`f1_micro` = 0.0007) because it rarely crosses its decision threshold;
  PLANT is higher (0.0287) but still low. F1 depends on a global
  threshold, so the **threshold-free P@k / R@k** are the fair,
  interpretable comparison here — and PLANT wins those cleanly.

> ℹ️ Numbers are from a single seed; treat the small head/overall
> improvements as directional and the large zero/few-shot gaps as the
> robust finding. `few_recall_at_5` was still rising at epoch 6
> (0.030 → 0.102 for PLANT), so longer training would likely widen the
> few/zero-shot margins rather than close them.

### Llama-3.1-8B-Instruct

Identical protocol to Mistral above (6 epochs, 167 test notes, best
checkpoint by `few_recall_at_5`, Stage 2 baseline vs. Stage 5 PLANT on the
same data/labels/buckets). The one config difference is the base model
(`meta-llama/Llama-3.1-8B-Instruct`); Stage 4 used the same OOM-safe
`batch=2 × grad_accum=4`, everything else matches. **Checkpoint note:**
`load_best_model_at_end` picked Stage 2's **epoch 3** (its `few_recall_at_5`
peaked there, then overfit down) and Stage 5's epoch 6 — each stage is
reported at its own best checkpoint, as the selection rule dictates.

**Headline — precision@(bucket median) per bucket:**

| Bucket (codes) | metric | Stage 2 (baseline) | Stage 5 (PLANT) | Lift |
| --- | --- | --- | --- | --- |
| **zero-shot** (358) | P@1 | 0.0075 | **0.2839** | ✅ **~38×** |
| **few-shot** (2,710) | P@5 | 0.0087 | **0.1340** | ✅ **~15×** |
| **head** (275) | P@8 | 0.1742 | **0.3423** | ✅ ~2.0× |
| **overall** (3,343) | P@15 | 0.0985 | **0.2695** | ✅ ~2.7× |

**PLANT wins every bucket, at every cutoff** — the same story as Mistral.
Llama's baseline peaks earlier (epoch 3) and is a touch weaker than
Mistral's at the head (P@8 0.174 vs. 0.277), but PLANT's numbers are
comparable-to-better than Mistral's (head P@8 0.342 vs. 0.308), and the
zero/few-shot lifts remain large (~38× / ~15×).

<details>
<summary>Full per-bucket scorecard (P@k / R@k; S2 @ epoch 3, S5 @ epoch 6)</summary>

| Bucket | Metric | Stage 2 | Stage 5 |
| --- | --- | --- | --- |
| **zero** | P@1 | 0.0075 | **0.2839** |
| | P@5 | 0.0085 | **0.1442** |
| | R@8 | 0.0178 | **0.3025** |
| **few** | P@1 | 0.0187 | **0.2229** |
| | P@5 | 0.0087 | **0.1340** |
| | R@8 | 0.0128 | **0.1510** |
| **head** | P@1 | 0.2466 | **0.5592** |
| | P@5 | 0.1970 | **0.4057** |
| | P@8 | 0.1742 | **0.3423** |
| **overall** | P@1 | 0.2453 | **0.5093** |
| | P@5 | 0.1726 | **0.4002** |
| | P@15 | 0.0985 | **0.2695** |
| | R@15 | 0.0857 | **0.2338** |

Stage 4's ranking quality was strong and stable across its 6 epochs
(`ndcg@25`: 0.735 → 0.766 → 0.778 → 0.779 → 0.780 → 0.780) — a higher
plateau than Mistral's (~0.58), which shows up as the stronger PLANT head.
Thresholded `f1_micro` is again near the floor for both (baseline 0.0033,
PLANT 0.0530), so the threshold-free P@k / R@k remain the fair comparison.

</details>

> ℹ️ Single seed, same caveats as Mistral. `few_recall_at_5` for PLANT was
> still rising at epoch 6 (0.049 → 0.119), so the few/zero-shot margins
> would likely widen with longer training.

### Phi-3-small-8k-instruct

Same protocol again (6 epochs, 167 test notes, best checkpoint by
`few_recall_at_5`, Stage 2 vs. Stage 5, same data/labels/buckets, Stage 4
at `batch=2 × grad_accum=4`). Base model
`microsoft/Phi-3-small-8k-instruct`, loaded via `trust_remote_code`; this
combo needed two extra install-time deps (`tiktoken`, `pytest` — see
[Errors we hit](#errors-we-hit-and-the-fixes-reproducibility)).
**Checkpoint note:** `load_best_model_at_end` selected Stage 2's **epoch 5**
and Stage 5's epoch 6 (each stage's own `few_recall_at_5` peak).

**Headline — precision@(bucket median) per bucket:**

| Bucket (codes) | metric | Stage 2 (baseline) | Stage 5 (PLANT) | Outcome |
| --- | --- | --- | --- | --- |
| **zero-shot** (358) | P@1 | 0.0112 | **0.1118** | ✅ ~10× |
| **few-shot** (2,710) | P@5 | **0.0673** | 0.0581 | ❌ **−14% — PLANT loses** |
| **head** (275) | P@8 | 0.1716 | **0.2901** | ✅ ~1.7× |
| **overall** (3,343) | P@15 | 0.1076 | **0.2141** | ✅ ~2.0× |

**Phi-3-small is the exception: PLANT wins zero-shot, head, and overall,
but *loses* the few-shot bucket** — and it loses it at every cutoff
(P@1 0.109 → 0.077, P@5 0.067 → 0.058, R@8 0.085 → 0.064). This is a real
effect, not a training fault, and it's worth stating plainly:

- Phi-3's **baseline is unusually strong on few-shot** to begin with
  (few P@5 0.067, vs. Mistral 0.055 and Llama 0.009) — a higher bar.
- Phi-3's **PLANT head barely learned the few bucket**: the selection
  metric `few_recall_at_5` plateaued at just **0.048**
  (0.031 → 0.036 → 0.039 → 0.045 → 0.048 → 0.048), against Mistral 0.102
  and Llama 0.119 at their epoch-6 peaks. The checkpoint that gets selected
  is weak exactly where the few bucket lives.

<details>
<summary>Full per-bucket scorecard (P@k / R@k; S2 @ epoch 5, S5 @ epoch 6)</summary>

| Bucket | Metric | Stage 2 | Stage 5 |
| --- | --- | --- | --- |
| **zero** | P@1 | 0.0112 | **0.1118** |
| | P@5 | 0.0077 | **0.0534** |
| | R@8 | 0.0194 | **0.1274** |
| **few** | P@1 | **0.1093** | 0.0770 |
| | P@5 | **0.0673** | 0.0581 |
| | R@8 | **0.0853** | 0.0639 |
| **head** | P@1 | 0.3130 | **0.5093** |
| | P@5 | 0.2045 | **0.3399** |
| | P@8 | 0.1716 | **0.2901** |
| **overall** | P@1 | 0.3267 | **0.4807** |
| | P@5 | 0.1930 | **0.3205** |
| | P@15 | 0.1076 | **0.2141** |
| | R@15 | 0.1104 | **0.1831** |

Bold = winner per row; the **few** rows are the only ones the baseline
wins on any model. Stage 4's `ndcg@25` plateaued Mistral-like
(0.533 → 0.552 → 0.561 → 0.567 → 0.570 → 0.571, ~0.57 — well below Llama's
~0.78). Thresholded `f1_micro` near the floor for both (baseline 0.0031,
PLANT 0.0195), so P@k / R@k remain the fair comparison.

</details>

> ℹ️ Single seed. **Takeaway across all three models:** the head / zero-shot
> / overall PLANT gains are the robust finding; the *few-shot* win is
> **not** universal — it tracks how strong the base model's own few-shot
> baseline is and how well its Stage-5 head fits the few bucket. Mistral and
> Llama win few-shot handily; Phi-3-small does not.
>
> One honest caveat on that loss: Phi-3's `few_recall_at_5` was still the
> lowest of the three and only *just* flattening at epoch 6 (0.048), so part
> of the gap may be undertraining rather than a ceiling — exactly the "1
> epoch gave a false *PLANT loses*" failure mode flagged in
> [Notes on the parameter choices](#notes-on-the-parameter-choices). A
> longer Phi-3-small run would be the clean way to tell an undertrained head
> from a genuine base-model effect; at a fixed 6 epochs, PLANT loses few-shot
> here.

## 🧪 **Training-free LLM baseline (context)**

The [Results](#-results-plant-vs-baseline) above show PLANT beating the
*trained* Stage 2 baseline by huge margins on rare/zero-shot codes. But that
baseline is a classifier — it **cannot** rank a code it never saw in
training, so some of that margin is structural, not a PLANT effect. To put
the numbers in honest context we add the third system from
[What we compare](#-what-we-compare-and-two-meanings-of-zerofew): the **same
base LLM, off the shelf, with no MIMIC training at all**, used as an
extract-then-link coder. This answers *"how much of PLANT's rare-code
advantage needs training in the first place?"*

> 🔁 **Reminder (two axes):** below, **zero/few‑shot** in a *column header*
> is the **prompting regime** (0 or 2 in-context examples given to the
> extractor); **zero/few/head** in a *row* is the **evaluation bucket** (a
> code's training frequency). See
> [What we compare](#-what-we-compare-and-two-meanings-of-zerofew).

### Method: extract → link → score

Fully reproducible via
[`scripts/train_free_baseline.py`](scripts/train_free_baseline.py) (driver:
[`scripts/run_train_free_baseline.sh`](scripts/run_train_free_baseline.sh)).
Per held-out note:

1. **Extract** — prompt an off-the-shelf **Instruct** model
   (`mistralai/Mistral-7B-Instruct-v0.3`, `meta-llama/Llama-3.1-8B-Instruct`)
   to list every diagnosis and procedure as short clinical phrases. The
   prompt contains **no target-label list** — only the note (plus, in the
   **few-shot** regime, 2 hand-authored worked examples). Truly
   training-free: raw instruct checkpoints, no Stage 1–5 weights.
2. **Link** — embed every mention and all **3,343 target-code descriptions**
   with **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) and take
   cosine similarity. Descriptions come from MIMIC-IV's own `d_icd_*`
   dictionaries (auto-downloaded from PhysioNet's **open-access** demo →
   **100% coverage** of both ICD-10-CM diagnoses and ICD-10-PCS procedures).
   Linking is **constrained to the 3,343 targets**, so the candidate space
   matches Stages 2/5 exactly.
3. **Score** — build a **dense** `(167 × 3343)` matrix where
   `s[note, code] = max over the note's mentions of cosine(mention, code_desc)`,
   then feed it (as "probs") through a **faithful copy** of the PLANT eval's
   bucketed `precision@k` (`bucketed_metrics`, mirroring
   `multilabel_classify.py`'s `compute_metrics_with_labels`). Same buckets,
   same per-note averaging → **directly comparable** to Stage 2 / Stage 5.

### The commands

Prereq: the [5-stage run](#-running-all-5-stages) must have produced
`labels_partition.json` (the bucket definitions) and
[`create_mimic4_few.py`](#-reproducing-the-dataset) the MIMIC-IV-few CSV.
The ICD-10 `d_icd_*` description dictionaries download automatically from
PhysioNet's open-access demo on first run.

```bash
cd ~/plant/scripts

# one-time: chat templates need jinja2>=3.1 (the pinned env ships 3.0.3)
pip install -U "jinja2>=3.1"

# --- auth + runtime env -------------------------------------------------
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"   # from `huggingface-cli login`
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export TOKENIZERS_PARALLELISM=False
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- all four combos at once: 2 models x {zero, few}, ~30-45 min each ----
# writes ./train_free_cache/{ments,metrics}_<model>_<mode>.json
bash run_train_free_baseline.sh

# --- or fire a single (model, regime) explicitly ------------------------
# --do all = extract (LLM mention extraction, GPU) then score (SapBERT link
#            + bucketed precision@k, fast). --mode = prompting regime.
python3 train_free_baseline.py --model "mistralai/Mistral-7B-Instruct-v0.3" --mode zero --do all
python3 train_free_baseline.py --model "mistralai/Mistral-7B-Instruct-v0.3" --mode few  --do all
python3 train_free_baseline.py --model "meta-llama/Llama-3.1-8B-Instruct"   --mode zero --do all
python3 train_free_baseline.py --model "meta-llama/Llama-3.1-8B-Instruct"   --mode few  --do all

# --- re-score without re-running the LLM (e.g. after editing the mention
#     filter) -- reuses the cached extractions ---------------------------
python3 train_free_baseline.py --model "mistralai/Mistral-7B-Instruct-v0.3" --mode zero --do score
```

Defaults (override via flags): `--data_csv`
`~/.xcube/data/mimic4_demo/mimic4_icd10_few.csv`, `--partition`
`~/plant/tmp_few_all/MIMIC4_DEMO/labels_partition.json`, `--cache_dir`
`./train_free_cache`. The linker (SapBERT) and off-the-shelf checkpoints
are fixed in the script.

### Results: training-free vs. trained

Precision@(bucket median). **S2** = trained baseline, **TF-0/TF-few** =
training-free zero-/few-shot prompting, **S5** = PLANT. Bold = best of the
four per row.

**Mistral-7B-Instruct-v0.3**

| Bucket | k | S2 (trained) | TF-0shot | TF-fewshot | S5 PLANT |
| --- | --- | --- | --- | --- | --- |
| **zero** | P@1 | 0.0055 | 0.2754 | 0.2575 | **0.3087** |
| **few** | P@5 | 0.0545 | 0.1485 | **0.1545** | 0.1160 |
| **head** | P@8 | 0.2766 | 0.2964 | 0.2927 | **0.3083** |
| **overall** | P@15 | 0.2221 | 0.2240 | 0.2271 | **0.2460** |

**Llama-3.1-8B-Instruct**

| Bucket | k | S2 (trained) | TF-0shot | TF-fewshot | S5 PLANT |
| --- | --- | --- | --- | --- | --- |
| **zero** | P@1 | 0.0075 | 0.2275 | 0.2335 | **0.2839** |
| **few** | P@5 | 0.0087 | **0.1389** | **0.1389** | 0.1340 |
| **head** | P@8 | 0.1742 | 0.2762 | 0.2837 | **0.3423** |
| **overall** | P@15 | 0.0985 | 0.2104 | 0.2228 | **0.2695** |

### What this establishes

1. **The trained baseline's rare/zero-shot floor is structural, not a
   knowledge gap.** A training-free LLM + entity-linking reaches zero-shot P@1
   ≈ 0.23–0.28, versus the trained Stage 2's ≈ 0.006–0.008 — a **~30–50×**
   gap in the exact buckets where PLANT also wins big. A classifier can't rank
   never-seen codes; an LLM's world knowledge can.
2. **Training-free is competitive with PLANT on rare codes — and *beats* it on
   few-shot.** Few-shot bucket P@5: Mistral **0.1545 (TF) > 0.1160 (PLANT)**;
   Llama **0.1389 (TF) > 0.1340 (PLANT)**. On the zero-shot bucket PLANT stays
   ahead but the margin is modest (0.31 vs 0.28; 0.28 vs 0.23).
3. **PLANT's uniquely training-dependent win is head + overall.** On common
   codes and global ranking the label-attention head adds what extraction
   can't: Mistral overall **0.2460 vs ≤0.227**; Llama head **0.3423 vs
   ≤0.284**, overall **0.2695 vs ≤0.223**. This is the honest headline —
   *PLANT's rare/zero-shot gains over the trained baseline are largely
   reproducible training-free; its head/overall gains are what training
   uniquely buys.*
4. **Prompting regime is a minor lever.** Few-shot examples help the
   few/overall buckets slightly and marginally hurt zero-shot — not decisive,
   and secondary to the bucket effects above.

> ℹ️ Single seed; off-the-shelf checkpoints; linking constrained to the 3,343
> targets. The dense max-similarity scoring gives every code a score from its
> best-matching extracted mention, so the training-free system is ranked by
> the identical bucketed `precision@k` as Stages 2/5 — no thresholding, a fair
> side-by-side. Phi-3-small was **not** run training-free (kept as the pure
> PLANT combo).

### Errors we hit (training-free)

<details>
<summary>1 — Few-shot <b>exemplar contamination</b> (Llama parroted the demo codes)</summary>

The first few-shot design built its worked examples from **train notes'
gold-code descriptions** (long formal ICD-10-PCS `long_title`s). Llama-3.1-8B
then **copied those verbatim into every test note** — e.g. "Insertion of
Infusion Device into Superior Vena Cava, Percutaneous Approach" appeared on
notes it had nothing to do with — spuriously boosting the exemplars' codes.
Mistral was largely robust, but the contamination invalidated Llama's
few-shot numbers.

**Fix (in the code):** `fewshot_exemplars()` now uses **2 hand-authored,
generic** snippets whose target lines are **short natural phrases** (not
long code descriptions), identical across models. Post-fix echo check:
zero occurrences of the parroted PCS string; the most-repeated mentions are
genuine recurring comorbidities (hypertension, CKD, atrial fibrillation).
</details>

<details>
<summary>2 — <code>apply_chat_template requires jinja2>=3.1</code>; and a stray preamble line</summary>

Two smaller ones: (a) the box shipped `jinja2 3.0.3`, so
`tokenizer.apply_chat_template` raised `ImportError` at the first prompt —
fixed with `pip install -U jinja2` (≥3.1). (b) Llama often prefixes its answer
with "Here is the list of distinct medical conditions and procedures", which
slipped past a naive length filter (the line is long); `is_junk()` now drops
such preambles by prefix/substring. Impact was negligible (one low-similarity
mention among ~40; re-scoring moved Llama-few head 0.2822→0.2837), but the
filter keeps the extraction clean.
</details>

> 🔁 **Reproduce:** `bash scripts/run_train_free_baseline.sh` runs all four
> combos (2 models × {zero, few}) on the same 167 notes; ~30–45 min per combo
> on one 80 GB+ GPU. Needs `labels_partition.json` (from the PLANT run) and the
> MIMIC-IV-few CSV; the `d_icd_*` dictionaries download automatically.
