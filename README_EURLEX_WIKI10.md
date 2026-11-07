# 🌱 PLANT on Eurlex-4K & Wiki10-31K

This is the Eurlex-4K / Wiki10-31K companion to [README.md](README.md). The
PLANT paradigm, architecture, and 5-stage pipeline are all unchanged — only
the dataset changes. Read [README.md](README.md) first for the concepts;
this file documents what's different when `--source_url="XURLs.EURLEX_4K"`
or `--source_url="XURLs.WIKI10_31K"` instead of `XURLs.MIMIC4_DEMO`.

Unlike [README_AMAZON3M.md](README_AMAZON3M.md), both datasets here have
**bounded** label spaces (thousands, not millions) — comparable in scale to
MIMIC-IV's ~8–17K ICD codes, not to Amazon-3M's 2.81M labels. That means
the plain dense Stage 1→5 commands apply directly; none of Amazon-3M's
candidate-generator / label-clustering machinery is needed for either
dataset. Where the two datasets differ from each other or from MIMIC-IV,
it's called out inline.

This file also covers a **3-model comparison** — Mistral-7B (already
validated on Amazon-3M), Llama-3.1-8B, and Phi-3-small-8k — instead of a
single base model. [README.md](README.md#-plant-pipeline) already
documents the conceptual shape of Stage 1→5; repeating that here for two
more datasets and three models would just be six copies of the same
commands. Instead this file skips straight to a runnable
[smoke test](#-smoke-testing-all-5-stages-on-a-sample-first), written
once, generically, against whatever `setup_plant_names.sh` exports for
the dataset/model pair you pick; see
[Choosing a Base Model](#-choosing-a-base-model) first.

> 💻 **Where this runs**: same setup as [README_AMAZON3M.md](README_AMAZON3M.md#-plant-on-amazon-3m) —
> assembled on the primary dev machine (CPU-only); actual training runs on
> a separate cloud GPU box. The box just clones this repository.

## 📋 Table of Contents

- [📊 Dataset: Eurlex-4K](#-dataset-eurlex-4k)
- [📊 Dataset: Wiki10-31K](#-dataset-wiki10-31k)
- [🤖 Choosing a Base Model](#-choosing-a-base-model)
- [⚙️ GPU Box Setup](#️-gpu-box-setup)
- [📝 Naming Conventions](#-naming-conventions)
- [🧪 Smoke-Testing All 5 Stages on a Sample First](#-smoke-testing-all-5-stages-on-a-sample-first)
  - [Building the sample](#building-the-sample)
  - [The commands](#the-commands)
- [🚀 Full-Scale Runs](#-full-scale-runs)
  - [Rare-label threshold, computed per dataset](#rare-label-threshold-computed-per-dataset)

## 📊 **Dataset: Eurlex-4K**

[Eurlex-4K](https://xcubebucket.s3.us-east-2.amazonaws.com/eurlex-4k/Eurlex-4k.tgz)
(`XURLs.EURLEX_4K` in `xcube/data/external.py`) is a multi-label legal
document classification dataset built from EU legislation: document text
as `text`, `;`-delimited EuroVoc descriptor labels as `labels`.

### 📥 Download Dataset

```python
# 🐍 Python REPL commands
from xcube.imports import *
p = untar_xxx(XURLs.EURLEX_4K)
print(f"Dataset location: {p}")
# Available files: eurlex-4k.csv  (columns: text, labels, is_valid)
# Also present (raw AttentionXML-format files, not used by this pipeline):
#   train.txt, train_texts.txt, train_labels.txt,
#   test.txt, test_texts.txt, test_labels.txt, rarelbs.csv
```

Measured directly on the downloaded `eurlex-4k.csv`:

| Quantity | Value |
| --- | --- |
| Rows (docs) | 19,314 (15,449 train / 3,865 test) |
| Unique labels | 3,956 |
| Median labels/doc | 6 (mean 5.32, max 24) |

### ⚠️ One-time dataset prep: the `_full` symlink

Same reasoning as [README_AMAZON3M.md's `_full` symlink section](README_AMAZON3M.md#️-one-time-dataset-prep-the-_full-symlink):
`bootl2r.py`'s Stage 3 derives the MIG-file prefix by stripping a trailing
`_full` from `--data`, and `get_data()` elsewhere reads `{source}/{data}.csv`
literally. Eurlex's raw CSV is `eurlex-4k.csv` — no `_full` suffix — so
symlink one into existence once per box:

```bash
python3 - <<'PY'
from xcube.imports import *
p = untar_xxx(XURLs.EURLEX_4K)
full = p / "eurlex-4k_full.csv"
if not full.exists():
    full.symlink_to("eurlex-4k.csv")
print(f"Ready: {full}")
PY
```

This step has already been done on the primary dev machine; **repeat it
once on the GPU box** after you pull the code there (the dataset itself is
downloaded separately via `untar_xxx`, it isn't part of the git sync).

## 📊 **Dataset: Wiki10-31K**

[Wiki10-31K](https://xcubebucket.s3.us-east-2.amazonaws.com/wiki10-31k/Wiki10-31K.tgz)
(`XURLs.WIKI10_31K` in `xcube/data/external.py`) is a multi-label tagging
dataset built from Wikipedia articles: article text as `text`,
`;`-delimited tag ids as `labels`.

### 📥 Download Dataset

```python
# 🐍 Python REPL commands
from xcube.imports import *
p = untar_xxx(XURLs.WIKI10_31K)
print(f"Dataset location: {p}")
# Available files: wiki10-31k.csv  (columns: text, labels, is_valid)
# Also present (raw AttentionXML-format files, not used by this pipeline):
#   train_raw_texts.txt, train_labels.txt, test_raw_texts.txt, test_labels.txt
```

Measured directly on the downloaded `wiki10-31k.csv`:

| Quantity | Value |
| --- | --- |
| Rows (docs) | 20,762 (14,146 train / 6,616 test) |
| Unique labels | 30,938 |
| Median labels/doc | 19 (mean 18.76, max 30) |

30,938 labels is larger than MIMIC-IV's ICD-code space but still nowhere
near Amazon-3M's 2.81M — a dense `(20,762 × 30,938)` boolean array is
~640MB (trivial), and a per-label `nn.Parameter(30938, 4096)` embedding
table is ~127M parameters (small next to a 7B backbone). Neither of
Amazon-3M's two blow-up failure modes applies here, so the dense track's
plain `MultiLabelBinarizer().fit_transform()` and per-label embedding
table are both fine as-is — no `--label_space=candidate` needed.

### ⚠️ Correction, found at full scale: dense label tensors need uint8, not float32

The "~640MB, trivial" estimate above is correct in principle but assumed
a boolean/byte representation; the actual code
(`multilabel_classify.py`'s Stage 2/5 dataset construction, the `else`
branch of the label-space dispatch) built the dense per-row label vector
as `torch.tensor(label_vec.toarray().ravel(), dtype=torch.float32)`.
`Dataset.from_dict` serializes that list-of-tensors column through
pyarrow as a `list<item: T>` array, and pyarrow infers `double` (8
bytes) for it — not the 4 bytes `torch.float32` suggests. At Wiki10-31K
full scale that's `20,762 × 30,938 × 8 bytes ≈ 5.1GB` for the column's
child buffer, over pyarrow's ~2GB per-buffer limit for standard
(non-`large_list`) list arrays. Full-scale Stage 2 crashed immediately
with `offset overflow while concatenating arrays, consider casting input
from `list<item: double>` to `large_list<item: double>` first` — a
pyarrow construction-time failure, not a training-time one, so it never
showed up as an OOM. The 600-row/150-label smoke sample never came close
to the buffer limit, which is why this wasn't caught earlier.

Fixed by building that tensor as `dtype=torch.uint8` instead of
`torch.float32` — labels are 0/1, so nothing is lost, and every
consumer already calls `labels.float()` before using them (all three
`MultilabelICDClassifier.forward` methods), so this is a storage-only
change with no other call site affected. At uint8 the same column is
`20,762 × 30,938 × 1 byte ≈ 650MB`, matching this section's original
"~640MB, trivial" estimate — the estimate was right, the code just
didn't match it. Eurlex-4K never hits this even before the fix
(`19,314 × 3,956 × 8 bytes ≈ 611MB`, already under the 2GB limit at
float64), which is also why Eurlex-4K's smoke and full-scale runs never
surfaced it.

**Correction, found immediately after: the above fix is necessary but not
sufficient.** Retrying Stage 2 with the `uint8` fix above got substantially
further — 16+ minutes into dataset construction, past the point the first
crash happened instantly — then hit the identical `offset overflow` error
again, now reported as `list<item: int64>` rather than `list<item: double>`.
The `uint8` dtype only survives as far as the *first* `Dataset.from_dict`
call. `preprocess_data` reads that dataset back out via
`dataset.map(process_batch, ...)` — which, without `.set_format("torch")`
in effect yet, decodes every column back into plain Python objects,
collapsing any Arrow-level width info (`uint8` or otherwise) into generic
Python `int`s. `process_batch` hands each row's `labels` (now a bare
`list[int]`) into `chunk_text`, which stores it verbatim as
`chunk_dict["labels"] = labels`. Every emitted chunk for a given document
carries its own full copy of that document's label vector, so for
multi-chunk documents (Wikipedia articles routinely exceed `max_length=512`
tokens) the row count feeding the *final* `Dataset.from_list(flattened_chunks)`
is total chunks, not total docs — larger than the original estimate, and
still built from bare Python ints, which pyarrow always infers as
`int64` regardless of the source dtype further upstream.

Same fix, applied at the point that actually matters: `chunk_text` now
stores `np.asarray(labels, dtype=np.uint8)` instead of the bare `labels`
list, confirmed via isolated repro (`Dataset.from_list` with a numpy-typed
column preserves `uint8`; a bare Python list is always `int64`, same
special-cased dtype-preserving path pyarrow already gave the first fix's
torch tensors). This is the same "guard the boundary where the crash
actually happens, not every place the wrong dtype could theoretically
leak in" lesson as Phi-3-small's flash-attn bf16 guard above — dtype
narrowing doesn't propagate through a plain-Python round trip, so it has
to be reapplied at the last construction point before Arrow serialization,
not just the first. One knock-on fix this required: `process_batch`'s
own-document chunk-consistency check compared `chunk["labels"]` with
`==`, which is fine for two lists but returns an elementwise array (not a
single bool) for two numpy arrays, breaking `all(...)`/`assert` — switched
to `np.array_equal`.

**Third correction — the real root cause, found by testing at true scale
instead of reasoning further from small repros.** The second fix also
wasn't enough: retried Stage 2 a third time, got past the identical
16-minute mark again, then hit `offset overflow` a third time, still
`list<item: int64>` despite `chunk_text` now storing genuinely-uint8
numpy arrays. Stepped back from patch-and-rerun (each attempt costs a
real 16+ minutes just to reach the failure point) and instead imported
`preprocess_data` directly for fast, isolated iteration against the real
full CSV. That located the actual mechanism: **it's an element-count
ceiling, not a byte-size one.** Arrow's standard `list<T>` type uses
int32 *offsets* — indices into the child values array — capped at
`2^31-1 ≈ 2.15B` elements, regardless of each element's byte width. And
the row count feeding this column isn't 20,762 (docs) — Wiki10-31K's
Wikipedia articles routinely need multiple `max_length=512` windows, so
`chunk_text` splits each doc into several chunks, *each carrying its own
full copy* of that doc's label vector (verified: 20,762 docs → 126,573
flattened chunks, a ~6.1x multiplier). `126,573 × 30,938 ≈ 3.9B`
elements — over the 2.15B ceiling at *any* item width, which is exactly
why narrowing `float32 → uint8` (both prior fixes) never had a chance:
narrower items don't reduce element *count*. Confirmed directly:
`Dataset.from_list` on ~130K real-shaped uint8-item rows still failed,
now as `ArrowInvalid: Value 2147499394 too large to fit in C integer
type` — 2147499394 is `2^31` plus change.

**Real fix:** every row's label vector is the same width
(`expected_labels_len`), so it doesn't need a variable-length `list<T>`
at all — a `pyarrow.FixedSizeListArray` has no per-row offsets buffer
whatsoever (row *i*'s slice is just `i × width`, computed, not stored),
so it's structurally immune to this failure regardless of row count, not
merely narrower. `preprocess_data` now pops `"labels"` out of every
chunk dict *before* `Dataset.from_list(flattened_chunks)` (which handles
every other column fine — none is remotely close to the 2.15B ceiling),
builds `pa.FixedSizeListArray.from_arrays(pa.array(labels_flat,
type=pa.uint8()), expected_labels_len)` from the popped values directly,
and reattaches it via `table.append_column("labels", labels_array)`.
Confirmed end-to-end against the complete real 20,762-row Wiki10-31K CSV
(not a repro or a subset): succeeds in ~25 minutes, produces the expected
86,795/39,778 train/test split, correct per-row label sums. The two
narrower-dtype fixes above are harmless no-ops now (uint8 is still a
reasonable Stage 1 intermediate dtype) but were never the actual fix —
left in place and left documented as-is since they're what the
investigation actually did, in order, not rewritten as if this were
found first.

### ⚠️ One-time dataset prep: the `_full` symlink

Same reasoning as Eurlex-4K above — `wiki10-31k.csv` has no `_full`
suffix, so:

```bash
python3 - <<'PY'
from xcube.imports import *
p = untar_xxx(XURLs.WIKI10_31K)
full = p / "wiki10-31k_full.csv"
if not full.exists():
    full.symlink_to("wiki10-31k.csv")
print(f"Ready: {full}")
PY
```

This step has already been done on the primary dev machine; **repeat it
once on the GPU box** after you pull the code there.

## 🤖 **Choosing a Base Model**

| Model | HF id | Params | Notes |
| --- | --- | --- | --- |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Already validated end-to-end on Amazon-3M ([README_AMAZON3M.md](README_AMAZON3M.md)); no extra setup. |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Gated on Hugging Face — accept Meta's license on the model page and use an authenticated HF token (`huggingface-cli login` or `HF_TOKEN`) before downloading. |
| Phi-3-small-8k-instruct | `microsoft/Phi-3-small-8k-instruct` | 7B | Size-matched to the other two (unlike the more commonly-used 3.8B Phi-3-mini). Loaded via `trust_remote_code=True` (already set in `finetune_llm.py`/`learning2rank.py`/`multilabel_classify.py`) from Microsoft's own `modeling_phi3_small.py`, not the upstreamed `transformers` class — needs the `flash_attn` package installed on the GPU box. |

### ⚠️ Porting note: LoRA `target_modules` differ per architecture

Every stage that attaches LoRA (`finetune_llm.py` Stage 1,
`learning2rank.py` Stage 3, `multilabel_classify.py` Stage 2/5) picks
`target_modules` off the base model's attention projection names.
Mistral/Llama use separate `q_proj`/`k_proj`/`v_proj`/`o_proj` — the
pipeline's original default. Phi-3 doesn't:

| Model family | `model_type` | Attention projections | `target_modules` used |
| --- | --- | --- | --- |
| Mistral / Llama | `mistral` / `llama` | separate `q_proj`, `k_proj`, `v_proj`, `o_proj` | `["q_proj","k_proj","v_proj","o_proj"]` |
| Phi-3-mini / -medium | `phi3` | fused `qkv_proj`, separate `o_proj` | `["qkv_proj","o_proj"]` |
| Phi-3-small | `phi3small` | fused `query_key_value`, separate `dense` | `["query_key_value","dense"]` |

This isn't just a naming mismatch to work around by hand — the original
hardcoded `["q_proj","k_proj","v_proj","o_proj"]` default doesn't error
out on Phi-3, since PEFT only requires *some* of the listed names to
match. It silently applies LoRA to `o_proj` alone (the only name that
happens to exist on Phi-3 too) and skips query/key/value entirely — a
quietly crippled adapter that never touches attention's q/k/v, the exact
thing PLANT's attention-transfer story depends on. All three scripts now
branch on `model.config.model_type` to pick the right set automatically;
nothing to pass on the command line. Verified against the real
`transformers` `Phi3ForCausalLM` class (for `phi3`) and against
`microsoft/Phi-3-small-8k-instruct`'s actual `modeling_phi3_small.py` (for
`phi3small`) — in both cases confirming LoRA attaches to the fused
qkv/query-key-value module, not just the output projection.

**Correction, found much later (Combo 3, Stage 5):** "all three
scripts" undercounted by one call site. `LTRModel._init_ground_model`
(`xcube/l2r/models/core.py`) is a *fourth*, separate place that builds
LoRA `target_modules` for the ground model — used whenever `LTRModel`
is reconstructed via `from_pretrained` (Stage 5 loading Stage 4's
local L2R output as its `--hub_model_id`). It only branched on
`"bert" in model_name.lower()`, else the flat
`q_proj`/`k_proj`/`v_proj`/`o_proj` default — no Phi-3 branch at all,
so Stage 5 hit `Ground model initialization failed` outright (a hard
error this time, not the earlier silent-partial-LoRA failure mode,
since `query_key_value`/`dense` don't exist on the model at all under
the wrong target list). Fixed by branching on
`self.ground_model.config.model_type` (the model is already loaded by
that point in the function, so this is the same canonical signal the
other three sites use — more robust than matching on `model_name`,
a free-form Hub id string). Edited via the source notebook
(`nbs/07_l2r.models.core.ipynb`) and re-exported with
`nbdev.nbdev_export()`.

### ⚠️ Porting note: Stage 4's tokenizer check assumed `vocab_size == len(tokenizer)`

`learning2rank.py`'s `verify_token_mappings` (used by Stage 4,
`launch_l2r_training`) checked each token id against
`tokenizer.vocab_size` and compared it to `len(df_toks)` (the token table
Stage 3 built). `tokenizer.vocab_size` reports only the base vocab and
excludes added/reserved special tokens — for Llama-3.1-8B-Instruct that's
`vocab_size=128000` vs the true `len(tokenizer)=128256` (256 reserved
special-token ids), confirmed directly against the real tokenizer. Stage
3's `df_toks` spans the full `len(tokenizer)` id space, so every one of
those 256 ids got treated as out-of-range, manufactured as a "mismatch",
and Stage 4 raised before writing anything: `Tokenizer's vocab size =
128000, while df_tok's size = 128256`.

Mistral-7B-Instruct-v0.3 happens to have `vocab_size == len(tokenizer) ==
32768` (also confirmed directly), which is why this didn't surface until
the Llama-3.1-8B combo — it isn't Llama-specific in cause, just in
exposure: any tokenizer with added/reserved special tokens beyond its
base vocab (the entire Llama-3 family, and many others) hits this. Fixed
by using `len(tokenizer)` throughout `verify_token_mappings` instead —
the only two spots in the codebase that read `tokenizer.vocab_size`,
confirmed via repo-wide search.

### ⚠️ Porting note: `AutoTokenizer.from_pretrained` needs `trust_remote_code=True`

Stage 1 crashed immediately on the Phi-3-small combo: `finetune_llm.py`'s
tokenizer load (`AutoTokenizer.from_pretrained(model_name)`, no
`trust_remote_code`) hit a `[y/N]` remote-code-trust prompt and raised
`ValueError` under the launch scripts' non-interactive shell — confirmed
directly by reproducing the exact prompt/error against the real
`microsoft/Phi-3-small-8k-instruct` tokenizer. Phi-3-small ships a custom
tiktoken-based tokenizer class (`Phi3SmallTokenizer`, loaded via
`trust_remote_code`, same mechanism as its custom model class — see the
LoRA porting note above); the *model* load a few lines below already had
`trust_remote_code=True`, the tokenizer load just didn't.

This is specific to **Phi-3-small**, not Phi-3 generally — Phi-3-mini and
-medium use a standard, fully-upstreamed tokenizer and load fine without
the flag (confirmed directly). Rather than branch on model family, every
`AutoTokenizer.from_pretrained` call in the codebase that loads a
base/adapter model now passes `trust_remote_code=True` unconditionally
(13 call sites across `finetune_llm.py`, `multilabel_classify.py`,
`bootl2r.py`, `learning2rank.py`, confirmed via repo-wide audit) — it's a
no-op for Mistral/Llama/Phi-3-mini, so this doesn't need
model-conditional logic. The two hardcoded
`microsoft/BiomedNLP-PubMedBERT-*` tokenizer loads in `learning2rank.py`
were deliberately left alone (standard BERT, nothing to trust).

### ⚠️ Porting note: model loads need an explicit `torch_dtype`

Phi-3-small failed differently at Stage 1 (CUDA OOM at the first
training step) and Stage 4 (`FlashAttention only support fp16 and bf16
data type`) — same root cause both times. None of the causal-LM
`from_pretrained` calls passed `torch_dtype`, so weights defaulted to
**fp32** (`bnb_4bit_compute_dtype=bfloat16` on the 4-bit stages only
governs the *quantized* linear layers — non-quantized submodules,
including the attention path, still follow `torch_dtype`). Phi-3-small's
custom block-sparse/flash-attention kernel only accepts fp16/bf16 and
sizes its buffers to the model's native `max_seq_len=8192` regardless of
`--max_length`, so fp32 either trips the kernel's hard dtype assert
(Stage 4) or blows past available memory before assert-time (Stage 1:
fp32 weights + fp32 8192-wide attention buffers). Mistral/Llama's
standard attention tolerates fp32, which is why Combos 1–2 never
surfaced this.

Fixed by adding `torch_dtype=torch.bfloat16` to the 4 live
model-loading call sites (`finetune_llm.py`'s fresh-train and
checkpoint-resume loads, `learning2rank.py`'s Stage 4 load,
`multilabel_classify.py`'s Stage 2/5 load) — matches the `bfloat16`
already used for `bnb_4bit_compute_dtype`, and is a no-op for
Mistral/Llama (standard bf16 loading). One citation correction for the
record: an earlier draft of this fix pointed at
`multilabel_classify.py:1832`, which turned out to be inside
`load_base_model_and_tokenizer_too_old` — confirmed dead code, never
called. The real live Stage 2/5 load is
`load_base_model_and_tokenizer`'s `model_info["model_class"].from_pretrained`
call, which is what actually got fixed.

### ⚠️ Porting note: two more precision issues surfaced once bf16 loading worked

Fixing the `torch_dtype` gap above got Stage 1 training and Stage 3
working, but exposed two more Phi-3-small-specific issues:

**Tokenizer reload after training.** Stage 1's post-training evaluation
step reloads the tokenizer from the local checkpoint dir
(`skip_push_to_hub` forces this). `save_pretrained` doesn't copy
Phi-3-small's custom `tokenization_phi3_small.py` into that dir, so the
reload failed: `does not appear to have a file named
tokenization_phi3_small.py`. Fixed by reloading the tokenizer from the
original base model id instead — LoRA fine-tuning never modifies the
tokenizer, so there's no reason to load it from the checkpoint at all.
(Found the same gap in an adjacent `pipeline()` call while fixing this —
missing `trust_remote_code=True`, same category as the 13 sites fixed
earlier, just a different API surface.)

**Fp32 LoRA adapters.** PEFT initializes new LoRA A/B weights in fp32
regardless of the base model's dtype — confirmed by reproducing it in
isolation against the exact `peft==0.20.0` + `bitsandbytes Linear4bit`
setup Stage 4 actually uses. Fixed with `model.to(torch.bfloat16)` right
after `get_peft_model()` in both Stage 4 (`learning2rank.py`) and Stage
2/5 (`multilabel_classify.py`) — verified this only moves the LoRA
layers and leaves bitsandbytes' 4-bit quantized weights (stored as
`uint8`) untouched. Also flipped both files' `TrainingArguments` from
`fp16=True` to `bf16=True`, consistent with the bf16 model loads and the
box's `accelerate` config.

One honest caveat: the originally-proposed mechanism was "the LoRA sum
upcasts to fp32 and that reaches the attention kernel directly." My own
isolated reproduction shows PEFT's LoRA-wrapped `Linear4bit` layer
recasts its *output* back to bf16 even with fp32 internal weights — so
that specific leak path didn't reproduce for me, though the underlying
fp32-weight-creation fact did. The fix is applied because it's safe and
removes a real inconsistency either way; if Stage 4 still fails after
this, the actual leak is likely elsewhere (backward-pass precision, or
something in Phi-3-small's real attention path not captured by a
simplified test) and needs further diagnosis.

### ⚠️ Porting note: Stage 2/5 also need the tokenizer from the base model, not a local dir

The tokenizer-reload fix in the previous note only patched
`finetune_llm.py`'s own reload. `multilabel_classify.py`'s
`load_base_model_and_tokenizer` (Stage 2/5) and
`load_custom_task_model_and_tokenizer` (its `--load_from_checkpoint`
path) had the identical bug: both loaded the tokenizer from whatever
`--hub_model_id` resolved to, which for a chained, `--skip_push_to_hub`
run is a **local directory** — `$STAGE1_LOCAL` for Stage 2,
`$STAGE4_LOCAL` for Stage 5 — that never received Phi-3-small's
custom `tokenization_phi3_small.py`, for the same `save_pretrained`
reason as before.

Fixed by adding a `model_name` parameter to both functions (mirroring
the `hub_model_id`/`model_name` split `learning2rank.py`'s Stage 4
function already had) and loading the tokenizer from `model_name` when
given, falling back to `hub_model_id` only when it isn't. Also added
`--model_name="$BASE_MODEL_NAME"` to Stage 5's command below — it was
the only one of the five stages missing it, so it was silently falling
back to `load_base_model_and_tokenizer`'s Mistral default.

### ⚠️ Porting note: root cause of the flash-attn fp32 assert — a buffer, not a weight

The `model.to(torch.bfloat16)` fix above was correct but incomplete —
Stage 4 still hit `FlashAttention only support fp16 and bf16 data
type`. Root-caused with a hook on
`flash_attn.flash_attn_interface.flash_attn_varlen_kvpacked_func`
(injected via `sitecustomize.py` so it reaches the `accelerate launch`
subprocess): both `q` and `kv` arrive as fp32.

The actual source is Phi-3-small's `RotaryEmbedding`
(`positional_embedding.py`), which builds its cos/sin cache with
`dtype = dtype or torch.get_default_dtype()` — fp32 by default,
independent of the model's own dtype. `apply_rotary_pos_emb` then
multiplies the bf16 query/key by that fp32 cache, upcasting the result;
`key` packs with `value` into `kv`, matching the captured
`q=float32 kv=float32`. `model.to(torch.bfloat16)` can't fix this
because the cache is a buffer built once at construction time from
`get_default_dtype()`, not a parameter reachable by `.to()`-ing the
model after the fact.

Two things were ruled out first:
- **Not `torch_dtype`.** Every non-quantized param (LoRA A/B, biases,
  layernorms) already probed as bf16; the fp32 is generated at
  runtime, not stored.
- **Not `accelerate`'s `mixed_precision`.** Re-running Stage 4 with
  `--precision=bf16` (→ `accelerate launch --mixed_precision=bf16`,
  matching `TrainingArguments(bf16=True)`) failed identically —
  Phi-3-small's block-sparse/flash kernels are custom autograd ops
  that autocast doesn't touch, so `mixed_precision` never reaches
  them.

**Fix:** wrap the base model's `from_pretrained(...)` call with
`torch.set_default_dtype(torch.bfloat16)` (restored in a `finally`) so
the rope cache is built bf16 from construction, in
`learning2rank.py`'s `load_base_model_and_tokenizer` (Stage 4),
`multilabel_classify.py`'s `load_base_model_and_tokenizer` (Stage 2/5
— same base model, same custom attention, so the same bug applies to
its forward pass), and `finetune_llm.py`'s post-training `pipeline()`
call (Stage 1's generation sanity check reloads the model and hits the
identical crash during `generate()`, despite already passing
`torch_dtype=torch.bfloat16` — that flag alone was never enough, as
Stage 4 had it too and still failed). Verified narrowly: casting `q`/
`kv` back to bf16 immediately before the kernel (via the same hook) let
Stage 4 training proceed past the point of failure.

Also reconciled `launches/launchmultilabelclas` and
`launches/launch_l2r_training`'s hardcoded/default
`--mixed_precision=fp16` to `bf16`, matching the `TrainingArguments`
in the scripts they invoke. Per the finding above this alone doesn't
fix anything (`mixed_precision` never reaches Phi-3-small's kernels
either way) — it's a consistency cleanup, not a functional fix.
Stage 3 (`launch_l2rboot`) was left untouched: `bootl2r.py` never
loads the base model itself, so its `--mixed_precision` flag is inert.

### ⚠️ Porting note: the rope fix wasn't the only fp32 source — guarding the kernel boundary instead

The rope fix above was necessary but not sufficient. Rerunning all 5
stages confirmed it worked (`RotaryEmbedding.cos_cached`/`sin_cached`/
`inv_freq` probed as bf16) — but Stage 2/5 and Stage 4 still hit the
identical flash-attn fp32 assert, with `q`/`kv` still arriving fp32 at
the kernel. Two more independent sources, found by process of
elimination:

- **Stage 2/5:** they load Stage 1's *saved* adapter via
  `PeftModel.from_pretrained(base_model, hub_model_id)`, which has no
  `.to(torch.bfloat16)` after it — only the *fresh* `get_peft_model()`
  path (used when there's no adapter to load yet) got that cast. The
  loaded adapter stays fp32.
- **Stage 4:** the base model probed fully bf16 (rope *and* LoRA) —
  yet training still fed fp32 to the kernel. The remaining candidates
  are `define_model()` (wraps the base model as `LTRModel.ground_model`
  and adds fp32 L2R heads) or `accelerator.prepare()`; which one
  reintroduces fp32 wasn't pinned down before switching strategy.

This is the third distinct fp32 source found one at a time (rope →
loaded adapter → wrapper/`prepare`), which makes per-source patching a
losing game — each fix just uncovers the next source, with no way to
know when it's the last one. Switched strategy: **guarantee bf16 at
the point flash-attn actually receives tensors**, rather than at every
place fp32 could theoretically leak in. `scripts/flash_attn_bf16_guard.py`
monkeypatches `flash_attn.flash_attn_interface.flash_attn_varlen_kvpacked_func`
to downcast `q`/`kv` to bf16 only when they arrive fp32 (a no-op for
already-bf16/fp16 tensors, so it's harmless for Mistral/Llama, which
don't hit this code path at all since they don't use
`trust_remote_code`), and is imported + installed at the top of
`finetune_llm.py`, `multilabel_classify.py`, and `learning2rank.py`
(Stage 3's `bootl2r.py` doesn't load the base model, so it's skipped).
This is the exact mechanism verified end-to-end earlier: casting `q`/
`kv` back to bf16 immediately before the kernel let Stage 4 training
proceed past the point of failure. Downcasting this late is numerically
safe — it only meets the kernel's own dtype requirement; accumulation
up to that point stays at whatever precision it already was.

One implementation detail that matters for it to actually take effect:
the dynamically-downloaded `modeling_phi3_small.py` binds
`flash_attn_varlen_kvpacked_func` via `from flash_attn... import ...`
the *first* time it's imported (triggered by the first
`trust_remote_code=True` model load in that process), so the patch
must already be in place on `flash_attn.flash_attn_interface` before
that happens — which is why `install_bf16_boundary_guard()` is called
at each script's top level, before any model-loading code runs, rather
than closer to the actual `from_pretrained()` call sites.

One remaining honest caveat: verification so far only exercised
Phi-3-small's *dense* attention path (`flash_attn_varlen_kvpacked_func`);
Phi-3-small alternates dense and block-sparse attention layers, and the
block-sparse kernel wasn't observed to be hit in these short
(`--max_length=512`) smoke runs, so it isn't patched here. If a longer
run exercises block-sparse layers and hits the same assert, the guard
will need a second patch target for whatever function those layers
call.

### ⚠️ Porting note: Phi-3-small's tied embeddings break safetensors save

The boundary guard above resolved every flash-attn error — Stage 2
trained all 59 steps and Stage 4 all 237, both cleanly. Both then
crashed at model-save instead, with an unhelpful empty error message
(the handler only logs `str(e)`, which was blank here):

```
Some tensors share memory ... [{'...embed_tokens.weight', '...lm_head.weight'}]
```

Phi-3-small sets `tie_word_embeddings=True` (Mistral/Llama are
`False`), so its `embed_tokens` and `lm_head` weights are the same
underlying tensor. `safetensors` refuses to serialize two output
entries that alias the same memory — this is *why* it's the first
model of the three to hit this: it's not a size or precision issue,
it's specific to which models tie their embeddings.

**First fix attempt (necessary, not sufficient):**
`safe_serialization=False` (writes `pytorch_model.bin` via
`torch.save`/`pickle`, which handles aliased tensors natively) at
every point a Phi-3-small-based model gets saved:
`save_safetensors=False` added to both files' `TrainingArguments`,
`learning2rank.py`'s explicit `trainer.model.save_pretrained(output_dir)`
call passing `safe_serialization=False` directly, and
`LTRModel.save_pretrained` (`xcube/l2r/models/core.py`) hardcoding
`safe_serialization=False` on `self.ground_model.save_pretrained(...)`
specifically, since it wasn't forwarding `**kwargs` there at all.

**What actually happened on rerun:** both stages trained to full
completion (Stage 2: 59/59, Stage 4: 237/237) — the boundary guard
fully held — but *both still crashed at save*, identically. The
`TrainingArguments`/`save_pretrained` fixes above never had a chance
to run: both `CustomTrainer` subclasses (`multilabel_classify.py` and
`learning2rank.py`) override `Trainer._save` and call
`safetensors.torch.save_file(...)` **directly**, completely bypassing
`self.args.save_safetensors`. `trainer.save_model()` calls this
override, not `Trainer`'s own `_save`, so the flag was simply never
consulted on the path that actually matters — `MultilabelICDClassifier`
being a plain `nn.Module` was true but irrelevant, since it never even
reaches `Trainer`'s generic (flag-respecting) fallback.

**Real fix:** made both `_save` overrides check the flag themselves:

```python
if getattr(self.args, "save_safetensors", True):
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
else:
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
```

This surfaced a coupled problem: both files' `_load_best_model`
(consulted whenever `load_best_model_at_end=True`, which both set) and
`save_and_push`'s required-file/best-checkpoint-verification checks
all hardcoded `model.safetensors` — writing `pytorch_model.bin`
instead would have traded a save error for a load error one step
later. Fixed every live (non-`_old`/non-dead) reader in both files to
check for `model.safetensors` first and fall back to
`pytorch_model.bin`, loading with `load_file`/`torch.load`
respectively. `LTRModel.from_pretrained` already had this fallback
built in (a pre-existing try/except around both filenames), so it
needed no change.

All of this is a harmless no-op for untied Mistral/Llama runs — same
principle as the boundary guard. The `xcube/l2r/models/core.py` change
was made via the source notebook (`nbs/07_l2r.models.core.ipynb`) and
re-exported with `nbdev.nbdev_export()`, not by hand-editing the
generated `.py` directly.

## ⚙️ **GPU Box Setup**

This section only covers what's specific to running *this* file's
combinations — see [README.md's Installation & Setup](README.md#-installation--setup)
for the general xcube bootstrap.

1. **Get the code.** Fresh box:
   ```bash
   git clone git@github.com:deb-official/plant.git ~/plant
   cd ~/plant && git checkout plant-full-scale
   ```
   Already have a checkout on this box? Sync it instead of re-cloning:
   ```bash
   cd ~/plant && git fetch origin && git checkout plant-full-scale && git pull
   ```

2. **Python environment:**
   ```bash
   cd ~/plant
   pip install -e .
   ```
   This installs the `transformers==4.49.0` pinned in `settings.ini` — the
   version this pipeline was actually validated against (the primary dev
   machine's environment has drifted to a newer `transformers`; a fresh
   GPU box following `settings.ini` as-is gets the tested combination).

3. **Model-specific extras** — only install the one(s) you need:
   - **Phi-3-small-8k-instruct** needs `flash_attn`, which needs an
     Ampere+/Hopper GPU (see the [LoRA porting note](#️-porting-note-lora-target_modules-differ-per-architecture)
     above — Turing GPUs can't run this model at all, dense-attention
     layers hard-`assert` on it). Deliberately **not** auto-installed by
     `pip install -e .` — see the comment above `requirements=` in
     `settings.ini` for why — so install it separately, after step 2:
     ```bash
     pip install flash-attn --no-build-isolation
     ```
     `tiktoken` and `pytest` (its custom tokenizer backend and a stray
     import in Microsoft's own modeling code — see the
     [`trust_remote_code` porting note](#️-porting-note-autotokenizerfrom_pretrained-needs-trust_remote_codetrue)
     above) **are** in `settings.ini`'s `requirements=`, so step 2 already
     installs them — nothing extra needed here.
   - **Llama-3.1-8B-Instruct** is gated on Hugging Face. Accept Meta's
     license on the model page with whichever HF account you'll
     authenticate as, then:
     ```bash
     huggingface-cli login   # or: export HF_TOKEN="..."
     ```
   - **Mistral-7B-Instruct-v0.3** needs nothing beyond step 2.

   No Weights & Biases setup needed for any model — `report_to="none"` is
   hardcoded in all three training scripts' `TrainingArguments`
   (`finetune_llm.py`, `learning2rank.py`, `multilabel_classify.py`)
   despite `wandb` being in `settings.ini`'s requirements (it's a
   transitive dependency of something else in that list, not something
   this pipeline calls into).

4. **Datasets** — download, `_full` symlink, and smoke-test sample, each
   once per dataset (not per model or per stage): the download+symlink
   commands are in [Dataset: Eurlex-4K](#-dataset-eurlex-4k) /
   [Dataset: Wiki10-31K](#-dataset-wiki10-31k) above, and the sample
   command is in [Building the sample](#building-the-sample) below.

5. **`accelerate` precision config** — the model loads now explicitly
   request `bfloat16` (see the
   [`torch_dtype` porting note](#️-porting-note-model-loads-need-an-explicit-torch_dtype)
   above), so set `accelerate`'s mixed-precision mode to match, rather
   than leaving it at whatever a generic setup script defaulted to
   (`fp16`, in at least one observed case):
   ```bash
   accelerate config   # choose bf16 when prompted for mixed precision
   ```
   This lives in `~/.cache/huggingface/accelerate/default_config.yaml`
   on each box (not part of this repo), so it's a one-time manual step
   per fresh box, same as the HF login above.

**Disk space**: budget ~15–30GB per base model (Hub repo totals —
Mistral-7B-Instruct-v0.3 ~29GB, Llama-3.1-8B-Instruct ~32GB,
Phi-3-small-8k-instruct ~15GB; `from_pretrained` typically only pulls the
`safetensors` shards, not every format in the repo, so actual usage tends
to run lower than these totals) plus a few hundred MB for each dataset.
Comfortable on a GPU box with 100GB+ free if running all three models.

Confirmed working end-to-end on an NVIDIA H100 80GB, all 5 stages:

| Combo | Status |
| --- | --- |
| Eurlex-4K + Mistral-7B-Instruct-v0.3 | ✅ (surfaced [the `DATASET`/`DATASET_SHORT` variable-collision fix](#the-commands)) |
| Eurlex-4K + Llama-3.1-8B-Instruct | ✅ (surfaced [the `vocab_size`/`len(tokenizer)` porting fix](#️-porting-note-stage-4s-tokenizer-check-assumed-vocab_size--lentokenizer)) |
| Eurlex-4K + Phi-3-small-8k-instruct | ✅ all 5 stages (surfaced the full `trust_remote_code` / `torch_dtype` / flash-attn boundary-guard / tied-embedding-save / `LTRModel._init_ground_model` LoRA-target chain documented above) |
| Wiki10-31K + Mistral-7B-Instruct-v0.3 | ✅ with Stages 2/4/5 at `--num_train_epochs=3` (surfaced [the Stage-4/Stage-5 undertraining finding](#️-stage-4-needs-enough-epochs--1-isnt-always-sufficient) — 1 epoch everywhere gave a false "PLANT loses" result, and bumping Stage 4 alone wasn't enough either) |
| Wiki10-31K + Llama-3.1-8B-Instruct | ✅ with Stages 2/4/5 at `--num_train_epochs=3` — second data point confirming [the Stage-4/Stage-5 undertraining finding](#️-stage-4-needs-enough-epochs--1-isnt-always-sufficient), margins comparable to or larger than Mistral's |
| Wiki10-31K + Phi-3-small-8k-instruct | ✅ with Stages 2/4/5 at `--num_train_epochs=3` — completes all 6 combos; PLANT wins every ranking metric here too, though the margins are inflated by an unusually weak Phi-3-small baseline on this dataset (see below) |

All six combos are now complete. Every one shows PLANT (Stage 5)
beating baseline (Stage 2) on every ranking metric — directional
results only (600-row smoke sample), not final-scale numbers.
Phi-3-small needed substantially more porting work to get there than
Mistral/Llama (see the "Porting note" subsections above); once fixed,
its PLANT lift follows the same pattern as the other two on Eurlex-4K.
Both Wiki10-31K combos needed Stages 2, 4, and 5 run for 3 epochs
instead of 1 to show this pattern at all — bumping Stage 4 alone
wasn't enough — see
["Stage 4 needs enough epochs"](#️-stage-4-needs-enough-epochs--1-isnt-always-sufficient)
below the commands for the full before/after numbers and why. Llama
and Phi-3-small both reproducing the win-everything pattern on
Wiki10-31K confirms this isn't a Mistral-specific quirk — though
Phi-3-small's win there comes with real caveats (see the table below).

Full 6-combo matrix, P@25 base→PLANT (recorded for all six; P@5 is
recorded for the Wiki10-31K combos below, not the Eurlex-4K ones) for
a one-screen view — every combo wins:

| # | Dataset × Model | P@25 base→PLANT |
| --- | --- | --- |
| 1 | Eurlex-4K × Mistral | 0.0323 → 0.0465 |
| 2 | Eurlex-4K × Llama | 0.0385 → 0.0540 |
| 3 | Eurlex-4K × Phi-3-small | 0.0308 → 0.0428 |
| 4 | Wiki10-31K × Mistral | 0.1463 → 0.1908 |
| 5 | Wiki10-31K × Llama | 0.1502 → 0.1938 |
| 6 | Wiki10-31K × Phi-3-small | 0.0394 → 0.1545 |

Combo 6 has the largest P@25 lift by far (+293%, vs +30–39% for the
other five), and at P@5 the gap is even more extreme (+697%). That's
misleading taken at face value: Phi-3-small's *baseline* on Wiki10-31K
is the anomaly (P@5=0.059, near-random), not its PLANT result — see
the caveat below the Wiki10-31K table.

The three Eurlex-4K runs below share the same box (NVIDIA H100 80GB,
Eurlex-4K sample, 600 rows / 150 labels) and epoch count (1 per stage,
including Stage 4 — sufficient for this dataset, unlike Wiki10-31K):

| Metric | Mistral (base→PLANT) | Llama (base→PLANT) | Phi-3-small (base→PLANT) |
| --- | --- | --- | --- |
| P@25 | 0.0323 → 0.0465 | 0.0385 → 0.0540 | 0.0308 → 0.0428 |
| R@25 | 0.3164 → 0.4850 | 0.3805 → 0.5401 | 0.2906 → 0.3900 |
| PSP@25 | 0.2617 → 0.4254 | 0.3189 → 0.4952 | 0.2463 → 0.3968 |
| F1-micro | 0.0352 → 0.0623 | 0.0272 → 0.0531 | 0.0348 → 0.1084 |

The Wiki10-31K runs below share the same box (NVIDIA H100 80GB,
Wiki10-31K sample, 600 rows / 150 labels), but Stages 2, 4, and 5 at
**3 epochs**, not 1 — see ["Stage 4 needs enough epochs"](#️-stage-4-needs-enough-epochs--1-isnt-always-sufficient)
below for why this dataset needed it and Eurlex-4K didn't. All three
models win on every ranking metric here, but read Phi-3-small's row
with the caveat right below the table — its huge deltas are inflated
by an unusually weak baseline, not an unusually strong PLANT result.

| Metric | Mistral (base→PLANT) | Llama (base→PLANT) | Phi-3-small (base→PLANT) |
| --- | --- | --- | --- |
| P@5 | 0.3988 → 0.5988 | 0.4065 → 0.5801 | 0.0591 → 0.4711 |
| P@25 | 0.1463 → 0.1908 | 0.1502 → 0.1938 | 0.0394 → 0.1545 |
| R@25 | 0.5753 → 0.7722 | 0.6252 → 0.7862 | 0.1272 → 0.6466 |
| PSP@25 | 0.5228 → 0.6748 | 0.4651 → 0.6861 | 0.1674 → 0.4954 |
| F1-micro | 0.0927 → 0.5087 | 0.3502 → 0.4277 | 0.0824 → 0.1500 |
| F1-macro | 0.0763 → 0.2079 | 0.0196 → 0.2184 | 0.0720 → 0.0726 (flat) |

**Read Phi-3-small's numbers carefully — the ratios are inflated by a
broken baseline, not a uniquely strong PLANT.** Its Stage 2 baseline
on Wiki10-31K nearly collapsed (P@5=0.059, near-random) even though
its Eurlex-4K baseline was fine and its own PLANT result here is
healthy — that's what produces the misleadingly huge percentage lifts
(P@5 +697%, R@25 +408%). In *absolute* terms, Phi-3-small's PLANT is
actually the weakest of the three on Wiki10-31K (P@5 0.471 vs
Mistral's 0.599 and Llama's 0.580) — PLANT rescued it from a bad
baseline back up to a normal PLANT level; that's a real result, but a
rescue, not a new high-water mark. The weak baseline itself (clean
run, no error — the pipeline just fit poorly) is worth a glance if
investigating further, but isn't chased down here.

Also unlike Mistral/Llama, Phi-3-small's tail didn't benefit:
`F1-macro` is flat (0.0720 → 0.0726) and `rare-F1-micro` actually
*dropped* (0.0212 → 0.0000) — the opposite direction from Mistral/
Llama, where it was 0 for both baseline and PLANT (nothing to drop).
For Phi-3-small on Wiki10-31K, PLANT's entire gain is head/mid-frequency
ranking; Mistral and Llama's F1-macro gains come from mid-frequency
not-rare labels too (e.g. Llama's `not_rare_precision` 0.219 → 0.313),
just without an existing rare-label baseline signal to lose. Not
enough rare-label support at this 600-row sample size to read much
into any of the three directions.

## 📝 **Naming Conventions**

Same mechanism as [README_AMAZON3M.md](README_AMAZON3M.md#-naming-conventions-for-amazon-3m):

```bash
source setup_plant_names.sh "<dataset>_full" "<model-hf-id>"
```

All Hub ids below use `deb101` (this repo's author). Reproducing this on
your own Hugging Face account: `setup_plant_names.sh` reads
`HF_USERNAME` from the environment (`HF_USERNAME=${HF_USERNAME:-"deb101"}`),
so `export HF_USERNAME="your-hf-username"` before sourcing it, and every
`STAGE*_HUB_ID` below is generated under your account instead.

`<dataset>` is `eurlex-4k` or `wiki10-31k`; `<model-hf-id>` is one of the
three rows above. `setup_plant_names.sh` derives a short, collision-free
tag per model (fixed here to handle Phi-3's context-length-in-name repo
ids like `Phi-3-small-8k-instruct`, which otherwise all collapsed to the
same short name):

| Model | `MODEL_SHORT` |
| --- | --- |
| Mistral-7B-Instruct-v0.3 | `mistral7b` |
| Llama-3.1-8B-Instruct | `llama8b` |
| Phi-3-small-8k-instruct | `phi3small` |

Example for Eurlex-4K + Mistral:

```bash
source setup_plant_names.sh "eurlex-4k_full" "mistralai/Mistral-7B-Instruct-v0.3"
```

prints (and exports) exactly:

| Stage | Hub ID | Output Dir |
| --- | --- | --- |
| 1 | `deb101/mistral-7b-eurlex-4k-adapt` | `mistral7b_eurlex-4k` |
| 2 | *(same as Stage 1)* | `eurlex-4k_classify_mistral7b` |
| 3 | `mistralai/Mistral-7B-Instruct-v0.3` (tokenizer only) | `eurlex-4k_l2rboot_mistral7b` |
| 4 | *(same as Stage 1)* | `eurlex-4k_l2rtrain_mistral7b` |
| 4 (L2R output) | `deb101/mistral-7b-eurlex-4k-adapt-l2r` | — |
| 5 | *(same as Stage 4 L2R output)* | `eurlex-4k_plantclassify_mistral7b` |

Swap `eurlex-4k` for `wiki10-31k`, or the model id for one of the other
two, and every name above follows the same pattern — `DATASET_NAME` is
exported for use as `--data` in every stage.

## 🧪 **Smoke-Testing All 5 Stages on a Sample First**

> 📍 Same dense track for both datasets and all three models — no
> `--label_space` flag, no candidate-generator machinery (see
> [Dataset: Wiki10-31K](#-dataset-wiki10-31k) above for why that's true
> even for its 30,938 labels).

Unlike [README_AMAZON3M.md's sample](README_AMAZON3M.md#-smoke-testing-all-5-stages-on-a-sample-first),
the point of a sample here isn't to dodge an OOM — Eurlex-4K (19,314 rows)
and Wiki10-31K (20,762 rows) are already small enough to just run in
full. It's for **speed**: `bootl2r.py`'s Stage 3 MIG bootstrap makes
`ceil(num_unique_labels / chnk_sz)` full-dataset passes (`chnk_sz=200` by
default) — 20 passes on Eurlex-4K's 3,956 labels, 155 passes on
Wiki10-31K's 30,938. A sample that caps the label vocabulary under
`chnk_sz` gets that down to a single pass, so a smoke test actually
finishes fast instead of just being smaller.

Two other constraints from Amazon-3M's sample section still apply here,
since they're properties of this codebase, not of any one dataset:

1. **Row count** — drives per-step training time (Stages 1, 2, 4, 5).
2. **A hidden cross-stage dependency**: `bootl2r.py`'s Stage 3
   (`get_info()` → `load_mlb_classes()`) reads
   `{base_dir}/labels_partition.json`, written only by
   `multilabel_classify.py`'s Stage 2 (or 5) `preprocess_data()`. Stage 3
   cannot run before Stage 2 has run at least once against the same
   `--base_dir` — running stages in order (1→2→3→4→5) satisfies this for
   free.

### Building the sample

`scripts/create_amazon3m_sample.py` is fully dataset-generic despite the
name — every path is a CLI flag (`--source_url`, `--csv_name`,
`--output_name`, ...), and it only assumes a `text,labels,is_valid` CSV
with `;`-delimited labels, which is exactly what both datasets' downloads
now produce (see the two [Dataset](#-dataset-eurlex-4k) sections above).
No new script needed — just different flags:

```bash
cd ~/plant/scripts

python3 create_amazon3m_sample.py \
    --source_url=XURLs.EURLEX_4K --csv_name=eurlex-4k \
    --output_name=eurlex-4k-sample_full \
    --top_k_labels=150 --target_rows=600 --valid_frac=0.10

python3 create_amazon3m_sample.py \
    --source_url=XURLs.WIKI10_31K --csv_name=wiki10-31k \
    --output_name=wiki10-31k-sample_full \
    --top_k_labels=150 --target_rows=600 --valid_frac=0.10
```

Both ran cleanly against the actual downloaded data:

| | Eurlex-4K (full → sample) | Wiki10-31K (full → sample) |
| --- | --- | --- |
| Rows | 19,314 → 600 (540 train / 60 eval) | 20,762 → 600 (540 train / 60 eval) |
| Unique labels | 3,956 → 150 | 30,938 → 150 |
| Median labels/doc | 6 → 3 | 19 → 6 |
| Sample file | `eurlex-4k-sample_full.csv` | `wiki10-31k-sample_full.csv` |

150 labels keeps Stage 3 to exactly one `chnk_sz=200` pass for either
dataset, same as Amazon-3M's sample.

### Isolating sample runs, avoiding Hub pollution, chaining stages locally

All three mechanisms from README_AMAZON3M.md apply here completely
unchanged (they're generic to the codebase, not Amazon-3M-specific):
[isolating sample runs from the real run](README_AMAZON3M.md#isolating-sample-runs-from-the-real-run),
[avoiding Hub pollution and slow uploads](README_AMAZON3M.md#avoiding-hub-pollution-and-slow-uploads),
and [chaining stages locally without the Hub](README_AMAZON3M.md#chaining-stages-locally-without-the-hub).
In short: `--base_dir="../tmp_sample"` keeps sample runs in their own
tree, `--skip_push_to_hub` avoids uploading multi-GB checkpoints for a
disposable run, and `--hub_model_id` can point at a previous stage's
**local** output dir instead of a Hub id when chaining stages without
pushing.

### The commands

One generic block, parameterized by dataset and model — set the three
variables at the top, then run as-is. Swap `DATASET_BASE`/`SOURCE_URL` for
the other dataset, or `MODEL` for either of the other two from
[Choosing a Base Model](#-choosing-a-base-model), and nothing else
changes.

> ⚠️ `setup_plant_names.sh` uses a variable literally named `DATASET`
> internally for its first positional arg (`scripts/setup_plant_names.sh:6`).
> Since it's `source`d (not run in a subshell), it silently overwrites any
> caller-side `$DATASET` of the same name — which is exactly what broke an
> earlier draft of this block (Stage 4 picked up the clobbered value and
> looked for `..._full-sample_tok_rank_per_lbl.ft` instead of the real
> `..._sample_tok_rank_per_lbl.ft`). The block below avoids the collision by
> naming the caller-side variable `DATASET_BASE` instead, and uses
> `$DATASET_SHORT` (which `setup_plant_names.sh` exports as
> `${DATASET%_*}` — the `_full`-stripped form) for Stage 4's
> `--data_l2r_fname_prefix` rather than reconstructing it by hand.

```bash
cd ~/plant/scripts

DATASET_BASE="eurlex-4k"                          # or "wiki10-31k"
SOURCE_URL="XURLs.EURLEX_4K"                       # or "XURLs.WIKI10_31K"
MODEL="mistralai/Mistral-7B-Instruct-v0.3"         # or the Llama/Phi-3 id

source setup_plant_names.sh "${DATASET_BASE}-sample_full" "$MODEL" "../tmp_sample"
# -> DATASET_NAME=${DATASET_BASE}-sample_full, DATASET_SHORT=${DATASET_BASE}-sample, BASE_DIR=../tmp_sample

STAGE1_LOCAL="$BASE_DIR/${SOURCE_URL#XURLs.}/$STAGE1_OUTPUT_DIR"
STAGE4_LOCAL="$BASE_DIR/${SOURCE_URL#XURLs.}/$STAGE4_OUTPUT_DIR"

# --- Stage 1 -------------------------------------------------------------
launches/launchLLM \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE1_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=1 \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 --skip_push_to_hub

# --- Stage 2 (also writes labels_partition.json, needed by Stage 3) ------
# 3 epochs, not 1 -- see "Stage 4 needs enough epochs" below: an
# isolation run showed bumping only Stage 4 leaves PLANT's head-ranking
# metrics behind baseline. The confirmed-working combination retrains
# baseline at the same epoch count as PLANT (apples-to-apples), not a
# proven-minimal one -- see that section before assuming Stage 2 must
# move too in your own experiments.
launches/launchmultilabelclas \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE2_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-classify" --do_fresh_training \
    --num_train_epochs=3 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_25" --rare_threshold=5 \
    --label_coverage_frac=1.0 --skip_push_to_hub

# --- Stage 3 (needs labels_partition.json from Stage 2 above) ------------
launches/launch_l2rboot \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" \
    --base_dir="$BASE_DIR" --output_dir="$STAGE3_OUTPUT_DIR" \
    --hub_model_id="$STAGE3_HUB_ID" --bs=8 --chnk_sz=200 --max_length=1024

# --- Stage 4 (3 epochs, not 1 -- see "Stage 4 needs enough epochs" below) --
launches/launch_l2r_training \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" \
    --data_l2r_fname_prefix="$DATASET_SHORT" --max_length=512 \
    --base_dir="$BASE_DIR" --l2r_boot_dir="$STAGE3_OUTPUT_DIR" \
    --output_dir="$STAGE4_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=3 --learning_rate=1e-4 --warmup_steps=0 \
    --metric_for_best_model="ndcg@25" \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 --skip_push_to_hub

# --- Stage 5 (hub_model_id = Stage 4's local L2R output) ------------------
# 3 epochs, not 1 -- a converged Stage 4 alone was NOT enough to make
# PLANT beat baseline on head-ranking metrics (P@5, PSP@5); Stage 5
# itself needs the extra epochs too. See "Stage 4 needs enough epochs"
# below for the isolation run that established this.
launches/launchmultilabelclas \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE5_OUTPUT_DIR" \
    --hub_model_id="$STAGE4_LOCAL" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-plantclassify" --do_fresh_training \
    --num_train_epochs=3 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_25" --rare_threshold=5 \
    --label_coverage_frac=1.0 --skip_push_to_hub
```

> ⚠️ **Phi-3 models only**: nothing changes in the commands above — the
> LoRA `target_modules` fix (see
> [Choosing a Base Model](#-choosing-a-base-model)) is automatic based on
> `model.config.model_type`, not a flag you need to pass.

### ⚠️ Stage 4 needs enough epochs — 1 isn't always sufficient

All three Eurlex-4K combos above won on every metric at Stage 4's
original `--num_train_epochs=1`. Wiki10-31K + Mistral did not — its
first run was a genuine counterexample to "PLANT always wins," and the
root cause turned out to be Stage 4 undertraining, not a dataset PLANT
doesn't work for.

**1 epoch — PLANT loses on every ranking metric:**

| Metric | Baseline | PLANT | Δ |
| --- | --- | --- | --- |
| P@5 | 0.4879 | 0.4674 | −4.2% |
| P@25 | 0.1679 | 0.1607 | −4.3% |
| R@25 | 0.6874 | 0.6630 | −3.5% |
| PSP@25 | 0.5598 | 0.5248 | −6.3% |
| F1-micro | 0.0834 | 0.3414 | +309% |
| rare-F1-micro | 0.0215 | 0.0000 | −100% |

F1-micro's huge jump is real but misleading here: it's driven entirely
by PLANT predicting far fewer, more-confident positives on head/not-rare
labels (`not_rare_precision` 0.044 → 0.248) — while it zeroed out rare
labels entirely (baseline had marginal rare-label signal; PLANT had
none). On the metrics that actually matter for XMC ranking quality
(P@k, R@k, PSP@k), PLANT was strictly worse.

**All stages at 3 epochs — same combo, PLANT wins by wide margins on
everything:**

| Metric | Baseline | PLANT | Δ |
| --- | --- | --- | --- |
| P@5 | 0.3988 | 0.5988 | +50% |
| P@25 | 0.1463 | 0.1908 | +30% |
| R@25 | 0.5753 | 0.7722 | +34% |
| PSP@25 | 0.5228 | 0.6748 | +29% |
| F1-micro | 0.0927 | 0.5087 | +449% |
| F1-macro | 0.0763 | 0.2079 | +172% |

**Correction — a follow-up experiment shows Stage 4 alone isn't the
fix.** The paragraph originally here credited Stage 4's better
attention init as *the* mechanism and left Stages 1/2/5 at 1 epoch in
the generic block. A dedicated isolation run (Stage 4 at 3 epochs,
every other stage still at 1) tested that directly — and falsifies it
as a complete explanation:

**Stage 4 = 3 epochs, Stages 1/2/5 = 1 epoch (isolation run):**

| Metric | Baseline (1ep) | PLANT (1ep, Stage 4 = 3ep) | Δ |
| --- | --- | --- | --- |
| P@5 | 0.5351 | 0.4698 | −12% |
| P@25 | 0.1711 | 0.1713 | ~tie |
| R@25 | 0.6927 | 0.7100 | +2% |
| PSP@5 | 0.4086 | 0.3687 | −10% |
| PSP@25 | 0.5653 | 0.5770 | +2% |
| F1-macro | 0.0325 | 0.0826 | +154% |
| rare-F1-micro | 0.0000 | 0.0894 | rare labels no longer zeroed |

Stage 4's own `ndcg@25` did improve monotonically with its extra
epochs (0.4903 → 0.5246 → 0.5305, vs ~0.487 in the all-1-epoch run) —
so the better attention init is real. But with Stage 5 still trained
for only 1 epoch, PLANT is still *worse* than baseline at the head
(P@5 −12%, PSP@5 −10%) and merely tied at deeper cutoffs (P@25, R@25,
PSP@25 within ±2%). What the better init clearly does move is the
tail: F1-macro +154%, and `rare-F1-micro` goes from exactly 0 to
0.089 — PLANT starts predicting rare labels it previously never
touched. That's the mechanism PLANT is designed to help with, so the
Stage-4 lift shows up there; it just doesn't reach the head-ranking
metrics on its own.

Lined up against the three regimes tested so far:

| Regime | Head ranking (P@5, PSP@5) | Tail (rare-F1) |
| --- | --- | --- |
| All 1 epoch | ~tied / PLANT slightly behind | zeroed for PLANT |
| Stage 4 = 3, rest = 1 | PLANT still behind | fixed (0 → 0.089) |
| All stages = 3 | PLANT wins decisively | (not separately reported) |

So the decisive factor in the all-3-epoch win was training **Stage 5
(PLANT) itself** for more epochs, not Stage 4's init quality in
isolation — the better init helps the tail, but the head-ranking
dominance only shows up once Stage 5 gets enough epochs to exploit it.
Note the all-3-epoch run also retrained Stage 2 (baseline) at 3
epochs, not just Stage 4/5 — so that comparison stayed apples-to-apples
(same epoch budget both sides) rather than PLANT simply outtraining a
frozen-at-1-epoch baseline. Because Stage 2 wasn't varied
independently of Stage 5 in either run, its own contribution isn't
fully isolated yet — the honest statement is "Stages 4 and 5 both
needed more epochs, and Stage 2 got 3 as well in the run that worked,"
not a precise minimal recipe.

**Why Eurlex-4K didn't need this and Wiki10-31K did:** both smoke
samples cap `top_k_labels=150`, so it isn't raw full-corpus label-space
size (Eurlex-4K's ~4K vs Wiki10-31K's ~31K unique labels) showing up
directly in this 600-row sample. The more likely driver is label
*density*: Wiki10-31K's sample has a higher median labels/doc than
Eurlex-4K's even after both are capped to the same top-150 labels (see
the sample-build table above — 6 vs 3 post-sample). More concurrent
labels per document plausibly makes the attention-to-relevance mapping
Stage 4 is learning, and the classify head Stage 5 fine-tunes on top of
it, harder to fit — needing more epochs on both to converge. Treat
this as a working hypothesis, not a proven mechanism — it's based on
one dataset pair at smoke scale, not a controlled ablation.

**Consequences for command design:**
- The generic block below now defaults Stages 2, 4, and 5 to
  `--num_train_epochs=3` (was `1` for all three), applied to *both*
  datasets — a no-op-or-better for Eurlex-4K (already winning at 1
  epoch; 3 can only help further) and the configuration actually
  confirmed to fix Wiki10-31K. Stage 1 stays at 1 (untested increase;
  not implicated by any experiment so far) and Stage 3 has no epoch
  parameter to change.
- **This is the confirmed-working combination (2/4/5 = 3), not a
  proven-minimal one.** A cleaner ablation — e.g. Stage 2 = 1, Stage
  4 = 3, Stage 5 = 3 — would tell you whether the baseline actually
  needs the extra epochs too, or whether it was incidental. Worth
  running if a tighter recipe (and shorter smoke-test runtime)
  matters; not required just to reproduce the win.
- **For full-scale (non-sample) runs on either dataset**, don't assume
  1 epoch is enough anywhere in Stages 2/4/5 just because it worked at
  smoke scale for Eurlex-4K. Watch Stage 4's own `ndcg@25` (its
  `--metric_for_best_model`) across epochs as a cheap early signal for
  whether the attention-transfer step has converged — but per the
  isolation run above, a converged Stage 4 alone does **not** mean
  Stage 5 is done training; Stage 5's own head-ranking metrics need to
  be watched directly, not inferred from Stage 4's.

**Second data point (Llama):** Wiki10-31K + Llama-3.1-8B, run with the
confirmed-working 2/4/5=3 config, reproduces the same win-everything
pattern as Mistral, with margins comparable to or larger than
Mistral's (e.g. PSP@5 +73%, F1-macro +1015%). This isn't a Mistral
quirk — the effect reproduces across base models on Wiki10-31K. One
difference from the earlier isolation-run data: at 3 epochs,
`rare-F1-micro` is 0 for *both* baseline and PLANT (not just PLANT) —
the F1-macro gain here comes entirely from PLANT activating
mid-frequency not-rare labels, with too little rare-label support at
this sample size to move the tail metric either direction.

**Third data point (Phi-3-small) — same config, cleanly, but with its
own caveat:** Wiki10-31K + Phi-3-small also completed the 2/4/5=3
config with zero errors — the entire Phi-3-small fix chain (flash-attn
boundary guard, tied-embedding save, `LTRModel` LoRA targets) carried
over to 3 epochs without anything new surfacing, confirming those
fixes weren't epoch-count-dependent. PLANT wins every ranking metric
here too, but Phi-3-small's Stage 2 baseline on Wiki10-31K nearly
collapsed (P@5=0.059, near-random — its Eurlex-4K baseline was fine),
which inflates the ratios (P@5 +697%) far beyond what PLANT's
*absolute* result justifies — in absolute terms Phi-3-small's PLANT is
the weakest of the three on Wiki10-31K, not the strongest. See the
full breakdown in the Wiki10-31K table above. Also, unlike Mistral/
Llama, Phi-3-small's `rare-F1-micro` *dropped* (0.021 → 0.000) rather
than staying at 0 for both sides — its baseline had some marginal
rare-label signal that PLANT lost, the opposite of what happened with
the better-fit Mistral/Llama baselines.

**Status**: all six combos (2 datasets × 3 base models) confirmed
working end-to-end (Stages 1→5) on an NVIDIA H100 80GB, all winning on
every ranking metric — see the combo status table above and the full
6-combo matrix further up.
Exact per-stage timing isn't recorded yet, and will now run somewhat
longer than the original estimate since Stages 2, 4, and 5 default to
3 epochs instead of 1 — Amazon-3M's smoke test took ~5–14 min total on an H100
for 600 rows/95 labels at 1 epoch throughout, which is a reasonable
floor given the similar sample size here, but treat that as an
estimate, not a measurement, until these runs report real numbers.

## 🚀 **Full-Scale Runs**

Same generic-block pattern as the smoke test above, pointed at the
real datasets instead of the 600-row/150-label sample. **Not yet run**
— these commands haven't been executed against the full data on any
box; they're the smoke test's pattern applied forward, informed by
what the smoke test found, not independently validated at this scale.

**What changes from the sample commands, and why:**
- `--data` uses `${DATASET_BASE}_full`, not `${DATASET_BASE}-sample_full`
  — the real CSV (`eurlex-4k_full.csv` / `wiki10-31k_full.csv`) created
  by the one-time `_full` symlink step in each dataset section above,
  not the `create_amazon3m_sample.py`-built sample.
- `setup_plant_names.sh` is called with no third argument, so
  `BASE_DIR` takes its own default (`../tmp`), not the sample's
  `../tmp_sample` — keeping full runs and sample runs in separate
  trees, same isolation mechanism [described earlier](#isolating-sample-runs-avoiding-hub-pollution-chaining-stages-locally),
  just the other side of it.
- No `--skip_push_to_hub`, and stages chain through real Hub ids
  (`$STAGE1_HUB_ID`, `$STAGE4_HUB_ID_L2R`) instead of local paths
  (`$STAGE1_LOCAL`, `$STAGE4_LOCAL`) — a full run's checkpoints are the
  actual deliverable, not a disposable smoke artifact, so there's no
  reason to avoid the Hub here. Needs a valid `HF_TOKEN` /
  `HF_USERNAME` with write access; will create real repos under your
  namespace (see [Naming Conventions](#-naming-conventions) for exactly
  which ones).
- **Epoch counts follow the smoke test's confirmed-working
  combination**: Stage 1 at 1 epoch (never implicated by any of the
  six smoke combos), Stages 2/4/5 at 3 epochs (see
  ["Stage 4 needs enough epochs"](#️-stage-4-needs-enough-epochs--1-isnt-always-sufficient) —
  1 epoch on Stages 2/4/5 produced a false "PLANT loses" result on
  Wiki10-31K, and bumping only Stage 4 wasn't sufficient either).
  This combination was validated at 600-row/150-label smoke scale on
  both datasets and all three models, **not yet at full scale** —
  treat it as the best available starting point, not a guarantee.
  Watch Stage 4's own `ndcg@25` across epochs as a cheap early signal
  either way (see the same section for why that alone isn't sufficient
  either — Stage 5's own metrics still need watching directly).
- **`--rare_threshold` is now dataset-specific** instead of the smoke
  test's flat `5` — computed from each dataset's actual full-corpus
  label frequency distribution, not carried over from the sample. See
  ["Rare-label threshold, computed per dataset"](#rare-label-threshold-computed-per-dataset)
  right below for the numbers and the reasoning.

### Rare-label threshold, computed per dataset

`--rare_threshold` only affects evaluation bucketing (`rare_labels`
vs `not_rare_labels` in `build_label_or_cluster_vectors`, feeding
`rare-F1-micro` / `not_rare_precision` etc.) — it doesn't filter
training data or restrict the model, so getting it wrong doesn't break
a run, it just makes "rare" mean something different than intended.
The smoke test's `5` was picked for a 600-row/150-label sample where
almost every label is inherently sparse; it doesn't reflect either
dataset's actual full-corpus tail. Computed directly from each full
CSV's label frequency (label string → number of documents it appears
in):

| Quantity | Eurlex-4K (3,956 labels) | Wiki10-31K (30,938 labels) |
| --- | --- | --- |
| Min frequency | 1 | 2 |
| 25th percentile | 2 | 2 |
| **Median** | **6** | **3** |
| 75th percentile | 20 | 6 |
| Mean | 25.97 | 12.59 |
| Max frequency | 1,253 | 16,756 |

Both distributions are heavily right-skewed (a long tail of
infrequent labels plus a handful of very common ones pulling the mean
well above the median) — normal for XMC label spaces, and part of why
PSP@k exists as a metric in the first place. **Chosen threshold: each
dataset's own median frequency** — `6` for Eurlex-4K, `3` for
Wiki10-31K. This is a simple, reproducible, per-dataset definition
("below-median frequency counts as rare") rather than reusing one
number that would mean very different things for the two label
spaces: `--rare_threshold=5` on the full data would call 43.3% of
Eurlex-4K's labels rare but 66.6% of Wiki10-31K's — a majority for one
dataset, still a large minority for the other, for the same nominal
number.

| | Eurlex-4K @ threshold=6 | Wiki10-31K @ threshold=3 |
| --- | --- | --- |
| Rare labels | 1,900 (48.0%) | 12,580 (40.7%) |
| Not-rare labels | 2,056 (52.0%) | 18,358 (59.3%) |

Two honest notes on this choice:
- A median split puts roughly half of each label *vocabulary* in the
  "rare" bucket — that reads as a lot for a word usually meaning
  "uncommon," but it's standard for XMC's power-law label
  distributions: a large fraction of distinct labels each individually
  rare, together still a minority of total label *instances*. If a
  smaller, more colloquially-"rare" tail is wanted instead (e.g. the
  bottom quartile), Eurlex-4K's 25th-percentile threshold is `3`
  (28.6% rare); Wiki10-31K's is stuck at either `2` (0% rare — its
  minimum frequency *is* 2, so this threshold value is vacuous) or `3`
  (40.7% rare) — there's no integer in between, because so much of its
  tail sits at exactly 2–3 occurrences. That coarseness is itself a
  property of Wiki10-31K's label space, not a computation error.
- These frequencies come directly from the full CSVs
  (`eurlex-4k_full.csv` / `wiki10-31k_full.csv`), not from whatever
  `sample_df_with_full_label_coverage(..., frac=1.0)` produces
  internally — at `frac=1.0` that function is expected to keep every
  row, so the two should match closely, but this wasn't verified
  against the pipeline's own internal count.

```bash
cd ~/plant/scripts

DATASET_BASE="eurlex-4k"                          # or "wiki10-31k"
SOURCE_URL="XURLs.EURLEX_4K"                       # or "XURLs.WIKI10_31K"
MODEL="mistralai/Mistral-7B-Instruct-v0.3"         # or the Llama/Phi-3 id
RARE_THRESHOLD=6                                   # Eurlex-4K=6 (median), Wiki10-31K=3 (median) --
                                                    # see "Rare-label threshold, computed per dataset" above

source setup_plant_names.sh "${DATASET_BASE}_full" "$MODEL"
# -> DATASET_NAME=${DATASET_BASE}_full, DATASET_SHORT=${DATASET_BASE}, BASE_DIR=../tmp (default)

# --- Stage 1 -------------------------------------------------------------
launches/launchLLM \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE1_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=1 \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1

# --- Stage 2 (also writes labels_partition.json, needed by Stage 3) ------
launches/launchmultilabelclas \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE2_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-classify" --do_fresh_training \
    --num_train_epochs=3 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_25" --rare_threshold="$RARE_THRESHOLD" \
    --label_coverage_frac=1.0

# --- Stage 3 (needs labels_partition.json from Stage 2 above) ------------
launches/launch_l2rboot \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" \
    --base_dir="$BASE_DIR" --output_dir="$STAGE3_OUTPUT_DIR" \
    --hub_model_id="$STAGE3_HUB_ID" --bs=8 --chnk_sz=200 --max_length=1024

# --- Stage 4 (3 epochs) ---------------------------------------------------
launches/launch_l2r_training \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" \
    --data_l2r_fname_prefix="$DATASET_SHORT" --max_length=512 \
    --base_dir="$BASE_DIR" --l2r_boot_dir="$STAGE3_OUTPUT_DIR" \
    --output_dir="$STAGE4_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=3 --learning_rate=1e-4 --warmup_steps=0 \
    --metric_for_best_model="ndcg@25" \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1

# --- Stage 5 (hub_model_id = Stage 4's pushed L2R model) ------------------
launches/launchmultilabelclas \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE5_OUTPUT_DIR" \
    --hub_model_id="$STAGE4_HUB_ID_L2R" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-plantclassify" --do_fresh_training \
    --num_train_epochs=3 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_25" --rare_threshold="$RARE_THRESHOLD" \
    --label_coverage_frac=1.0
```

> ⚠️ **Phi-3 models only**: same note as the smoke test — nothing
> changes in the commands above for Phi-3-small/-mini/-medium; the
> LoRA `target_modules` selection is automatic based on
> `model.config.model_type`.

`--label_coverage_frac=1.0` is carried over unchanged from the smoke
test — it uses the entire dataset either way (it's not a subsampling
knob you'd lower here), kept for consistency with the smoke test's
flags rather than because it needs tuning at this scale.

**Stage 3 runtime will differ sharply between the two datasets, and
won't match the smoke test's ratio.** The smoke sample capped both
datasets to the same 150 labels, so Stage 3's MIG bootstrap took the
same one `--chnk_sz=200` pass for either. At full scale there's no
label cap: Eurlex-4K's 3,956 labels need ~20 chunks, Wiki10-31K's
30,938 labels need ~155 chunks — roughly **8× more Stage 3 chunking
work for Wiki10-31K than for Eurlex-4K**, on top of both datasets
having ~30× more rows than the sample. Budget Wiki10-31K's full run
accordingly; this isn't a bug, just a consequence of its much larger
label space, and `--chnk_sz` wasn't tuned or retested at this scale.

**No timing data yet for either dataset at full scale.** As a rough,
unvalidated extrapolation: Eurlex-4K's full CSV is ~32× the sample's
row count (19,314 vs 600), Wiki10-31K's is ~35× (20,762 vs 600), and
Stages 2/4/5 now run 3× the epochs the smoke test's *original*
1-epoch-everywhere baseline used — treat any back-of-envelope estimate
from the smoke-test timing as a low floor, not a prediction, until a
full run actually reports numbers.
