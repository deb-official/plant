#!/usr/bin/env python3
"""Training-free (zero-/few-shot) LLM baseline for the MIMIC-IV-few PLANT
experiment -- see README_PLANT_FEW_SHOT.md, section "Training-free LLM baseline".

WHY: to set PLANT's rare/zero-shot improvements in context against what the SAME
off-the-shelf LLM can do with NO task training at all. The trained baseline
(Stage 2) structurally cannot surface codes it never saw in training; this
pipeline shows how much of PLANT's rare-code advantage is reachable training-free.

SCHEME (per test note):
  1. EXTRACT diagnosis/procedure mention phrases with an off-the-shelf *Instruct*
     LLM -- NO target-label list in the prompt. Two prompting regimes:
       --mode zero : instruction only (0 in-context examples)
       --mode few  : instruction + 2 hand-authored worked examples
  2. LINK each mention to an ICD-10 code by SapBERT nearest-neighbour over the
     target-code descriptions (MIMIC-IV demo d_icd dictionaries -- the exact
     coding vocabulary; auto-downloaded from PhysioNet's OPEN-access demo).
  3. SCORE: build a DENSE (n_notes x n_labels) similarity matrix where
       s[note, code] = max over the note's mentions of cosine(mention, code_desc)
     then feed that (as "probs") + the gold multi-hot through a FAITHFUL COPY of
     the PLANT eval's bucketed precision@k / recall@k (multilabel_classify.py,
     compute_metrics_with_labels) so numbers are directly comparable to the
     Stage 2 baseline and the Stage 5 PLANT head.

TERMINOLOGY -- two independent axes both named "zero/few" (see README):
  * evaluation BUCKET  = partition of the LABEL SPACE by TRAIN frequency
                         (zero = 0 train occ, few = 1..rare_threshold-1, head = >=).
  * prompting REGIME   = number of in-context EXAMPLES given to the LLM extractor
                         (--mode zero vs --mode few). Applies to THIS baseline only.
  Every regime is scored on every bucket.

CACHING: description embeddings -> <cache>/desc_emb.npy; extracted mentions ->
<cache>/ments_<model>_<mode>.json; metrics -> <cache>/metrics_<model>_<mode>.json.
Re-run `--do score` after editing the mention filter without re-running the LLM.

Deps: transformers, torch (+flash-attn), scikit-learn, pandas, numpy. Extra
packages some base models need (Llama HF gating; Phi-3-small tiktoken/pytest) are
documented in the README; this baseline paper uses Mistral-7B and Llama-3.1-8B.

USAGE:
  python train_free_baseline.py --model mistralai/Mistral-7B-Instruct-v0.3 --mode zero
  python train_free_baseline.py --model meta-llama/Llama-3.1-8B-Instruct   --mode few
See scripts/run_train_free_baseline.sh for the exact 4-combo reproduction driver.
"""
import os, re, json, argparse, urllib.request
import numpy as np, pandas as pd

SAPBERT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
K_VALUES = [1, 5, 8, 15, 25]
# PhysioNet MIMIC-IV *demo* is open-access (no credentials) and ships the exact
# d_icd dictionaries used to code these notes -> 100% description coverage.
PHYSIONET = "https://physionet.org/files/mimic-iv-demo/2.2/hosp/{}.csv.gz"

# ---------------------------------------------------------------- data helpers
def load_classes_and_buckets(partition):
    p = json.load(open(partition))
    classes = p["classes"]                          # canonical column order
    idx = {c: i for i, c in enumerate(classes)}
    zero = sorted(idx[c] for c in p["zero"])
    rare = sorted(idx[c] for c in p["rare"])        # zero + few
    not_rare = sorted(idx[c] for c in p["not_rare"])  # head
    return classes, idx, zero, rare, not_rare

def fetch_dicts(cache):
    paths = []
    for name in ("d_icd_diagnoses", "d_icd_procedures"):
        dst = os.path.join(cache, f"{name}.csv.gz")
        if not os.path.exists(dst):
            print(f"downloading {name} from PhysioNet (open-access demo)...", flush=True)
            urllib.request.urlretrieve(PHYSIONET.format(name), dst)
        paths.append(dst)
    return paths

def load_descriptions(classes, cache):
    d = {}
    for f in fetch_dicts(cache):
        df = pd.read_csv(f, dtype=str)
        df = df[df.icd_version == "10"]
        for c, t in zip(df.icd_code, df.long_title):
            d[c.strip()] = str(t)
    norm = lambda c: c.replace(".", "").strip()
    missing = [c for c in classes if norm(c) not in d]
    if missing:
        raise SystemExit(f"{len(missing)} target codes lack a description, e.g. {missing[:5]}")
    return [d[norm(c)] for c in classes]

def load_test_notes(data_csv):
    df = pd.read_csv(data_csv)
    return df[df.is_valid == True].reset_index(drop=True)   # 167 held-out notes

def gold_matrix(df, idx):
    Y = np.zeros((len(df), len(idx)), dtype=np.int32)
    for i, lab in enumerate(df["labels"].fillna("")):
        for c in str(lab).split(";"):
            c = c.strip()
            if c in idx:
                Y[i, idx[c]] = 1
    return Y

# ---------------------------------------------------------------- SapBERT linker
def sapbert_encode(texts, batch=256, max_length=64):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(SAPBERT)
    mdl = AutoModel.from_pretrained(SAPBERT).cuda().eval().half()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i+batch], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to("cuda")
            cls = mdl(**enc).last_hidden_state[:, 0, :]   # [CLS] -- SapBERT convention
            cls = torch.nn.functional.normalize(cls, dim=1)
            out.append(cls.float().cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 768), np.float32)

def desc_embeddings(classes, cache):
    p = os.path.join(cache, "desc_emb.npy")
    if os.path.exists(p):
        return np.load(p)
    E = sapbert_encode(load_descriptions(classes, cache))
    np.save(p, E)
    return E

# ---------------------------------------------------------------- LLM extraction
ZERO_INSTR = (
    "You are an expert clinical coder. Read the hospital discharge summary and "
    "list EVERY distinct medical condition (diagnosis) AND every procedure or "
    "operation documented for this patient. Write one item per line as a short "
    "clinical phrase (the kind of term a coder would look up in an index). "
    "Include both diagnoses and procedures. Do NOT output codes, numbers, or "
    "explanations -- only the phrases, one per line."
)

def fewshot_exemplars():
    """Hand-authored, generic exemplars demonstrating the FORMAT (short clinical
    phrases, diagnoses + procedures) without injecting echo-prone code
    descriptions. NB: an earlier version fed train notes' *gold-code long_titles*
    as target lines; Llama-3.1-8B then parroted those verbatim PCS descriptions
    (e.g. "Insertion of Infusion Device into Superior Vena Cava...") into every
    note -- exemplar contamination. Short generic mentions avoid that and are
    identical across models. See README 'Errors we hit' (training-free)."""
    return [
        ("A 68-year-old man was admitted with crushing substernal chest pain. "
         "Troponin was elevated and the ECG showed ST elevations. He underwent "
         "cardiac catheterization and a drug-eluting stent was placed in the LAD. "
         "His course was complicated by acute kidney injury that resolved with "
         "hydration. He has type 2 diabetes and hypertension.",
         ["Chest pain", "ST-elevation myocardial infarction",
          "Cardiac catheterization", "Coronary stent placement",
          "Acute kidney injury", "Type 2 diabetes mellitus", "Hypertension"]),
        ("An 82-year-old woman presented with altered mental status and fever. "
         "She was found to have a urinary tract infection with sepsis and was "
         "started on intravenous antibiotics. A central venous catheter was placed "
         "for access. Her history includes atrial fibrillation on warfarin and "
         "chronic kidney disease.",
         ["Altered mental status", "Urinary tract infection", "Sepsis",
          "Insertion of central venous catheter", "Atrial fibrillation",
          "Chronic kidney disease"]),
    ]

def make_prompt(note_text, mode, tok, note_chars=16000):
    note = re.sub(r"\s+", " ", str(note_text))[:note_chars]
    if mode == "few":
        demo = ZERO_INSTR + "\n\nHere are worked examples of the task:\n"
        for note_x, lines in fewshot_exemplars():
            demo += ("\nDischarge summary:\n\"\"\"\n" + note_x +
                     "\n\"\"\"\nList:\n" + "\n".join(lines) + "\n")
        demo += ("\nNow do the same for the following discharge summary.\n\n"
                 "Discharge summary:\n\"\"\"\n" + note + "\n\"\"\"\nList:")
        msgs = [{"role": "user", "content": demo}]
    else:
        msgs = [{"role": "user", "content": ZERO_INSTR +
                 "\n\nDischarge summary:\n\"\"\"\n" + note + "\n\"\"\"\nList:"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

_JUNK_PREFIX = ("here is", "here are", "here's", "the following", "below is",
                "below are", "these are", "based on", "sure", "certainly",
                "of course", "i have", "i've", "this is a list", "list of",
                "the list", "note:", "please note", "in summary", "to summarize")
_JUNK_SUB = ("list of distinct medical", "medical conditions and procedures",
             "documented for this patient", "discharge summary")

def is_junk(m):
    """Drop LLM meta/preamble lines (e.g. 'Here is the list of distinct medical
    conditions and procedures') that are not clinical mentions -- they slip past a
    length filter because they're long, so match by prefix/substring."""
    low = m.lower()
    if low.startswith(_JUNK_PREFIX) or any(s in low for s in _JUNK_SUB):
        return True
    if len(m) < 20 and low.rstrip(":").strip() in (
            "diagnoses", "diagnosis", "procedures", "procedure", "conditions",
            "medications", "none", "list"):
        return True
    return False

def parse_mentions(text, cap=60):
    out, seen = [], set()
    for ln in text.splitlines():
        ln = re.sub(r"^[\-\*•\d\.\)\s]+", "", ln.strip()).strip(" .;:-")
        if not ln or len(ln) < 3 or is_junk(ln):
            continue
        key = ln.lower()
        if key not in seen:
            seen.add(key); out.append(ln)
    return out[:cap]

def tag(model_id):
    return model_id.split("/")[-1].replace(".", "-")

def extract(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    df = load_test_notes(a.data_csv)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2").eval()
    results = {}
    for i, r in df.iterrows():
        prompt = make_prompt(r["text"], a.mode, tok)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=30000).to("cuda")
        with torch.no_grad():
            gen = mdl.generate(**enc, max_new_tokens=512, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        txt = tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        ments = parse_mentions(txt)
        results[str(r["note_id"])] = ments
        if i < 2 or i % 40 == 0:
            print(f"[{i+1}/{len(df)}] {r['note_id']}: {len(ments)} mentions; "
                  f"e.g. {ments[:4]}", flush=True)
    out = os.path.join(a.cache_dir, f"ments_{tag(a.model)}_{a.mode}.json")
    json.dump(results, open(out, "w"), indent=1)
    print("saved", out, "| avg mentions/note:",
          round(np.mean([len(v) for v in results.values()]), 1))

# ---------------------------------------------------------------- metric (faithful copy)
def bucketed_metrics(probs, labels, rare, not_rare, zero):
    """Faithful copy of multilabel_classify.py compute_metrics_with_labels: each
    bucket restricts BOTH the ranking and the gold to that bucket's columns
    (rank within bucket); precision@k = mean(tp/k), recall@k averaged over ALL
    notes (0 for a note with no true code in the bucket)."""
    from sklearn.metrics import f1_score
    m = {"f1_micro": f1_score(labels, (probs > 0.5).astype(int),
                              average="micro", zero_division=0)}
    def pk_rk(P, L, k):
        topk = np.argsort(P, axis=1)[:, -k:]
        pred = np.zeros_like(L)
        for i in range(len(L)):
            pred[i, topk[i]] = 1
        tp = np.sum(pred * L, axis=1)
        cnt = np.sum(L, axis=1)
        return float(np.mean(tp / k)), float(np.mean(np.where(cnt > 0, tp / cnt, 0)))
    for k in K_VALUES:
        m[f"precision_at_{k}"], m[f"recall_at_{k}"] = pk_rk(probs, labels, k)
    zset = set(zero)
    for name, bidx in (("zero", sorted(zset)),
                       ("few", [i for i in rare if i not in zset]),
                       ("head", list(not_rare))):
        for k in K_VALUES:
            p, r = pk_rk(probs[:, bidx], labels[:, bidx], k)
            m[f"{name}_precision_at_{k}"], m[f"{name}_recall_at_{k}"] = p, r
    return m

def score(a):
    classes, idx, zero, rare, not_rare = load_classes_and_buckets(a.partition)
    df = load_test_notes(a.data_csv)
    Y = gold_matrix(df, idx)
    E = desc_embeddings(classes, a.cache_dir)
    ments = json.load(open(os.path.join(a.cache_dir, f"ments_{tag(a.model)}_{a.mode}.json")))
    # re-apply junk filter here so cached extractions clean up without re-running the LLM
    per_note = [[m for m in ments.get(str(nid), []) if not is_junk(m)]
                for nid in df["note_id"]]
    flat, spans = [], []
    for lst in per_note:
        spans.append((len(flat), len(flat) + len(lst))); flat.extend(lst)
    M = sapbert_encode(flat, max_length=32)
    S = np.zeros((len(df), len(classes)), dtype=np.float32)
    for i, (s, e) in enumerate(spans):
        if e > s:
            S[i] = (M[s:e] @ E.T).max(axis=0)          # dense best-mention cosine per code
    m = bucketed_metrics(S, Y, rare, not_rare, zero)
    empty = sum(1 for s, e in spans if e == s)
    print(f"\n===== TRAINING-FREE  {tag(a.model)}  mode={a.mode} =====")
    print(f"notes={len(df)}  avg mentions/note={np.mean([e-s for s,e in spans]):.1f}  "
          f"notes-with-0-mentions={empty}")
    print("  headline precision@(bucket median):")
    for name, k in (("zero", 1), ("few", 5), ("head", 8), ("overall", 15)):
        key = f"precision_at_{k}" if name == "overall" else f"{name}_precision_at_{k}"
        print(f"    {name:8} P@{k:<2} = {m[key]:.4f}")
    json.dump({k: float(v) for k, v in m.items()},
              open(os.path.join(a.cache_dir, f"metrics_{tag(a.model)}_{a.mode}.json"), "w"),
              indent=1)

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id of an off-the-shelf Instruct LLM")
    ap.add_argument("--mode", choices=["zero", "few"], required=True,
                    help="prompting regime: 0 or 2 in-context examples")
    ap.add_argument("--do", choices=["extract", "score", "all"], default="all")
    ap.add_argument("--data_csv",
                    default=os.path.expanduser("~/.xcube/data/mimic4_demo/mimic4_icd10_few.csv"),
                    help="MIMIC-IV-few CSV built by create_mimic4_few.py")
    ap.add_argument("--partition",
                    default=os.path.expanduser("~/plant/tmp_few_all/MIMIC4_DEMO/labels_partition.json"),
                    help="labels_partition.json emitted by the PLANT pipeline (bucket defs)")
    ap.add_argument("--cache_dir", default="./train_free_cache",
                    help="descriptions / embeddings / mentions / metrics cache")
    a = ap.parse_args()
    os.makedirs(a.cache_dir, exist_ok=True)
    if a.do in ("extract", "all"):
        extract(a)
    if a.do in ("score", "all"):
        score(a)
