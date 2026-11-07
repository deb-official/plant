from xcube.imports import *

# Builds MIMIC-IV-few, a few-shot ICD-10 coding slice of mimic4_icd10_full.csv
# (shipped in XURLs.MIMIC4_DEMO), for the PLANT few-shot experiment -- see
# README_PLANT_FEW_SHOT.md. The idea: pick a set of "rare" target codes whose
# TRAINING-split frequency falls in a chosen range [min_freq, max_freq], then
# keep only the discharge summaries that carry at least one of those codes (any
# split), dropping every note that has none.
#
# By default (--label_scope=all) each kept note keeps its FULL original labels,
# so the label space is every ICD-10 code that co-occurs on the selected notes
# (~3K), a genuine extreme-multi-label task (~17 codes/note). The rare target
# codes (train freq in range) are what SELECTED the notes, but evaluation is
# done by bucketing every eval code by its TRAINING-split frequency into
# zero-shot (0) / few-shot (1..few_max) / head (>few_max) and scoring each
# bucket separately (see README_PLANT_FEW_SHOT.md) -- this is PLANT's native
# regime and lets us measure zero-shot and few-shot generalization directly.
# --label_scope=targets instead strips each note's labels to only the target
# codes (a ~200-code single-label few-shot task); useful, but collapses the
# zero/head buckets away.
#
# Note counts are quantized by the source CSV's frequency distribution (it is
# already curated -- see README_PLANT_FEW_SHOT.md): whole frequency bands come
# in or out together, so a pure range cannot hit an arbitrary note target. Pass
# --max_notes to cap the kept notes at ~that many for quick training: the kept
# set is deterministically subsampled (fixed --seed, proportional to each
# split's share). Some target codes may then lose all their notes; the script
# reports how many still appear so this is visible, not silent.

def compute_train_frequencies(csv_path, chunksize):
    "Count each label's frequency in the training split only, in one pass."
    counter = Counter()
    for chunk in pd.read_csv(csv_path, usecols=['labels', 'split'], chunksize=chunksize):
        train = chunk[chunk['split'] == 'train']
        for labels in train['labels'].astype(str):
            counter.update(l.strip() for l in labels.split(';') if l.strip())
    return counter

def proportional_subsample(kept, max_notes, seed):
    "Deterministically cut `kept` down to ~max_notes, keeping each split's share (largest-remainder allocation)."
    if len(kept) <= max_notes:
        return kept
    kept = kept.sort_values('note_id').reset_index(drop=True)   # stable, order-independent
    counts = kept['split'].value_counts()
    exact = {sp: max_notes * n / len(kept) for sp, n in counts.items()}
    alloc = {sp: int(v) for sp, v in exact.items()}
    remainder = max_notes - sum(alloc.values())
    for sp in sorted(exact, key=lambda s: exact[s] - alloc[s], reverse=True)[:remainder]:
        alloc[sp] += 1
    parts = [grp.sample(n=min(len(grp), alloc[sp]), random_state=seed)
             for sp, grp in kept.groupby('split')]
    return pd.concat(parts).sort_values('note_id').reset_index(drop=True)

def build_few(
    csv_path,             # Source CSV (mimic4_icd10_full.csv): text,labels,split,is_valid + metadata cols
    output_csv,           # Where to write the filtered few-shot CSV
    output_codes,         # Where to write the target-code list (one code per line, sorted)
    min_freq=3,           # Rare target codes: train frequency >= this ...
    max_freq=7,           # ... and <= this
    max_notes=800,        # Cap kept notes at ~this many (deterministic subsample); None/0 = no cap
    label_scope='all',    # 'all' = keep full labels (multi-label, bucketed eval); 'targets' = strip to target codes
    seed=42,              # Seed for the --max_notes subsample
    chunksize=200_000,    # Rows per pandas chunk when scanning the source CSV
):
    counter = compute_train_frequencies(csv_path, chunksize)
    print(f"Unique codes in training split: {len(counter)}")

    few = {c for c, n in counter.items() if min_freq <= n <= max_freq}
    if not few:
        raise ValueError(f"No codes have train freq in [{min_freq}, {max_freq}].")
    print(f"Selected {len(few)} target codes with train freq in [{min_freq}, {max_freq}]")
    output_codes.write_text("\n".join(sorted(few)) + "\n")

    # Keep notes (any split) carrying >=1 target code; add num_few_codes = how
    # many of the note's labels are targets. With label_scope='targets' (default)
    # each kept note's `labels` are also *stripped to only the target codes*, so
    # the label space is the ~200 few-shot codes -- this makes every downstream
    # metric (P@k, F1, etc.) measure the few-shot objective directly instead of
    # being dominated by common co-occurring head codes. 'all' keeps full labels.
    kept_chunks, total = [], 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        total += len(chunk)
        codes = chunk['labels'].astype(str).map(
            lambda s: [l.strip() for l in s.split(';') if l.strip()])
        nfew = codes.map(lambda cs: sum(c in few for c in cs))
        keep = nfew > 0
        sub = chunk[keep].copy()
        sub['num_few_codes'] = nfew[keep]
        if label_scope == 'targets':
            sub['labels'] = codes[keep].map(
                lambda cs: ';'.join(sorted(c for c in cs if c in few)))
        kept_chunks.append(sub)
    kept = pd.concat(kept_chunks, ignore_index=True)
    print(f"Notes with >=1 target code: {len(kept)} / {total}  (label_scope={label_scope})")

    if max_notes:
        kept = proportional_subsample(kept, max_notes, seed)
        print(f"Capped to {len(kept)} notes (--max_notes={max_notes}, --seed={seed})")

    assert (kept['num_few_codes'] >= 1).all()
    kept.to_csv(output_csv, index=False)

    # Transparency: how many target codes survive in the (possibly capped) note set.
    present = set()
    for s in kept['labels'].astype(str):
        present.update(l.strip() for l in s.split(';') if l.strip() in few)
    print(f"Target codes still represented: {len(present)} / {len(few)}")
    print(kept['split'].value_counts().to_string())
    print(f"Wrote {len(kept)} notes to {output_csv}")
    print(f"Wrote {len(few)} target codes to {output_codes}")

def main(
    source_url: str = "XURLs.MIMIC4_DEMO",
    csv_name: str = "mimic4_icd10_full",       # basename (no .csv) of the source file inside the downloaded dir
    output_name: str = "mimic4_icd10_few",     # basename (no .csv) for the few-shot CSV
    min_freq: int = 3,
    max_freq: int = 7,
    max_notes: int = 800,
    label_scope: str = "all",
    seed: int = 42,
    chunksize: int = 200_000,
):
    source_dir = untar_xxx(eval(source_url))
    csv_path = source_dir / f"{csv_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found -- download it first, e.g. `untar_xxx({source_url})`")
    build_few(
        csv_path,
        source_dir / f"{output_name}.csv",
        source_dir / f"{output_name}_codes.txt",
        min_freq=min_freq, max_freq=max_freq, max_notes=max_notes,
        label_scope=label_scope, seed=seed, chunksize=chunksize,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MIMIC-IV-few, a few-shot ICD-10 slice of mimic4_icd10_full.csv for the PLANT pipeline.")
    parser.add_argument("--source_url", type=str, default="XURLs.MIMIC4_DEMO", help="XURLs constant to resolve/download the source dataset (default: XURLs.MIMIC4_DEMO)")
    parser.add_argument("--csv_name", type=str, default="mimic4_icd10_full", help="Basename (no .csv) of the source CSV inside the downloaded dir (default: mimic4_icd10_full)")
    parser.add_argument("--output_name", type=str, default="mimic4_icd10_few", help="Basename (no .csv) for the output CSV; a matching _codes.txt is written alongside (default: mimic4_icd10_few)")
    parser.add_argument("--min_freq", type=int, default=3, help="Rare target codes have train frequency >= this (default: 3)")
    parser.add_argument("--max_freq", type=int, default=7, help="Rare target codes have train frequency <= this (default: 7)")
    parser.add_argument("--max_notes", type=int, default=800, help="Cap kept notes at ~this many via deterministic, split-proportional subsample; 0 = no cap (default: 800)")
    parser.add_argument("--label_scope", type=str, default="all", choices=["targets", "all"], help="'all' keeps full original labels (multi-label, zero/few/head bucketed eval); 'targets' strips to only the target codes (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the --max_notes subsample (default: 42)")
    parser.add_argument("--chunksize", type=int, default=200_000, help="Rows per pandas chunk when scanning the source CSV (default: 200000)")
    args = parser.parse_args()
    main(**vars(args))
