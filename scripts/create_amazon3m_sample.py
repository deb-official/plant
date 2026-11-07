from xcube.imports import *

# Builds a small, frequency-filtered sample of Amazon-3M for smoke-testing the
# PLANT pipeline (see README_AMAZON3M.md) before committing to a full-scale
# GPU run. A plain random row sample would not help here: Amazon-3M's labels
# are ~2.8M mostly-unique "related product" ids, so a random subset of rows
# still has a huge label vocabulary. This instead fixes the label vocabulary
# first (top-k most frequent labels globally) and only keeps rows that still
# have at least one label after restricting to that vocabulary, so the
# resulting sample has a small, bounded number of unique labels -- keeping
# Stage 3's MIG bootstrap to a single label-chunk pass (chnk_sz=200 default)
# and the classifier heads in Stages 2/4/5 small.

def compute_label_frequencies(csv_path, chunksize):
    "Count label frequency across the whole `csv_path` in one pass."
    counter = Counter()
    for chunk in pd.read_csv(csv_path, usecols=['labels'], chunksize=chunksize):
        for labels in chunk['labels'].astype(str):
            counter.update(labels.split(';'))
    return counter

def build_sample(
    csv_path,           # Source CSV with text,labels,is_valid columns (';'-delimited labels)
    output_path,        # Where to write the filtered sample CSV
    top_k_labels=150,   # Keep only this many most-frequent labels (< chnk_sz keeps Stage 3 to one pass)
    target_rows=600,    # Final sample size
    valid_frac=0.10,    # Fraction of the sample marked is_valid=True (eval split)
    pool_multiplier=4,  # Collect this many x target_rows as a candidate pool before subsampling
    chunksize=300_000,  # Rows per pandas chunk when scanning the (potentially huge) source CSV
    seed=42,
):
    random.seed(seed)

    counter = compute_label_frequencies(csv_path, chunksize)
    top_labels = set(l for l, _ in counter.most_common(top_k_labels))
    print(f"Selected top {len(top_labels)} labels by global frequency "
          f"(min freq among them: {min(counter[l] for l in top_labels)})")

    kept_rows = []
    pool_target = target_rows * pool_multiplier
    for ci, chunk in enumerate(pd.read_csv(csv_path, usecols=['text', 'labels', 'is_valid'], chunksize=chunksize)):
        for text, labels, is_valid in zip(chunk['text'], chunk['labels'].astype(str), chunk['is_valid']):
            filtered = [l for l in labels.split(';') if l in top_labels]
            if not filtered:
                continue
            kept_rows.append((text, ';'.join(filtered), is_valid))
        print(f"chunk {ci}: kept so far = {len(kept_rows)}")
        if len(kept_rows) >= pool_target:
            break

    if len(kept_rows) < target_rows:
        raise ValueError(
            f"Only found {len(kept_rows)} candidate rows with >=1 label in the "
            f"top-{top_k_labels} vocabulary, need {target_rows}. Increase "
            f"top_k_labels or pool_multiplier."
        )

    random.shuffle(kept_rows)
    sample = kept_rows[:target_rows]

    # Re-derive is_valid ourselves (deterministic, exact split) rather than
    # trust the original column, since heavy label-filtering can skew it.
    n_valid = max(1, int(len(sample) * valid_frac))
    is_valid_flags = [False] * (len(sample) - n_valid) + [True] * n_valid
    random.shuffle(is_valid_flags)

    df_out = pd.DataFrame({
        'text': [r[0] for r in sample],
        'labels': [r[1] for r in sample],
        'is_valid': is_valid_flags,
    })
    df_out.to_csv(output_path, index=False)

    final_labels = set(l for labels in df_out['labels'] for l in labels.split(';'))
    print(f"Wrote {len(df_out)} rows to {output_path}")
    print(f"Final unique label count in sample: {len(final_labels)}")
    print(f"Train rows: {(~df_out['is_valid']).sum()}, Eval rows: {df_out['is_valid'].sum()}")
    print(df_out['labels'].str.split(';').str.len().describe())

def main(
    source_url: str = "XURLs.AMAZON_3M",
    csv_name: str = "amazon-3m",             # basename (no .csv) of the source file inside the downloaded dir
    output_name: str = "amazon-3m-sample_full",  # basename (no .csv) for the sample; keep the "_full" suffix -- see README_AMAZON3M.md's naming-convention section
    top_k_labels: int = 150,
    target_rows: int = 600,
    valid_frac: float = 0.10,
    pool_multiplier: int = 4,
    chunksize: int = 300_000,
    seed: int = 42,
):
    source_dir = untar_xxx(eval(source_url))
    csv_path = source_dir / f"{csv_name}.csv"
    output_path = source_dir / f"{output_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found -- download it first, e.g. `untar_xxx({source_url})`")

    build_sample(
        csv_path, output_path,
        top_k_labels=top_k_labels, target_rows=target_rows, valid_frac=valid_frac,
        pool_multiplier=pool_multiplier, chunksize=chunksize, seed=seed,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a small, frequency-filtered Amazon-3M sample for smoke-testing the PLANT pipeline.")
    parser.add_argument("--source_url", type=str, default="XURLs.AMAZON_3M", help="XURLs constant to resolve/download the source dataset (default: XURLs.AMAZON_3M)")
    parser.add_argument("--csv_name", type=str, default="amazon-3m", help="Basename (no .csv) of the source CSV inside the downloaded dir (default: amazon-3m)")
    parser.add_argument("--output_name", type=str, default="amazon-3m-sample_full", help="Basename (no .csv) for the output sample CSV (default: amazon-3m-sample_full)")
    parser.add_argument("--top_k_labels", type=int, default=150, help="Keep only this many most-frequent labels; keep < chnk_sz (default 200) for a single-pass Stage 3 (default: 150)")
    parser.add_argument("--target_rows", type=int, default=600, help="Final sample size (default: 600)")
    parser.add_argument("--valid_frac", type=float, default=0.10, help="Fraction of the sample marked is_valid=True (default: 0.10)")
    parser.add_argument("--pool_multiplier", type=int, default=4, help="Collect this many x target_rows as a candidate pool before subsampling (default: 4)")
    parser.add_argument("--chunksize", type=int, default=300_000, help="Rows per pandas chunk when scanning the source CSV (default: 300000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    main(**vars(args))
