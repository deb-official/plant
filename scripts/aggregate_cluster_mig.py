from xcube.imports import *

# Phase 3's Stage 4 router needs one combined `_tok_rank_per_lbl.ft`-shaped
# feather file at cluster granularity, but Phase 4's per-cluster Stage 3 run
# (`bootl2r.py --label_space=cluster`) writes one *separate* file set per
# cluster instead (see README_AMAZON3M.md / the full-scale
# candidate-generator plan). This script aggregates those per-cluster
# outputs into a single combined file set, using the SAME three-file naming
# convention `learning2rank.py` already expects (`{prefix}_tok_rank_per_lbl.ft`,
# `{prefix}_tok.ft`, `{prefix}_lbl.ft`) -- so Stage 4's existing data-loading
# code needs zero changes; just point its `--data_l2r_fname_prefix` at this
# script's `--output_prefix` instead of the original per-label prefix.
#
# Aggregation: for each cluster, a token's per-cluster score is the MAX of
# its per-label bcx_mutual_info within that cluster ("how informative is
# this token for detecting ANY label in this cluster" -- the natural choice
# for a coarse router). Per-token ranks are then recomputed across clusters
# (not copied from the per-label ranks, which were relative to a different,
# larger set of columns).


def load_cluster_tok_rank(l2r_boot_dir, fname, cluster_id):
    path = os.path.join(l2r_boot_dir, f"{fname}_cluster{cluster_id}_tok_rank_per_lbl.ft")
    return pd.read_feather(path)


def aggregate_cluster_scores(l2r_boot_dir, fname, num_clusters):
    """For every cluster, collapse its per-label bcx_mutual_info into one
    per-token max score. Returns columns ['token', 'label', 'bcx_mutual_info']
    where 'label' is actually the cluster id -- kept as 'label' so
    learning2rank.py's existing schema expectations are met unchanged."""
    rows = []
    for cluster_id in range(num_clusters):
        df = load_cluster_tok_rank(l2r_boot_dir, fname, cluster_id)
        per_token_max = df.groupby("token")["bcx_mutual_info"].max().reset_index()
        per_token_max["label"] = cluster_id
        rows.append(per_token_max)
    return pd.concat(rows, ignore_index=True)


def add_ranks(combined):
    """Recompute per-cluster-column token ranks (descending bcx_mutual_info,
    0-indexed, rank 0 = highest), matching bootl2r.py's
    compute_token_label_ranks semantics."""
    combined = combined.sort_values(["label", "token"]).reset_index(drop=True)
    combined["rank"] = (
        combined.groupby("label")["bcx_mutual_info"].rank(method="first", ascending=False).astype(int) - 1
    )
    return combined


def main(
    l2r_boot_dir: str,             # dir containing bootl2r.py --label_space=cluster's per-cluster output files
    fname: str,                    # the {fname} prefix bootl2r.py used (e.g. "amazon-3m-sample")
    output_prefix: str,            # new combined-file prefix -- pass this as learning2rank.py's --data_l2r_fname_prefix
    num_clusters: int = None,      # inferred from base_dir's cluster_assignments.json if not given
    base_dir: str = None,          # only needed to infer num_clusters when not given explicitly
):
    if num_clusters is None:
        assert base_dir is not None, "Provide --num_clusters or --base_dir (to read cluster_assignments.json)"
        with open(os.path.join(base_dir, "cluster_assignments.json"), "r") as f:
            num_clusters = json.load(f)["num_clusters"]
    print(f"Aggregating {num_clusters} clusters' per-label MIG scores into one per-cluster score set...")

    combined = aggregate_cluster_scores(l2r_boot_dir, fname, num_clusters)
    combined = add_ranks(combined)
    combined = combined[["token", "label", "bcx_mutual_info", "rank"]]

    out_tok_rank_path = os.path.join(l2r_boot_dir, f"{output_prefix}_tok_rank_per_lbl.ft")
    combined.to_feather(out_tok_rank_path)
    print(f"Wrote {out_tok_rank_path} ({len(combined)} rows)")

    # _tok.ft is identical across clusters (same tokenizer/vocab every time) -- reuse cluster0's verbatim.
    src_tok_path = os.path.join(l2r_boot_dir, f"{fname}_cluster0_tok.ft")
    out_tok_path = os.path.join(l2r_boot_dir, f"{output_prefix}_tok.ft")
    shutil.copyfile(src_tok_path, out_tok_path)
    print(f"Copied {src_tok_path} -> {out_tok_path}")

    df_lbs = pd.DataFrame({"lbl": range(num_clusters), "lbl_val": [str(c) for c in range(num_clusters)]})
    out_lbl_path = os.path.join(l2r_boot_dir, f"{output_prefix}_lbl.ft")
    df_lbs.to_feather(out_lbl_path)
    print(f"Wrote {out_lbl_path} ({len(df_lbs)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate bootl2r.py --label_space=cluster's per-cluster MIG outputs into one combined cluster-level file set for Stage 4's router.")
    parser.add_argument("--l2r_boot_dir", type=str, required=True, help="Stage 3 output dir containing the per-cluster {fname}_cluster{id}_tok_rank_per_lbl.ft files.")
    parser.add_argument("--fname", type=str, required=True, help="The {fname} prefix bootl2r.py used (e.g. 'amazon-3m-sample').")
    parser.add_argument("--output_prefix", type=str, required=True, help="New combined-file prefix -- pass this as learning2rank.py's --data_l2r_fname_prefix.")
    parser.add_argument("--num_clusters", type=int, default=None, help="Number of clusters (inferred from --base_dir's cluster_assignments.json if not given).")
    parser.add_argument("--base_dir", type=str, default=None, help="Dir containing cluster_assignments.json, only needed if --num_clusters is not given.")
    args = parser.parse_args()
    main(**vars(args))
