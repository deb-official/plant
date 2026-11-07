from xcube.imports import *
from scipy.sparse import vstack
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sentence_transformers import SentenceTransformer

# Phase 1 of the full-scale candidate-generator/hard-negative-mining subsystem
# (see README_AMAZON3M.md's "2.8M-label problem" section). Amazon-3M's 2.81M
# labels are too many for Stages 2/4/5 to score densely against every
# document -- this script groups labels into a flat set of clusters so a
# lightweight router can shortlist a handful of clusters per document, and the
# fine-grained stages only ever score labels within those clusters (true
# positives + same-cluster hard negatives) instead of the whole vocabulary.
#
# Requires `labels_partition.json` to already exist under `base_dir` (written
# by Stage 2's `preprocess_data`, see multilabel_classify.py) -- this is the
# same hard dependency Stage 3's MIG bootstrap already has on that file, and
# for the same reason: it fixes the label vocabulary and its column order,
# which `cluster_assignments.json` reuses verbatim so the two files stay
# index-aligned.
#
# Amazon-3M labels have no text of their own (they're opaque related-product
# ids), so each label's "identity" for clustering is reconstructed from the
# documents that carry it: labels_f = normalize(Y^T @ X), where Y is the
# sparse (doc x label) indicator and X is each document's SentenceTransformer
# embedding. This is the same construction AttentionXML's label tree uses
# (~/AttentionXML/deepxml/cluster.py), with dense sentence embeddings standing
# in for AttentionXML's TF-IDF document features. Clustering itself is a flat
# (single-level, not a multi-level cascade) recursive balanced 2-means,
# adapted from that same file's `split_node`.


def load_mlb_classes(base_dir):
    "Load the label vocabulary + column order fixed by Stage 2's preprocess_data."
    labels_file = os.path.join(base_dir, "labels_partition.json")
    with open(labels_file, "r") as f:
        return json.load(f)["classes"]


def build_doc_label_matrix(csv_path, mlb_classes, chunksize=300_000):
    "Sparse (num_docs, num_labels) doc-label indicator, built one CSV chunk at a time -- never densified."
    mlb = MultiLabelBinarizer(classes=mlb_classes, sparse_output=True)
    mlb.fit([])  # classes are fixed; no label data needed to determine them
    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=["labels"], chunksize=chunksize):
        labels = chunk["labels"].astype(str).apply(lambda x: x.split(";") if x else [])
        chunks.append(mlb.transform(labels))
    return vstack(chunks).tocsr()


def embed_documents(texts, model_name="all-MiniLM-L6-v2", batch_size=256, device=None):
    "One batch-inference pass over every document's text."
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return np.asarray(embeddings, dtype=np.float32)


def label_features(Y, X):
    "labels_f = normalize(Y^T @ X): each label's feature vector = normalized sum of its documents' embeddings."
    return normalize(Y.T @ X)


def split_node(label_idx, label_feats, eps, rng):
    """Balanced 2-means split of one label cluster into two, by cosine similarity.
    Adapted from ~/AttentionXML/deepxml/cluster.py:split_node."""
    n = len(label_idx)
    c1, c2 = rng.choice(n, 2, replace=False)
    centers = label_feats[[c1, c2]]
    old_sim, new_sim = -1e9, -1.0
    left_idx = right_idx = None
    while new_sim - old_sim >= eps:
        sims = label_feats @ centers.T  # (n, 2)
        order = np.argsort(sims[:, 1] - sims[:, 0])
        left_idx, right_idx = order[: n // 2], order[n // 2:]
        old_sim, new_sim = new_sim, (sims[left_idx, 0].sum() + sims[right_idx, 1].sum()) / n
        centers = normalize(np.stack([
            label_feats[left_idx].sum(axis=0),
            label_feats[right_idx].sum(axis=0),
        ]))
    return (label_idx[left_idx], label_feats[left_idx]), (label_idx[right_idx], label_feats[right_idx])


def build_flat_clusters(label_feats, target_cluster_size, eps=1e-4, seed=42):
    """Recursive balanced 2-means, stopping each branch once its leaf size is
    <= target_cluster_size. Returns a cluster_id per label (0..num_clusters-1),
    aligned with label_feats' row order -- i.e. with mlb_classes' order."""
    rng = np.random.default_rng(seed)
    num_labels = label_feats.shape[0]
    queue = [(np.arange(num_labels), label_feats)]
    leaves = []
    while queue:
        idx, feats = queue.pop()
        if len(idx) <= target_cluster_size:
            leaves.append(idx)
            continue
        left, right = split_node(idx, feats, eps, rng)
        queue.append(left)
        queue.append(right)

    cluster_ids = np.empty(num_labels, dtype=np.int64)
    for cluster_id, idx in enumerate(leaves):
        cluster_ids[idx] = cluster_id
    return cluster_ids, len(leaves)


def main(
    base_dir: str,                          # dir containing labels_partition.json (Stage 2 output)
    csv_path: str,                          # source CSV with text,labels columns (same corpus Stage 2 ran against)
    target_cluster_size: int = 1000,        # recursion stops once a cluster's label count is <= this
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    chunksize: int = 300_000,
    seed: int = 42,
    device: str = None,
):
    mlb_classes = load_mlb_classes(base_dir)
    print(f"Loaded {len(mlb_classes)} labels from {base_dir}/labels_partition.json")

    print("Building sparse doc-label matrix...")
    Y = build_doc_label_matrix(csv_path, mlb_classes, chunksize=chunksize)
    print(f"Y: {Y.shape}, {Y.nnz} nonzeros")

    print("Reading document text...")
    texts = pd.concat(
        chunk["text"] for chunk in pd.read_csv(csv_path, usecols=["text"], chunksize=chunksize)
    ).astype(str).tolist()
    assert len(texts) == Y.shape[0], f"Row count mismatch: {len(texts)} texts vs {Y.shape[0]} label rows"

    print(f"Embedding {len(texts)} documents with {embedding_model}...")
    X = embed_documents(texts, model_name=embedding_model, batch_size=batch_size, device=device)

    print("Computing label features (labels_f = normalize(Y^T @ X))...")
    label_feats = label_features(Y, X)

    print(f"Clustering {len(mlb_classes)} labels, target cluster size {target_cluster_size}...")
    cluster_ids, num_clusters = build_flat_clusters(label_feats, target_cluster_size, seed=seed)
    print(f"Produced {num_clusters} clusters (avg size {len(mlb_classes) / num_clusters:.1f})")

    cluster_assignments_path = os.path.join(base_dir, "cluster_assignments.json")
    with open(cluster_assignments_path, "w") as f:
        json.dump({"cluster_id": cluster_ids.tolist(), "num_clusters": int(num_clusters)}, f)
    print(f"Wrote {cluster_assignments_path}")

    cluster_to_labels = defaultdict(list)
    for label_idx, cid in enumerate(cluster_ids):
        cluster_to_labels[int(cid)].append(label_idx)
    cluster_to_labels_path = os.path.join(base_dir, "cluster_to_labels.json")
    with open(cluster_to_labels_path, "w") as f:
        json.dump(cluster_to_labels, f)
    print(f"Wrote {cluster_to_labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a flat label clustering for the full-scale candidate-generator subsystem (see README_AMAZON3M.md).")
    parser.add_argument("--base_dir", type=str, required=True, help="Directory containing labels_partition.json, written by Stage 2's preprocess_data (same dependency Stage 3 already has).")
    parser.add_argument("--csv_path", type=str, required=True, help="Source CSV with text,labels columns (same corpus Stage 2 was run against).")
    parser.add_argument("--target_cluster_size", type=int, default=1000, help="Recursion stops once a cluster's label count is <= this (default: 1000).")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model for document embeddings (default: all-MiniLM-L6-v2).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for SentenceTransformer encoding (default: 256).")
    parser.add_argument("--chunksize", type=int, default=300_000, help="Rows per pandas chunk when scanning the source CSV (default: 300000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for cluster-split initialization (default: 42).")
    parser.add_argument("--device", type=str, default=None, help="Device for SentenceTransformer, e.g. cuda/cpu (default: auto).")
    args = parser.parse_args()
    main(**vars(args))
