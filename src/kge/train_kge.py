"""
train_kge.py
------------
Knowledge Graph Embedding with PyKEEN.
Trains TransE and RotatE on the scientists knowledge graph.

Usage (from project root):
    python src/kge/train_kge.py

Dependencies:
    pip install pykeen torch scikit-learn matplotlib

Output:
    kge_datasets/train.txt
    kge_datasets/valid.txt
    kge_datasets/test.txt
    kge_datasets/results_TransE.json
    kge_datasets/results_RotatE.json
    kge_datasets/comparison_report.txt
    kge_datasets/tsne_TransE.png
    kge_datasets/tsne_RotatE.png
"""

import json
import random
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
# Use aligned_kb.nt: ~52 clean ontology properties instead of
# ~601 raw verbs → each relation seen ~380x instead of ~33x
KB_PATH     = "kg_artifacts/aligned_kb.nt"
KGE_DIR     = Path("kge_datasets")
TRAIN_PATH  = KGE_DIR / "train.txt"
VALID_PATH  = KGE_DIR / "valid.txt"
TEST_PATH   = KGE_DIR / "test.txt"
REPORT_PATH = KGE_DIR / "comparison_report.txt"

KGE_DIR.mkdir(exist_ok=True)

# ── Training configuration ────────────────────────────────────
EMBEDDING_DIM = 100    # increased capacity
NUM_EPOCHS    = 300    # more epochs to converge
BATCH_SIZE    = 32     # larger batch = stabler gradients
LEARNING_RATE = 0.001  # lower lr = smoother convergence
RANDOM_SEED   = 42

random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════
#  STEP 1 — Load and prepare triples
# ══════════════════════════════════════════════════════════════

def load_triples(kb_path):
    """Load URI-to-URI triples from an NT file, filtering out literals."""
    from rdflib import Graph, URIRef

    g = Graph()
    g.parse(kb_path, format="nt")

    triples = []
    skip_predicates = {"sameAs", "type", "subClassOf", "domain", "range"}

    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
            s_str = str(s).split("/")[-1].replace("#", "_")
            p_str = str(p).split("/")[-1].replace("#", "_")
            o_str = str(o).split("/")[-1].replace("#", "_")
            if p_str not in skip_predicates:
                triples.append((s_str, p_str, o_str))

    return list(set(triples))


def split_triples(triples, train_ratio=0.8, valid_ratio=0.1):
    """80/10/10 split ensuring no entity appears only in valid/test."""
    random.shuffle(triples)

    n       = len(triples)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = triples[:n_train]
    valid = triples[n_train:n_train + n_valid]
    test  = triples[n_train + n_valid:]

    train_entities = {e for s, p, o in train for e in (s, o)}
    valid = [(s, p, o) for s, p, o in valid if s in train_entities and o in train_entities]
    test  = [(s, p, o) for s, p, o in test  if s in train_entities and o in train_entities]

    return train, valid, test


def save_triples(triples, path):
    with open(path, "w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")
    print(f"  Saved: {path} ({len(triples)} triples)")


# ══════════════════════════════════════════════════════════════
#  STEP 2 — Train KGE models with PyKEEN
# ══════════════════════════════════════════════════════════════

def train_model(model_name, train_path, valid_path, test_path):
    """Train a KGE model with PyKEEN and return metrics."""
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    print(f"\n{'─'*50}")
    print(f"Training: {model_name}")
    print(f"{'─'*50}")

    training   = TriplesFactory.from_path(train_path)
    validation = TriplesFactory.from_path(
        valid_path,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    testing = TriplesFactory.from_path(
        test_path,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )

    print(f"  Entities  : {training.num_entities}")
    print(f"  Relations : {training.num_relations}")
    print(f"  Train     : {training.num_triples}")
    print(f"  Valid     : {validation.num_triples}")
    print(f"  Test      : {testing.num_triples}")

    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model=model_name,
        model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=LEARNING_RATE),
        training_kwargs=dict(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE),
        random_seed=RANDOM_SEED,
        evaluation_kwargs=dict(use_tqdm=False),
    )

    metrics = result.metric_results.to_dict()
    mrr    = metrics.get("both.realistic.inverse_harmonic_mean_rank", 0)
    hits1  = metrics.get("both.realistic.hits_at_1", 0)
    hits3  = metrics.get("both.realistic.hits_at_3", 0)
    hits10 = metrics.get("both.realistic.hits_at_10", 0)

    summary = {
        "model": model_name,
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "epochs":        NUM_EPOCHS,
            "batch_size":    BATCH_SIZE,
            "lr":            LEARNING_RATE,
        },
        "metrics": {
            "MRR":     round(mrr,    4),
            "Hits@1":  round(hits1,  4),
            "Hits@3":  round(hits3,  4),
            "Hits@10": round(hits10, 4),
        },
        "dataset": {
            "entities":  training.num_entities,
            "relations": training.num_relations,
            "train":     training.num_triples,
            "valid":     validation.num_triples,
            "test":      testing.num_triples,
        }
    }

    print(f"\n  Metrics {model_name}:")
    print(f"    MRR     : {mrr:.4f}")
    print(f"    Hits@1  : {hits1:.4f}")
    print(f"    Hits@3  : {hits3:.4f}")
    print(f"    Hits@10 : {hits10:.4f}")

    model_dir = KGE_DIR / model_name
    model_dir.mkdir(exist_ok=True)
    result.save_to_directory(str(model_dir))

    return summary, result


# ══════════════════════════════════════════════════════════════
#  STEP 3 — Nearest neighbor analysis
# ══════════════════════════════════════════════════════════════

def nearest_neighbors(result, entity_name, top_k=5):
    """Find closest entities in embedding space using cosine similarity."""
    import torch

    model        = result.model
    entity_to_id = result.training.entity_to_id

    if entity_name not in entity_to_id:
        print(f"  Entity '{entity_name}' not found.")
        return []

    id_to_entity = {v: k for k, v in entity_to_id.items()}
    entity_id    = entity_to_id[entity_name]

    all_embeddings = model.entity_representations[0](
        torch.arange(len(entity_to_id))
    ).detach()

    query = all_embeddings[entity_id].unsqueeze(0)
    similarities = torch.nn.functional.cosine_similarity(query, all_embeddings)
    similarities[entity_id] = -1  # exclude the query entity itself

    top_ids   = similarities.topk(top_k).indices.tolist()
    neighbors = [(id_to_entity[i], round(similarities[i].item(), 4)) for i in top_ids]

    return neighbors


# ══════════════════════════════════════════════════════════════
#  STEP 4 — t-SNE visualization
# ══════════════════════════════════════════════════════════════

def plot_tsne(result, model_name):
    """Generate a t-SNE scatter plot of entity embeddings."""
    try:
        import torch
        import numpy as np
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model        = result.model
        entity_to_id = result.training.entity_to_id

        embeddings = model.entity_representations[0](
            torch.arange(len(entity_to_id))
        ).detach().numpy()

        # RotatE produces complex embeddings — take real part only
        if np.iscomplexobj(embeddings):
            embeddings = embeddings.real

        labels = list(entity_to_id.keys())

        tsne   = TSNE(n_components=2, random_state=RANDOM_SEED,
                      perplexity=min(30, len(labels) - 1))
        coords = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=40)

        main_entities = {
            "Galileo", "Newton", "Aristotle", "Alvarez",
            "Plato", "Socrates", "Copernicus", "Leibniz",
            "Einstein", "Darwin", "Fermi", "Marie_Curie"
        }
        for i, label in enumerate(labels):
            if label in main_entities:
                ax.annotate(label, (coords[i, 0], coords[i, 1]),
                            fontsize=9, ha="center")

        ax.set_title(f"t-SNE of entity embeddings — {model_name}")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

        out_path = KGE_DIR / f"tsne_{model_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  t-SNE saved: {out_path}")

    except ImportError as e:
        print(f"  t-SNE skipped (missing module: {e})")
        print("  Run: pip install scikit-learn matplotlib")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Step 1 — Prepare data
    print("=" * 60)
    print("STEP 1 — Triple preparation")
    print("=" * 60)

    triples = load_triples(KB_PATH)
    print(f"URI-to-URI triples loaded: {len(triples)}")

    if len(triples) < 10:
        print("ERROR: not enough triples to train a KGE model.")
        exit(1)

    train, valid, test = split_triples(triples)
    print(f"\n80/10/10 split:")
    print(f"  Train: {len(train)}")
    print(f"  Valid: {len(valid)}")
    print(f"  Test : {len(test)}")

    save_triples(train, TRAIN_PATH)
    save_triples(valid, VALID_PATH)
    save_triples(test,  TEST_PATH)

    # Step 2 — Train models
    print("\n" + "=" * 60)
    print("STEP 2 — KGE model training")
    print("=" * 60)

    try:
        summary_transe, result_transe = train_model("TransE", TRAIN_PATH, VALID_PATH, TEST_PATH)
        summary_rotate, result_rotate = train_model("RotatE", TRAIN_PATH, VALID_PATH, TEST_PATH)
    except Exception as e:
        print(f"\nERROR during training: {e}")
        print("Make sure PyKEEN and PyTorch are installed: pip install pykeen torch")
        exit(1)

    # Step 3 — Nearest neighbors
    print("\n" + "=" * 60)
    print("STEP 3 — Nearest neighbor analysis (TransE)")
    print("=" * 60)

    for entity in ["Galileo", "Aristotle", "Newton", "Einstein", "Darwin"]:
        neighbors = nearest_neighbors(result_transe, entity)
        if neighbors:
            print(f"\n  Neighbors of {entity}:")
            for name, score in neighbors:
                print(f"    {name:35s}  sim={score}")

    # Step 4 — t-SNE
    print("\n" + "=" * 60)
    print("STEP 4 — t-SNE visualization")
    print("=" * 60)

    plot_tsne(result_transe, "TransE")
    plot_tsne(result_rotate, "RotatE")

    # Step 5 — Comparison report
    print("\n" + "=" * 60)
    print("STEP 5 — Comparison report")
    print("=" * 60)

    with open(KGE_DIR / "results_TransE.json", "w") as f:
        json.dump(summary_transe, f, indent=2)
    with open(KGE_DIR / "results_RotatE.json", "w") as f:
        json.dump(summary_rotate, f, indent=2)

    report = f"""
=== KGE Comparison Report ===

Dataset
  Source KB      : {KB_PATH}
  Total triples  : {len(triples)}
  Train          : {len(train)}
  Valid          : {len(valid)}
  Test           : {len(test)}

Configuration
  Embedding dim  : {EMBEDDING_DIM}
  Epochs         : {NUM_EPOCHS}
  Batch size     : {BATCH_SIZE}
  Learning rate  : {LEARNING_RATE}

Results
  Model      MRR      Hits@1   Hits@3   Hits@10
  -------------------------------------------------------
  TransE     {summary_transe['metrics']['MRR']:<8.4f} {summary_transe['metrics']['Hits@1']:<8.4f} {summary_transe['metrics']['Hits@3']:<8.4f} {summary_transe['metrics']['Hits@10']:.4f}
  RotatE     {summary_rotate['metrics']['MRR']:<8.4f} {summary_rotate['metrics']['Hits@1']:<8.4f} {summary_rotate['metrics']['Hits@3']:<8.4f} {summary_rotate['metrics']['Hits@10']:.4f}
"""
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved: {REPORT_PATH}")
    print("\nKGE training complete.")
