"""
train_kge.py
------------
Knowledge Graph Embedding avec PyKEEN.
Entraîne TransE et RotatE sur le KB des scientifiques.

Usage (depuis la racine du projet) :
    python src/kge/train_kge.py

Dépendances :
    pip install pykeen torch

Output :
    kge_datasets/train.txt
    kge_datasets/valid.txt
    kge_datasets/test.txt
    kge_datasets/results_TransE.json
    kge_datasets/results_RotatE.json
    kge_datasets/comparison_report.txt
"""

import json
import random
import os
from pathlib import Path

# ── Chemins ──────────────────────────────────────────────────
KB_PATH      = "kg_artifacts/expanded_reasoned.nt"
KGE_DIR      = Path("kge_datasets")
TRAIN_PATH   = KGE_DIR / "train.txt"
VALID_PATH   = KGE_DIR / "valid.txt"
TEST_PATH    = KGE_DIR / "test.txt"
REPORT_PATH  = KGE_DIR / "comparison_report.txt"

KGE_DIR.mkdir(exist_ok=True)

# ── Config d'entraînement ────────────────────────────────────
EMBEDDING_DIM = 50      # petit car KB petit
NUM_EPOCHS    = 100
BATCH_SIZE    = 16
LEARNING_RATE = 0.01
RANDOM_SEED   = 42

random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 1 — Préparer les triplets depuis le KB
# ══════════════════════════════════════════════════════════════

def load_triples(kb_path):
    """Charge les triplets depuis un fichier NT, filtre les literals."""
    from rdflib import Graph, URIRef

    g = Graph()
    g.parse(kb_path, format="nt")

    triples = []
    for s, p, o in g:
        # On garde uniquement les triplets URI→URI (pas les literals)
        if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
            s_str = str(s).split("/")[-1].replace("#", "_")
            p_str = str(p).split("/")[-1].replace("#", "_")
            o_str = str(o).split("/")[-1].replace("#", "_")
            # Ignorer les triplets owl:sameAs et rdf:type (trop génériques)
            if p_str not in {"sameAs", "type", "subClassOf", "domain", "range"}:
                triples.append((s_str, p_str, o_str))

    # Dédupliquer
    triples = list(set(triples))
    return triples


def split_triples(triples, train_ratio=0.8, valid_ratio=0.1):
    """Split 80/10/10 en s'assurant qu'aucune entité n'apparaît seulement en valid/test."""
    random.shuffle(triples)

    n = len(triples)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = triples[:n_train]
    valid = triples[n_train:n_train + n_valid]
    test  = triples[n_train + n_valid:]

    # Entités vues à l'entraînement
    train_entities = set()
    for s, p, o in train:
        train_entities.add(s)
        train_entities.add(o)

    # Filtrer valid/test : garder seulement les triplets dont les entités sont dans train
    valid = [(s, p, o) for s, p, o in valid if s in train_entities and o in train_entities]
    test  = [(s, p, o) for s, p, o in test  if s in train_entities and o in train_entities]

    return train, valid, test


def save_triples(triples, path):
    with open(path, "w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")
    print(f"  Sauvegardé : {path} ({len(triples)} triplets)")


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 2 — Entraînement PyKEEN
# ══════════════════════════════════════════════════════════════

def train_model(model_name, train_path, valid_path, test_path):
    """Entraîne un modèle KGE avec PyKEEN et retourne les métriques."""
    import torch
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    print(f"\n{'─'*50}")
    print(f"Entraînement : {model_name}")
    print(f"{'─'*50}")

    # Charger les triplets
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

    print(f"  Entités    : {training.num_entities}")
    print(f"  Relations  : {training.num_relations}")
    print(f"  Train      : {training.num_triples}")
    print(f"  Valid      : {validation.num_triples}")
    print(f"  Test       : {testing.num_triples}")

    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model=model_name,
        model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=LEARNING_RATE),
        training_kwargs=dict(
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
        ),
        random_seed=RANDOM_SEED,
        evaluation_kwargs=dict(use_tqdm=False),
    )

    # Extraire les métriques
    metrics = result.metric_results.to_dict()

    mrr    = metrics.get("both.realistic.inverse_harmonic_mean_rank", 0)
    hits1  = metrics.get("both.realistic.hits_at_1", 0)
    hits3  = metrics.get("both.realistic.hits_at_3", 0)
    hits10 = metrics.get("both.realistic.hits_at_10", 0)

    summary = {
        "model":  model_name,
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

    print(f"\n  Métriques {model_name} :")
    print(f"    MRR     : {mrr:.4f}")
    print(f"    Hits@1  : {hits1:.4f}")
    print(f"    Hits@3  : {hits3:.4f}")
    print(f"    Hits@10 : {hits10:.4f}")

    # Sauvegarder les embeddings pour l'analyse t-SNE plus tard
    model_dir = KGE_DIR / model_name
    model_dir.mkdir(exist_ok=True)
    result.save_to_directory(str(model_dir))

    return summary, result


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 3 — Analyse nearest neighbors
# ══════════════════════════════════════════════════════════════

def nearest_neighbors(result, entity_name, top_k=5):
    """Trouve les voisins les plus proches d'une entité dans l'espace d'embedding."""
    import torch

    model = result.model
    entity_to_id = result.training.entity_to_id

    if entity_name not in entity_to_id:
        print(f"  Entité '{entity_name}' non trouvée.")
        return []

    id_to_entity = {v: k for k, v in entity_to_id.items()}
    entity_id    = entity_to_id[entity_name]

    # Récupérer tous les embeddings
    all_embeddings = model.entity_representations[0](
        torch.arange(len(entity_to_id))
    ).detach()

    query = all_embeddings[entity_id].unsqueeze(0)

    # Distance cosinus
    similarities = torch.nn.functional.cosine_similarity(query, all_embeddings)
    similarities[entity_id] = -1  # exclure l'entité elle-même

    top_ids = similarities.topk(top_k).indices.tolist()
    neighbors = [(id_to_entity[i], round(similarities[i].item(), 4)) for i in top_ids]

    return neighbors


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 4 — t-SNE visualization
# ══════════════════════════════════════════════════════════════

def plot_tsne(result, model_name):
    """Génère un plot t-SNE des embeddings d'entités."""
    try:
        import torch
        import numpy as np
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model = result.model
        entity_to_id = result.training.entity_to_id

        embeddings = model.entity_representations[0](
            torch.arange(len(entity_to_id))
        ).detach().numpy()

        # RotatE produit des embeddings complexes → prendre la partie réelle
        if np.iscomplexobj(embeddings):
            embeddings = embeddings.real

        labels = list(entity_to_id.keys())

        # t-SNE
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED,
                    perplexity=min(30, len(labels) - 1))
        coords = tsne.fit_transform(embeddings)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=40)

        # Annoter les entités principales
        main_entities = {
            "Galileo", "Newton", "Aristotle", "Alvarez",
            "Plato", "Socrates", "Copernicus", "Leibniz"
        }
        for i, label in enumerate(labels):
            if label in main_entities:
                ax.annotate(label, (coords[i, 0], coords[i, 1]),
                           fontsize=9, ha="center")

        ax.set_title(f"t-SNE des embeddings d'entités — {model_name}")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

        out_path = KGE_DIR / f"tsne_{model_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  t-SNE sauvegardé : {out_path}")

    except ImportError as e:
        print(f"  t-SNE ignoré (module manquant : {e})")
        print("  → pip install scikit-learn matplotlib")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Préparer les données
    print("="*60)
    print("ÉTAPE 1 — Préparation des triplets")
    print("="*60)

    triples = load_triples(KB_PATH)
    print(f"Triplets URI→URI chargés : {len(triples)}")

    if len(triples) < 10:
        print("ERREUR : pas assez de triplets pour entraîner un modèle KGE.")
        print("Vérifier que expanded_reasoned.nt existe et contient des triplets URI→URI.")
        exit(1)

    train, valid, test = split_triples(triples)
    print(f"\nSplit 80/10/10 :")
    print(f"  Train : {len(train)}")
    print(f"  Valid : {len(valid)}")
    print(f"  Test  : {len(test)}")

    save_triples(train, TRAIN_PATH)
    save_triples(valid, VALID_PATH)
    save_triples(test,  TEST_PATH)

    # 2. Entraîner les modèles
    print("\n" + "="*60)
    print("ÉTAPE 2 — Entraînement des modèles KGE")
    print("="*60)

    try:
        summary_transe, result_transe = train_model("TransE", TRAIN_PATH, VALID_PATH, TEST_PATH)
        summary_rotate, result_rotate = train_model("RotatE", TRAIN_PATH, VALID_PATH, TEST_PATH)
    except Exception as e:
        print(f"\nERREUR pendant l'entraînement : {e}")
        print("→ Vérifier que PyKEEN et PyTorch sont installés : pip install pykeen torch")
        exit(1)

    # 3. Nearest neighbors
    print("\n" + "="*60)
    print("ÉTAPE 3 — Analyse nearest neighbors (TransE)")
    print("="*60)

    for entity in ["Galileo", "Aristotle", "Newton"]:
        neighbors = nearest_neighbors(result_transe, entity)
        if neighbors:
            print(f"\n  Voisins de {entity} :")
            for name, score in neighbors:
                print(f"    {name:30s}  sim={score}")

    # 4. t-SNE
    print("\n" + "="*60)
    print("ÉTAPE 4 — Visualisation t-SNE")
    print("="*60)

    plot_tsne(result_transe, "TransE")
    plot_tsne(result_rotate, "RotatE")

    # 5. Rapport de comparaison
    print("\n" + "="*60)
    print("ÉTAPE 5 — Rapport de comparaison")
    print("="*60)

    # Sauvegarder les résultats JSON
    with open(KGE_DIR / "results_TransE.json", "w") as f:
        json.dump(summary_transe, f, indent=2)
    with open(KGE_DIR / "results_RotatE.json", "w") as f:
        json.dump(summary_rotate, f, indent=2)

    # Rapport textuel
    report = f"""
=== Rapport de comparaison KGE ===

Dataset
  Triplets total  : {len(triples)}
  Train           : {len(train)}
  Valid           : {len(valid)}
  Test            : {len(test)}

Configuration
  Embedding dim   : {EMBEDDING_DIM}
  Epochs          : {NUM_EPOCHS}
  Batch size      : {BATCH_SIZE}
  Learning rate   : {LEARNING_RATE}

Résultats
  Modèle       MRR      Hits@1   Hits@3   Hits@10
  {'─'*55}
  TransE       {summary_transe['metrics']['MRR']:<8.4f} {summary_transe['metrics']['Hits@1']:<8.4f} {summary_transe['metrics']['Hits@3']:<8.4f} {summary_transe['metrics']['Hits@10']:.4f}
  RotatE       {summary_rotate['metrics']['MRR']:<8.4f} {summary_rotate['metrics']['Hits@1']:<8.4f} {summary_rotate['metrics']['Hits@3']:<8.4f} {summary_rotate['metrics']['Hits@10']:.4f}

Note : KB de petite taille ({len(triples)} triplets vs 50k-200k recommandés).
Les métriques sont instables et doivent être interprétées avec précaution.
Ce résultat est discuté dans la section Critical Reflection du rapport.
"""
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Rapport sauvegardé : {REPORT_PATH}")
    print("\nEntraînement KGE terminé.")
