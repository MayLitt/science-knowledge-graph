"""
build_private_kb.py
--------------------
Builds the private RDF graph from extracted and cleaned relations.

Reads  : data/extracted_relations_cleaned.csv
Writes : kg_artifacts/graph.nt

Usage (from project root):
    python src/kg/build_private_kb.py

Dependencies:
    pip install rdflib pandas
"""

import pandas as pd
import re
from rdflib import Graph, Namespace
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
INPUT_FILE  = "data/extracted_relations_cleaned.csv"
OUTPUT_FILE = "kg_artifacts/graph.nt"

# ── Namespace ────────────────────────────────────────────────
EX = Namespace("http://example.org/")


# ── Utilities ────────────────────────────────────────────────

def uri_safe(text: str) -> str:
    """Convert text into a safe URI fragment."""
    text = str(text).strip()

    # Remove unicode / accents
    text = text.encode("ascii", "ignore").decode()

    # Replace non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9_]", "_", text)

    # Remove duplicate underscores
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def is_valid(text: str) -> bool:
    """Basic validation for entity/predicate."""
    return text and text.lower() != "nan" and len(text.strip()) > 0


# ── Main builder ─────────────────────────────────────────────

def build_graph(input_path: str, output_path: str):
    df = pd.read_csv(input_path, encoding="utf-8")

    print(f"Relations loaded : {len(df)}")
    print(f"Columns          : {list(df.columns)}")

    g = Graph()
    g.bind("ex", EX)

    added = 0
    skipped = 0
    errors = 0

    for _, row in df.iterrows():
        subject   = str(row.get("subject", "")).strip()
        predicate = str(row.get("predicate", "")).strip()
        obj       = str(row.get("object", "")).strip()

        # Validation
        if not (is_valid(subject) and is_valid(predicate) and is_valid(obj)):
            skipped += 1
            continue

        try:
            # URI construction
            s = EX[uri_safe(subject)]
            p = EX[uri_safe(predicate)]
            o = EX[uri_safe(obj)]

            # Add triple
            g.add((s, p, o))
            added += 1

        except Exception:
            errors += 1
            continue

    # Save graph
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    g.serialize(output_path, format="nt")

    # Logs
    print("\nGraph construction complete:")
    print(f"  Triples added   : {added}")
    print(f"  Rows skipped    : {skipped}")
    print(f"  Errors          : {errors}")
    print(f"  File created    : {output_path}")

    # Statistics
    entities = set()
    predicates = set()

    for s, p, o in g:
        entities.add(str(s))
        entities.add(str(o))
        predicates.add(str(p))

    print("\nKB Statistics:")
    print(f"  Unique entities   : {len(entities)}")
    print(f"  Unique predicates : {len(predicates)}")
    print(f"  Total triples     : {len(g)}")

    return g


# ── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    build_graph(INPUT_FILE, OUTPUT_FILE)