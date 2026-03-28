"""
align_predicates.py
-------------------
Maps raw verb predicates from private_kb.nt to ontology/schema properties,
then outputs an aligned RDF graph as aligned_kb.nt.
"""

from rdflib import Graph, Namespace

EX  = Namespace("http://example.org/")
DBO = Namespace("http://dbpedia.org/ontology/")
SCH = Namespace("https://schema.org/")

# Predicate alignment map
# raw verb → (ontology property URI, confidence)
PREDICATE_MAP = {
    # Birth / death
    "bear":         (EX.bornIn,          0.9),
    "die":          (EX.diedIn,          0.9),

    # Education
    "graduate":     (EX.studiedAt,       0.85),
    "enrol":        (EX.studiedAt,       0.80),
    "study":        (EX.studiedAt,       0.80),

    # Work / career
    "complete":     (EX.workedAt,        0.70),
    "spend":        (EX.workedAt,        0.65),
    "arrive":       (EX.workedAt,        0.60),
    "come":         (EX.workedAt,        0.55),
    "return":       (EX.workedAt,        0.55),
    "leave":        (EX.workedAt,        0.50),
    "go":           (EX.workedAt,        0.50),

    # Social / collaboration
    "influence":    (EX.influencedBy,     0.85),
    "teach":        (EX.taughtBy,         0.85),
    "collaborate":  (EX.collaboratedWith, 0.90),
    "talk":         (EX.collaboratedWith, 0.60),
    "represent":    (EX.collaboratedWith, 0.55),
    "tell":         (EX.collaboratedWith, 0.55),
    "ask":          (EX.collaboratedWith, 0.50),
    "summon":       (EX.collaboratedWith, 0.50),
    "send":         (EX.collaboratedWith, 0.50),

    # Authorship / works
    "write":        (EX.authorOf,        0.90),
    "build":        (EX.authorOf,        0.70),
    "devise":       (EX.authorOf,        0.75),
    "produce":      (EX.authorOf,        0.70),
    "create":       (EX.authorOf,        0.70),
    "improve":      (EX.authorOf,        0.65),
    "detect":       (EX.knownFor,        0.75),

    # Membership
    "elect":        (EX.memberOf,        0.85),
    "become":       (EX.memberOf,        0.65),
    "form":         (EX.memberOf,        0.60),

    # Known for
    "discover":     (EX.knownFor,        0.90),
    "find":         (EX.knownFor,        0.70),
    "expand":       (EX.knownFor,        0.65),
}

CONFIDENCE_THRESHOLD = 0.55

def is_year(uri):
    label = str(uri).split("/")[-1]
    return label.isdigit() and len(label) == 4

def looks_like_place(uri):
    label = str(uri).split("/")[-1].lower()
    place_keywords = ["rome", "florence", "athens", "cambridge", "london", "italy", "greece"]
    return any(k in label for k in place_keywords)

def looks_like_person(uri):
    label = str(uri).split("/")[-1]
    return "_" in label and not label.isdigit()

def align(input_path="private_kb.nt",
          output_path="aligned_kb.nt",
          alignment_log="alignment_report.txt"):

    g_in  = Graph()
    g_out = Graph()

    g_in.parse(input_path, format="nt")
    print(f"Loaded triples: {len(g_in)}")

    kept = 0
    dropped = 0
    aligned = 0
    log_lines = []

    for s, p, o in g_in:
        raw_verb = str(p).replace("http://example.org/", "")

        if raw_verb in PREDICATE_MAP:
            new_pred, confidence = PREDICATE_MAP[raw_verb]

            if confidence >= CONFIDENCE_THRESHOLD:

                # ── Semantic validation ──
                if new_pred == EX.authorOf and (is_year(o) or looks_like_place(o)):
                    dropped += 1
                    log_lines.append(f"SEMANTIC_DROP {raw_verb:15s} (invalid authorOf object)")
                    continue

                if new_pred == EX.bornIn and is_year(o):
                    dropped += 1
                    log_lines.append(f"SEMANTIC_DROP {raw_verb:15s} (year cannot be place)")
                    continue

                # ── Valid triple ──
                g_out.add((s, new_pred, o))
                aligned += 1
                kept += 1
                log_lines.append(
                    f"ALIGNED  [{confidence:.0%}]  {raw_verb:15s} → {str(new_pred).split('/')[-1]}"
                )

            else:
                dropped += 1
                log_lines.append(
                    f"DROPPED  [{confidence:.0%}]  {raw_verb:15s}  (confidence too low)"
                )

        else:
            g_out.add((s, p, o))
            kept += 1
            log_lines.append(
                f"KEPT_RAW          {raw_verb:15s}  (no mapping found)"
            )

    g_out.serialize(output_path, format="nt")

    with open(alignment_log, "w", encoding="utf-8") as f:
        f.write("=== Predicate Alignment Report ===\n\n")
        f.write(f"Initial triples  : {len(g_in)}\n")
        f.write(f"Aligned triples  : {aligned}\n")
        f.write(f"Kept triples     : {kept}\n")
        f.write(f"Dropped triples  : {dropped}\n\n")
        f.write("\n".join(sorted(set(log_lines))))

    print(f"Aligned  : {aligned}")
    print(f"Kept     : {kept}")
    print(f"Dropped  : {dropped}")
    print(f"Output   : {output_path}")
    print(f"Log file : {alignment_log}")


if __name__ == "__main__":
    align(
        input_path="kg_artifacts/graph.nt",
        output_path="kg_artifacts/aligned_kb.nt",
        alignment_log="kg_artifacts/alignment_report.txt"
    )