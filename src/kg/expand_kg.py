"""
expand_kg.py
------------
Enrichit le KB privé en interrogeant Wikidata via SPARQL
pour chaque entité liée (owl:sameAs), puis ajoute les triplets
récupérés au KB aligné.

Usage (depuis la racine du projet) :
    python src/kg/expand_kg.py

Input  : kg_artifacts/aligned_kb.nt
         kg_artifacts/alignment.ttl
Output : kg_artifacts/expanded.nt
         kg_artifacts/expansion_report.txt
"""

import time
import requests
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import OWL, RDF, RDFS, XSD

# ── Namespaces ──────────────────────────────────────────────
EX  = Namespace("http://example.org/")
WD  = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
DBO = Namespace("http://dbpedia.org/ontology/")
SCH = Namespace("https://schema.org/")

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "ScienceKGBot/1.0 (student project; contact: student@example.com)",
    "Accept": "application/sparql-results+json"
}

# ── Mapping propriétés Wikidata → propriétés locales ────────
# (propriété Wikidata, label lisible, propriété locale)
PROPERTIES = [
    ("P569",  "birthDate",   EX.birthDate),
    ("P570",  "deathDate",   EX.deathDate),
    ("P19",   "birthPlace",  EX.bornIn),
    ("P20",   "deathPlace",  EX.diedIn),
    ("P69",   "studiedAt",   EX.studiedAt),
    ("P108",  "workedAt",    EX.workedAt),
    ("P101",  "fieldOfWork", EX.knownFor),
    ("P737",  "influencedBy",EX.influencedBy),
    ("P800",  "notableWork", EX.authorOf),
    ("P166",  "award",       EX.knownFor),
]


def build_sparql_query(wd_id: str) -> str:
    prop_lines = "\n  ".join(
        f"OPTIONAL {{ wd:{wd_id} wdt:{pid} ?{label} . }}"
        for pid, label, _ in PROPERTIES
    )
    return f"""
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT * WHERE {{
  {prop_lines}
  OPTIONAL {{ wd:{wd_id} rdfs:label ?name FILTER(lang(?name) = "en") }}
}}
LIMIT 1
"""


def query_wikidata(wd_id: str) -> dict:
    query = build_sparql_query(wd_id)
    try:
        resp = requests.get(
            WIKIDATA_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=15
        )
        resp.raise_for_status()
        results = resp.json()["results"]["bindings"]
        return results[0] if results else {}
    except Exception as e:
        print(f"  ⚠ Erreur Wikidata pour {wd_id} : {e}")
        return {}


def load_entity_links(alignment_path: str) -> dict:
    """Retourne {ex_uri: wd_id} depuis le fichier alignment.ttl"""
    g = Graph()
    g.parse(alignment_path, format="turtle")

    links = {}
    for s, p, o in g.triples((None, OWL.sameAs, None)):
        o_str = str(o)
        if "wikidata.org/entity/" in o_str:
            wd_id = o_str.split("/")[-1]
            links[str(s)] = wd_id

    # Dédupliquer : garder une seule entrée par entité locale
    seen_wd = {}
    deduped = {}
    for ex_uri, wd_id in links.items():
        if wd_id not in seen_wd:
            seen_wd[wd_id] = ex_uri
            deduped[ex_uri] = wd_id

    return deduped


def expand(
    kb_path="kg_artifacts/aligned_kb.nt",
    alignment_path="kg_artifacts/alignment.ttl",
    output_path="kg_artifacts/expanded.nt",
    report_path="kg_artifacts/expansion_report.txt"
):
    # Charger le KB existant
    g = Graph()
    g.parse(kb_path, format="nt")
    initial_count = len(g)
    print(f"KB chargé : {initial_count} triplets")

    # Charger les liens d'entités
    entity_links = load_entity_links(alignment_path)
    print(f"Entités liées à Wikidata : {len(entity_links)}")

    report_lines = []
    total_added = 0

    for ex_uri, wd_id in entity_links.items():
        entity_name = ex_uri.split("/")[-1]
        print(f"\n→ {entity_name} (wd:{wd_id})")
        report_lines.append(f"\n=== {entity_name} (wd:{wd_id}) ===")

        result = query_wikidata(wd_id)
        added = 0

        for pid, label, local_prop in PROPERTIES:
            if label in result:
                raw_val = result[label]["value"]
                val_type = result[label].get("type", "literal")

                subject = URIRef(ex_uri)

                # Valeur URI (entité Wikidata) → on crée un nœud local
                if val_type == "uri" and "wikidata.org" in raw_val:
                    wd_entity_id = raw_val.split("/")[-1]
                    obj = URIRef(f"http://example.org/wd_{wd_entity_id}")
                    # Ajouter un label si dispo
                    if "name" in result:
                        g.add((obj, RDFS.label, Literal(result["name"]["value"])))
                elif val_type == "uri":
                    obj = URIRef(raw_val)
                else:
                    # Literal (date, texte)
                    if "T" in raw_val and raw_val.startswith(("-", "1", "2", "3")):
                        obj = Literal(raw_val[:10], datatype=XSD.date)
                    else:
                        obj = Literal(raw_val)

                g.add((subject, local_prop, obj))
                added += 1
                report_lines.append(f"  + {label}: {raw_val[:80]}")
                print(f"    + {label}: {raw_val[:60]}")

        total_added += added
        report_lines.append(f"  → {added} triplets ajoutés")

        # Pause pour respecter le rate limit Wikidata
        time.sleep(1)

    # Sauvegarder
    g.serialize(output_path, format="nt")

    final_count = len(g)
    summary = (
        f"\n=== Résumé expansion ===\n"
        f"Triplets initiaux  : {initial_count}\n"
        f"Triplets ajoutés   : {total_added}\n"
        f"Triplets finaux    : {final_count}\n"
        f"Entités enrichies  : {len(entity_links)}\n"
        f"Fichier            : {output_path}\n"
    )
    print(summary)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)
        f.write("\n".join(report_lines))

    print(f"Rapport : {report_path}")


if __name__ == "__main__":
    expand()
