"""
rag_pipeline.py (v2 — amélioré)
--------------------------------
Pipeline RAG : Question en langage naturel → SPARQL → Résultat
avec mécanisme de self-repair et déréférencement des IDs Wikidata.

Améliorations v2 :
  - PREFIX ex: forcé dans chaque prompt → moins d'erreurs LLM
  - Déréférencement wd_QXXX → labels lisibles (ex: wd_Q333 → "astronomy")
  - Schema summary enrichi avec ex:sharedInstitution

Usage (depuis la racine du projet) :
    python src/rag/rag_pipeline.py        # mode interactif
    python src/rag/rag_pipeline.py eval   # mode évaluation

Dépendances :
    pip install rdflib requests
"""

import requests
import json
import sys
from rdflib import Graph, Namespace
from rdflib.namespace import RDFS

# ── Config ───────────────────────────────────────────────────
KB_PATH      = "kg_artifacts/expanded_reasoned.nt"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
MAX_RETRIES  = 3

EX = Namespace("http://example.org/")

# ── Table de déréférencement Wikidata ────────────────────────
WD_LABELS = {
    "wd_Q13375":   "Pisa",
    "wd_Q193510":  "University of Padua",
    "wd_Q645663":  "University of Pisa",
    "wd_Q307":     "Galileo Galilei",
    "wd_Q333":     "astronomy",
    "wd_Q395":     "mathematics",
    "wd_Q413":     "physics",
    "wd_Q5891":    "philosophy",
    "wd_Q316":     "love",
    "wd_Q336":     "science",
    "wd_Q7754":    "mathematics",
    "wd_Q11023":   "engineering",
    "wd_Q816425":  "biology",
    "wd_Q935":     "Isaac Newton",
    "wd_Q35794":   "University of Cambridge",
    "wd_Q745967":  "University of Oxford",
    "wd_Q131262":  "Jagiellonian University",
    "wd_Q154561":  "University of Leipzig",
    "wd_Q193093":  "Platonic Academy",
    "wd_Q868":     "Aristotle",
    "wd_Q859":     "Plato",
    "wd_Q913":     "Socrates",
    "wd_Q9191":    "Descartes",
    "wd_Q36330":   "Spinoza",
    "wd_Q83041":   "Pythagoras",
    "wd_Q969370":  "Regiomontanus",
    "wd_Q93038":   "Enrico Fermi",
    "wd_Q9047":    "Leibniz",
    "wd_Q46830":   "Robert Hooke",
    "wd_Q9353":    "John Locke",
    "wd_Q8409":    "Alexander the Great",
    "wd_Q619":     "Copernicus",
    "wd_Q92935":   "Luis Alvarez",
    "wd_Q99951011":"Galileo Medal",
    "wd_Q833163":  "FRS Medal",
    "wd_Q15631401":"Fellow of the Royal Society",
    "wd_Q840924":  "Nobel Prize in Physics",
    "wd_Q329757":  "Royal Medal",
    "wd_Q123885":  "Royal Society",
    "wd_Q2031761": "Arcetri",
    "wd_Q3316008": "Woolsthorpe",
    "wd_Q288781":  "Kensington",
    "wd_Q2079":    "Leipzig",
    "wd_Q1715":    "Hanover",
    "wd_Q154804":  "Berlin Academy of Sciences",
    "wd_Q189441":  "Warmia Chapter",
    "wd_Q497115":  "Frombork",
    "wd_Q47554":   "Torun",
    "wd_Q2428909": "Isle of Wight",
    "wd_Q84":      "London",
    "wd_Q586762":  "Walter Alvarez",
    "wd_Q505549":  "Berkeley California",
    "wd_Q49108":   "University of Chicago",
    "wd_Q3138607": "Lawrence Berkeley Laboratory",
    "wd_Q177524":  "Manhattan Project",
    "wd_Q675765":  "De revolutionibus",
    "wd_Q74263":   "Principia Mathematica",
    "wd_Q219207":  "Dialogue Concerning Two World Systems",
    "wd_Q69539":   "Nicomachean Ethics",
    "wd_Q123397":  "Republic (Plato)",
    "wd_Q150008":  "Monadology",
    "wd_Q231949":  "Essay Concerning Human Understanding",
    "wd_Q170282":  "Micrographia",
    "wd_Q1524":    "Athens",
    "wd_Q2044":    "Florence",
    "wd_Q220":     "Rome",
    "wd_Q350":     "Cambridge",
    "wd_Q62":      "San Francisco",
    "wd_Q41":      "Greece",
    "wd_Q38":      "Italy",
    "wd_Q846127":  "Stagira",
    "wd_Q21235810":"Chalcis",
    "wd_Q213679":  "Pella",
    "wd_Q5684":    "Babylon",
    "wd_Q1825419": "Wrington",
    "wd_Q2233631": "Oates Essex",
    "wd_Q34433":   "University of Oxford",
}

# ── Schema summary enrichi ───────────────────────────────────
SCHEMA_SUMMARY = """You are a SPARQL expert. You MUST always start your query with:
PREFIX ex: <http://example.org/>

The knowledge graph contains information about famous scientists.

Available predicates:
  ex:bornIn          - place where a person was born
  ex:diedIn          - place where a person died
  ex:studiedAt       - institution where a person studied
  ex:workedAt        - institution where a person worked
  ex:influencedBy    - person or idea that influenced someone
  ex:authorOf        - work authored by a person
  ex:knownFor        - discovery or achievement a person is known for
  ex:collaboratedWith - person collaborated with
  ex:memberOf        - organisation a person belongs to
  ex:birthDate       - birth date (literal)
  ex:deathDate       - death date (literal)
  ex:indirectlyInfluencedBy - transitive influence chain (inferred by SWRL)
  ex:sharedInstitution - two scientists who studied at the same institution (inferred by SWRL)

Key entities (copy exact spelling including capital letters):
  ex:Galileo, ex:Newton, ex:Isaac_Newton, ex:Aristotle, ex:Plato, ex:Socrates
  ex:Copernicus, ex:Leibniz, ex:Robert_Hooke, ex:John_Locke, ex:Luis_Alvarez
  ex:Florence, ex:Athens, ex:Rome, ex:Cambridge, ex:Royal_Society

Rules:
1. ALWAYS include: PREFIX ex: <http://example.org/>
2. Use SELECT queries only
3. Keep queries simple - one or two triple patterns maximum
4. Return ONLY the SPARQL query, no explanation, no markdown, no backticks
5. Never use FILTER with function syntax like FILTER(ex:something(?x))

Example queries:
  Q: Where was Galileo born?
  A: PREFIX ex: <http://example.org/>
     SELECT ?place WHERE { ex:Galileo ex:bornIn ?place . }

  Q: Who influenced Newton?
  A: PREFIX ex: <http://example.org/>
     SELECT ?person WHERE { ex:Newton ex:influencedBy ?person . }

  Q: Which scientists shared the same institution?
  A: PREFIX ex: <http://example.org/>
     SELECT ?s1 ?s2 WHERE { ?s1 ex:sharedInstitution ?s2 . }
"""


# ══════════════════════════════════════════════════════════════
#  Utilitaires
# ══════════════════════════════════════════════════════════════

def dereference(value: str) -> str:
    """Convertit un ID Wikidata en label lisible."""
    key = value.replace(" ", "_")
    if key in WD_LABELS:
        return WD_LABELS[key]
    # Nettoyer les underscores pour les entités locales
    return value.replace("_", " ")


def load_kb(path):
    g = Graph()
    g.parse(path, format="nt")
    print(f"KB chargé : {len(g)} triplets")
    return g


# ══════════════════════════════════════════════════════════════
#  Génération SPARQL via Ollama
# ══════════════════════════════════════════════════════════════

def generate_sparql(question: str, error_feedback: str = None) -> str:
    if error_feedback:
        prompt = f"""{SCHEMA_SUMMARY}

The previous SPARQL query failed with this error:
{error_feedback}

Write a simpler corrected SPARQL query for:
{question}

Remember: PREFIX ex: <http://example.org/> is MANDATORY on the first line.
Return ONLY the SPARQL query."""
    else:
        prompt = f"""{SCHEMA_SUMMARY}

Write a SPARQL SELECT query to answer:
{question}

Return ONLY the SPARQL query."""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    raw = response.json()["response"].strip()

    # Nettoyer les backticks markdown
    if "```" in raw:
        lines = raw.split("\n")
        cleaned = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            cleaned.append(line)
        raw = "\n".join(cleaned).strip()

    # Forcer le PREFIX si absent
    if "PREFIX ex:" not in raw:
        raw = "PREFIX ex: <http://example.org/>\n" + raw

    return raw


# ══════════════════════════════════════════════════════════════
#  Exécution SPARQL + déréférencement
# ══════════════════════════════════════════════════════════════

def execute_sparql(g: Graph, query: str):
    results = g.query(query)
    rows = []
    for row in results:
        row_data = []
        for val in row:
            if val is not None:
                raw = str(val).split("/")[-1]
                label = dereference(raw)
                row_data.append(label)
        if row_data:
            rows.append(row_data)
    return rows


# ══════════════════════════════════════════════════════════════
#  Pipeline avec self-repair
# ══════════════════════════════════════════════════════════════

def ask(g: Graph, question: str, verbose: bool = True) -> dict:
    if verbose:
        print(f"\n{'─'*60}")
        print(f"Question : {question}")

    query    = None
    results  = []
    error    = None
    attempts = 0
    repaired = False

    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt

        try:
            query = generate_sparql(question, error_feedback=error)
            if verbose:
                label = "SPARQL généré :" if attempt == 1 else f"SPARQL corrigé (tentative {attempt}) :"
                print(f"\n{label}")
                print(query)
        except Exception as e:
            error = f"Ollama error: {e}"
            continue

        try:
            results = execute_sparql(g, query)
            error = None
            if attempt > 1:
                repaired = True
            break
        except Exception as e:
            error = str(e)
            if verbose:
                print(f"\n⚠ Erreur SPARQL (tentative {attempt}) : {error[:120]}")

    if error:
        answer = "Échec après self-repair"
    elif not results:
        answer = "Aucun résultat trouvé"
    else:
        answer = ", ".join([" | ".join(r) for r in results[:5]])

    if verbose:
        print(f"\nRéponse : {answer}")
        if repaired:
            print(f"✓ Self-repair réussi en {attempts} tentatives")

    return {
        "question": question,
        "sparql":   query,
        "results":  results,
        "answer":   answer,
        "attempts": attempts,
        "repaired": repaired,
        "error":    error,
    }


# ══════════════════════════════════════════════════════════════
#  Évaluation baseline vs RAG
# ══════════════════════════════════════════════════════════════

EVAL_QUESTIONS = [
    "Where was Galileo born?",
    "Who influenced Aristotle?",
    "Where did Newton study?",
    "What is Galileo known for?",
    "Which scientists shared the same institution?",
]

BASELINE = {
    "Where was Galileo born?"                       : ["pisa", "florence"],
    "Who influenced Aristotle?"                     : ["plato", "socrates"],
    "Where did Newton study?"                       : ["cambridge"],
    "What is Galileo known for?"                    : ["astronomy", "physics", "dialogue"],
    "Which scientists shared the same institution?" : ["locke", "hooke", "newton"],
}


def evaluate(g: Graph):
    print("\n" + "═"*60)
    print("ÉVALUATION — Baseline vs RAG")
    print("═"*60)

    results_log = []

    for question in EVAL_QUESTIONS:
        result = ask(g, question, verbose=True)

        keywords  = BASELINE.get(question, [])
        got_lower = result["answer"].lower()
        match     = any(kw in got_lower for kw in keywords)

        print(f"  Attendu  : {', '.join(keywords)}")
        print(f"  Match    : {'✓' if match else '✗'}")

        results_log.append({
            "question": question,
            "expected": keywords,
            "got":      result["answer"],
            "match":    match,
            "attempts": result["attempts"],
            "repaired": result["repaired"],
        })

    # Rapport
    print("\n" + "═"*60)
    print("RAPPORT D'ÉVALUATION")
    print("═"*60)
    print(f"\n{'Question':<45} {'Match':<8} {'Tentatives':<12} {'Self-repair'}")
    print("─"*80)
    for r in results_log:
        print(f"{r['question']:<45} {'✓' if r['match'] else '✗':<8} {r['attempts']:<12} {'Oui' if r['repaired'] else 'Non'}")

    matched = sum(1 for r in results_log if r["match"])
    print(f"\nScore RAG      : {matched}/{len(results_log)}")
    print(f"Score baseline : 0/{len(results_log)} (pas de SPARQL automatique)")

    with open("kg_artifacts/rag_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)
    print("\nRapport sauvegardé : kg_artifacts/rag_evaluation.json")


# ══════════════════════════════════════════════════════════════
#  Mode interactif
# ══════════════════════════════════════════════════════════════

def interactive(g: Graph):
    print("\n" + "═"*60)
    print("RAG DEMO — Mode interactif (v2)")
    print("Tape 'quit' pour quitter, 'eval' pour l'évaluation")
    print("═"*60)

    while True:
        try:
            question = input("\nQuestion > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "eval":
            evaluate(g)
            continue
        ask(g, question, verbose=True)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        requests.get("http://localhost:11434", timeout=3)
    except Exception:
        print("ERREUR : Ollama ne répond pas.")
        print("Lance 'ollama serve' dans un autre terminal.")
        sys.exit(1)

    g = load_kb(KB_PATH)

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate(g)
    else:
        interactive(g)
