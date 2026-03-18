# science-knowledge-graph

A full Knowledge Graph pipeline applied to famous scientists — from web crawling to RAG-powered SPARQL querying.

Built as part of a Web Mining & Semantics course project.

---

## Overview

This project builds a private Knowledge Graph (KG) about famous scientists using a complete pipeline:

1. **Web Crawling** — scrape biographical pages about scientists
2. **Information Extraction** — NER (spaCy) + relation extraction
3. **KB Construction** — RDF graph + OWL ontology (rdflib)
4. **Alignment** — entity/predicate linking to Wikidata / DBpedia
5. **Reasoning** — SWRL rules with OWLReady2
6. **KGE** — Knowledge Graph Embeddings (TransE, RotatE via PyKEEN)
7. **RAG** — Natural Language → SPARQL with self-repair (Ollama)

---

## Project Structure

```
science-knowledge-graph/
├── src/
│   ├── crawl/
│   │   └── crawler.py              # Web scraper (trafilatura)
│   ├── ie/
│   │   ├── extract_entities.py     # NER with spaCy
│   │   ├── extract_relations.py    # Subject-verb-object extraction
│   │   └── clean_relations.py      # Relation filtering
│   ├── kg/
│   │   └── build_private_kb.py     # RDF graph construction (rdflib)
│   ├── reason/
│   │   └── swrl_rules.py           # SWRL reasoning (OWLReady2)
│   ├── kge/
│   │   └── train_kge.py            # KGE training (PyKEEN)
│   └── rag/
│       └── rag_pipeline.py         # NL→SPARQL + self-repair (Ollama)
├── data/
│   ├── crawler_output.jsonl
│   ├── extracted_knowledge.csv
│   ├── extracted_relations.csv
│   ├── extracted_relations_cleaned.csv
│   └── README.md
├── kg_artifacts/
│   ├── ontology.ttl
│   ├── graph.nt
│   ├── expanded.nt
│   └── alignment.ttl
├── kge_datasets/
│   ├── train.txt
│   ├── valid.txt
│   └── test.txt
├── reports/
│   └── final_report.pdf
├── notebooks/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) (for the RAG module)

### Setup

```bash
git clone https://github.com/<your-username>/science-knowledge-graph.git
cd science-knowledge-graph
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

---

## How to Run Each Module

### 1. Crawl

```bash
python src/crawl/crawler.py
# Output: data/crawler_output.jsonl
```

### 2. Entity & Relation Extraction

```bash
python src/ie/extract_entities.py
# Output: data/extracted_knowledge.csv

python src/ie/extract_relations.py
# Output: data/extracted_relations.csv

python src/ie/clean_relations.py
# Output: data/extracted_relations_cleaned.csv
```

### 3. Build Knowledge Graph

```bash
python src/kg/build_private_kb.py
# Output: kg_artifacts/graph.nt
```

### 4. SWRL Reasoning

```bash
python src/reason/swrl_rules.py
# Output: reasoning results in terminal
```

### 5. KGE Training

```bash
python src/kge/train_kge.py
# Output: kge_datasets/ + evaluation metrics
```

### 6. RAG Demo (NL → SPARQL)

```bash
# Start Ollama first
ollama serve

# Run the RAG pipeline
python src/rag/rag_pipeline.py
```

---

## RAG Demo

The RAG pipeline takes a natural language question, generates a SPARQL query using an LLM (via Ollama), executes it against the local RDF graph, and auto-repairs the query if execution fails.

Example:

```
Question: Who did Galileo collaborate with?
→ SPARQL generated → executed → result returned
```

A screenshot of the demo is available in `reports/`.

---

## Hardware Requirements

- RAM: 8 GB minimum (16 GB recommended for spaCy transformer model)
- GPU: optional but speeds up KGE training significantly
- Disk: ~2 GB for model weights (spaCy + Ollama LLM)

---

## KB Statistics

| Metric | Value |
|---|---|
| Source pages crawled | 4 |
| Entities extracted | TBD |
| Relations (raw) | TBD |
| Relations (cleaned) | TBD |
| RDF triples | TBD |

*To be updated after full pipeline run.*

---

## Requirements

See `requirements.txt` for full list. Key dependencies:

```
rdflib
spacy
pandas
requests
trafilatura
owlready2
pykeen
ollama
```

---

## License

MIT
