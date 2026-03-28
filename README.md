# Scientists & Discoveries Knowledge Graph

A full Knowledge Graph pipeline applied to famous scientists — from web crawling to RAG-powered SPARQL querying.

Built as part of a Web Mining & Semantics course project.

---

## Overview

This project builds a private Knowledge Graph (KG) about famous scientists using a complete pipeline:

1. **Web Crawling** — scrape biographical pages about scientists
2. **Information Extraction** — NER (spaCy) + relation extraction
3. **KB Construction** — RDF graph + OWL ontology (rdflib)
4. **Alignment** — entity/predicate linking to Wikidata / DBpedia
5. **SPARQL Expansion** — enrich KB by querying Wikidata
6. **Reasoning** — SWRL rules with OWLReady2
7. **KGE** — Knowledge Graph Embeddings (TransE, RotatE via PyKEEN)
8. **RAG** — Natural Language to SPARQL with self-repair (Ollama + llama3.2)

---

## Project Structure

```
science-knowledge-graph/
├── src/
│   ├── crawl/
│   │   └── crawler.py                  # Web scraper (trafilatura)
│   ├── ie/
│   │   ├── extract_entities.py         # NER with spaCy (en_core_web_sm)
│   │   ├── extract_relations.py        # Subject-verb-object extraction
│   │   └── clean_relations.py          # Relation filtering
│   ├── kg/
│   │   ├── build_private_kb.py         # RDF graph construction (rdflib)
│   │   ├── align_predicates.py         # Predicate alignment to ontology
│   │   └── expand_kg.py                # SPARQL expansion via Wikidata
│   ├── reason/
│   │   └── swrl_rules.py               # SWRL reasoning (OWLReady2)
│   ├── kge/
│   │   └── train_kge.py                # KGE training (PyKEEN)
│   └── rag/
│       └── rag_pipeline.py             # NL to SPARQL + self-repair (Ollama)
├── data/
│   ├── crawler_output.jsonl
│   ├── extracted_knowledge.csv
│   ├── extracted_relations.csv
│   └── extracted_relations_cleaned.csv
├── kg_artifacts/
│   ├── ontology.ttl                    # OWL ontology (6 classes, 13 properties)
│   ├── graph.nt                        # Initial RDF graph (25,028 triples)
│   ├── alignment.ttl                   # Entity linking to Wikidata (21 entities)
│   ├── aligned_kb.nt                   # KB with aligned predicates
│   ├── expanded.nt                     # KB after Wikidata expansion (24,656 triples)
│   ├── expanded_reasoned.nt            # KB after SWRL inference (24,951 triples)
│   ├── alignment_report.txt            # Predicate alignment log
│   ├── expansion_report.txt            # Wikidata expansion log
│   └── rag_evaluation.json             # RAG evaluation results
├── kge_datasets/
│   ├── RotatE/
│   │   ├── training_triples/
│   │   ├── metadata.json
│   │   ├── results.json
│   │   └── trained_model.pkl
│   ├── TransE/
│   │   ├── training_triples/
│   │   ├── metadata.json
│   │   ├── results.json
│   │   └── trained_model.pkl
│   ├── comparison_report.txt
│   ├── results_RotatE.json
│   ├── results_TransE.json
│   ├── test.txt
│   ├── train.txt
│   ├── tsne_RotatE.png
│   ├── tsne_TransE.png
│   └── valid.txt
├── reports/
│   └── final_report.pdf
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) (for the RAG module)

### Setup

```bash
git clone https://github.com/<your-username>/science-knowledge-graph.git
cd science-knowledge-graph
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Ollama setup

```bash
# Install Ollama from https://ollama.com/download
# Then pull the model:
ollama pull llama3.2
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

### 3. Build & Align Knowledge Graph

```bash
python src/kg/build_private_kb.py
# Output: kg_artifacts/graph.nt

python src/kg/align_predicates.py
# Output: kg_artifacts/aligned_kb.nt

python src/kg/expand_kg.py
# Output: kg_artifacts/expanded.nt
```

### 4. SWRL Reasoning

```bash
python src/reason/swrl_rules.py
# Output: kg_artifacts/expanded_reasoned.nt
```

### 5. KGE Training

```bash
python src/kge/train_kge.py
# Output: kge_datasets/ (splits, models, t-SNE plots, metrics)
```

### 6. RAG Demo (NL to SPARQL)

```bash
# Ollama runs automatically in the background after install

# Interactive mode:
python src/rag/rag_pipeline.py

# Evaluation mode (10 questions, baseline vs RAG):
python src/rag/rag_pipeline.py eval
```

---

## RAG Demo

The RAG pipeline converts a natural language question into a SPARQL query using llama3.2 via Ollama, executes it against the local RDF graph, and auto-repairs the query if execution fails (up to 3 attempts).

```
Question > Where was Galileo born?

SPARQL generated:
PREFIX ex: <http://example.org/>
SELECT ?place WHERE { ex:Galileo ex:bornIn ?place }

Answer: Pisa
```

RAG evaluation score: **8/10** questions correctly answered.

---

## KB Statistics

| Metric | Value |
|---|---|
| Source pages crawled | 16 |
| Entities extracted | 11,435 |
| Relations (raw) | 27,227 |
| Relations (cleaned) | 25,273 |
| RDF triples — initial | 25,028 |
| RDF triples — after alignment | 24,467 |
| RDF triples — after Wikidata expansion | 24,656 |
| RDF triples — after SWRL reasoning | 24,951 |
| Entities linked to Wikidata | 21 |
| KGE train / valid / test | 19,572 / 2,354 / 2,355 |

---

## Hardware Requirements

- RAM: 8 GB minimum (16 GB recommended for spaCy transformer model)
- GPU: optional but speeds up KGE training
- Disk: ~4 GB (spaCy model ~500 MB + Ollama llama3.2 ~2 GB)

---

## License

MIT
