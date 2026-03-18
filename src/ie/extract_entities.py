import json
import spacy
import pandas as pd

INPUT_FILE = "crawler_output.jsonl"
OUTPUT_FILE = "extracted_knowledge.csv"

# Types d'entités à conserver
ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "DATE"}


def load_documents(jsonl_file):
    documents = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def extract_entities(text, nlp):
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        if ent.label_ in ALLOWED_LABELS:
            entities.append({
                "entity": ent.text,
                "label": ent.label_
            })

    return entities


if __name__ == "__main__":
    print("Chargement du modèle spaCy (ça peut prendre un peu de temps)...")
    nlp = spacy.load("en_core_web_trf")

    documents = load_documents(INPUT_FILE)

    rows = []

    for doc in documents:
        text = doc["text"]
        source_url = doc["url"]

        entities = extract_entities(text, nlp)

        for ent in entities:
            rows.append({
                "entity": ent["entity"],
                "label": ent["label"],
                "source_url": source_url
            })

    df = pd.DataFrame(rows)
    df.drop_duplicates(inplace=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Extraction terminée")
    print(f"Nombre total d'entités extraites: {len(df)}")
    print(f"Fichier créé: {OUTPUT_FILE}")
