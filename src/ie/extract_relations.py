import json
import spacy
import pandas as pd

INPUT_FILE = "crawler_output.jsonl"
OUTPUT_FILE = "extracted_relations.csv"

ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "DATE"}


def load_documents(jsonl_file):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


if __name__ == "__main__":
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_trf")

    rows = []

    for doc_data in load_documents(INPUT_FILE):
        text = doc_data["text"]
        source_url = doc_data["url"]

        doc = nlp(text)

        for sent in doc.sents:
            entities = [ent for ent in sent.ents if ent.label_ in ALLOWED_LABELS]

            if len(entities) < 2:
                continue

            # Identify the main verb of the sentence
            verbs = [token for token in sent if token.pos_ == "VERB"]

            if not verbs:
                continue

            verb = verbs[0].lemma_

            # Create simple relations between entity pairs
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    rows.append({
                        "subject": entities[i].text,
                        "predicate": verb,
                        "object": entities[j].text,
                        "source_url": source_url,
                        "sentence": sent.text.strip()
                    })

    df = pd.DataFrame(rows)
    df.drop_duplicates(inplace=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Relation extraction completed")
    print(f"Total relations extracted: {len(df)}")
    print(f"Output file: {OUTPUT_FILE}")