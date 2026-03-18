import pandas as pd

INPUT_FILE = "extracted_relations.csv"
OUTPUT_FILE = "extracted_relations_cleaned.csv"

GENERIC_VERBS = {
    "be", "have", "do", "make", "use", "say"
}

MAX_ENTITY_LENGTH = 7  # en nombre de mots

df = pd.read_csv(INPUT_FILE)

initial_count = len(df)

# 1. supprimer prédicats génériques
df = df[~df["predicate"].isin(GENERIC_VERBS)]

# 2. supprimer auto-relations
df = df[df["subject"] != df["object"]]

# 3. supprimer entités trop longues
df = df[
    (df["subject"].str.split().str.len() <= MAX_ENTITY_LENGTH) &
    (df["object"].str.split().str.len() <= MAX_ENTITY_LENGTH)
]

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("Nettoyage terminé")
print(f"Relations initiales : {initial_count}")
print(f"Relations après nettoyage : {len(df)}")
print(f"Fichier créé : {OUTPUT_FILE}")
