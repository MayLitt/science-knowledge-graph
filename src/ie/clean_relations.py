import pandas as pd

INPUT_FILE = "extracted_relations.csv"
OUTPUT_FILE = "extracted_relations_cleaned.csv"

GENERIC_VERBS = {
    "be", "have", "do", "make", "use", "say"
}

MAX_ENTITY_LENGTH = 7  # maximum number of words per entity

df = pd.read_csv(INPUT_FILE)

initial_count = len(df)

# 1. Remove generic predicates
df = df[~df["predicate"].isin(GENERIC_VERBS)]

# 2. Remove self-relations
df = df[df["subject"] != df["object"]]

# 3. Remove overly long entities
df = df[
    (df["subject"].str.split().str.len() <= MAX_ENTITY_LENGTH) &
    (df["object"].str.split().str.len() <= MAX_ENTITY_LENGTH)
]

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("Cleaning completed")
print(f"Initial relations: {initial_count}")
print(f"Relations after cleaning: {len(df)}")
print(f"Output file: {OUTPUT_FILE}")