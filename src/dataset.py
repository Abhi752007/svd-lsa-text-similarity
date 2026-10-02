import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# Categories we selected for the project
categories = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "sci.space",
    "talk.politics.misc",
]

# Load the dataset
dataset = fetch_20newsgroups(
    subset="train",
    categories=categories,
    remove=("headers", "footers", "quotes"),
    shuffle=True,
    random_state=42,
)

# Number of valid documents required from each category
documents_per_category = 200

rows = []

# Select 200 valid documents from each category
for category_id, category in enumerate(dataset.target_names):

    count = 0

    for text, target in zip(dataset.data, dataset.target):

        # Only consider documents belonging to this category
        if target != category_id:
            continue

        # Skip missing or empty documents
        if not isinstance(text, str) or not text.strip():
            continue

        rows.append({
            "document_id": len(rows),
            "category": category,
            "text": text,
        })

        count += 1

        # Stop once we have 200 valid documents
        if count == documents_per_category:
            break

    print(f"{category}: {count} valid documents selected")

# Convert to a DataFrame
df = pd.DataFrame(rows)

# Save the dataset
df.to_csv("data/documents.csv", index=False)

print("\nDataset created successfully")
print("Total documents:", len(df))

print("\nDocuments per category:")
print(df["category"].value_counts())

print("\nSaved to: data/documents.csv")