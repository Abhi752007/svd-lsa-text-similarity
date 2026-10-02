import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from preprocessing import preprocess_text


# Categories we want to use for the project
categories = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "sci.space",
    "talk.politics.misc",
]


# Number of usable documents required from each category
documents_per_category = 200


# Download/load the 20 Newsgroups training dataset.
#
# We remove headers, footers, and quoted replies because these can
# contain metadata or repeated text that can artificially affect
# document similarity.
dataset = fetch_20newsgroups(
    subset="train",
    categories=categories,
    remove=("headers", "footers", "quotes"),
    shuffle=True,
    random_state=42,
)


rows = []


# Process each category separately so that the final dataset
# contains exactly the same number of usable documents per category.
for category_id, category in enumerate(dataset.target_names):

    count = 0

    # Go through the documents belonging to the current category.
    for text, target in zip(dataset.data, dataset.target):

        # Skip documents belonging to other categories.
        if target != category_id:
            continue

        # Skip missing or invalid raw documents.
        if not isinstance(text, str) or not text.strip():
            continue

        # Apply the SAME preprocessing used by the main
        # preprocessing stage.
        cleaned_text = preprocess_text(text)

        # A document may contain text initially but become empty
        # after preprocessing. Do not include such documents.
        if not cleaned_text:
            continue

        # Store the original text and category.
        #
        # The processed text will be generated again by
        # preprocessing.py and saved separately.
        rows.append({
            "document_id": len(rows),
            "category": category,
            "text": text,
        })

        count += 1

        # Stop only after we have enough documents that are
        # confirmed to survive preprocessing.
        if count == documents_per_category:
            break

    print(f"{category}: {count} valid documents selected")


# Convert the selected documents into a DataFrame.
df = pd.DataFrame(rows)


# Save the raw selected documents.
df.to_csv("data/documents.csv", index=False)


# Display useful information for verification.
print("\nDataset created successfully")
print("Total documents:", len(df))

print("\nDocuments per category:")
print(df["category"].value_counts())

print("\nSaved to: data/documents.csv")