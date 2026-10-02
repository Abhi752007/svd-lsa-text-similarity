import re
import pandas as pd


def preprocess_text(text):
    """
    Clean a document before converting it into numerical features.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Keep only alphabetic characters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def preprocess_dataset(input_file="data/documents.csv",
                       output_file="data/processed_documents.csv"):

    # Load the dataset
    df = pd.read_csv(input_file)

    # Apply preprocessing to every document
    df["processed_text"] = df["text"].apply(preprocess_text)

    # Keep only the information we need
    processed_df = df[["document_id", "category", "processed_text"]]

    # Save the processed dataset
    processed_df.to_csv(output_file, index=False)

    print("Preprocessing completed successfully")
    print("Documents processed:", len(processed_df))
    print("Saved to:", output_file)


if __name__ == "__main__":
    preprocess_dataset()