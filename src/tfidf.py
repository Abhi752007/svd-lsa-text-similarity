import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_matrix(
    input_file="data/processed_documents.csv",
):
    # Load processed documents
    df = pd.read_csv(input_file)

    # Extract the cleaned text
    documents = df["processed_text"]

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        max_df=0.95
    )

    # Convert documents into TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Get vocabulary
    feature_names = vectorizer.get_feature_names_out()

    print("TF-IDF created successfully")
    print("Number of documents:", tfidf_matrix.shape[0])
    print("Number of terms:", tfidf_matrix.shape[1])
    print("Matrix shape:", tfidf_matrix.shape)
    print("Non-zero values:", tfidf_matrix.nnz)

    print("\nFirst 20 terms:")
    print(feature_names[:20])

    return tfidf_matrix, vectorizer


if __name__ == "__main__":
    create_tfidf_matrix()