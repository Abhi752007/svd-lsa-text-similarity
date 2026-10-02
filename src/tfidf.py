import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_matrix(
    input_file="data/processed_documents.csv"
):
    # Load processed documents
    df = pd.read_csv(input_file)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        max_df=0.95
    )

    # Convert documents to TF-IDF
    tfidf_matrix = vectorizer.fit_transform(
        df["processed_text"]
    )

    return tfidf_matrix, vectorizer, df