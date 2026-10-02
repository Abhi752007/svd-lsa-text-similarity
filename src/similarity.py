import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_documents(
    input_file="data/processed_documents.csv",
    document_index=0,
    n_components=100,
    top_n=5
):
    # Load processed documents
    df = pd.read_csv(input_file)

    documents = df["processed_text"]

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    # Apply SVD
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )

    latent_matrix = svd.fit_transform(tfidf_matrix)

    # Calculate similarity between the selected document
    # and every other document
    similarities = cosine_similarity(
        latent_matrix[document_index].reshape(1, -1),
        latent_matrix
    )[0]

    # Create a copy of the similarities
    results = df.copy()
    results["similarity"] = similarities

    # Remove the document itself
    results = results[results.index != document_index]

    # Sort by similarity
    results = results.sort_values(
        by="similarity",
        ascending=False
    )

    print("Query document:")
    print(df.iloc[document_index]["processed_text"][:500])

    print("\nTop similar documents:\n")

    for rank, (_, row) in enumerate(results.head(top_n).iterrows(), start=1):
        print(f"Rank {rank}")
        print(f"Category: {row['category']}")
        print(f"Similarity: {row['similarity']:.4f}")
        print(f"Text: {row['processed_text'][:200]}")
        print("-" * 60)


if __name__ == "__main__":
    find_similar_documents()