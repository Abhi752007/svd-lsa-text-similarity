import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_documents(
    matrix,
    input_file="data/processed_documents.csv",
    document_index=0,
    top_n=5
):
    # Load document data
    df = pd.read_csv(input_file)

    # Calculate similarities
    similarities = cosine_similarity(
        matrix[document_index].reshape(1, -1),
        matrix
    )[0]

    results = df.copy()
    results["similarity"] = similarities

    # Remove the query itself
    results = results[results.index != document_index]

    # Highest similarity first
    results = results.sort_values(
        by="similarity",
        ascending=False
    )

    print("Query document:")
    print(df.iloc[document_index]["processed_text"][:500])

    print("\nTop similar documents:\n")

    for rank, (_, row) in enumerate(
        results.head(top_n).iterrows(),
        start=1
    ):
        print(f"Rank {rank}")
        print(f"Category: {row['category']}")
        print(f"Similarity: {row['similarity']:.4f}")
        print(f"Text: {row['processed_text'][:200]}")
        print("-" * 60)