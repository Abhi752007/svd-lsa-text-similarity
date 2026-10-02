import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_documents(
    latent_matrix,
    input_file="data/processed_documents.csv",
    document_index=0,
    top_n=5
):
    # Load processed documents so we can display
    # the category and text of the results.
    df = pd.read_csv(input_file)

    # Calculate cosine similarity between the selected
    # document and every document in the latent space.
    similarities = cosine_similarity(
        latent_matrix[document_index].reshape(1, -1),
        latent_matrix
    )[0]

    # Create a copy of the document data.
    results = df.copy()

    # Add the similarity score for every document.
    results["similarity"] = similarities

    # Remove the query document itself.
    results = results[results.index != document_index]

    # Sort from highest similarity to lowest similarity.
    results = results.sort_values(
        by="similarity",
        ascending=False
    )

    # Display the query document.
    print("Query document:")
    print(df.iloc[document_index]["processed_text"][:500])

    print("\nTop similar documents:\n")

    # Display the top results.
    for rank, (_, row) in enumerate(
        results.head(top_n).iterrows(),
        start=1
    ):
        print(f"Rank {rank}")
        print(f"Category: {row['category']}")
        print(f"Similarity: {row['similarity']:.4f}")
        print(f"Text: {row['processed_text'][:200]}")
        print("-" * 60)