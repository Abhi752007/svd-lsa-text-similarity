from src.tfidf import create_tfidf_matrix
from src.svd import create_svd_representation
from src.prediction import find_similar_input


K = 20
TOP_N = 5


def main():

    print("\n=== SVD-Based Text Similarity ===")

    # Build TF-IDF
    tfidf_matrix, vectorizer, df = create_tfidf_matrix()

    # Build LSA
    latent_matrix, svd = create_svd_representation(
        tfidf_matrix,
        n_components=K
    )

    print("\nModel ready.")
    print("-" * 50)

    # Get user input
    user_text = input("Enter your text: ")

    # Find similar documents
    result = find_similar_input(
        user_text,
        vectorizer,
        svd,
        latent_matrix,
        top_n=TOP_N
    )

    if result is None:
        print("\nNo usable text was provided.")
        return

    print("\n" + "=" * 50)
    print(f"Predicted topic: {result['predicted_topic']}")
    print("=" * 50)

    print("\nTop similar documents:\n")

    for rank, document in enumerate(
        result["similar_documents"],
        start=1
    ):
        print(f"{rank}. {document['category']}")
        print(f"   Similarity: {document['similarity']:.4f}")
        print(f"   {document['text'][:180]}...")
        print()


if __name__ == "__main__":
    main()