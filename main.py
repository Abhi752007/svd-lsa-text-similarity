from src.tfidf import create_tfidf_matrix
from src.svd import create_svd_representation
from src.prediction import find_similar_input


# SVD rank used by the demo
K = 20

# Number of similar documents to show
TOP_N = 5


# Build TF-IDF representation
tfidf_matrix, vectorizer, df = create_tfidf_matrix()


# Build LSA representation
latent_matrix, svd = create_svd_representation(
    tfidf_matrix,
    n_components=K
)


# Get user input
user_text = input("\nEnter a document or text: ")


# Run prediction
result = find_similar_input(
    user_text,
    vectorizer,
    svd,
    latent_matrix,
    top_n=TOP_N
)


if result is None:
    print("Input text is empty after preprocessing.")

else:
    print(f"\nPredicted topic: {result['predicted_topic']}")

    print("\nTop similar documents:\n")

    for rank, document in enumerate(
        result["similar_documents"],
        start=1
    ):
        print(f"Rank {rank}")
        print(f"Category: {document['category']}")
        print(f"Similarity: {document['similarity']:.4f}")
        print(f"Text: {document['text'][:200]}")
        print("-" * 60)