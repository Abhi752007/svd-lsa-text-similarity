import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def precision_at_k(
    matrix,
    labels,
    query_index,
    top_n=5
):
    # Compare query with all documents
    similarities = cosine_similarity(
        matrix[query_index].reshape(1, -1),
        matrix
    )[0]

    # Ignore the query itself
    similarities[query_index] = -1

    # Get top results
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    # Count matching categories
    same_category = sum(
        labels[index] == labels[query_index]
        for index in top_indices
    )

    return same_category / top_n


def evaluate_matrix(
    matrix,
    labels,
    query_indices,
    top_n=5
):
    # Calculate Precision@5 for each query
    scores = [
        precision_at_k(
            matrix,
            labels,
            index,
            top_n
        )
        for index in query_indices
    ]

    # Return average precision
    return np.mean(scores)


def evaluate_by_category(
    matrix,
    labels,
    top_n=5
):
    # Store category-wise scores
    category_scores = {}

    # Evaluate each category separately
    for category in np.unique(labels):

        # Get documents from this category
        category_indices = np.where(labels == category)[0]

        # Calculate scores for this category
        score = evaluate_matrix(
            matrix,
            labels,
            category_indices,
            top_n
        )

        category_scores[category] = score

    return category_scores