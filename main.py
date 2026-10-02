import pandas as pd

from src.tfidf import create_tfidf_matrix
from src.svd import create_svd_representation
from src.evaluation import (
    evaluate_matrix,
    evaluate_by_category
)


# SVD ranks to test
K_VALUES = [20, 50, 100, 200]

# Number of results per query
TOP_N = 5


# Create TF-IDF once
tfidf_matrix, vectorizer, df = create_tfidf_matrix()

# Get category labels
labels = df["category"].to_numpy()

# Store all experiment results
results = []


# --------------------------------
# TF-IDF evaluation
# --------------------------------

print("\n--- TF-IDF ---")

tfidf_score = evaluate_matrix(
    tfidf_matrix,
    labels,
    range(len(labels)),
    TOP_N
)

print(f"Overall Precision@5: {tfidf_score:.4f}")

tfidf_categories = evaluate_by_category(
    tfidf_matrix,
    labels,
    TOP_N
)

results.append({
    "method": "TF-IDF",
    "k": None,
    "category": "Overall",
    "precision_at_5": tfidf_score
})

for category, score in tfidf_categories.items():
    print(f"{category}: {score:.4f}")

    results.append({
        "method": "TF-IDF",
        "k": None,
        "category": category,
        "precision_at_5": score
    })


# --------------------------------
# LSA evaluation
# --------------------------------

print("\n--- LSA ---")

for K in K_VALUES:

    # Create latent representation
    latent_matrix, svd = create_svd_representation(
        tfidf_matrix,
        n_components=K
    )

    # Overall score
    overall_score = evaluate_matrix(
        latent_matrix,
        labels,
        range(len(labels)),
        TOP_N
    )

    print(f"\nk = {K}")
    print(f"Overall Precision@5: {overall_score:.4f}")

    results.append({
        "method": "LSA",
        "k": K,
        "category": "Overall",
        "precision_at_5": overall_score
    })

    # Category-wise scores
    category_scores = evaluate_by_category(
        latent_matrix,
        labels,
        TOP_N
    )

    for category, score in category_scores.items():
        print(f"{category}: {score:.4f}")

        results.append({
            "method": "LSA",
            "k": K,
            "category": category,
            "precision_at_5": score
        })


# --------------------------------
# Save results
# --------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/precision_at_5.csv",
    index=False
)

print("\nResults saved to: results/precision_at_5.csv")