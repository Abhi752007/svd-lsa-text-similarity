from sklearn.decomposition import TruncatedSVD


def create_svd_representation(
    tfidf_matrix,
    n_components=100
):
    # Create SVD model
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )

    # Reduce TF-IDF into LSA space
    latent_matrix = svd.fit_transform(tfidf_matrix)

    return latent_matrix, svd