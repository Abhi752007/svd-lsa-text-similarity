from sklearn.decomposition import TruncatedSVD


def create_svd_representation(
    tfidf_matrix,
    n_components=100
):
    # Create the SVD model.
    #
    # n_components is the value of k:
    # the number of latent semantic dimensions we keep.
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )

    # Reduce the TF-IDF matrix into the latent semantic space.
    latent_matrix = svd.fit_transform(tfidf_matrix)

    print("SVD completed successfully")
    print("Original TF-IDF shape:", tfidf_matrix.shape)
    print("Latent matrix shape:", latent_matrix.shape)
    print("Number of components:", n_components)

    return latent_matrix, svd