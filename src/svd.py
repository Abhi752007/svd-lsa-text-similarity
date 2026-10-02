import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def create_svd_representation(
    input_file="data/processed_documents.csv",
    n_components=100
):
    # Load processed documents
    df = pd.read_csv(input_file)

    #Extract documents
    documents = df["processed_text"]

    # Create TF-IDF Matrix
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    # Create SVD model
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    # Reduce TF-IDF matrix to latent semantic space
    latent_matrix = svd.fit_transform(tfidf_matrix)

    print("SVD completed successfully")
    print("Original TF-IDF shape:", tfidf_matrix.shape)
    print("Latent matrix shape:", latent_matrix.shape)
    print("Number of components:", n_components)   

    return latent_matrix, svd

if __name__ == "__main__":
    create_svd_representation()