import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text


def find_similar_input(
    user_text,
    vectorizer,
    svd,
    latent_matrix,
    input_file="data/processed_documents.csv",
    top_n=5
):
    # Load document information
    df = pd.read_csv(input_file)

    # Clean the user's text
    cleaned_text = preprocess_text(user_text)

    if not cleaned_text:
        return None

    # Convert user text to TF-IDF
    user_tfidf = vectorizer.transform([cleaned_text])

    # Convert TF-IDF to LSA
    user_latent = svd.transform(user_tfidf)

    # Compare input with all documents
    similarities = cosine_similarity(
        user_latent,
        latent_matrix
    )[0]

    # Get top similar documents
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Calculate topic scores
    topic_scores = {}

    for index in top_indices:
        category = df.iloc[index]["category"]
        similarity = similarities[index]

        topic_scores[category] = (
            topic_scores.get(category, 0) + similarity
        )

    # Select highest-scoring topic
    predicted_topic = max(
        topic_scores,
        key=topic_scores.get
    )

    # Store document results
    similar_documents = []

    for index in top_indices:
        similar_documents.append({
            "category": df.iloc[index]["category"],
            "similarity": similarities[index],
            "text": df.iloc[index]["processed_text"]
        })

    return {
        "predicted_topic": predicted_topic,
        "similar_documents": similar_documents
    }