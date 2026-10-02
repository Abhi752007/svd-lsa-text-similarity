# Dataset

This project uses the 20 Newsgroups text dataset provided through
scikit-learn's `fetch_20newsgroups` loader.

## Selected categories

- comp.graphics
- sci.space
- rec.sport.baseball
- talk.politics.misc
- sci.med

## Dataset construction

We use a balanced subset of approximately 200 documents per category.

Headers, footers, and quoted replies are removed during dataset loading
to reduce metadata leakage and keep the similarity experiment focused on
document content.

The dataset is generated programmatically using a fixed random seed so
that team members can reproduce the same dataset.

## Important

The category label is retained for evaluation only.

It is not used when calculating TF-IDF, SVD, or cosine similarity.
