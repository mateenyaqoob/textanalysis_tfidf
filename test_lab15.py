"""
Automated Pytest Suite for Lab 15: TF-IDF & Unstructured Data Triage
This script validates student implementations in student_starter_script.py.
"""

## Setup & Dataset Loading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources quietly
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load 4 real-world unstructured user discussion categories
CATEGORIES = [
    "sci.electronics",
    "rec.autos",
    "comp.sys.mac.hardware",
    "sci.med",
]

print("Loading dataset...")
raw_data = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
)

df = pd.DataFrame(
    {
        "text": raw_data.data,
        "true_label": [raw_data.target_names[i] for i in raw_data.target],
    }
)

# Keep posts with meaningful length (> 50 characters)
df = df[df["text"].str.strip().str.len() > 50].reset_index(drop=True)
print(f"Loaded {len(df)} unstructured user posts.\n")


# %% [markdown]
# ## Step 1: Text Cleaning, Tokenization & Normalization (10 mins)

# %%
sample_post = df["text"].iloc[0][:400]
print("--- Raw Sample Post ---")
print(sample_post)
print("-" * 40)

# TODO 1.1: Tokenize `sample_post` using `word_tokenize`, convert to lowercase,
# and keep only alphabetic tokens (`w.isalpha()`).
tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]

# TODO 1.2: Remove standard English stopwords using `stopwords.words('english')`.
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]

# TODO 1.3: Apply PorterStemmer and WordNetLemmatizer (with pos='v') to `filtered_tokens`.
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(w, pos="v") for w in filtered_tokens]

if filtered_tokens is not None:
    print("Filtered Tokens:  ", filtered_tokens[:12])
    print("Stemmed Roots:    ", stemmed_tokens[:12])
    print("Lemmatized Words: ", lemmatized_tokens[:12])


# %% [markdown]
# ## Step 2: Bag of Words vs. TF-IDF Feature Extraction (15 mins)

# %%
# TODO 2.1: Initialize CountVectorizer and TfidfVectorizer with:
# stop_words='english', max_features=4000, min_df=5
bow_vec = CountVectorizer(
    stop_words="english",
    max_features=4000,
    min_df=5,
)
tfidf_vec = TfidfVectorizer(
    stop_words="english",
    max_features=4000,
    min_df=5,
)

# TODO 2.2: Fit and transform `df['text']` to create `X_bow` and `X_tfidf`
X_bow = bow_vec.fit_transform(df["text"])
X_tfidf = tfidf_vec.fit_transform(df["text"])

# TODO 2.3: Calculate matrix sparsity (% of zero entries in `X_tfidf`)
if X_tfidf is not None:
    total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
    sparsity = 100.0 * (1.0 - (X_tfidf.nnz / total_elements))
    print(f"\nTF-IDF Matrix Shape: {X_tfidf.shape} | Sparsity: {sparsity:.2f}%")

    # TODO 2.4: Compute mean score per feature across documents and plot Top 10 TF-IDF words
    tfidf_means = np.asarray(X_tfidf.mean(axis=0)).ravel()
    top_idx = tfidf_means.argsort()[-10:]
    top_words = tfidf_vec.get_feature_names_out()[top_idx]
    top_scores = tfidf_means[top_idx]

    plt.figure(figsize=(8, 4.5))
    plt.barh(top_words, top_scores, color="teal")
    plt.xlabel("Average TF-IDF Weight")
    plt.title("Top 10 Most Influential Words (Baseline TF-IDF)")
    plt.tight_layout()
    plt.savefig("top_features.png", dpi=150)
    plt.show()


# %% [markdown]
# ## Step 3: Automated Topic Discovery & Spatial Mapping (15 mins)

# %%
if X_tfidf is not None:
    # TODO 3.1: Fit KMeans with `n_clusters=4, random_state=42, n_init=10` on `X_tfidf`
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    feature_names = tfidf_vec.get_feature_names_out()
    centers = kmeans.cluster_centers_

    # TODO 3.2: Print the top 8 defining TF-IDF words per cluster centroid
    # and plot a seaborn heatmap (`cluster_heatmap.png`) of the top 5 words per cluster.
    print("\n--- Top 8 words per cluster centroid ---")
    top5_words = []
    for i in range(4):
        top8_idx = centers[i].argsort()[-8:][::-1]
        top8_words = feature_names[top8_idx]
        print(f"Cluster {i}: {', '.join(top8_words)}")

        # take top-5 per cluster for the heatmap
        top5_words.append(list(feature_names[centers[i].argsort()[-5:][::-1]]))

    # Build a 4 x 5 heatmap matrix of centroid weights
    heatmap_data = np.zeros((4, 5))
    heatmap_cols = []
    for i in range(4):
        for j, word in enumerate(top5_words[i]):
            heatmap_data[i, j] = centers[i, feature_names.tolist().index(word)]
        heatmap_cols.append(f"C{i}_" + "|".join(top5_words[i]))
    # rename columns to be unique per row for display clarity
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=[f"Cluster {i}" for i in range(4)],
        columns=[f"Top{j+1}" for j in range(5)],
    )

    plt.figure(figsize=(9, 4.5))
    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Cluster Centroid Weights (Top-5 TF-IDF Words per Cluster)")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.show()

    # Also print the word order per column for interpretability
    for i in range(4):
        print(f"  Heatmap row Cluster {i}: " + " | ".join(top5_words[i]))

    # TODO 3.3: Project `X_tfidf` to 2 dimensions using `TruncatedSVD(n_components=2, random_state=42)`
    # and save a 2D scatter plot (`spatial_mapping.png`) colored by `df["cluster"]`.
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X_tfidf)

    plt.figure(figsize=(9, 6))
    palette = sns.color_palette("tab10", n_colors=4)
    for c in range(4):
        mask = df["cluster"].values == c
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            alpha=0.6,
            color=palette[c],
            label=f"Cluster {c}",
        )
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.title("2D Spatial Mapping of TF-IDF Document Clusters")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.show()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %%
# Challenge 1 (Noise Filtering):
# TODO 4.1: Build a cleaner `TfidfVectorizer` using `token_pattern=r'\b[a-zA-Z]{3,}\b'`,
# `max_df=0.5`, and custom conversational stopwords (e.g., 'know', 'like', 'just', 'think', 'does', 'use', 'thanks').
# Re-fit KMeans and compare cluster keywords!
conversational_stops = [
    "know", "like", "just", "think", "does", "use", "thanks",
    "would", "could", "really", "want", "get", "one", "also",
    "well", "thing", "going", "much", "many", "even", "make",
]

tfidf_clean = TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    min_df=5,
    max_features=4000,
    stop_words=conversational_stops + list(stopwords.words("english")),
)
X_tfidf_clean = tfidf_clean.fit_transform(df["text"])
km_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
clean_labels = km_clean.fit_predict(X_tfidf_clean)

print("\n--- Challenge 1: Cleaned TF-IDF clusters ---")
clean_feat = tfidf_clean.get_feature_names_out()
for i in range(4):
    top8 = clean_feat[km_clean.cluster_centers_[i].argsort()[-8:][::-1]]
    print(f"Clean Cluster {i}: {', '.join(top8)}")


# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Run `pd.crosstab(df['true_label'], df['cluster'])` and inspect overlapping categories.
crosstab = pd.crosstab(df["true_label"], df["cluster"])
print("\n--- Challenge 2: Crosstab (true_label x cluster) ---")
print(crosstab)

# Identify which pair of true labels most often shares a cluster
# For each pair of labels, count max overlap per cluster
label_pairs = []
labels = crosstab.index.tolist()
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        # sum of the minimum of the two label distributions across clusters
        overlap = np.minimum(crosstab.loc[labels[i]], crosstab.loc[labels[j]]).sum()
        label_pairs.append((labels[i], labels[j], overlap))
label_pairs.sort(key=lambda x: -x[2])
print("\nMost overlapping label pairs (by cluster co-occupancy):")
for a, b, o in label_pairs[:3]:
    print(f"  {a}  <->  {b}: overlap score {o}")


# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 vector norms (`np.linalg.norm`) of the longest and shortest posts in `X_bow` vs `X_tfidf`.
lengths = df["text"].str.len().values
longest_idx = int(np.argmax(lengths))
shortest_idx = int(np.argmin(lengths))

norms_bow = np.linalg.norm(X_bow.toarray(), axis=1)
norms_tfidf = np.linalg.norm(X_tfidf.toarray(), axis=1)

print("\n--- Challenge 3: L2 Norms (BoW vs TF-IDF) ---")
print(f"Longest  post idx={longest_idx}: len={lengths[longest_idx]:>6d} | "
      f"||x_bow||={norms_bow[longest_idx]:.3f} | ||x_tfidf||={norms_tfidf[longest_idx]:.3f}")
print(f"Shortest post idx={shortest_idx}: len={lengths[shortest_idx]:>6d} | "
      f"||x_bow||={norms_bow[shortest_idx]:.3f} | ||x_tfidf||={norms_tfidf[shortest_idx]:.3f}")
print(f"BoW norm ratio (long/short): {norms_bow[longest_idx] / norms_bow[shortest_idx]:.2f}x")
print(f"TF-IDF norm ratio (long/short): {norms_tfidf[longest_idx] / norms_tfidf[shortest_idx]:.2f}x")
print("NOTE: TF-IDF norms ≈ 1.0 for every doc because TfidfVectorizer uses norm='l2'.")


# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return or print the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` (test with doc_id=412).
    """
    if doc_id < 0 or doc_id >= matrix.shape[0]:
        raise IndexError(f"doc_id {doc_id} out of range (0..{matrix.shape[0]-1})")

    row = matrix.getrow(doc_id) if hasattr(matrix, "getrow") else matrix[doc_id]
    row = row.tocsr() if hasattr(row, "tocsr") else row

    feature_names = vectorizer.get_feature_names_out()
    nz_idx = row.nonzero()[1]
    weights = np.asarray(row[nz_idx]).ravel()

    order = np.argsort(weights)[::-1][:top_n]
    top_idx = nz_idx[order]
    top_weights = weights[order]

    print(f"\n--- Document #{doc_id} top {top_n} TF-IDF features ---")
    for rank, (idx, w) in enumerate(zip(top_idx, top_weights), start=1):
        print(f"  {rank}. {feature_names[idx]:<20s}  weight = {w:.6f}")

    # Return a structured payload for audit logs
    return [
        {"feature": str(feature_names[idx]), "weight": float(w)}
        for idx, w in zip(top_idx, top_weights)
    ]


# Test on the required doc
_ = explain_document(412, X_tfidf, tfidf_vec, top_n=5)