"""
Lab 15: Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques

Instructions:
Complete each TODO section below. You can run this file interactively cell-by-cell
in VS Code / PyCharm / Spyder (using the `# %%` markers) or directly via terminal:
    python lab15_starter.py
"""

# %% [markdown]
# ## Setup & Dataset Loading

# %%
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
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
bow_vec = CountVectorizer(stop_words="english", max_features=4000, min_df=5)
tfidf_vec = TfidfVectorizer(stop_words="english", max_features=4000, min_df=5)

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
    plt.close()
    print("Saved top_features.png")


# %% [markdown]
# ## Step 3: Automated Topic Discovery & Spatial Mapping (15 mins)

# %%
if X_tfidf is not None:
    # TODO 3.1: Fit KMeans with `n_clusters=4, random_state=42, n_init=10` on `X_tfidf`
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    # TODO 3.2: Print the top 8 defining TF-IDF words per cluster centroid
    # and plot a seaborn heatmap (`cluster_heatmap.png`) of the top 5 words per cluster.
    feature_names = tfidf_vec.get_feature_names_out()
    centroids = kmeans.cluster_centers_

    print("\n--- Top 8 TF-IDF words per cluster (baseline) ---")
    for i, centroid in enumerate(centroids):
        top8 = centroid.argsort()[-8:][::-1]
        print(f"Cluster {i}: {', '.join(feature_names[top8])}")

    top_n_heat = 5
    heat_indices = []
    for centroid in centroids:
        heat_indices.extend(centroid.argsort()[-top_n_heat:][::-1].tolist())
    heat_indices = list(dict.fromkeys(heat_indices))

    heat_df = pd.DataFrame(
        centroids[:, heat_indices],
        index=[f"Cluster {i}" for i in range(kmeans.n_clusters)],
        columns=feature_names[heat_indices],
    )

    plt.figure(figsize=(12, 4.5))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlOrRd")
    plt.title("Cluster-by-Word Importance (Top 5 words per cluster)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.close()
    print("Saved cluster_heatmap.png")

    # TODO 3.3: Project `X_tfidf` to 2 dimensions using `TruncatedSVD(n_components=2, random_state=42)`
    # and save a 2D scatter plot (`spatial_mapping.png`) colored by `df["cluster"]`.
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X_tfidf)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=df["cluster"],
        cmap="tab10",
        alpha=0.55,
        s=18,
    )
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.title("Document Clusters in 2D Space (TruncatedSVD)")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.close()
    print("Saved spatial_mapping.png")


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %%
# Challenge 1 (Noise Filtering):
# TODO 4.1: Build a cleaner `TfidfVectorizer` using `token_pattern=r'\b[a-zA-Z]{3,}\b'`,
# `max_df=0.5`, and custom conversational stopwords (e.g., 'know', 'like', 'just', 'think', 'does', 'use', 'thanks').
# Re-fit KMeans and compare cluster keywords!
CONVERSATIONAL_FILLERS = [
    "know",
    "like",
    "just",
    "think",
    "does",
    "use",
    "thanks",
]
custom_stopwords = list(set(ENGLISH_STOP_WORDS).union(CONVERSATIONAL_FILLERS))

tfidf_clean = TfidfVectorizer(
    stop_words=custom_stopwords,
    max_features=4000,
    min_df=5,
    max_df=0.5,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
)
X_tfidf_clean = tfidf_clean.fit_transform(df["text"])
kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = kmeans_clean.fit_predict(X_tfidf_clean)

clean_names = tfidf_clean.get_feature_names_out()
print("\n--- Challenge 1: Top 8 words after noise filtering ---")
for i, centroid in enumerate(kmeans_clean.cluster_centers_):
    top8 = centroid.argsort()[-8:][::-1]
    print(f"Clean Cluster {i}: {', '.join(clean_names[top8])}")


# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Run `pd.crosstab(df['true_label'], df['cluster'])` and inspect overlapping categories.
print("\n--- Challenge 2: Cluster vs true-label crosstab (baseline) ---")
crosstab = pd.crosstab(df["true_label"], df["cluster"])
print(crosstab)

print("\n--- Challenge 2: Cluster vs true-label crosstab (cleaned) ---")
crosstab_clean = pd.crosstab(df["true_label"], df["cluster_clean"])
print(crosstab_clean)

# Shared vocabulary pulling electronics and Mac hardware together
overlap_mask = df["true_label"].isin(["sci.electronics", "comp.sys.mac.hardware"])
overlap_mean = np.asarray(X_tfidf[overlap_mask.values].mean(axis=0)).ravel()
overlap_top_idx = overlap_mean.argsort()[-15:][::-1]
print("\nShared high-weight words in electronics + Mac hardware posts:")
print(", ".join(feature_names[overlap_top_idx]))

# Bigrams to separate general electronics from Mac hardware
tfidf_ngram = TfidfVectorizer(
    stop_words=custom_stopwords,
    max_features=4000,
    min_df=5,
    max_df=0.5,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    ngram_range=(1, 2),
)
X_tfidf_ngram = tfidf_ngram.fit_transform(df["text"])
kmeans_ngram = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_ngram"] = kmeans_ngram.fit_predict(X_tfidf_ngram)
ngram_names = tfidf_ngram.get_feature_names_out()

print("\n--- Challenge 2: Top 8 words/bigrams per cluster ---")
for i, centroid in enumerate(kmeans_ngram.cluster_centers_):
    top8 = centroid.argsort()[-8:][::-1]
    print(f"Bigram Cluster {i}: {', '.join(ngram_names[top8])}")

print("\n--- Challenge 2: Crosstab with bigrams ---")
print(pd.crosstab(df["true_label"], df["cluster_ngram"]))


# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 vector norms (`np.linalg.norm`) of the longest and shortest posts in `X_bow` vs `X_tfidf`.
text_lengths = df["text"].str.len()
longest_idx = int(text_lengths.idxmax())
shortest_idx = int(text_lengths.idxmin())

bow_long_norm = float(np.linalg.norm(X_bow[longest_idx].toarray()))
bow_short_norm = float(np.linalg.norm(X_bow[shortest_idx].toarray()))
tfidf_long_norm = float(np.linalg.norm(X_tfidf[longest_idx].toarray()))
tfidf_short_norm = float(np.linalg.norm(X_tfidf[shortest_idx].toarray()))

print("\n--- Challenge 3: Document-length bias (L2 norms) ---")
print(f"Longest post id={longest_idx}, chars={text_lengths.loc[longest_idx]}")
print(f"Shortest post id={shortest_idx}, chars={text_lengths.loc[shortest_idx]}")
print(f"BoW    ||longest||_2={bow_long_norm:.4f}  ||shortest||_2={bow_short_norm:.4f}")
print(f"TF-IDF ||longest||_2={tfidf_long_norm:.4f}  ||shortest||_2={tfidf_short_norm:.4f}")


# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return or print the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` (test with doc_id=412).
    """
    names = vectorizer.get_feature_names_out()
    weights = np.asarray(matrix[doc_id].todense()).ravel()
    nonzero = np.where(weights > 0)[0]
    if nonzero.size == 0:
        explanation = []
    else:
        ranked = nonzero[np.argsort(weights[nonzero])[::-1][:top_n]]
        explanation = [(str(names[i]), float(weights[i])) for i in ranked]

    print(f"\nExplain Document #{doc_id} (top {top_n} TF-IDF features):")
    for term, weight in explanation:
        print(f"  {term:20s}  {weight:.6f}")
    return explanation


explain_document(412, X_tfidf, tfidf_vec, top_n=5)
