"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques

Instructions:
Complete each TODO section below. You can run this file interactively cell-by-cell
in VS Code / PyCharm / Spyder (using the `# %%` markers) or directly via terminal:
    python lab15_starter.py
"""

# ## Setup & Dataset Loading

# %%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
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


# %% [markdown]
# ## Step 3: Automated Topic Discovery & Spatial Mapping (15 mins)

# %%
if X_tfidf is not None:
    # TODO 3.1: Fit KMeans with `n_clusters=4, random_state=42, n_init=10` on `X_tfidf`
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    # TODO 3.2: Print the top 8 defining TF-IDF words per cluster centroid
    # and plot a seaborn heatmap (`cluster_heatmap.png`) of the top 5 words per cluster.
    terms = tfidf_vec.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    print("\n--- Top 8 Words per Cluster Centroid ---")
    top_5_words_set = set()
    for i in range(4):
        top_8 = [terms[ind] for ind in order_centroids[i, :8]]
        print(f"Cluster {i}: {', '.join(top_8)}")
        for ind in order_centroids[i, :5]:
            top_5_words_set.add(terms[ind])

    # Heatmap of top 5 words per cluster
    heatmap_words = sorted(list(top_5_words_set))
    heatmap_indices = [tfidf_vec.vocabulary_[w] for w in heatmap_words]
    heatmap_data = kmeans.cluster_centers_[:, heatmap_indices]
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=[f"Cluster {i}" for i in range(4)],
        columns=heatmap_words,
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Centroid Weight'})
    plt.title("Cluster-by-Word Importance Heatmap (Top 5 Words per Cluster)")
    plt.xlabel("Defining Vocabulary")
    plt.ylabel("Topic Clusters")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.close()

    # TODO 3.3: Project `X_tfidf` to 2 dimensions using `TruncatedSVD(n_components=2, random_state=42)`
    # and save a 2D scatter plot (`spatial_mapping.png`) colored by `df["cluster"]`.
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    df["svd_x"] = X_svd[:, 0]
    df["svd_y"] = X_svd[:, 1]

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        df["svd_x"],
        df["svd_y"],
        c=df["cluster"],
        cmap="tab10",
        alpha=0.6,
        s=18,
        edgecolors="none",
    )
    plt.colorbar(scatter, label="KMeans Cluster")
    plt.title("2D Spatial Document Clusters (TruncatedSVD)")
    plt.xlabel("Latent Semantic Component 1")
    plt.ylabel("Latent Semantic Component 2")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.close()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %%
# Challenge 1 (Noise Filtering):
# TODO 4.1: Build a cleaner `TfidfVectorizer` using `token_pattern=r'\b[a-zA-Z]{3,}\b'`,
# `max_df=0.5`, and custom conversational stopwords (e.g., 'know', 'like', 'just', 'think', 'does', 'use', 'thanks').
# Re-fit KMeans and compare cluster keywords!
custom_fillers = ["know", "like", "just", "think", "does", "use", "thanks", "good", "time", "people", "way"]
custom_stopwords = list(set(stopwords.words("english")).union(set(custom_fillers)))

clean_tfidf_vec = TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    min_df=5,
    max_features=4000,
    stop_words=custom_stopwords,
)
X_tfidf_clean = clean_tfidf_vec.fit_transform(df["text"])
kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = kmeans_clean.fit_predict(X_tfidf_clean)

clean_terms = clean_tfidf_vec.get_feature_names_out()
clean_order_centroids = kmeans_clean.cluster_centers_.argsort()[:, ::-1]

print("\n--- Challenge 1: Cleaned Clusters (Top 6 Words) ---")
for i in range(4):
    top_clean = [clean_terms[ind] for ind in clean_order_centroids[i, :6]]
    print(f"Clean Cluster {i}: {', '.join(top_clean)}")


# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Run `pd.crosstab(df['true_label'], df['cluster'])` and inspect overlapping categories.
print("\n--- Challenge 2: Crosstab (True Labels vs Baseline Clusters) ---")
crosstab_res = pd.crosstab(df["true_label"], df["cluster"])
print(crosstab_res)


# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 vector norms (`np.linalg.norm`) of the longest and shortest posts in `X_bow` vs `X_tfidf`.
doc_lengths = df["text"].str.len()
longest_idx = doc_lengths.idxmax()
shortest_idx = doc_lengths.idxmin()

bow_longest_norm = np.linalg.norm(X_bow[longest_idx].toarray())
bow_shortest_norm = np.linalg.norm(X_bow[shortest_idx].toarray())
tfidf_longest_norm = np.linalg.norm(X_tfidf[longest_idx].toarray())
tfidf_shortest_norm = np.linalg.norm(X_tfidf[shortest_idx].toarray())

print("\n--- Challenge 3: Norm Comparison ---")
print(f"Longest Post length (chars): {doc_lengths[longest_idx]}")
print(f"Shortest Post length (chars): {doc_lengths[shortest_idx]}")
print(f"BoW L2 Norm: Longest = {bow_longest_norm:.2f}, Shortest = {bow_shortest_norm:.2f}")
print(f"TF-IDF L2 Norm: Longest = {tfidf_longest_norm:.2f}, Shortest = {tfidf_shortest_norm:.2f}")


# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix=None, vectorizer=None, top_n=5):
    """
    TODO 4.4: Return or print the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` (test with doc_id=412).
    """
    if matrix is None:
        matrix = X_tfidf
    if vectorizer is None:
        vectorizer = tfidf_vec

    feature_names = vectorizer.get_feature_names_out()
    row = matrix[doc_id].toarray().ravel()
    nonzero_indices = row.nonzero()[0]

    # Sort nonzero features by weight descending
    sorted_nonzero = nonzero_indices[np.argsort(row[nonzero_indices])[::-1]]
    top_indices = sorted_nonzero[:top_n]

    results = []
    for idx in top_indices:
        results.append((feature_names[idx], float(row[idx])))

    return results


if __name__ == "__main__" or True:
    explanation_412 = explain_document(412, X_tfidf, tfidf_vec, top_n=5)
    print(f"\n--- Challenge 4: Top 5 Features for Document #412 (Cluster: {df['cluster'].iloc[412]}, Label: {df['true_label'].iloc[412]}) ---")
    for term, weight in explanation_412:
        print(f"  {term:<15}: {weight:.4f}")
