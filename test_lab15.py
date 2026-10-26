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
matplotlib.use("Agg")
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
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    print("\n--- Top 8 Words per Cluster Centroid ---")
    top_words_heatmap = []
    for cluster_id in range(4):
        top_8_words = [feature_names[ind] for ind in order_centroids[cluster_id, :8]]
        print(f"Cluster {cluster_id}: {', '.join(top_8_words)}")
        top_5_words = [feature_names[ind] for ind in order_centroids[cluster_id, :5]]
        top_words_heatmap.extend(top_5_words)

    # Unique top words preserving order for heatmap
    unique_heatmap_words = list(dict.fromkeys(top_words_heatmap))
    word_indices = [tfidf_vec.vocabulary_[w] for w in unique_heatmap_words]
    heatmap_matrix = kmeans.cluster_centers_[:, word_indices]

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=unique_heatmap_words,
        yticklabels=[f"Cluster {i}" for i in range(4)],
    )
    plt.title("Cluster Centroid Importance Heatmap (Top Words)")
    plt.xlabel("Terms")
    plt.ylabel("Clusters")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.close()
    print("Saved cluster_heatmap.png")

    # TODO 3.3: Project `X_tfidf` to 2 dimensions using `TruncatedSVD(n_components=2, random_state=42)`
    # and save a 2D scatter plot (`spatial_mapping.png`) colored by `df["cluster"]`.
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords_2d = svd.fit_transform(X_tfidf)
    df["svd_dim1"] = coords_2d[:, 0]
    df["svd_dim2"] = coords_2d[:, 1]

    plt.figure(figsize=(9, 6))
    scatter = sns.scatterplot(
        data=df,
        x="svd_dim1",
        y="svd_dim2",
        hue="cluster",
        palette="tab10",
        alpha=0.6,
        s=30,
    )
    plt.title("TruncatedSVD 2D Spatial Mapping of Document Clusters")
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.legend(title="Cluster")
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
custom_stopwords = list(
    set(stopwords.words("english")).union(
        {
            "know",
            "like",
            "just",
            "think",
            "does",
            "use",
            "thanks",
            "good",
            "time",
            "way",
            "want",
            "really",
            "say",
            "make",
            "people",
            "work",
            "new",
        }
    )
)

clean_tfidf_vec = TfidfVectorizer(
    stop_words=custom_stopwords,
    max_features=4000,
    min_df=5,
    max_df=0.5,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
)
X_clean_tfidf = clean_tfidf_vec.fit_transform(df["text"])
clean_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clean_clusters = clean_kmeans.fit_predict(X_clean_tfidf)
clean_features = clean_tfidf_vec.get_feature_names_out()
clean_order_centroids = clean_kmeans.cluster_centers_.argsort()[:, ::-1]

print("\n--- Challenge 1: Cleaned TF-IDF Top 8 Words per Cluster ---")
for c in range(4):
    c_words = [clean_features[ind] for ind in clean_order_centroids[c, :8]]
    print(f"Clean Cluster {c}: {', '.join(c_words)}")


# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Run `pd.crosstab(df['true_label'], df['cluster'])` and inspect overlapping categories.
print("\n--- Challenge 2: Confusion Cross-Tabulation (True Labels vs Clusters) ---")
crosstab_res = pd.crosstab(df["true_label"], df["cluster"])
print(crosstab_res)


# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 vector norms (`np.linalg.norm`) of the longest and shortest posts in `X_bow` vs `X_tfidf`.
doc_lengths = df["text"].str.split().str.len()
longest_doc_idx = doc_lengths.idxmax()
shortest_doc_idx = doc_lengths.idxmin()

bow_norm_longest = np.linalg.norm(X_bow[longest_doc_idx].toarray())
bow_norm_shortest = np.linalg.norm(X_bow[shortest_doc_idx].toarray())

tfidf_norm_longest = np.linalg.norm(X_tfidf[longest_doc_idx].toarray())
tfidf_norm_shortest = np.linalg.norm(X_tfidf[shortest_doc_idx].toarray())

print("\n--- Challenge 3: Vector L2 Norms ---")
print(f"Longest doc ({doc_lengths[longest_doc_idx]} words) - BoW L2 Norm: {bow_norm_longest:.4f}, TF-IDF L2 Norm: {tfidf_norm_longest:.4f}")
print(f"Shortest doc ({doc_lengths[shortest_doc_idx]} words) - BoW L2 Norm: {bow_norm_shortest:.4f}, TF-IDF L2 Norm: {tfidf_norm_shortest:.4f}")


# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id=412, matrix=None, vectorizer=None, top_n=5):
    """
    TODO 4.4: Return or print the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` (test with doc_id=412).
    """
    if matrix is None:
        matrix = X_tfidf
    if vectorizer is None:
        vectorizer = tfidf_vec

    feature_names = vectorizer.get_feature_names_out()
    row = matrix[doc_id]
    non_zero_indices = row.indices
    non_zero_values = row.data

    if len(non_zero_indices) == 0:
        return []

    sorted_order = np.argsort(non_zero_values)[::-1][:top_n]
    explanation = [
        (feature_names[non_zero_indices[i]], float(non_zero_values[i]))
        for i in sorted_order
    ]
    return explanation


if __name__ == "__main__" or True:
    explanation_412 = explain_document(doc_id=412, matrix=X_tfidf, vectorizer=tfidf_vec, top_n=5)
    print("\n--- Challenge 4: Top 5 Features for Document #412 ---")
    print(f"Document #412 True Label: {df['true_label'].iloc[412]} | Assigned Cluster: {df['cluster'].iloc[412]}")
    for term, weight in explanation_412:
        print(f"  {term:<15}: {weight:.4f}")
