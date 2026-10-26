"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques

Completed version. Run cell-by-cell (# %% markers) or directly:
    python lab15_starter.py
"""


# ## Setup & Dataset Loading

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)
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

# TODO 1.1: Tokenize, lowercase, keep alphabetic tokens only
tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]

# TODO 1.2: Remove standard English stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]

# TODO 1.3: Porter stemming vs. WordNet lemmatization (pos='v')
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
# TODO 2.1: Initialize both vectorizers
bow_vec = CountVectorizer(stop_words="english", max_features=4000, min_df=5)
tfidf_vec = TfidfVectorizer(stop_words="english", max_features=4000, min_df=5)

# TODO 2.2: Fit and transform the corpus
X_bow = bow_vec.fit_transform(df["text"])
X_tfidf = tfidf_vec.fit_transform(df["text"])

# TODO 2.3: Calculate matrix sparsity (% of zero entries in X_tfidf)
if X_tfidf is not None:
    total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
    sparsity = 100.0 * (1.0 - (X_tfidf.nnz / total_elements))
    print(f"\nTF-IDF Matrix Shape: {X_tfidf.shape} | Sparsity: {sparsity:.2f}%")

    # TODO 2.4: Mean score per feature across documents, plot Top 10 TF-IDF words
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
    # TODO 3.1: Fit KMeans (4 clusters) on X_tfidf
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    # TODO 3.2: Top 8 defining TF-IDF words per cluster centroid + heatmap of top 5
    feature_names = tfidf_vec.get_feature_names_out()
    centers = kmeans.cluster_centers_

    print("\n=== Top 8 words per cluster (baseline) ===")
    for c in range(4):
        top8 = centers[c].argsort()[::-1][:8]
        print(f"Cluster {c}: {', '.join(feature_names[top8])}")

    heat_words = []
    for c in range(4):
        for i in centers[c].argsort()[::-1][:5]:
            if feature_names[i] not in heat_words:
                heat_words.append(feature_names[i])
    heat_idx = [list(feature_names).index(w) for w in heat_words]
    heat_df = pd.DataFrame(
        centers[:, heat_idx].T,
        index=heat_words,
        columns=[f"Cluster {c}" for c in range(4)],
    )

    plt.figure(figsize=(7, 8))
    sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Top 5 Centroid Words per Cluster (TF-IDF weight)")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.show()

    # TODO 3.3: 2-D projection with TruncatedSVD, scatter coloured by cluster
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X_tfidf)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1], c=df["cluster"], cmap="viridis", s=8, alpha=0.6
    )
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.title("2-D Spatial Mapping of Posts (TruncatedSVD on TF-IDF)")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.show()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %%
# Challenge 1 (Noise Filtering):
# TODO 4.1: Cleaner TfidfVectorizer + re-fit KMeans, compare cluster keywords
custom_stop = {"know", "like", "just", "think", "does", "use", "thanks"}
clean_stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop))

tfidf_clean = TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    min_df=5,
    max_features=4000,
    stop_words=clean_stop_words,
)
X_clean = tfidf_clean.fit_transform(df["text"])
kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = kmeans_clean.fit_predict(X_clean)

clean_features = tfidf_clean.get_feature_names_out()
print("\n=== Top 8 words per cluster (cleaned vectorizer) ===")
for c in range(4):
    top8 = kmeans_clean.cluster_centers_[c].argsort()[::-1][:8]
    print(f"Cluster {c}: {', '.join(clean_features[top8])}")

# Why does a filler word still get weight? IDF of a word in 25% of posts:
print(f"\nIDF check: log(1 / 0.25) = {np.log(1 / 0.25):.3f}  (filler word in 25% of posts)")
print(f"IDF check: log(1 / 0.01) = {np.log(1 / 0.01):.3f}  (rare word in 1% of posts)")


# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Crosstab of ground-truth labels vs. clusters
print("\n=== Crosstab: true_label vs. baseline cluster ===")
ct = pd.crosstab(df["true_label"], df["cluster"])
print(ct)

print("\n=== Crosstab: true_label vs. cleaned cluster ===")
print(pd.crosstab(df["true_label"], df["cluster_clean"]))

# Shared technical words between the two hardware categories
elec_mask = (df["true_label"] == "sci.electronics").values
mac_mask = (df["true_label"] == "comp.sys.mac.hardware").values
elec_mean = np.asarray(X_tfidf[elec_mask].mean(axis=0)).ravel()
mac_mean = np.asarray(X_tfidf[mac_mask].mean(axis=0)).ravel()
top_elec = set(tfidf_vec.get_feature_names_out()[elec_mean.argsort()[::-1][:60]])
top_mac = set(tfidf_vec.get_feature_names_out()[mac_mean.argsort()[::-1][:60]])
shared_words = sorted(top_elec & top_mac)
print("\nShared top words (electronics vs. mac hardware):", shared_words)

# Bigrams to separate the two hardware groups
tfidf_bi = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 2), max_features=6000, min_df=5
)
X_bi = tfidf_bi.fit_transform(df["text"])
km_bi = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_bi"] = km_bi.fit_predict(X_bi)
print("\n=== Crosstab: true_label vs. bigram cluster ===")
print(pd.crosstab(df["true_label"], df["cluster_bi"]))
bi_features = tfidf_bi.get_feature_names_out()
bigram_hits = [t for t in bi_features if " " in t]
print("Example bigram features:", bigram_hits[:15])


# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 norms of longest vs. shortest posts in BoW and TF-IDF
text_len = df["text"].str.len().values
nonzero_rows = np.asarray(X_bow.sum(axis=1)).ravel() > 0  # posts with >=1 in-vocab word
longest_i = int(np.argmax(text_len))
shortest_i = int(np.where(nonzero_rows)[0][np.argmin(text_len[nonzero_rows])])


def row_norm(matrix, i):
    return float(np.linalg.norm(matrix[i].toarray().ravel()))


print("\n=== L2 norms (Challenge 3) ===")
print(f"Longest post  (idx {longest_i}, {text_len[longest_i]} chars):")
print(f"   BoW norm   = {row_norm(X_bow, longest_i):.3f}")
print(f"   TF-IDF norm = {row_norm(X_tfidf, longest_i):.3f}")
print(f"Shortest post (idx {shortest_i}, {text_len[shortest_i]} chars):")
print(f"   BoW norm   = {row_norm(X_bow, shortest_i):.3f}")
print(f"   TF-IDF norm = {row_norm(X_tfidf, shortest_i):.3f}")


# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    Return (and print) the top_n highest-weighted non-zero TF-IDF terms and
    their exact weights for a single document `doc_id`.
    """
    row = matrix[doc_id]
    if hasattr(row, "tocoo"):
        row = row.tocoo()
        cols, vals = row.col, row.data
    else:  # dense fallback
        arr = np.asarray(row).ravel()
        cols = np.nonzero(arr)[0]
        vals = arr[cols]

    feature_names = vectorizer.get_feature_names_out()
    order = np.argsort(vals)[::-1][:top_n]
    result = [(str(feature_names[cols[i]]), float(vals[i])) for i in order]

    print(f"\nTop {top_n} TF-IDF terms for document #{doc_id}:")
    for term, weight in result:
        print(f"   {term:<15s} {weight:.4f}")
    return result


# Audit of document #412
if len(df) > 412:
    audit = explain_document(412, X_tfidf, tfidf_vec, top_n=5)
    print(
        f"Doc #412 true label: {df['true_label'].iloc[412]} | "
        f"assigned cluster: {df['cluster'].iloc[412]}"
    )
    print("Doc #412 text preview:", df["text"].iloc[412][:300].replace("\n", " "))
