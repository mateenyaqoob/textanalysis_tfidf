"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques

Instructions:
Complete each TODO section below. You can run this file interactively cell-by-cell
in VS Code / PyCharm / Spyder (using the `# %%` markers) or directly via terminal:
    python lab15_starter.py
"""

import sys
import matplotlib

# Use a non-interactive backend unless running inside Jupyter / an interactive
# session, so the script (and the pytest import) never blocks on plt.show().
SHOW_PLOTS = "ipykernel" in sys.modules or hasattr(sys, "ps1")
if not SHOW_PLOTS:
    matplotlib.use("Agg")


# ## Setup & Dataset Loading

# %%
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
nltk.download("omw-1.4", quiet=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

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

    # Stemming chops suffixes heuristically (can produce non-words such as 'anyon'),
    # while lemmatization maps to valid dictionary forms.
    comparison = pd.DataFrame(
        {"token": filtered_tokens, "porter_stem": stemmed_tokens, "lemma_v": lemmatized_tokens}
    )
    print("\nStemmer vs. Lemmatizer (tokens where outputs differ):")
    print(comparison[comparison["porter_stem"] != comparison["lemma_v"]].head(12).to_string(index=False))


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
    bow_sparsity = 100.0 * (1.0 - (X_bow.nnz / (X_bow.shape[0] * X_bow.shape[1])))
    print(f"BoW Matrix Shape:    {X_bow.shape} | Sparsity: {bow_sparsity:.2f}%")

    # TODO 2.4: Compute mean score per feature across documents and plot Top 10 TF-IDF words
    tfidf_means = np.asarray(X_tfidf.mean(axis=0)).ravel()
    top_idx = tfidf_means.argsort()[-10:]
    top_words = tfidf_vec.get_feature_names_out()[top_idx]
    top_scores = tfidf_means[top_idx]
    print("Top 10 words by average TF-IDF:", ", ".join(top_words[::-1]))

    plt.figure(figsize=(8, 4.5))
    plt.barh(top_words, top_scores, color="teal")
    plt.xlabel("Average TF-IDF Weight")
    plt.title("Top 10 Most Influential Words (Baseline TF-IDF)")
    plt.tight_layout()
    plt.savefig("top_features.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# %% [markdown]
# ## Step 3: Automated Topic Discovery & Spatial Mapping (15 mins)

# %%
if X_tfidf is not None:
    # TODO 3.1: Fit KMeans with `n_clusters=4, random_state=42, n_init=10` on `X_tfidf`
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)
    print("\nCluster sizes:", df["cluster"].value_counts().sort_index().to_dict())

    # TODO 3.2: Print the top 8 defining TF-IDF words per cluster centroid
    # and plot a seaborn heatmap (`cluster_heatmap.png`) of the top 5 words per cluster.
    feature_names = tfidf_vec.get_feature_names_out()
    centroids = kmeans.cluster_centers_
    order_centroids = centroids.argsort(axis=1)[:, ::-1]

    print("\nTop 8 defining words per cluster (baseline TF-IDF):")
    for c in range(kmeans.n_clusters):
        print(f"  Cluster {c}: {', '.join(feature_names[order_centroids[c, :8]])}")

    # Union of each cluster's top 5 words forms the heatmap columns
    heatmap_idx = []
    for c in range(kmeans.n_clusters):
        for i in order_centroids[c, :5]:
            if i not in heatmap_idx:
                heatmap_idx.append(i)
    heatmap_df = pd.DataFrame(
        centroids[:, heatmap_idx],
        index=[f"Cluster {c}" for c in range(kmeans.n_clusters)],
        columns=feature_names[heatmap_idx],
    )

    plt.figure(figsize=(12, 4))
    sns.heatmap(heatmap_df, cmap="YlGnBu", annot=True, fmt=".3f", annot_kws={"size": 7})
    plt.title("Cluster Centroid Weights: Top 5 TF-IDF Words per Cluster")
    plt.xlabel("Term")
    plt.ylabel("K-Means Cluster")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # TODO 3.3: Project `X_tfidf` to 2 dimensions using `TruncatedSVD(n_components=2, random_state=42)`
    # and save a 2D scatter plot (`spatial_mapping.png`) colored by `df["cluster"]`.
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_tfidf)
    print(f"\nTruncatedSVD explained variance (2 components): {svd.explained_variance_ratio_.sum():.4f}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster"], cmap="tab10", s=8, alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.title("2D Spatial Mapping of User Posts (TruncatedSVD on TF-IDF)")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %%
def top_cluster_terms(model, vectorizer, n=8):
    """Top-n centroid terms for every cluster of a fitted KMeans model."""
    names = vectorizer.get_feature_names_out()
    order = model.cluster_centers_.argsort(axis=1)[:, ::-1]
    return {c: list(names[order[c, :n]]) for c in range(model.n_clusters)}


def cluster_purity(labels, clusters):
    """Fraction of documents that belong to their cluster's majority true label."""
    ct = pd.crosstab(labels, clusters)
    return ct.max(axis=0).sum() / ct.values.sum()


# Challenge 1 (Noise Filtering):
# TODO 4.1: Build a cleaner `TfidfVectorizer` using `token_pattern=r'\b[a-zA-Z]{3,}\b'`,
# `max_df=0.5`, and custom conversational stopwords (e.g., 'know', 'like', 'just', 'think', 'does', 'use', 'thanks').
# Re-fit KMeans and compare cluster keywords!
if X_tfidf is not None:
    filler_words = ["know", "like", "just", "think", "does", "use", "thanks"]
    print("\n[Challenge 1] Document frequency and IDF of filler words:")
    for w in filler_words:
        doc_freq = df["text"].str.lower().str.contains(rf"\b{w}\b", regex=True).mean()
        idf_val = tfidf_vec.idf_[tfidf_vec.vocabulary_[w]] if w in tfidf_vec.vocabulary_ else float("nan")
        print(f"  {w:<7} DF={doc_freq:6.2%}  sklearn idf={idf_val:.3f}  plain log(N/DF)={np.log(1 / doc_freq):.3f}")

    conversational_stops = [
        "know", "like", "just", "think", "does", "use", "thanks", "don", "want", "good",
        "need", "make", "used", "really", "way", "people", "time", "say", "said", "going",
        "got", "did", "new", "anybody", "anyone", "doesn", "didn", "isn", "sure", "thing",
        "things", "right", "look", "help", "work", "better", "little", "lot", "problem",
        "edu", "com", "mail", "post", "email", "advance", "appreciated", "question", "info",
        "information", "probably", "years", "using",
    ]
    # A repeated sci.med signature block ("Gordon Banks N3JXP ... geb@cadre.dsl.pitt.edu
    # ... shameful to surrender it too soon") survives remove=('footers',) and forms its own
    # 71-post "topic" in the baseline. It is noise, not vocabulary, so drop it too.
    signature_noise = [
        "geb", "cadre", "dsl", "pitt", "chastity", "shameful", "surrender", "intellect",
        "skepticism", "banks", "gordon", "jxp",
    ]
    custom_stops = list(ENGLISH_STOP_WORDS.union(conversational_stops + signature_noise))
    clean_tfidf_vec = TfidfVectorizer(
        stop_words=custom_stops,
        token_pattern=r"\b[a-zA-Z]{3,}\b",
        max_df=0.5,
        min_df=5,
        max_features=4000,
    )
    X_clean = clean_tfidf_vec.fit_transform(df["text"])
    kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster_clean"] = kmeans_clean.fit_predict(X_clean)

    baseline_terms = top_cluster_terms(kmeans, tfidf_vec)
    clean_terms = top_cluster_terms(kmeans_clean, clean_tfidf_vec)
    print("\n[Challenge 1] Baseline vs. cleaned cluster keywords:")
    for c in range(4):
        print(f"  Baseline cluster {c}: {', '.join(baseline_terms[c])}")
    for c in range(4):
        print(f"  Cleaned  cluster {c}: {', '.join(clean_terms[c])}")
    print(f"  Purity  baseline={cluster_purity(df['true_label'], df['cluster']):.3f} | "
          f"cleaned={cluster_purity(df['true_label'], df['cluster_clean']):.3f}")


# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Run `pd.crosstab(df['true_label'], df['cluster'])` and inspect overlapping categories.
if X_tfidf is not None:
    crosstab = pd.crosstab(df["true_label"], df["cluster"])
    print("\n[Challenge 2] Baseline crosstab (true_label x cluster):")
    print(crosstab)
    print("\n[Challenge 2] Cleaned-vectorizer crosstab:")
    print(pd.crosstab(df["true_label"], df["cluster_clean"]))

    # Map each cluster to its dominant true label, then look at hardware posts that land
    # in the "other" hardware cluster. The cleaned clustering is used here because it is the
    # one with a distinct electronics-dominated cluster.
    cluster_to_label = crosstab.idxmax(axis=0).to_dict()
    clean_cluster_to_label = pd.crosstab(df["true_label"], df["cluster_clean"]).idxmax(axis=0).to_dict()
    df["predicted_label"] = df["cluster_clean"].map(clean_cluster_to_label)
    hw = ["sci.electronics", "comp.sys.mac.hardware"]
    print("\n[Challenge 2] True label vs. dominant label of assigned (cleaned) cluster:")
    print(pd.crosstab(df["true_label"], df["predicted_label"]))

    mis = df[
        df["true_label"].isin(hw)
        & df["predicted_label"].isin(hw)
        & (df["true_label"] != df["predicted_label"])
    ]
    print(f"\n[Challenge 2] Electronics <-> Mac misclustered posts: {len(mis)}")
    if len(mis) > 0:
        clean_names = clean_tfidf_vec.get_feature_names_out()
        mis_means = np.asarray(X_clean[mis.index.to_numpy()].mean(axis=0)).ravel()
        print("  Top TF-IDF terms in misclustered hardware posts:",
              ", ".join(clean_names[mis_means.argsort()[::-1][:15]]))
        for i in mis.index[:3]:
            snippet = " ".join(df.loc[i, "text"].split())[:160]
            print(f"  doc {i} [{df.loc[i, 'true_label']} -> {df.loc[i, 'predicted_label']}]: {snippet}")

    # Terms carrying high average weight in BOTH hardware categories
    elec_mean = np.asarray(X_clean[np.flatnonzero(df["true_label"] == hw[0])].mean(axis=0)).ravel()
    mac_mean = np.asarray(X_clean[np.flatnonzero(df["true_label"] == hw[1])].mean(axis=0)).ravel()
    overlap_score = np.minimum(elec_mean, mac_mean)
    print("  Terms weighted highly in both categories:",
          ", ".join(clean_names[overlap_score.argsort()[::-1][:12]]))

    # Bigrams capture phrases ('power supply', 'mac plus', 'hard drive') that are more
    # category-specific than their individual words.
    bigram_vec = TfidfVectorizer(
        stop_words=custom_stops,
        token_pattern=r"\b[a-zA-Z]{3,}\b",
        max_df=0.5,
        min_df=5,
        max_features=8000,
        ngram_range=(1, 2),
    )
    X_bigram = bigram_vec.fit_transform(df["text"])
    kmeans_bigram = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster_bigram"] = kmeans_bigram.fit_predict(X_bigram)
    print("\n[Challenge 2] Bigram (1,2) crosstab:")
    print(pd.crosstab(df["true_label"], df["cluster_bigram"]))
    print(f"  Purity  baseline={cluster_purity(df['true_label'], df['cluster']):.3f} | "
          f"cleaned={cluster_purity(df['true_label'], df['cluster_clean']):.3f} | "
          f"bigram={cluster_purity(df['true_label'], df['cluster_bigram']):.3f}")
    bigram_names = bigram_vec.get_feature_names_out()
    bigram_order = kmeans_bigram.cluster_centers_.argsort(axis=1)[:, ::-1]
    for c in range(4):
        phrases = [bigram_names[i] for i in bigram_order[c] if " " in bigram_names[i]][:6]
        print(f"  Bigram cluster {c} top phrases: {', '.join(phrases)}")


# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 vector norms (`np.linalg.norm`) of the longest and shortest posts in `X_bow` vs `X_tfidf`.
if X_tfidf is not None:
    doc_lengths = df["text"].str.split().str.len()
    longest_idx = int(doc_lengths.idxmax())
    # Shortest post that keeps at least one vocabulary term (an all-zero row has norm 0
    # in both representations and says nothing about length bias).
    non_empty = np.diff(X_bow.indptr) > 0
    shortest_idx = int(doc_lengths[non_empty].idxmin())

    bow_long = np.linalg.norm(X_bow[longest_idx].toarray())
    bow_short = np.linalg.norm(X_bow[shortest_idx].toarray())
    tfidf_long = np.linalg.norm(X_tfidf[longest_idx].toarray())
    tfidf_short = np.linalg.norm(X_tfidf[shortest_idx].toarray())

    print(f"\n[Challenge 3] Longest post: doc {longest_idx} ({doc_lengths[longest_idx]} words) | "
          f"Shortest post: doc {shortest_idx} ({doc_lengths[shortest_idx]} words)")
    print(f"  BoW    L2 norm  longest={bow_long:.3f}  shortest={bow_short:.3f}  "
          f"ratio={bow_long / max(bow_short, 1e-12):.1f}x")
    print(f"  TF-IDF L2 norm  longest={tfidf_long:.3f}  shortest={tfidf_short:.3f}")
    print(f"  Short post text: {' '.join(df.loc[shortest_idx, 'text'].split())[:120]}")

    bow_norms = np.sqrt(np.asarray(X_bow.multiply(X_bow).sum(axis=1)).ravel())
    length_bins = pd.qcut(doc_lengths, 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"])
    print("  Mean BoW L2 norm by post-length quartile:",
          pd.Series(bow_norms).groupby(length_bins, observed=True).mean().round(2).to_dict())

    # Consequence: KMeans directly on raw counts
    kmeans_bow = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster_bow"] = kmeans_bow.fit_predict(X_bow)
    print("  KMeans on raw BoW counts -> cluster sizes:",
          df["cluster_bow"].value_counts().sort_index().to_dict())
    print("  Median words per post in each BoW cluster:",
          doc_lengths.groupby(df["cluster_bow"]).median().to_dict())
    print("  KMeans on TF-IDF         -> cluster sizes:",
          df["cluster"].value_counts().sort_index().to_dict())
    print(f"  Purity  BoW={cluster_purity(df['true_label'], df['cluster_bow']):.3f} | "
          f"TF-IDF={cluster_purity(df['true_label'], df['cluster']):.3f}")


# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return or print the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` (test with doc_id=412).
    """
    row = matrix[doc_id].tocsr()
    names = vectorizer.get_feature_names_out()
    order = np.argsort(row.data)[::-1][:top_n]
    return [(str(names[row.indices[i]]), float(row.data[i])) for i in order]


if X_tfidf is not None:
    audit_id = 412
    audit_cluster = int(df.loc[audit_id, "cluster"])
    print(f"\n[Challenge 4] Audit for Document #{audit_id}")
    print(f"  True label: {df.loc[audit_id, 'true_label']} | Assigned cluster: {audit_cluster} "
          f"(dominant label: {cluster_to_label[audit_cluster]})")
    print(f"  Text: {' '.join(df.loc[audit_id, 'text'].split())[:300]}")
    for term, weight in explain_document(audit_id, X_tfidf, tfidf_vec, top_n=5):
        print(f"    {term:<15} {weight:.4f}")

    # Distance to every centroid shows why the assigned cluster won
    dists = kmeans.transform(X_tfidf[audit_id])[0]
    print("  Euclidean distance to each centroid:", {c: round(float(d), 4) for c, d in enumerate(dists)})
