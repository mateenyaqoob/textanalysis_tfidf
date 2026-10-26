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

# TODO 1.1: Tokenize, lowercase, keep only alphabetic tokens
tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]

# TODO 1.2: Remove standard English stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]

# TODO 1.3: Stemming vs. lemmatization (pos='v')
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
# TODO 2.1: Initialize vectorizers
bow_vec = CountVectorizer(stop_words="english", max_features=4000, min_df=5)
tfidf_vec = TfidfVectorizer(stop_words="english", max_features=4000, min_df=5)

# TODO 2.2: Fit and transform
X_bow = bow_vec.fit_transform(df["text"])
X_tfidf = tfidf_vec.fit_transform(df["text"])

# TODO 2.3: Sparsity (% of zero entries in X_tfidf)
if X_tfidf is not None:
    total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
    sparsity = 100.0 * (1.0 - (X_tfidf.nnz / total_elements))
    print(f"\nTF-IDF Matrix Shape: {X_tfidf.shape} | Sparsity: {sparsity:.2f}%")

    # TODO 2.4: Top 10 words by average TF-IDF weight
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
def top_terms_per_cluster(model, vectorizer, n=8):
    """Return {cluster_id: [top n terms]} from the KMeans centroids."""
    terms = vectorizer.get_feature_names_out()
    order = model.cluster_centers_.argsort()[:, ::-1]
    return {c: [terms[i] for i in order[c, :n]] for c in range(model.n_clusters)}


if X_tfidf is not None:
    # TODO 3.1: KMeans on the TF-IDF matrix
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    # TODO 3.2: Top 8 words per cluster + heatmap of top 5 words per cluster
    print("\n--- Top 8 Words per Cluster (Baseline) ---")
    baseline_top = top_terms_per_cluster(kmeans, tfidf_vec, n=8)
    for c, words in baseline_top.items():
        print(f"Cluster {c}: {', '.join(words)}")

    feature_names = tfidf_vec.get_feature_names_out()
    heat_words = []
    for c in range(kmeans.n_clusters):
        for w in baseline_top[c][:5]:
            if w not in heat_words:
                heat_words.append(w)
    heat_idx = [list(feature_names).index(w) for w in heat_words]
    heat_df = pd.DataFrame(
        kmeans.cluster_centers_[:, heat_idx],
        index=[f"Cluster {c}" for c in range(kmeans.n_clusters)],
        columns=heat_words,
    )

    plt.figure(figsize=(12, 4))
    sns.heatmap(heat_df, cmap="YlGnBu", annot=True, fmt=".3f", cbar_kws={"label": "Centroid weight"})
    plt.title("Cluster-by-Word Importance (Top 5 Words per Cluster)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.show()

    # TODO 3.3: 2D projection with TruncatedSVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_tfidf)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster"], cmap="viridis", s=8, alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.title("2D Spatial Mapping of Posts (TruncatedSVD on TF-IDF)")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.show()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %%
# Challenge 1 (Noise Filtering):
# TODO 4.1: Cleaner vectorizer: 3+ letter words only, drop terms in >50% of posts,
# and add conversational filler to the stopword list.
custom_stops = ["know", "like", "just", "think", "does", "use", "thanks",
                "don", "want", "good", "need", "say", "way", "make", "really"]
clean_stop_words = list(ENGLISH_STOP_WORDS.union(custom_stops))

tfidf_clean = TfidfVectorizer(
    stop_words=clean_stop_words,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    min_df=5,
    max_features=4000,
)
X_tfidf_clean = tfidf_clean.fit_transform(df["text"])

kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = kmeans_clean.fit_predict(X_tfidf_clean)

print("\n--- Top 8 Words per Cluster (Cleaned Vectorizer) ---")
for c, words in top_terms_per_cluster(kmeans_clean, tfidf_clean, n=8).items():
    print(f"Cluster {c}: {', '.join(words)}")


# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Compare clusters with the hidden true labels
print("\n--- Crosstab: True Label vs. Baseline Cluster ---")
print(pd.crosstab(df["true_label"], df["cluster"]))

print("\n--- Crosstab: True Label vs. Cleaned Cluster ---")
print(pd.crosstab(df["true_label"], df["cluster_clean"]))

# Bigram experiment: phrases like 'hard drive', 'power supply' can separate the two hardware groups
tfidf_bigram = TfidfVectorizer(
    stop_words=clean_stop_words,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    ngram_range=(1, 2),
    max_df=0.5,
    min_df=5,
    max_features=4000,
)
X_tfidf_bigram = tfidf_bigram.fit_transform(df["text"])
kmeans_bigram = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_bigram"] = kmeans_bigram.fit_predict(X_tfidf_bigram)

print("\n--- Crosstab: True Label vs. Bigram Cluster ---")
print(pd.crosstab(df["true_label"], df["cluster_bigram"]))

print("\n--- Top 8 Terms per Cluster (Unigrams + Bigrams) ---")
for c, words in top_terms_per_cluster(kmeans_bigram, tfidf_bigram, n=8).items():
    print(f"Cluster {c}: {', '.join(words)}")


# Challenge 3 (Rant Length Bias):
# TODO 4.3: L2 norms of the longest vs. shortest post in BoW vs. TF-IDF
doc_lengths = df["text"].str.split().str.len()
longest_id = int(doc_lengths.idxmax())
shortest_id = int(doc_lengths.idxmin())

bow_long = np.linalg.norm(X_bow[longest_id].toarray())
bow_short = np.linalg.norm(X_bow[shortest_id].toarray())
tfidf_long = np.linalg.norm(X_tfidf[longest_id].toarray())
tfidf_short = np.linalg.norm(X_tfidf[shortest_id].toarray())

print("\n--- L2 Norm Comparison ---")
print(f"Longest post  (#{longest_id}, {doc_lengths[longest_id]} words): "
      f"BoW = {bow_long:.2f} | TF-IDF = {tfidf_long:.2f}")
print(f"Shortest post (#{shortest_id}, {doc_lengths[shortest_id]} words): "
      f"BoW = {bow_short:.2f} | TF-IDF = {tfidf_short:.2f}")


# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id`, as a list of (term, weight) tuples.
    """
    row = matrix[doc_id].toarray().ravel()
    terms = vectorizer.get_feature_names_out()
    nonzero_idx = np.flatnonzero(row)
    ranked = nonzero_idx[np.argsort(row[nonzero_idx])[::-1]][:top_n]
    return [(str(terms[i]), float(row[i])) for i in ranked]


AUDIT_DOC = 412
print(f"\n--- Explainability Audit: Document #{AUDIT_DOC} ---")
print(f"True label: {df.loc[AUDIT_DOC, 'true_label']} | Assigned cluster: {df.loc[AUDIT_DOC, 'cluster']}")
for term, weight in explain_document(AUDIT_DOC, X_tfidf, tfidf_vec, top_n=5):
    print(f"   {term:<15} {weight:.4f}")
