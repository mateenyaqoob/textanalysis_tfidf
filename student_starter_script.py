"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques

Instructions:
Complete each TODO section below. You can run this file interactively cell-by-cell
in VS Code / PyCharm / Spyder (using the `# %%` markers) or directly via terminal:
    python student_starter_script.py
"""


# %% [markdown]
# ## Setup & Dataset Loading

# %%
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
    plt.show()


# %% [markdown]
# ## Step 3: Automated Topic Discovery & Spatial Mapping (15 mins)

# %%
if X_tfidf is not None:
    # TODO 3.1: Fit KMeans with `n_clusters=4, random_state=42, n_init=10` on `X_tfidf`
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    # TODO 3.2: Print the top 8 defining TF-IDF words per cluster centroid
    # and plot a seaborn heatmap (`cluster_heatmap.png`) of the top 5 words per cluster.

    # TODO 3.3: Project `X_tfidf` to 2 dimensions using `TruncatedSVD(n_components=2, random_state=42)`
    # and save a 2D scatter plot (`spatial_mapping.png`) colored by `df["cluster"]`.
    terms = tfidf_vec.get_feature_names_out()

    def top_cluster_words(model, n):
        return [
            [terms[j] for j in model.cluster_centers_[c].argsort()[::-1][:n]]
            for c in range(model.n_clusters)
        ]

    for c, words in enumerate(top_cluster_words(kmeans, 8)):
        print(f"Cluster {c}: {', '.join(words)}")

    heat_words = []
    for words in top_cluster_words(kmeans, 5):
        heat_words += [w for w in words if w not in heat_words]
    heat_idx = [list(terms).index(w) for w in heat_words]
    heat_df = pd.DataFrame(
        kmeans.cluster_centers_[:, heat_idx],
        index=[f"Cluster {c}" for c in range(kmeans.n_clusters)],
        columns=heat_words,
    )
    plt.figure(figsize=(12, 4))
    sns.heatmap(heat_df, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Top Words per Cluster (Centroid TF-IDF Weight)")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.show()

    coords = TruncatedSVD(n_components=2, random_state=42).fit_transform(X_tfidf)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1], c=df["cluster"], cmap="tab10", s=8, alpha=0.6
    )
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.title("TF-IDF Documents in 2D (TruncatedSVD)")
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
FILLER = ["know", "like", "just", "think", "does", "use", "thanks", "don", "ve", "ll", "really", "good", "got", "want", "need", "way"]
clean_stop_words = list(set(CountVectorizer(stop_words="english").get_stop_words()) | set(FILLER))
clean_vec = TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    min_df=5,
    max_features=4000,
    stop_words=clean_stop_words,
)
X_clean = clean_vec.fit_transform(df["text"])
kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = kmeans_clean.fit_predict(X_clean)
clean_terms = clean_vec.get_feature_names_out()
print("\n--- Cluster keywords after noise filtering ---")
for c in range(4):
    top = kmeans_clean.cluster_centers_[c].argsort()[::-1][:8]
    print(f"Cluster {c}: {', '.join(clean_terms[top])}")

# Why does a filler word still score highly? Inspect the baseline IDF values.
baseline_terms = list(tfidf_vec.get_feature_names_out())
for w in ["know", "like", "think", "just", "does", "thanks", "use"]:
    if w in baseline_terms:
        i = baseline_terms.index(w)
        df_frac = X_tfidf[:, i].nnz / X_tfidf.shape[0]
        print(f"{w:8s} DF={df_frac:6.1%}  idf={tfidf_vec.idf_[i]:.3f}")


# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Run `pd.crosstab(df['true_label'], df['cluster'])` and inspect overlapping categories.
print("\n--- Crosstab: true label vs. cluster (baseline) ---")
print(pd.crosstab(df["true_label"], df["cluster"]))

# Bigrams to separate electronics from Mac hardware
bigram_vec = TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    min_df=5,
    max_features=6000,
    ngram_range=(1, 2),
    stop_words=clean_stop_words,
)
X_bigram = bigram_vec.fit_transform(df["text"])
df["cluster_bigram"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_bigram)
print("\n--- Crosstab with bigrams (1,2) ---")
print(pd.crosstab(df["true_label"], df["cluster_bigram"]))

# Shared technical words in the two hardware categories
hw = df["true_label"].isin(["sci.electronics", "comp.sys.mac.hardware"])
hw_vec = CountVectorizer(token_pattern=r"\b[a-zA-Z]{3,}\b", stop_words=clean_stop_words, min_df=5)
hw_counts = hw_vec.fit_transform(df.loc[hw, "text"])
hw_labels = df.loc[hw, "true_label"].to_numpy()
elec = np.asarray(hw_counts[hw_labels == "sci.electronics"].sum(axis=0)).ravel()
mac = np.asarray(hw_counts[hw_labels == "comp.sys.mac.hardware"].sum(axis=0)).ravel()
shared = np.minimum(elec, mac)
print("\nMost shared words (electronics & mac.hardware):",
      list(hw_vec.get_feature_names_out()[shared.argsort()[::-1][:15]]))


# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 vector norms (`np.linalg.norm`) of the longest and shortest posts in `X_bow` vs `X_tfidf`.
word_counts = df["text"].str.split().str.len()
longest, shortest = int(word_counts.idxmax()), int(word_counts.idxmin())
print(f"\nLongest post: #{longest} ({word_counts[longest]} words) | Shortest post: #{shortest} ({word_counts[shortest]} words)")
for name, M in [("X_bow", X_bow), ("X_tfidf", X_tfidf)]:
    print(f"{name:8s} norm(longest)={np.linalg.norm(M[longest].toarray()):8.3f}  "
          f"norm(shortest)={np.linalg.norm(M[shortest].toarray()):8.3f}")


# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return or print the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` (test with doc_id=412).
    """
    row = matrix[doc_id].tocoo()
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(row.col, row.data), key=lambda t: t[1], reverse=True)[:top_n]
    return [(str(terms[j]), float(w)) for j, w in ranked]


if len(df) > 412:
    print("\n--- explain_document(412) ---")
    print(f"True label: {df['true_label'][412]} | Cluster: {df['cluster'][412]}")
    for term, weight in explain_document(412, X_tfidf, tfidf_vec, top_n=5):
        print(f"{term:15s} {weight:.4f}")
