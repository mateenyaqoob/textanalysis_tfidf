"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques

Run cell-by-cell in VS Code (Jupyter extension, `# %%` markers) or from the terminal:
    python lab15_starter.py
"""


# %% [markdown]
# ## Setup & Dataset Loading

# %%
import matplotlib

# Use a non-interactive backend when there is no display (e.g. pytest / GitHub Actions)
import os
if os.environ.get("DISPLAY") is None and os.name != "nt" and "VSCODE_PID" not in os.environ:
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
# ## Step 1: Text Cleaning, Tokenization & Normalization

# %%
sample_post = df["text"].iloc[0][:400]
print("--- Raw Sample Post ---")
print(sample_post)
print("-" * 40)

# TODO 1.1: tokenize, lowercase, keep alphabetic tokens only
tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]

# TODO 1.2: remove English stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]

# TODO 1.3: stemming vs. lemmatization (verbs)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(w, pos="v") for w in filtered_tokens]

print("Filtered Tokens:  ", filtered_tokens[:12])
print("Stemmed Roots:    ", stemmed_tokens[:12])
print("Lemmatized Words: ", lemmatized_tokens[:12])


# %% [markdown]
# ## Step 2: Bag of Words vs. TF-IDF Feature Extraction

# %%
# TODO 2.1
bow_vec = CountVectorizer(stop_words="english", max_features=4000, min_df=5)
tfidf_vec = TfidfVectorizer(stop_words="english", max_features=4000, min_df=5)

# TODO 2.2
X_bow = bow_vec.fit_transform(df["text"])
X_tfidf = tfidf_vec.fit_transform(df["text"])

# TODO 2.3: sparsity (% zero entries)
total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
sparsity = 100.0 * (1.0 - (X_tfidf.nnz / total_elements))
print(f"\nTF-IDF Matrix Shape: {X_tfidf.shape} | Sparsity: {sparsity:.2f}%")

# TODO 2.4: top 10 words by mean TF-IDF
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
plt.close()


# %% [markdown]
# ## Step 3: Automated Topic Discovery & Spatial Mapping

# %%
def top_terms_per_cluster(model, vectorizer, n=8):
    """Return {cluster_id: [top n terms]} from a fitted KMeans model's centroids."""
    terms = vectorizer.get_feature_names_out()
    order = model.cluster_centers_.argsort()[:, ::-1]
    return {c: [terms[i] for i in order[c, :n]] for c in range(model.n_clusters)}


# TODO 3.1
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_tfidf)

# TODO 3.2: top 8 words per cluster + heatmap of top 5
feature_names = tfidf_vec.get_feature_names_out()
print("\n--- Baseline cluster keywords ---")
for c, words in top_terms_per_cluster(kmeans, tfidf_vec, n=8).items():
    print(f"Cluster {c} ({(df['cluster'] == c).sum()} posts): {', '.join(words)}")

order = kmeans.cluster_centers_.argsort()[:, ::-1]
heat_idx = list(dict.fromkeys(order[:, :5].ravel()))  # unique top-5 terms, all clusters
heat_df = pd.DataFrame(
    kmeans.cluster_centers_[:, heat_idx],
    index=[f"Cluster {c}" for c in range(4)],
    columns=feature_names[heat_idx],
)
plt.figure(figsize=(12, 3.8))
sns.heatmap(heat_df, cmap="YlGnBu", annot=True, fmt=".3f", cbar_kws={"label": "Centroid weight"})
plt.title("Top 5 TF-IDF Words per Cluster Centroid")
plt.tight_layout()
plt.savefig("cluster_heatmap.png", dpi=150)
plt.show()
plt.close()

# TODO 3.3: 2D projection with TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X_tfidf)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=df["cluster"], palette="tab10", s=12, alpha=0.6)
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.title("2D Spatial Mapping of Posts (TruncatedSVD on TF-IDF)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("spatial_mapping.png", dpi=150)
plt.show()
plt.close()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics

# %%
# Challenge 1 (Noise Filtering)
# TODO 4.1: cleaner vectorizer: letters only (3+ chars), drop terms in >50% of posts,
# plus conversational filler stopwords.
CONVERSATIONAL_STOPWORDS = [
    "know", "like", "just", "think", "does", "use", "thanks", "used", "using",
    "want", "need", "good", "make", "way", "really", "say", "said", "going",
    "anybody", "anyone", "people", "thing", "things", "time", "did", "got",
    "don", "doesn", "didn", "isn", "ve", "ll", "re", "new", "help", "appreciated",
]
clean_stop_words = list(ENGLISH_STOP_WORDS.union(CONVERSATIONAL_STOPWORDS))

tfidf_clean_vec = TfidfVectorizer(
    stop_words=clean_stop_words,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    min_df=5,
    max_features=4000,
)
X_tfidf_clean = tfidf_clean_vec.fit_transform(df["text"])
kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = kmeans_clean.fit_predict(X_tfidf_clean)

print("\n--- Cleaned cluster keywords (Challenge 1) ---")
for c, words in top_terms_per_cluster(kmeans_clean, tfidf_clean_vec, n=8).items():
    print(f"Cluster {c} ({(df['cluster_clean'] == c).sum()} posts): {', '.join(words)}")

# Share of posts containing "know" (to reason about its IDF)
know_df = (X_bow[:, bow_vec.vocabulary_["know"]] > 0).sum() / X_bow.shape[0] if "know" in bow_vec.vocabulary_ else 0
print(f"\n'know' appears in {know_df:.1%} of posts -> smoothed idf = "
      f"{tfidf_vec.idf_[tfidf_vec.vocabulary_['know']]:.3f}" if "know" in tfidf_vec.vocabulary_ else "")

# %%
# Challenge 2 (Hardware Overlap Diagnosis)
# TODO 4.2
print("\n--- Crosstab: true label vs. baseline cluster ---")
ct = pd.crosstab(df["true_label"], df["cluster"])
print(ct)
print("\n--- Crosstab: true label vs. cleaned cluster ---")
print(pd.crosstab(df["true_label"], df["cluster_clean"]))

# Shared vocabulary between electronics and mac hardware (highest mean TF-IDF in both)
clean_terms = tfidf_clean_vec.get_feature_names_out()
elec_mask = (df["true_label"] == "sci.electronics").values
mac_mask = (df["true_label"] == "comp.sys.mac.hardware").values
elec_mean = np.asarray(X_tfidf_clean[elec_mask].mean(axis=0)).ravel()
mac_mean = np.asarray(X_tfidf_clean[mac_mask].mean(axis=0)).ravel()
shared_score = np.minimum(elec_mean, mac_mean)
shared_top = clean_terms[shared_score.argsort()[::-1][:12]]
print("\nShared electronics/Mac vocabulary:", ", ".join(shared_top))

# Bigram experiment
tfidf_bigram_vec = TfidfVectorizer(
    stop_words=clean_stop_words,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    ngram_range=(1, 2),
    max_df=0.5,
    min_df=5,
    max_features=8000,
)
X_tfidf_bigram = tfidf_bigram_vec.fit_transform(df["text"])
kmeans_bigram = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_bigram"] = kmeans_bigram.fit_predict(X_tfidf_bigram)
print("\n--- Crosstab: true label vs. bigram cluster ---")
print(pd.crosstab(df["true_label"], df["cluster_bigram"]))


def cluster_purity(labels, clusters):
    table = pd.crosstab(labels, clusters)
    return table.max(axis=0).sum() / table.values.sum()


print(f"\nPurity  baseline: {cluster_purity(df['true_label'], df['cluster']):.3f} | "
      f"cleaned: {cluster_purity(df['true_label'], df['cluster_clean']):.3f} | "
      f"bigram: {cluster_purity(df['true_label'], df['cluster_bigram']):.3f}")

# %%
# Challenge 3 (Rant Length Bias)
# TODO 4.3
lengths = df["text"].str.split().str.len()
longest_id, shortest_id = int(lengths.idxmax()), int(lengths.idxmin())
for name, M in [("X_bow", X_bow), ("X_tfidf", X_tfidf)]:
    long_norm = np.linalg.norm(M[longest_id].toarray())
    short_norm = np.linalg.norm(M[shortest_id].toarray())
    print(f"{name:8s} | longest post (#{longest_id}, {lengths[longest_id]} words) ||x||2 = {long_norm:8.3f} | "
          f"shortest post (#{shortest_id}, {lengths[shortest_id]} words) ||x||2 = {short_norm:6.3f}")


# %%
# Challenge 4 (Stakeholder Explainability Audit)
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return the top_n highest-weighted non-zero TF-IDF terms for document `doc_id`
    as a list of (term, weight) tuples, sorted by weight descending.
    """
    row = matrix[doc_id]
    if hasattr(row, "tocoo"):
        row = row.tocoo()
        indices, weights = row.col, row.data
    else:
        dense = np.asarray(row).ravel()
        indices = np.nonzero(dense)[0]
        weights = dense[indices]

    terms = vectorizer.get_feature_names_out()
    order = np.argsort(weights)[::-1][:top_n]
    return [(str(terms[indices[i]]), float(weights[i])) for i in order]


doc_id = 412
print(f"\n--- Explainability audit: Document #{doc_id} ---")
print(f"True label: {df.loc[doc_id, 'true_label']} | Assigned cluster: {df.loc[doc_id, 'cluster']}")
for term, weight in explain_document(doc_id, X_tfidf, tfidf_vec, top_n=5):
    print(f"  {term:<15s} {weight:.4f}")
