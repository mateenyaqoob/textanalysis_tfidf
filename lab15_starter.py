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
    terms = tfidf_vec.get_feature_names_out()
    centroids = kmeans.cluster_centers_
    order = centroids.argsort(axis=1)[:, ::-1]  # har cluster ke words, highest weight pehle

    print("\n--- Top 8 Words per Cluster ---")
    for c in range(kmeans.n_clusters):
        top8 = [terms[i] for i in order[c, :8]]
        size = (df["cluster"] == c).sum()
        print(f"Cluster {c} ({size} posts): {', '.join(top8)}")

    heat_idx = list(dict.fromkeys(order[:, :5].ravel()))  # har cluster ke top 5, bina duplicate
    heat_df = pd.DataFrame(
        centroids[:, heat_idx],
        index=[f"Cluster {c}" for c in range(kmeans.n_clusters)],
        columns=terms[heat_idx],
    )
    plt.figure(figsize=(12, 4))
    sns.heatmap(heat_df, cmap="YlGnBu", linewidths=0.5)
    plt.title("Cluster vs. Top-5 Word Importance (Centroid TF-IDF Weights)")
    plt.xlabel("Word")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.show()

    # TODO 3.3: Project `X_tfidf` to 2 dimensions using `TruncatedSVD(n_components=2, random_state=42)`
    # and save a 2D scatter plot (`spatial_mapping.png`) colored by `df["cluster"]`.
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_tfidf)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster"], cmap="tab10", s=8, alpha=0.6)
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
# %%
# Challenge 1 (Noise Filtering):
# TODO 4.1
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def cluster_purity(labels):
    """% of posts that fall in their cluster's majority true category."""
    ct = pd.crosstab(df["true_label"], labels)
    return 100.0 * ct.max(axis=0).sum() / ct.values.sum()

def print_top_words(model, vectorizer, labels, n=8):
    terms = vectorizer.get_feature_names_out()
    order = model.cluster_centers_.argsort(axis=1)[:, ::-1]
    for c in range(model.n_clusters):
        size = (labels == c).sum()
        print(f"Cluster {c} ({size} posts): {', '.join(terms[order[c, :n]])}")

# Why does a filler word like 'know' still get weight? Check its document frequency and IDF.
vocab = tfidf_vec.vocabulary_
n_docs = X_tfidf.shape[0]
print("--- Baseline IDF of conversational filler words ---")
for w in ["know", "like", "just", "think", "does", "use", "thanks"]:
    if w in vocab:
        j = vocab[w]
        doc_freq = 100.0 * X_bow[:, j].getnnz() / n_docs
        print(f"{w:>8}: appears in {doc_freq:5.1f}% of posts | idf = {tfidf_vec.idf_[j]:.3f}")

filler_words = [
    "know", "like", "just", "think", "does", "use", "thanks", "don", "doesn", "didn",
    "really", "want", "need", "good", "make", "way", "people", "got", "going", "say",
    "thing", "things", "used", "anybody", "anyone", "sure", "right", "said", "isn",
]
custom_stop_words = list(ENGLISH_STOP_WORDS.union(filler_words))

tfidf_clean_vec = TfidfVectorizer(
    stop_words=custom_stop_words,
    token_pattern=r"\b[a-zA-Z]{3,}\b",  # only words with 3+ letters, so no numbers
    max_df=0.5,
    max_features=4000,
    min_df=5,
)
X_tfidf_clean = tfidf_clean_vec.fit_transform(df["text"])
kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = kmeans_clean.fit_predict(X_tfidf_clean)

print("\n--- BEFORE cleaning: top 8 words per cluster ---")
print_top_words(kmeans, tfidf_vec, df["cluster"])
print("\n--- AFTER cleaning: top 8 words per cluster ---")
print_top_words(kmeans_clean, tfidf_clean_vec, df["cluster_clean"])
print(f"\nCluster purity  baseline: {cluster_purity(df['cluster']):.1f}%  |  cleaned: {cluster_purity(df['cluster_clean']):.1f}%")


# %%
# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2
print("--- Crosstab: true category vs. cleaned cluster ---")
ct_clean = pd.crosstab(df["true_label"], df["cluster_clean"])
print(ct_clean)

# Name each cluster by its majority true category, then find misclustered posts
cluster_to_label = ct_clean.idxmax(axis=0)
df["predicted_label"] = df["cluster_clean"].map(cluster_to_label)
hw = ["sci.electronics", "comp.sys.mac.hardware"]
mis = df[df["true_label"].isin(hw) & (df["true_label"] != df["predicted_label"])]
print(f"\nMisclustered hardware posts (electronics/mac): {len(mis)}")

# Shared vocabulary: words strong in BOTH electronics and mac posts
terms_clean = tfidf_clean_vec.get_feature_names_out()
elec_mean = np.asarray(X_tfidf_clean[(df["true_label"] == "sci.electronics").values].mean(axis=0)).ravel()
mac_mean = np.asarray(X_tfidf_clean[(df["true_label"] == "comp.sys.mac.hardware").values].mean(axis=0)).ravel()
shared = np.minimum(elec_mean, mac_mean)
print("Top shared words (electronics & mac):", ", ".join(terms_clean[shared.argsort()[::-1][:10]]))

if len(mis) > 0:
    mis_mean = np.asarray(X_tfidf_clean[mis.index].mean(axis=0)).ravel()
    print("Top words inside misclustered posts: ", ", ".join(terms_clean[mis_mean.argsort()[::-1][:10]]))
    print("\nSample misclustered posts:")
    for i in mis.index[:3]:
        snippet = " ".join(df.loc[i, "text"].split())[:200]
        print(f"  [#{i}] true={df.loc[i, 'true_label']} -> cluster={df.loc[i, 'predicted_label']}: {snippet}...")

# Add bigrams so phrases like 'power supply' or 'mac ii' become their own features
tfidf_bigram_vec = TfidfVectorizer(
    stop_words=custom_stop_words,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    min_df=5,
    max_features=8000,
    ngram_range=(1, 2),
)
X_tfidf_bigram = tfidf_bigram_vec.fit_transform(df["text"])
kmeans_bigram = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_bigram"] = kmeans_bigram.fit_predict(X_tfidf_bigram)

print("\n--- Crosstab with bigrams (ngram_range=(1, 2)) ---")
print(pd.crosstab(df["true_label"], df["cluster_bigram"]))
print("\n--- Top 8 terms per cluster with bigrams ---")
print_top_words(kmeans_bigram, tfidf_bigram_vec, df["cluster_bigram"])
print(f"\nCluster purity  unigram (cleaned): {cluster_purity(df['cluster_clean']):.1f}%  |  bigram: {cluster_purity(df['cluster_bigram']):.1f}%")


# %%
# Challenge 3 (Rant Length Bias):
# TODO 4.3
word_counts = df["text"].str.split().str.len()
has_vocab = np.asarray(X_bow.getnnz(axis=1)) > 0  # skip posts with no vocabulary words at all
longest_id = word_counts[has_vocab].idxmax()
shortest_id = word_counts[has_vocab].idxmin()

print("--- L2 norm: longest vs. shortest post ---")
for name, i in [("Longest", longest_id), ("Shortest", shortest_id)]:
    bow_norm = np.linalg.norm(X_bow[i].toarray())
    tfidf_norm = np.linalg.norm(X_tfidf[i].toarray())
    print(f"{name:>8} post #{i}: {word_counts[i]:5d} words | BoW norm = {bow_norm:8.2f} | TF-IDF norm = {tfidf_norm:.4f}")

bow_norms = np.sqrt(np.asarray(X_bow.multiply(X_bow).sum(axis=1)).ravel())
print(f"\nCorrelation(word count, BoW norm) = {np.corrcoef(word_counts, bow_norms)[0, 1]:.3f}")

kmeans_bow = KMeans(n_clusters=4, random_state=42, n_init=10)
bow_labels = kmeans_bow.fit_predict(X_bow)
print("\n--- KMeans on raw BoW counts: cluster sizes ---")
print(pd.Series(bow_labels).value_counts().sort_index().to_string())
print(f"\nCluster purity  raw BoW: {cluster_purity(bow_labels):.1f}%  |  TF-IDF: {cluster_purity(df['cluster']):.1f}%")


# %%
# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return or print the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` (test with doc_id=412).
    """
    row = matrix[doc_id].toarray().ravel()
    terms = vectorizer.get_feature_names_out()
    nonzero = np.flatnonzero(row)
    top = nonzero[np.argsort(row[nonzero])[::-1][:top_n]]
    return [(str(terms[j]), round(float(row[j]), 4)) for j in top]


doc_id = 412
print(f"--- Audit: Document #{doc_id} ---")
print(f"True category: {df.loc[doc_id, 'true_label']} | Assigned cluster: {df.loc[doc_id, 'cluster']}")
for term, weight in explain_document(doc_id, X_tfidf, tfidf_vec, top_n=5):
    print(f"  {term:<15} {weight:.4f}")

cluster_terms = tfidf_vec.get_feature_names_out()[kmeans.cluster_centers_[df.loc[doc_id, "cluster"]].argsort()[::-1][:8]]
print(f"Cluster {df.loc[doc_id, 'cluster']} top words: {', '.join(cluster_terms)}")
print("\nPost snippet:", " ".join(df.loc[doc_id, "text"].split())[:300])
# %%
