"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Roll no: 25i8033

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
# ## Step 1: Text Cleaning, Tokenization & Normalization
# %%
sample_post = df["text"].iloc[0][:400]
print("--- Raw Sample Post ---")
print(sample_post)
print("-" * 40)

# TODO 1.1: lowercase, tokenize, keep alphabetic tokens only
tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]

# TODO 1.2: remove English stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]

# TODO 1.3: PorterStemmer vs WordNetLemmatizer (pos='v') on the same tokens
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(w, pos="v") for w in filtered_tokens]

if filtered_tokens is not None:
    print("Filtered Tokens: ", filtered_tokens[:12])
    print("Stemmed Roots:   ", stemmed_tokens[:12])
    print("Lemmatized Words:", lemmatized_tokens[:12])


# %% [markdown]
# ## Step 2: Bag of Words vs. TF-IDF Feature Extraction
# %%
# TODO 2.1: vectorizers
bow_vec = CountVectorizer(stop_words="english", max_features=4000, min_df=5)
tfidf_vec = TfidfVectorizer(stop_words="english", max_features=4000, min_df=5)

# TODO 2.2: fit & transform
X_bow = bow_vec.fit_transform(df["text"])
X_tfidf = tfidf_vec.fit_transform(df["text"])

# TODO 2.3: sparsity of the TF-IDF matrix
if X_tfidf is not None:
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


# %% [markdown]
# ## Step 3: Automated Topic Discovery & Spatial Mapping
# %%
if X_tfidf is not None:
    # TODO 3.1: KMeans on TF-IDF
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    # TODO 3.2: top 8 words per centroid + heatmap of top 5 per cluster
    terms = tfidf_vec.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    print("\n--- Top 8 words per cluster ---")
    for c in range(kmeans.n_clusters):
        words = [terms[i] for i in order_centroids[c, :8]]
        print(f"Cluster {c}: {', '.join(words)}")

    heat_terms = []
    for c in range(kmeans.n_clusters):
        for i in order_centroids[c, :5]:
            if terms[i] not in heat_terms:
                heat_terms.append(terms[i])
    heat_idx = [list(terms).index(t) for t in heat_terms]
    heat_df = pd.DataFrame(
        kmeans.cluster_centers_[:, heat_idx],
        index=[f"Cluster {c}" for c in range(kmeans.n_clusters)],
        columns=heat_terms,
    )
    plt.figure(figsize=(12, 3.8))
    sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Cluster Centroid Weights: Top 5 Words per Cluster")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.show()

    # TODO 3.3: 2D projection with TruncatedSVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X_tfidf)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1], c=df["cluster"], cmap="tab10", s=8, alpha=0.6
    )
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.title("Spatial Mapping of Posts (TF-IDF -> 2D SVD)")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.show()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics
# %%
# ---------------------------------------------------------------
# Challenge 1: conversational filler & numeric noise
# ---------------------------------------------------------------
custom_stops = list(ENGLISH_STOP_WORDS) + [
    "know", "like", "just", "think", "does", "use", "thanks", "don", "ve",
    "ll", "really", "going", "want", "need", "good", "make", "way", "used",
]
clean_vec = TfidfVectorizer(
    stop_words=custom_stops,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    min_df=5,
    max_features=4000,
)
X_clean = clean_vec.fit_transform(df["text"])
kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = kmeans_clean.fit_predict(X_clean)

clean_terms = clean_vec.get_feature_names_out()
clean_order = kmeans_clean.cluster_centers_.argsort()[:, ::-1]
print("\n--- Challenge 1: top 8 words per cluster AFTER noise filtering ---")
for c in range(kmeans_clean.n_clusters):
    print(f"Cluster {c}: {', '.join(clean_terms[i] for i in clean_order[c, :8])}")

# ---------------------------------------------------------------
# Challenge 2: hardware vocabulary overlap
# ---------------------------------------------------------------
print("\n--- Challenge 2: true label vs. cluster (baseline) ---")
ct = pd.crosstab(df["true_label"], df["cluster"])
print(ct)

# Which two categories overlap the most? (cosine similarity of cluster distributions)
ct_norm = ct.div(ct.sum(axis=1), axis=0)
labels = list(ct_norm.index)
best_pair, best_sim = None, -1.0
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        a, b = ct_norm.iloc[i].values, ct_norm.iloc[j].values
        sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        if sim > best_sim:
            best_pair, best_sim = (labels[i], labels[j]), sim
print(f"\nMost overlapping categories: {best_pair[0]} <-> {best_pair[1]} "
      f"(cluster-distribution cosine similarity = {best_sim:.3f})")

# Shared technical vocabulary between those two categories
terms_all = tfidf_vec.get_feature_names_out()
top_by_label = {}
for lab in best_pair:
    mask = (df["true_label"] == lab).values
    means = np.asarray(X_tfidf[mask].mean(axis=0)).ravel()
    top_by_label[lab] = set(terms_all[means.argsort()[-40:]])
shared = sorted(top_by_label[best_pair[0]] & top_by_label[best_pair[1]])
print(f"Shared top-40 words: {shared}")

# Peek at a few misclustered posts
majority_cluster = ct.idxmax(axis=1)  # label -> dominant cluster
mis = df[df.apply(lambda r: r["cluster"] != majority_cluster[r["true_label"]], axis=1)]
print(f"\nMisclustered posts: {len(mis)} of {len(df)}")
for _, row in mis[mis["true_label"].isin(best_pair)].head(3).iterrows():
    print(f"\n[{row['true_label']} -> cluster {row['cluster']}] "
          f"{row['text'].strip()[:200]!r}")

# Bigrams
bigram_vec = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 2), max_features=6000, min_df=5
)
X_bi = bigram_vec.fit_transform(df["text"])
km_bi = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_bigram"] = km_bi.fit_predict(X_bi)
print("\n--- Challenge 2: true label vs. cluster (with bigrams) ---")
print(pd.crosstab(df["true_label"], df["cluster_bigram"]))

# ---------------------------------------------------------------
# Challenge 3: rant length bias
# ---------------------------------------------------------------
doc_len = df["text"].str.split().str.len().values
longest, shortest = int(doc_len.argmax()), int(doc_len.argmin())


def row_norm(M, i):
    return float(np.linalg.norm(M[i].toarray().ravel()))


print("\n--- Challenge 3: L2 norms ---")
print(f"Longest post  (#{longest}, {doc_len[longest]} words)")
print(f"  BoW norm   : {row_norm(X_bow, longest):.3f}")
print(f"  TF-IDF norm: {row_norm(X_tfidf, longest):.3f}")
print(f"Shortest post (#{shortest}, {doc_len[shortest]} words)")
print(f"  BoW norm   : {row_norm(X_bow, shortest):.3f}")
print(f"  TF-IDF norm: {row_norm(X_tfidf, shortest):.3f}")


# ---------------------------------------------------------------
# Challenge 4: stakeholder explainability audit
# ---------------------------------------------------------------
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """Return the top_n highest-weighted non-zero TF-IDF terms for one document
    as a list of (term, weight) tuples, and print them."""
    row = matrix[doc_id].tocoo()
    terms = vectorizer.get_feature_names_out()
    pairs = sorted(zip(row.col, row.data), key=lambda p: p[1], reverse=True)[:top_n]
    result = [(str(terms[c]), float(w)) for c, w in pairs]

    print(f"\n--- Why was document #{doc_id} routed where it was? ---")
    if "cluster" in df.columns:
        print(f"Assigned cluster: {int(df['cluster'].iloc[doc_id])} | "
              f"true label: {df['true_label'].iloc[doc_id]}")
    for term, weight in result:
        print(f"  {term:<15} {weight:.4f}")
    return result


explain_document(412, X_tfidf, tfidf_vec, top_n=5)
