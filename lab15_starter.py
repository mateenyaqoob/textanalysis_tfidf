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
    terms = np.array(tfidf_vec.get_feature_names_out())
    order_centroids = kmeans.cluster_centers_.argsort(axis=1)[:, ::-1]

    print("\n--- Top 8 words per cluster ---")
    for c in range(kmeans.n_clusters):
        print(f"Cluster {c}: {', '.join(terms[order_centroids[c, :8]])}")

    heat_idx = []
    for c in range(kmeans.n_clusters):
        for i in order_centroids[c, :5]:
            if i not in heat_idx:
                heat_idx.append(i)
    heat_df = pd.DataFrame(
        kmeans.cluster_centers_[:, heat_idx],
        index=[f"Cluster {c}" for c in range(kmeans.n_clusters)],
        columns=terms[heat_idx],
    )
    plt.figure(figsize=(12, 4))
    sns.heatmap(heat_df, cmap="YlOrRd", annot=False)
    plt.title("Cluster-by-Word Importance (K-Means Centroids)")
    plt.xticks(rotation=70, ha="right")
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
    plt.title("Document Map: TF-IDF Projected to 2D (colored by cluster)")
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
print("\n=== Challenge 1: Noise Filtering ===")
custom_stopwords = list(
    set(stopwords.words("english"))
    | {"know", "like", "just", "think", "does", "use", "used", "using", "thanks",
       "thank", "need", "want", "get", "got", "good", "make", "really", "anyone",
       "please", "help", "way", "say", "said", "going", "tell", "look", "does"}
)
tfidf_clean = TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    stop_words=custom_stopwords,
    max_df=0.5,
    min_df=5,
    max_features=4000,
)
X_clean = tfidf_clean.fit_transform(df["text"])
kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_clean)

clean_terms = np.array(tfidf_clean.get_feature_names_out())
clean_order = kmeans_clean.cluster_centers_.argsort(axis=1)[:, ::-1]
print("BEFORE (baseline) top words:")
for c in range(4):
    print(f"  Cluster {c}: {', '.join(terms[order_centroids[c, :8]])}")
print("AFTER (refined) top words:")
for c in range(4):
    print(f"  Cluster {c}: {', '.join(clean_terms[clean_order[c, :8]])}")


# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Run `pd.crosstab(df['true_label'], df['cluster'])` and inspect overlapping categories.
print("\n=== Challenge 2: Hardware Overlap ===")
ct = pd.crosstab(df["true_label"], df["cluster"])
print(ct)

dominant = ct.idxmax(axis=1)
elec_cluster = dominant["sci.electronics"]
mis = df[(df["true_label"] == "comp.sys.mac.hardware") & (df["cluster"] == elec_cluster)]
print(f"\nMac-hardware posts routed to the electronics cluster ({elec_cluster}): {len(mis)}")
if len(mis) > 0:
    mis_mean = np.asarray(X_tfidf[mis.index.to_numpy()].mean(axis=0)).ravel()
    print("Words pulling them together:", list(terms[mis_mean.argsort()[-15:][::-1]]))
    for t in mis["text"].head(3):
        print("  ---", t[:200].replace("\n", " "))

# Bigrams
tfidf_bi = TfidfVectorizer(
    ngram_range=(1, 2),
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    stop_words=custom_stopwords,
    max_df=0.5,
    min_df=5,
    max_features=6000,
)
X_bi = tfidf_bi.fit_transform(df["text"])
km_bi = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_bi)
print("\nCrosstab with bigrams:")
print(pd.crosstab(df["true_label"], km_bi.labels_))
bi_terms = np.array(tfidf_bi.get_feature_names_out())
bi_means = np.asarray(X_bi.mean(axis=0)).ravel()
print("Top bigrams:", [w for w in bi_terms[bi_means.argsort()[::-1]] if " " in w][:15])


# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 vector norms (`np.linalg.norm`) of the longest and shortest posts in `X_bow` vs `X_tfidf`.
print("\n=== Challenge 3: Rant Length Bias ===")
doc_len = np.asarray(X_bow.sum(axis=1)).ravel()
nonempty = np.where(doc_len > 0)[0]
longest = nonempty[doc_len[nonempty].argmax()]
shortest = nonempty[doc_len[nonempty].argmin()]

for name, i in [("Longest", longest), ("Shortest", shortest)]:
    bow_norm = np.linalg.norm(X_bow[i].toarray())
    tfidf_norm = np.linalg.norm(X_tfidf[i].toarray())
    print(f"{name} post (#{i}, {int(doc_len[i])} tokens): "
          f"BoW norm = {bow_norm:.2f} | TF-IDF norm = {tfidf_norm:.4f}")


# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return or print the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` (test with doc_id=412).
    """
    names = vectorizer.get_feature_names_out()
    row = matrix[doc_id]
    cols, vals = row.indices, row.data  # non-zero entries only
    order = vals.argsort()[::-1][:top_n]
    result = [(str(names[cols[i]]), float(vals[i])) for i in order]

    print(f"\nDocument #{doc_id} | true label: {df['true_label'].iloc[doc_id]} "
          f"| assigned cluster: {df['cluster'].iloc[doc_id]}")
    for rank, (word, weight) in enumerate(result, 1):
        print(f"  {rank}. {word:<15} TF-IDF = {weight:.4f}")
    return result


print("\n=== Challenge 4: Explainability ===")
explain_document(412, X_tfidf, tfidf_vec, top_n=5)
