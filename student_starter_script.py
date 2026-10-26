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


# TODO 1.1: Tokenize, lowercase, and keep alphabetic tokens
tokens = [
    w.lower()
    for w in word_tokenize(sample_post)
    if w.isalpha()
]

# TODO 1.2: Remove standard English stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [
    w for w in tokens
    if w not in stop_words
]

# TODO 1.3: Apply stemming and lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [
    stemmer.stem(w)
    for w in filtered_tokens
]

lemmatized_tokens = [
    lemmatizer.lemmatize(w, pos="v")
    for w in filtered_tokens
]

print("Filtered Tokens:  ", filtered_tokens[:12])
print("Stemmed Roots:    ", stemmed_tokens[:12])
print("Lemmatized Words: ", lemmatized_tokens[:12])

# %% [markdown]
# ## Step 2: Bag of Words vs. TF-IDF Feature Extraction (15 mins)


# TODO 2.1: Initialize vectorizers
bow_vec = CountVectorizer(
    stop_words="english",
    max_features=4000,
    min_df=5
)

tfidf_vec = TfidfVectorizer(
    stop_words="english",
    max_features=4000,
    min_df=5
)

# TODO 2.2: Fit and transform the corpus
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



if X_tfidf is not None:
    # TODO 3.1: Fit KMeans
    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    # TODO 3.2: Display top 8 words for each cluster
    feature_names = tfidf_vec.get_feature_names_out()

    for cluster_id, centroid in enumerate(kmeans.cluster_centers_):
        top_indices = centroid.argsort()[-8:][::-1]
        top_terms = feature_names[top_indices]
        print(f"\nCluster {cluster_id} top words: {', '.join(top_terms)}")

    # Heatmap using the top 5 words from each cluster
    top_indices_all = set()
    for centroid in kmeans.cluster_centers_:
        top_indices_all.update(centroid.argsort()[-5:])

    selected_indices = sorted(top_indices_all)
    heatmap_data = kmeans.cluster_centers_[:, selected_indices]
    heatmap_words = feature_names[selected_indices]

    plt.figure(figsize=(12, 5))
    sns.heatmap(
        heatmap_data,
        xticklabels=heatmap_words,
        yticklabels=[f"Cluster {i}" for i in range(4)],
        cmap="YlGnBu"
    )
    plt.title("Cluster-by-Word TF-IDF Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.show()

    # TODO 3.3: Reduce TF-IDF vectors to two dimensions
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_tfidf)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=df["cluster"],
        cmap="viridis",
        s=12,
        alpha=0.7
    )
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.title("Document Clusters in 2D")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.show()



# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics

# %%
# Challenge 1: Noise Filtering

custom_stopwords = set(stopwords.words("english"))
custom_stopwords.update([
    "know", "like", "just", "think", "does",
    "use", "thanks"
])

clean_tfidf_vec = TfidfVectorizer(
    stop_words=list(custom_stopwords),
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    max_features=4000,
    min_df=5
)

X_clean_tfidf = clean_tfidf_vec.fit_transform(df["text"])

clean_kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["clean_cluster"] = clean_kmeans.fit_predict(X_clean_tfidf)

clean_features = clean_tfidf_vec.get_feature_names_out()

print("\n=== Challenge 1: Cleaned Cluster Keywords ===")
for cluster_id, centroid in enumerate(clean_kmeans.cluster_centers_):
    top_indices = centroid.argsort()[-8:][::-1]
    print(
        f"Clean cluster {cluster_id}: "
        + ", ".join(clean_features[top_indices])
    )


# Challenge 2: Hardware Overlap Diagnosis

print("\n=== Challenge 2: True Labels vs. Baseline Clusters ===")
label_cluster_table = pd.crosstab(
    df["true_label"],
    df["cluster"]
)
print(label_cluster_table)

print("\n=== True Labels vs. Cleaned Clusters ===")
clean_label_cluster_table = pd.crosstab(
    df["true_label"],
    df["clean_cluster"]
)
print(clean_label_cluster_table)


# Challenge 3: Rant Length Bias

# Find the shortest and longest posts by character count.
post_lengths = df["text"].str.len()
shortest_id = int(post_lengths.idxmin())
longest_id = int(post_lengths.idxmax())

def compare_document_norms(doc_id, description):
    bow_norm = np.linalg.norm(X_bow.getrow(doc_id).toarray())
    tfidf_norm = np.linalg.norm(X_tfidf.getrow(doc_id).toarray())

    print(
        f"{description} post (row {doc_id}, "
        f"{post_lengths.iloc[doc_id]} characters): "
        f"BoW L2 norm = {bow_norm:.4f}, "
        f"TF-IDF L2 norm = {tfidf_norm:.4f}"
    )

print("\n=== Challenge 3: Document Vector Norms ===")
compare_document_norms(shortest_id, "Shortest")
compare_document_norms(longest_id, "Longest")


# Challenge 4: Stakeholder Explainability Audit

def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    Return and print the top_n terms and weights for a document row.
    doc_id is a zero-based row index in the supplied matrix.
    """
    if doc_id < 0 or doc_id >= matrix.shape[0]:
        raise IndexError(
            f"doc_id must be between 0 and {matrix.shape[0] - 1}"
        )

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    row = matrix.getrow(doc_id)
    feature_names = vectorizer.get_feature_names_out()

    if row.nnz == 0:
        print(f"Document {doc_id} has no nonzero features.")
        return []

    order = np.argsort(row.data)[::-1][:top_n]

    results = [
        (feature_names[row.indices[i]], float(row.data[i]))
        for i in order
    ]

    print(f"\nTop {top_n} terms for document {doc_id}:")
    for term, weight in results:
        print(f"  {term}: {weight:.6f}")

    return results


print("\n=== Challenge 4: Explain Document 412 ===")
explain_document(412, X_tfidf, tfidf_vec, top_n=5)
