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
tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]  # Replace None with your list comprehension

# TODO 1.2: Remove standard English stopwords using `stopwords.words('english')`.
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]  # Replace None

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
    # TODO 3.1: K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    # TODO 3.2: Top 8 words per cluster
    feature_names = tfidf_vec.get_feature_names_out()
    centers = kmeans.cluster_centers_

    for i in range(4):
        top_idx = centers[i].argsort()[-8:][::-1]
        print(f"Cluster {i}: {', '.join(feature_names[top_idx])}")

    # Heatmap of top 5 words per cluster
    top_words_idx = np.unique(
        np.concatenate([centers[i].argsort()[-5:] for i in range(4)])
    )

    plt.figure(figsize=(12, 5))
    sns.heatmap(
        centers[:, top_words_idx],
        xticklabels=feature_names[top_words_idx],
        yticklabels=[f"Cluster {i}" for i in range(4)],
        cmap="YlGnBu"
    )
    plt.title("Top TF-IDF Words per Cluster")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.show()

    # TODO 3.3: 2D Spatial Mapping
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_tfidf)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=df["cluster"],
        palette="tab10",
        s=15
    )
    plt.title("2D Spatial Mapping of Text Clusters")
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
custom_stopwords = list(CountVectorizer(stop_words="english").get_stop_words()) + [
    "know", "like", "just", "think", "does", "use", "thanks"
]

clean_vec = TfidfVectorizer(
    stop_words=custom_stopwords,
    token_pattern=r'\b[a-zA-Z]{3,}\b',
    max_df=0.5,
    max_features=4000,
    min_df=5
)

X_clean = clean_vec.fit_transform(df["text"])

clean_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clean_kmeans.fit(X_clean)

clean_words = clean_vec.get_feature_names_out()

for i, center in enumerate(clean_kmeans.cluster_centers_):
    top_idx = center.argsort()[-8:][::-1]
    print(f"Clean Cluster {i}: {', '.join(clean_words[top_idx])}")


# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Run `pd.crosstab(df['true_label'], df['cluster'])` and inspect overlapping categories.
print("\nCluster vs Actual Category:")
print(pd.crosstab(df["true_label"], df["cluster"]))

# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 vector norms (`np.linalg.norm`) of the longest and shortest posts in `X_bow` vs `X_tfidf`.
longest = df["text"].str.len().idxmax()
shortest = df["text"].str.len().idxmin()

for name, idx in [("Longest", longest), ("Shortest", shortest)]:
    bow_norm = np.linalg.norm(X_bow[idx].toarray())
    tfidf_norm = np.linalg.norm(X_tfidf[idx].toarray())

    print(f"{name} post: BoW Norm = {bow_norm:.4f}, TF-IDF Norm = {tfidf_norm:.4f}")

# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """Return the top TF-IDF terms and their weights for a document."""
    row = matrix[doc_id].toarray().ravel()
    feature_names = vectorizer.get_feature_names_out()

    top_indices = row.argsort()[::-1][:top_n]

    return [(feature_names[i], float(row[i]))
            for i in top_indices if row[i] > 0]


print("\nDocument 412 - Top TF-IDF Terms:")
print(explain_document(412, X_tfidf, tfidf_vec))
# Challenge 2: Inspect hardware vocabulary overlap
hardware = df[
    df["true_label"].isin([
        "sci.electronics",
        "comp.sys.mac.hardware"
    ])
]

shared_words = (
    set(bow_vec.build_analyzer()( " ".join(
        hardware[hardware["true_label"] == "sci.electronics"]["text"]
    )))
    &
    set(bow_vec.build_analyzer()( " ".join(
        hardware[hardware["true_label"] == "comp.sys.mac.hardware"]["text"]
    )))
)

print("\nShared hardware terms:")
print(sorted(shared_words & {
    "drive", "card", "monitor", "power",
    "cable", "disk", "memory", "scsi"
}))

bigram_vec = TfidfVectorizer(
    stop_words="english",
    max_features=4000,
    min_df=5,
    ngram_range=(1, 2)
)

X_bigram = bigram_vec.fit_transform(df["text"])
bigram_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["bigram_cluster"] = bigram_kmeans.fit_predict(X_bigram)

print("\nBigram cluster comparison:")
print(pd.crosstab(df["true_label"], df["bigram_cluster"]))