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
lemmatized_tokens = [lemmatizer.lemmatize(w, pos='v') for w in filtered_tokens]

if filtered_tokens is not None:
    print("Filtered Tokens:  ", filtered_tokens[:12])
    print("Stemmed Roots:    ", stemmed_tokens[:12])
    print("Lemmatized Words: ", lemmatized_tokens[:12])


# %% [markdown]
# ## Step 2: Bag of Words vs. TF-IDF Feature Extraction (15 mins)

# %%
# TODO 2.1: Initialize CountVectorizer and TfidfVectorizer with:
# stop_words='english', max_features=4000, min_df=5
bow_vec = CountVectorizer(stop_words='english', max_features=4000, min_df=5)
tfidf_vec = TfidfVectorizer(stop_words='english', max_features=4000, min_df=5)

# TODO 2.2: Fit and transform `df['text']` to create `X_bow` and `X_tfidf`
X_bow = bow_vec.fit_transform(df['text'])
X_tfidf = tfidf_vec.fit_transform(df['text'])

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
    plt.close()


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
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    
    heatmap_data = {}
    for i in range(4):
        print(f"Cluster {i} top 8 words:")
        cluster_words = [terms[ind] for ind in order_centroids[i, :8]]
        print(cluster_words)
        # For heatmap, let's store top 5 words and their centroids weights
        for ind in order_centroids[i, :5]:
            word = terms[ind]
            if word not in heatmap_data:
                heatmap_data[word] = [0.0]*4
            heatmap_data[word][i] = kmeans.cluster_centers_[i, ind]
            
    df_hm = pd.DataFrame(heatmap_data).T
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_hm, annot=True, cmap="YlGnBu")
    plt.title("Top 5 TF-IDF Terms per Cluster Centroid")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.close()

    # TODO 3.3: Project `X_tfidf` to 2 dimensions using `TruncatedSVD(n_components=2, random_state=42)`
    # and save a 2D scatter plot (`spatial_mapping.png`) colored by `df["cluster"]`.
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X_tfidf)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=df["cluster"], palette="viridis", alpha=0.6)
    plt.title("2D Spatial Mapping of Document Clusters")
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.close()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %%
# Challenge 1 (Noise Filtering):
# TODO 4.1: Build a cleaner `TfidfVectorizer` using `token_pattern=r'\b[a-zA-Z]{3,}\b'`,
# `max_df=0.5`, and custom conversational stopwords (e.g., 'know', 'like', 'just', 'think', 'does', 'use', 'thanks').
# Re-fit KMeans and compare cluster keywords!
custom_stopwords = list(stopwords.words("english")) + ['know', 'like', 'just', 'think', 'does', 'use', 'thanks']
tfidf_vec_clean = TfidfVectorizer(
    stop_words=custom_stopwords, 
    token_pattern=r'\b[a-zA-Z]{3,}\b',
    max_df=0.5,
    max_features=4000,
    min_df=5
)
X_tfidf_clean = tfidf_vec_clean.fit_transform(df['text'])
kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_clean = kmeans_clean.fit_predict(X_tfidf_clean)

# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Run `pd.crosstab(df['true_label'], df['cluster'])` and inspect overlapping categories.
cross_tab = pd.crosstab(df['true_label'], df['cluster'])
print("\n--- Crosstab of True Labels vs. Clusters ---")
print(cross_tab)

# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 vector norms (`np.linalg.norm`) of the longest and shortest posts in `X_bow` vs `X_tfidf`.
lens = df['text'].str.len()
longest_idx = lens.idxmax()
shortest_idx = lens.idxmin()

bow_longest_norm = np.linalg.norm(X_bow[longest_idx].toarray())
bow_shortest_norm = np.linalg.norm(X_bow[shortest_idx].toarray())
tfidf_longest_norm = np.linalg.norm(X_tfidf[longest_idx].toarray())
tfidf_shortest_norm = np.linalg.norm(X_tfidf[shortest_idx].toarray())
print(f"\nLongest document length: {lens[longest_idx]} characters | Shortest: {lens[shortest_idx]}")
print(f"BoW L2 Norm - Longest: {bow_longest_norm:.2f}, Shortest: {bow_shortest_norm:.2f}")
print(f"TF-IDF L2 Norm - Longest: {tfidf_longest_norm:.2f}, Shortest: {tfidf_shortest_norm:.2f}")

# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return or print the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` (test with doc_id=412).
    """
    row = matrix[doc_id].toarray().ravel()
    non_zero_indices = row.nonzero()[0]
    feature_names = vectorizer.get_feature_names_out()
    items = [(feature_names[i], row[i]) for i in non_zero_indices]
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    return items_sorted[:top_n]

explanation_412 = explain_document(412, X_tfidf, tfidf_vec, top_n=5)
print("\n--- TF-IDF Explanation for Document 412 ---")
for term, weight in explanation_412:
    print(f"{term}: {weight:.4f}")
