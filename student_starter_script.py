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

# Tokenize, lowercase, and retain alphabetic words.
tokens = [word.lower() for word in word_tokenize(sample_post) if word.isalpha()]

# TODO 1.2: Remove standard English stopwords using `stopwords.words('english')`.
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if word not in stop_words]

# TODO 1.3: Apply PorterStemmer and WordNetLemmatizer (with pos='v') to `filtered_tokens`.
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(word, pos="v") for word in filtered_tokens]

if filtered_tokens is not None:
    print("Filtered Tokens:  ", filtered_tokens[:12])
    print("Stemmed Roots:    ", stemmed_tokens[:12])
    print("Lemmatized Words: ", lemmatized_tokens[:12])


# %% [markdown]
# ## Step 2: Bag of Words vs. TF-IDF Feature Extraction (15 mins)

# %%
# Initialize both vectorizers with the requested vocabulary limits.
bow_vec = CountVectorizer(stop_words="english", max_features=4000, min_df=5)
tfidf_vec = TfidfVectorizer(stop_words="english", max_features=4000, min_df=5)

# Fit and transform the full discussion corpus.
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
    plt.close()


# %% [markdown]
# ## Step 3: Automated Topic Discovery & Spatial Mapping (15 mins)

# %%
if X_tfidf is not None:
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    feature_names = tfidf_vec.get_feature_names_out()
    centroid_order = kmeans.cluster_centers_.argsort(axis=1)[:, ::-1]
    for cluster_id, feature_indices in enumerate(centroid_order):
        words = feature_names[feature_indices[:8]]
        print(f"Cluster {cluster_id} top words: {', '.join(words)}")

    heatmap_indices = np.unique(centroid_order[:, :5])
    heatmap_words = feature_names[heatmap_indices]
    sns.heatmap(
        kmeans.cluster_centers_[:, heatmap_indices],
        xticklabels=heatmap_words,
        yticklabels=[f"Cluster {i}" for i in range(kmeans.n_clusters)],
        cmap="YlGnBu",
    )
    plt.title("Cluster Centroid Feature Importance")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.close()

    svd = TruncatedSVD(n_components=2, random_state=42)
    projection = svd.fit_transform(X_tfidf)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        projection[:, 0], projection[:, 1], c=df["cluster"], cmap="tab10", s=12, alpha=0.7
    )
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.title("Document Clusters in 2D TF-IDF Space")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.close()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %%
# Challenge 1 (Noise Filtering):
custom_stop_words = set(stopwords.words("english")) | {
    "know", "like", "just", "think", "does", "use", "thanks"
}
clean_tfidf_vec = TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    stop_words=sorted(custom_stop_words),
    max_df=0.5,
    max_features=4000,
    min_df=5,
)
X_tfidf_clean = clean_tfidf_vec.fit_transform(df["text"])
clean_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_tfidf_clean)
clean_features = clean_tfidf_vec.get_feature_names_out()
for cluster_id, feature_indices in enumerate(clean_kmeans.cluster_centers_.argsort(axis=1)[:, ::-1]):
    print(f"Filtered cluster {cluster_id} top words: {', '.join(clean_features[feature_indices[:8]])}")


# Challenge 2 (Hardware Overlap Diagnosis):
cluster_crosstab = pd.crosstab(df["true_label"], df["cluster"])
print("\nTrue category by baseline cluster:\n", cluster_crosstab)


# Challenge 3 (Rant Length Bias):
post_lengths = df["text"].str.len()
longest_doc_id = int(post_lengths.idxmax())
shortest_doc_id = int(post_lengths.idxmin())
for description, doc_id in (("Longest", longest_doc_id), ("Shortest", shortest_doc_id)):
    bow_norm = np.linalg.norm(X_bow[doc_id].toarray())
    tfidf_norm = np.linalg.norm(X_tfidf[doc_id].toarray())
    print(f"{description} post (doc {doc_id}) L2 norms: BoW={bow_norm:.3f}, TF-IDF={tfidf_norm:.3f}")


# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix=None, vectorizer=None, top_n=5):
    """
    Return the top_n highest-weighted TF-IDF terms and their exact weights.
    """
    matrix = X_tfidf if matrix is None else matrix
    vectorizer = tfidf_vec if vectorizer is None else vectorizer
    if doc_id < 0 or doc_id >= matrix.shape[0]:
        raise IndexError(f"doc_id must be between 0 and {matrix.shape[0] - 1}")
    if top_n < 0:
        raise ValueError("top_n must be non-negative")

    row = matrix.getrow(doc_id)
    if row.nnz == 0 or top_n == 0:
        return []
    order = np.argsort(row.data)[::-1][:top_n]
    terms = vectorizer.get_feature_names_out()
    explanation = [(terms[row.indices[index]], float(row.data[index])) for index in order]
    print(f"Document {doc_id} top TF-IDF terms: {explanation}")
    return explanation


document_412_explanation = explain_document(doc_id=412, top_n=5)
