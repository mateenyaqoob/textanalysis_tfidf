"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques

Run interactively cell-by-cell (# %% markers) or via terminal:
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
# ## Step 1: Text Cleaning, Tokenization & Normalization

# %%
sample_post = df["text"].iloc[0][:400]
print("--- Raw Sample Post ---")
print(sample_post)
print("-" * 40)

# TODO 1.1: tokenize, lowercase, keep alphabetic tokens only
tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]

# TODO 1.2: remove standard English stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]

# TODO 1.3: PorterStemmer vs WordNetLemmatizer (pos='v')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(w, pos="v") for w in filtered_tokens]

if filtered_tokens is not None:
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

# TODO 2.3
if X_tfidf is not None:
    total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
    sparsity = 100.0 * (1.0 - (X_tfidf.nnz / total_elements))
    print(f"\nTF-IDF Matrix Shape: {X_tfidf.shape} | Sparsity: {sparsity:.2f}%")

    # TODO 2.4
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
    # TODO 3.1
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    # TODO 3.2: top 8 words per centroid + heatmap of top 5
    terms = tfidf_vec.get_feature_names_out()
    print("\n--- Top 8 words per cluster ---")
    for c in range(4):
        idx = kmeans.cluster_centers_[c].argsort()[::-1][:8]
        print(f"Cluster {c}: {', '.join(terms[idx])}")

    heat_words = []
    for c in range(4):
        for i in kmeans.cluster_centers_[c].argsort()[::-1][:5]:
            if terms[i] not in heat_words:
                heat_words.append(terms[i])
    word_idx = [list(terms).index(w) for w in heat_words]
    heat_df = pd.DataFrame(
        kmeans.cluster_centers_[:, word_idx],
        index=[f"Cluster {c}" for c in range(4)],
        columns=heat_words,
    )
    plt.figure(figsize=(12, 3.5))
    sns.heatmap(heat_df, cmap="YlGnBu", annot=False)
    plt.title("Top Words per Cluster (Centroid Weights)")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.show()

    # TODO 3.3: 2D projection
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X_tfidf)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1], c=df["cluster"], cmap="tab10", s=8, alpha=0.6
    )
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.title("Spatial Mapping of Posts (TruncatedSVD of TF-IDF)")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.show()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics

# %%
# Challenge 1 (Noise Filtering)
# TODO 4.1
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

custom_stops = list(ENGLISH_STOP_WORDS.union(
    {"know", "like", "just", "think", "does", "use", "thanks", "don", "ve", "ll"}
))
clean_vec = TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    min_df=5,
    max_features=4000,
    stop_words=custom_stops,
)
X_clean = clean_vec.fit_transform(df["text"])
km_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = km_clean.fit_predict(X_clean)
clean_terms = clean_vec.get_feature_names_out()
print("\n--- Cleaner vectorizer: top 8 words per cluster ---")
for c in range(4):
    idx = km_clean.cluster_centers_[c].argsort()[::-1][:8]
    print(f"Cluster {c}: {', '.join(clean_terms[idx])}")

# Challenge 2 (Hardware Overlap Diagnosis)
# TODO 4.2
print("\n--- Crosstab: true label vs baseline cluster ---")
print(pd.crosstab(df["true_label"], df["cluster"]))
print("\n--- Crosstab: true label vs cleaner cluster ---")
print(pd.crosstab(df["true_label"], df["cluster_clean"]))

# Bigram experiment
bi_vec = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 2), max_features=6000, min_df=5, max_df=0.5
)
X_bi = bi_vec.fit_transform(df["text"])
df["cluster_bigram"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_bi)
print("\n--- Crosstab: true label vs bigram cluster ---")
print(pd.crosstab(df["true_label"], df["cluster_bigram"]))

# Challenge 3 (Rant Length Bias)
# TODO 4.3
lengths = df["text"].str.split().str.len()
longest, shortest = int(lengths.idxmax()), int(lengths.idxmin())
for name, M in [("BoW", X_bow), ("TF-IDF", X_tfidf)]:
    print(
        f"{name}: longest post (id {longest}, {lengths[longest]} words) norm = "
        f"{np.linalg.norm(M[longest].toarray()):.3f} | shortest post "
        f"(id {shortest}, {lengths[shortest]} words) norm = "
        f"{np.linalg.norm(M[shortest].toarray()):.3f}"
    )


# Challenge 4 (Stakeholder Explainability Audit)
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """Print and return the top_n highest-weighted TF-IDF terms for one document."""
    row = matrix[doc_id].toarray().ravel()
    terms = vectorizer.get_feature_names_out()
    top = row.argsort()[::-1][:top_n]
    result = [(terms[i], float(row[i])) for i in top if row[i] > 0]
    print(f"\nDocument #{doc_id} top {top_n} TF-IDF terms:")
    for term, w in result:
        print(f"  {term:<15} {w:.4f}")
    return result


explain_document(412, X_tfidf, tfidf_vec, top_n=5)
print("Doc 412 true label:", df["true_label"].iloc[412], "| cluster:", df["cluster"].iloc[412])
