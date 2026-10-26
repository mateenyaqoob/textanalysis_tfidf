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


def show_plot():
    """Show the figure when the script is run directly; just close it when the
    file is imported by pytest (so the autograder never blocks on a plot window)."""
    if __name__ == "__main__":
        plt.show()
    plt.close()


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

# TODO 1.1: Tokenize, lowercase, keep only alphabetic tokens
tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]

# TODO 1.2: Remove standard English stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]

# TODO 1.3: PorterStemmer vs WordNetLemmatizer (pos='v') on the same tokens
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
# TODO 2.1: Initialize vectorizers
bow_vec = CountVectorizer(stop_words="english", max_features=4000, min_df=5)
tfidf_vec = TfidfVectorizer(stop_words="english", max_features=4000, min_df=5)

# TODO 2.2: Fit and transform the corpus
X_bow = bow_vec.fit_transform(df["text"])
X_tfidf = tfidf_vec.fit_transform(df["text"])

# TODO 2.3: Matrix sparsity (% of zero entries in X_tfidf)
if X_tfidf is not None:
    total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
    sparsity = 100.0 * (1.0 - (X_tfidf.nnz / total_elements))
    print(f"\nBoW Matrix Shape: {X_bow.shape}")
    print(f"TF-IDF Matrix Shape: {X_tfidf.shape} | Sparsity: {sparsity:.2f}%")

    # TODO 2.4: Top 10 words by average TF-IDF score
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
    show_plot()


# %% [markdown]
# ## Step 3: Automated Topic Discovery & Spatial Mapping (15 mins)

# %%
def top_terms_per_cluster(model, vectorizer, n_terms=8):
    """Return {cluster_id: [top n_terms words]} using the cluster centroids."""
    terms = vectorizer.get_feature_names_out()
    order = model.cluster_centers_.argsort()[:, ::-1]
    return {c: [terms[i] for i in order[c, :n_terms]] for c in range(model.n_clusters)}


if X_tfidf is not None:
    # TODO 3.1: KMeans on the TF-IDF matrix
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)
    print("\nCluster sizes:", df["cluster"].value_counts().sort_index().to_dict())

    # TODO 3.2: Top 8 words per cluster + heatmap of top 5 words per cluster
    feature_names = tfidf_vec.get_feature_names_out()
    top8 = top_terms_per_cluster(kmeans, tfidf_vec, n_terms=8)
    print("\nTop 8 TF-IDF words per cluster (baseline):")
    for c, words in top8.items():
        print(f"  Cluster {c}: {', '.join(words)}")

    heat_words = []
    for c in range(kmeans.n_clusters):
        for w in top8[c][:5]:
            if w not in heat_words:
                heat_words.append(w)
    word_idx = [list(feature_names).index(w) for w in heat_words]
    heat_df = pd.DataFrame(
        kmeans.cluster_centers_[:, word_idx],
        index=[f"Cluster {c}" for c in range(kmeans.n_clusters)],
        columns=heat_words,
    )
    plt.figure(figsize=(12, 3.8))
    sns.heatmap(heat_df, cmap="YlGnBu", annot=True, fmt=".3f", cbar_kws={"label": "Centroid TF-IDF weight"})
    plt.title("TF-IDF Word Importance per Cluster (top 5 words each)")
    plt.xlabel("Word")
    plt.ylabel("Cluster")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    show_plot()

    # TODO 3.3: 2D projection with TruncatedSVD (works directly on sparse matrices)
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X_tfidf)
    df["svd_x"], df["svd_y"] = coords[:, 0], coords[:, 1]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df["svd_x"], df["svd_y"], c=df["cluster"], cmap="tab10", s=8, alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.title("Spatial Document Mapping (TF-IDF + TruncatedSVD, coloured by K-Means cluster)")
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    show_plot()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %%
# Challenge 1 (Noise Filtering):
# TODO 4.1: Cleaner TF-IDF: 3+ letter alphabetic tokens, max_df=0.5, custom filler stopwords
filler_words = {"know", "like", "just", "think", "does", "use", "thanks",
                "don", "want", "good", "used", "need", "way", "make", "really"}
# Signature / e-mail noise found in the baseline clusters (a repeated sci.med sign-off
# "Gordon Banks N3JXP ... Skepticism is the chastity of the intellect ..." forms its own cluster)
signature_noise = {"geb", "pitt", "dsl", "cadre", "jxp", "chastity", "shameful", "surrender",
                   "skepticism", "intellect", "gordon", "banks", "edu", "com"}
custom_stop_words = list(ENGLISH_STOP_WORDS.union(filler_words).union(signature_noise))

tfidf_clean_vec = TfidfVectorizer(
    stop_words=custom_stop_words,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    max_features=4000,
    min_df=5,
)
X_tfidf_clean = tfidf_clean_vec.fit_transform(df["text"])
kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = kmeans_clean.fit_predict(X_tfidf_clean)

print("\nTop 8 words per cluster (CLEANED vocabulary):")
for c, words in top_terms_per_cluster(kmeans_clean, tfidf_clean_vec, 8).items():
    print(f"  Cluster {c}: {', '.join(words)}")

# Document frequency of the filler words in the raw corpus (why they get weight)
_filler_df = CountVectorizer(vocabulary=sorted(filler_words), binary=True).fit_transform(df["text"])
filler_doc_pct = pd.Series(100 * np.asarray(_filler_df.sum(axis=0)).ravel() / len(df),
                           index=sorted(filler_words)).round(1).sort_values(ascending=False)
print("\n% of posts containing each filler word:\n", filler_doc_pct.to_string())


# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Compare clusters with the hidden ground truth
ct_baseline = pd.crosstab(df["true_label"], df["cluster"])
ct_clean = pd.crosstab(df["true_label"], df["cluster_clean"])
print("\nCrosstab (baseline TF-IDF clusters):\n", ct_baseline)
print("\nCrosstab (cleaned TF-IDF clusters):\n", ct_clean)


def purity(ct):
    """Share of posts that fall in the majority true label of their cluster."""
    return ct.max(axis=0).sum() / ct.values.sum()


# Shared technical words between electronics and Mac hardware posts
elec = df["true_label"] == "sci.electronics"
mac = df["true_label"] == "comp.sys.mac.hardware"
clean_terms = tfidf_clean_vec.get_feature_names_out()
elec_mean = np.asarray(X_tfidf_clean[elec.values].mean(axis=0)).ravel()
mac_mean = np.asarray(X_tfidf_clean[mac.values].mean(axis=0)).ravel()
shared_score = np.minimum(elec_mean, mac_mean)          # high in BOTH groups
shared_words = [clean_terms[i] for i in shared_score.argsort()[::-1][:10]]
print("\nWords strongly shared by sci.electronics and comp.sys.mac.hardware:", shared_words)

# Electronics posts that landed in the cluster dominated by Mac hardware
mac_cluster = ct_clean.loc["comp.sys.mac.hardware"].idxmax()
misclustered = df[elec & (df["cluster_clean"] == mac_cluster)]
print(f"Electronics posts placed in the Mac-hardware cluster ({mac_cluster}): {len(misclustered)}")

# Bigrams: phrases such as 'power supply', 'hard drive', 'apple ii' become features
tfidf_bigram_vec = TfidfVectorizer(
    stop_words=custom_stop_words,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    ngram_range=(1, 2),
    max_df=0.5,
    max_features=8000,
    min_df=5,
)
X_tfidf_bigram = tfidf_bigram_vec.fit_transform(df["text"])
kmeans_bigram = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_bigram"] = kmeans_bigram.fit_predict(X_tfidf_bigram)
ct_bigram = pd.crosstab(df["true_label"], df["cluster_bigram"])
bigram_terms = [t for t in tfidf_bigram_vec.get_feature_names_out() if " " in t]
print("\nCrosstab (unigrams + bigrams):\n", ct_bigram)
print(f"\nCluster purity -> baseline: {purity(ct_baseline):.3f} | cleaned: {purity(ct_clean):.3f} "
      f"| cleaned + bigrams: {purity(ct_bigram):.3f}")
print("Example bigram features:", bigram_terms[:15])
print("\nTop 8 words per cluster (unigrams + bigrams):")
for c, words in top_terms_per_cluster(kmeans_bigram, tfidf_bigram_vec, 8).items():
    print(f"  Cluster {c}: {', '.join(words)}")


# Challenge 3 (Rant Length Bias):
# TODO 4.3: L2 norms of the longest and shortest post in X_bow vs X_tfidf
text_len = df["text"].str.len()
longest_id, shortest_id = int(text_len.idxmax()), int(text_len.idxmin())


def row_norm(matrix, i):
    return float(np.linalg.norm(matrix[i].toarray()))


norm_table = pd.DataFrame(
    {
        "characters": [text_len[longest_id], text_len[shortest_id]],
        "BoW L2 norm": [row_norm(X_bow, longest_id), row_norm(X_bow, shortest_id)],
        "TF-IDF L2 norm": [row_norm(X_tfidf, longest_id), row_norm(X_tfidf, shortest_id)],
    },
    index=[f"longest post (#{longest_id})", f"shortest post (#{shortest_id})"],
)
print("\nLength bias check:\n", norm_table.round(3))


# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` as a list of (term, weight) tuples.
    """
    row = matrix[doc_id].toarray().ravel()
    terms = vectorizer.get_feature_names_out()
    nonzero = np.flatnonzero(row)
    ranked = nonzero[np.argsort(row[nonzero])[::-1]][:top_n]
    explanation = [(str(terms[i]), round(float(row[i]), 4)) for i in ranked]

    print(f"\nDocument #{doc_id} | true label: {df.loc[doc_id, 'true_label']} | "
          f"assigned cluster: {df.loc[doc_id, 'cluster']}")
    for term, weight in explanation:
        print(f"  {term:<15} {weight:.4f}")
    return explanation


if X_tfidf.shape[0] > 412:
    explanation_412 = explain_document(412, X_tfidf, tfidf_vec, top_n=5)
