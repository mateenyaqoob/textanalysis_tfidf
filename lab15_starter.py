"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques
"""

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

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

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

df = df[df["text"].str.strip().str.len() > 50].reset_index(drop=True)
print(f"Loaded {len(df)} unstructured user posts.\n")


# %% Step 1: Text Cleaning, Tokenization & Normalization
sample_post = df["text"].iloc[0][:400]
print("--- Raw Sample Post ---")
print(sample_post)
print("-" * 40)

tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]

stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(w, pos="v") for w in filtered_tokens]

if filtered_tokens is not None:
    print("Filtered Tokens:  ", filtered_tokens[:12])
    print("Stemmed Roots:    ", stemmed_tokens[:12])
    print("Lemmatized Words: ", lemmatized_tokens[:12])


# %% Step 2: Bag of Words vs. TF-IDF Feature Extraction
bow_vec = CountVectorizer(stop_words="english", max_features=4000, min_df=5)
tfidf_vec = TfidfVectorizer(stop_words="english", max_features=4000, min_df=5)

X_bow = bow_vec.fit_transform(df["text"])
X_tfidf = tfidf_vec.fit_transform(df["text"])

if X_tfidf is not None:
    total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
    sparsity = 100.0 * (1.0 - (X_tfidf.nnz / total_elements))
    print(f"\nTF-IDF Matrix Shape: {X_tfidf.shape} | Sparsity: {sparsity:.2f}%")

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


# %% Step 3: Automated Topic Discovery & Spatial Mapping
if X_tfidf is not None:
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    terms = tfidf_vec.get_feature_names_out()
    order = kmeans.cluster_centers_.argsort()[:, ::-1]

    for c in range(4):
        print(f"Cluster {c}: {', '.join(terms[i] for i in order[c, :8])}")

    heat_idx = []
    for c in range(4):
        for i in order[c, :5]:
            if i not in heat_idx:
                heat_idx.append(i)

    heat_df = pd.DataFrame(
        kmeans.cluster_centers_[:, heat_idx].T,
        index=terms[heat_idx],
        columns=[f"Cluster {c}" for c in range(4)],
    )
    plt.figure(figsize=(7, 8))
    sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Top Words per Cluster Centroid")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.show()

    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X_tfidf)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=df["cluster"], cmap="tab10", s=8, alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.title("Spatial Mapping of Posts (TF-IDF -> SVD 2D)")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.show()


# %% Step 4: Problem-Solving Diagnostics

# Challenge 1 (Noise Filtering)
custom_stops = list(
    ENGLISH_STOP_WORDS.union(
        {"know", "like", "just", "think", "does", "use", "thanks", "thank", "good",
         "don", "ve", "doesn", "didn", "really", "want", "need", "got", "getting",
         "say", "make", "going", "way", "used", "using", "anybody", "anyone", "help"}
    )
)

tfidf_clean = TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    stop_words=custom_stops,
    max_df=0.5,
    min_df=5,
    max_features=4000,
)
X_clean = tfidf_clean.fit_transform(df["text"])

kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = kmeans_clean.fit_predict(X_clean)

terms_clean = tfidf_clean.get_feature_names_out()
order_clean = kmeans_clean.cluster_centers_.argsort()[:, ::-1]
print("\n=== Refined vectorizer: top words per cluster ===")
for c in range(4):
    print(f"Cluster {c}: {', '.join(terms_clean[i] for i in order_clean[c, :8])}")

if "know" in bow_vec.vocabulary_:
    share = (X_bow[:, bow_vec.vocabulary_["know"]] > 0).sum() / X_bow.shape[0]
    print(f"\n'know' appears in {share:.1%} of posts")


# Challenge 2 (Hardware Overlap Diagnosis)
ct = pd.crosstab(df["true_label"], df["cluster"])
print("\n=== Crosstab ===")
print(ct)

cluster_major = ct.idxmax(axis=0)
print("\nMajority label per cluster:\n", cluster_major)

elec_in_mac = df[
    (df["true_label"] == "sci.electronics")
    & (df["cluster"].map(cluster_major) == "comp.sys.mac.hardware")
]
mac_in_elec = df[
    (df["true_label"] == "comp.sys.mac.hardware")
    & (df["cluster"].map(cluster_major) == "sci.electronics")
]
mis_idx = list(elec_in_mac.index) + list(mac_in_elec.index)
print(f"\nMisclustered electronics<->mac posts: {len(mis_idx)}")

if mis_idx:
    mis_means = np.asarray(X_tfidf[mis_idx].mean(axis=0)).ravel()
    print("Top shared words in misclustered posts:",
          list(tfidf_vec.get_feature_names_out()[mis_means.argsort()[-12:][::-1]]))
    for i in mis_idx[:3]:
        print("\n--- sample misclustered post ---")
        print(df.loc[i, "true_label"], "|", df.loc[i, "text"][:250].replace("\n", " "))

tfidf_bi = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 2), max_df=0.5, min_df=5, max_features=6000
)
X_bi = tfidf_bi.fit_transform(df["text"])
kmeans_bi = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_bi"] = kmeans_bi.fit_predict(X_bi)
print("\n=== Crosstab with bigrams ===")
print(pd.crosstab(df["true_label"], df["cluster_bi"]))

terms_bi = tfidf_bi.get_feature_names_out()
order_bi = kmeans_bi.cluster_centers_.argsort()[:, ::-1]
for c in range(4):
    print(f"Cluster {c}: {', '.join(terms_bi[i] for i in order_bi[c, :8])}")


# Challenge 3 (Rant Length Bias)
lengths = df["text"].str.len()
nonempty = np.asarray((X_bow.sum(axis=1) > 0)).ravel()

longest_i = int(lengths.idxmax())
shortest_i = int(lengths[nonempty].idxmin())


def row_norm(M, i):
    return float(np.linalg.norm(M[i].toarray()))


print(f"\nLongest post  (#{longest_i}, {lengths[longest_i]} chars)")
print(f"Shortest post (#{shortest_i}, {lengths[shortest_i]} chars)")
print(f"BoW   L2 norm -> longest: {row_norm(X_bow, longest_i):.3f} | shortest: {row_norm(X_bow, shortest_i):.3f}")
print(f"TFIDF L2 norm -> longest: {row_norm(X_tfidf, longest_i):.3f} | shortest: {row_norm(X_tfidf, shortest_i):.3f}")


# Challenge 4 (Stakeholder Explainability Audit)
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """Return (and print) the top_n non-zero TF-IDF terms and exact weights for one document."""
    row = matrix[doc_id].tocoo()
    terms = vectorizer.get_feature_names_out()
    pairs = sorted(zip(terms[row.col], row.data), key=lambda p: p[1], reverse=True)[:top_n]
    result = [(str(t), float(w)) for t, w in pairs]

    print(f"\nDocument #{doc_id} - top {top_n} TF-IDF features:")
    for t, w in result:
        print(f"  {t:<15} {w:.4f}")
    return result


if len(df) > 412:
    print(f"\nTrue label: {df.loc[412, 'true_label']} | Assigned cluster: {df.loc[412, 'cluster']}")
    explain_document(doc_id=412, matrix=X_tfidf, vectorizer=tfidf_vec, top_n=5)
    