
"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques

This script covers:
1. Text cleaning, tokenization, stemming, and lemmatization
2. Bag of Words vs. TF-IDF feature extraction
3. K-Means clustering and topic visualization
4. Noise filtering
5. Hardware overlap diagnosis
6. Rant length bias
7. Stakeholder explainability audit

Run:
    python lab15_starter.py
"""

# ============================================================
# SETUP AND DATASET LOADING
# ============================================================

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


# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# Dataset categories
CATEGORIES = [
    "sci.electronics",
    "rec.autos",
    "comp.sys.mac.hardware",
    "sci.med",
]

print("=" * 65)
print("UNSTRUCTURED TEXT ANALYSIS AND AUTOMATED TRIAGE WITH TF-IDF")
print("=" * 65)

print("\nLoading dataset...")

raw_data = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
)

df = pd.DataFrame({
    "text": raw_data.data,
    "true_label": [
        raw_data.target_names[i]
        for i in raw_data.target
    ],
})

# Remove posts shorter than or equal to 50 characters
df = df[
    df["text"].str.strip().str.len() > 50
].reset_index(drop=True)

print(f"Loaded {len(df)} unstructured user posts.")

print("\nCategory distribution:")
print(df["true_label"].value_counts())


# ============================================================
# STEP 1: TEXT CLEANING, TOKENIZATION AND NORMALIZATION
# ============================================================

print("\n" + "=" * 65)
print("STEP 1: TEXT CLEANING AND NORMALIZATION")
print("=" * 65)

sample_post = df["text"].iloc[0][:400]

print("\n--- Raw Sample Post ---")
print(sample_post)
print("-" * 40)


# TODO 1.1: Tokenize, lowercase, and keep alphabetic tokens
tokens = [
    word.lower()
    for word in word_tokenize(sample_post)
    if word.isalpha()
]

print("\nOriginal alphabetic tokens:")
print(tokens[:20])


# TODO 1.2: Remove English stopwords
stop_words = set(stopwords.words("english"))

filtered_tokens = [
    word
    for word in tokens
    if word not in stop_words
]

print("\nFiltered tokens:")
print(filtered_tokens[:20])


# TODO 1.3: Apply stemming and lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [
    stemmer.stem(word)
    for word in filtered_tokens
]

lemmatized_tokens = [
    lemmatizer.lemmatize(word, pos="v")
    for word in filtered_tokens
]

print("\nFiltered Tokens:  ", filtered_tokens[:12])
print("Stemmed Roots:    ", stemmed_tokens[:12])
print("Lemmatized Words: ", lemmatized_tokens[:12])


# ============================================================
# STEP 2: BAG OF WORDS VS. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 65)
print("STEP 2: BAG OF WORDS VS. TF-IDF")
print("=" * 65)


# TODO 2.1: Initialize the vectorizers
bow_vec = CountVectorizer(
    stop_words="english",
    max_features=4000,
    min_df=5,
)

tfidf_vec = TfidfVectorizer(
    stop_words="english",
    max_features=4000,
    min_df=5,
)


# TODO 2.2: Fit and transform the dataset
X_bow = bow_vec.fit_transform(df["text"])

X_tfidf = tfidf_vec.fit_transform(df["text"])


print("\nBag of Words matrix shape:", X_bow.shape)
print("TF-IDF matrix shape:", X_tfidf.shape)


# TODO 2.3: Calculate TF-IDF matrix sparsity
total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]

sparsity = 100.0 * (
    1.0 - X_tfidf.nnz / total_elements
)

print(f"\nTF-IDF Matrix Shape: {X_tfidf.shape}")
print(f"TF-IDF Matrix Sparsity: {sparsity:.2f}%")


# TODO 2.4: Find and plot the top 10 average TF-IDF features
tfidf_means = np.asarray(
    X_tfidf.mean(axis=0)
).ravel()

top_idx = tfidf_means.argsort()[-10:]

top_words = tfidf_vec.get_feature_names_out()[top_idx]
top_scores = tfidf_means[top_idx]

plt.figure(figsize=(9, 5))

plt.barh(
    top_words,
    top_scores,
    color="teal",
)

plt.xlabel("Average TF-IDF Weight")
plt.ylabel("Word")

plt.title("Top 10 Most Influential Words (Baseline TF-IDF)")

plt.tight_layout()
plt.savefig("top_features.png", dpi=150)
plt.show()

print("\nSaved visualization: top_features.png")


# ============================================================
# STEP 3: AUTOMATED TOPIC DISCOVERY AND SPATIAL MAPPING
# ============================================================

print("\n" + "=" * 65)
print("STEP 3: K-MEANS CLUSTERING AND TOPIC DISCOVERY")
print("=" * 65)


# TODO 3.1: Fit K-Means with four clusters
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10,
)

df["cluster"] = kmeans.fit_predict(X_tfidf)

print("\nK-Means clustering completed.")

print("\nNumber of documents in each cluster:")
print(df["cluster"].value_counts().sort_index())


# Get feature names
feature_names = tfidf_vec.get_feature_names_out()


# TODO 3.2: Print the top eight defining words per cluster
print("\n--- Top 8 Keywords Per Cluster ---")

for cluster_id, centroid in enumerate(kmeans.cluster_centers_):

    top_indices = centroid.argsort()[-8:][::-1]

    top_cluster_words = feature_names[top_indices]

    print(f"\nCluster {cluster_id}:")
    print(", ".join(top_cluster_words))


# Build a heatmap using the top five words per cluster
cluster_top_words = {}

for cluster_id, centroid in enumerate(kmeans.cluster_centers_):

    top_indices = centroid.argsort()[-5:][::-1]

    cluster_top_words[cluster_id] = list(
        feature_names[top_indices]
    )


# Collect the union of top words from all clusters
heatmap_words = list(
    dict.fromkeys(
        word
        for words in cluster_top_words.values()
        for word in words
    )
)

# Get the centroid weights for those words
word_to_index = {
    word: index
    for index, word in enumerate(feature_names)
}

heatmap_data = np.array([
    [
        kmeans.cluster_centers_[cluster_id][
            word_to_index[word]
        ]
        for word in heatmap_words
    ]
    for cluster_id in range(4)
])

heatmap_df = pd.DataFrame(
    heatmap_data,
    index=[f"Cluster {i}" for i in range(4)],
    columns=heatmap_words,
)

plt.figure(figsize=(14, 5))

sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
)

plt.title("Top 5 Keywords Per Cluster: TF-IDF Centroid Weights")
plt.xlabel("Top Cluster Keywords")
plt.ylabel("K-Means Cluster")

plt.tight_layout()
plt.savefig("cluster_heatmap.png", dpi=150)
plt.show()

print("\nSaved visualization: cluster_heatmap.png")


# TODO 3.3: Project TF-IDF into two dimensions using TruncatedSVD
svd = TruncatedSVD(
    n_components=2,
    random_state=42,
)

X_2d = svd.fit_transform(X_tfidf)

df["svd_1"] = X_2d[:, 0]
df["svd_2"] = X_2d[:, 1]

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="svd_1",
    y="svd_2",
    hue="cluster",
    palette="tab10",
    alpha=0.65,
    s=35,
)

plt.title("2D Spatial Mapping of TF-IDF Document Clusters")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")

plt.legend(title="Cluster")
plt.tight_layout()

plt.savefig("spatial_mapping.png", dpi=150)
plt.show()

print("Saved visualization: spatial_mapping.png")

print(
    "\nExplained variance by the two SVD components:",
    f"{svd.explained_variance_ratio_.sum():.4f}",
)


# ============================================================
# STEP 4: PROBLEM-SOLVING DIAGNOSTICS
# ============================================================


# ------------------------------------------------------------
# CHALLENGE 1: NOISE FILTERING
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("CHALLENGE 1: NOISE FILTERING")
print("=" * 65)


# Define custom conversational stopwords
custom_stopwords = set(stopwords.words("english")).union({
    "know",
    "like",
    "just",
    "think",
    "does",
    "use",
    "thanks",
    "thank",
    "really",
    "going",
    "get",
    "would",
    "could",
    "also",
    "well",
    "one",
    "even",
    "much",
    "many",
    "something",
    "anything",
    "want",
    "said",
    "say",
    "make",
    "made",
    "thing",
    "things",
})


# Build a cleaner TF-IDF vectorizer
clean_tfidf_vec = TfidfVectorizer(
    stop_words=list(custom_stopwords),
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_features=4000,
    min_df=5,
    max_df=0.5,
)

# Fit and transform the original documents
X_clean_tfidf = clean_tfidf_vec.fit_transform(
    df["text"]
)

print("\nOriginal TF-IDF shape:", X_tfidf.shape)
print("Cleaned TF-IDF shape:", X_clean_tfidf.shape)

print(
    "Original vocabulary size:",
    len(tfidf_vec.get_feature_names_out()),
)

print(
    "Cleaned vocabulary size:",
    len(clean_tfidf_vec.get_feature_names_out()),
)


# Fit a new K-Means model on cleaned TF-IDF
clean_kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10,
)

df["clean_cluster"] = clean_kmeans.fit_predict(
    X_clean_tfidf
)


# Helper function for displaying cluster keywords
def print_cluster_keywords(model, vectorizer, top_n=10):
    """
    Print the top-weighted keywords for every K-Means cluster.
    """

    feature_names = vectorizer.get_feature_names_out()

    for cluster_id, centroid in enumerate(
        model.cluster_centers_
    ):

        top_indices = centroid.argsort()[-top_n:][::-1]

        top_words = feature_names[top_indices]

        print(f"\nCluster {cluster_id}:")
        print(", ".join(top_words))


print("\n--- Original TF-IDF Cluster Keywords ---")

print_cluster_keywords(
    kmeans,
    tfidf_vec,
    top_n=10,
)

print("\n--- Cleaned TF-IDF Cluster Keywords ---")

print_cluster_keywords(
    clean_kmeans,
    clean_tfidf_vec,
    top_n=10,
)


# Compare the vocabulary sizes and cluster assignments
print("\n--- Noise Filtering Comparison ---")

print(
    "Original number of features:",
    X_tfidf.shape[1],
)

print(
    "Cleaned number of features:",
    X_clean_tfidf.shape[1],
)

print(
    "Original TF-IDF nonzero entries:",
    X_tfidf.nnz,
)

print(
    "Cleaned TF-IDF nonzero entries:",
    X_clean_tfidf.nnz,
)

print("\nOriginal cluster sizes:")
print(df["cluster"].value_counts().sort_index())

print("\nCleaned cluster sizes:")
print(df["clean_cluster"].value_counts().sort_index())

print(
    "\nInterpretation: Compare the keywords and cluster sizes "
    "before and after filtering. Removing conversational noise "
    "may improve topic separation, but the outcome must be "
    "evaluated using the actual results."
)


# ------------------------------------------------------------
# CHALLENGE 2: HARDWARE OVERLAP DIAGNOSIS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("CHALLENGE 2: HARDWARE OVERLAP DIAGNOSIS")
print("=" * 65)


# Compare true categories with original K-Means clusters
confusion_table = pd.crosstab(
    df["true_label"],
    df["cluster"],
    rownames=["Actual Category"],
    colnames=["Predicted Cluster"],
)

print("\nOriginal TF-IDF: Actual Categories vs. Clusters")
print(confusion_table)


# Compare true categories with cleaned K-Means clusters
clean_confusion_table = pd.crosstab(
    df["true_label"],
    df["clean_cluster"],
    rownames=["Actual Category"],
    colnames=["Cleaned Cluster"],
)

print("\nCleaned TF-IDF: Actual Categories vs. Clusters")
print(clean_confusion_table)


# Plot the original category-cluster overlap
plt.figure(figsize=(9, 5))

sns.heatmap(
    confusion_table,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title("Actual Categories vs. Original K-Means Clusters")
plt.xlabel("K-Means Cluster")
plt.ylabel("Actual Category")

plt.tight_layout()
plt.savefig("category_cluster_overlap.png", dpi=150)
plt.show()


# Plot the cleaned category-cluster overlap
plt.figure(figsize=(9, 5))

sns.heatmap(
    clean_confusion_table,
    annot=True,
    fmt="d",
    cmap="Greens",
)

plt.title("Actual Categories vs. Cleaned K-Means Clusters")
plt.xlabel("Cleaned K-Means Cluster")
plt.ylabel("Actual Category")

plt.tight_layout()
plt.savefig("clean_category_cluster_overlap.png", dpi=150)
plt.show()

print(
    "\nSaved visualizations: category_cluster_overlap.png "
    "and clean_category_cluster_overlap.png"
)

print(
    "\nInterpretation: Large counts indicate where documents "
    "from a true category are concentrated. Categories appearing "
    "across several clusters may share vocabulary or may not "
    "be cleanly separated by the selected features."
)


# ------------------------------------------------------------
# CHALLENGE 3: RANT LENGTH BIAS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("CHALLENGE 3: RANT LENGTH BIAS")
print("=" * 65)


# Identify longest and shortest posts by character count
post_lengths = df["text"].str.len()

longest_idx = post_lengths.idxmax()
shortest_idx = post_lengths.idxmin()

print("\nLongest post index:", longest_idx)
print(
    "Longest post character count:",
    post_lengths.loc[longest_idx],
)

print("\nShortest post index:", shortest_idx)
print(
    "Shortest post character count:",
    post_lengths.loc[shortest_idx],
)


def get_l2_norm(matrix, row_idx):
    """
    Calculate the L2 norm of one document vector.

    Only the selected sparse row is converted to a dense array.
    """

    row = matrix[row_idx].toarray().ravel()

    return np.linalg.norm(row)


# Bag of Words L2 norms
bow_longest_norm = get_l2_norm(
    X_bow,
    longest_idx,
)

bow_shortest_norm = get_l2_norm(
    X_bow,
    shortest_idx,
)


# TF-IDF L2 norms
tfidf_longest_norm = get_l2_norm(
    X_tfidf,
    longest_idx,
)

tfidf_shortest_norm = get_l2_norm(
    X_tfidf,
    shortest_idx,
)


# Create a comparison table
norm_comparison = pd.DataFrame({
    "Document": [
        "Longest Post",
        "Shortest Post",
    ],

    "Character Length": [
        post_lengths.loc[longest_idx],
        post_lengths.loc[shortest_idx],
    ],

    "Bag of Words L2 Norm": [
        bow_longest_norm,
        bow_shortest_norm,
    ],

    "TF-IDF L2 Norm": [
        tfidf_longest_norm,
        tfidf_shortest_norm,
    ],
})

print("\nL2 Norm Comparison:")
print(norm_comparison.to_string(index=False))


print("\n--- Interpretation ---")

if bow_longest_norm > bow_shortest_norm:

    print(
        "The longest post has a larger Bag of Words L2 norm. "
        "Its vector contains more or larger term counts under "
        "the selected vocabulary."
    )

else:

    print(
        "The longest post does not have a larger Bag of Words "
        "L2 norm. Character length alone does not determine "
        "the number or distribution of vocabulary terms."
    )


if tfidf_longest_norm > 0 and tfidf_shortest_norm > 0:

    print(
        "Both TF-IDF vectors have L2 norms approximately equal "
        "to 1 because TfidfVectorizer applies L2 normalization "
        "by default."
    )

elif tfidf_longest_norm == 0 or tfidf_shortest_norm == 0:

    print(
        "At least one TF-IDF vector is zero. This can happen "
        "when a document contains no terms from the fitted "
        "vocabulary after preprocessing."
    )


print(
    "\nConclusion: Bag of Words counts are influenced by term "
    "frequency, while normalized TF-IDF vectors reduce the "
    "direct effect of document length on vector magnitude."
)


# ------------------------------------------------------------
# CHALLENGE 4: STAKEHOLDER EXPLAINABILITY AUDIT
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("CHALLENGE 4: STAKEHOLDER EXPLAINABILITY AUDIT")
print("=" * 65)


def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    Display and return the top_n highest-weighted terms
    for one document.

    Parameters
    ----------
    doc_id : int
        Zero-based row index of the document.
    matrix : sparse matrix
        Document-term matrix, such as X_tfidf.
    vectorizer : fitted vectorizer
        Vectorizer used to create the matrix.
    top_n : int
        Number of nonzero terms to display.

    Returns
    -------
    pandas.DataFrame
        Terms and exact weights sorted in descending order.
    """

    # Validate the document index
    if not isinstance(doc_id, (int, np.integer)):
        raise TypeError("doc_id must be an integer.")

    if doc_id < 0 or doc_id >= matrix.shape[0]:
        raise IndexError(
            f"doc_id must be between 0 and {matrix.shape[0] - 1}."
        )

    # Extract the document vector
    document_vector = matrix[doc_id].toarray().ravel()

    # Get the vocabulary
    feature_names = vectorizer.get_feature_names_out()

    # Find nonzero features
    nonzero_indices = np.flatnonzero(
        document_vector > 0
    )

    if len(nonzero_indices) == 0:

        print(f"\nDocument {doc_id} has no nonzero features.")

        return pd.DataFrame(
            columns=["Term", "Weight"]
        )

    # Sort nonzero features by descending weight
    sorted_indices = nonzero_indices[
        np.argsort(
            document_vector[nonzero_indices]
        )[::-1]
    ]

    # Select the top N terms
    top_indices = sorted_indices[:top_n]

    # Construct the explanation table
    explanation = pd.DataFrame({
        "Term": feature_names[top_indices],

        "Weight": document_vector[top_indices],
    })

    print(f"\nDocument ID: {doc_id}")

    print(
        "Document length:",
        len(df.iloc[doc_id]["text"]),
        "characters",
    )

    print(
        "Actual category:",
        df.iloc[doc_id]["true_label"],
    )

    if "cluster" in df.columns:

        print(
            "Original K-Means cluster:",
            df.iloc[doc_id]["cluster"],
        )

    if "clean_cluster" in df.columns:

        print(
            "Cleaned K-Means cluster:",
            df.iloc[doc_id]["clean_cluster"],
        )

    print("\nOriginal document excerpt:")
    print(df.iloc[doc_id]["text"][:500])

    print(f"\nTop {len(explanation)} TF-IDF terms:")

    print(
        explanation.to_string(
            index=False,
            formatters={
                "Weight": lambda value: f"{value:.6f}"
            },
        )
    )

    return explanation


# Test with document ID 412
explanation_412 = explain_document(
    doc_id=412,
    matrix=X_tfidf,
    vectorizer=tfidf_vec,
    top_n=5,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("LAB COMPLETED")
print("=" * 65)

print("\nDataset size:", len(df))

print("Original TF-IDF features:", X_tfidf.shape[1])

print("Cleaned TF-IDF features:", X_clean_tfidf.shape[1])

print("\nGenerated files:")
print("1. top_features.png")
print("2. cluster_heatmap.png")
print("3. spatial_mapping.png")
print("4. category_cluster_overlap.png")
print("5. clean_category_cluster_overlap.png")

print("\nAll lab steps and four diagnostic challenges completed.")


"""
Automated Pytest Suite for Lab 15: TF-IDF & Unstructured Data Triage
This script validates student implementations in student_starter_script.py.
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
