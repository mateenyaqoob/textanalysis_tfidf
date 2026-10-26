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

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load dataset
CATEGORIES = [
    "sci.electronics",
    "rec.autos",
    "comp.sys.mac.hardware",
    "sci.med"
]

print("Loading dataset...")

raw_data = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes")
)

df = pd.DataFrame({
    "text": raw_data.data,
    "true_label": [raw_data.target_names[i] for i in raw_data.target]
})

# Keep meaningful posts
df = df[df["text"].str.strip().str.len() > 50].reset_index(drop=True)

print(f"Loaded {len(df)} unstructured user posts.")

# Task 1: Text Cleaning and Normalization

sample_post = df["text"].iloc[0][:400]

print("--- Raw Sample Post ---")
print(sample_post)
print("-" * 40)

# TODO 1.1: Tokenization and lowercasing
tokens = [
    w.lower()
    for w in word_tokenize(sample_post)
    if w.isalpha()
]

# TODO 1.2: Remove stopwords
stop_words = set(stopwords.words("english"))

filtered_tokens = [
    w for w in tokens
    if w not in stop_words
]

# TODO 1.3: Stemming and Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]

lemmatized_tokens = [
    lemmatizer.lemmatize(w, pos="v")
    for w in filtered_tokens
]

print("Filtered Tokens:  ", filtered_tokens[:12])
print("Stemmed Roots:    ", stemmed_tokens[:12])
print("Lemmatized Words: ", lemmatized_tokens[:12])


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

# TODO 2.2: Fit and transform
X_bow = bow_vec.fit_transform(df["text"])
X_tfidf = tfidf_vec.fit_transform(df["text"])

print("BoW Matrix Shape:", X_bow.shape)
print("TF-IDF Matrix Shape:", X_tfidf.shape)

# TODO 2.3: Calculate sparsity
total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]

sparsity = 100.0 * (1.0 - X_tfidf.nnz / total_elements)

print(f"TF-IDF Sparsity: {sparsity:.2f}%")

# TODO 2.4: Top 10 TF-IDF words
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

# Task 3: Automated Topic Discovery

# TODO 3.1: Apply K-Means clustering
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_tfidf)

# Print top 8 words per cluster
feature_names = tfidf_vec.get_feature_names_out()
centers = kmeans.cluster_centers_

print("--- Top 8 Words per Cluster ---")

for i, center in enumerate(centers):
    top_indices = center.argsort()[-8:][::-1]
    print(f"Cluster {i}:", ", ".join(feature_names[top_indices]))


# TODO 3.2: Cluster importance heatmap

# Collect top 5 features from each cluster
top_indices = []

for center in centers:
    indices = center.argsort()[-5:][::-1]
    top_indices.extend(indices)

# Remove duplicate feature indices
top_indices = list(dict.fromkeys(top_indices))

heatmap_data = pd.DataFrame(
    centers[:, top_indices],
    columns=feature_names[top_indices],
    index=[f"Cluster {i}" for i in range(4)]
)

plt.figure(figsize=(12, 5))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu"
)

plt.xlabel("Important Words")
plt.ylabel("Clusters")
plt.title("Cluster-Word Importance Heatmap")

plt.tight_layout()
plt.savefig("cluster_heatmap.png", dpi=150)
plt.show()


# TODO 3.3: Spatial mapping using TruncatedSVD

svd = TruncatedSVD(
    n_components=2,
    random_state=42
)

X_2d = svd.fit_transform(X_tfidf)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=X_2d[:, 0],
    y=X_2d[:, 1],
    hue=df["cluster"],
    palette="Set2",
    alpha=0.6,
    s=20
)

plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.title("2D Spatial Mapping of Document Clusters")

plt.tight_layout()
plt.savefig("spatial_mapping.png", dpi=150)
plt.show()


# Custom conversational stopwords
custom_words = {
    "know", "like", "think", "just",
    "does", "thanks", "use"
}

# Combine with standard English stopwords
all_stopwords = set(stopwords.words("english")).union(custom_words)

# Improved TF-IDF Vectorizer
clean_tfidf_vec = TfidfVectorizer(
    stop_words=list(all_stopwords),
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    max_features=4000,
    min_df=5
)

X_clean = clean_tfidf_vec.fit_transform(df["text"])

# Refit K-Means
clean_kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clean_clusters = clean_kmeans.fit_predict(X_clean)

clean_features = clean_tfidf_vec.get_feature_names_out()

# Compare original vs cleaned cluster words
print("--- Original Cluster Words ---")

for i, center in enumerate(kmeans.cluster_centers_):
    idx = center.argsort()[-8:][::-1]
    print(f"Cluster {i}:", ", ".join(feature_names[idx]))

print("\n--- Cleaned Cluster Words ---")

for i, center in enumerate(clean_kmeans.cluster_centers_):
    idx = center.argsort()[-8:][::-1]
    print(f"Cluster {i}:", ", ".join(clean_features[idx]))



print("--- Original Cluster Comparison ---")

original_table = pd.crosstab(
    df["true_label"],
    df["cluster"]
)

print(original_table)

# Inspect common cluster assignments
print("\nElectronics vs Mac Hardware:")

print(original_table.loc[
    ["sci.electronics", "comp.sys.mac.hardware"]
])

# Find cluster with highest combined presence of both categories
hardware_counts = original_table.loc[
    ["sci.electronics", "comp.sys.mac.hardware"]
]

overlap_cluster = hardware_counts.min(axis=0).idxmax()

print("\nCluster with strongest shared hardware presence:", overlap_cluster)

# Show sample posts from both categories in that cluster
for category in ["sci.electronics", "comp.sys.mac.hardware"]:
    samples = df[
        (df["true_label"] == category) &
        (df["cluster"] == overlap_cluster)
    ]["text"].head(2)

    print(f"\n--- {category} ---")
    for post in samples:
        print(post[:400].replace("\n", " "))
        print("-" * 40)

# Apply bigrams
bigram_vec = TfidfVectorizer(
    stop_words="english",
    max_features=4000,
    min_df=5,
    ngram_range=(1, 2)
)

X_bigram = bigram_vec.fit_transform(df["text"])

bigram_kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["bigram_cluster"] = bigram_kmeans.fit_predict(X_bigram)

print("\n--- After Adding Bigrams ---")

bigram_table = pd.crosstab(
    df["true_label"],
    df["bigram_cluster"]
)

print(bigram_table)

# Print top words and phrases
bigram_features = bigram_vec.get_feature_names_out()

print("\nTop Features After Bigrams:")

for i, center in enumerate(bigram_kmeans.cluster_centers_):
    idx = center.argsort()[-8:][::-1]
    print(f"Cluster {i}:", ", ".join(bigram_features[idx]))

# Check technical words shared by both hardware categories

technical_words = [
    "drive", "monitor", "card", "power", "disk",
    "board", "cable", "chip", "port", "memory",
    "hardware", "computer", "device", "controller"
]

for word in technical_words:
    electronics_count = df[
        df["true_label"] == "sci.electronics"
    ]["text"].str.contains(
        rf"\b{word}\b", case=False, regex=True
    ).sum()

    mac_count = df[
        df["true_label"] == "comp.sys.mac.hardware"
    ]["text"].str.contains(
        rf"\b{word}\b", case=False, regex=True
    ).sum()

    if electronics_count > 0 and mac_count > 0:
        print(f"{word}: Electronics={electronics_count}, Mac={mac_count}")


# Calculate word count per document
word_counts = df["text"].str.split().str.len()

# Identify longest and shortest documents
longest_idx = word_counts.idxmax()
shortest_idx = word_counts.idxmin()

print("Longest Post Words:", word_counts[longest_idx])
print("Shortest Post Words:", word_counts[shortest_idx])

# Compare vector norms
for name, idx in [
    ("Longest Post", longest_idx),
    ("Shortest Post", shortest_idx)
]:
    bow_norm = np.linalg.norm(X_bow[idx].toarray())
    tfidf_norm = np.linalg.norm(X_tfidf[idx].toarray())

    print(f"\n{name}")
    print(f"BoW L2 Norm: {bow_norm:.4f}")
    print(f"TF-IDF L2 Norm: {tfidf_norm:.4f}")

def explain_document(doc_id, matrix, vectorizer, top_n=5):

    # Select document
    row = matrix.getrow(doc_id)

    # Get non-zero features and weights
    indices = row.indices
    weights = row.data

    # Find highest-weighted features
    top_positions = np.argsort(weights)[::-1][:top_n]

    feature_names = vectorizer.get_feature_names_out()

    explanation = [
        (feature_names[indices[i]], float(weights[i]))
        for i in top_positions
    ]

    return explanation


# Test Document 412
result = explain_document(
    doc_id=412,
    matrix=X_tfidf,
    vectorizer=tfidf_vec,
    top_n=5
)

print("--- Document 412 Explanation ---")

for word, weight in result:
    print(f"{word}: {weight:.4f}")

print("\nAssigned Cluster:", df.loc[412, "cluster"])
print("True Category:", df.loc[412, "true_label"])

