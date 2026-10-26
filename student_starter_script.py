"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques

Instructions:
Complete each TODO section below. You can run this file interactively cell-by-cell
in VS Code / PyCharm / Spyder (using the `# %%` markers) or directly via terminal:
    python student_starter_script.py
"""


# ## Setup & Dataset Loading

# %%
import sys

import matplotlib

# Use a non-blocking backend outside notebooks / interactive windows so that
# `python student_starter_script.py` and `pytest` (which imports this module) never hang on plt.show().
INTERACTIVE = "ipykernel" in sys.modules
if not INTERACTIVE:
    matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources quietly
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


def show_plot():
    """Display the figure in interactive sessions; otherwise just release it."""
    if INTERACTIVE:
        plt.show()
    else:
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

    # Side-by-side view of where the stemmer and lemmatizer disagree
    diffs = [(w, s, l) for w, s, l in zip(filtered_tokens, stemmed_tokens, lemmatized_tokens) if s != l]
    print("\nStemmer vs. Lemmatizer disagreements (word -> stem | lemma):")
    for w, s, l in diffs[:10]:
        print(f"  {w:<15} -> {s:<12} | {l}")


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
    print(f"BoW Matrix Shape:    {X_bow.shape} | Non-zeros: {X_bow.nnz}")

    # TODO 2.4: Compute mean score per feature across documents and plot Top 10 TF-IDF words
    tfidf_means = np.asarray(X_tfidf.mean(axis=0)).ravel()
    top_idx = tfidf_means.argsort()[-10:]
    top_words = tfidf_vec.get_feature_names_out()[top_idx]
    top_scores = tfidf_means[top_idx]

    print("\nTop 10 words by average TF-IDF weight:")
    for w, s in zip(top_words[::-1], top_scores[::-1]):
        print(f"  {w:<12} {s:.4f}")

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
def top_terms_per_cluster(model, feature_names, n=8):
    """Return {cluster_id: [top-n terms]} ranked by centroid TF-IDF weight."""
    order = model.cluster_centers_.argsort()[:, ::-1]
    return {c: [feature_names[i] for i in order[c, :n]] for c in range(model.n_clusters)}


if X_tfidf is not None:
    # TODO 3.1: Fit KMeans with `n_clusters=4, random_state=42, n_init=10` on `X_tfidf`
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    # TODO 3.2: Print the top 8 defining TF-IDF words per cluster centroid
    # and plot a seaborn heatmap (`cluster_heatmap.png`) of the top 5 words per cluster.
    feature_names = tfidf_vec.get_feature_names_out()
    baseline_terms = top_terms_per_cluster(kmeans, feature_names, n=8)

    print("\nBaseline cluster sizes:", df["cluster"].value_counts().sort_index().to_dict())
    print("Top 8 terms per cluster (baseline TF-IDF):")
    for c, terms in baseline_terms.items():
        print(f"  Cluster {c}: {', '.join(terms)}")

    # Union of each cluster's top-5 terms -> rows = clusters, cols = words, values = centroid weight
    heatmap_idx = []
    for c in range(kmeans.n_clusters):
        for i in kmeans.cluster_centers_[c].argsort()[::-1][:5]:
            if i not in heatmap_idx:
                heatmap_idx.append(i)
    heatmap_df = pd.DataFrame(
        kmeans.cluster_centers_[:, heatmap_idx],
        index=[f"Cluster {c}" for c in range(kmeans.n_clusters)],
        columns=feature_names[heatmap_idx],
    )

    plt.figure(figsize=(12, 3.8))
    sns.heatmap(heatmap_df, cmap="YlGnBu", annot=True, fmt=".3f", cbar_kws={"label": "Centroid TF-IDF weight"})
    plt.title("Cluster-by-Word Importance (Top 5 Words per Cluster)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    show_plot()

    # TODO 3.3: Project `X_tfidf` to 2 dimensions using `TruncatedSVD(n_components=2, random_state=42)`
    # and save a 2D scatter plot (`spatial_mapping.png`) colored by `df["cluster"]`.
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_tfidf)
    print(f"\nTruncatedSVD explained variance (2 comps): {svd.explained_variance_ratio_.sum():.4f}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster"], cmap="tab10", s=8, alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.title("2D Spatial Mapping of Posts (TF-IDF -> TruncatedSVD)")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    show_plot()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %%
# Challenge 1 (Noise Filtering):
# TODO 4.1: Build a cleaner `TfidfVectorizer` using `token_pattern=r'\b[a-zA-Z]{3,}\b'`,
# `max_df=0.5`, and custom conversational stopwords (e.g., 'know', 'like', 'just', 'think', 'does', 'use', 'thanks').
# Re-fit KMeans and compare cluster keywords!
print("\n" + "=" * 60)
print("CHALLENGE 1: Conversational filler & numeric noise")
print("=" * 60)

# Why `know` still scores: idf = ln((1+N)/(1+df)) + 1 (sklearn smooth_idf) -> at df = 25% of N it is ~2.39,
# i.e. a filler word keeps ~2.4x the weight multiplier of a word that occurs everywhere.
N = X_tfidf.shape[0]
idf_lookup = dict(zip(feature_names, tfidf_vec.idf_))
for w in ["know", "like", "just", "think", "does", "use", "thanks"]:
    if w in idf_lookup:
        doc_freq = int((X_bow[:, bow_vec.vocabulary_[w]] > 0).sum())
        print(f"  {w:<7} df={doc_freq:4d} ({100 * doc_freq / N:5.1f}% of posts)  idf={idf_lookup[w]:.3f}")
print(f"  idf if df = 25% of N: {np.log((1 + N) / (1 + 0.25 * N)) + 1:.3f}")

numeric_terms = [t for t in feature_names if any(ch.isdigit() for ch in t)]
print(f"  Numeric/alphanumeric tokens in baseline vocabulary: {len(numeric_terms)} (e.g. {numeric_terms[:8]})")

CUSTOM_STOPWORDS = [
    "know", "like", "just", "think", "does", "use", "thanks", "used", "using", "want", "need",
    "good", "make", "way", "really", "say", "said", "going", "got", "people", "time", "thing",
    "things", "don", "doesn", "didn", "ve", "ll", "isn", "let", "right", "sure", "anybody",
    "anyone", "help", "appreciated", "advance", "post", "question", "problem", "new", "work",
    # Email / quoting artefacts that survive remove=("headers", "footers", "quotes")
    "edu", "com", "mail", "email", "info", "information", "net", "org", "address", "reply", "article", "writes",
    # A recurring sci.med signature block ("Gordon Banks N3JXP ... Skepticism is the chastity of the intellect")
    # that otherwise hijacks an entire cluster of 71 posts
    "gordon", "banks", "n3jxp", "skepticism", "chastity", "intellect", "geb", "cadre", "dsl", "pitt",
    "shameful", "surrender", "soon",
]
clean_stopwords = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))

tfidf_clean_vec = TfidfVectorizer(
    stop_words=clean_stopwords,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    min_df=5,
    max_features=4000,
)
X_tfidf_clean = tfidf_clean_vec.fit_transform(df["text"])
kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = kmeans_clean.fit_predict(X_tfidf_clean)
clean_terms = top_terms_per_cluster(kmeans_clean, tfidf_clean_vec.get_feature_names_out(), n=8)

print("\nTop 8 terms per cluster -- BASELINE vs CLEANED:")
for c in range(4):
    print(f"  Baseline C{c}: {', '.join(baseline_terms[c])}")
for c in range(4):
    print(f"  Cleaned  C{c}: {', '.join(clean_terms[c])}")
ari_base = adjusted_rand_score(df["true_label"], df["cluster"])
ari_clean = adjusted_rand_score(df["true_label"], df["cluster_clean"])
print(f"\n  Adjusted Rand Index vs. true labels: baseline={ari_base:.3f} | cleaned={ari_clean:.3f}")


# %%
# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Run `pd.crosstab(df['true_label'], df['cluster'])` and inspect overlapping categories.
print("\n" + "=" * 60)
print("CHALLENGE 2: Hardware vocabulary overlap")
print("=" * 60)


def majority_mapping(labels_col):
    """Map each cluster id to the true label that dominates it."""
    ct = pd.crosstab(df["true_label"], df[labels_col])
    return ct, ct.idxmax(axis=0).to_dict()


crosstab_base, cluster_to_label = majority_mapping("cluster")
print("Baseline crosstab (rows = true label, cols = cluster):")
print(crosstab_base)
print("Cluster -> majority label:", cluster_to_label)

crosstab_clean, clean_to_label = majority_mapping("cluster_clean")
print("\nCleaned-vectorizer crosstab:")
print(crosstab_clean)
print("Cluster -> majority label:", clean_to_label)

# Misclustered hardware posts = electronics posts landing in a Mac-dominated cluster and vice versa
ELEC, MAC = "sci.electronics", "comp.sys.mac.hardware"
pred_label = df["cluster_clean"].map(clean_to_label)
mis_mask = ((df["true_label"] == ELEC) & (pred_label == MAC)) | ((df["true_label"] == MAC) & (pred_label == ELEC))
print(f"\nElectronics <-> Mac misclustered posts (cleaned model): {mis_mask.sum()}")

# Terms carrying the most weight in BOTH categories = vocabulary that pulls them together
clean_names = tfidf_clean_vec.get_feature_names_out()
elec_mean = np.asarray(X_tfidf_clean[(df["true_label"] == ELEC).values].mean(axis=0)).ravel()
mac_mean = np.asarray(X_tfidf_clean[(df["true_label"] == MAC).values].mean(axis=0)).ravel()
mis_mean = np.asarray(X_tfidf_clean[mis_mask.values].mean(axis=0)).ravel()
shared_score = np.minimum(elec_mean, mac_mean)
shared_terms = [clean_names[i] for i in shared_score.argsort()[::-1][:12]]
mis_terms = [clean_names[i] for i in mis_mean.argsort()[::-1][:12]]
print("Top shared Electronics/Mac terms:", shared_terms)
print("Top terms inside misclustered posts:", mis_terms)
for i in np.where(mis_mask)[0][:3]:
    snippet = " ".join(df.loc[i, "text"].split())[:160]
    print(f"  [#{i} true={df.loc[i, 'true_label']}, routed={pred_label[i]}] {snippet}...")

# Bigrams: phrases such as 'power supply' vs 'mac ii' / 'hard drive' disambiguate shared unigrams
tfidf_bigram_vec = TfidfVectorizer(
    stop_words=clean_stopwords,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    min_df=5,
    max_features=8000,
    ngram_range=(1, 2),
)
X_tfidf_bigram = tfidf_bigram_vec.fit_transform(df["text"])
kmeans_bigram = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_bigram"] = kmeans_bigram.fit_predict(X_tfidf_bigram)
crosstab_bigram, bigram_to_label = majority_mapping("cluster_bigram")
print("\nBigram (1,2) crosstab:")
print(crosstab_bigram)
bigram_names = tfidf_bigram_vec.get_feature_names_out()
top_bigrams = [bigram_names[i] for i in np.asarray(X_tfidf_bigram.sum(axis=0)).ravel().argsort()[::-1] if " " in bigram_names[i]][:15]
print("Most weighted bigrams:", top_bigrams)
ari_bigram = adjusted_rand_score(df["true_label"], df["cluster_bigram"])
print(f"  ARI (all 4 classes): baseline={ari_base:.3f} | cleaned={ari_clean:.3f} | cleaned+bigrams={ari_bigram:.3f}")

# Pair-specific view: how cleanly are Electronics and Mac posts separated from each other?
hw_mask = df["true_label"].isin([ELEC, MAC])
for col in ["cluster_clean", "cluster_bigram"]:
    _, mapping = majority_mapping(col)
    routed = df.loc[hw_mask, col].map(mapping)
    swapped = ((df.loc[hw_mask, "true_label"] == ELEC) & (routed == MAC)).sum() + \
              ((df.loc[hw_mask, "true_label"] == MAC) & (routed == ELEC)).sum()
    pair_ari = adjusted_rand_score(df.loc[hw_mask, "true_label"], df.loc[hw_mask, col])
    print(f"  {col:<15} Electronics<->Mac swaps={swapped:4d} | pair ARI={pair_ari:.3f}")


# %%
# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 vector norms (`np.linalg.norm`) of the longest and shortest posts in `X_bow` vs `X_tfidf`.
print("\n" + "=" * 60)
print("CHALLENGE 3: Rant length bias")
print("=" * 60)

word_counts = df["text"].str.split().str.len()
has_features = np.asarray(X_bow.sum(axis=1)).ravel() > 0  # ignore posts with zero in-vocabulary tokens
longest_id = int(word_counts.idxmax())
shortest_id = int(word_counts[has_features].idxmin())

for name, idx in [("Longest", longest_id), ("Shortest", shortest_id)]:
    bow_norm = np.linalg.norm(X_bow[idx].toarray())
    tfidf_norm = np.linalg.norm(X_tfidf[idx].toarray())
    print(f"  {name:<8} post #{idx:<5} words={word_counts[idx]:<6} ||x_bow||2={bow_norm:9.3f}  ||x_tfidf||2={tfidf_norm:.3f}")

bow_norms = np.sqrt(np.asarray(X_bow.multiply(X_bow).sum(axis=1)).ravel())
print(f"  BoW norm range across corpus: {bow_norms[has_features].min():.2f} - {bow_norms.max():.2f}")
print(f"  Correlation(word count, BoW norm) = {np.corrcoef(word_counts, bow_norms)[0, 1]:.3f}")

kmeans_bow = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_bow"] = kmeans_bow.fit_predict(X_bow)
print("  KMeans on raw BoW counts -> cluster sizes:", df["cluster_bow"].value_counts().sort_index().to_dict())
print(f"  ARI: BoW={adjusted_rand_score(df['true_label'], df['cluster_bow']):.3f} | TF-IDF={ari_base:.3f}")


# %%
# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return or print the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` (test with doc_id=412).
    """
    row = matrix[doc_id].tocsr()
    terms = vectorizer.get_feature_names_out()
    order = row.data.argsort()[::-1][:top_n]  # only non-zero entries are stored in the sparse row
    explanation = [(str(terms[row.indices[i]]), float(row.data[i])) for i in order]

    print(f"\nDocument #{doc_id} -- top {len(explanation)} non-zero TF-IDF features:")
    for term, weight in explanation:
        print(f"  {term:<15} {weight:.4f}")
    return explanation


print("\n" + "=" * 60)
print("CHALLENGE 4: Explainability audit")
print("=" * 60)
AUDIT_DOC = 412
audit_explanation = explain_document(AUDIT_DOC, X_tfidf, tfidf_vec, top_n=5)
audit_cluster = int(df.loc[AUDIT_DOC, "cluster"])
print(f"  True label: {df.loc[AUDIT_DOC, 'true_label']} | Routed to cluster {audit_cluster} "
      f"(majority: {cluster_to_label[audit_cluster]}) | cluster top terms: {', '.join(baseline_terms[audit_cluster][:5])}")
print("  Post snippet:", " ".join(df.loc[AUDIT_DOC, "text"].split())[:200], "...")
