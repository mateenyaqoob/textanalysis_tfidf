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
lemmatized_tokens = [lemmatizer.lemmatize(w, pos="v") for w in filtered_tokens]

if filtered_tokens is not None:
    print("Filtered Tokens:  ", filtered_tokens[:12])
    print("Stemmed Roots:    ", stemmed_tokens[:12])
    print("Lemmatized Words: ", lemmatized_tokens[:12])

    # Side-by-side view of where the two normalizers disagree
    diffs = [
        (w, s, l)
        for w, s, l in zip(filtered_tokens, stemmed_tokens, lemmatized_tokens)
        if s != l
    ]
    print("\nWord -> Stem vs Lemma (where they differ):")
    for w, s, l in diffs[:10]:
        print(f"  {w:<14s} -> {s:<12s} | {l}")


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
    print(f"\nBoW Matrix Shape:    {X_bow.shape}")
    print(f"TF-IDF Matrix Shape: {X_tfidf.shape} | Sparsity: {sparsity:.2f}%")

    # TODO 2.4: Compute mean score per feature across documents and plot Top 10 TF-IDF words
    tfidf_means = np.asarray(X_tfidf.mean(axis=0)).ravel()
    top_idx = tfidf_means.argsort()[-10:]
    top_words = tfidf_vec.get_feature_names_out()[top_idx]
    top_scores = tfidf_means[top_idx]
    print("Top 10 words by average TF-IDF:", list(top_words[::-1]))

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
def cluster_top_words(model, vectorizer, n=8):
    """Return {cluster_id: [top-n centroid terms]}."""
    terms = vectorizer.get_feature_names_out()
    order = model.cluster_centers_.argsort()[:, ::-1]
    return {c: list(terms[order[c, :n]]) for c in range(model.n_clusters)}


if X_tfidf is not None:
    # TODO 3.1: Fit KMeans with `n_clusters=4, random_state=42, n_init=10` on `X_tfidf`
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    # TODO 3.2: Print the top 8 defining TF-IDF words per cluster centroid
    # and plot a seaborn heatmap (`cluster_heatmap.png`) of the top 5 words per cluster.
    terms = tfidf_vec.get_feature_names_out()
    print("\n=== Top 8 words per cluster (baseline TF-IDF) ===")
    for c, words in cluster_top_words(kmeans, tfidf_vec, n=8).items():
        print(f"Cluster {c} ({(df['cluster'] == c).sum()} posts): {', '.join(words)}")

    heat_words = []
    for words in cluster_top_words(kmeans, tfidf_vec, n=5).values():
        heat_words += [w for w in words if w not in heat_words]
    heat_idx = [list(terms).index(w) for w in heat_words]
    heat_df = pd.DataFrame(
        kmeans.cluster_centers_[:, heat_idx],
        index=[f"Cluster {c}" for c in range(4)],
        columns=heat_words,
    )
    plt.figure(figsize=(12, 3.8))
    sns.heatmap(heat_df, cmap="YlGnBu", annot=True, fmt=".3f", cbar_kws={"label": "Centroid TF-IDF weight"})
    plt.title("Cluster x Word Importance (top 5 words per cluster)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.show()

    # TODO 3.3: Project `X_tfidf` to 2 dimensions using `TruncatedSVD(n_components=2, random_state=42)`
    # and save a 2D scatter plot (`spatial_mapping.png`) colored by `df["cluster"]`.
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X_tfidf)
    print(f"\nSVD explained variance (2 comps): {svd.explained_variance_ratio_.sum():.3%}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=df["cluster"], cmap="tab10", s=8, alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.title("2D Spatial Mapping of Posts (TruncatedSVD on TF-IDF)")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.show()


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %%
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

CATS = sorted(df["true_label"].unique())


def show_clusters(model, vectorizer, labels, tag, n=8):
    """Print top-n centroid words per cluster with the majority true label."""
    majority = pd.crosstab(labels, model.labels_).idxmax(axis=0)
    for c, words in cluster_top_words(model, vectorizer, n=n).items():
        print(f"  {tag} {c} [{majority[c]:<21s}] {', '.join(words)}")


# Challenge 1 (Noise Filtering):
# TODO 4.1: Build a cleaner `TfidfVectorizer` using `token_pattern=r'\b[a-zA-Z]{3,}\b'`,
# `max_df=0.5`, and custom conversational stopwords (e.g., 'know', 'like', 'just', 'think', 'does', 'use', 'thanks').
# Re-fit KMeans and compare cluster keywords!
N = X_tfidf.shape[0]
print("\n=== Challenge 1: why filler words survive ===")
print(f"Textbook IDF for a word in 25% of posts: log(1/0.25) = {np.log(4):.3f}")
print(f"sklearn smoothed IDF at 25%: 1 + ln((1+N)/(1+0.25N)) = {1 + np.log((1 + N) / (1 + 0.25 * N)):.3f} "
      f"(max possible = {1 + np.log((1 + N) / 2):.3f})")
for w in ["know", "like", "just", "does", "use", "thanks", "think"]:
    j = tfidf_vec.vocabulary_[w]
    df_pct = 100 * (X_bow[:, bow_vec.vocabulary_[w]] > 0).sum() / N
    rank = int((tfidf_means > tfidf_means[j]).sum()) + 1
    print(f"  {w:<7s} in {df_pct:4.1f}% of posts | idf={tfidf_vec.idf_[j]:.2f} | avg TF-IDF rank #{rank} of {len(tfidf_means)}")

FILLER_WORDS = [
    "know", "like", "just", "think", "does", "use", "thanks", "used", "using", "want", "need",
    "good", "make", "way", "really", "say", "said", "people", "time", "don", "doesn", "didn",
    "isn", "got", "going", "thing", "things", "anybody", "anyone", "help", "appreciated",
    "advance", "mail", "email", "post", "posted", "question", "sure", "lot", "right", "edu",
    "com", "work", "new", "problem", "probably", "information", "info", "years",
]
# A signature block repeated verbatim in ~72 sci.med posts ("Gordon Banks N3JXP ... geb@cadre.dsl.pitt.edu")
# hijacks a whole cluster in the baseline. Its tokens are boilerplate, not content.
SIGNATURE_WORDS = [
    "gordon", "banks", "n3jxp", "geb", "cadre", "dsl", "pitt", "skepticism",
    "chastity", "intellect", "shameful", "surrender", "soon",
]
custom_stopwords = list(TfidfVectorizer(stop_words="english").get_stop_words()) + FILLER_WORDS + SIGNATURE_WORDS


def clean_vectorizer(**overrides):
    params = dict(
        stop_words=custom_stopwords,
        token_pattern=r"\b[a-zA-Z]{3,}\b",
        max_df=0.5,
        min_df=5,
        max_features=4000,
        sublinear_tf=True,  # 1 + log(tf): a word repeated 20x is not 20x more important
    )
    params.update(overrides)
    return TfidfVectorizer(**params)


tfidf_clean = clean_vectorizer()
X_clean = tfidf_clean.fit_transform(df["text"])
kmeans_clean = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_clean"] = kmeans_clean.fit_predict(X_clean)

print("\nCluster keywords — baseline vs cleaned:")
show_clusters(kmeans, tfidf_vec, df["true_label"], "BASE ")
show_clusters(kmeans_clean, tfidf_clean, df["true_label"], "CLEAN")

print("\nAblation (Adjusted Rand Index vs true labels, higher = better):")
print(f"  baseline TF-IDF                           {adjusted_rand_score(df['true_label'], df['cluster']):.3f}")
ablation_stops = list(TfidfVectorizer(stop_words="english").get_stop_words()) + FILLER_WORDS
for label, vec in [
    ("+ token_pattern, max_df, filler words", clean_vectorizer(stop_words=ablation_stops, sublinear_tf=False)),
    ("+ signature words removed", clean_vectorizer(sublinear_tf=False)),
    ("+ sublinear_tf (final cleaned)", tfidf_clean),
]:
    Xa = X_clean if vec is tfidf_clean else vec.fit_transform(df["text"])
    la = df["cluster_clean"] if vec is tfidf_clean else KMeans(4, random_state=42, n_init=10).fit_predict(Xa)
    print(f"  {label:<41s} {adjusted_rand_score(df['true_label'], la):.3f}")


# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Run `pd.crosstab(df['true_label'], df['cluster'])` and inspect overlapping categories.
print("\n=== Challenge 2: crosstab (baseline clusters) ===")
print(pd.crosstab(df["true_label"], df["cluster"]))
print("\nCrosstab (cleaned clusters):")
print(pd.crosstab(df["true_label"], df["cluster_clean"]))

# Vocabulary overlap between the TRUE categories: cosine similarity of their mean TF-IDF vectors.
clean_terms = tfidf_clean.get_feature_names_out()
cat_centroids = np.vstack([np.asarray(X_clean[(df["true_label"] == c).values].mean(axis=0)).ravel() for c in CATS])
overlap = pd.DataFrame(cosine_similarity(cat_centroids), index=CATS, columns=CATS)
with pd.option_context("display.max_columns", None, "display.width", 200):
    print("\nCategory-to-category vocabulary similarity (cosine of mean TF-IDF vectors):")
    print(overlap.round(3))
pair_scores = {(a, b): overlap.at[a, b] for i, a in enumerate(CATS) for b in CATS[i + 1:]}
cat_a, cat_b = max(pair_scores, key=pair_scores.get)
print(f"Most overlapping pair: {cat_a} <-> {cat_b} (cosine {pair_scores[(cat_a, cat_b)]:.3f})")

vec_a, vec_b = cat_centroids[CATS.index(cat_a)], cat_centroids[CATS.index(cat_b)]
shared_words = list(clean_terms[np.minimum(vec_a, vec_b).argsort()[::-1][:10]])
print(f"Shared technical words (high in BOTH categories): {shared_words}")

# Misclustered posts: true label is one of the pair, but the post sits in the cluster owned by the other.
owner = pd.crosstab(df["true_label"], df["cluster_clean"]).idxmax(axis=0)
df["predicted_clean"] = df["cluster_clean"].map(owner)
mis = df[df["true_label"].isin([cat_a, cat_b]) & df["predicted_clean"].isin([cat_a, cat_b])
         & (df["true_label"] != df["predicted_clean"])]
print(f"{len(mis)} posts misrouted between the two. Examples:")
for idx in mis.index[:4]:
    row_terms = set(clean_terms[X_clean[idx].indices])
    hits = [w for w in shared_words if w in row_terms]
    snippet = " ".join(df.at[idx, "text"].split())[:130]
    print(f"  #{idx} [{df.at[idx, 'true_label']} -> {df.at[idx, 'predicted_clean']}] shared={hits} | {snippet}...")

# Bigrams: do phrases separate general electronics from Mac hardware?
tfidf_bigram = clean_vectorizer(ngram_range=(1, 2), max_features=8000)
X_bigram = tfidf_bigram.fit_transform(df["text"])
bigram_terms = tfidf_bigram.get_feature_names_out()
bi_centroids = np.vstack([np.asarray(X_bigram[(df["true_label"] == c).values].mean(axis=0)).ravel() for c in CATS])
bi_a, bi_b = bi_centroids[CATS.index(cat_a)], bi_centroids[CATS.index(cat_b)]
print(f"\nBigrams: {cat_a} <-> {cat_b} cosine {pair_scores[(cat_a, cat_b)]:.3f} -> "
      f"{cosine_similarity([bi_a], [bi_b])[0, 0]:.3f}")
for name, mine, other in [(cat_a, bi_a, bi_b), (cat_b, bi_b, bi_a)]:
    order = (mine - other).argsort()[::-1]
    print(f"  Distinctive bigrams for {name}: {[bigram_terms[j] for j in order if ' ' in bigram_terms[j]][:8]}")

pair_mask = df["true_label"].isin([cat_a, cat_b]).values
for name, X_pair in [("unigrams", X_clean[pair_mask]), ("uni+bigrams", X_bigram[pair_mask])]:
    accs = []
    for seed in range(5):
        km2 = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(X_pair)
        ct = pd.crosstab(df.loc[pair_mask, "true_label"], km2.labels_).values
        accs.append(max(ct.trace(), ct[::-1].trace()) / pair_mask.sum())
    print(f"  2-way KMeans on {cat_a} + {cat_b} only, {name:<11s}: accuracy {np.mean(accs):.1%} "
          f"(5 seeds, {min(accs):.1%}-{max(accs):.1%})")


# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 vector norms (`np.linalg.norm`) of the longest and shortest posts in `X_bow` vs `X_tfidf`.
lengths = df["text"].str.split().str.len()
has_terms = np.asarray(X_bow.sum(axis=1)).ravel() > 0  # skip posts whose every word was filtered out
longest = int(lengths.idxmax())
shortest = int(lengths[has_terms].idxmin())
print("\n=== Challenge 3: length bias ===")
for name, idx in [("Longest", longest), ("Shortest", shortest)]:
    bow_norm = np.linalg.norm(X_bow[idx].toarray())
    tfidf_norm = np.linalg.norm(X_tfidf[idx].toarray())
    print(f"{name:<8s} post #{idx}: {lengths[idx]:5d} words | ||x||_2 BoW = {bow_norm:7.2f} | TF-IDF = {tfidf_norm:.2f}")
print(f"({(~has_terms).sum()} posts have an all-zero vector after stopword/min_df filtering and were skipped.)")

bow_norms = np.sqrt(np.asarray(X_bow.multiply(X_bow).sum(axis=1)).ravel())
print(f"Correlation(post length, BoW norm) = {np.corrcoef(lengths, bow_norms)[0, 1]:.2f}")
kmeans_bow = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_bow)
bow_sizes = np.bincount(kmeans_bow.labels_)
print(f"KMeans on raw BoW  -> cluster sizes {sorted(bow_sizes.tolist(), reverse=True)}, "
      f"ARI={adjusted_rand_score(df['true_label'], kmeans_bow.labels_):.3f}")
print(f"KMeans on TF-IDF   -> cluster sizes {sorted(np.bincount(kmeans.labels_).tolist(), reverse=True)}, "
      f"ARI={adjusted_rand_score(df['true_label'], kmeans.labels_):.3f}")
for c in np.where(bow_sizes < 0.05 * len(df))[0]:
    members = np.where(kmeans_bow.labels_ == c)[0]
    print(f"  BoW cluster {c}: {len(members)} post(s), median length {int(lengths.iloc[members].median())} words "
          f"(corpus median {int(lengths.median())})")


# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return or print the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` (test with doc_id=412).
    """
    row = matrix[doc_id].tocoo()
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(row.col, row.data), key=lambda x: x[1], reverse=True)[:top_n]
    explanation = [(str(terms[j]), round(float(w), 4)) for j, w in ranked]
    print(f"\nDocument #{doc_id} - top {len(explanation)} TF-IDF features:")
    for term, weight in explanation:
        print(f"  {term:<15s} {weight:.4f}")
    return explanation


DOC_ID = 412
print("\n=== Challenge 4: explainability audit ===")
base_owner = pd.crosstab(df["true_label"], df["cluster"]).idxmax(axis=0)
print(f"Doc #{DOC_ID} true label: {df.at[DOC_ID, 'true_label']} | assigned cluster {df.at[DOC_ID, 'cluster']} "
      f"(majority label: {base_owner[df.at[DOC_ID, 'cluster']]})")
print("Text:", " ".join(df.at[DOC_ID, "text"].split())[:300])
audit = explain_document(DOC_ID, X_tfidf, tfidf_vec, top_n=5)

x412 = X_tfidf[DOC_ID].toarray().ravel()
print("Distance to each centroid (KMeans picks the smallest):",
      {c: round(float(d), 4) for c, d in enumerate(np.linalg.norm(kmeans.cluster_centers_ - x412, axis=1))})
print("Centroid weight of the post's top terms (rows = term, cols = cluster):")
print(pd.DataFrame(
    {c: [kmeans.cluster_centers_[c, tfidf_vec.vocabulary_[t]] for t, _ in audit] for c in range(4)},
    index=[t for t, _ in audit],
).round(4))
