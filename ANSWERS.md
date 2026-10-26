# Lab 15 – Problem-Solving Answers

Dataset: 20 Newsgroups (`sci.electronics`, `rec.autos`, `comp.sys.mac.hardware`, `sci.med`), 3,703 posts after dropping posts of 50 characters or fewer. TF-IDF: 4,000 features, `min_df=5`. K-Means: `k=4`, `random_state=42`.

---

## Challenge 1: Conversational Filler & Numeric Noise

**Why does `know` still get a substantial weight at 25% document frequency?**

TF-IDF is `TF × log(N / DF)`. IDF only reaches zero when a word appears in *every* document. At 25% DF, `log(N/DF) = log(4) ≈ 1.39`, which is small but not zero. Scikit-Learn's smoothed variant, `ln((1+N)/(1+DF)) + 1`, gives about 2.4, so the word is never fully suppressed. A word is also *dense* when it appears in a quarter of all posts. Its TF-IDF is nonzero in thousands of rows, so its **average** weight across the corpus is high even though each individual weight is modest. The baseline top-10 chart shows this: `know`, `like`, `just`, `don`, `does` sit at the top, with `car` as the only real topic word in the list. (`don` comes from "don't" being split by the default tokenizer.) A long list of fillers that appear everywhere adds up to a lot of weight, which pulls the K-Means centroids toward generic chatter instead of topics.

**Refinement used**

```python
TfidfVectorizer(
    stop_words=list(custom_stopwords),     # English + know, like, just, think, does, use, thanks, ...
    token_pattern=r"\b[a-zA-Z]{3,}\b",     # alphabetic tokens of 3+ letters: removes numbers and fragments like "don"/"n3jxp"-style IDs
    max_df=0.5,                            # drop terms in more than 50% of posts
    max_features=4000,
    min_df=5,
)
```

**How the cluster top words changed**

| | Original TF-IDF | Cleaned TF-IDF |
|---|---|---|
| Cluster 0 (largest, ~2,400 posts) | know, like, don, just, does | *[paste from `print_cluster_keywords` output]* |
| Junk cluster (~71 posts, all `sci.med`) | cadre, dsl, n3jxp, chastity, surrender | *[paste cleaned keywords]* |
| Autos | car, cars, engine, new | *[paste]* |
| Mac hardware | mac, apple, drive, monitor, card | *[paste]* |

In the original run, the biggest cluster was labelled by filler words (`know, like, don, just, does`), which means it was a "generic chatter" bucket rather than a topic. Filtering and `max_df` remove that vocabulary, so the cluster labels shift toward topical words. The `sci.med` junk cluster (a small group of near-duplicate posts sharing a signature block with `cadre, dsl, n3jxp, chastity, surrender`) survives cleaning (72 posts), because those tokens are rare and repeated, which is exactly what TF-IDF rewards. Removing that block would need an explicit duplicate or signature filter, not just stopwords.

**Honest assessment of the result:** cleaning made the keywords more meaningful, but it did not clearly improve the separation. Mac hardware stays well isolated (579 posts in cluster 0), while the autos cluster actually got smaller (507 → 420 posts), and the largest cluster (cluster 2) still absorbs most of `sci.med` and `sci.electronics`.

---

## Challenge 2: Hardware Vocabulary Overlap

**Crosstab (original clusters)**

| Actual category | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| comp.sys.mac.hardware | 269 | 0 | 2 | **639** |
| rec.autos | 396 | 0 | **507** | 2 |
| sci.electronics | **860** | 0 | 20 | 60 |
| sci.med | **875** | 71 | 0 | 2 |

**Which two categories overlap most?** `sci.electronics` and `comp.sys.mac.hardware`. Cluster 3 is the "Mac hardware" cluster (keywords: mac, apple, drive, monitor, card), and 60 `sci.electronics` posts land in it. In the other direction, 269 `mac.hardware` posts fall into the generic cluster 0, which is where nearly all electronics posts sit. Mac and electronics posts are therefore mixed in both directions. (`sci.med` also lands almost entirely in cluster 0 with electronics, but that is because cluster 0 is a catch-all bucket, not a vocabulary overlap.)

**Shared technical words pulling them together:** `drive`, `card`, `monitor`, plus typically `board`, `power`, `memory`, `video`. The first three appear directly in the Mac cluster's top keywords. Confirm the rest with:

```python
overlap = df[(df.true_label == "sci.electronics") & (df.cluster == 3)]
vec = TfidfVectorizer(stop_words="english", max_features=2000)
M = vec.fit_transform(overlap["text"])
print(vec.get_feature_names_out()[np.asarray(M.mean(0)).ravel().argsort()[-10:]])
```

**How bigrams help:** single words like `drive`, `card`, `board`, and `power` are ambiguous and appear in both communities. Bigrams (`ngram_range=(1, 2)`) add context-specific phrases such as `power supply`, `circuit board`, `serial port` (electronics) versus `hard drive`, `mac plus`, `logic board`, `apple monitor` (Mac). A post's vector then includes phrases that point to one community, which raises the weight of discriminating features relative to the shared single words:

```python
TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000, min_df=5, max_df=0.5)
```

---

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

| Document | Chars | BoW L2 norm | TF-IDF L2 norm |
|---|---|---|---|
| Longest post | *[fill from output]* | *[larger]* | ≈ 1.0 |
| Shortest post | *[fill from output]* | *[much smaller]* | ≈ 1.0 *(or 0 if it has no vocabulary terms)* |

**Why K-Means fails on raw BoW counts:** K-Means minimizes Euclidean distance. In BoW, every extra word adds to the counts, so vector magnitude grows with post length. A 2,000-word rant is far from the origin, and a 20-word question sits near it. Distances therefore mostly reflect *how long* a post is rather than *what it is about*. Short posts of any topic clump near the origin and get grouped together, while long posts become outliers or dominate and drag centroids. This is consistent with the oversized generic cluster 0 in the results.

**How `TfidfVectorizer` fixes it:** its default `norm="l2"` scales every document vector to unit length, so every post lies on the unit sphere. Euclidean distance between unit vectors is a monotonic function of cosine similarity (`‖a−b‖² = 2 − 2·cos`), so clustering depends on the *direction* (the mix of words), not on length. IDF weighting also down-weights terms common to all posts.

---

## Challenge 4: Regulated Compliance Explainability

```python
explanation_412 = explain_document(doc_id=412, matrix=X_tfidf, vectorizer=tfidf_vec, top_n=5)
```

Output for Document #412 *(paste from your run)*:

| Term | Weight |
|---|---|
| *[term 1]* | *[0.xxxxxx]* |
| *[term 2]* | *[0.xxxxxx]* |
| *[term 3]* | *[0.xxxxxx]* |
| *[term 4]* | *[0.xxxxxx]* |
| *[term 5]* | *[0.xxxxxx]* |

**Justification (3 sentences):**
TF-IDF assigns each document a sparse vector in which every number is the weight of a specific, human-readable word, so an auditor can see exactly which terms placed Document #412 in its cluster and reproduce that result deterministically. A large language model gives no such traceable, reproducible account of its decision, which makes it hard to defend in a regulated adverse-event audit that requires documented reasoning. TF-IDF is therefore the preferred transparent baseline, and any more complex model can be benchmarked against it and used only where its extra accuracy justifies the loss of explainability.
