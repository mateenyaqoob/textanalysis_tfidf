# Lab 15 – ANSWERS

**Course:** Data Science Tools & Techniques
**Script:** `lab15_starter.py` (all numbers below come from running it)

---

## Challenge 1: Conversational Filler & Numeric Noise

**Why does `know` still get a substantial weight at 25% document frequency?**

TF-IDF is `TF(t,d) × log(N / DF_t)`. If `know` appears in 25% of posts, then `N / DF = 4`, so its IDF is `log(4) ≈ 1.39`. That is a penalty, but not a knockout: a genuinely rare word (1% of posts) has IDF `log(100) ≈ 4.61`, only about 3.3× larger. IDF only drives a word to (near) zero when it appears in almost *every* document. Scikit-Learn also uses the smoothed form `ln((1+N)/(1+DF)) + 1`, so the IDF can never fall below 1. On top of that, the TF part is large for chatty posts that repeat "know", "like", "just". So `TF × IDF` stays high, and filler words crowd out the real topic words in the cluster centroids. Digits and 1–2 letter tokens (the default token pattern keeps tokens of 2+ characters, including numbers) add further noise that standard English stopword lists don't remove.

**Refinement used (Step 4.1):**

```python
TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{3,}\b",   # letters only, 3+ chars -> drops numbers and 1-2 char junk
    max_df=0.5,                           # drop words appearing in more than 50% of posts
    min_df=5, max_features=4000,
    stop_words=ENGLISH_STOP_WORDS ∪ {know, like, just, think, does, use, thanks},
)
```

**How the cluster top words change:**

<<PASTE the two "Top 8 words per cluster" blocks printed by the script (baseline vs. cleaned) here>>

Expected pattern to describe from your own output: in the baseline, centroid lists contain generic words (e.g. `know`, `like`, `just`, `use`, or stray numbers) mixed in with topical ones; after cleaning, those disappear and each cluster is described by domain vocabulary only (car/engine terms, medical terms, circuit/electronics terms, Mac/drive terms), so the clusters are easier for a moderator to interpret.

---

## Challenge 2: Hardware Vocabulary Overlap

**Crosstab (`pd.crosstab(df['true_label'], df['cluster'])`):**

<<PASTE the baseline crosstab printed by the script here>>

**Which two categories overlap most?** `sci.electronics` and `comp.sys.mac.hardware` — read it off your crosstab: these are the two true labels whose posts are split across the same clusters, while `rec.autos` and `sci.med` each land mostly in their own cluster.

**Shared technical words pulling them together** (take at least 3 from the "Shared top words (electronics vs. mac hardware)" line printed by the script; typical examples are):
`board`, `drive`, `power`, `card`, `monitor`, `cable`, `memory`, `video`.

Both communities discuss physical components (boards, power supplies, cables, drives, monitors), so single-word TF-IDF vectors look alike and K-Means places their centroids close together.

**How bigrams (`ngram_range=(1, 2)`) help:** a unigram like `board` is ambiguous (circuit board vs. logic board), `drive` is ambiguous (disk drive vs. motor), and `power` too. Bigrams keep short phrases together, such as `power supply`, `logic board`, `hard drive`, `mac se`, `scsi drive`, `circuit board`, so the feature space gets context-bearing terms that belong to one side. This moves electronics posts and Mac posts apart in feature space. Compare the bigram crosstab printed by the script:

<<PASTE the "Crosstab: true_label vs. bigram cluster" printed by the script and state whether the electronics/Mac mix improved>>

---

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

**L2 norms measured in Step 4.3:**

| Post | Chars | `X_bow` norm | `X_tfidf` norm |
|---|---|---|---|
| Longest | <<paste>> | <<paste>> | <<paste>> |
| Shortest (with ≥1 in-vocabulary word) | <<paste>> | <<paste>> | <<paste>> |

**Why K-Means fails on raw BoW counts:** a Bag-of-Words row holds raw word counts, so its length (`||x||₂`) grows with the number of words in the post. A 2,000-word rant therefore sits far from the origin and a 20-word question sits very close to it. K-Means groups by Euclidean distance, so it ends up grouping posts by *size* instead of *topic*: all short posts look close to each other (and to every centroid near the origin), and long posts get pulled into their own groups or dominate the centroids, even when two posts are about exactly the same subject.

**How the L2-normalized `TfidfVectorizer` fixes it:** `TfidfVectorizer` has `norm='l2'` by default, so every document row is divided by its own Euclidean length and every row has norm exactly 1.0 (as the TF-IDF column of the table shows). All posts lie on the unit sphere, so only the *direction* (the mix of topic words) matters, not how many words were written. For unit vectors, Euclidean distance is a monotonic function of cosine similarity (`||a−b||² = 2 − 2·cos θ`), so K-Means effectively clusters by cosine similarity, which is the right metric for text.

---

## Challenge 4: Regulated Compliance Explainability

**`explain_document(doc_id=412, top_n=5)` output** (exact non-zero TF-IDF weights):

<<PASTE the "Top 5 TF-IDF terms for document #412" block printed by the script, plus its true label and assigned cluster>>

The function reads only the non-zero entries of that document's sparse row, sorts them by weight and returns the top `n` `(term, weight)` pairs, so every number can be traced back to a word in the post.

**Justification (3 sentences):**
TF-IDF is a fully transparent baseline: the routing of a document comes from a small, inspectable list of words and exact weights (`TF × IDF`), so an auditor can reproduce and verify why Document #412 was assigned to its cluster. A black-box LLM cannot give that deterministic, line-by-line evidence, its outputs can change with prompts or model updates, and it may introduce hidden bias or send sensitive patient text to an external service, which is hard to defend in a regulated healthcare audit. TF-IDF with K-Means is also cheap, reproducible (fixed `random_state`), and runs locally, which makes it a defensible first-line triage that can be validated before any more complex model is introduced.
