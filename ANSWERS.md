# Lab 15 – Problem-Solving Answers (TF-IDF & Unstructured Text Triage)

> All numbers below come from running `lab15_starter.py` on the full dataset: **3,703 posts**, TF-IDF matrix **(3703, 4000)**, sparsity **99.04%**. Cluster IDs (0–3) are arbitrary labels chosen by K-Means, so they are read through the crosstabs, not by their number.

---

## Challenge 1: Conversational Filler & Numeric Noise

**Why does `know` still receive a substantial weight?**
With TF-IDF = TF × log(N / DF), a word that appears in 25% of posts has DF = N/4, so
IDF = log(4) ≈ 1.39 (natural log), or 0.60 with log10. sklearn's smoothed version gives ln((1+N)/(1+DF)) + 1 ≈ 2.39.
IDF only reaches 0 when a word is in *every* document, so 25% is nowhere near cancelling it. A genuinely topical word that appears in about 1% of posts gets ln(100) ≈ 4.6, which is only about 3× more. Chatty posts also repeat `know`, `like`, `just`, `don` many times, so their **TF is high**, and TF × a moderate IDF still gives a large weight.

In my run, these filler words appeared in a large share of posts: `like` 23.4%, `know` 22.7%, `just` 21.2%, `don` 20.3%, `does` 17.3%, `use` 17.1%, `think` 15.2%, `thanks` 13.0%. As a result, the baseline "Top 10 Most Influential Words" chart was dominated by `know, like, just, don, does, use, thanks, good, think`. `car` was the only real topic word in it.

**Refinement used:**
```python
TfidfVectorizer(stop_words = ENGLISH_STOP_WORDS + filler words (know, like, just, think, does, use,
                             thanks, don, want, good, used, need, way, make, really) + signature noise,
                token_pattern = r'\b[a-zA-Z]{3,}\b',   # letters only, 3+ chars: removes numbers and fragments
                max_df = 0.5, min_df = 5, max_features = 4000)
```
- `token_pattern` drops numbers, part numbers and 1–2 letter fragments.
- `max_df = 0.5` removes any word that appears in more than half of all posts.
- Custom stopwords remove the conversational filler that the standard English list misses.

**How the cluster top words changed:**

| | Cluster top words |
|---|---|
| **Baseline** | C0 (2,400 posts): `know, like, don, just, does, use, thanks, good` (a filler cluster, no topic) · C1 (71 posts): `cadre, dsl, n3jxp, chastity, surrender, intellect, shameful, skepticism` (one doctor's repeated e-mail signature) · C2: `car, cars, engine, new, just, ford, dealer, like` · C3: `mac, apple, drive, monitor, card, thanks, does, know` |
| **Cleaned** | C0: `doctor, msg, medical, people, disease, pain, patients` (**medical**) · C1: `car, cars, engine, oil, dealer, miles, ford` (**autos**) · C2: `mac, apple, drive, monitor, card, scsi, video, simms` (**Mac**) · C3: `information, time, info, power, mail, right, sure` (general / electronics) |

The filler cluster and the signature cluster disappeared, and a real **medical** cluster appeared. Filler words also dropped out of the cars and Mac clusters. Cluster purity (the share of posts that belong to their cluster's majority label) rose from **0.565 to 0.681**.

Extra finding: `remove=('footers',)` did not catch that repeated signature, so its words had to be added to the stopword list. Even then, `soon` (from "...surrender it too soon") still shows up in the medical cluster.

---

## Challenge 2: Hardware Vocabulary Overlap

`pd.crosstab(df['true_label'], df['cluster_clean'])`:

| true_label | C0 medical | C1 autos | C2 Mac | C3 general |
|---|---|---|---|---|
| comp.sys.mac.hardware | 3 | 1 | **612** | 294 |
| rec.autos | 8 | **427** | 2 | 468 |
| sci.electronics | 7 | 19 | 62 | **852** |
| sci.med | **629** | 0 | 2 | 317 |

**The two most overlapping categories are `sci.electronics` and `comp.sys.mac.hardware`.** Electronics never gets a topic cluster of its own. 852 electronics posts fall into the general "technical" cluster C3 together with 294 Mac posts, and another **62 electronics posts are pulled into the Mac cluster**.

**Shared words pulling them together** (highest average TF-IDF in *both* groups): `power`, `software`, `using`, `work`, `number`, plus help-request words such as `help`, `mail` and `advance` (as in "thanks in advance"). Both groups ask hardware questions about power, chips, drives and software, so their vectors point in similar directions.

**How bigrams (`ngram_range=(1, 2)`) help:** single words like `power`, `drive`, `card` are ambiguous, but phrases are specific. Phrases such as `power supply`, `hard drive`, `video card`, `access time` and `acid battery` become their own features. Electronics phrases (`acid battery`, `power supply`) pull electronics posts away from Mac posts, and Mac phrases keep the Mac cluster tight (`mac, apple, drive, monitor, card, scsi, simms, video`).

In my run, electronics posts wrongly placed in the Mac cluster fell from **62 to 55**. Overall purity, however, was **0.659**, slightly below the cleaned unigram model (0.681). Bigrams add useful context, but they also double the number of sparse features and created a large mixed "general" cluster. So they help separate the specific hardware topics, but the gain must be checked with the crosstab rather than assumed.

---

## Challenge 3: Rant Length Bias (BoW vs TF-IDF)

| Post | Characters | ‖x‖₂ in X_bow | ‖x‖₂ in X_tfidf |
|---|---|---|---|
| Longest (#310) | 72,161 | **276.241** | **1.0** |
| Shortest (#1159) | 52 | **2.000** | **1.0** |

- In **Bag of Words** a vector's length grows with the length of the post, because the counts add up. The longest rant is about **138×** longer than the short question.
- **K-Means uses Euclidean distance to cluster centroids.** With raw counts, that distance is driven by *how much someone wrote*, not *what they wrote about*. Long rants sit far from everything, and short posts all collapse near the origin regardless of topic, so the clusters reflect post length instead of subject.
- **`TfidfVectorizer` applies L2 normalisation by default (`norm='l2'`).** Every post becomes a unit vector (‖x‖₂ = 1.0), so only its *direction* (its relative word mix) remains. Euclidean distance between unit vectors then behaves like cosine similarity, and a 20-word question and a 2,000-word rant on the same topic land close together.

---

## Challenge 4: Regulated Compliance Explainability

`explain_document(412, X_tfidf, tfidf_vec, top_n=5)` returns a list of `(term, weight)` tuples: the exact non-zero TF-IDF features with the highest weights.

Document #412: true label `rec.autos`, assigned to **cluster 0** (the baseline generic/filler cluster).

| Term | TF-IDF weight |
|---|---|
| discussion | 0.4524 |
| feel | 0.3989 |
| days | 0.3855 |
| need | 0.2885 |
| meant | 0.2667 |

The explanation shows exactly *why* this cars post was routed to the generic queue: none of its strongest terms are car-specific (no `car`, `engine` or `dealer`). Its highest-weighted words are general conversation words, so it sits closest to the general cluster's centroid. An auditor can see this mis-routing and its cause immediately, and could fix it, for example by adding `discussion`, `feel` and `need` to the stopword list.

**Justification (3 sentences):** TF-IDF gives every routing decision an exact, reproducible audit trail: each document is a vector of named words with numeric weights, and its cluster is simply the nearest centroid, so an auditor can see precisely which terms drove the assignment, as shown for document #412. A black-box LLM makes decisions using billions of opaque parameters, can change its output between runs or model versions, and cannot provide a verifiable, deterministic reason, which is a compliance liability when adverse-event reports must be traceable. TF-IDF is also fast, cheap and stable, which makes it the right transparent baseline: any more complex model must first beat it and then explain itself just as clearly.
