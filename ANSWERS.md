# Lab 15 – Answers: Unstructured Text Analysis & Automated Triage with TF-IDF

All numbers below come from running `lab15_starter.py` on the 4-category 20 Newsgroups subset
(`sci.electronics`, `rec.autos`, `comp.sys.mac.hardware`, `sci.med`) after keeping posts longer than 50 characters: **3,703 posts**.

---

## Part 1 – Summary of Results

**Step 1 – Stemming vs. Lemmatization** (first sample post)

| Filtered token | PorterStemmer | WordNetLemmatizer (`pos='v'`) |
|---|---|---|
| cable | cabl | cable |
| release | releas | release |
| locked | lock | lock |
| ignition | ignit | ignition |
| left | left | **leave** |
| someone | someon | someone |

The stemmer chops suffixes by rule and often produces non-words (`cabl`, `ignit`). The lemmatizer uses a dictionary and returns real words. It also finds the true verb root (`left` → `leave`), which the stemmer misses.

**Step 2 – Vectorization:** TF-IDF matrix shape `(3703, 4000)`, sparsity **99.04%**. Each post uses only about 1% of the vocabulary. See `top_features.png`.

**Step 3 – Baseline K-Means (k = 4) on TF-IDF:**

| Cluster | Posts | Top 8 words |
|---|---|---|
| 0 | 2400 | know, like, don, just, does, use, thanks, good |
| 1 | 71 | cadre, dsl, n3jxp, chastity, surrender, intellect, shameful, skepticism |
| 2 | 529 | car, cars, engine, new, just, ford, dealer, like |
| 3 | 703 | mac, apple, drive, monitor, card, thanks, does, know |

Only two clusters are real topics (autos, Mac). Cluster 0 is a "conversational filler" bucket holding 65% of all posts. Cluster 1 is one user's repeated e-mail signature (`geb@cadre.dsl.pitt.edu`, "Skepticism is the chastity of the intellect…") that `remove=("footers",)` failed to strip. See `cluster_heatmap.png` and `spatial_mapping.png`.

---

## Challenge 1: Conversational Filler & Numeric Noise

**Why does `know` still get a substantial weight?**

$$\text{TFIDF}_{t,d} = \text{TF}_{t,d} \times \log\left(\frac{N}{\text{DF}_t}\right)$$

IDF penalises a word only in proportion to how *many* documents contain it. It reaches zero only when a word appears in **every** document. A word in 25% of posts gets

$$\log\left(\frac{N}{0.25N}\right) = \log(4) \approx 1.39$$

That is a reduced weight, but far from zero. In our data `know` appears in **22.7%** of posts. Scikit-learn's smoothed IDF, $\ln\frac{1+N}{1+DF}+1$, gives it **2.48**. The rarest allowed word (DF = 5) gets about 7.4, so `know` still keeps roughly a third of the maximum per-occurrence weight.

| Word | % of posts | IDF |
|---|---|---|
| know | 22.7% | 2.480 |
| like | 23.4% | 2.452 |
| just | 21.2% | 2.550 |
| does | 17.3% | 2.753 |
| use | 17.1% | 2.767 |
| think | 15.2% | 2.884 |
| thanks | 13.0% | 3.035 |

Many posts are short questions ("does anyone know…, thanks"). In those posts the filler words make up a large share of the content, so after L2 normalisation they dominate the vector. K-Means then groups these posts by *writing style* rather than *topic*. This is exactly what created baseline Cluster 0 (2,400 posts).

**Refinement:** `TfidfVectorizer(token_pattern=r'\b[a-zA-Z]{3,}\b', max_df=0.5, stop_words=ENGLISH_STOP_WORDS + 29 filler words)`
- The `token_pattern` drops numbers and 1–2 letter tokens.
- `max_df=0.5` drops any word found in more than half of the posts.
- The custom list removes conversational verbs and fillers.

**How the cluster top words changed:**

| | Before cleaning | After cleaning |
|---|---|---|
| Filler cluster | know, like, don, just, does, use, thanks, good (2400) | **gone** |
| Autos | car, cars, engine, new, **just**, ford, dealer, **like** | car, cars, engine, new, **oil**, ford, dealer, **miles** |
| Mac | mac, apple, drive, monitor, card, **thanks, does, know** | mac, apple, drive, monitor, card, **scsi, video, simms** |
| Signature | cadre, dsl, n3jxp, chastity, … | geb, pitt, dsl, cadre, chastity, … |
| New catch-all | – | time, edu, information, new, work, help, problem, years (2460) |

The topical clusters became much cleaner: autos and Mac lost all filler words and gained real domain terms (`oil`, `miles`, `scsi`, `simms`). However, overall cluster purity stayed about the same (**56.5% → 55.0%**). A new catch-all cluster formed around the *next* layer of generic words. Two examples: `edu`, which comes from e-mail addresses, and `time`/`work`/`help`. Noise filtering is iterative. Removing one layer of filler exposes the next. Further steps would be to stop-list `edu`/`time`, strip the signature lines, or raise `k`.

---

## Challenge 2: Hardware Vocabulary Overlap

Crosstab of true category vs. cleaned clusters:

```
cluster_clean           0    1    2    3
true_label
comp.sys.mac.hardware   0    1  612  297
rec.autos               0  477    2  426
sci.electronics         0   20   57  863
sci.med                72    0    2  874
```

**Which categories overlap most?** `sci.electronics` is the category that never gets a cluster of its own:
- **91.8%** of its posts (863/940) fall into the generic catch-all cluster 3, alongside `sci.med`.
- Of its posts that land in a *topical* cluster, most go to the **Mac** cluster (57 posts), not the autos cluster (20).
- In the other direction, 297 Mac posts (32.6%) also slip into the generic cluster.

Overall, **1,238** electronics/Mac posts are misclustered. Electronics and Mac hardware are the most confusable pair because they share a hardware/computing vocabulary.

**Shared technical words pulling them together** (high average TF-IDF in *both* categories, and the top terms inside misclustered posts):
- `power`
- `chip`
- `circuit`
- `software`
- `battery`
- `phone`
- `using`

They also share the generic noise words `edu`, `work` and `help`. Sample misclustered posts confirm this. Post #2 is an electronics post about a "555 oscillator chip", and post #6 is a Mac post about SyQuest optical drives. Both use generic hardware language rather than category-specific terms.

**Effect of bigrams (`ngram_range=(1, 2)`):**

```
cluster_bigram           0    1    2   3
true_label
comp.sys.mac.hardware    0  647  263   0
rec.autos              398    2  505   0
sci.electronics         18   85  837   0
sci.med                  0    2  875  71
```

In principle, bigrams help separate the two domains. A unigram like `power` or `drive` is ambiguous, but the phrases `power supply` and `op amp` are clearly electronics, while `mac ii`, `scsi drive` and `powerbook battery` are clearly Mac. Bigrams preserve that local context.

In this run, the effect was mixed:
- The Mac cluster captured more Mac posts (**612 → 647**).
- It also absorbed more electronics posts (57 → 85).
- Overall purity changed little (**55.0% → 53.8%**).
- The top bigrams that did surface were mostly signature boilerplate (`dsl pitt`, `geb cadre`, `shameful surrender`).

Three reasons explain why bigrams did not help much here:
- Bigrams are even sparser than unigrams. With `min_df=5`, few discriminative phrases survive, and the 8,000-feature budget is still dominated by unigrams.
- Repeated boilerplate becomes *more* distinctive as bigrams.
- With `k = 4`, the large generic cluster continues to absorb short posts.

Bigrams are a useful tool, but they work best combined with signature removal and a larger `k` (or LSA before K-Means).

---

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

| Post | Words | ‖x‖₂ in `X_bow` | ‖x‖₂ in `X_tfidf` |
|---|---|---|---|
| Longest (#310) | 11,765 | **276.24** | 1.0000 |
| Shortest (#2721) | 4 | **3.00** | 1.0000 |

Correlation between post word-count and BoW norm: **0.925**.

**Why K-Means fails on raw BoW:** K-Means assigns points by Euclidean distance. In raw count space a vector's length grows with the length of the post, so a long rant and a short question about the *same* topic are far apart simply because one has ~90× larger counts. The distance is dominated by **how much** someone wrote, not **what** they wrote about. Our run shows this clearly:
- **3,692 of 3,703 posts** (99.7%) landed in a single cluster.
- The other three "clusters" contain just 8, 2 and 1 extremely long outlier posts.
- Purity was **25.7%**, no better than always guessing the largest category (948/3703 = 25.6%).

**How TF-IDF fixes it:** Scikit-learn's `TfidfVectorizer` uses `norm='l2'` by default. Every document vector is divided by its own length, so every post has ‖x‖₂ = 1 regardless of whether it has 4 or 11,765 words. All posts lie on the unit sphere. Euclidean distance between them then depends only on the *direction* of the vector, i.e. the relative mix of words, and becomes equivalent to cosine distance: $\|a-b\|^2 = 2(1-\cos(a,b))$. With the length bias removed, the same K-Means reached **56.5%** purity and found real topic clusters.

---

## Challenge 4: Regulated Compliance Explainability

`explain_document(doc_id, matrix, vectorizer, top_n=5)` returns the `top_n` non-zero TF-IDF features of a document as a list of `(term, weight)` tuples:

```python
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    row = matrix[doc_id].toarray().ravel()
    terms = vectorizer.get_feature_names_out()
    nonzero = np.flatnonzero(row)
    top = nonzero[np.argsort(row[nonzero])[::-1][:top_n]]
    return [(str(terms[j]), round(float(row[j]), 4)) for j in top]
```

**Audit of Document #412:**

True category: `rec.autos`. Assigned cluster: **0** (centroid top words: know, like, don, just, does, use, thanks, good).

| Term | TF-IDF weight |
|---|---|
| discussion | 0.4524 |
| feel | 0.3989 |
| days | 0.3855 |
| need | 0.2885 |
| meant | 0.2667 |

> Post: *"Could we plase cease this discussion. I fail to see why people feel the need to expound upon this issue for days and days on end. These areas are not meant for this type of discussion…"*

**Why it was routed to Cluster 0:** The post contains no automotive vocabulary at all. It is a meta-complaint about a thread. Every one of its five strongest features is a generic conversational word, so its vector lies closest to the generic "conversation" centroid (Cluster 0) rather than the autos centroid (car, engine, ford, dealer). The explanation therefore lets the auditor see immediately that this is not a model error on automotive content: the post is off-topic and should be flagged for human review.

**Why TF-IDF over a black-box LLM in regulated compliance (3 sentences):**
1. TF-IDF is fully deterministic and traceable: every routing decision can be reproduced exactly and broken down into specific words with exact numeric weights, as shown above for Document #412, which is the kind of evidence an auditor can verify and file.
2. A black-box LLM produces decisions from billions of opaque parameters, may give different outputs for the same input, and can only offer post-hoc explanations that are not guaranteed to reflect how it actually decided, which fails audit and traceability requirements for patient adverse-event reports.
3. TF-IDF is therefore the right transparent baseline: it is cheap, keeps sensitive patient text in-house, and its failure modes (like filler-word clusters or signature noise) are visible and fixable, so any more complex model can later be justified against a baseline that regulators can fully inspect.
