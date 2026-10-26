# Lab 15 — Problem-Solving Answers

**Student:** Syed Shahbaz Ali Shah — 24i-8002

All numbers below are produced by Step 4 of `student_starter_script.py` (run `python student_starter_script.py`).
Corpus: 3,703 posts from `sci.electronics`, `rec.autos`, `comp.sys.mac.hardware` and `sci.med`.

## Part 1 summary

| Step | Result |
|---|---|
| 1. Cleaning | Stemming cuts words to non-words (`cable → cabl`, `ignition → ignit`, `someone → someon`). Lemmatizing with `pos='v'` keeps real words and fixes irregular verbs (`left → leave`, `locked → lock`). |
| 2. BoW vs TF-IDF | Both matrices are 3,703 × 4,000. TF-IDF sparsity is **99.04%**: each post uses fewer than 1% of the vocabulary. |
| 2. Top features | `know, like, just, car, don, does, use, thanks, good, think`. Nine of the ten are conversational filler (see Challenge 1). |
| 3. Clustering | The baseline KMeans gives clusters of 2,381, 699, 552 and 71 posts. Only two clusters are clean topics (Mac = `mac, apple, drive`; Autos = `car, engine, ford`). |

---

## Challenge 1: Conversational filler & numeric noise

**Why `know` still gets a high weight.** IDF only *dampens* common words; it never removes them.
A word in 25% of posts gets IDF = log(1/0.25) = **1.39**. Scikit-learn's smoothed version gives
1 + ln((1+N)/(1+0.25N)) = **2.39**. That is about 28% of the maximum (8.52, for a word in a single post),
not zero. Filler words also have a high **TF**, because people type them in almost every post. A
moderate IDF multiplied by a high TF, repeated across ~800 posts, puts them at the top of the average
ranking:

| Word | % of posts | IDF | Avg TF-IDF rank |
|---|---|---|---|
| know | 22.7% | 2.48 | **#1** of 4,000 |
| like | 23.4% | 2.45 | #2 |
| just | 21.2% | 2.55 | #3 |
| does | 17.3% | 2.75 | #6 |
| thanks | 13.0% | 3.04 | #8 |

NLTK and scikit-learn stopword lists target grammatical words (`the`, `of`), not forum chatter. So
these words pass through, and the baseline produces a 2,381-post "chatter" cluster whose top words
are `know, like, don, just, does, use, thanks, good`.

**Refinement.** I added `token_pattern=r'\b[a-zA-Z]{3,}\b'`, which drops numbers and 1–2 letter
fragments; `max_df=0.5`; and ~50 custom stopwords (filler such as `know/like/thanks`, plus email noise
such as `edu/com/mail`). I also found a second noise source: a **signature block repeated verbatim in
72 `sci.med` posts** ("Gordon Banks N3JXP … *Skepticism is the chastity of the intellect* …
geb@cadre.dsl.pitt.edu"). Its words took over a whole cluster in the baseline (cluster 0: `cadre, dsl,
n3jxp, chastity, intellect`). Adding those signature words to the stopword list, and using `sublinear_tf=True`
(1 + log tf), produced:

| Cluster | Baseline top words | Cleaned top words |
|---|---|---|
| 0 | cadre, dsl, n3jxp, chastity, intellect *(signature)* | car, cars, engine, ford, dealer, miles, oil, price → **Autos** |
| 1 | know, like, don, just, does, use, thanks *(chatter)* | doctor, disease, medical, msg, cause, treatment, patients, pain → **Med** |
| 2 | mac, apple, drive, monitor, card, **does, thanks, know** | mac, apple, drive, monitor, card, scsi, video, disk → **Mac** |
| 3 | car, cars, engine, **just**, new, ford, dealer, **like** | looking, power, phone, circuit, current, line, number, data → **Electronics** |

Each cleaned cluster now corresponds to one real category. Agreement with the true labels
(Adjusted Rand Index) changes step by step as follows:

| Step | ARI |
|---|---|
| Baseline | 0.232 |
| + token_pattern, max_df, filler words | 0.159 |
| + signature words removed | 0.249 |
| + sublinear_tf | **0.314** |

Removing filler alone made the score *worse* (0.159). The signature cluster was still taking one of
the four slots, so the remaining topics had to share three clusters. Each noise source has to be
fixed for the cleaning to pay off.

---

## Challenge 2: Hardware vocabulary overlap

**Crosstab, baseline clusters:**

| true \ cluster | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| comp.sys.mac.hardware | 0 | 279 | **629** | 2 |
| rec.autos | 0 | 377 | 2 | **526** |
| sci.electronics | 0 | **850** | 66 | 24 |
| sci.med | 71 | **875** | 2 | 0 |

In the baseline, Electronics never gets a cluster of its own. It is absorbed into the chatter cluster
together with Med.

**Which two categories overlap most?** The KMeans assignments mix the overlap with the noise problems
above. So I also measured it directly: the cosine similarity between each true category's average
TF-IDF vector.

| | mac.hardware | autos | electronics | med |
|---|---|---|---|---|
| mac.hardware | 1 | 0.394 | **0.566** | 0.364 |
| autos | | 1 | 0.517 | 0.439 |
| electronics | | | 1 | 0.491 |

**`sci.electronics` ↔ `comp.sys.mac.hardware` overlap the most (0.566).** In the cleaned clustering,
**341 posts** are routed between the two: 297 Mac posts land in the Electronics cluster and 44
Electronics posts land in the Mac cluster.

**Shared technical words pulling them together:** `power`, `chip`, `software`, `computer`, `pin`,
`data`, `run`. Both groups talk about power supplies, chips and pins. Many misrouted Mac posts are
about generic hardware (a scanner roller, display pixels, an optical drive) with no Mac-specific word
in them.

**Do bigrams help?** Partly. Bigrams surface phrases that clearly belong to one side:

- **Mac:** `hard drive`, `video card`, `pds slot`, `clock speed`, `mac plus`, `floppy drive`, `vram simms`
- **Electronics:** `radio shack`, `radar detector`, `lead acid`, `pink noise`, `phone company`, `copy protection`

A single word like `card`, `drive` or `power` is ambiguous; `video card` or `lead acid` is not. Bigrams
lower the similarity between the two categories from **0.566 to 0.551**, and they make the cluster
labels far easier for a moderator to read. However, they did **not** improve routing accuracy here.
Running KMeans with 2 clusters on just these two categories reaches **80.4%** with unigrams and
**80.1%** with uni+bigrams (averaged over 5 seeds). Most bigrams occur in only a handful of posts, so
they barely move the centroids. Bigrams help explain the clusters more than they help separate them.

---

## Challenge 3: Rant length bias

| Post | Words | ‖x‖₂ in `X_bow` | ‖x‖₂ in `X_tfidf` |
|---|---|---|---|
| Longest (#310) | 11,765 | **276.29** | 1.00 |
| Shortest non-empty (#2721) | 4 | **3.00** | 1.00 |

(5 posts become all-zero vectors once stopwords and `min_df` are applied, so I skipped them when
choosing the shortest post.)

**Why KMeans fails on raw BoW.** KMeans assigns points by Euclidean distance. In raw counts, a
vector's length grows with the length of the post: the correlation between post length and BoW norm
is **0.93**. A 12,000-word rant is therefore far from *every* other post simply because it is long,
not because of its topic. On BoW, KMeans gave clusters of **3,692, 8, 2 and 1** posts. Three of the
four clusters only isolate the longest rants (median lengths 4,421–11,765 words, against a corpus
median of 83), and the ARI is **0.000**, i.e. no better than random.

**How TF-IDF fixes it.** `TfidfVectorizer` applies L2 normalization by default (`norm='l2'`), so every
post has length exactly 1.0. Euclidean distance between unit vectors is a direct function of cosine
similarity (‖a−b‖² = 2 − 2·cos θ). Clustering therefore compares *which words* a post uses and in what
proportions, not how many words it has. That alone takes the cluster sizes to 2,381 / 699 / 552 / 71
and the ARI to 0.232.

---

## Challenge 4: Regulated compliance explainability

`explain_document(412, X_tfidf, tfidf_vec, top_n=5)` returns a list of (term, weight) pairs:

| Term | TF-IDF weight |
|---|---|
| discussion | 0.4524 |
| feel | 0.3989 |
| days | 0.3855 |
| need | 0.2885 |
| meant | 0.2667 |

**What the audit shows.** Post #412 reads: *"Could we please cease this discussion … These areas are
not meant for this type of discussion."* It is a meta-complaint from `rec.autos`, not a medical
report. None of its top terms is about any topic, and its distances to the four centroids are almost
equal (0.992, 1.005, 1.006, 1.186). It was routed to cluster 1, the generic chatter cluster, simply
because that centroid is the least specific. The explanation itself shows the routing has low
confidence and should go to a human reviewer.

**Why TF-IDF over a black-box LLM (3 sentences).** Every routing decision made with TF-IDF and KMeans
can be traced to the exact terms, weights and centroid distances shown above, so an auditor can
reproduce it by hand, and the same input always gives the same output. An LLM's decision comes from
billions of parameters, can change between model versions or between runs, and gives only a
generated rationale that is not guaranteed to reflect how it actually decided. That is not enough for
regulated adverse-event workflows, where each decision must be documented, reproducible and
defensible. As shown with post #412, the transparent baseline also reveals when it is uncertain
(near-equal centroid distances, no topical terms), so those cases can be flagged for human review
instead of being misrouted silently.
