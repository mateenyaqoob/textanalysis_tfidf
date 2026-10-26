# Lab 15 Problem-Solving Answers

These answers are supported by the diagnostics in Step 4 of `lab15_starter.py` (3,703 posts after length filtering).

---

## Challenge 1: Conversational Filler & Numeric Noise

**Why `know` still gets a substantial TF-IDF weight at 25% document frequency**

Recall:

$$\text{TFIDF}_{t,d} = \text{TF}_{t,d} \times \log\left(\frac{N}{\text{DF}_t}\right)$$

If `know` appears in 25% of posts, then \(\text{DF}_t = 0.25N\) and the IDF term is only \(\log(4) \approx 1.39\). That is a mild penalty, not a knockout: IDF reaches 0 only when a term is in *every* document. Conversational fillers also have **high TF** inside a post (“I know… you know…”), so the product remains large. Scikit-Learn’s default `smooth_idf=True` formula, \(\log\frac{1+N}{1+\text{DF}_t}+1\), adds a +1 offset, so the weight never collapses even for common terms. Numbers slip through the default token pattern `(?u)\b\w\w+\b`, which treats digits as word characters.

**What changed after the cleaner vectorizer**

`TfidfVectorizer` was rebuilt with `token_pattern=r'\b[a-zA-Z]{3,}\b'` (alphabetic tokens of length ≥ 3, so stray numbers drop), `max_df=0.5`, and custom stopwords `{know, like, just, think, does, use, thanks}` on top of English stops.

Baseline cluster keywords:

| Cluster | Top words |
|---------|-----------|
| 0 (generic dump) | know, like, don, just, does, use, thanks, good |
| 1 (signature spam) | cadre, dsl, n3jxp, chastity, surrender, intellect, shameful, skepticism |
| 2 (autos) | car, cars, engine, new, just, ford, dealer, like |
| 3 (Mac) | mac, apple, drive, monitor, card, thanks, does, know |

After filtering:

| Clean cluster | Top words |
|---------------|-----------|
| 0 | dsl, cadre, chastity, shameful, intellect, surrender, skepticism, geb |
| 1 | don, good, people, time, used, edu, information, want |
| 2 | car, cars, engine, new, ford, **oil**, dealer, good |
| 3 | mac, apple, drive, monitor, card, **scsi, video, simms** |

Fillers (`know`, `like`, `just`, `does`, `thanks`) disappear from the centroids. The Mac queue picks up hardware-specific terms (`scsi`, `simms`, `video`) instead of politeness words, and autos picks up `oil`. `max_df=0.5` alone would **not** have removed `know` (it is only in ~25% of posts); the custom stoplist was required.

---

## Challenge 2: Hardware Vocabulary Overlap

Baseline `pd.crosstab(df['true_label'], df['cluster'])`:

| true_label | Cluster 0 (generic) | Cluster 1 (spam) | Cluster 2 (autos) | Cluster 3 (Mac) |
|---|---:|---:|---:|---:|
| comp.sys.mac.hardware | 269 | 0 | 2 | **639** |
| rec.autos | 396 | 0 | **507** | 2 |
| sci.electronics | **860** | 0 | 20 | 60 |
| sci.med | **875** | 71 | 0 | 2 |

The two categories that overlap the most are **`sci.electronics` and `comp.sys.mac.hardware`**. Electronics has no dedicated cluster: 860 of its posts collapse into the generic filler cluster alongside 269 Mac posts, and another 60 electronics posts are pulled into the Mac-dominated cluster 3.

Inspecting those 60 misclustered electronics posts, the highest-weight technical terms they share with Mac hardware are **`drive`**, **`card`**, **`monitor`**, plus related component words (`board`, `ram`, `mhz`, `video`). A typical electronics post assigned to the Mac queue discusses CD-ROM drives, cards, and monitors — the same component vocabulary Mac users use — so unigram K-Means cannot tell a generic SCSI/video card from a Mac NuBus card.

**How bigrams help:** `ngram_range=(1, 2)` adds collocations such as `hard drive`, `power mac`, `apple macintosh`, or `scsi card`. A unigram `drive` is ambiguous (disk drive vs. car drive vs. voltage drive); a bigram is not. In this run the Mac centroid still surfaces Mac-specific tokens (`scsi`, `simms`, `video`) while the signature-spam cluster is cleanly captured by phrases (`shameful surrender`, `geb cadre`, `dsl pitt`). Distinctive multi-word product names pull Mac hardware away from general electronics even when they share the unigrams `card`, `drive`, and `board`.

---

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

Longest post: id 310, 72,161 characters. Shortest post: id 1159, 52 characters.

| Representation | \(\|x_{\text{longest}}\|_2\) | \(\|x_{\text{shortest}}\|_2\) |
|---|---:|---:|
| Bag of Words (`X_bow`) | **276.24** | **2.00** |
| TF-IDF (`X_tfidf`) | **1.00** | **1.00** |

K-Means minimizes Euclidean distance to centroids. Raw BoW counts grow with document length, so a 2,000-word rant has a huge vector magnitude and dominates the centroid, while a 20-word question sits near the origin and is treated as a different “topic” even if it uses the same words. That is a length artifact, not a topical one.

Scikit-Learn’s `TfidfVectorizer` applies **L2 normalization** (`norm='l2'`) after TF-IDF weighting, so every document vector has \(\|x\|_2 = 1\) (except all-zero rows). Euclidean distance then depends only on the **direction** of the vector — the mix of distinctive terms — not on how long the user ranted. That is why the two norms collapse to 1.00 / 1.00 and why TF-IDF is the safer input to K-Means for variable-length forum text.

---

## Challenge 4: Regulated Compliance Explainability

`explain_document(doc_id=412, top_n=5)` on the baseline TF-IDF matrix:

| Rank | Term | TF-IDF weight |
|---:|---|---:|
| 1 | discussion | 0.452421 |
| 2 | feel | 0.398890 |
| 3 | days | 0.385506 |
| 4 | need | 0.288483 |
| 5 | meant | 0.266672 |

Document #412 is a short meta-post (“Could we please cease this discussion…”) that K-Means placed in the generic cluster (cluster 0). An auditor can point to these five non-zero weights and see that routing was driven by conversational meta-language, not by medical or hardware terminology.

**Why TF-IDF is preferred over a black-box LLM as a transparent baseline (3 sentences).**  
TF-IDF produces an explicit, reproducible document-term matrix in which every routing score is a finite sum of named term weights, so an auditor can recover the exact features that sent Document #412 to its cluster. A black-box LLM may generate a fluent rationale that is not faithful to the true decision path and cannot be re-derived from a frozen, inspectable vector. In a regulated adverse-event workflow that requires an audit trail, a fully explainable TF-IDF + K-Means baseline is the appropriate first control even if a more accurate model is evaluated later.
