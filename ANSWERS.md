# Lab 15 – Answers

All numbers below come from running `student_starter_script.py` (3,703 posts after the >50-character filter, `random_state=42`).

## Challenge 1: Conversational Filler & Numeric Noise

**Why `know` still gets a large weight.** TF-IDF = TF × log(N / DF). The IDF only penalises a word logarithmically, so a word in about 25% of posts is not pushed anywhere near zero. In the baseline vectorizer `know` appears in 22.7% of posts and has idf = 2.48, `like` 23.4% / 2.45, `just` 21.2% / 2.55, `thanks` 13.0% / 3.04. Their weights are only slightly below those of genuine topic words (rarer terms such as `scsi` have a higher idf but a low TF). Because these words are common and frequently repeated inside a post (high TF), they still accumulate a large mean weight in the K-Means centroids. NLTK/scikit-learn stopword lists contain function words (`the`, `and`), not conversational filler such as `know`, `thanks` or `use`.

**Fix.** `TfidfVectorizer(token_pattern=r'\b[a-zA-Z]{3,}\b', max_df=0.5, stop_words=<english + custom filler>)`. The token pattern drops numbers and 1–2 letter fragments. `max_df=0.5` removes terms in more than half the posts, and the custom list removes filler such as `know`, `like`, `just`, `think`, `does`, `use`, `thanks`, `don`, `ve`, `ll`.

**How the cluster top words change.**

| | Baseline | After filtering |
|---|---|---|
| Autos | car, cars, engine, new, **just**, ford, dealer, **like** | car, cars, engine, new, oil, ford, dealer, miles |
| Mac | mac, apple, drive, monitor, card, **thanks, does, know** | mac, apple, drive, monitor, card, scsi, simms, video |
| Generic blob | **know, like, don, just, does, use, thanks, good** | people, time, edu, used, information, make, new, help |
| Small outlier | dsl, cadre, n3jxp, chastity, shameful, … | geb, pitt, cadre, dsl, chastity, intellect, … |

The autos and Mac clusters now show domain vocabulary (`oil`, `miles`, `scsi`, `simms`, `video`) instead of filler. The remaining "generic" cluster is still the least interpretable (medical/electronics posts mixed together), and the small cluster (~72 posts) is a set of `sci.med` posts that share the same signature text, not a real topic.

## Challenge 2: Hardware Vocabulary Overlap

Baseline `pd.crosstab(true_label, cluster)`:

| true_label | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| comp.sys.mac.hardware | 269 | 0 | 2 | 639 |
| rec.autos | 396 | 0 | 507 | 2 |
| sci.electronics | 860 | 0 | 20 | 60 |
| sci.med | 875 | 71 | 0 | 2 |

Cluster 0 is a catch-all containing most `sci.electronics` and `sci.med` posts, and 269 `comp.sys.mac.hardware` posts. Among the hardware categories, **`sci.electronics` and `comp.sys.mac.hardware`** overlap most: both are technical, question-and-answer style posts about devices. (`sci.electronics` and `sci.med` also merge, but that is the generic-vocabulary cluster rather than shared topic words.)

**Shared technical words** (highest min-count across the two categories): `power`, `cable`, `data`, `software`, `run`, `problem`, `using`, `work`. Words such as `power`, `cable`, `data`, `software` appear naturally in both electronics and Mac hardware questions, so unigrams alone pull those posts together.

**Do bigrams help?** With `ngram_range=(1, 2)` the Mac cluster went from 639 to 634 correct Mac posts and `sci.electronics` from 860/60/20 to 855/19/66 – essentially no change in this run. Bigrams such as `power supply`, `hard drive`, `mac se`, `disk drive` are more specific than their single words, so they should separate "power" (electronics) from "power mac" (Apple), but K-Means with only 4 clusters on a 6,000-feature space is dominated by the big generic-vocabulary cluster, so the effect is small here. More clusters (e.g. k=6–8) or supervised classification would be needed to exploit them fully.

## Challenge 3: Rant Length Bias

Longest post: #310 (11,765 words). Shortest post: #1926 (a single long token).

| Matrix | ‖longest‖₂ | ‖shortest‖₂ |
|---|---|---|
| `X_bow` | 276.241 | 0.000 |
| `X_tfidf` | 1.000 | 0.000 |

(The shortest post's vector is all zeros: its one token was removed by stopword/`min_df` filtering – it is a "noise" post that no model can place.)

**Why K-Means fails on raw BoW.** Raw counts grow with document length, so the longest post has a vector more than 270× larger than a typical short one. K-Means uses Euclidean distance, so distance is dominated by length instead of topic: a 20-word question about Macs is "far" from a 2,000-word rant about Macs, and long documents of different topics are drawn together/outlying simply because of their magnitude. Centroids are also pulled toward the long posts.

**How TF-IDF fixes it.** `TfidfVectorizer` L2-normalises every row (`norm='l2'`), so every non-empty document has norm exactly 1.0 (see the table). Euclidean distance between unit vectors is a monotonic function of cosine similarity, so K-Means now compares word *proportions* (direction) and ignores post length. IDF additionally down-weights words that appear everywhere.

## Challenge 4: Regulated Compliance Explainability

`explain_document(doc_id=412, top_n=5)` returns the five largest non-zero TF-IDF weights for the document:

| term | weight |
|---|---|
| discussion | 0.4524 |
| feel | 0.3989 |
| days | 0.3855 |
| need | 0.2885 |
| meant | 0.2667 |

Document #412 is actually a `rec.autos`-labelled post (a complaint asking people to stop a long discussion thread), and it was assigned to cluster 0, the generic-vocabulary cluster, because none of its words are topic-specific. The same function works for any document, including `sci.med` reports.

**Why TF-IDF over a black-box LLM as a transparent baseline (3 sentences).**
1. Every routing decision can be traced to exact, human-readable terms and numeric weights, as printed above, so an auditor can reproduce and challenge why a report was assigned to a queue.
2. The pipeline is deterministic (fixed `random_state`, no hidden prompt or model drift) and runs locally, so no patient text is sent to a third-party service and results are repeatable for an audit trail.
3. An LLM can give a fluent but unverifiable rationale, whereas TF-IDF's explanation *is* the computation, which is what regulated workflows need as a baseline before any more complex model is justified.
