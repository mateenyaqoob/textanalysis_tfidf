# Lab 15 — Problem-Solving Answers

**Student:** Khizar Mohammad Hashmi , 26I-8009
**Script:** `student_starter_script.py` (Step 4 generates every number quoted below; `random_state=42` throughout)

**Corpus:** 3,703 posts after the >50-character filter. Baseline TF-IDF matrix: 3,703 × 4,000, **99.04% sparse**.

---

## Challenge 1: Conversational Filler & Numeric Noise

**Why `know` still gets a large weight.** IDF only penalises a term heavily when it appears in *almost every* document. With the textbook formula, a word in 25% of posts gets `log(N / 0.25N) = log 4 ≈ 1.39`. That is far from zero. Scikit-Learn's smoothed IDF, `ln((1+N)/(1+DF)) + 1`, gives **≈ 2.39** for the same word. The measured values agree: `know` appears in 22.7% of posts with idf = 2.48, `like` in 23.4% with 2.45, and `just` in 21.2% with 2.55. The TF factor makes it worse because filler words repeat within a post. In short posts each repeat also takes a large share of the L2-normalised vector. The result: `know`, `like`, `just`, `don`, `does`, `use`, `thanks`, `good` and `think` fill 9 of the top 10 baseline features (`top_features.png`). The baseline vocabulary also kept **212 numeric tokens** (`00`, `000`, `01`, …).

**Refinement.** I used `token_pattern=r'\b[a-zA-Z]{3,}\b'` to drop numbers and 1–2 letter fragments. I added `max_df=0.5` and Scikit-Learn's English stopwords plus three custom groups:

1. conversational filler (`know, like, just, think, does, use, thanks, …`)
2. email artefacts (`edu, com, mail, info, writes, …`)
3. one sci.med signature block (`gordon, banks, n3jxp, skepticism, chastity, intellect, geb, cadre, dsl, pitt, …`)


| Cluster | Baseline top words                                                 | Cleaned top words                                                         |
| ------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| 0       | mac, thanks, apple, does, drive, know, monitor, card               | power, looking, phone, lot, current, circuit, radio, try*(electronics)*   |
| 1       | like, don, just, know, use, good, think, people*(filler sink)*     | car, cars, engine, dealer, oil, miles, ford, speed*(autos)*               |
| 2       | dsl, cadre, chastity, n3jxp, intellect, surrender, …*(signature)* | msg, doctor, medical, disease, food, patients, pain, treatment*(medical)* |
| 3       | car, cars, engine, just, new, dealer, ford, like                   | mac, apple, drive, monitor, card, scsi, video, software*(Mac)*            |

In the baseline, two of the four clusters are wasted. One is a sink for 2,183 filler-heavy posts. The other holds the 71 posts that carry Gordon Banks' signature, which `remove=("footers")` failed to strip. After cleaning, there is **one clear topic cluster per domain**, and the Adjusted Rand Index against the true labels rises from **0.250 to 0.269**.

Removing the filler alone was **not** enough. That run scored ARI 0.176 because the signature cluster stayed and `edu/mail/info` formed a new junk cluster. Stopword lists need to be audited against what the clusters actually show, not only against a generic English list.

---

## Challenge 2: Hardware Vocabulary Overlap

**Most-overlapping pair: `sci.electronics` ↔ `comp.sys.mac.hardware`.** The cleaned-model crosstab:


| true label \ cluster  | 0 (elec) | 1 (autos) | 2 (med) | 3 (Mac) |
| --------------------- | -------- | --------- | ------- | ------- |
| comp.sys.mac.hardware | **300**  | 0         | 0       | 610     |
| rec.autos             | 493      | 405       | 5       | 2       |
| sci.electronics       | 853      | 16        | 5       | **66**  |
| sci.med               | 375      | 0         | 571     | 2       |

**366 posts** cross between the Electronics and Mac clusters (300 + 66). The same pair also overlaps in the baseline: 164 Electronics posts sit in the Mac cluster, and both groups leak into the filler cluster.

**Shared technical words pulling them together** (high mean TF-IDF in *both* classes, and dominant inside the misclustered posts): `power`, `chip`, `board`, `computer`, `software`, `card`, `data`, `ram`, `monitor`, `keyboard`, `mouse`, `printer`, `pin`. Example misroutes:

- **#6:** Mac post about a SyQuest/optical drive purchase, routed to Electronics.
- **#18:** Mac post comparing a IIci, routed to Electronics because its generic `buy / run / fast` vocabulary matches the catch-all Electronics centroid.
- **#312:** Electronics post about an A/D board on an AT bus (`computer, board, mhz`), close to Mac hardware.

**Bigrams.** In principle `ngram_range=(1, 2)` turns ambiguous unigrams into domain-specific phrases. The highest-weighted bigrams split cleanly: Electronics gets `power supply`, `radio shack`, `radar detector`, `phone number`, and Mac gets `hard drive`, `hard disk`, `video card`, `copy protection`. Bigrams also tell `power supply` (electronics) apart from `power mac` (Mac).

**Measured result:** on this corpus bigrams gave **no gain**. Electronics↔Mac swaps went from 366 to 370, the pair-ARI from 0.351 to 0.349, and the 4-class ARI from 0.269 to 0.233. With `min_df=5`, each post contains only a few surviving bigrams, so they carry little centroid mass. Doubling the vocabulary to 8,000 also spreads the L2 norm thinner, so unigram overlap still dominates the distances. Bigrams help most when combined with a supervised or feature-selected model, or with `sublinear_tf`/larger corpora. Unsupervised K-Means alone does not benefit.

---

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)


| Post                                      | Words  | ‖x‖₂ in`X_bow` | ‖x‖₂ in`X_tfidf` |
| ----------------------------------------- | ------ | ----------------- | ------------------- |
| Longest (#310)                            | 11,765 | **276.28**        | 1.000               |
| Shortest with ≥1 vocabulary term (#2721) | 4      | **3.00**          | 1.000               |

Across the corpus, BoW norms range from 1.0 to 276.3 and correlate with post length at **r = 0.925**.

**Why K-Means fails on raw counts.** K-Means minimises squared Euclidean distance, and `‖a − b‖² = ‖a‖² + ‖b‖² − 2a·b`. With raw counts, the `‖a‖²` term grows with post length and swamps the topical `a·b` term. A 2,000-word rant is far from *every* centroid, whatever it is about. The algorithm therefore spends its clusters isolating long posts: on `X_bow`, the cluster sizes were **3,692 / 8 / 1 / 2** with **ARI ≈ 0.000** (no topical signal at all). It clusters by length, not topic.

**How TF-IDF fixes it.** `TfidfVectorizer` applies `norm='l2'` by default, so every row has ‖x‖₂ = 1. Then `‖a − b‖² = 2(1 − cos θ)`, and Euclidean K-Means becomes cosine-similarity clustering. Distance depends only on the *direction* of the vector (the mix of terms), not its magnitude. A 4-word question and an 11,765-word rant about the same topic land near each other. This is why the TF-IDF baseline reaches ARI 0.250.

---

## Challenge 4: Regulated Compliance Explainability

`explain_document(doc_id, matrix, vectorizer, top_n=5)` reads the stored non-zero entries of the document's sparse row. It returns the top-`n` `(term, weight)` tuples and also prints them.

**Document #412** (baseline `X_tfidf`):


| Rank | Term       | TF-IDF weight |
| ---- | ---------- | ------------- |
| 1    | discussion | 0.4524        |
| 2    | feel       | 0.3989        |
| 3    | days       | 0.3855        |
| 4    | need       | 0.2885        |
| 5    | meant      | 0.2667        |

**Audit trace.** In this dataset, post #412 is a `rec.autos` meta-complaint: *"Could we please cease this discussion… people feel the need to expound upon this issue for days and days…"*. It contains no domain vocabulary. Its top terms are conversational, so it was routed to baseline **Cluster 1**, the generic "filler" cluster (top words: *like, don, just, know, use*). The auditor can see exactly why, and can see that the routing is a known failure mode that the Challenge 1 cleaning addresses, not a hidden judgement.

**Why TF-IDF over a black-box LLM as the compliance baseline (3 sentences):**
TF-IDF is fully deterministic and decomposable: every routing decision reduces to a finite list of named terms, with exact weights computed from a published formula, that an auditor can recompute by hand. An LLM's decision is spread across billions of parameters, can vary with prompt wording, sampling or model version, and its post-hoc rationales are not guaranteed to reflect the actual computation. That makes it hard to reproduce or defend under regulatory scrutiny. A transparent TF-IDF baseline therefore gives a reproducible audit trail and a performance floor that any more complex model must justify exceeding.
