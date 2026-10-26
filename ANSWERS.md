# Lab 15 – Problem-Solving Answers

All numbers below come from running `lab15_starter.py` (identical to `student_starter_script.py`) on the 4-category 20 Newsgroups subset. 3,703 posts remain after the >50-character filter. Full console output is in `run_output.txt`.

**Baseline pipeline summary**

| Item | Result |
|---|---|
| TF-IDF matrix | 3,703 × 4,000, **99.04 % sparse** |
| Top 10 words by average TF-IDF | know, like, just, don, does, car, use, thanks, good, think |
| Baseline K-Means clusters (top words) | 0: know, like, don, just, does… (2,400 posts, generic chatter) · 1: cadre, dsl, n3jxp, chastity, surrender… (71 posts, a signature block) · 2: car, cars, engine, ford, dealer… (529) · 3: mac, apple, drive, monitor, card… (703) |
| Baseline purity | 0.565 |

---

## Challenge 1: Conversational Filler & Numeric Noise

**Why does `know` still get a large weight?**
In TF-IDF, $\text{TFIDF}_{t,d} = \text{TF}_{t,d} \times \log(N/\text{DF}_t)$. The IDF factor is the only thing that penalises common words, and it shrinks slowly because it is a logarithm. A word in 25 % of posts gets $\log(1/0.25) = \log 4 \approx 1.39$. That is lower than a rare word's IDF but still far from zero. Only a word in 100 % of posts gets $\log 1 = 0$. In our corpus, `know` appears in **22.7 %** of posts: plain IDF = 1.48, and scikit-learn's smoothed IDF $\ln\frac{1+N}{1+\text{DF}}+1$ = **2.48**. The smoothed form adds +1, so no word is ever weighted 0.

`know` is also spread across all four topics, so it appears in thousands of documents with a moderate TF each time. When you average over the corpus, these many small weights add up. That is why `know`, `like`, `just`, `don` and `does` top the baseline top-10 chart. They also dominate the 2,400-post "catch-all" cluster 0 because it has no topical words of its own. Standard stop-word lists only cover function words ("the", "and"). Conversational fillers like these are left in.

**Refinement.** The refined vectorizer uses:
- `token_pattern=r'\b[a-zA-Z]{3,}\b'`: removes numbers, stray digits, and 1–2-letter fragments such as `ve`, `ll` and `s`.
- `max_df=0.5`: removes any term that appears in more than half of the posts.
- Custom stop words: the English list plus fillers (`know, like, just, think, does, use, thanks, don, want, good, need, really, people, time, edu, com, mail, …`). It also includes a repeated sci.med signature block (`geb, cadre, dsl, pitt, chastity, shameful, surrender, intellect, skepticism`). That block survives `remove=('footers',)` and formed a whole 71-post "topic" in the baseline.

Note: `max_df=0.5` alone does almost nothing here, because no filler word goes above ~23 % DF. The explicit stop-word list does the real work.

**How the cluster top words change:**

| Cluster | Baseline | Cleaned |
|---|---|---|
| A | know, like, don, just, does, use, thanks, good | **doctor, msg, soon, medical, disease, pain, patients, food** (sci.med) |
| B | cadre, dsl, n3jxp, chastity, surrender, intellect… | **power, looking, phone, high, current, circuit, number** (sci.electronics) |
| C | car, cars, engine, **new, just**, ford, dealer, **like** | car, cars, engine, dealer, **oil, miles**, ford, **price** (rec.autos) |
| D | mac, apple, drive, monitor, card, **thanks, does, know** | mac, apple, drive, monitor, card, **scsi, video, simms** (mac hardware) |

The filler-word cluster and the signature cluster both disappear. In their place we get a real **medical** cluster and an **electronics** cluster, so the four clusters now line up with the four domains. Purity rises from **0.565 to 0.634**.

---

## Challenge 2: Hardware Vocabulary Overlap

**Baseline `pd.crosstab(df['true_label'], df['cluster'])`:**

```
cluster                  0   1    2    3
comp.sys.mac.hardware  269   0    2  639
rec.autos              396   0  507    2
sci.electronics        860   0   20   60
sci.med                875  71    0    2
```

**Cleaned vectorizer:**

```
cluster_clean            0    1    2    3
comp.sys.mac.hardware    1  348  561    0
rec.autos                5  554    2  344
sci.electronics          2  879   45   14
sci.med                563  384    1    0
```

**Most-overlapping pair: `sci.electronics` and `comp.sys.mac.hardware`.** In the baseline, the Mac cluster (3) takes in 60 electronics posts, and 269 Mac posts fall into the generic cluster alongside 860 electronics posts. After cleaning, **348 Mac posts** land in the electronics-dominated cluster and **45 electronics posts** land in the Mac cluster. In total, 393 hardware posts end up in the other hardware queue. That is the largest error between any two specific topics. (Autos and med also leak into the electronics cluster, but mainly because it acts as the leftover "general" queue.)

**Shared technical words pulling them together.** These terms carry high average TF-IDF in *both* categories: **power, software, data, chip, computer, board, pin, run, buy**. Among the misclustered posts, the strongest terms are **mouse, keyboard, printer, modem, card, software, computer, fax, price**. These are peripheral and component words that a Mac owner and an electronics hobbyist both use. Examples:
- doc 167 (electronics → baseline Mac cluster): a "CD-ROM based system … 4-disk changer hooked to a dedicated MS-DOS computer".
- doc 257 (electronics → baseline Mac cluster): wiring a "Sun CD-ROM drive" to an amplifier.
- doc 11 (Mac → cleaned electronics cluster): fixing a scanner's "rubber carriage roller".

Cleaning moves docs 167 and 257 back to the electronics cluster. Doc 11 stays misplaced because its vocabulary (tape, roller, scanner) has nothing Mac-specific.

**How bigrams help (`ngram_range=(1, 2)`).** As single words, `power`, `card`, `drive` and `supply` are ambiguous. As phrases they become specific to one domain. The bigram model's Mac cluster is defined by **hard drive, hard disk, video card, pds slot, clock speed**. Its electronics/general cluster is defined by **radio shack, radar detector, laser printers, mailing list**. Purity rises to **0.665** (vs. 0.634 for unigrams only), and Mac posts in the correct cluster go from 561 to **625**.

Bigrams help but do not solve the problem fully: 284 Mac posts still sit in the general/electronics cluster. Short Mac posts often contain no Mac-specific phrase at all, and `power supply` still appears on both sides.

---

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

| Post | Words | ‖x‖₂ in `X_bow` | ‖x‖₂ in `X_tfidf` |
|---|---|---|---|
| Longest (doc 310) | 11,765 | **276.24** | **1.000** |
| Shortest non-empty (doc 2721) | 4 | **3.00** | **1.000** |

The raw-count vector of the rant is **92× longer** than the short post's. The mean BoW norm climbs steadily with post length: 3.42 → 5.44 → 7.71 → 17.06 across the length quartiles. (The very shortest post by word count has *no* vocabulary terms left after filtering, so its vector is all zeros in both matrices. That is why the table uses the shortest *non-empty* post.)

**Why K-Means fails on raw BoW.** K-Means assigns points to centroids by **Euclidean distance**. With raw counts, a vector's length mostly reflects how long the post is, not what it is about. A 2,000-word rant about cars and a 2,000-word rant about medicine are both far from the origin. Every short post sits close to the origin, whatever its topic. The squared-distance objective is dominated by the few huge vectors, so K-Means spends its clusters isolating outliers. Our run shows exactly this: K-Means on `X_bow` produced clusters of size **{8, 3692, 2, 1}**. The three small clusters hold posts with median lengths of 4,421, 7,538 and 11,765 words. Everything else is lumped together, and purity falls to **0.257**, which is chance level for 4 classes.

**How `TfidfVectorizer` fixes it.** By default (`norm='l2'`), scikit-learn divides every row by its Euclidean norm, so every document sits on the unit sphere (‖x‖₂ = 1, as the table shows). For unit vectors, $\|a-b\|^2 = 2 - 2\cos(a,b)$. Euclidean K-Means then effectively clusters by **cosine similarity**, which depends on the *proportions* of the words used rather than the total count. A 20-word question and a 2,000-word rant about the same topic now point in the same direction. TF-IDF K-Means gives topical clusters with purity 0.565 at baseline and 0.634 after cleaning. IDF also helps a second way: it down-weights the common words that long rants repeat most.

---

## Challenge 4: Regulated Compliance Explainability

`explain_document(doc_id, matrix, vectorizer, top_n=5)` reads the document's sparse row, sorts its non-zero entries by weight, and returns `(term, weight)` tuples:

```
Document #412 — assigned to baseline cluster 0
  discussion   0.4524
  feel         0.3989
  days         0.3855
  need         0.2885
  meant        0.2667
Euclidean distance to centroids: {0: 0.9918, 1: 1.1859, 2: 1.0069, 3: 1.0051}
```

**Audit finding.** Document #412 reads: *"Could we please cease this discussion … people feel the need to expound upon this issue for days and days … take your thought elsewhere."* Its five strongest terms (discussion, feel, days, need, meant) carry no medical, automotive or hardware vocabulary. It therefore lies nearest (distance 0.992) to cluster 0, the generic-conversation centroid, rather than to the car (1.007) or Mac (1.005) centroids. Its ground-truth label is actually `rec.autos`, and nothing in it describes an adverse event. The correct audit response is "routed to the general queue because it is off-topic meta-discussion", and the weights above prove it. In a real adverse-event workflow this is exactly the kind of misrouting an auditor needs to be able to see.

**Why TF-IDF over a black-box LLM (3 sentences).**
TF-IDF routing is fully deterministic and reproducible: every decision comes down to a sparse vector of named terms with exact numeric weights and a measurable distance to each cluster centroid, so an auditor can re-run it and get the same answer. A black-box LLM's routing depends on billions of opaque parameters, can change with prompt wording, sampling or model updates, and offers only after-the-fact explanations that may not reflect how it actually decided. In regulated settings such as pharmacovigilance or healthcare audits, where decisions must be traceable, validated and defensible, a transparent TF-IDF baseline gives a verifiable audit trail and a benchmark that any more complex model must be justified against.
