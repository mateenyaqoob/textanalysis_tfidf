# Lab 15 Answers: Unstructured Text Analysis & Automated Triage with TF-IDF

**Roll No:** 25i-8049
**Course:** Data Science Tools & Techniques

## Pipeline Summary

| Item | Result |
|---|---|
| Posts loaded (after removing posts ≤ 50 characters) | 3,703 |
| TF-IDF matrix shape | (3703, 4000) |
| Sparsity | 99.04% |

**Stemming vs. lemmatization (Step 1).** On the sample post about a Honda trunk release, the Porter stemmer produced non-words such as `cabl`, `releas`, `ignit` and `someon`, because it cuts word endings by rule. The WordNet lemmatizer (with `pos='v'`) kept real dictionary words (`cable`, `release`, `ignition`, `someone`) and correctly mapped `left` to `leave` and `locked` to `lock`. Lemmatized output is easier for humans to read, which matters for an explainable triage system.

**Baseline clusters (Step 3).** Only two of the four K-Means clusters matched a real topic:

| Cluster | Top words | Interpretation |
|---|---|---|
| 0 | know, like, don, just, does, use, thanks, good | No topic: a catch-all of 2,400 posts (65% of the data) |
| 1 | cadre, dsl, n3jxp, chastity, surrender, intellect, shameful, skepticism | One `sci.med` author's email signature (71 posts) |
| 2 | car, cars, engine, new, just, ford, dealer, like | Autos (95.8% pure) |
| 3 | mac, apple, drive, monitor, card, thanks, does, know | Mac Hardware (90.9% pure) |

The spatial map confirms this: the signature cluster sits far away from all other posts, while the other three clusters overlap in one dense band.

---

## Challenge 1: Conversational Filler & Numeric Noise

**Why does a filler word like `know` still get a substantial weight?**

The IDF part of the formula only punishes a word in proportion to the log of how common it is. If `know` appears in 25% of posts:

$$\log\left(\frac{N}{DF_t}\right) = \log\left(\frac{1}{0.25}\right) = \log 4 \approx 1.39$$

Scikit-learn's smoothed version adds 1, giving about 2.39. A genuinely rare word that appears in 1% of posts gets about 5.6. So a filler word is only reduced by a factor of roughly 2 to 3, not removed. The TF part then multiplies this weight by how often the word is used, and conversational posts repeat words like `know` and `like` a lot. On top of that, the "average TF-IDF" ranking rewards words that are non-zero in many documents, so widespread filler rises to the top. This is exactly what happened: 9 of the top 10 baseline words (`know`, `like`, `just`, `don`, `does`, `use`, `thanks`, `good`, `think`) are filler, and only `car` is a topic word.

**Effect of the refined vectorizer** (`token_pattern=r'\b[a-zA-Z]{3,}\b'`, `max_df=0.5`, custom conversational stopwords):

| Cluster | Top words after cleaning |
|---|---|
| Autos | car, cars, engine, oil, new, dealer, ford, miles |
| Mac Hardware | mac, apple, drive, monitor, card, scsi, video, simms |
| Signature | cadre, dsl, chastity, shameful, surrender, intellect, skepticism, geb |
| Catch-all | people, time, used, edu, information, new, help, work |

The cluster vocabulary became clearly more meaningful. Filler words disappeared from every cluster, and the topic clusters gained specific terms (`oil`, `miles`, `scsi`, `video`, `simms`). However, the cluster *membership* barely changed: the catch-all cluster still holds 2,532 posts (68%), and Mac purity only rose from 90.9% to 92.6%. Cleaning improved how explainable the clusters are, but it did not fix the structure, because one of only four clusters is still spent on the repeated email signature. A further fix would be to add signature tokens (`cadre`, `dsl`, `geb`, `pitt`, `chastity`, …) to the stopword list, or to strip signature lines before vectorizing.

---

## Challenge 2: Hardware Vocabulary Overlap

**Baseline crosstab:**

| True label | Cluster 0 (catch-all) | Cluster 1 (signature) | Cluster 2 (autos) | Cluster 3 (Mac) |
|---|---|---|---|---|
| comp.sys.mac.hardware | 269 | 0 | 2 | 639 |
| rec.autos | 396 | 0 | 507 | 2 |
| sci.electronics | 860 | 0 | 20 | 60 |
| sci.med | 875 | 71 | 0 | 2 |

**Which categories overlap most?** `sci.electronics` and `comp.sys.mac.hardware`. Electronics never gets a cluster of its own. Apart from the catch-all, the topical cluster it leaks into most is the Mac cluster (60 posts), and that leakage continues in every version of the experiment (60 → 43 → 42 posts). Both groups discuss physical hardware, so they share technical words that appear in the Mac cluster's top terms: **`drive`, `card`, `monitor`, `video`, `power`**. A word like `drive` means a disk drive in a Mac post but can also appear in electronics posts about drive circuits or motor drives; `power` (power supply, power consumption) and `card` (expansion cards, circuit cards) are used the same way in both groups.

**Effect of bigrams** (`ngram_range=(1, 2)`):

| True label | Cluster 0 (Mac) | Cluster 1 (catch-all) | Cluster 2 (Medical) | Cluster 3 (signature) |
|---|---|---|---|---|
| comp.sys.mac.hardware | 607 | 297 | 6 | 0 |
| rec.autos | 1 | 879 | 25 | 0 |
| sci.electronics | 42 | 890 | 8 | 0 |
| sci.med | 1 | 180 | 696 | 71 |

Bigrams turn ambiguous single words into specific phrases ("hard drive" vs. "drive circuit", "power supply" vs. "power mac"), and the Mac cluster became the purest it has been in any run (93.2%, with only 42 electronics posts). Bigrams also finally gave `sci.med` its own cluster (696 posts, 94.7% pure, with terms like `doctor`, `medical`, `disease`, `pain`).

But bigrams did **not** give Electronics its own cluster. Instead, Autos and Electronics merged into one catch-all cluster (879 + 890 posts) whose top terms (`car`, `cars`, `engine`, `power`, `used`) mix both topics. Two reasons limit the effect: `max_features=4000` leaves little room for bigrams once unigrams are counted, and the strongest bigrams in the whole corpus were the signature's (`dsl pitt`, `cadre dsl`, `shameful surrender`), which reinforced the wasted signature cluster. Bigrams help separate general electronics from Mac hardware at the word level, but with only four clusters and one lost to the signature, K-Means still cannot give all four topics their own group.

---

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

| Post | Length | BoW L2 norm | TF-IDF L2 norm |
|---|---|---|---|
| Longest (#310) | 11,765 words | 276.24 | 1.00 |
| Shortest (#1926) | 1 word | 0.00 | 0.00 |

**Why K-Means fails on raw Bag of Words.** K-Means groups points by Euclidean distance. In raw counts, a vector's length grows with the number of words in the post, so the 11,765-word post has a norm of 276, far larger than a typical short question. Two posts on the same topic but of very different lengths end up far apart, while long posts on different topics can end up close to each other. K-Means then clusters partly by *post length* instead of *topic*, and very long rants pull centroids toward themselves.

**How TF-IDF fixes this.** Scikit-learn's `TfidfVectorizer` applies L2 normalization by default, so every post vector is scaled to length 1 (the longest post has a norm of exactly 1.00). Only the *direction* of the vector, meaning the relative mix of words, remains. For unit vectors, Euclidean distance depends only on cosine similarity:

$$\|a - b\|^2 = 2 - 2\cos(a, b)$$

So K-Means on normalized TF-IDF compares *what* posts talk about, not *how much* they wrote.

**Note on the shortest post.** Post #1926 has a norm of 0 in both matrices. It passed the "more than 50 characters" filter but contains only one whitespace-separated token, which is not in the 4,000-term vocabulary (or is a stopword). Its vector is therefore all zeros, and normalization cannot help, because there is nothing to scale. This is another real-world trap: such empty vectors carry no information, and they should be filtered out before clustering.

---

## Challenge 4: Regulated Compliance Explainability

**`explain_document(doc_id=412, top_n=5)` output:**

| Term | TF-IDF weight |
|---|---|
| discussion | 0.4524 |
| feel | 0.3989 |
| days | 0.3855 |
| need | 0.2885 |
| meant | 0.2667 |

Document #412 is actually a `rec.autos` post (the README assumes a `sci.med` document, but in this dataset #412 belongs to Autos). It was routed to Cluster 0, the catch-all cluster. The explanation shows exactly why: none of its five strongest terms is car vocabulary, so nothing pulled it toward the Autos centroid, and generic words like `discussion`, `feel` and `need` matched the catch-all cluster instead. An auditor can see this misrouting and its cause immediately, without trusting the system blindly.

**Why TF-IDF is preferred over a black-box LLM as a transparent baseline:**
Every TF-IDF routing decision can be traced to a short list of exact words and numeric weights, as shown for Document #412, and running the same pipeline again gives exactly the same result. A black-box LLM's decision depends on billions of hidden parameters, can change between runs or model versions, and cannot be fully reproduced or explained to a regulator. In an audited workflow such as adverse-event reporting, the ability to show a regulator the exact evidence behind each decision, including when the decision was wrong, matters more than a small gain in accuracy, which makes TF-IDF the safer baseline.
