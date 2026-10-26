# Lab 15 – Answers

Dataset: 3,703 posts, TF-IDF matrix shape (3703, 4000), sparsity 99.04%.

## Challenge 1: Conversational Filler & Numeric Noise

**Why "know" still gets a substantial weight.**
IDF = log(N / DF). If "know" appears in 25% of posts, N/DF = 4, so IDF = log(4) ≈ 1.39 (scikit-learn's smoothed version gives about 2.39). That is not close to zero. IDF only becomes ~0 when a word appears in nearly every document. Filler words also have high term frequency inside the posts that use them, so TF × IDF stays large. Since they appear across all four topics, their centroid weights are high in every cluster and they swamp the real topic words.

**Baseline top words (stopwords='english' only):**

| Cluster | Top words |
|---|---|
| 0 | cadre, dsl, n3jxp, chastity, intellect, shameful, surrender, skepticism |
| 1 | know, like, don, just, does, use, thanks, good |
| 2 | mac, apple, drive, monitor, card, does, thanks, know |
| 3 | car, cars, engine, just, new, ford, dealer, like |

**After refinement** (`token_pattern=r'\b[a-zA-Z]{3,}\b'`, `max_df=0.5`, custom stopwords: know, like, just, think, does, use, thanks, don, did, ve, really, good, need):

| Cluster | Top words |
|---|---|
| 0 | cadre, dsl, chastity, intellect, surrender, shameful, skepticism, geb |
| 1 | people, time, edu, used, information, want, help, make |
| 2 | mac, apple, drive, monitor, card, scsi, video, simms |
| 3 | car, cars, engine, new, ford, oil, dealer, miles |

Filler words disappeared from every cluster. Cluster 2 gained hardware terms (scsi, video, simms) and cluster 3 gained car terms (oil, miles), so both are more topical. Cluster 1 is still a generic catch-all (people, time, information), and cluster 0 is a tiny cluster driven by a repeated signature/quoted text (cadre, dsl, chastity).

## Challenge 2: Hardware Vocabulary Overlap

```
cluster                 0    1    2    3
true_label
comp.sys.mac.hardware   0  279  629    2
rec.autos               0  377    2  526
sci.electronics         0  850   66   24
sci.med                71  875    2    0
```

- Cluster 2 is the "Mac hardware" cluster and cluster 3 is "autos".
- The two hardware categories overlap the most: `sci.electronics` has 66 posts inside the Mac cluster, and `comp.sys.mac.hardware` has 279 posts in the generic cluster 1 where electronics also sits (850). Electronics has no cluster of its own.
- Shared technical words pulling them together (from the cluster 2 centroid): **drive, card, monitor, video, scsi**.
- **Bigrams (`ngram_range=(1, 2)`)**: phrases such as "hard drive", "power supply" or "logic board" carry more context than single words like "drive" or "power", so in principle they can separate general electronics from Mac hardware. In my run the effect was small: the Mac cluster went from 629 to 599 posts and electronics was still mostly in the generic cluster (888). Bigram clusters were nearly the same as the unigram ones, and top bigrams were dominated by the signature text ("cadre dsl", "intellect geb"). So bigrams help in principle, but here the generic/filler vocabulary dominates the clustering more than phrases do.

## Challenge 3: Rant Length Bias

| Matrix | Longest post norm | Shortest post norm |
|---|---|---|
| X_bow | 276.286 | 2.000 |
| X_tfidf | 1.000 | 1.000 |

KMeans uses Euclidean distance. In raw Bag of Words the long post has a huge vector (norm 276 vs 2), so distance is driven by document length rather than topic. Long posts sit far from the origin and from everything else, and short posts all collapse near the origin regardless of topic. TfidfVectorizer applies L2 normalization by default, so every document has norm 1 and lies on the unit sphere. Distance then depends on the direction (which words) and not on how long the post is.

## Challenge 4: Regulated Compliance Explainability

`explain_document(doc_id=412, top_n=5)` output:

| Term | TF-IDF weight |
|---|---|
| discussion | 0.4524 |
| feel | 0.3989 |
| days | 0.3855 |
| need | 0.2885 |
| meant | 0.2667 |

**Justification (3 sentences):**
TF-IDF is fully transparent: every routing decision can be traced to specific words and exact numeric weights, as shown above for Document #412, so an auditor can reproduce and verify it. It is deterministic and cheap to run, so the same input always gives the same output and the model can be versioned and re-audited, which a black-box LLM cannot guarantee. This makes TF-IDF a defensible, explainable baseline for regulated workflows, and any more complex model can be compared against it.
