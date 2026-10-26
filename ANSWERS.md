# Part 2: Problem-Solving Answers

> All numbers below come from running `python lab15_starter.py` (3,703 posts, 4,000 TF-IDF features, 99.04% sparse).

## Challenge 1: Conversational Filler & Numeric Noise

TF-IDF is `TF * log(N / DF)`. A word in 25% of posts has an IDF of `log(1/0.25) = log 4 ≈ 1.39`, only moderately lower
than a topical word in 2% of posts (`log 50 ≈ 3.9`). Filler like `know` also has a high TF in chatty posts, so the
product stays substantial: IDF down-weights common words but never zeroes them unless they appear in nearly every
document. Standard stopword lists also don't contain `know`, `thanks`, `don`, or stray numbers.

Fix: `TfidfVectorizer(token_pattern=r'\b[a-zA-Z]{3,}\b', max_df=0.5, stop_words=<english + custom filler>)`.
This drops numbers and 1-2 letter tokens, removes words in more than half the posts, and removes the custom filler.

**Result.** Baseline cluster 0 was pure filler: `know, like, don, just, does, use, thanks, good`. After cleaning, the filler
words are gone from every cluster. The car cluster (`car, cars, engine, new, ford, oil, dealer`) and the Mac cluster
(`mac, apple, drive, monitor, card, scsi, video, simms`) became cleaner and more technical (`scsi`, `simms`, `oil`).
Cluster 0 turned into a small cluster of one author's signature/header remnants (`geb, pitt, cadre, dsl, chastity`),
and the generic remainder is still a catch-all (`good, people, time, edu, used, information`). So cleaning improves
interpretability, but it does not fully separate medicine and electronics.

## Challenge 2: Hardware Vocabulary Overlap

Baseline crosstab (`pd.crosstab(df['true_label'], df['cluster'])`):

| true_label | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| comp.sys.mac.hardware | 269 | 0 | 2 | 639 |
| rec.autos | 396 | 0 | 507 | 2 |
| sci.electronics | 860 | 0 | 20 | 60 |
| sci.med | 875 | 71 | 0 | 2 |

Clusters 2 (cars) and 3 (Mac) are the only ones that map cleanly to a category. `sci.electronics` and `sci.med` are
mostly absorbed into the generic filler cluster 0, so electronics is not separated from medicine. Of the two hardware
categories, `sci.electronics` and `comp.sys.mac.hardware` are the pair that share vocabulary: 60 electronics posts
land in the Mac cluster, and 269 Mac posts fall into the generic cluster alongside electronics. Shared technical words
pulling them together include `drive`, `card`, `monitor`, `power`, `board`, `memory`, `chip` (`drive`, `card`, `monitor`
are among the top centroid terms of the Mac cluster, and electronics posts about the same parts match them).

Bigrams (`ngram_range=(1, 2)`) moved the numbers only slightly (electronics in the Mac cluster went 60 → 18, Mac
posts in the generic cluster 269 → 308). They help because phrases like `hard drive`, `logic board`, or `mac se` are more
specific than their single words, while electronics phrases (`power supply`, `op amp`) stay distinct. The main
limitation is that KMeans is still dominated by the large generic cluster, so bigrams alone do not fully fix it.

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

| matrix | longest post L2 norm | shortest post L2 norm |
|---|---|---|
| `X_bow` | 276.241 | 2.000 |
| `X_tfidf` | 1.000 | 1.000 |

In raw counts the longest post is about 138x farther from the origin than the shortest. KMeans uses Euclidean
distance, so a long rant sits far from short posts even when they share a topic; clusters group by length rather than
topic. `TfidfVectorizer` L2-normalizes each row (`norm='l2'`), so every document is a unit vector, and Euclidean
distance then reflects direction (word mix), not document length.

## Challenge 4: Regulated Compliance Explainability

`explain_document(412, X_tfidf, tfidf_vec, top_n=5)` outputs the exact top 5 non-zero TF-IDF features for post #412:

| term | weight |
|---|---|
| discussion | 0.4524 |
| feel | 0.3989 |
| days | 0.3855 |
| need | 0.2885 |
| meant | 0.2667 |

TF-IDF is preferred as a transparent baseline because every routing decision traces to specific words and weights that an
auditor can verify and reproduce. A black-box LLM can't give a faithful, deterministic account of why it chose a
cluster, and its outputs may vary between runs or model versions. TF-IDF + KMeans is also cheap, runs on-premises
without sending patient data to a third party, and gives a fixed baseline against which more complex models can be validated.
