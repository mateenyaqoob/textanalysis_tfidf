# Lab 15: Problem-Solving Answers

## Challenge 1: Conversational Filler and Numeric Noise

TF-IDF is calculated as:

\[
\operatorname{TFIDF}(t,d)=\operatorname{TF}(t,d)\times\log\left(\frac{N}{\operatorname{DF}(t)}\right)
\]

If a filler word such as `know` occurs in 25% of documents, its inverse document frequency is approximately \(\log(4)=1.386\), using the natural logarithm. This remains positive, so the word can receive a substantial TF-IDF weight when it occurs frequently in a document. Standard English stopword lists do not remove every conversational filler word.

To reduce noise, I created a custom stopword list containing `know`, `like`, `just`, `think`, `does`, `use`, and `thanks`. I also used `token_pattern=r'\b[a-zA-Z]{3,}\b'` to retain alphabetic tokens of at least three characters, `max_df=0.5` to exclude terms occurring in more than half of the documents, and `min_df=5` to remove very rare terms.

**Observed change in cluster keywords:**

In the baseline model, Cluster 0 contained terms such as `know`, `like`, `don`, `just`, `does`, `use`, and `thanks`. Cluster 2 included `car`, `cars`, `engine`, `new`, and `ford`, while Cluster 3 included `mac`, `apple`, `drive`, `monitor`, and `card`.

After cleaning, the conversational filler words disappeared from the leading keywords. The cleaned automotive cluster retained terms such as `car`, `cars`, `engine`, `oil`, and `ford`, while the Mac hardware cluster included `mac`, `apple`, `drive`, `monitor`, `card`, and `scsi`. The results show that custom stopwords can make cluster keywords more topic-focused. Some generic terms, such as `would`, `one`, and `get`, remain, so the cleaning process does not eliminate all general vocabulary.

The token pattern also excludes short tokens and numeric-only tokens. This reduces noise, although it can remove meaningful short technical abbreviations as well.

## Challenge 2: Hardware Overlap Diagnosis

I used `pd.crosstab()` to compare the original newsgroup labels with the clusters discovered by KMeans. The baseline cluster table showed the following patterns:

- `comp.sys.mac.hardware` was concentrated in Cluster 3, with 639 posts, but 269 posts were assigned to Cluster 0.
- `rec.autos` was split mainly between Cluster 2 (507 posts) and Cluster 0 (396 posts).
- `sci.electronics` was concentrated in Cluster 0, with 860 posts.
- `sci.med` was concentrated in Cluster 0, with 875 posts, while 71 posts were assigned to Cluster 1.

After cleaning, the Mac hardware category remained concentrated in Clean Cluster 3 (599 posts), with 309 posts in Clean Cluster 1. The automotive category remained mainly in Clean Cluster 2 (512 posts) and Clean Cluster 1 (393 posts). Electronics was concentrated in Clean Cluster 1 (860 posts), and medical posts were mainly in Clean Cluster 1 (874 posts).

These results show that the clusters do not perfectly match the original categories. In particular, electronics and medical posts are assigned to the same cleaned cluster, indicating that the model has grouped documents based on shared vocabulary rather than fully separating their subject categories. The automotive and Mac hardware categories also show some overlap.

KMeans assigns arbitrary numeric cluster IDs, so Cluster 0 in the baseline model does not necessarily represent the same topic as Clean Cluster 0 in the cleaned model.

## Challenge 3: Rant Length Bias

I compared the L2 norms of the Bag of Words and TF-IDF vectors for the shortest and longest posts.

| Document | Length | BoW L2 norm | TF-IDF L2 norm |
|---|---:|---:|---:|
| Shortest post (row 1159) | 52 characters | 2.0000 | 1.0000 |
| Longest post (row 310) | 72,161 characters | 276.2408 | 1.0000 |

The longest post has a much larger Bag of Words norm because the vector contains raw term counts, and long documents often contain more word occurrences. The shortest post has a BoW norm of 2.0000, compared with 276.2408 for the longest post.

The TF-IDF vectors both have an L2 norm of 1.0000. This is expected because `TfidfVectorizer` uses L2 normalization by default for nonempty documents. Consequently, the magnitude of a TF-IDF vector is not directly proportional to document length, although the words and their relative weights can still differ.

This comparison demonstrates how TF-IDF normalization reduces the direct influence of document length compared with raw term counts.

## Challenge 4: Stakeholder Explainability Audit

I implemented `explain_document(doc_id, matrix, vectorizer, top_n=5)` to identify the highest-weighted terms and their exact weights for a selected document. The function checks the document index, extracts the corresponding sparse matrix row, sorts its nonzero weights, and returns the top terms.

For document 412, the function returned:

| Term | TF-IDF weight |
|---|---:|
| discussion | 0.452421 |
| feel | 0.398890 |
| days | 0.385506 |
| need | 0.288483 |
| meant | 0.266672 |

The term `discussion` has the highest weight among these five terms, followed by `feel` and `days`. These weights help explain which terms are most prominent in the document's TF-IDF representation.

This function supports stakeholder explainability by showing the actual words and weights used to represent an individual document, rather than presenting only its assigned cluster.
