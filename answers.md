# Lab 15: Problem-Solving Diagnostics (ANSWERS.md)

## Challenge 1: Conversational Filler & Numeric Noise
**Task:** Explain why a filler word like `know` still receives a substantial weight if it appears in 25% of posts. Then, refine `TfidfVectorizer` and explain how cluster top words change.

**Answer:**
The TF-IDF formula is \(TF_{t,d} \times \log(\frac{N}{DF_t})\). If a filler word like "know" appears in 25% of posts, its Document Frequency (DF) is \(0.25N\). The Inverse Document Frequency (IDF) becomes \(\log(\frac{N}{0.25N}) = \log(4) \approx 1.386\). Because the Term Frequency (TF) in those posts is still being calculated, the word receives a positive, substantial weight. TF-IDF down-weights common words, but does not completely eliminate them unless the word appears in 100% of the documents.

By refining the vectorizer using `token_pattern=r'\b[a-zA-Z]{3,}\b'`, `max_df=0.5`, and a custom stopword list (removing words like 'know', 'like', 'just'), the clusters become much more distinct. The top words shift from generic conversational verbs (e.g., "think", "use", "know") to highly specific domain nouns (e.g., "mac", "drive", "patient", "car"), making the topical queues much clearer for human moderators.

## Challenge 2: Hardware Vocabulary Overlap
**Task:** Identify which two categories overlap the most. List 3 shared technical words pulling them together, and explain how bigrams help.

**Answer:**
Based on the `pd.crosstab` output, **`sci.electronics`** and **`comp.sys.mac.hardware`** overlap the most. Because a Mac *is* an electronic device, they share heavily overlapping technical vocabulary. 

At least 3 shared technical words pulling them together are: **"drive"**, **"board"**, and **"monitor"** (others include "system", "power", "card", "memory").

Adding bigrams (`ngram_range=(1, 2)`) helps separate them because it captures compound terms that carry specific context. For example, "hard drive" or "logic board" strongly indicates Mac hardware, whereas "power supply" or "video card" leans toward general electronics. Bigrams retain the surrounding context that unigrams (single words) lose.

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)
**Task:** Compare the L2 norm of the longest and shortest posts in `X_bow` versus `X_tfidf`. Explain why KMeans fails on raw BoW and how TF-IDF fixes it.

**Answer:**
In `X_bow`, the L2 norm ratio of the longest post to the shortest post is extremely high (often 10x to 50x larger). In `X_tfidf`, the L2 norm ratio is exactly **1.0** for every document because `TfidfVectorizer` applies L2 normalization by default.

`KMeans` relies on Euclidean distance. If raw Bag of Words is used, the Euclidean distance is heavily dominated by document length. A 2,000-word rant will be mathematically "farther" from a 20-word question simply because it has more words, causing KMeans to cluster by length rather than topic. TF-IDF's L2 normalization forces every document vector onto the unit sphere (length = 1). This ensures that distance only measures the *angle* (topic similarity) and not the magnitude (word count), allowing KMeans to successfully cluster by subject matter.

## Challenge 4: Regulated Compliance Explainability
**Task:** Implement `explain_document(doc_id=412, top_n=5)` and write a 3-sentence justification for TF-IDF over black-box LLMs.

**Answer:**
*(Note: The implementation of `explain_document` is included in `student_starter_script.py`. It uses `matrix.getrow(doc_id)` to extract non-zero features and `argsort` to return the top 5 terms and weights.)*

**Justification for Auditors:**
TF-IDF produces a document-term matrix where every feature weight is the direct mathematical product of transparent term frequency and inverse document frequency calculations. This ensures that an auditor can trace exactly which specific words drove a routing decision without relying on complex, post-hoc approximations. A black-box LLM provides no such intrinsic per-feature audit trail, making TF-IDF the preferred transparent baseline for meeting strict regulated explainability requirements (such as GDPR "right to explanation").