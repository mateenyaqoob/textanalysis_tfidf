# Lab 15: Unstructured Text Analysis & Automated Triage

## Challenge 1: Conversational Filler & Numeric Noise

The simplified TF-IDF formula is:

\[
\operatorname{TFIDF}(t,d)
=
\operatorname{TF}(t,d)
\times
\log\left(\frac{N}{DF_t}\right)
\]

A word such as `know` can still receive a substantial TF-IDF weight even if it appears in 25% of posts. Its inverse document frequency is \(\ln(4)\approx1.386\) under the simplified formula, so it is not considered sufficiently common to receive a near-zero weight. TF-IDF measures how informative a term is across the corpus; it does not know whether the term is useful for identifying a topic.

I refined the vectorizer by adding custom stopwords, setting `max_df=0.5`, and using `token_pattern=r"\b[a-zA-Z]{3,}\b"`. This removes specified conversational filler, excludes terms appearing in more than half of documents, and ignores numeric tokens and words shorter than three letters. In the original model, Cluster 0 contained filler words such as `know`, `like`, `don`, and `just`. After cleaning, its top terms changed to `geb`, `pitt`, `cadre`, `dsl`, `chastity`, `intellect`, `shameful`, and `surrender`, showing that removing filler words changes which features dominate a cluster.

## Challenge 2: Hardware Vocabulary Overlap

The original crosstab shows that most `comp.sys.mac.hardware` posts were assigned to Cluster 3, which has prominent terms such as `mac`, `apple`, `drive`, `monitor`, and `card`. Most `sci.electronics` posts were assigned to Cluster 0, which contains posts from several categories. Therefore, the crosstab alone does not establish that these two categories overlap the most; example posts and their shared technical vocabulary must also be inspected.

To investigate the overlap, I would inspect posts from both categories that were assigned to the same cluster and identify at least three technical terms that occur in both. Adding bigrams through `ngram_range=(1, 2)` lets TF-IDF represent two-word phrases, not just individual terms. Phrases such as `video card`, `hard drive`, or `power supply` may provide more context for distinguishing general electronics from Mac hardware, but the result must be checked against the new cluster keywords and crosstab.

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

The longest post was 72,161 characters, while the shortest was 52 characters. The Bag of Words vector norm was approximately 276.24 for the longest post and 2.00 for the shortest post. In contrast, the TF-IDF vector norms were approximately 1.00 for both documents.

Bag of Words stores raw word counts, so longer documents can have much larger vector magnitudes and disproportionately influence Euclidean distances used by K-Means. By default, Scikit-learn's `TfidfVectorizer` applies L2 normalization, making each nonempty document vector have a norm of approximately one. This reduces the effect of document length and helps clustering focus more on relative word importance, although it does not guarantee perfect topic separation.

## Challenge 4: Regulated Compliance Explainability

For Document #412, the five highest non-zero TF-IDF features were `discussion` (0.452421), `feel` (0.398890), `days` (0.385506), `need` (0.288483), and `meant` (0.266672). These weights provide a reproducible, document-specific explanation of which terms were most influential in its vector representation. TF-IDF is a transparent baseline for regulated workflows because its feature weights and vocabulary can be inspected and reproduced, unlike a black-box LLM whose generated explanations may be harder to audit consistently. However, TF-IDF weights explain the document representation, not the full causal reason for a K-Means assignment, and a regulated system would still need validation, audit trails, and human oversight.
