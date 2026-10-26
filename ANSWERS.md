# Lab 15 — Problem-Solving Answers

## Challenge 1: Conversational Filler & Numeric Noise

A word in 25% of posts can still have significant TF-IDF weight: its IDF is `log(N / 0.25N) = log(4) ≈ 1.386` (before multiplying by term frequency). The refined vectorizer uses alphabetic words of at least three letters, `max_df=0.5`, and custom stopwords. Previously, Cluster 0 was dominated by `know`, `like`, `just`, `does`, `use`, and `thanks`. After cleaning, those words disappeared from the top cluster terms; topic-specific words such as `mac`, `apple`, `drive`, `car`, `engine`, and `ford` stood out more clearly, although some generic words remained.

## Challenge 2: Hardware Vocabulary Overlap

Electronics and Mac Hardware overlapped in the original mixed Cluster 0 (860 Electronics posts and 269 Mac posts). Verified shared technical words include `drive` (38 Electronics / 140 Mac posts), `monitor` (12 / 105), and `card` (23 / 78). Bigrams can preserve context from two-word phrases, potentially distinguishing hardware topics. After adding bigrams, Electronics posts in the Mac-dominant cluster fell from 60 to 18, but Mac posts in mixed Cluster 0 rose from 269 to 308. Thus, separation improved in one respect but not consistently overall.

## Challenge 3: Rant Length Bias

The longest post had **11,765 words**: BoW L2 norm **276.2408**, TF-IDF L2 norm **1.0000**. The shortest had **1 word**: both norms **0.0000**, because its term(s) did not survive vocabulary filtering. Raw BoW counts make longer posts have larger vector magnitudes, which can distort Euclidean-distance K-Means. Default L2-normalized TF-IDF gives **non-zero** document vectors unit norm, reducing length bias; zero vectors remain zero.

## Challenge 4: Regulated Compliance Explainability

Document **412** was assigned to **Cluster 0**; its true category was **`rec.autos`**. Its five highest non-zero TF-IDF features were:

| Term | Weight |
|---|---:|
| discussion | 0.4524 |
| feel | 0.3989 |
| days | 0.3855 |
| need | 0.2885 |
| meant | 0.2667 |

TF-IDF exposes the exact words and weights used to represent each document. Unlike a black-box LLM, its feature construction and K-Means assignments can be reproduced and audited, although these top terms alone do not fully explain the centroid-distance decision. This makes TF-IDF a transparent baseline for regulated workflows, with human review still necessary.
