# TF-IDF Text Analysis — Diagnostic Challenges

## Challenge 1: Conversational Filler & Numeric Noise

Words like `know` can still receive high TF-IDF scores because appearing in 25% of documents gives an IDF of `log(4) ≈ 1.386`.

I applied `token_pattern=r'\b[a-zA-Z]{3,}\b'`, `max_df=0.5`, and custom stopwords to remove filler words and numeric noise.

The clusters became more meaningful. For example, the Mac cluster changed from `mac, apple, drive, monitor, card, thanks, does, know` to `mac, apple, drive, monitor, card, scsi, video, simms`. However, some noise remained.

## Challenge 2: Hardware Vocabulary Overlap

Electronics and Mac hardware showed overlap. In the original clustering, 269 Mac posts were grouped with 860 electronics posts in Cluster 0.

Common technical words included `cable`, `card`, `disk`, `drive`, `memory`, `monitor`, `power`, and `scsi`.

Bigrams capture two-word phrases, which can provide more context. After adding `ngram_range=(1, 2)`, Mac posts in the Mac-dominated cluster decreased from 639 to 600, while electronics posts in that cluster decreased from 60 to 35. The results were mixed, with less overlap but fewer Mac posts correctly grouped.

## Challenge 3: Rant Length Bias

| Document | BoW L2 Norm | TF-IDF L2 Norm |
|---|---:|---:|
| Longest | 276.2408 | 1.0000 |
| Shortest | 2.0000 | 1.0000 |

Raw BoW gives longer documents larger vector magnitudes, which can affect K-Means clustering. TF-IDF uses L2 normalization, keeping nonzero document vectors at unit length and reducing document-length bias.

## Challenge 4: Regulated Compliance Explainability

Top five TF-IDF words for Document 412:

| Word | Weight |
|---|---:|
| discussion | 0.452421 |
| feel | 0.398890 |
| days | 0.385506 |
| need | 0.288483 |
| meant | 0.266672 |

TF-IDF is easier to explain because every word has a measurable weight. Unlike a black-box LLM, its feature scores can be directly checked by an auditor. This makes it a useful baseline for regulated applications, although the scores alone cannot establish whether a routing decision is correct.