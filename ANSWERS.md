
# TF-IDF and Text Clustering

## Challenge 1: Conversational Filler and Numeric Noise

The word `know` can receive a substantial TF-IDF weight because it appears in only 25% of the posts, rather than all posts. Its inverse document frequency is proportional to log(N/DF), which equals log(4) when the document frequency is 25% of the corpus. We reduced conversational noise using custom stopwords, `token_pattern=r'\b[a-zA-Z]{3,}\b'`, and `max_df=0.5`. After cleaning, clusters became more interpretable, including recognizable automobile, Mac hardware, and medical vocabulary.

## Challenge 2: Hardware Vocabulary Overlap

The `sci.electronics` and `comp.sys.mac.hardware` categories overlap because they discuss related technical subjects. Shared terms such as `drive`, `card`, and `monitor` can pull their posts into the same cluster. Adding bigrams allows the model to consider word combinations as features, which can provide more context than individual words. In our results, the Mac hardware and automobile clusters were recognizable, but electronics and medical posts still showed considerable overlap; therefore, bigrams did not fully separate all categories.

## Challenge 3: Rant Length Bias

The longest document had a Bag of Words L2 norm of 276.2408, while the shortest had a norm of 2.0000. Both TF-IDF vectors had an L2 norm of 1.0000 because `TfidfVectorizer` uses L2 normalization by default. Raw Bag of Words counts can make longer documents dominate Euclidean distances used by K-Means, whereas L2-normalized TF-IDF reduces the direct influence of document length.

## Challenge 4: Regulated Compliance Explainability

For document 412, the five highest-weighted non-zero TF-IDF features were:

| Term | TF-IDF weight |
|---|---:|
| discussion | 0.4524 |
| feel | 0.3989 |
| days | 0.3855 |
| need | 0.2885 |
| meant | 0.2667 |

TF-IDF is a transparent baseline because an auditor can inspect the exact terms and numerical weights contributing to a document's representation. Unlike a black-box LLM, this representation uses a defined and reproducible mathematical calculation rather than opaque generated reasoning. However, TF-IDF explains which words are important in the representation; the top terms alone do not fully explain why K-Means assigned the document to a particular cluster.
