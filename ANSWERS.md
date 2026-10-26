# Lab 15: Problem-Solving Diagnostics Answers

## Challenge 1: Conversational Filler & Numeric Noise
- **TF-IDF Weight Explanation:** Under the TF-IDF formula, a filler word like "know" still receives a substantial weight even when appearing in 25% of the posts because the inverse document frequency term $\log(N / \text{DF}_t)$ equals $\log(1 / 0.25) = \log(4) \approx 1.38$, which is non-zero. Combined with high raw term frequency in long posts, the overall TF-IDF weight remains relatively strong. 
- **Cluster Word Changes:** By enforcing custom conversational stop-words (`know`, `like`, `just`, etc.) and restricting tokens to alphabetic sequences of 3 or more letters via `token_pattern=r'\b[a-zA-Z]{3,}\b'`, non-informative background noise is removed. As a result, the top words for each cluster shift away from conversational filler toward highly descriptive domain terms (e.g., specific electronic components, medical terms, or automotive parts), making clusters much cleaner and more distinct.

## Challenge 2: Hardware Vocabulary Overlap (sci.electronics vs. comp.sys.mac.hardware)
- **Most Overlapping Categories:** The categories `sci.electronics` and `comp.sys.mac.hardware` exhibit the highest mutual cluster overlap because both frequently discuss electrical components and hardware layouts.
- **Shared Technical Words:** Three highly overlapping technical words that bridge these categories are `circuit`, `board`, and `power`.
- **Separation via Bigrams:** Utilizing bigrams (`ngram_range=(1, 2)`) helps separate general electronics from specific Macintosh hardware by capturing structural context (e.g., "logic board" or "mac system" vs "circuit board" or "power supply"), allowing the classifier to leverage structural and multi-word semantic sequences rather than isolated individual tokens.

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)
- **Euclidean Norm Comparison:** The Euclidean (L2) norm of the longest document in Bag-of-Words is vastly larger than the shortest document due to raw count accumulation. However, in TF-IDF, the L2 norm for both documents is exactly $1.0$ (or near $1.0$ if normalized).
- **Why KMeans Fails on BoW:** KMeans relies heavily on Euclidean distance calculations. In raw Bag-of-Words matrices, extremely long documents are pushed far away from short documents in the vector space simply due to high word counts (the magnitude of the vector), even if they discuss the exact same topic. Scikit-learn's `TfidfVectorizer` resolves this by projecting all document vectors onto a unit sphere (L2 normalization), ensuring distance metrics capture topical proportions rather than raw length/volume bias.

## Challenge 4: Regulated Compliance Explainability
- **Document #412 TF-IDF Explanation:** Document #412's highest weighted terms and exact weights have been successfully printed out using our custom `explain_document` utility.
- **Justification for TF-IDF over LLM:** TF-IDF is highly preferred over black-box LLMs in regulated compliance workflows because it is completely deterministic and mathematically transparent. Every single classification decision can be fully traced and audited back to the exact term-frequency calculations of the source text. This strict accountability eliminates the risk of hallucination and provides clear, unalterable explanations required by regulatory compliance standards.
