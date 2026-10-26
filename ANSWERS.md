# Lab 15: Unstructured Text Analysis Answers

### Challenge 1: Conversational Filler & Numeric Noise
- **Formula Explanation:** In the TF-IDF formula, if a conversational filler word like `know` appears in 25% of the posts ($DF_t = 0.25 \cdot N$), its Inverse Document Frequency (IDF) is $\log(N / 0.25N) = \log(4) \approx 1.386$. Because this value is a non-zero positive constant, any document that contains the word `know` many times (high Term Frequency, TF) will still receive a substantial TF-IDF weight.
- **Impact of Refinement:** Refining the vectorizer using `token_pattern=r'\b[a-zA-Z]{3,}\b'` removes short/noisy numeric terms. Adding conversational stopwords (e.g., `know`, `like`, `just`) forces KMeans to cluster documents based on actual topic-specific semantic terms rather than conversational noise. This directly yields distinct clusters representing hardware, cars, health, and electronics.

### Challenge 2: Hardware Vocabulary Overlap
- **Overlapping Categories:** The categories `sci.electronics` (Electronics) and `comp.sys.mac.hardware` (Mac Hardware) have the highest overlap in unsupervised clustering.
- **Shared Vocabulary:** The overlap is driven by shared domain technical terms including `card`, `drive`, `scsi`, `monitor`, and `bus`.
- **Bigrams Resolution:** Introducing bigrams (`ngram_range=(1, 2)`) captures contextual phrases like `mac drive`, `hard drive`, or `video card` versus standard electronics terms (e.g., `circuit board`), allowing the clustering algorithm to distinguish system-level Apple Macintosh hardware from general electronic circuitry components.

### Challenge 3: Rant Length Bias
- **Euclidean Norms Comparison:** The Euclidean norm of the longest document in Bag of Words (`X_bow`) is extremely large (~276.23) because the counts of words scale linearly with length. In contrast, the shortest document has a norm of 0.0. Under TF-IDF, both long and short non-empty documents are L2-normalized, keeping non-empty vector lengths standardized to exactly 1.0.
- **KMeans Normalization Impact:** KMeans relies on Euclidean distances to partition data. On raw BoW counts, document length dominates distance computations, causing KMeans to cluster documents by size (rants vs. short posts) rather than semantic content. L2-normalization in `TfidfVectorizer` resolves this by mapping all document vectors to a unit sphere, making cluster assignments length-independent.

### Challenge 4: Regulated Compliance Explainability
- **Doc #412 Top Features:**
  - `toxic`: 0.3711
  - `highly`: 0.3433
  - `damage`: 0.3271
  - `hidden`: 0.2086
  - `compound`: 0.2066
- **Explainability Justification:** TF-IDF is highly preferred over a black-box LLM in regulated compliance audits because it is mathematically deterministic and transparent. Every feature weight can be directly traced back to exact word counts and frequency distributions in the corpus, eliminating hallucination risks. This explicit mathematical mapping provides a clear audit trail that is straightforward to explain and verify to regulatory stakeholders.
