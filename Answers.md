# Lab 15: Diagnostic Challenges & Problem-Solving Answers

## Challenge 1: Conversational Filler & Numeric Noise

* **TF-IDF Weight Explanation:** 
  According to the TF-IDF formula:
  $$\text{TFIDF}_{t,d} = \text{TF}_{t,d} \times \log\left(\frac{N}{\text{DF}_t}\right)$$
  Even if a conversational filler word like `know` appears in 25% of posts, its Inverse Document Frequency (IDF) is $\log(N / 0.25N) = \log(4) \approx 0.602$. Because this logarithmic multiplier is still substantial, high term frequency ($\text{TF}$) in individual posts allows filler words to accumulate high overall weights, drowning out rare, domain-specific vocabulary.
* **Impact of Refinement:** 
  Implementing `token_pattern=r'\b[a-zA-Z]{3,}\b'` filters out stray numeric tokens and short noise words, while `max_df=0.5` automatically prunes terms appearing in more than 50% of documents. Combined with custom conversational stopwords, this shifts cluster top words away from conversational filler toward true domain-specific terminology (e.g., hardware components, medical symptoms).

---

## Challenge 2: Hardware Overlap Diagnosis

* **Overlapping Categories:** 
  Running `pd.crosstab(df['true_label'], df['cluster'])` reveals that `sci.electronics` and `comp.sys.mac.hardware` exhibit significant cross-cluster misclassification. 
* **Shared Vocabulary & Bigrams:** 
  Misclustered posts share technical vocabulary such as `board`, `power`, `pin`, `cable`, and `card`. Adding bigrams via `ngram_range=(1, 2)` resolves this ambiguity by capturing distinct compound phrases (e.g., distinguishing general electronics terms like `power supply` or `circuit board` from Mac-specific hardware tokens like `logic board` or `adb port`).

---

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

* **Vector Norm Comparison:** 
  The Euclidean norms ($\|x\|_2$) for Bag of Words (`X_bow`) scale linearly with document length, resulting in massive norm discrepancies between 2,000-word rants and 20-word questions. In contrast, `X_tfidf` applies L2 normalization, bounding vector lengths.
* **Effect on K-Means:** 
  `KMeans` uses Euclidean distance to assign cluster centroids. On raw Bag of Words counts, distance calculations are dominated by document length rather than semantic content, causing short posts and long posts to separate artificially. Scikit-Learn's L2-normalized TF-IDF vectorizer projects all documents onto a unit hypersphere ($\|x\|_2 = 1$), eliminating length bias.

---

## Challenge 4: Regulated Compliance Explainability

* **Document #412 Top Features (Example Output):**
  * `patient` (Weight: 0.42)
  * `dose` (Weight: 0.38)
  * `symptoms` (Weight: 0.31)
  * `reaction` (Weight: 0.25)
  * `mg` (Weight: 0.19)
* **Regulatory Justification:** 
  TF-IDF is preferred over black-box LLMs in regulated compliance workflows because every classification decision is 100% mathematically transparent and reproducible. Auditors can directly trace a document's routing back to explicit term frequencies and document frequencies without facing hidden hallucinations or opaque neural network weights. This explainable feature attribution satisfies strict legal and healthcare accountability standards.
