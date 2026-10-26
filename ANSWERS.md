# Lab 15: Diagnostic Challenges & Problem-Solving Answers

**Course:** Data Science Tools & Techniques  
**Topic:** NLP Pipeline, Bag of Words (BoW), TF-IDF Vectorization, and Unsupervised Topic Discovery  
**Student:** Obaid Ullah Raja ([baidi-prob](https://github.com/baidi-prob))

---

## Challenge 1: Conversational Filler & Numeric Noise

### Mathematical Explanation of TF-IDF on High-Frequency Filler Words
The TF-IDF weight for term $t$ in document $d$ is given by:
$$\text{TFIDF}_{t,d} = \text{TF}_{t,d} \times \log\left(\frac{N}{\text{DF}_t}\right)$$

In Scikit-Learn's default smoothed formulation:
$$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}_t}\right) + 1$$

If a conversational filler word like `know` appears in 25% of posts ($\text{DF}_t = 0.25 N$):
$$\text{IDF}(\text{know}) \approx \ln\left(\frac{1}{0.25}\right) + 1 = \ln(4) + 1 \approx 1.386 + 1 = 2.386$$

Because $\text{IDF}$ remains a positive, non-trivial multiplier ($> 2.3$), any post where a user uses conversational filler repeatedly (e.g., *"I know that you know..."*, leading to high local term frequency $\text{TF}_{t,d}$) generates a substantial TF-IDF score. Because standard English stopword lists focus on structural words (pronouns, prepositions, conjunctions) rather than colloquial discourse markers, words like `know`, `like`, `just`, `think`, and `does` evade filtering and contaminate the top centroid vocabulary.

### Refinements & Impact on Clusters
By refining `TfidfVectorizer` with:
- `token_pattern=r'\b[a-zA-Z]{3,}\b'`: Eliminates stray digits, hex numbers, and two-letter abbreviations.
- `max_df=0.5`: Automatically filters out corpus-wide ubiquitous words appearing in over 50% of documents.
- `stop_words=custom_stopwords`: Explicitly excludes common conversational fillers (`know`, `like`, `just`, `think`, `does`, `use`, `thanks`, `good`, `time`, `people`, `way`).

**Cluster Keyword Shift:**
- **Baseline Clusters:** Dominated by conversational noise in Cluster 0 (`know, like, don, just, does, use, thanks, good`), creating an artificial "conversational style" cluster that pulled in documents from all four distinct domains.
- **Refined Clusters:** Centroids cleanly crystallize into distinct topical vocabulary:
  - Cluster 2 focuses on automotive mechanics (`car`, `cars`, `engine`, `dealer`, `ford`).
  - Cluster 3 isolates computer hardware (`mac`, `apple`, `drive`, `monitor`, `card`).
  - Remaining clusters highlight medical and technical discussions without being masked by conversational filler.

---

## Challenge 2: Hardware Vocabulary Overlap (`sci.electronics` vs. `comp.sys.mac.hardware`)

### Crosstab Analysis
The crosstab of hidden ground-truth labels against unsupervised KMeans baseline clusters:

| True Category | Cluster 0 (Conversational) | Cluster 1 (Outliers) | Cluster 2 (Automotive) | Cluster 3 (Mac Hardware) |
| :--- | :---: | :---: | :---: | :---: |
| **comp.sys.mac.hardware** | 269 | 0 | 2 | **639** |
| **rec.autos** | 396 | 0 | **507** | 2 |
| **sci.electronics** | **860** | 0 | 20 | **60** |
| **sci.med** | 875 | 71 | 0 | 2 |

### Shared Technical Vocabulary
The two categories that overlap the most are **`sci.electronics`** and **`comp.sys.mac.hardware`**. Over 60 posts from `sci.electronics` clustered directly with Mac hardware (Cluster 3), and both heavily share vocabulary in Cluster 0.

Inspection of misclassified documents reveals shared technical terms such as:
1. **`power`** / **`supply`** (power supply units, voltage, battery power)
2. **`card`** / **`board`** (circuit boards, logic boards, video cards)
3. **`chip`** / **`circuit`** / **`pin`** (integrated circuits, chipsets, connector pinouts)
4. **`monitor`** / **`cable`** / **`memory`** (peripheral connections, RAM modules, display sync)

### Resolving Overlap with Bigrams (`ngram_range=(1, 2)`)
Single unigrams like `power`, `board`, `card`, and `port` are ambiguous because both a hobbyist soldering an amplifier and an Apple technician repairing a Macintosh discuss them. 

Incorporating bigrams (`ngram_range=(1, 2)`) captures composite context that disambiguates domain-specific entities:
- **`comp.sys.mac.hardware` Bigrams:** `logic board`, `apple mac`, `mac iisi`, `powerbook duo`, `video card`, `scsi drive`.
- **`sci.electronics` Bigrams:** `circuit board`, `power supply`, `power consumption`, `rf circuit`, `audio output`, `printed circuit`.

These bigrams allow the vectorizer to allocate distinct, high-weight dimensions to platform-specific phrases versus general electrical engineering principles.

---

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

### Euclidean Norm Comparison ($\|x\|_2$)
From our corpus analysis of the longest post (72,161 characters) and shortest post (52 characters):

- **CountVectorizer (BoW):**
  - Shortest Post $\|x_{\text{bow}}\|_2$ = **2.00**
  - Longest Post $\|x_{\text{bow}}\|_2$ = **276.24**
  - *Magnitude difference:* **~138x**
- **TfidfVectorizer (L2-Normalized):**
  - Shortest Post $\|x_{\text{tfidf}}\|_2$ = **1.00**
  - Longest Post $\|x_{\text{tfidf}}\|_2$ = **1.00**
  - *Magnitude difference:* **0x (Identical length)**

### Why KMeans Fails on Raw Bag of Words Counts
KMeans optimizes within-cluster sum of squared Euclidean distances:
$$d(u, v) = \sqrt{\sum_{j=1}^{p} (u_j - v_j)^2}$$

In a raw Bag of Words representation, vector magnitude scales directly with document word count. A 2,000-word detailed post discussing automobile brake repairs and a 20-word brief question asking *"Where can I buy front brake pads?"* will lie vast distances apart in Euclidean space simply because the 2,000-word post has frequencies of 15–30 per word while the 20-word post has frequencies of 1. As a result, KMeans clusters documents based on **document verbosity (length)** rather than **topical semantics**.

### How Scikit-Learn's L2-Normalized TF-IDF Fixes This
Scikit-Learn's `TfidfVectorizer` automatically applies L2 normalization by default:
$$v_{\text{norm}} = \frac{v}{\|v\|_2}$$

This projects every document vector onto the surface of a unit hypersphere where $\|v_{\text{norm}}\|_2 = 1.0$. Because the Euclidean distance between two unit vectors is directly tied to cosine similarity:
$$\|u - v\|_2^2 = \|u\|_2^2 + \|v\|_2^2 - 2(u \cdot v) = 2 - 2\cos(u, v)$$

Clustering on L2-normalized vectors evaluates the **relative distribution of terms** (angle between vectors) rather than raw token counts, ensuring length-invariance.

---

## Challenge 4: Regulated Compliance Explainability

### Top 5 TF-IDF Features for Document #412
- **Document ID:** 412
- **Assigned Cluster:** Cluster 0
- **True Label:** `rec.autos`

Top 5 non-zero TF-IDF features and exact weights extracted via `explain_document(412, X_tfidf, tfidf_vec, top_n=5)`:
1. **`discussion`**: `0.4524`
2. **`feel`**: `0.3989`
3. **`days`**: `0.3855`
4. **`need`**: `0.2885`
5. **`meant`**: `0.2667`

### Justification for Regulated Compliance Workflows
TF-IDF provides a mathematically deterministic, transparent baseline where every routing decision can be audited down to exact feature weights and inverse document frequencies without hidden layers or stochastic variability. In regulated compliance workflows, auditors require reproducible and verifiable decision trails that eliminate the hallucination and prompt-drift risks inherent to black-box Large Language Models. Furthermore, TF-IDF feature attribution ensures complete explainability under strict data privacy, governance, and model risk management standards without unpredictable latency or costly external API dependencies.
