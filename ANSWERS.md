# Lab 15 — Problem-Solving Answers

All supporting code is in **Step 4** of `lab15_starter.py`. Running the script prints the
cluster keywords, crosstabs, vector norms and the Document #412 audit referenced below.

---

## Challenge 1: Conversational Filler & Numeric Noise

**Why does `know` still receive a substantial weight?**

IDF only punishes words that appear in almost *every* document. If `know` appears in 25% of posts:

$$\text{IDF} = \log\left(\frac{N}{DF}\right) = \log\left(\frac{1}{0.25}\right) = \log(4) \approx 1.39$$

That is far from zero (scikit-learn's smoothed version, `ln((1+N)/(1+DF)) + 1`, gives ≈ 2.39).
A rare topical word such as `carburetor` may score 4–6, so `know` is only discounted by a factor of
2–4, not removed. Filler words also have a **high term frequency** ("I know", "you know", "does anyone
know"), so `TF × IDF` stays large enough for them to appear among the top centroid words and pull
unrelated posts into the same cluster.

**Refinement applied** (`tfidf_clean_vec` in Step 4):

- `token_pattern=r'\b[a-zA-Z]{3,}\b'`: keeps only words of 3+ letters, so stray numbers (`00`, `10`,
  `486`) and contraction fragments (`ve`, `ll`, `re`) are dropped.
- `max_df=0.5`: removes any term that appears in more than half of all posts.
- **Custom stopwords** added to scikit-learn's English list: `know, like, just, think, does, use,
  thanks, used, using, want, need, good, make, way, really, say, people, time, ...`

**How the cluster top words change:**
In the baseline clusters, generic words (`know`, `like`, `just`, `does`, `thanks`) and numbers are mixed
into the top keywords, which makes the clusters hard to name. After cleaning, those words disappear and
are replaced by domain vocabulary such as *car, engine, oil, dealer* (autos), *doctor, patients, disease,
medical* (medicine), *mac, apple, drive, monitor* (Mac hardware) and *circuit, voltage, power, wire*
(electronics). Each cluster becomes readable as a clear topic queue.

---

## Challenge 2: Hardware Vocabulary Overlap

**Which two categories overlap the most?**
`pd.crosstab(df['true_label'], df['cluster'])` shows that **`sci.electronics` and
`comp.sys.mac.hardware`** overlap the most. `rec.autos` and `sci.med` each map mostly to a single cluster,
while electronics and Mac posts are split across the same cluster(s).

**Shared technical words pulling them together:**
Both communities write about hardware, so they share words such as **`power`, `board`, `chip`, `card`,
`memory`, `drive`, `monitor`, `video`, `port`, `cable`**. The script prints the strongest shared terms
under *"Shared electronics/Mac vocabulary"*, which ranks words with a high average TF-IDF in **both**
categories.

**How bigrams (`ngram_range=(1, 2)`) help:**
Single words are ambiguous: `power` can be "power supply" (electronics) or "PowerBook" (Mac), and `drive`
can be a "hard drive" or something in a car. With bigrams, the vectorizer also learns phrases that belong
to only one community, such as **`hard drive`, `scsi drive`, `apple monitor`** for Mac hardware and
**`power supply`, `op amp`, `circuit diagram`** for electronics. These phrases add context that single
words lose, so K-Means can separate the two groups better. The script compares the cluster purity of the
baseline, cleaned and bigram models to show this.

---

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

**Comparison of Euclidean norms ($\|x\|_2$):**

- **`X_bow` (raw counts):** the longest post has a **very large** norm because every repeated word adds
  to its counts, while the shortest post has a **tiny** norm. The two differ by orders of magnitude.
- **`X_tfidf`:** **both posts have a norm of exactly 1.0**, because `TfidfVectorizer` normalizes every row
  to unit length by default (`norm='l2'`). A post can only show 0 if all of its words were filtered out.

**Why K-Means fails on raw Bag of Words:**
K-Means groups points by **Euclidean distance**. With raw counts, a 2,000-word rant sits far from the
origin and a 20-word question sits close to it, so distances are driven by **post length, not topic**.
Two short posts on completely different subjects end up "closer" than a short and a long post on the
same subject, and clusters form around *long vs. short* instead of *cars vs. medicine*.

**How L2-normalized TF-IDF fixes it:**
Scaling every vector to length 1 removes the length effect and keeps only the **direction**, meaning the
relative mix of words. For unit vectors, Euclidean distance is directly tied to cosine similarity
($\|a-b\|^2 = 2 - 2\cos(a,b)$), so K-Means now compares **what** a post talks about, not **how much** it
says.

---

## Challenge 4: Regulated Compliance Explainability

**Implementation:**
`explain_document(doc_id, matrix, vectorizer, top_n=5)` in `lab15_starter.py` takes the TF-IDF row for
the document, keeps only its non-zero entries, sorts them by weight and returns the top `top_n` as
`(term, weight)` pairs. Calling `explain_document(412, X_tfidf, tfidf_vec, top_n=5)` prints Document
#412's true label, its assigned cluster, and the **exact five words and weights** that placed it there.
An auditor can match these words against the top words of that cluster's centroid to see why the post
was routed to it.

**Justification (3 sentences):**
TF-IDF gives a fixed, reproducible score to every word, so an auditor can see exactly which words in
Document #412, and with what weights, placed it closest to its cluster. Each weight can be recalculated
by hand from word counts and document frequencies, which creates a complete, documented audit trail for
regulators who require every decision to be traceable and open to challenge. A black-box LLM may give
different answers to the same input, cannot show a faithful word-level reason for its decision, and can
invent explanations, so TF-IDF is the safer transparent baseline for regulated compliance workflows.
