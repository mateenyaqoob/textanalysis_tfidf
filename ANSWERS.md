# Lab 15 Answers: TF-IDF & Unstructured Text Triage

> Fill in the numbers/word lists marked **[from your output]** after you run `python lab15_starter.py`.

## Challenge 1: Conversational Filler & Numeric Noise

**Why does `know` still get a substantial weight at 25% document frequency?**
TF-IDF is `TF(t,d) x log(N / DF_t)`. If `know` appears in 25% of posts, then `N / DF = 4` and `log(4)` is roughly 1.39 (natural log). That is a small *penalty*, not a removal: the IDF term only goes to zero when a word appears in *every* document. Meanwhile `know` can have a high raw term frequency inside the posts that use it, so `TF x IDF` stays large. A word that is common but not universal is not suppressed enough, and the standard English stopword list does not contain chatty words like `know`, `like`, `think`, `just`, `thanks`.

**Fix:** `TfidfVectorizer(token_pattern=r'\b[a-zA-Z]{3,}\b', max_df=0.5, stop_words=<English + custom filler>)`.
- `token_pattern` drops numbers and 1-2 letter fragments.
- `max_df=0.5` drops any word appearing in more than half the posts.
- Custom stopwords remove filler that is below that threshold.

**How cluster top words change:** [from your output] Before, centroids contained generic words (e.g. `like`, `know`, `just`, `use`). After filtering, each cluster's top words become domain vocabulary (e.g. car/engine/dealer for autos, drug/patient/doctor for medical, mac/drive/scsi for Mac hardware, circuit/voltage for electronics), so the clusters are easier to label.

## Challenge 2: Hardware Vocabulary Overlap

**Crosstab:** [from your output: paste the `pd.crosstab(df['true_label'], df['cluster'])` table].

**Most overlapping categories:** `sci.electronics` and `comp.sys.mac.hardware` (the script also computes this automatically as the pair with the most similar cluster distribution). Both discuss hardware, so their posts are split across the same clusters.

**Shared technical words pulling them together:** [from your output: "Shared top-40 words"]. Typical examples: `drive`, `card`, `power`, `scsi`, `board`, `chip`, `memory`, `monitor`.

**How bigrams help:** With `ngram_range=(1, 2)`, ambiguous single words become more specific phrases, e.g. `hard drive`, `power supply`, `mac ii`, `floppy drive` versus `circuit board`, `op amp`. A phrase like `power supply` or `mac ii` is much more category-specific than `power` or `drive` alone, so the two hardware groups separate better. The crosstab with bigrams is printed by the script for comparison.

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

[from your output: "Challenge 3: L2 norms"]

- **BoW:** the norm of a raw count vector grows with document length. A 2,000-word rant has a very large norm; a 20-word question has a tiny one.
- **TF-IDF:** `TfidfVectorizer` L2-normalizes each row by default (`norm='l2'`), so every non-empty document has norm 1.0, regardless of length.

**Why KMeans fails on raw BoW:** KMeans uses Euclidean distance. With raw counts, distance is dominated by document *magnitude* (length), not topic. Long posts sit far from the origin and from everything else, so short posts clump together and long posts form their own clusters or distort centroids, so clusters group by length rather than subject. L2-normalized TF-IDF puts all documents on the unit sphere, so Euclidean distance becomes proportional to cosine distance and depends on the *direction* (word mix), i.e. the topic.

## Challenge 4: Regulated Compliance Explainability

Output of `explain_document(doc_id=412, X_tfidf, tfidf_vec, top_n=5)`: [from your output: 5 terms and exact weights].

**Justification (3 sentences):**
TF-IDF routing is fully transparent: the assigned cluster can be traced to a short list of named words and their exact numeric weights, which an auditor can verify and reproduce deterministically. A black-box LLM cannot give a faithful, reproducible account of why it made a decision, and its outputs may vary between runs or model versions, which is hard to defend in a regulated adverse-event workflow. TF-IDF is also cheap, runs on-premises without sending patient data to a third party, and so makes a strong, auditable baseline against which any more complex model must be justified.
