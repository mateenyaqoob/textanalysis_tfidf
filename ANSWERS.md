# Lab 15 – Problem-Solving Answers

> Numbers marked **[FILL IN]** come from running `python lab15_starter.py` on your machine. Copy them from the terminal output.

## Challenge 1: Conversational Filler & Numeric Noise

**Why `know` still gets a substantial weight.** With the formula TF-IDF = TF × log(N / DF), a word in 25% of posts has an IDF of log(1 / 0.25) = log 4 ≈ 1.39 (sklearn's smoothed variant gives a similar value of about 1.29). That is a penalty, not a zero. Only a word appearing in every post gets an IDF near 0. So a filler word that appears several times in a chatty post keeps a high TF and a non-trivial IDF, and its weight stays comparable to real content words. Standard stopword lists also don't contain `know`, `think`, `thanks` or `use`, so nothing removes them.

**Refinement.** I used `token_pattern=r'\b[a-zA-Z]{3,}\b'` (drops numbers and 1–2 letter fragments), `max_df=0.5` (drops terms in more than half of posts), and a custom stopword list (sklearn's English stopwords plus `know, like, just, think, does, use, thanks, don, ve, ll`).

**How the cluster top words change.** Before: **[FILL IN baseline top words per cluster]**. After: **[FILL IN cleaner top words per cluster]**. Expected pattern: generic filler and number fragments disappear from the top 8, and domain words (for example engine/car/dealer, patient/doctor/disease, drive/mac/apple, circuit/voltage/power) move up.

## Challenge 2: Hardware Vocabulary Overlap

**Most overlapping categories.** From `pd.crosstab(df['true_label'], df['cluster'])`: **[FILL IN, expected `sci.electronics` and `comp.sys.mac.hardware`]**.

**Shared technical words.** Misclustered posts typically share words such as `power`, `board`, `chip`, `battery`, `drive`, `card`, `monitor`, `cable`, `supply`. **[Replace with 3+ words you actually saw in misclustered posts.]**

**Bigrams.** With `ngram_range=(1, 2)`, single ambiguous words like `power` or `board` become discriminative phrases such as `power supply` vs `power mac`, `logic board`, `hard drive`, `mac plus`. Phrases carry context that single tokens lose, so electronics-bench vocabulary and Mac product vocabulary separate better. Crosstab with bigrams: **[FILL IN whether overlap decreased]**.

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

**Norms (longest vs shortest post).**

| Matrix | Longest post ‖x‖₂ | Shortest post ‖x‖₂ |
|---|---|---|
| `X_bow` | **[FILL IN]** (large) | **[FILL IN]** (small) |
| `X_tfidf` | **[FILL IN]** (≈ 1.0) | **[FILL IN]** (≈ 1.0) |

**Why KMeans struggles on raw BoW.** KMeans uses Euclidean distance. Raw counts scale with document length, so a 2,000-word rant has a huge vector norm and a 20-word question has a tiny one. Distance is then dominated by length, not topic: two short posts on different topics can look closer than a short and a long post on the same topic, and long posts pull centroids toward themselves.

**How TF-IDF fixes it.** `TfidfVectorizer` applies L2 normalization by default (`norm='l2'`), so every row has unit length. Euclidean distance between unit vectors depends on the angle between them (it is monotonic in cosine similarity), so clustering reflects vocabulary mix rather than length. IDF also down-weights terms common across the corpus.

## Challenge 4: Regulated Compliance Explainability

`explain_document(412, X_tfidf, tfidf_vec, top_n=5)` output:

**[FILL IN the 5 terms and weights printed by the script, plus doc 412's true label and cluster.]**

**Justification (3 sentences).** TF-IDF is fully transparent: every routing decision traces back to exact, auditable term weights that a reviewer can inspect and reproduce deterministically. A black-box LLM cannot give a faithful, repeatable account of why it produced an output, which is hard to defend under regulatory audit, and it adds cost, data-privacy exposure and version drift. TF-IDF with KMeans is cheap, runs on-premises, and gives a stable baseline against which any more complex model can be validated.
