# Answers

## Challenge 1: Conversational filler and numeric noise

"know" shows up in about 23% of the posts (842 of 3703). TF-IDF is `TF * log(N / DF)`, and with a DF that high
the log part is still positive (sklearn's smoothed idf for "know" is about 2.48). It only becomes zero
if a word appears in every document. So the idf penalty is small, not zero, and since "know" is used a
lot inside each post it has a high TF. TF times that idf gives it a good-sized weight, which is why it
ends up at the top of the cluster 0 keywords.

I changed the vectorizer to use `token_pattern=r'\b[a-zA-Z]{3,}\b'`, `max_df=0.5` and extra stopwords
(know, like, just, think, does, use, thanks). The numbers and one or two letter tokens are gone now.
In the baseline cluster 0 was "know, like, don, just, does, use, thanks, good". After cleaning the
filler words are gone from the top words, and cluster 1 now has "don, good, people, time, used,
information", which is more general discussion vocabulary. The car and Mac clusters stayed almost the
same (car, cars, engine, ford / mac, apple, drive, monitor, scsi). One small cluster (71 `sci.med` posts) is still made up of odd
words like "cadre, dsl, chastity", which look like they come from a repeated signature in those posts.

## Challenge 2: Electronics vs Mac hardware overlap

From `pd.crosstab(df['true_label'], df['cluster'])`, `sci.electronics` and `comp.sys.mac.hardware` overlap
the most. 60 electronics posts landed in the Mac cluster (cluster 3). Most of the rest of electronics
went to cluster 0, which is the general "everything else" cluster.

Words that pull them together (top words of those misclustered posts): card, drive, monitor, software,
ram, mhz, board. These are hardware terms that both groups use, e.g. someone asking about a video card
or drive in an electronics group looks the same as a Mac video card question.

With `ngram_range=(1, 2)` the features include two word phrases (for example "hard drive"), so
shared single words like "drive" or "card" get some context. In my run the number of electronics posts
in the Mac cluster dropped from 60 to 35, while the Mac cluster kept 600 of its posts. It is not a
perfect separation, but it is better.

## Challenge 3: Rant length bias

Longest post (4483 kept words): BoW norm = 276.24, TF-IDF norm = 1.00.
Shortest post (1 kept word): BoW norm = 1.00, TF-IDF norm = 1.00.

In raw counts the long post has a vector that is hundreds of times bigger than the short one. KMeans
uses Euclidean distance, so the long rants end up far from everything else and the short question
sits close to the origin. Clusters then form by post length and not by topic. `TfidfVectorizer`
L2-normalizes every row to length 1, so every document lies on the unit sphere and only its direction
(the mix of words) matters. Length no longer affects the distance.

## Challenge 4: Explainability for a regulated audit

`explain_document(412, X_tfidf, tfidf_vec, top_n=5)` prints the top 5 non-zero terms and weights:

| Term | Weight |
|---|---|
| discussion | 0.4524 |
| feel | 0.3989 |
| days | 0.3855 |
| need | 0.2885 |
| meant | 0.2667 |

(In this dataset doc 412 is actually a `rec.autos` post, assigned to cluster 0, not a `sci.med`
one. The function works the same way for any document.)

Justification: TF-IDF gives the auditor the exact words and numeric weights that put a document in a
cluster, so every routing decision can be checked and reproduced by hand. An LLM can't show that, since its
decision comes from millions of parameters and an explanation it writes may not match what really drove
the output. For regulated work, a transparent, deterministic baseline is easier to validate, document
and defend, and it can be run locally without sending patient data to an external model.
