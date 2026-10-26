# Diagnostic Answers

## Challenge 1: Conversational Filler & Numeric Noise

If `know` appears in 25% of the documents, its inverse document frequency is approximately $\log(1/0.25)=\log(4)$, or 1.386 with the natural logarithm used by scikit-learn. That is not close to zero, so the term can still contribute substantial TF-IDF weight when it occurs repeatedly in an individual post. The filtered vectorizer removes tokens shorter than three letters, terms appearing in more than half the posts, and the specified conversational words; its printed cluster keywords should therefore emphasize more discriminating domain vocabulary than the baseline.

## Challenge 2: Hardware Vocabulary Overlap

The two categories expected to overlap most are `sci.electronics` and `comp.sys.mac.hardware`, because Mac hardware discussions also use general electronics terminology. Shared terms that can pull posts together include `card`, `drive`, `monitor`, and `modem`; the script prints the true-label-by-cluster crosstab for checking the actual split in this dataset. Adding bigrams gives the model more specific phrases such as `video card`, `hard drive`, and `Mac OS`, which preserve useful context that isolated words lose and can distinguish general components from Mac-specific discussions.

## Challenge 3: Rant Length Bias

The script prints the L2 norms for the longest and shortest posts in both matrices. With raw Bag of Words, the longer post generally has a much larger norm because repeated word counts accumulate, so Euclidean distance and K-Means can be driven by document length instead of topic. Scikit-learn's default TF-IDF normalization scales each nonempty document vector to unit L2 norm, reducing this length bias while retaining the relative importance of its terms.

## Challenge 4: Regulated Compliance Explainability

`explain_document(doc_id=412, top_n=5)` returns and prints the exact five highest nonzero TF-IDF terms and weights for document 412; this result is also assigned to `document_412_explanation` when the script runs. TF-IDF is preferred as a transparent baseline because each routing signal is an inspectable term with a reproducible numeric weight. Unlike a black-box LLM, the feature representation does not require an opaque generated rationale, making the decision easier to reproduce, audit, and challenge in a regulated workflow.
