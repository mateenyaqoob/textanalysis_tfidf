# Unstructured Text Analysis & Automated Triage with TF-IDF

**Course:** Data Science Tools & Techniques  
**Topic:** NLP Pipeline, Bag of Words (BoW), TF-IDF Vectorization, and Unsupervised Topic Discovery  

---

## Scenario: User Triage & Bottleneck Discovery

An online tech and consumer platform has accumulated nearly 4,000 unstructured user posts and support discussions across four domains (*Electronics*, *Mac Hardware*, *Automobiles*, and *Medical/Health*). Human moderators are experiencing reading fatigue, and leadership needs an automated, **100% explainable** pipeline to group incoming posts into topical queues and surface defining vocabulary without relying on expensive black-box models.

In this activity, you will build an end-to-end NLP pipeline that standardizes messy user text, compares **Bag of Words** against **TF-IDF**, discovers hidden topical clusters with **K-Means**, and diagnoses real-world unstructured text pitfalls.

---

## Learning Objectives

1. **Clean & Normalize Text:** Apply tokenization, stopword filtering, heuristic stemming (`PorterStemmer`), and morphological lemmatization (`WordNetLemmatizer`).
2. **Vectorize Unstructured Data:** Build document-term matrices using `CountVectorizer` and `TfidfVectorizer`, and quantify matrix sparsity.
3. **Discover & Visualize Topics:** Extract top TF-IDF features, map document similarity in 2D space using `TruncatedSVD`, and cluster documents via `KMeans`.
4. **Diagnose Real-World Text Traps:** Solve conversational noise, vocabulary overlap, document-length bias, and stakeholder explainability challenges.

---

## Repository Structure

```text
.
├── README.md                 # Lab instructions, problem-solving prompts, and rubric
├── requirements.txt          # Python package dependencies
├── .gitignore                # Standard Python/Jupyter Git ignore rules
├── student_starter_script.py          # Student starter script with TODO blocks (# %% cell compatible)
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Starter Script
`student_starter_script.py` uses `# %%` interactive cell markers. You can either:
* Open it in **VS Code** (with the Jupyter extension) or **Spyder/PyCharm** to run cell-by-cell like a notebook, **or**
* Run it directly from your terminal:
```bash
python student_starter_script.py
```
*(The dataset downloads automatically via Scikit-Learn's `fetch_20newsgroups`—no manual CSV setup required!)*

---

## Part 1: Programming Experience (40 Minutes)

Complete all `TODO` sections inside `lab15_starter.py`:

* **Step 1: Text Cleaning, Tokenization & Normalization**
  * Tokenize raw user posts into lowercase alphabetic words.
  * Filter out standard English stopwords using NLTK.
  * Compare `PorterStemmer` vs. `WordNetLemmatizer` outputs on the exact same tokens.
* **Step 2: Bag of Words vs. TF-IDF Feature Extraction**
  * Transform the corpus using both `CountVectorizer` and `TfidfVectorizer` (`max_features=4000`, `min_df=5`).
  * Compute the percentage of zero entries (**sparsity**) in the TF-IDF matrix.
  * Plot a horizontal bar chart (`top_features.png`) of the top 10 words ranked by average TF-IDF score.
* **Step 3: Automated Topic Discovery & Spatial Mapping**
  * Fit `KMeans(n_clusters=4)` directly on the TF-IDF matrix.
  * Generate a cluster-by-word importance heatmap (`cluster_heatmap.png`) from the cluster centroids.
  * Project the sparse TF-IDF vectors into 2D using `TruncatedSVD` and plot the spatial document clusters (`spatial_mapping.png`).

---

## Part 2: Problem-Solving Experience (20 Minutes)

Create a new file named `ANSWERS.md` in your repository and answer the four diagnostic challenges below (supported by the code you write in **Step 4** of `student_starter_script.py`):

### Challenge 1: Conversational Filler & Numeric Noise
User posts contain conversational filler (`know`, `like`, `think`, `just`, `does`, `thanks`, `use`) and stray numbers that bypass standard English stopword lists.
* **Task:** Recall the TF-IDF formula:
  $$\text{TFIDF}_{t,d} = \text{TF}_{t,d} \times \log\left(\frac{N}{\text{DF}_t}\right)$$
  Explain why a filler word like `know` still receives a substantial weight if it appears in 25% of posts. Then, refine `TfidfVectorizer` using `token_pattern=r'\b[a-zA-Z]{3,}\b'`, `max_df`, and custom stopwords. How do your cluster top words change?

### Challenge 2: Hardware Vocabulary Overlap (`sci.electronics` vs. `comp.sys.mac.hardware`)
Run `pd.crosstab(df['true_label'], df['cluster'])` to compare your unsupervised clusters against the hidden ground-truth categories.
* **Task:** Identify which two categories overlap the most. Inspect misclustered posts and list at least 3 shared technical words pulling them together. How does adding bigrams (`ngram_range=(1, 2)`) help separate general electronics from Mac hardware?

### Challenge 3: Rant Length Bias (BoW vs. TF-IDF)
Online forums contain both 2,000-word rants and concise 20-word questions.
* **Task:** Compare the Euclidean norm ($\|x\|_2$) of the longest post and shortest post in `X_bow` versus `X_tfidf`. Explain why `KMeans` fails on raw Bag of Words counts for variable-length user text, and how Scikit-Learn's L2-normalized `TfidfVectorizer` fixes this.

### Challenge 4: Regulated Compliance Explainability
Suppose the `sci.med` posts represent patient adverse-event reports in a regulated healthcare audit. An auditor asks *why* Document `#412` was routed to its assigned cluster.
* **Task:** Implement `explain_document(doc_id=412, top_n=5)` to output the exact top 5 non-zero TF-IDF features and weights for that post. Write a 3-sentence justification explaining why TF-IDF is preferred over a black-box LLM as a transparent baseline in regulated compliance workflows.

---

## 📤 Submission Instructions

This activity uses GitHub Classroom and Automated Grading. Your progress and solutions will be submitted automatically through Git.

Step 1: Accept the Assignment Invitation
Click the GitHub Classroom assignment link provided by your instructor to generate your own private student repository (e.g., data-science-lab15-yourusername).
Clone your newly created repository to your local machine or open it directly in VS Code / GitHub Codespaces.

Step 2: Complete the Programming & Problem-Solving Tasks

Open student_starter_script.py in your code editor.
Fill in the missing code blocks for Text Cleaning & Normalization, Bag-of-Words vs. TF-IDF Extraction, and K-Means Clustering as outlined in the script comments.
Write your short answers for the Problem-Solving Questions as comments or within a markdown summary.

Step 3: Test Your Code Locally (Recommended)

Before submitting, run the test suite on your own machine to ensure all checks pass:

# Install dependencies
pip install -r requirements.txt
pip install pytest

# Run the test suite
pytest test_lab15.py

Step 4: Submit via Git Push

Once you have completed the lab and verified your code works, commit and push your changes to your remote repository:
git add lab15_starter.py
git commit -m "Complete Lab 15 TF-IDF and K-Means clustering pipeline"
git push origin main

Step 5: Verify Your Automated Grade

Go to your repository page on GitHub.Click on the Actions tab at the top.
Check the latest workflow run. If your code passes all pytest checks, a green checkmark ($\checkmark$) will appear, indicating your assignment has been successfully evaluated and submitted!
