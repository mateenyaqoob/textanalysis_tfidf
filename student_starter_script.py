"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques
"""
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import text

# NLTK Downloads
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample Post Tokenization & Stemming
sample_post = "I'm looking for some help with my Mac hardware. Specifically, the graphics card seems to be overheating when running games. Any advice? 12345!"
tokens = word_tokenize(sample_post.lower())
all_alphabetic_tokens = [w for w in tokens if w.isalpha()]

stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in all_alphabetic_tokens if w not in stop_words]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(w, pos='v') for w in filtered_tokens]

# Fetching Dataset
categories = ['sci.electronics', 'comp.sys.mac.hardware', 'rec.autos', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
df = pd.DataFrame({'text': newsgroups.data, 'true_label': newsgroups.target})

# Vectorization
bow_vec = CountVectorizer(stop_words='english', max_features=4000, min_df=5)
tfidf_vec = TfidfVectorizer(stop_words='english', max_features=4000, min_df=5)
X_bow = bow_vec.fit_transform(df['text'])
X_tfidf = tfidf_vec.fit_transform(df['text'])

total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
non_zero_elements = X_tfidf.nnz
sparsity = 100.0 * (1.0 - (non_zero_elements / total_elements))

# Compute mean score per feature across documents and plot Top 10
mean_tfidf = np.asarray(X_tfidf.mean(axis=0)).ravel()
feature_names = np.array(tfidf_vec.get_feature_names_out())
top_indices = mean_tfidf.argsort()[-10:]
top_features = feature_names[top_indices]
top_scores = mean_tfidf[top_indices]

plt.figure(figsize=(10, 6))
plt.barh(top_features, top_scores, color='skyblue')
plt.xlabel('Average TF-IDF Score')
plt.ylabel('Features')
plt.title('Top 10 Words Ranked by Average TF-IDF Score')
plt.tight_layout()
plt.savefig('top_features.png')
plt.close()

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_tfidf)

# Heatmap of cluster centroids
centroids = kmeans.cluster_centers_
top_words_per_cluster = {}
for cluster_idx in range(4):
    sorted_centroid_indices = centroids[cluster_idx].argsort()[::-1]
    top_words_per_cluster[cluster_idx] = sorted_centroid_indices[:5]

visualized_word_indices = np.unique([idx for indices in top_words_per_cluster.values() for idx in indices])
visualized_words = feature_names[visualized_word_indices]
heatmap_data = centroids[:, visualized_word_indices]

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, xticklabels=visualized_words, yticklabels=[f'Cluster {i}' for i in range(4)], cmap='Blues', fmt=".4f")
plt.title("Cluster-by-Word Importance (Top 5 Words per Cluster Centroid)")
plt.tight_layout()
plt.savefig('cluster_heatmap.png')
plt.close()

# TruncatedSVD for 2D Spatial visualization
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X_tfidf)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='viridis', alpha=0.6, edgecolors='none', s=20)
plt.colorbar(scatter, label='Cluster ID')
plt.title("2D Spatial Document Clusters (TruncatedSVD)")
plt.tight_layout()
plt.savefig('spatial_mapping.png')
plt.close()

# Refined TfidfVectorizer
custom_stopwords = ['know', 'like', 'just', 'think', 'does', 'use', 'thanks', 'don', 'good']
stop_words_refined = list(text.ENGLISH_STOP_WORDS.union(custom_stopwords))
tfidf_vec_refined = TfidfVectorizer(
    stop_words=stop_words_refined,
    token_pattern=r'\b[a-zA-Z]{3,}\b',
    max_df=0.5,
    max_features=4000,
    min_df=5
)
X_tfidf_refined = tfidf_vec_refined.fit_transform(df['text'])
kans_refined = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster_refined'] = kans_refined.fit_predict(X_tfidf_refined)

# Explain Document function
def explain_document(doc_id, X_matrix, vectorizer, top_n=5):
    feats = np.array(vectorizer.get_feature_names_out())
    doc_row = X_matrix[doc_id].toarray().flatten()
    nz = doc_row.nonzero()[0]
    sorted_idx = nz[doc_row[nz].argsort()[::-1]]
    return [(feats[i], doc_row[i]) for i in sorted_idx[:top_n]]
