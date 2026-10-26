"""
Automated Pytest Suite for Lab 15: TF-IDF & Unstructured Data Triage
This script validates student implementations in student_starter_script.py / lab15_starter.py.
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# Import student starter script or check functions
try:
    import lab15_starter as student
except ImportError:
    student = None


def test_starter_file_exists():
    assert os.path.exists("lab15_starter.py"), "lab15_starter.py is missing from the repository."


def test_dataset_loading():
    assert student is not None, "Could not import lab15_starter.py"
    assert hasattr(student, "df"), "DataFrame 'df' is missing."
    assert len(student.df) > 100, "Dataset contains fewer than 100 rows."
    assert "text" in student.df.columns, "Column 'text' is missing from DataFrame."


def test_step1_tokenization():
    assert hasattr(student, "filtered_tokens"), "Variable 'filtered_tokens' not found."
    assert isinstance(student.filtered_tokens, list), "filtered_tokens must be a Python list."
    assert len(student.filtered_tokens) > 0, "filtered_tokens list is empty."
    assert all(isinstance(w, str) for w in student.filtered_tokens), "All tokens must be strings."


def test_step2_vectorization():
    assert hasattr(student, "X_tfidf"), "Matrix 'X_tfidf' not found."
    assert hasattr(student, "tfidf_vec"), "Vectorizer 'tfidf_vec' not found."
    
    # Verify TF-IDF matrix properties
    shape = student.X_tfidf.shape
    assert shape[0] > 0 and shape[1] > 0, "X_tfidf matrix has invalid dimensions."
    
    # Verify sparsity calculation
    total_elements = shape[0] * shape[1]
    expected_sparsity = 100.0 * (1.0 - (student.X_tfidf.nnz / total_elements))
    assert hasattr(student, "sparsity"), "Variable 'sparsity' not found."
    assert np.isclose(student.sparsity, expected_sparsity, atol=0.1), "Sparsity calculation is incorrect."


def test_step3_clustering():
    assert hasattr(student, "kmeans"), "KMeans model 'kmeans' not found."
    assert isinstance(student.kmeans, KMeans), "kmeans must be an instance of KMeans."
    assert student.kmeans.n_clusters == 4, "KMeans should be configured with 4 clusters."


def test_challenge_functions():
    assert hasattr(student, "explain_document"), "Function 'explain_document' is missing."
    # Test explain_document signature and output format if implemented
    try:
        explanation = student.explain_document(0, student.X_tfidf, student.tfidf_vec, top_n=3)
        if explanation is not None:
            assert isinstance(explanation, list), "explain_document must return a list of tuples."
            if len(explanation) > 0:
                assert len(explanation[0]) == 2, "Each item must be a (term, weight) tuple."
    except Exception as e:
        pytest.fail(f"explain_document raised an exception during test execution: {e}")
