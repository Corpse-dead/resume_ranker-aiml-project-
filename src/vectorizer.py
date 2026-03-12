"""
TF-IDF Vectorizer Module
Converts text documents into TF-IDF vectors.
TF-IDF reflects how important a word is to a document.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class DocumentVectorizer:
    """
    Vectorize text documents using TF-IDF.
    """
    
    def __init__(self, max_features=None, ngram_range=(1, 2)):
        """
        Initialize TF-IDF Vectorizer.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=1,  # Minimum document frequency
            smooth_idf=True,  # Smooth IDF weights
            use_idf=True  # Enable IDF weighting
        )
        self.feature_names = None
    
    def fit_transform(self, documents):
        """
        Transform documents to TF-IDF vectors.
        """
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return tfidf_matrix
    
    def transform(self, documents):
        """
        Transform documents to TF-IDF vectors using the learned vocabulary.
        
        Args:
            documents (list): List of preprocessed text documents
            
        Returns:
            sparse matrix: TF-IDF weighted document-term matrix
        """
        return self.vectorizer.transform(documents)
    
    def get_feature_names(self):
        """
        Get the feature names (words) used by the vectorizer.
        
        Returns:
            array: Array of feature names
        """
        return self.feature_names
    
    def get_tfidf_scores(self, document_vector, top_n=10):
        """
        Get the top N words with highest TF-IDF scores for a document.
        
        Args:
            document_vector: TF-IDF vector for a single document
            top_n (int): Number of top words to return
            
        Returns:
            list: List of tuples (word, score) sorted by score
        """
        if self.feature_names is None:
            return []
        
        # Convert sparse matrix to dense array
        scores = np.asarray(document_vector.todense()).flatten()
        
        # Get indices of top N scores
        top_indices = scores.argsort()[-top_n:][::-1]
        
        # Create list of (word, score) tuples
        top_words = [(self.feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        
        return top_words


def create_tfidf_vectors(documents):
    """
    Convenience function to create TF-IDF vectors from documents.
    
    Args:
        documents (list): List of preprocessed text documents
        
    Returns:
        tuple: (vectorizer, tfidf_matrix)
    """
    vectorizer = DocumentVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix


if __name__ == "__main__":
    # Test the vectorizer module
    sample_docs = [
        "python developer machine learning experience",
        "marketing manager strong communication skills",
        "data analyst skilled python statistics"
    ]
    
    vectorizer = DocumentVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sample_docs)
    
    print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
    print("\nFeature Names (sample):")
    print(vectorizer.get_feature_names()[:10])
    
    print("\nTop words in first document:")
    top_words = vectorizer.get_tfidf_scores(tfidf_matrix[0], top_n=5)
    for word, score in top_words:
        print(f"  {word}: {score:.4f}")
