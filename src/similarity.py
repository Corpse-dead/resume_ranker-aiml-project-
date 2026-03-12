"""
Similarity Calculation Module
Calculates similarity between documents using Cosine Similarity.
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SimilarityCalculator:
    """
    Calculate similarity between text documents.
    """
    
    def __init__(self):
        pass
    
    def calculate_cosine_similarity(self, vector1, vector2):
        """
        Calculate cosine similarity between two vectors.
        Range: -1 to 1 (1 = identical, 0 = no similarity)
        """
        similarity = cosine_similarity(vector1, vector2)
        return similarity[0][0]
    
    def calculate_similarities(self, query_vector, document_vectors):
        """
        Calculate cosine similarity between a query and multiple documents.
        
        Args:
            query_vector: TF-IDF vector of the query (job description)
            document_vectors: TF-IDF vectors of documents (resumes)
            
        Returns:
            array: Array of similarity scores for each document
        """
        # Calculate cosine similarity between query and all documents
        similarities = cosine_similarity(query_vector, document_vectors)
        
        # Flatten the result to get a 1D array
        return similarities.flatten()
    
    def get_top_matches(self, query_vector, document_vectors, document_names, top_n=None):
        """
        Get the top N most similar documents to the query.
        
        Args:
            query_vector: TF-IDF vector of the query
            document_vectors: TF-IDF vectors of documents
            document_names (list): Names/identifiers of documents
            top_n (int): Number of top matches to return (None = all)
            
        Returns:
            list: List of tuples (document_name, similarity_score) sorted by similarity
        """
        # Calculate similarities
        similarities = self.calculate_similarities(query_vector, document_vectors)
        
        # Create list of (name, score) tuples
        results = list(zip(document_names, similarities))
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N or all results
        if top_n:
            return results[:top_n]
        return results
    
    def similarity_to_percentage(self, similarity_score):
        """
        Convert similarity score to percentage.
        
        Args:
            similarity_score (float): Similarity score (0 to 1)
            
        Returns:
            float: Percentage representation (0 to 100)
        """
        return similarity_score * 100


def calculate_document_similarity(query_vector, document_vectors):
    """
    Convenience function to calculate similarity between query and documents.
    
    Args:
        query_vector: TF-IDF vector of the query
        document_vectors: TF-IDF vectors of documents
        
    Returns:
        array: Array of similarity scores
    """
    calculator = SimilarityCalculator()
    return calculator.calculate_similarities(query_vector, document_vectors)


if __name__ == "__main__":
    # Test the similarity module
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Sample documents
    job_desc = "python developer machine learning"
    resumes = [
        "python developer machine learning experience",
        "marketing manager communication skills",
        "data analyst python statistics"
    ]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    all_docs = [job_desc] + resumes
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    # Separate query and documents
    query_vector = tfidf_matrix[0]
    document_vectors = tfidf_matrix[1:]
    
    # Calculate similarities
    calculator = SimilarityCalculator()
    similarities = calculator.calculate_similarities(query_vector, document_vectors)
    
    print("Similarity Scores:")
    for i, score in enumerate(similarities, 1):
        print(f"  Resume {i}: {score:.4f} ({calculator.similarity_to_percentage(score):.2f}%)")
    
    # Get ranked matches
    resume_names = [f"Resume {i}" for i in range(1, len(resumes) + 1)]
    ranked = calculator.get_top_matches(query_vector, document_vectors, resume_names)
    
    print("\nRanked Resumes:")
    for rank, (name, score) in enumerate(ranked, 1):
        print(f"  {rank}. {name}: {score:.4f}")
