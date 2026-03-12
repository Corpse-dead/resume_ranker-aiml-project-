"""
Resume Ranker Module
Integrates all components to rank resumes based on job description similarity.
"""

import os
import pandas as pd
from .preprocess import TextPreprocessor
from .vectorizer import DocumentVectorizer
from .similarity import SimilarityCalculator


class ResumeRanker:
    """
    Rank resumes based on job description similarity.
    """
    
    def __init__(self):
        """Initialize resume ranker."""
        self.preprocessor = TextPreprocessor()
        self.vectorizer = DocumentVectorizer()
        self.similarity_calculator = SimilarityCalculator()
        
        self.job_description = None
        self.resumes = {}
        self.rankings = None
    
    def load_job_description(self, file_path):
        """Load job description from file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Job description file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.job_description = f.read()
        
        return self.job_description
    
    def load_resumes(self, resume_folder):
        """Load all resume files from folder."""
        if not os.path.exists(resume_folder):
            raise FileNotFoundError(f"Resume folder not found: {resume_folder}")
        
        # Get all text files in the folder
        resume_files = [f for f in os.listdir(resume_folder) if f.endswith('.txt')]
        
        if not resume_files:
            raise ValueError(f"No .txt resume files found in {resume_folder}")
        
        # Load each resume
        for resume_file in resume_files:
            file_path = os.path.join(resume_folder, resume_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                self.resumes[resume_file] = f.read()
        
        return self.resumes
    
    def rank_resumes(self):
        """Rank resumes based on similarity to job description."""
        if not self.job_description:
            raise ValueError("Job description not loaded. Call load_job_description() first.")
        
        if not self.resumes:
            raise ValueError("No resumes loaded. Call load_resumes() first.")
        
        # Preprocess job description
        print("Preprocessing job description...")
        preprocessed_job = self.preprocessor.clean_text(self.job_description)
        
        # Preprocess all resumes
        print(f"Preprocessing {len(self.resumes)} resumes...")
        resume_names = list(self.resumes.keys())
        resume_texts = list(self.resumes.values())
        preprocessed_resumes = self.preprocessor.preprocess_documents(resume_texts)
        
        # Create TF-IDF vectors
        print("Creating TF-IDF vectors...")
        all_documents = [preprocessed_job] + preprocessed_resumes
        tfidf_matrix = self.vectorizer.fit_transform(all_documents)
        
        # Separate job description vector from resume vectors
        job_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]
        
        # Calculate similarities
        print("Calculating similarity scores...")
        similarities = self.similarity_calculator.calculate_similarities(
            job_vector, resume_vectors
        )
        
        # Create results dataframe
        results = []
        for resume_name, similarity_score in zip(resume_names, similarities):
            results.append({
                'Resume': resume_name,
                'Similarity Score': round(similarity_score, 4),
                'Percentage': f"{round(similarity_score * 100, 2)}%"
            })
        
        # Create DataFrame and sort by similarity score
        self.rankings = pd.DataFrame(results)
        self.rankings = self.rankings.sort_values('Similarity Score', ascending=False)
        self.rankings['Rank'] = range(1, len(self.rankings) + 1)
        
        # Reorder columns
        self.rankings = self.rankings[['Rank', 'Resume', 'Similarity Score', 'Percentage']]
        
        return self.rankings
    
    def get_rankings(self):
        """
        Get the computed rankings.
        
        Returns:
            pandas.DataFrame: Rankings dataframe
        """
        if self.rankings is None:
            raise ValueError("Rankings not computed yet. Call rank_resumes() first.")
        return self.rankings
    
    def display_rankings(self):
        """
        Display the rankings in a formatted way.
        """
        if self.rankings is None:
            raise ValueError("Rankings not computed yet. Call rank_resumes() first.")
        
        print("\n" + "="*70)
        print("RESUME RANKING RESULTS".center(70))
        print("="*70 + "\n")
        
        for _, row in self.rankings.iterrows():
            print(f"Rank {row['Rank']}: {row['Resume']}")
            print(f"  Similarity Score: {row['Similarity Score']} ({row['Percentage']})")
            print()
        
        print("="*70)
    
    def save_rankings(self, output_file):
        """
        Save rankings to a CSV file.
        
        Args:
            output_file (str): Path to output CSV file
        """
        if self.rankings is None:
            raise ValueError("Rankings not computed yet. Call rank_resumes() first.")
        
        self.rankings.to_csv(output_file, index=False)
        print(f"\nRankings saved to: {output_file}")


if __name__ == "__main__":
    # Test the ranker module
    print("Testing Resume Ranker Module...")
    
    ranker = ResumeRanker()
    
    # This is just a test - actual paths will be provided by main.py
    print("\nNote: This is a test module.")
    print("Use main.py to run the complete resume ranking system.")
