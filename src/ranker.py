"""
Resume Ranker Module
Simple implementation for ranking resumes using TF-IDF + cosine similarity.
"""

import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeRanker:
    """
    Rank resumes based on job description similarity.
    """
    
    def __init__(self):
        """Initialize resume ranker."""
        self.job_description = None
        self.resumes = {}
        self.rankings = None

    @staticmethod
    def _clean_text(text):
        """Basic cleaning to keep processing easy to explain in viva."""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
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

        resume_files = sorted(
            filename for filename in os.listdir(resume_folder) if filename.endswith('.txt')
        )
        
        if not resume_files:
            raise ValueError(f"No .txt resume files found in {resume_folder}")
        
        self.resumes = {}
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
        
        print("Cleaning text...")
        cleaned_job = self._clean_text(self.job_description)
        resume_names = list(self.resumes.keys())
        cleaned_resumes = [self._clean_text(self.resumes[name]) for name in resume_names]
        
        print("Creating TF-IDF vectors...")
        all_documents = [cleaned_job] + cleaned_resumes
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_documents)
        
        job_vector = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]
        
        print("Calculating similarity scores...")
        similarities = cosine_similarity(job_vector, resume_vectors).flatten()

        rows = [
            {
                'Resume': name,
                'Similarity Score': round(score, 4),
                'Percentage': f"{round(score * 100, 2)}%"
            }
            for name, score in zip(resume_names, similarities)
        ]

        self.rankings = pd.DataFrame(rows).sort_values(
            by='Similarity Score', ascending=False
        ).reset_index(drop=True)
        self.rankings['Rank'] = self.rankings.index + 1
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
        
        print("\n" + "="*60)
        print("Resume Ranking Results")
        print("="*60 + "\n")
        
        for _, row in self.rankings.iterrows():
            print(f"Rank {row['Rank']}: {row['Resume']}")
            print(f"  Similarity Score: {row['Similarity Score']} ({row['Percentage']})")
            print()
        
        print("="*60)
    
    def save_rankings(self, output_file):
        """
        Save rankings to a CSV file.
        
        Args:
            output_file (str): Path to output CSV file
        """
        if self.rankings is None:
            raise ValueError("Rankings not computed yet. Call rank_resumes() first.")

        self.rankings.to_csv(output_file, index=False)
