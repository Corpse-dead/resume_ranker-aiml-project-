"""
Resume Ranking System - Main Entry Point

This is the main script to run the Resume Ranking System.
It automatically ranks candidate resumes based on similarity to a job description
using Natural Language Processing techniques.

B.Tech CSE (AI/ML) Project
Date: March 2026
"""

import os
import sys
from src.ranker import ResumeRanker


def print_banner():
    """Print a welcome banner."""
    print("\n" + "="*70)
    print("RESUME RANKING SYSTEM".center(70))
    print("Powered by NLP & Machine Learning".center(70))
    print("="*70 + "\n")


def print_section(title):
    """Print a section header."""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print('─'*70)


def main():
    """
    Main function to execute the resume ranking system.
    """
    # Print welcome banner
    print_banner()
    
    # Define file paths
    job_description_path = os.path.join('data', 'job_description.txt')
    resumes_folder = os.path.join('data', 'resumes')
    output_file = 'rankings_output.csv'
    
    try:
        # Initialize the ranker
        print_section("INITIALIZING SYSTEM")
        print("✓ Loading Resume Ranking System...")
        ranker = ResumeRanker()
        print("✓ System initialized successfully!\n")
        
        # Load job description
        print_section("LOADING JOB DESCRIPTION")
        print(f"Reading from: {job_description_path}")
        job_desc = ranker.load_job_description(job_description_path)
        print(f"✓ Job description loaded ({len(job_desc)} characters)\n")
        
        # Display job description preview
        print("Job Description Preview:")
        print("-" * 70)
        preview = job_desc[:300] + "..." if len(job_desc) > 300 else job_desc
        print(preview)
        print("-" * 70)
        
        # Load resumes
        print_section("LOADING RESUMES")
        print(f"Reading from folder: {resumes_folder}")
        resumes = ranker.load_resumes(resumes_folder)
        print(f"✓ Loaded {len(resumes)} resume(s):\n")
        for i, resume_name in enumerate(resumes.keys(), 1):
            print(f"  {i}. {resume_name}")
        
        # Rank resumes
        print_section("PROCESSING & RANKING")
        rankings = ranker.rank_resumes()
        print("✓ Ranking completed successfully!\n")
        
        # Display rankings
        print_section("RESULTS")
        ranker.display_rankings()
        
        # Save rankings
        print_section("SAVING RESULTS")
        ranker.save_rankings(output_file)
        print(f"✓ Results saved to: {output_file}\n")
        
        # Summary
        print_section("SUMMARY")
        print(f"Total Resumes Analyzed: {len(resumes)}")
        print(f"Best Match: {rankings.iloc[0]['Resume']}")
        print(f"Best Score: {rankings.iloc[0]['Similarity Score']} "
              f"({rankings.iloc[0]['Percentage']})")
        print()
        
        print("="*70)
        print("PROCESS COMPLETED SUCCESSFULLY!".center(70))
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure the following structure exists:")
        print("  - data/job_description.txt")
        print("  - data/resumes/ (folder with .txt resume files)")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERROR: An unexpected error occurred:")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)


def test_system():
    """
    Test function to verify all modules are working.
    """
    print("Testing Resume Ranking System modules...")
    print()
    
    try:
        from src.preprocess import TextPreprocessor
        print("✓ Preprocessing module loaded")
        
        from src.vectorizer import DocumentVectorizer
        print("✓ Vectorizer module loaded")
        
        from src.similarity import SimilarityCalculator
        print("✓ Similarity module loaded")
        
        from src.ranker import ResumeRanker
        print("✓ Ranker module loaded")
        
        print("\n✓ All modules loaded successfully!")
        print("\nSystem is ready to use. Run 'python main.py' to start ranking resumes.\n")
        
    except ImportError as e:
        print(f"❌ Error loading modules: {e}")
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt\n")
        sys.exit(1)


if __name__ == "__main__":
    # Check if user wants to test the system
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_system()
    else:
        main()
