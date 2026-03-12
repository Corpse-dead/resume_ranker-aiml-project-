"""
Text Preprocessing Module
Handles text preprocessing tasks:
- Lowercasing
- Removing punctuation
- Removing stopwords
- Tokenization
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (will only download if not already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """
    Preprocess text data for NLP tasks.
    """
    
    def __init__(self, language='english'):
        """
        Initialize preprocessor with stopwords.
        """
        self.stop_words = set(stopwords.words(language))
    
    def clean_text(self, text):
        """
        Clean and preprocess the input text.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Join tokens back
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def preprocess_documents(self, documents):
        """
        Preprocess multiple documents.
        
        Args:
            documents (list): List of text documents
            
        Returns:
            list: List of preprocessed documents
        """
        return [self.clean_text(doc) for doc in documents]


def preprocess_text(text):
    """
    Convenience function to preprocess a single text string.
    
    Args:
        text (str): Text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    preprocessor = TextPreprocessor()
    return preprocessor.clean_text(text)


if __name__ == "__main__":
    # Test the preprocessing module
    sample_text = """
    Hello! I'm a Python Developer with 5+ years of experience in Machine Learning.
    Email: test@example.com | Website: www.example.com
    Skills: Python, TensorFlow, scikit-learn, NLP, Data Analysis!!!
    """
    
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.clean_text(sample_text)
    
    print("Original Text:")
    print(sample_text)
    print("\nCleaned Text:")
    print(cleaned)
