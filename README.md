# Resume Ranking System 🎯

An AI-powered Natural Language Processing system that automatically ranks candidate resumes based on their similarity to a given job description using Machine Learning techniques.

## 📋 Project Overview

This is a B.Tech Computer Science (AI/ML specialization) academic project. The system uses NLP techniques including TF-IDF vectorization and cosine similarity to match resumes with job descriptions and provide ranked results.

### Key Features

✅ **Text Preprocessing**: Comprehensive text cleaning including lowercasing, punctuation removal, stopword removal, and tokenization

✅ **TF-IDF Vectorization**: Converts text documents into numerical feature vectors

✅ **Cosine Similarity**: Calculates similarity scores between job descriptions and resumes

✅ **Automatic Ranking**: Ranks all resumes in descending order of relevance

✅ **Detailed Output**: Provides resume names, similarity scores, and percentage matches

## 🗂️ Project Structure

```
resume_ranker/
│
├── data/
│   ├── resumes/
│   │   ├── resume1.txt          # Sample resume 1 (Python ML Developer)
│   │   ├── resume2.txt          # Sample resume 2 (Marketing Manager)
│   │   └── resume3.txt          # Sample resume 3 (Data Analyst)
│   │
│   └── job_description.txt      # Sample job description
│
├── src/
│   ├── __init__.py              # Package initializer
│   ├── preprocess.py            # Text preprocessing module
│   ├── vectorizer.py            # TF-IDF vectorization module
│   ├── similarity.py            # Cosine similarity calculation
│   └── ranker.py                # Main ranking logic
│
├── main.py                      # Entry point to run the system
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🔧 Technologies Used

- **Python 3.8+**
- **scikit-learn**: Machine learning library for TF-IDF and cosine similarity
- **NLTK**: Natural Language Toolkit for text preprocessing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd resume_ranker

# Or simply download and extract the project folder
```

### Step 2: Install Required Packages

```bash
pip install -r requirements.txt
```

This will install:
- scikit-learn
- nltk
- pandas
- numpy

### Step 3: Verify Installation

```bash
python main.py --test
```

This command will test all modules and confirm the system is ready to use.

## 🚀 How to Run

### Basic Usage

1. Ensure you have a job description file at `data/job_description.txt`
2. Place resume files (`.txt` format) in the `data/resumes/` folder
3. Run the main script:

```bash
python main.py
```

### Using Your Own Data

#### Adding Your Job Description

Edit or replace `data/job_description.txt` with your job posting content.

#### Adding Resumes

1. Convert your resumes to `.txt` format
2. Place them in the `data/resumes/` folder
3. Name them descriptively (e.g., `john_smith.txt`, `jane_doe.txt`)

## 📊 Output

### Console Output

The system will display:
- Job description preview
- List of loaded resumes
- Processing steps
- Ranked results with scores

Example output:
```
======================================================================
                        RESUME RANKING RESULTS
======================================================================

Rank 1: resume1.txt
  Similarity Score: 0.8234 (82.34%)

Rank 2: resume3.txt
  Similarity Score: 0.6521 (65.21%)

Rank 3: resume2.txt
  Similarity Score: 0.1045 (10.45%)

======================================================================
```

### CSV Output

Results are also saved to `rankings_output.csv` with columns:
- Rank
- Resume
- Similarity Score
- Percentage

## 🧪 Testing Individual Modules

Each module can be tested independently:

```bash
# Test preprocessing
cd src
python preprocess.py

# Test vectorizer
python vectorizer.py

# Test similarity calculator
python similarity.py
```

## 📚 How It Works

### 1. Text Preprocessing

The system cleans and normalizes text by:
- Converting to lowercase
- Removing URLs and emails
- Removing punctuation and special characters
- Tokenizing text into words
- Removing stopwords (common words like 'the', 'is', 'and')
- Filtering short words

### 2. Feature Extraction (TF-IDF)

**TF-IDF** (Term Frequency-Inverse Document Frequency) converts text into numerical vectors:
- **TF**: How often a term appears in a document
- **IDF**: How unique/important a term is across all documents
- Higher scores = more important and distinctive terms

### 3. Similarity Calculation

**Cosine Similarity** measures the angle between document vectors:
- Score of 1.0 = identical documents
- Score of 0.0 = no similarity
- Score between 0 and 1 = degree of similarity

### 4. Ranking

Resumes are sorted by similarity score in descending order, with the best match ranked first.

## 🎯 Sample Results

Using the provided sample data:

**Job Description**: Python Developer with ML and Data Analysis skills

**Expected Rankings**:
1. **resume1.txt** (John Smith - Python ML Developer) → **Highest Score** (~82%)
   - Strong match due to Python, ML, TensorFlow, scikit-learn, data analysis

2. **resume3.txt** (Michael Chen - Data Analyst) → **Medium Score** (~65%)
   - Good match for Python, data analysis, statistics

3. **resume2.txt** (Sarah Johnson - Marketing Manager) → **Low Score** (~10%)
   - Poor match due to different domain (marketing vs technical)

## 🔄 Workflow Diagram

```
┌─────────────────────┐
│  Job Description    │
│  & Resumes (Text)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Preprocessing      │
│  - Lowercasing      │
│  - Remove stopwords │
│  - Tokenization     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  TF-IDF             │
│  Vectorization      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Cosine Similarity  │
│  Calculation        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Ranked Results     │
│  Display & Save     │
└─────────────────────┘
```

## 🚀 Future Enhancements

Potential improvements for the project:

1. **PDF Support**: Add PDF resume parsing using PyMuPDF
2. **Web Interface**: Create a Streamlit UI for easy interaction
3. **Skill Extraction**: Identify and extract specific skills from resumes
4. **Advanced Matching**: Implement deep learning models (BERT, transformers)
5. **Multi-language Support**: Extend to handle resumes in different languages
6. **Database Integration**: Store results in a database
7. **Batch Processing**: Handle large volumes of resumes efficiently

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'nltk'`
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: NLTK data not found errors
- **Solution**: The system automatically downloads required NLTK data, but if it fails:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```

**Issue**: No resumes found
- **Solution**: Ensure resume files are `.txt` format and in `data/resumes/` folder

**Issue**: UnicodeDecodeError
- **Solution**: Ensure text files are saved in UTF-8 encoding

## 📝 Academic Notes

### Key Concepts Demonstrated

- **Natural Language Processing**: Text preprocessing and feature extraction
- **Machine Learning**: TF-IDF vectorization and similarity measurement
- **Software Engineering**: Modular design, code organization, documentation
- **Data Science**: Working with text data and generating insights

### Learning Outcomes

✅ Understanding of NLP preprocessing techniques

✅ Implementation of TF-IDF vectorization

✅ Application of cosine similarity for text matching

✅ Building end-to-end ML systems

✅ Python programming and software design

## 📄 License

This project is created for academic purposes. Feel free to use and modify for educational projects.

## 👨‍💻 Author

B.Tech Computer Science (AI/ML) academic project.

## 📧 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments for detailed explanations
3. Test individual modules to isolate issues

---

**Happy Ranking! 🎉**
