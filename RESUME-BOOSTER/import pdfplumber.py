
import pdfplumber
import spacy
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Ensure the 'resumes' folder exists
resume_folder = "resumes"
if not os.path.exists(resume_folder):
    print(f"Creating folder: {os.path.abspath(resume_folder)}")
    os.makedirs(resume_folder)
else:
    print(f"Folder already exists: {os.path.abspath(resume_folder)}")

print(f"Checking folder existence: {resume_folder} -> Exists? {os.path.exists(resume_folder)}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Function to calculate resume score
def calculate_resume_score(resume_texts, job_description):
    vectorizer = TfidfVectorizer()
    texts = [job_description] + resume_texts
    tfidf_matrix = vectorizer.fit_transform(texts)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return scores.flatten()

# Main execution
if __name__ == "__main__":
    job_description = "Looking for a Python Developer with experience in NLP, machine learning, and OpenCV."
    job_description = preprocess_text(job_description)
    
    resume_texts = []
    resume_files = []
    
    if os.path.exists(resume_folder):
        resume_files_list = os.listdir(resume_folder)
        if not resume_files_list:
            print("Warning: No resume files found in the 'resumes' folder!")
        else:
            print(f"Found {len(resume_files_list)} resume(s): {resume_files_list}")
            for file in resume_files_list:
                if file.endswith(".pdf"):
                    resume_text = extract_text_from_pdf(os.path.join(resume_folder, file))
                    resume_texts.append(preprocess_text(resume_text))
                    resume_files.append(file)
    else:
        print("Error: 'resumes' folder is missing!")
    
    if resume_texts:
        scores = calculate_resume_score(resume_texts, job_description)
        
        # Display results
        ranked_resumes = sorted(zip(resume_files, scores), key=lambda x: x[1], reverse=True)
        print("\nRanked Resumes Based on Job Description:")
        for rank, (file, score) in enumerate(ranked_resumes, start=1):
            print(f"{rank}. {file} - Score: {score:.2f}")
    else:
        print("No resumes to analyze.")
