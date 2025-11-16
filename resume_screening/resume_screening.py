import os
import PyPDF2
from docx import Document
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('en_core_web_sm')

SKILLS = ["FullStack Developer", "Python"]
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def extract_skills(text):
    found_skills = [skill for skill in SKILLS if skill in text]
    return found_skills

def load_resume_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format")

def main():
    resume_folder = r"D:\personal\python\resume_screening\resumes"
    print("Resume folder path:", resume_folder)

    if not os.path.exists(resume_folder):
        print("Creating folder...")
        os.makedirs(resume_folder, exist_ok=True)
    else:
        print("Folder already exists.")

    resume_texts = []
    resume_files = []

    for filename in os.listdir(resume_folder):
        if filename.endswith('.pdf') or filename.endswith('.docx'):
            file_path = os.path.join(resume_folder, filename)
            raw_text = load_resume_text(file_path)
            clean_text = preprocess_text(raw_text)
            resume_texts.append(clean_text)
            resume_files.append(filename)

    if len(resume_texts) == 0:
        print(f"No resumes found in '{resume_folder}'. Please add resume files and try again.")
        return

    job_description = """
    We are looking for a Python developer with experience in machine learning,
    data analysis, and web frameworks like Django or Flask.
    Must have skills in SQL and NLP.
    """
    job_description_clean = preprocess_text(job_description)

    documents = resume_texts + [job_description_clean]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    job_vector = tfidf_matrix[-1]
    resume_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(resume_vectors, job_vector).flatten()

    ranked_resumes = sorted(zip(resume_files, similarities), key=lambda x: x[1], reverse=True)

    print("Candidate Ranking Based on Job Description Similarity:\n")
    for rank, (filename, score) in enumerate(ranked_resumes, 1):
        print(f"{rank}. {filename} - Similarity Score: {score:.4f}")
        idx = resume_files.index(filename)
        skills = extract_skills(resume_texts[idx])
        print(f"   Extracted Skills: {skills}\n")

if __name__ == "__main__":
    main()
