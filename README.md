# Resume Screening System

## Overview

This is a Python-based resume screening system that ranks candidate resumes based on their similarity to a given job description. It supports PDF and DOCX resume formats and uses natural language processing (NLP) with TF-IDF vectorization and cosine similarity to match resumes with job requirements.

---

## Features

- Extracts text from PDF and DOCX resume files.
- Preprocesses text using spaCy (lemmatization, stopword removal).
- Supports keyword-based skill extraction from resumes.
- Uses TF-IDF and cosine similarity to rank resumes against a job description.
- Outputs ranked list of candidates with similarity scores and extracted skills.

---

## Requirements

Make sure you have the following installed before running the project:

- **Python 3.7+**
- **pip** (Python package manager)

### Python packages

Install the required Python packages using pip:

```bash
pip install PyPDF2 python-docx spacy scikit-learn
````

### spaCy English model

Download the spaCy English model used for text preprocessing:

```bash
python -m spacy download en_core_web_sm
```

---

## Setup

1. Clone or download the repository.

2. Place your resume files (`.pdf` or `.docx`) inside the resumes folder at:

   ```
   D:\personal\python\resume_screening\resumes
   ```

   (You can change this path in the `main()` function if needed.)

3. Customize the job description inside the script as per your needs.

---

## Running the Program

Run the main Python script:

```bash
python resume_screening.py
```

The script will read resumes, preprocess them, compare them with the job description, and output a ranked list of candidates along with extracted skills.

---

## Customization

* Modify the `SKILLS` list in the script to include skills relevant to your domain.
* Update the `job_description` variable with your specific job requirements.
* Change the resume folder path if needed.

---

## License

This project is open-source and available under the MIT License.

---

## Acknowledgements

* [spaCy](https://spacy.io/)
* [PyPDF2](https://pypi.org/project/PyPDF2/)
* [python-docx](https://python-docx.readthedocs.io/en/latest/)
* [scikit-learn](https://scikit-learn.org/stable/)

---

If you have any questions or suggestions, feel free to open an issue or submit a pull request.

```

---

Would you like me to generate a `requirements.txt` file or help you with a GitHub repo setup?
```
