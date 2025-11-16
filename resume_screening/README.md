# Resume Screening & Analysis Tool

A comprehensive web application for analyzing resumes, extracting keywords, skills, and contact information, and ranking candidates based on job description similarity. Built with Flask and powered by NLP (spaCy) and machine learning (scikit-learn).

## Features

### Single Resume Analysis
- **Text Extraction**: Extract text from PDF and DOCX resume files
- **Keyword Extraction**: Advanced keyword extraction using TF-IDF and NLP techniques
- **Skills Detection**: Automatically detect predefined skills from resumes
- **Contact Information**: Extract emails, phone numbers, LinkedIn, and GitHub profiles
- **Named Entity Recognition**: Identify organizations, products, and technologies
- **Visualizations**: Interactive keyword frequency charts using Plotly

### Batch Resume Processing
- **Batch Analysis**: Process multiple resumes from a folder
- **Job Description Matching**: Rank resumes by similarity to a job description using cosine similarity
- **Comparative Analysis**: View all candidates side-by-side with similarity scores
- **Aggregate Statistics**: See overall keyword frequency across all resumes

## Project Structure

```
resume_screening/
├── flask_app.py          # Main Flask application
├── app.py                # Streamlit alternative implementation
├── resume_screening.py   # Simple CLI script
├── requirements.txt      # Python dependencies
├── templates/           # Flask HTML templates
│   ├── base.html        # Base template with styling
│   ├── index.html       # Main upload page
│   └── results.html     # Results display page
├── resumes/             # Sample resume files (PDF/DOCX)
└── README.md           # This file
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### 1. Clone or Download the Project

Navigate to the project directory:
```bash
cd resume_screening
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install spaCy Language Model

The application uses spaCy for advanced NLP features. Install the English language model:

```bash
python -m spacy download en_core_web_sm
```

**Note:** If the spaCy model is not installed, the application will still work but with reduced functionality (basic keyword extraction without advanced NLP features).

## Running the Application

### Option 1: Flask Web Application (Recommended)

Run the main Flask application:

```bash
python flask_app.py
```

The application will start on `http://localhost:8000` or `http://127.0.0.1:8000`

Open your web browser and navigate to the URL to access the application.

**To run on a different port:**
Edit `flask_app.py` and change the port in the last line:
```python
app.run(host='0.0.0.0', port=YOUR_PORT, debug=True)
```

### Option 2: Streamlit Application

If you prefer Streamlit's interface:

```bash
streamlit run app.py
```

### Option 3: Command Line Script

For simple batch processing from command line:

```bash
python resume_screening.py
```

**Note:** You may need to update the `resume_folder` path in `resume_screening.py` to match your system.

## Usage Guide

### Single Resume Analysis

1. **Access the Application**: Open `http://localhost:8000` in your browser
2. **Upload Resume**: 
   - Click "Choose File" under "Single Resume Analysis"
   - Select a PDF or DOCX file
3. **Configure Options**:
   - Set the number of top keywords to extract (default: 20)
   - Toggle "Show Named Entities" to display NLP entities
   - Toggle "Show Contact Info" to extract contact details
4. **Analyze**: Click "Analyze Resume" button
5. **View Results**: 
   - See extracted text preview
   - View top keywords with frequency chart
   - Check detected skills
   - Review contact information and named entities

### Batch Resume Processing

1. **Prepare Resume Folder**: 
   - Create a folder containing multiple resume files (PDF or DOCX)
   - Example: `E:\Python\resume_screening\resumes`
2. **Enter Folder Path**:
   - In the "Batch Resume Processing" section
   - Enter the full path to your resume folder
3. **Enter Job Description**:
   - Paste or type the job description in the text area
   - This will be used to calculate similarity scores
4. **Configure Options**: Same as single resume analysis
5. **Process**: Click "Process Resumes" button
6. **View Results**:
   - See all resumes ranked by similarity score
   - View similarity score chart
   - Check detailed analysis for each candidate
   - Review aggregate keyword frequency

## Supported File Formats

- **PDF**: `.pdf` files (using PyPDF2)
- **DOCX**: Microsoft Word documents (using python-docx)

## Configuration

### Environment Variables

You can set a custom Flask secret key using an environment variable:

**Windows (PowerShell):**
```powershell
$env:FLASK_SECRET_KEY="your-secret-key-here"
python flask_app.py
```

**Linux/Mac:**
```bash
export FLASK_SECRET_KEY="your-secret-key-here"
python flask_app.py
```

### Skills List

To customize the skills detection, edit the `SKILLS` list in `flask_app.py`:

```python
SKILLS = [
    "FullStack Developer", "Python", "JavaScript", "Java", 
    "React", "Node.js", "Django", "Flask", "SQL", "MongoDB", 
    "Machine Learning", "Data Analysis", "NLP"
]
```

## Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
   - **Error**: `OSError: Can't find model 'en_core_web_sm'`
   - **Solution**: Run `python -m spacy download en_core_web_sm`
   - **Note**: The app will work without it, but with reduced functionality

2. **Port Already in Use**
   - **Error**: `Address already in use`
   - **Solution**: Change the port in `flask_app.py` or stop the process using port 8000

3. **File Upload Fails**
   - **Error**: `Unsupported file format`
   - **Solution**: Ensure files are PDF or DOCX format

4. **Folder Path Not Found (Batch Processing)**
   - **Error**: `Resume folder does not exist`
   - **Solution**: 
     - Use absolute paths (e.g., `E:\Python\resume_screening\resumes`)
     - On Windows, use backslashes or forward slashes
     - Ensure the folder contains PDF or DOCX files

5. **Import Errors**
   - **Error**: `ModuleNotFoundError`
   - **Solution**: 
     - Ensure virtual environment is activated
     - Run `pip install -r requirements.txt` again
     - Check Python version (requires 3.8+)

### Performance Tips

- For large batch processing (100+ resumes), processing may take several minutes
- The application processes files sequentially for reliability
- Consider processing in smaller batches for better performance

## Dependencies

- **Flask** (>=3.0.0): Web framework
- **PyPDF2** (>=3.0.0): PDF text extraction
- **python-docx** (>=1.0.0): DOCX text extraction
- **spacy** (>=3.7.0): Natural language processing
- **scikit-learn** (>=1.3.0): Machine learning (TF-IDF, cosine similarity)
- **pandas** (>=2.0.0): Data manipulation (for Streamlit version)
- **plotly** (>=5.17.0): Interactive visualizations
- **gunicorn** (>=21.2.0): Production WSGI server (optional)

## Development

### Running in Development Mode

The Flask app runs in debug mode by default. To disable:

```python
app.run(host='0.0.0.0', port=8000, debug=False)
```

### Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:8000 flask_app:app
```

## License

This project is open source and available for educational and commercial use.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review error messages in the application
3. Check console output for detailed error information

---

**Happy Resume Screening!** 🎯

