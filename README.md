# Resume Evaluation and Recommendation Script

This repository contains a Python script (`test.py`) that evaluates resumes, provides recommendations for improvement and analyzes their fit against job descriptions using large language models (LLMs).

## Features

- **Resume Scoring**: Generate a numerical score (0-100) to evaluate the quality of a resume.
- **Recommendation Generation**: Provide detailed suggestions to improve the resume.
- **Detailed Analysis**: Match resumes to job descriptions and provide a detailed fit analysis.

## Prerequisites

### Python Version
- Python 3.8 or later

### Required Libraries
Install the required dependencies:
```bash
pip install PyPDF2 sentence-transformers langchain faiss-cpu
```

## Usage 1

1. **Replace the Resume**:
   - Place the resume you want to evaluate in the `resume/` folder.
   - Ensure the file is in PDF format.
   - Example:
     ```
     project/resume/
     └── my-resume.pdf
     ```

2. **Add Job Descriptions**:
   - Place the job descriptions you want to match the resume against in the `job-description/` folder.
   - Ensure all job descriptions are in PDF format.
   - Example:
     ```
     project/job-description/
     ├── jd1.pdf
     ├── jd2.pdf
     └── jd3.pdf
     ```

3. **Execute the script using the following command**:
   - Example:
      ```bash
      python test.py
      ```
