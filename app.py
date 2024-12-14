import os
import streamlit as st
from PyPDF2 import PdfReader
from src.ScoreAgent import ScoreAgent
from src.RecommenderAgent import RecommenderAgent
from src.AnalyzerAgent import AnalyzerAgent

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit UI
st.title("Resume Evaluation and Job Matching System")
st.write("Upload your resume and job descriptions to get a score, recommendations, and a detailed analysis.")

# Upload Resume
uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
uploaded_jds = st.file_uploader("Upload Job Descriptions (PDFs)", type=["pdf"], accept_multiple_files=True)

if st.button("Evaluate"):
    if not uploaded_resume:
        st.error("Please upload a resume.")
    elif not uploaded_jds:
        st.error("Please upload at least one job description.")
    else:
        # Extract Resume Text
        st.write("Extracting text from resume...")
        resume_text = extract_text_from_pdf(uploaded_resume)

        # Save the job descriptions locally and extract text
        job_description_texts = {}
        for jd_file in uploaded_jds:
            jd_path = f"temp_jd_{jd_file.name}"
            with open(jd_path, "wb") as f:
                f.write(jd_file.read())
            job_description_texts[jd_file.name] = extract_text_from_pdf(jd_path)
            os.remove(jd_path)  # Clean up the temporary file

        # Initialize Agents
        st.write("Initializing agents and processing data...")
        scoreAgent = ScoreAgent()
        recommenderAgent = RecommenderAgent()
        analyzerAgent = AnalyzerAgent(resume_text=resume_text)

        # Load Job Descriptions into AnalyzerAgent
        analyzerAgent.load_job_descriptions(job_description_texts)
        analyzerAgent.split_text_into_chunks()
        analyzerAgent.generate_embeddings()
        analyzerAgent.initialize_vectorstore()

        # Generate Outputs
        st.write("Generating results...")
        score = scoreAgent.generate_score(resume_text)
        recommendation = recommenderAgent.generate_recommendation(resume_text)
        analyzerAgent.retrieve_documents()
        analysis = analyzerAgent.generate_detailed_analysis()

        # Display Results
        st.subheader("Results")
        st.write("**Resume Score:**")
        st.success(score)
        st.write("**Recommendations:**")
        # st.info(recommendation)
        st.json(recommendation)
        st.write("**Detailed Analysis:**")
        # st.text_area("Analysis", analysis, height=300)
        st.json(analysis)

st.write("Note: All data is processed locally. No files are uploaded to external servers.")
