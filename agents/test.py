from PyPDF2 import PdfReader
from ScoreAgent import ScoreAgent
from RecommenderAgent import RecommenderAgent
from AnalyzerAgent import AnalyzerAgent

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Example: Load the PDF file
resume_path = "../resume/business-resume.pdf"
resume_text = extract_text_from_pdf(resume_path)

# print("Extracted Resume Text:")
# print(resume_text)

scoreAgent = ScoreAgent()
score = scoreAgent.generate_score(resume_text)

recommenderAgent = RecommenderAgent()
recommendation = recommenderAgent.generate_recommendation(resume_text)

analyzerAgent = AnalyzerAgent(resume_text=resume_text)
analyzerAgent.load_job_descriptions("../jd")
analyzerAgent.split_text_into_chunks()
analyzerAgent.generate_embeddings()
analyzerAgent.initialize_vectorstore()
analyzerAgent.retrieve_documents()
analysis = analyzerAgent.generate_detailed_analysis()

print("-" * 50)
print(score)
print("-" * 50)
print(recommendation)
print("-" * 50)
print(analysis)
