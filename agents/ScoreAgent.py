from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
import json

class ScoreAgent:
    def __init__(self, llm_model="llama3.2"):
        self.llm = Ollama(model=llm_model)
        self.template = """
            You are an expert evaluator of resumes for job applications.
            Grade the provided resume on a scale of 0 to 100 based on the following criteria:
            1. Clarity and Organization:
            - Is the resume easy to read and logically structured?
            - Are sections like Education, Experience, Skills, and Certifications clearly defined?
            2. Quality of Achievements and Skills Mentioned
            - Are the achievements specific, measurable, and impactful (e.g., "Increased revenue by 20%" or "Developed an AI model with 95% accuracy")?
            - Are the skills aligned with the desired role (e.g., technical, soft skills)?
            3. Educational Background:
            - Check for relevance and quality of the candidate's university or educational institution.
            - Assess if the degree(s) mentioned align with the requirements of the job.
            4. Certifications and Training:
            - Evaluate the quality and relevance of certifications (e.g., AWS Certified, PMP, Coursera courses).
            - Consider the recency and credibility of these certifications.
            5. Work Experience:
            - Assess the depth and breadth of the candidate's work experience.
            - Check for progression (e.g., junior to senior roles) and alignment with the job requirements.
            6. Technical and Industry-Specific Skills:
            - Look for advanced skills, tools, and technologies relevant to the job (e.g., Python, TensorFlow, Agile methodologies).
            - Evaluate familiarity with industry best practices.
            7. Extracurricular and Additional Activities (Optional):
            - Consider any extracurricular achievements, volunteer work, or personal projects that enhance the candidate’s profile.
            8. No need to provide score breakdown, just return the result in json format.
            
            Provide the score in the following JSON format:
            {{
            "score": <integer between 0 and 100>
            }}

            Resume:
            {text}

            JSON response:
        """
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=self.template
        )

    def generate_score(self, resume_text):
        formatted_prompt = self.prompt.format(
            text=resume_text
        )
        response = self.llm(formatted_prompt)
        return json.loads(response)
