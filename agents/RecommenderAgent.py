from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
import json

class RecommenderAgent:
    def __init__(self, llm_model="llama3.2"):
        self.llm = Ollama(model=llm_model)
        self.template = """
            You are an expert career coach and resume advisor. Your task is to review the provided resume and generate detailed recommendations for improving it based on the following factors:
            1. Clarity and Organization:
            - Is the resume easy to read and logically structured?
            - Are sections like Education, Experience, Skills, and Certifications clearly defined and formatted?
            - Suggest improvements for better readability and organization (e.g., use of bullet points, section headers, consistent formatting).
            2. Quality of Achievements and Skills:
            - Are the achievements specific, measurable, and impactful (e.g., "Increased sales by 30%" or "Designed an algorithm with 95% accuracy")?
            - Suggest improvements for presenting achievements more effectively using action verbs and quantifiable metrics.
            - Identify any missing or poorly highlighted skills relevant to the target job.
            3. Educational Background:
            - Assess how the candidate's educational qualifications are presented.
            - Recommend improvements, such as emphasizing degrees, coursework, or relevant academic achievements.
            4. Certifications and Training:
            - Evaluate the relevance and quality of certifications mentioned (e.g., AWS Certified, PMP, or other industry-recognized certifications).
            - Suggest adding certifications or training that align with the candidate's career goals or the target job.
            5. Work Experience:
            - Assess the depth and relevance of the candidate's work experience.
            - Provide suggestions for enhancing descriptions of responsibilities, achievements, and job titles.
            - Recommend rephrasing or reorganizing work experience to better showcase progression or accomplishments.
            6. Technical and Industry-Specific Skills:
            - Evaluate the technical and industry-specific skills mentioned.
            - Suggest skills or tools the candidate should add to make the resume more competitive.
            7. Extracurricular and Additional Activities (Optional):
            - Highlight any personal projects, volunteer work, or extracurricular activities that could enhance the resume.
            - Suggest ways to include additional activities that demonstrate leadership, teamwork, or other transferable skills.
            8. General Feedback:
            - Provide any additional recommendations for improving the overall impression of the resume.
            - Identify any critical missing information (e.g., contact details, LinkedIn profile, professional summary).
            9. No need to provide anything else, just return the result in json format.
            
            Provide your recommendations in the following JSON format:
            {{
            "recommendations": [
                {{
                "section": "<section name>",
                "suggestion": "<specific recommendation for improvement>"
                }},
                {{
                "section": "<section name>",
                "suggestion": "<specific recommendation for improvement>"
                }}
            ]
            }}

            Resume:
            {text}

            JSON response:
        """
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=self.template
        )

    def generate_recommendation(self, resume_text):
        formatted_prompt = self.prompt.format(
            text=resume_text
        )
        response = self.llm(formatted_prompt)
        return json.loads(response)
