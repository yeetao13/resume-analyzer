import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class AnalyzerAgent:
    def __init__(self, resume_text, llm_model="llama3.2", embedding_model="all-MiniLM-L6-v2"):
        self.llm = Ollama(model=llm_model)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vectorstore = None
        self.resume_text = resume_text

        self.pdf_texts = {}
        self.chunks = {}
        self.embeddings = {}
        self.query = "Which job description fits this resume: " + resume_text

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file.
        """
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def load_job_descriptions(self, pdf_dir):
        """
        Extract text from all PDFs in a directory.
        """
        # pdf_texts = {}
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, filename)
                self.pdf_texts[filename] = self.extract_text_from_pdf(pdf_path)
        print("+"*50)
        print(len(self.pdf_texts))
        print(self.pdf_texts)
        print("+"*50)
        # return pdf_texts

    def split_text_into_chunks(self, chunk_size=500, chunk_overlap=50):
        """
        Split text into chunks for processing.
        """
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # chunks = {}
        for filename, text in self.pdf_texts.items():
            self.chunks[filename] = splitter.split_text(text)
        # return chunks

    def generate_embeddings(self):
        # self.embeddings = {}
        for filename, text_chunks in self.chunks.items():
            self.embeddings[filename] = [self.embedding_model.encode(chunk) for chunk in text_chunks]
        # return embeddings

    def initialize_vectorstore(self):
        """
        Create and initialize a FAISS vectorstore.
        """
        all_chunks = []
        all_embeddings = []
        for filename, text_chunks in self.chunks.items():
            for i, chunk in enumerate(text_chunks):
                all_chunks.append(Document(page_content=chunk, metadata={"source": filename, "chunk_index": i}))
                all_embeddings.append(self.embeddings[filename][i])

        # Convert embeddings to NumPy array
        all_embeddings_np = np.array(all_embeddings)

        # Initialize FAISS index
        dimension = all_embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(all_embeddings_np)

        # Create a docstore
        docstore = InMemoryDocstore({str(i): all_chunks[i] for i in range(len(all_chunks))})
        index_to_docstore_id = {i: str(i) for i in range(len(all_chunks))}

        self.vectorstore = FAISS(
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function=lambda text: self.embedding_model.encode([text])[0]
        )

    def retrieve_documents(self, top_k=3):
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized.")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        retrieved_docs = retriever.get_relevant_documents(self.query)

        retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
        retrieved_jd = ",".join([doc.metadata["source"] for doc in retrieved_docs])
        retrieved_jd=set(retrieved_jd.split(","))
        return {
            "retrieved_context": retrieved_context,
            "retrieved_jd": retrieved_jd
            }

    # def generate_summary(self):
    #     template = """
    #     You are an expert career coach. Summarize the key details of the following resume into a concise professional summary.

    #     Resume:
    #     {resume_text}

    #     Professional Summary:
    #     """
    #     prompt = PromptTemplate(input_variables=["resume_text"], template=template)
    #     formatted_prompt = prompt.format(resume_text=self.resume_text)
    #     return self.llm(formatted_prompt)

    def generate_detailed_analysis(self):
        retrieved_context=self.retrieve_documents()["retrieved_context"]
        retrieved_jd=self.retrieve_documents()["retrieved_jd"]

        print("="*50)
        print(retrieved_context)
        print(retrieved_jd)
        print("="*50)

        template = """
        You are a career advisor and professional evaluator. Your task is to analyze a candidate's resume, 
        the retrieved job descriptions, and the most suitable job description PDF to generate a detailed summary.

        Provide a comprehensive analysis of the candidate's fit for the most suitable job, including the following:
        1. An overall evaluation of how well the candidate's resume aligns with the suitable job description.
        2. Key strengths and qualifications in the resume that make the candidate a good match for the job.
        3. Any gaps or areas where the candidate may need improvement to better meet the job requirements.
        4. A professional recommendation on whether the candidate should pursue this job and why.

        Resume:
        {resume_text}

        Retrieved Job Descriptions:
        {retrieved_context}

        Most Suitable Job Description (PDF):
        {retrieved_jd}

        Generate a detailed summary of the candidate's fit for the job:
        """
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain=(
            RunnablePassthrough()
            | prompt
            | self.llm
            | StrOutputParser()
        )
        response=rag_chain.invoke({"resume_text":self.resume_text,
                                   "retrieved_context":retrieved_context,
                                   "retrieved_jd": retrieved_jd})

        # formatted_prompt = prompt.format(
        #     resume_text=self.resume_text,
        #     retrieved_context=self.retrieve_documents()["retrieved_context"],
        #     retrieved_jd=self.retrieve_documents()["retrieved_jd"]
        # )
        return {
            "response": response,
            "retrieved_jd": retrieved_jd
        }