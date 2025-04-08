class ResumeLoader():
    def __init__(self , resume_name):
    
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.vectorstores import FAISS
        from langchain.embeddings import OpenAIEmbeddings

        self.my_pdf_loader = PyPDFLoader(resume_name)
        self.pdf_pages = my_pdf_loader.load()
        self.resume_text_string = " ".join([page.page_content for page in pdf_pages])
        
        self.chunker = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.resume_in_chumks = chunker.split_text(resume_text_string)

        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_texts(resume_in_chumks, embedding=embeddings)




    def score_resume_using_context(job_description):

        from langchain.chains import RetrievalQA
        from langchain.chat_models import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4")
        RAG_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff"
        )

        query = f"""
        Given the following job description:

        {job_description}

        Evaluate candidate’s resume based on: Relevant experience (1–5), Technical skill match (1–5) , Communication & clarity (1–5)

        Provide a JSON output.
        """
        response = RAG_chain.run(query)
        return response ## parse this JSON into dictionary



class QuestionEval():
    def __init__(self):
        
        import os
        from langchain.chains import LLMChain
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import PromptTemplate
        self.LLMChain = LLMChain


        os.environ["OPENAI_API_KEY"] = "key here..."
        self.llm_model = ChatOpenAI(model="gpt-4", temperature=0)

        input_variables=["question", "answer"],
        self.score_prompt = PromptTemplate(
        template="""
        You are a technical interviewer.
        Question: {question}
        Candidate's Answer: {answer}
        Evaluate this answer based on:
        Relevance (1–5),Clarity (1–5),Technical Accuracy (1–5)

        Provide your response as a JSON:
            {{
            "Relevance": ...,
            "Clarity": ...,
            "Accuracy": ...,
            }}
        """
        )

    def get_score(Question, Answer):
        chain = self.LLMChain(llm=self.llm_model, prompt=self.score_prompt)
        score_resp = chain.run({"question" : Question , "answer" : Answer})
        return score_resp #convert in dic first




