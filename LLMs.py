class ResumeLoader():
    def __init__(self, resume_name, llm):

        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.vectorstores import FAISS
        from langchain.embeddings import OpenAIEmbeddings

        self.llm = llm
        self.my_pdf_loader = PyPDFLoader(resume_name)
        self.pdf_pages = my_pdf_loader.load()
        self.resume_text_string = " ".join(
            [page.page_content for page in pdf_pages])

        self.chunker = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.resume_in_chumks = chunker.split_text(resume_text_string)

        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_texts(
            resume_in_chumks, embedding=embeddings)

    def score_resume_using_context(job_description):

        from langchain.chains import RetrievalQA
        from langchain.chat_models import ChatOpenAI

        RAG_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff"
        )

        query = f"""
        Given the following job description:

        {job_description}

        Evaluate candidate’s resume based on: Relevant experience (0–10), Technical skill match (1–10)

        Provide a JSON output.
        """
        response = RAG_chain.run(query)
        return response  # parse this JSON into dictionary


class InterviewSystem():
    def __init__(self, doc, evaluator, llm):
        from langchain.vectorstores import FAISS
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains import RetrievalQA
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import LLMChain
        import os

        self.llm
        self.evaluator
        loader = TextLoader(doc)
        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Embed & store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        self.retriever = vectorstore.as_retriever()

        self.question_prompt = PromptTemplate(
            input_variables=["context", "difficulty", "concept"],
            template="""
            You're an AI Interviewer. Generate a {difficulty}-level conceptual question based on the topic "{concept}" using the following context:
            {context}

            Question:"""
        )
        self.qa_chain = LLMChain(llm=self.llm, prompt=self.question_prompt)

        pass

    def start(self, callback):

        # Interview state
        Q_COUNT = 0
        T = 10
        N = 3

        incorrect_easy = 0
        performance = []
        level = "easy"
        concept = "start"

        while Q_COUNT < T:
            if incorrect_easy >= N:
                concept = input(
                    "Please mention the concept you're confident about: ")
                level = "easy"
                incorrect_easy = 0  # reset
            else:
                # Optional: auto-select a concept from context or previous answer
                concept = "basic topic" if concept == "start" else concept

            # Fetch context
            docs = self.retriever.get_relevant_documents(concept)
            context = docs[0].page_content if docs else ""

            # Generate question
            question = self.qa_chain.run(
                context=context, difficulty=level, concept=concept)
            print(f"\nQ{Q_COUNT + 1}: {question}")
            student_answer = callback(question)

            # Evaluate (simple eval for now)
            print("Evaluating...")
            # Simulate LLM check or just accept any answer as correct/incorrect
            question_attr = (concept, level)
            t_score = self.evaluator.get_score(
                question_attr, question, student_answer)

            correct = t_score > 6  # or one can make some function out of it

            if correct:
                if level == "easy":
                    level = "moderate"
                incorrect_easy = 0
            else:
                if level == "easy":
                    incorrect_easy += 1
                else:
                    level = "easy"

            Q_COUNT += 1


class QuestionAnswerEval():
    def __init__(self, llm):
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        scoring_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="""
            Evaluate the following answer for both technical relevance and accuracy. Give each a score between 0 and 10.

            Question: {question}
            Answer: {answer}

            Return response in json :
            Relevance: <score>
            Accuracy: <score>
            """
        )

        comm_prompt = PromptTemplate(
            input_variables=["answer"],
            template="""
            Evaluate the communication quality of the following answer. Rate each on a scale from 0 to 10:

            Answer: {answer}

            Return response in json:
            Grammar: <score>
            Vocabulary: <score>
            Appropriate Words: <score>
            Articulation: <score>
            Clarity: <score>
            """
        )

        self.score_chain = LLMChain(llm=llm, prompt=scoring_prompt)
        self.comm_chain = LLMChain(llm=llm, prompt=comm_prompt)
        self.tech_scores = []
        self.comm_scores = []
        self.question_attributes = []

        self.question_answers_log = []

    def get_score(self, question_attr, question, answer):
        # attrs
        self.question_attributes.append(question_attr)
        self.question_answers_log.append((question, answer))

        # TECHNICAL SCORING

        tech_eval = self.score_chain.run(question=question, answer=answer)
        relevance = float(tech_eval.split("Relevance:")
                          [1].split("\n")[0].strip())
        accuracy = float(tech_eval.split("Accuracy:")[1].strip())
        tech_score = 0.6 * relevance + 0.4 * accuracy
        self.tech_scores.append(tech_score)

        # COMMUNICATION SCORING
        comm_eval = self.comm_chain.run(answer=answer)
        g = float(comm_eval.split("Grammar:")[1].split("\n")[0])
        v = float(comm_eval.split("Vocabulary:")[1].split("\n")[0])
        a = float(comm_eval.split("Appropriate Words:")[1].split("\n")[0])
        ar = float(comm_eval.split("Articulation:")[1].split("\n")[0])
        c = float(comm_eval.split("Clarity:")[1].split("\n")[0])
        comm_score = 0.2 * (g + v + a + ar + c)
        self.comm_scores.append(comm_score)

        return tech_score, comm_score

    def get_cummulative_score(self):
        technical_score = sum(self.tech_scores) / len(self.tech_scores)
        communication_score = sum(self.comm_scores) / len(self.comm_scores)

        return technical_score, communication_score

    def get_analytics_plots(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import defaultdict

        # Group scores by concept and level
        # concept -> level -> [scores]
        data = defaultdict(lambda: defaultdict(list))

        for i, attr in enumerate(self.question_attributes):
            concept = attr['concept']
            level = attr['level']
            tech = self.tech_scores[i]
            comm = self.comm_scores[i]
            data[concept][level].append((tech, comm))

        # Plot each concept
        for concept, level_scores in data.items():
            levels = ['easy', 'medium', 'hard']
            tech_avgs = []
            comm_avgs = []

            for level in levels:
                scores = level_scores.get(level, [])
                if scores:
                    tech_avg = sum([t for t, _ in scores]) / len(scores)
                    comm_avg = sum([c for _, c in scores]) / len(scores)
                else:
                    tech_avg = comm_avg = 0
                tech_avgs.append(tech_avg)
                comm_avgs.append(comm_avg)

            x = np.arange(len(levels))
            width = 0.35

            fig, ax = plt.subplots()
            bars1 = ax.bar(x - width/2, tech_avgs, width,
                           label='Technical Score', color='skyblue')
            bars2 = ax.bar(x + width/2, comm_avgs, width,
                           label='Communication Score', color='salmon')

            ax.set_ylabel('Score')
            ax.set_title(f'Scores by Level for Concept: {concept}')
            ax.set_xticks(x)
            ax.set_xticklabels(levels)
            ax.set_ylim(0, 10)
            ax.legend()

            for bar in bars1 + bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')

            plt.tight_layout()
            plt.show()
