# AI driven interview

## Summary

This system integrates **resume analysis** and **adaptive AI-driven interviews** using Large Language Models (LLMs) to evaluate candidate fitness for technical roles.

###  Tech Stack
- **Language**: Python  
- **LLM Platform**: OpenAI via LangChain  
- **Vector Store**: FAISS  
- **Embeddings**: OpenAI Embeddings  
- **Visualization**: Matplotlib, NumPy  

---

### Components

1. **ResumeLoader**  
   - Loads and parses resumes from PDFs  
   - Chunks and embeds content  
   - Matches resume against job description using Retrieval QA

2. **InterviewSystem**  
   - Contextual, adaptive interview generator  
   - Adjusts question difficulty dynamically  
   - Uses knowledge base to generate relevant questions  

3. **QuestionAnswerEval**  
   - Evaluates answers on:
     - **Technical Score**: Relevance + Accuracy
     - **Communication Score**: Grammar, Vocabulary, Clarity, etc.  
   - Provides visual analytics  

4. **EasyScoring**  
   - Combines resume and interview scores  
   - Returns a vector representing:
     - Technical Fit
     - Experience Relevance
     - Communication Skill  

---

### GenAI Strategy

- **LLM Customization**: OpenAI GPT with controlled temperature (0.5)  
- **Prompt Engineering**: Dynamic, context-aware prompts for both questioning and evaluation  
- **Evaluation**: JSON-based structured outputs scored using custom weighted formulas  
- **Multimodal Vision** (Future Scope): Extend text-based evaluation to include audio (via speech-to-text) and video (expression analysis)




## figure showing overall system components and data flow
![image](https://github.com/green-gray-gaurav/AI-driven-interview/blob/AI/OVERVIEW_ARCH.drawio.png)


## scoring schema suggested
> * can use G-index or weighted sum
> * G-index is my original, computes the score of answer by --accuracy * relevance + clarity




## figure showing the structure of QuestionEval, InterviewSystem ,ResumeLoader , EasyScoring system
![image](https://github.com/green-gray-gaurav/AI-driven-interview/blob/AI/ARCHI_ES_IS_QE.drawio.png)


## figure showing the radar chart of individual performance
![image](https://github.com/green-gray-gaurav/AI-driven-interview/blob/master/Screenshot%202025-04-09%20000330.png)

## bar chart to get detail insights on answers given by candidate around specific concepts
![image](https://github.com/green-gray-gaurav/AI-driven-interview/blob/AI/Screenshot%202025-04-09%20154311.png)
