from langchain.chains import LLMChain
from langchain.llms import OpenAI
from scoring_system import EasyScoring


from LLMs import ResumeLoader, InterviewSystem, QuestionAnswerEval
llm = OpenAI(temperature=0.5)

doc_path  =  "doc.txt"
job_des = "this job is for AI guys"


evaluator = QuestionAnswerEval(llm)
IS = InterviewSystem(doc_path , evaluator , llm)

candiadate_resume = "resume.pdf"
RL = ResumeLoader(candiadate_resume , llm)

resume_score = RL.score_resume_using_context(job_des)

def get_answer(question):

    #get the answer from the student through the UI inetrface
    answer = ""
    return answer
    pass


#this will start the interview qustioning
IS.start(get_answer)

interview_score = evaluator.get_cummulative_score()

performance_vector = EasyScoring().getScore(resume_score , interview_score)




