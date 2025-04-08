
# set up the ui
# get all the question in the question array

from utils import Questioning_system
from LLMs import ResumeLoader
from LLMs import QuestionEval
from scoring_system import EasyScoring


questionlist = []
QS = Questioning_system()
QS.getQuestionFromList(questionlist)

LLM_EVAL = QuestionEval()

job_des = ""

resume_name = None  # input()

RL = ResumeLoader(None)
resume_score_in_dic = RL.score_resume_using_context(job_des)


while QS.hasQuestsions:
    question = QS.askQuestion()

    answer = ""

    QS.evaluateAnswer(answer, LLM_EVAL)


interview_score = QS.getQuestionsScore()


ES = EasyScoring()


performance_vector = EasyScoring.getScore(
    score_from_resume=resume_score_in_dic, score_from_interview=interview_score)


print(performance_vector)
