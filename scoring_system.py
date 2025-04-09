
class EasyScoring():
    def __init__(self):
        pass

    def getScore(self, score_from_resume, interview_score):
        # for a qusetion we get these scores
        # {
        #   "Accuracy": 1,
        #   "Relevance": 5,
        #   "Clarity": 4,
        # }
        # and we can compute G-index = accuracy * relevance + clarity
        # or we can compte total_score = weighted_avg([accuracy, clarity, relevance], weights=[0.5, 0.25, 0.25])
        # total score is what used in further computations

        # let assume we got techinal 
        # --------------from inetrview score-------

        tech_score , com_score = interview_score

        # --------------from resume score

        technical_match = "Resume-job match score" #make sure to score in 0 - 10
        
        #exp 10 could mean 10+ also
        experience , technical_match  = score_from_resume

    

        # augmented score
        aug_technical_score = tech_score * technical_match
        

        
        candidate_score_vector = [aug_technical_score,
                                  experience, com_score]

        # -----------standardist the vector values in 0-5 range

        return candidate_score_vector
        # this vector is capable of giving the complete cinforamtion

        # ----------------------- make a radar chart out of this
