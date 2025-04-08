
class EasyScoring():
    def __init__(self):
        pass

    def getScore(self, score_from_resume, score_from_interview):
        # for a qusetion we get these scores
        # {
        #   "Accuracy": 1,
        #   "Relevance": 5,
        #   "Clarity": 4,
        # }
        # and we can compute G-index = accuracy * relevance + clarity
        # or we can compte total_score = weighted_avg([accuracy, clarity, relevance], weights=[0.5, 0.25, 0.25])
        # total score is what used in further computations

        # let assume we got techinal, aptitude, concept calrity question so we calculate the score for them

        # --------------from inetrview score-------

        technical_score = "Avg. score from technical answers"
        aptitude_score = "Avg. score from aptitude answers"
        clarity_score = " Avg. score from concept clarity answers"

        # --------------from resume score

        technical_match = "Resume-job match score (0 to 1 or 0–5 scaled to 0–1)"
        clarity_match = " How confidently and clearly the candidate answers (could be model-rated, 0–1)"

        # -----------from resume you can also get
        experience = None
        # -----------from techincal round you can also get communcation skills -------s
        # way of speaking
        # grammar
        # fluency of language
        # content-> appropriate word used, vocabulary
        # articulation
        communication_score = None

        # augmented score
        aug_technical_score = technical_score * technical_match
        aug_clarity_score = clarity_score * clarity_match

        mind_fitness_score = weighted_average([
            aug_technical_score,
            aptitude_score,
            aug_clarity_score
        ], weights=[0.4, 0.3, 0.3])

        candidate_score_vector = [mind_fitness_score,
                                  experience, communication_score]

        # -----------standardist the vector values in 0-5 range

        return candidate_score_vector
        # this vector is capable of giving the complete cinforamtion

        # ----------------------- make a radar chart out of this
