from ScoreHelpers import *
import time

def score_evaluation_5_evolution(matrix, mentors, mentees, person_dict, list):
    reward_similar_text(matrix, mentors, mentees)
    # reward_similar_text(matrix, mentors, mentees, list[0])
    matrix = reward_same_industry(matrix, mentors, mentees, list[0])
    matrix = reward_priorities(matrix, mentors, mentees, list[1])
    matrix = reward_high_lix(matrix, mentors, mentees, list[2])
    matrix = reward_clustered_mentors(matrix, mentors, mentees, person_dict, list[3])
    matrix = punish_short_mentee_profile(matrix, mentors, mentees, list[4])
    matrix = punish_lack_of_linkedin(matrix, mentors, mentees, list[5])

    return matrix

def pairwise(iterable):
    '''

    :param iterable:
    :return:
    '''
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


class IScoring:
    '''
    Interface for creating scoring functions.
    '''

    def __init__(self, mentors, mentees, person_dict=None):
        self.mentors = mentors
        self.mentees = mentees
        self.assignment_matrix = [[len(self.mentors)*2] * len(self.mentees) for i in range(len(self.mentors))]
        self.person_dict = person_dict
        self.arg_dict = {}

    def calculate_score(self):
        pass

    def _calculate_score(self, *args):
        '''
        Receives a number of functions and calculates the final score.
        :param args: function, score. E.g. reward_xxx, 200, reward_yyy, 100 etc

        :return:
        '''
        self.set_args(args=args)
        print("function_name;time_from_start")
        start_time = time.time()
        for function, score in pairwise(args):
            # If we don't want to set the score, ignore it and default to method value
            if score is None:
                self.assignment_matrix = function(matrix=self.assignment_matrix,
                                                  mentors=self.mentors,
                                                  mentees=self.mentees,
                                                  person_dict=self.person_dict)
            else:
                self.assignment_matrix = function(matrix=self.assignment_matrix,
                                                  mentors=self.mentors,
                                                  mentees=self.mentees,
                                                  score=score,
                                                  person_dict=self.person_dict)

            print("{};{}".format(function.__name__, time.time() - start_time))
    def set_args(self, args):
        arg_dict = {}
        for function, score in pairwise(args):
            arg_dict[function.__name__] = score

        self.arg_dict = arg_dict

'''Scorefunction 1: 46%
(243) matching_result_20201220_21_26_22.csv
rating_20201220_21_26_49.csv
'''
class ScoreFunction5(IScoring):
    '''
    Only takes priorities into account
    '''

    def calculate_score(self):
        self._calculate_score(
            reward_priorities, 150
        )

        return self.assignment_matrix

'''
Scorefunction 2: 
(244) matching_result_20201220_22_05_24.csv
'''
class ScoreFunction6(IScoring):
    '''
    Take industry and priorities 
    into account
    '''

    def calculate_score(self):
        self._calculate_score(
            reward_priorities, 150,
            reward_same_industry, 50
        )

        return self.assignment_matrix


class ScoreFunction7(IScoring):
    '''
    Take industry, priorities, 
    short mentee profiles, lack of linkedin profile, lix 
    into account
    '''

    def calculate_score(self):
        self._calculate_score(
            reward_priorities, 150,
            reward_same_industry, 50,
            punish_short_mentee_profile, 100,
        )

        return self.assignment_matrix


class ScoreFunction8(IScoring):
    def calculate_score(self):
        self._calculate_score(
            reward_priorities, 150,
            reward_same_industry, 50,
            punish_short_mentee_profile, 100,
            punish_lack_of_linkedin, 30,
            reward_female_gender, 30,
            reward_similar_text, 20,
            reward_clustered_mentors, 20
        )

        return self.assignment_matrix


# reward_generalist_or_specialist
class ScoreFunction9(IScoring):
    
    def calculate_score(self):
        self._calculate_score(
            reward_same_industry, 50,
            reward_priorities, 100,
            punish_short_mentee_profile, 50,
            reward_potential_priorities_mentee, 50
        )

        return self.assignment_matrix


class ScoreFunction10(IScoring):
    
    ''' They is an upper limitation to what priorities can do. '''
    def calculate_score(self):
        self._calculate_score(
            reward_same_industry, 50,
            reward_priorities, 100,
            punish_short_mentee_profile, 50,
            punish_lack_of_linkedin, 20,
            reward_potential_priorities_mentee, 50,
            reward_potential_priorities_mentor, 40,
            punish_bachelor_mentees, 20,
            reward_female_gender, 20,
        )

        return self.assignment_matrix


# 63 %
class ScoreFunction11(IScoring):
    
    ''' They is an upper limitation to what priorities can do. '''
    def calculate_score(self):
        self._calculate_score(
            reward_same_industry, 50,
            reward_priorities, 150,
            punish_short_mentee_profile, 50,
            punish_lack_of_linkedin, 20,
            reward_potential_priorities_mentee, 50,
            reward_potential_priorities_mentor, 40,
            reward_potential_priorities_mentee_factor_refactored, 20,
            reward_potential_priorities_mentor_factor_refactored, 20,
            punish_bachelor_mentees, 20,
            reward_female_gender, 10,
            punish_missing_linkedin_picture, 10
        )

        return self.assignment_matrix


class ScoreFunction12(IScoring):
    
    ''' They is an upper limitation to what priorities can do. '''
    def calculate_score(self):
        self._calculate_score(
            reward_same_industry, 50,
            reward_priorities, 150,
            punish_short_mentee_profile, 50,
            punish_lack_of_linkedin, 20,
            reward_potential_priorities_mentee, 50,
            reward_potential_priorities_mentor, 40,
            reward_potential_priorities_mentee_factor_refactored, 20,
            reward_potential_priorities_mentor_factor_refactored, 20,
            punish_bachelor_mentees, 20,
            reward_female_gender, 10,
            punish_missing_linkedin_picture, 10,
            reward_similarity_keywords_to_texts_new, 20
        )

        return self.assignment_matrix

''' We still need to utilize: 
    - Reward potential priorities mentors
    - Reward potential mentees and mentors refactored
    - Reward potential mentees and mentors factor refactored 
    - LSH
    - GloVe (similarity keywords to text) '''


# 80% - top so far
class ScoreFunction13(IScoring):
    def calculate_score(self):
        self._calculate_score(
            #reward_similar_text, 0,
            reward_same_industry, 50,
            reward_priorities, 110,
            #reward_clustered_mentors, 10,
            punish_short_mentee_profile, 50,
            punish_lack_of_linkedin, 20,
            reward_potential_priorities_mentee, 60,
            reward_potential_priorities_mentor, 50,
            punish_bachelor_mentees, 20,
            reward_female_gender, 20,
            punish_missing_linkedin_picture,20,
            reward_similarity_keywords_to_texts, 20
        )

        return self.assignment_matrix


class ScoreFunction14(IScoring):
    def calculate_score(self):
        self._calculate_score(
            reward_same_industry, 50,
            reward_priorities, 130,
            punish_short_mentee_profile, 50,
            punish_lack_of_linkedin, 20,
            reward_potential_priorities_mentee, 60,
            reward_potential_priorities_mentor, 50,
            punish_bachelor_mentees, 20,
            reward_female_gender, 20,
            punish_missing_linkedin_picture,20,
            reward_similarity_keywords_to_texts_new, 30
        )

        return self.assignment_matrix


class ScoreFunction15(IScoring):
    def calculate_score(self):
        self._calculate_score(
            reward_same_industry, 50,
            reward_priorities, 130,
            punish_short_mentee_profile, 50,
            punish_lack_of_linkedin, 20,
            reward_potential_priorities_mentee_lsh, 60,
            reward_potential_priorities_mentor_lsh, 50,
            punish_bachelor_mentees, 20,
            reward_female_gender, 20,
            punish_missing_linkedin_picture, 20,
            reward_similarity_keywords_to_texts_new, 30
        )

        return self.assignment_matrix


class ScoreFunction30(IScoring):
    def calculate_score(self):
        self._calculate_score(
            reward_potential_priorities_mentee_lsh, 10,
            reward_potential_priorities_mentor_lsh, 10
        )

        return self.assignment_matrix

class ScoreFunction2(IScoring):
    def calculate_score(self):
        self._calculate_score(
            generate_random_score,0
        )

        return self.assignment_matrix

class ScoreFunction20(IScoring):
    def calculate_score(self):
        self._calculate_score(
            reward_potential_priorities_mentee_lsh, 10
        )

        return self.assignment_matrix


class ScoreFunction21(IScoring):
    def calculate_score(self):
        self._calculate_score(
            reward_potential_priorities_mentee_lsh, 10
        )

        return self.assignment_matrix

class ScoreFunctionTime(IScoring):
    def calculate_score(self):
        self._calculate_score(
            reward_priorities,0,
            reward_same_industry,0,
            punish_short_mentee_profile,0,
            punish_lack_of_linkedin,0,
            reward_female_gender,0,
            reward_potential_priorities_mentee,0,
            reward_potential_priorities_mentor,0,
            punish_bachelor_mentees, 0,
            punish_missing_linkedin_picture,0,
            reward_similarity_keywords_to_texts_new,0,
            reward_similar_text,0,
            reward_clustered_mentors,0,
        )

        return self.assignment_matrix

class ScoreFunction0(IScoring):
    def calculate_score(self):
        
        self._calculate_score(
            reward_same_industry, 150,
            reward_similarity_keywords_to_texts_new, 10,
        )

        return self.assignment_matrix
