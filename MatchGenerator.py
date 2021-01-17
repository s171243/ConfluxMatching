import os
import math
import importlib, inspect
from datetime import datetime
from KeywordExtraction import get_keyword_list
import global_vars

import DataStore
from Hungarian import HungarianAlgorithm
from Scoring import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MatchGenerator:

    def __init__(self, mentors=None, mentees=None, person_dict=None):
        self.ds = DataStore.DataStore()
        self.matching = None
        self.matching_list = None

        if mentors is None and mentees is None and person_dict is None:
            self.mentors, self.mentees, self.person_dict = self.ds.load_data("initial_data_{}".format(global_vars.ROUND))
        else:
            self.mentors = mentors
            self.mentees = mentees
            self.person_dict = person_dict

        self.assignment_matrix = None
        self.now = datetime.now()  # current date and time
        self.now_str = self.now.strftime("%Y%m%d_%H_%M_%S")

    def add_dummy_rows(self):
        if len(self.mentors) != len(self.mentees):
            diff = len(self.mentees) - len(self.mentors)
            for _ in range(diff):
                dummy_mentor = [0 for _ in self.assignment_matrix[0]]
                self.assignment_matrix.append(dummy_mentor)

    def set_rematch(self, rematch):
        self.ds.rematch = rematch

    def format_matching_result(self, mentors, mentees):
        matching_list = []
        no_mentors = len(mentors)
        no_mentees = len(mentees)

        for i in range(no_mentors):
            for j in range(no_mentees):
                if self.matching[i][j][1] == 2:
                    if self.matching[i][j][0] == 0:
                        print("{} {}".format(mentors[i].name, mentees[j].name))
                        continue

                    matching_list.append((mentors[i], mentees[j], self.matching[i][j][0]))

        return matching_list

    def get_str_id(self):
        return self.ds.get_now()

    def run_hungarian(self):
        hungarian = HungarianAlgorithm(self.assignment_matrix, self.mentors, self.mentees)
        self.matching = hungarian.run()
        self.matching_list = self.format_matching_result(self.mentors, self.mentees)
        return self.ds.save_matches(self.matching_list)
        '''
        total_sum = sum + sum2
        total_count = count + count2
        print("Total average:  {}".format(float(total_sum/total_count)))
        print("Average of former: {}".format(float(sum/count)))
        '''

    #def analyze_hungarian_result(self):
        # print("########## The former match scores ##########")
        real_match_scores = self.get_real_match_scores()
        hungarian_match_scores = self.get_hungarian_match_scores()

        # self.show_boxplots(hungarian_match_scores, real_match_scores)
        # ax1.boxplot(real_match_scores)
        # ax1.boxplot(hungarian_match_scores)
        # plt.show()

    def get_hungarian_match_scores(self):
        hungarian_match_scores = []
        for mentor, mentee, score in self.matching_list:
            hungarian_match_scores.append(score)
        return hungarian_match_scores

    def count_matches(self, mentors, mentees):
        count = 0
        no_mentors = len(mentors)
        no_mentees = len(mentees)

        for i in range(no_mentors):
            for j in range(no_mentees):
                if self.matching[i][j][1] == 2:
                    count += 1

        return count

    def count_mentees_matched_original(self):
        count = 0
        for j, mentee in enumerate(self.mentees):
            count += 1 if mentee.former_match is not None and not math.isnan(mentee.former_match) else 0

        return count

    def count_mentees_matched_again(self):
        count = 0
        for i, mentor in enumerate(self.mentors):
            for j, mentee in enumerate(self.mentees):
                if self.matching[i][j][1] == 2:
                    count += 1 if mentee.former_match is not None and not math.isnan(mentee.former_match) else 0

        return count

    def get_real_match_scores(self):
        count = 0
        sum = 0
        real_match_scores = []
        for i, mentor in enumerate(self.mentors):
            for j, mentee in enumerate(self.mentees):
                if mentor.former_match == mentee.id:
                    count += 1
                    sum += self.matching[i][j][0]
                    real_match_scores.append(self.matching[i][j][0])
                    # print("Match: {} and {} with score: {}".format(mentor.id, mentee.id, self.score_matrix[i][j]))
        return real_match_scores

    def show_boxplots(self, hungarian_match_scores, real_match_scores):
        min_value = min(min(real_match_scores), min(hungarian_match_scores))
        max_value = max(max(real_match_scores), max(hungarian_match_scores))
        tick_length = 50
        fig1, ax = plt.subplots(nrows=2)
        rounded_min = round(min_value / tick_length) * tick_length
        rounded_max = round(max_value / tick_length) * tick_length
        ax[0].boxplot(real_match_scores, vert=False, labels=None, autorange=True, widths=[0.5])
        ax[1].boxplot(hungarian_match_scores, vert=False, labels=None, autorange=True, widths=[0.5])
        plt.sca(ax[0])
        plt.xticks(np.arange(rounded_min, rounded_max, tick_length))
        plt.yticks([])
        plt.sca(ax[1])
        plt.xticks(np.arange(rounded_min, rounded_max, tick_length))
        plt.yticks([])
        ax[0].set_title('Manual Matching Scores')
        ax[1].set_title('Algorithmic Matching Scores')

    def calculate_percentage_rematched(self):
        number_old_matches = 0
        number_non_old_matches = 0

        for mentor, mentee, score in self.matching_list:
            if mentee.id in self.person_dict and mentor.id in self.person_dict:
                if mentee.id == self.person_dict[mentor.id].former_match:
                    number_old_matches += 1
                else:
                    number_non_old_matches += 1

        # print("RESULT:\nWe had {} matches that were the same\nWe had {} new matches".format(number_old_matches,
        #                 s                                                                   number_non_old_matches))
        # print("{:d} matches were the same That's a correct-match percentage of {:.3g}%".format(number_old_matches,
        #                                                                                        number_old_matches / (
        #                                                                                                number_non_old_matches + number_old_matches) * 100))

        return number_old_matches

    def calculate_score_object(self, score_class):
        self.class_for_score = score_class(mentors=self.mentors,
                                           mentees=self.mentees,
                                           person_dict=self.person_dict)

        self.assignment_matrix = self.class_for_score.calculate_score()
        self.add_dummy_rows()

    def save_matching_configuration(self):
        arg_dict = self.class_for_score.arg_dict
        arg_dict['id'] = self.now_str
        output_path = "data/matching_configuration_2.csv"
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            df2 = pd.DataFrame([arg_dict])
            df3 = df.append(df2)
            df3.to_csv(output_path, index=False, mode='w')
        else:
            df = pd.DataFrame([arg_dict])
            df.to_csv(output_path, index=False, mode='w')

    def save_hungarian_result(self):
        hungarian_scores = self.get_hungarian_match_scores()
        real_scores = self.get_real_match_scores()
        columns = ['id', 'old_q1', 'old_q2', 'old_q3', 'old_min', 'old_max',
                   'new_q1', 'new_q2', 'new_q3', 'new_min', 'new_max', 'new_repeat']
        data = {
            'id': self.now_str,
            'old_q1': int(np.quantile(real_scores, 0.25)),
            'old_q2': int(np.quantile(real_scores, 0.50)),
            'old_q3': int(np.quantile(real_scores, 0.75)),
            'old_min': int(min(real_scores)),
            'old_max': int(max(real_scores)),
            'new_q1': int(np.quantile(hungarian_scores, 0.25)),
            'new_q2': int(np.quantile(hungarian_scores, 0.50)),
            'new_q3': int(np.quantile(hungarian_scores, 0.75)),
            'new_min': int(min(hungarian_scores)),
            'new_max': int(max(hungarian_scores)),
            'new_repeat': self.calculate_percentage_rematched(),
            'matched_again': self.count_mentees_matched_again(),
            'real_avg': sum(real_scores)/len(real_scores),
            'new_avg': sum(hungarian_scores)/len(hungarian_scores),
            'avg_all': sum(hungarian_scores)
        }

        print(
            "Number of students that were matched with this algorithm that were also matched (not necessarily with the same person): {} af {}".format(
                self.count_mentees_matched_again(), self.count_mentees_matched_original()))
        output_path = "data/matching_results.csv"

        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            df2 = pd.DataFrame([data])
            df3 = df.append(df2)
            df3.to_csv(output_path, index=False, mode='w')
        else:
            df = pd.DataFrame([data])
            df.to_csv(output_path, index=False, mode='w')

        # df.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))


def get_score_function_from_user():
    global score_function
    print("Welcome to the matching program! Which matching algorithm do you want to use?")
    class_names = inspect.getmembers(importlib.import_module("Scoring"), inspect.isclass)
    # class_names.sort(key=lambda name, _: int(re.sub('\D', '', name)))

    first_number = -1
    for i, (name, cls) in enumerate(class_names):
        if "ScoreFunction" in name:
            if first_number == -1:
                first_number = i
            print("({}): {}".format(i, name))
    print("\nWrite the number to choose an algorithm. Write anything else to give up.\n\n")
    result = input("")
    if int(result) in range(first_number, len(class_names)):
        print("\nThe score function {} has been chosen".format(class_names[int(result)][0]))
        return class_names[int(result)][1]

    return None


if __name__ == "__main__":
    TOGGLE_CMDLINE = False
    match_gen = MatchGenerator()
    print("Started MatchGenerator")
    numbers = [0, 5, 10, 15, 25, 50, 70]
    highest_number = 0

    # match_gen.calculate_score_evolution_2(score_evaluation_5_evolution, [5, 10, 10, 10, 50, 50])

    # match_gen.calculate_score_2(score_evaluation_5)

    if TOGGLE_CMDLINE:
        score_function = get_score_function_from_user()
        if score_function is None:
            score_function = ScoreFunction5
    else:
        score_function = ScoreFunction5

    match_gen.calculate_score_object(score_function)
    print("Calculated scores")
    match_gen.run_hungarian()
    print("Hungarian done")
    number_old_matches = match_gen.calculate_percentage_rematched()
    print("Percentage calculated")
    match_gen.analyze_hungarian_result()
    match_gen.save_hungarian_result()
    match_gen.save_matching_configuration()
    # print("Score: {}. That's: {:. 2f}".format(number_old_matches, ))"""
