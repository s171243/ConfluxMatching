import pandas as pd
import csv
import copy
import sys
import Scoring as score
import MatchRater5000 as gui_test
from MatchRater5000 import show_matches
from DataStore import DataStore
from global_vars import ROUND

class HungarianAlgorithm:
    def __init__(self, assignment_matrix, mentors, mentees):
        self.score_matrix = assignment_matrix # the artist formerly known as assignment matrix
        self.equality_matrix = None
        self.matrix_size = None
        self.matching = None
        self.matching_size = None
        self.mentors = mentors
        self.mentees = mentees
        self.no_mentors = None
        pass

    def create_assignment_matrix(self):
        mentors, mentees, person_dict = ds.load_data("initial_data_{}".format(ROUND))

        # mentors, mentees = loader.load_data("initial_data.csv")
        self.mentors = mentors
        self.mentees = mentees

        self.no_mentors = len(mentors)
        no_mentees = len(mentees)
    
        matrix = [[self.no_mentors]*no_mentees for i in range(self.no_mentors)]

        self.score_matrix = score.score_evaluation_5(matrix, mentors, mentees, person_dict)

        self.no_mentors = len(self.score_matrix)
        no_mentees = len(self.score_matrix[0])

        if self.no_mentors != no_mentees:
            diff = no_mentees - self.no_mentors
            for _ in range(diff):
                dummy_mentor = [0 for _ in self.score_matrix[0]]
                self.score_matrix.append(dummy_mentor)

        return mentors, mentees

    def create_job_assignment_matrix(self):
        import random
        assignment_matrix1 = [[42, 35, 28, 21],
                              [30, 25, 20, 15],
                              [30, 25, 20, 15],
                              [24, 20, 16, 12]]

        assignment_matrix2 = [[2, 3, 6],
                              [2, 4, 4],
                              [1, 3, 2]]

        assignment_matrix3 = [[7, 5, 6, 9],
                              [8, 7, 7, 5],
                              [3, 1, 0, 9],
                              [7, 5, 5, 9]]

        assignment_matrix4 = [[62, 73, 61, 30],
                              [18, 93, 96, 25],
                              [14, 35, 43, 27],
                              [36, 21, 72, 32]]

        assignment_matrix5 = [[85, 26, 67, 79],
                              [66, 34, 65, 12],
                              [25, 67, 39, 51],
                              [51, 32, 55, 45]]

        assignment_matrix6 = [[18, 85, 70, 31],
                              [94, 89, 19, 73],
                              [98, 58, 74, 65],
                              [25, 20, 96, 49]]

        assignment_matrix7 = [[77, 49, 35, 57],
                              [15, 67, 71, 65],
                              [41, 50, 69, 36],
                              [67, 97, 34, 20]]

        assignment_matrix8 = [[79, 87, 78, 27],
                              [61, 71, 40, 45],
                              [16, 18, 40, 38],
                              [70, 75, 31, 25]]

        assignment_matrix9 = [[83, 45, 19, 70],
                              [86, 46, 11, 28],
                              [91, 41, 38, 31],
                              [50, 10, 20, 81]]

        assignment_matrix10 = [[40, 35, 61, 13],
                               [28, 64, 18, 38],
                               [28, 53, 33, 15],
                               [25, 42, 26, 16]]

        assignment_matrix11 = [[25, 88, 79, 76],
                               [49, 24, 21, 76],
                               [80, 33, 88, 16],
                               [31, 73, 49, 80]]

        assignment_matrix12 = [[74, 47, 56, 45],
                               [83, 83, 76, 36],
                               [12, 69, 65, 10],
                               [89, 50, 73, 41]]

        assignment_matrix13 = [[73, 41, 68, 57],
                               [86, 15, 75, 31],
                               [88, 92, 93, 46],
                               [42, 85, 31, 33]]

        assignment_matrix14 = [[73, 41, 68, 57, 17],
                               [86, 15, 75, 31, 2],
                               [88, 92, 93, 46, 39],
                               [42, 85, 31, 33, 0]]

        assignment_matrix_random = [[random.randint(10,99) for i in range(12)] for j in range(7)]

        assignment_matrix = assignment_matrix_random

        no_mentors = len(assignment_matrix)
        no_mentees = len(assignment_matrix[0])

        if no_mentors != no_mentees:
            diff = no_mentees - no_mentors
            for i in range(diff):
                dummy_mentor = [0 for _ in assignment_matrix[0]]
                assignment_matrix.append(dummy_mentor)
        
        return assignment_matrix

    def init_row_aux_vars(self):
        return [max(row) for row in self.score_matrix]

    def init_col_aux_vars(self):
        return [0 for _ in self.score_matrix[0]]

    def create_equality_matrix(self, assignment_matrix, aux_row_vars, aux_col_vars):
        # Initialize empty equality graph
        equality_matrix = [[None for _ in range(self.matrix_size)] for _ in range(self.matrix_size)]

        for idx, row in enumerate(assignment_matrix):
            for idy, score in enumerate(row):
                aux_sum = (aux_row_vars[idx] + aux_col_vars[idy])

                equality_matrix[idx][idy] = (score, 1 if score == aux_sum else 0)

        self.equality_matrix = equality_matrix

    def get_column(self, matrix, i):
        return [row[i] for row in matrix]

    def find_initial_matching(self):
        matching = [[None for _ in range(self.matrix_size)] for _ in range(self.matrix_size)]

        matched_mentees = []
        matched_mentors = []

        # TODO: Maybe refactor the matched_mentees and matched_mentors list to a function: Has_matching()
        for idx, row in enumerate(self.equality_matrix):
            for idy, (score, status) in enumerate(row):
                if status == 1 and (idy not in matched_mentees) and (idx not in matched_mentors):
                    matching[idx][idy] = (score, 2)
                    matched_mentors.append(idx)
                    matched_mentees.append(idy)
                else:
                    matching[idx][idy] = (score, status)

        self.matching = matching

    def find_maximal_matching(self):
        self.find_initial_matching()
        col_of_squares = [None] * self.matrix_size
        row_of_triangles = [None] * self.matrix_size
        self.find_augmenting_path(row_of_triangles=row_of_triangles, col_of_squares=col_of_squares)

    def find_smallest_diff(self, col_of_squares, row_of_triangles, aux_row_vars, aux_col_vars):
        smallest_diff = sys.maxsize

        for mentor_id, square in enumerate(col_of_squares):
            if square != "S":
                continue
            for mentee_id, triangle in enumerate(row_of_triangles):
                if triangle == "T":
                    continue
                diff = (aux_row_vars[mentor_id] + aux_col_vars[mentee_id]) - self.equality_matrix[mentor_id][mentee_id][0]
                smallest_diff = min(smallest_diff, diff)

        if smallest_diff == sys.maxsize:
            smallest_diff = 0

        return smallest_diff

    def update_aux_numbers(self, col_of_squares, row_of_triangles, aux_row_vars, aux_col_vars):
        smallest_diff = self.find_smallest_diff(col_of_squares, row_of_triangles, aux_row_vars,
                                                aux_col_vars)

        for mentor_id, square in enumerate(col_of_squares):
            if square == "S":
                aux_row_vars[mentor_id] -= smallest_diff
        for mentee_id, triangle in enumerate(row_of_triangles):
            if triangle == "T":
                aux_col_vars[mentee_id] += smallest_diff

        return aux_row_vars, aux_col_vars

    def find_initial_squares(self, col_of_squares):
        path = []
        for idx, row in enumerate(self.matching):
            if not self.has_matching(row):
                col_of_squares[idx] = "S"
                # We add:
                # 1) The index,
                # 2) The type
                # "predecessor" to the path, in this no predecessor, presented as case -1,
                # as this is the first to be marked
                path.append((idx, "Mentor", -1))

        return col_of_squares, path

    def has_matching(self, entry):
        for _, (_, status) in enumerate(entry):
            if status == 2:
                return True
        return False

    def mark_squares(self, col_of_squares, row_of_triangles, path):
        # Initialize unmatched mentor vertices

        for mentee_id, triangle in enumerate(row_of_triangles):
            if triangle != "T":
                continue
            for mentor_id, square in enumerate(col_of_squares):
                if self.matching[mentor_id][mentee_id][1] == 2 and square != "S":
                    col_of_squares[mentor_id] = "S"
                    path.append((mentor_id, "Mentor", mentee_id))

        return col_of_squares, row_of_triangles, path

    def mark_triangles(self, col_of_squares, row_of_triangles, path, terminated):
        # TODO: Figure out if we should split this into two different functions (terminated = False or = None)
        for mentor_id, square in enumerate(col_of_squares):
            if square != "S":
                continue
            for mentee_id, triangle in enumerate(row_of_triangles):
                if self.matching[mentor_id][mentee_id][1] != 0 and triangle != "T":
                    row_of_triangles[mentee_id] = "T"
                    path.append((mentee_id, "Mentee", mentor_id))
                    mentee_column = self.get_column(self.matching, mentee_id)

                    if not (self.has_matching(mentee_column)) and terminated != None:
                        terminated = True
                        return row_of_triangles, path, terminated
        return row_of_triangles, path, terminated

    def extend_matching(self, path):
        path.reverse()
        new_matches = []
        old_matches = []

        curr_index = 0

        while path[curr_index][2] != -1:
            for idx, elem in enumerate(path, 1):
                if path[curr_index][2] == path[curr_index + idx][0] and path[curr_index][1] != path[curr_index + idx][
                    1]:
                    mentor_idx = path[curr_index + (0 if path[curr_index][1] == "Mentor" else idx)][0]
                    mentee_idx = path[curr_index + (0 if path[curr_index][1] == "Mentee" else idx)][0]

                    curr_index += idx

                    if self.matching[mentor_idx][mentee_idx][1] == 1:
                        new_matches.append((mentor_idx, mentee_idx))
                        break

                    elif self.matching[mentor_idx][mentee_idx][1] == 2:
                        old_matches.append((mentor_idx, mentee_idx))
                        break

        for mentor_idx, mentee_idx in old_matches:
            weight = self.matching[mentor_idx][mentee_idx][0]
            self.matching[mentor_idx][mentee_idx] = (weight, 1)
        for mentor_idx, mentee_idx in new_matches:
            weight = self.matching[mentor_idx][mentee_idx][0]
            self.matching[mentor_idx][mentee_idx] = (weight, 2)


    def find_augmenting_path(self, row_of_triangles, col_of_squares):
        col_of_squares_copy = [None] * self.matrix_size
        row_of_triangles_copy = [None] * self.matrix_size

        col_of_squares, path = self.find_initial_squares(col_of_squares=col_of_squares)

        while not (col_of_squares == col_of_squares_copy and row_of_triangles == row_of_triangles_copy):
            col_of_squares_copy = col_of_squares[:]
            row_of_triangles_copy = row_of_triangles[:]

            col_of_squares, row_of_triangles, path = self.mark_squares(col_of_squares, row_of_triangles, path)
            row_of_triangles, path, terminated = self.mark_triangles(col_of_squares, row_of_triangles, path, terminated=False)

            if terminated:
                self.extend_matching(path)
                col_of_squares = [None] * self.matrix_size
                col_of_squares, path = self.find_initial_squares(col_of_squares)
                row_of_triangles = [None] * self.matrix_size
                terminated = False

    def find_size_of_matching(self):
        matching_size = 0
        total_score = 0

        for mentor_id, row in enumerate(self.matching):
            if sum(status == 2 for _, status in row) > 1:
                raise Exception('\033[91m' + "Error. Mentor with more than one match.")
            for mentee_id, (score, status) in enumerate(row):
                if status == 2:
                    total_score += score
                    matching_size += 1

        for mentee_id, _ in enumerate(self.matching[0]):
            mentee_col = self.get_column(self.matching, mentee_id)
            if sum(status == 2 for _, status in mentee_col) > 1:
                raise Exception('\033[91m' + "Error. Mentee with more than one match.")

        self.matching_size = matching_size

    def print_matrix(self, matrix):
        [print(row) for row in matrix]

    def perform_marking_procedure(self):
        col_of_squares = [None] * self.matrix_size
        row_of_triangles = [None] * self.matrix_size
        col_of_squares_copy = [None] * self.matrix_size
        row_of_triangles_copy = [None] * self.matrix_size
        col_of_squares, path = self.find_initial_squares(col_of_squares)

        while not (col_of_squares == col_of_squares_copy and row_of_triangles == row_of_triangles_copy):
            col_of_squares_copy = col_of_squares[:]
            row_of_triangles_copy = row_of_triangles[:]

            col_of_squares, row_of_triangles, path = self.mark_squares(col_of_squares, row_of_triangles, path)
            row_of_triangles, path, terminated = self.mark_triangles(col_of_squares, row_of_triangles, path,
                                                                     terminated=None)

        return col_of_squares, row_of_triangles

    def hungarian_algorithm(self, aux_row_vars, aux_col_vars, iteration):
        self.create_equality_matrix(self.score_matrix, aux_row_vars, aux_col_vars)

        self.find_maximal_matching()

        col_of_squares, row_of_triangles = self.perform_marking_procedure()

        # Base case
        if self.is_perfect_matching():
            return self.matching

        aux_row_vars, aux_col_vars = self.update_aux_numbers(col_of_squares, row_of_triangles, aux_row_vars, aux_col_vars)

        iteration += 1
        return self.hungarian_algorithm(aux_row_vars, aux_col_vars, iteration)

    def is_perfect_matching(self):
        self.find_size_of_matching()
        return self.matching_size == self.matrix_size


    def run(self):
        self.matrix_size = len(self.score_matrix)

        aux_row_vars = self.init_row_aux_vars()
        aux_col_vars = self.init_col_aux_vars()

        # Master method for the hungarian algorithm / job assignment
        self.hungarian_algorithm(aux_row_vars, aux_col_vars, 1)

        return self.matching

    def hungarian_main(self):
        global ds
        ds = DataStore()
        mentors, mentees = self.create_assignment_matrix()
        self.matrix_size = len(self.score_matrix)
        
        aux_row_vars = self.init_row_aux_vars()
        aux_col_vars = self.init_col_aux_vars()

        # Master method for the hungarian algorithm / job assignment
        self.hungarian_algorithm(aux_row_vars, aux_col_vars, 1)

        matching_list = self.format_matching_result(mentors, mentees)
        # ds.save_matches(matching_list)

        show_matches(matching_list)

        return matching_list

    def hungarian_main_test(self):
        self.score_matrix = self.create_job_assignment_matrix()
        self.matrix_size = len(self.score_matrix)

        aux_row_vars = self.init_row_aux_vars()
        aux_col_vars = self.init_col_aux_vars()

        # Master method for the hungarian algorithm / job assignment
        self.hungarian_algorithm(aux_row_vars, aux_col_vars, 1)
    
    def hungarian_main_test_suite(self, assignment_matrix):
        self.score_matrix = assignment_matrix[:]

        self.no_mentors = len(self.score_matrix)
        no_mentees = len(self.score_matrix[0])

        if self.no_mentors != no_mentees:
            diff = no_mentees - self.no_mentors

            for _ in range(diff):
                dummy_mentor = [0 for _ in self.score_matrix[0]]
                self.score_matrix.append(dummy_mentor)

        self.matrix_size = no_mentees

        aux_row_vars = self.init_row_aux_vars()
        aux_col_vars = self.init_col_aux_vars()

        # Master method for the hungarian algorithm / job assignment
        self.hungarian_algorithm(aux_row_vars, aux_col_vars, 1)

        matching_without_dummies = []

        for i in range(self.no_mentors):
            matching_without_dummies.append(self.matching[i])

        return self.score_matrix, matching_without_dummies


if __name__ == "__main__":
    hg = HungarianAlgorithm()
    hg.hungarian_main()