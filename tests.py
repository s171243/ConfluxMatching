import unittest
import hungarian_algorithm as hg
import score
import load_data

class TestHungarianAlgorithmMethods(unittest.TestCase):
    def test_matching_result(self):
        a1 = [[42, 35, 28, 21],
              [30, 25, 20, 15],
              [30, 25, 20, 15],
              [24, 20, 16, 12]]

        a2 = [[2, 3 ,6],
              [2, 4, 4],
              [1, 3, 2]]

        a3 = [[7, 5, 6, 9],
              [8, 7, 7, 5],
              [3, 1, 0, 9],
              [7, 5, 5, 9]] 

        m1 = [[(42, 2), (35, 0), (28, 0), (21, 0)],
              [(30, 1), (25, 2), (20, 1), (15, 0)],
              [(30, 1), (25, 1), (20, 2), (15, 0)],
              [(24, 0), (20, 0), (16, 1), (12, 2)]]

        m2 = [[(2, 0), (3, 0), (6, 2)],
              [(2, 2), (4, 1), (4, 1)],
              [(1, 1), (3, 2), (2, 0)]]

        m3 = [[(7, 1), (5, 0), (6, 2), (9, 1)],
              [(8, 1), (7, 2), (7, 1), (5, 0)],
              [(3, 0), (1, 0), (0, 0), (9, 2)],
              [(7, 2), (5, 0), (5, 0), (9, 1)]]

        self.assertEqual(hg.hungarian_algorithm(a1, hg.init_row_aux_vars(a1), hg.init_row_aux_vars(a1), 1), m1)
        self.assertEqual(hg.hungarian_algorithm(a2, hg.init_row_aux_vars(a2), hg.init_row_aux_vars(a2), 1), m2)
        self.assertEqual(hg.hungarian_algorithm(a3, hg.init_row_aux_vars(a3), hg.init_row_aux_vars(a3), 1), m3)

class TestScoreEvaluationMethods(unittest.TestCase):
    def test_score_evaluation_1(self):
        pass
        #self.assertEqual(fun(3), 4)

class TestLoadDataMethods(unittest.TestCase):
    def test(self):
        pass
        #self.assertEqual(fun(3), 4)

if __name__ == '__main__':
    unittest.main()