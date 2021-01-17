import heapq
import json
import operator
import random as random
import re
from typing import List
import time

import gender_guesser.detector as gender
import language_tool_python
import nltk
import numpy as np
import textstat
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import global_vars
from Caching import *
from ClusteringMentors import SimilarityCalc
from KeywordExtraction import get_keyword_list
from LocalitySensitiveHashing import *
from objects import Mentee, Mentor
from SemanticSimilarity import Glove, Glove_new, glove_score_1v1

gender_detector = gender.Detector()
''' If the mentee has many entries in their LinkedIn, they will be rewarded.'''

def get_person_index_for_id(id, person_list):
    for idx,person in enumerate(person_list):
        if str(person.id) == str(id):
            return idx

    return None

def reward_potential_priorities_mentee_lsh(matrix, mentors, mentees, score=10, **kwargs):
    mentee_contents = []

    for mentee in mentees:
        mentee_contents.append(mentee.content)

    # Returns top k similar mentees for each mentee
    text_sim_lsh = TextSimilarityWithLSH(mentee_contents, 0.4, 64, 2)
    text_similarities = text_sim_lsh.text_similarity_with_lsh()
    count = len(text_similarities)
    total_cands = 0

    for i,sim in enumerate(text_similarities):
        for j in sim:
            total_cands += len(sim)
            if j != i:
                if mentees[i].industry == mentees[j].industry:
                    for priority in mentees[j].priorities:
                        mentor_idx = get_person_index_for_id(priority, mentors)
                        if mentor_idx != None:
                            matrix[int(mentor_idx)][i] += score
    
    print("Average candidate pairs of mentees:")
    print(total_cands / count)
    return matrix


def reward_potential_priorities_mentor_lsh(matrix, mentors, mentees, score=10, **kwargs):
    mentor_contents = []

    for mentor in mentors:
        mentor_contents.append(mentor.content)

    # Returns top k similar mentees for each mentee
    text_sim_lsh = TextSimilarityWithLSH(mentor_contents, 0.4, 64, 2)
    text_similarities = text_sim_lsh.text_similarity_with_lsh()
    total = len(text_similarities)
    count = 0

    for i,sim in enumerate(text_similarities):
        for j in sim:
            count += len(sim)
            if j != i:
                if bool(set(mentors[i].industry) & set(mentors[j].industry)):
                    for priority in mentors[j].priorities:
                        mentee_idx = get_person_index_for_id(priority, mentees)
                        if mentee_idx != None:
                            matrix[i][int(mentee_idx)] += score
  
    return matrix


def reward_test_lsh_falses(matrix, mentors, mentees, score=10, **kwargs):
    lt = LemmaTokenizer()
    threshold = 0.6
    bands = 20
    rows = 5
    false_negatives = 0
    false_positives = 0
    total = 0

    mentee_contents = []
    for mentee in mentees:
        mentee_contents.append(mentee.content)

    start_time = time.time()
    text_sim_lsh = TextSimilarityWithLSH(mentee_contents, threshold, bands, rows)
    text_similarities = text_sim_lsh.text_similarity_with_lsh() 
    elapsed_time = time.time() - start_time
    
    for i,sim in enumerate(text_similarities):
        print("Reached index:   " + str(i))
        for j in sim:
            total += 1
            # Calculate actual jaccard similarity
            mentee_i = re.sub(r'[^\w\s]','', str(mentees[i].content).lower())
            i_tokens = lt.__call__(mentee_i)
            mentee_j = re.sub(r'[^\w\s]','', str(mentees[j].content).lower())
            j_tokens = lt.__call__(mentee_j)

            jaccard = jaccard_similarity(i_tokens, j_tokens)

            if threshold > jaccard:
                false_positives += 1
        # Now we need a loop to check all them that haven't been put in buckets

        for j in range(0,len(mentees)):
            if j not in sim:
                mentee_i = re.sub(r'[^\w\s]','', str(mentees[i].content).lower())
                i_tokens = lt.__call__(mentee_i)
                mentee_j = re.sub(r'[^\w\s]','', str(mentees[j].content).lower())
                j_tokens = lt.__call__(mentee_j)
                total += 1

                jaccard2 = jaccard_similarity(i_tokens, j_tokens)

                if jaccard2 > threshold:
                    false_negatives += 1
                
    print("Threshold:    " + str(threshold))
    print("Bands:   " + str(bands))
    print("Rows:   " + str(rows))
    print("These are the numbers of false positives:")
    print(false_positives)
    print("These are the numbers of false negatives:")
    print(false_negatives)
    print("Total comparisons:")
    print(total)
    print("Elapsed time:")
    print(elapsed_time)

    return matrix

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

''' We still need to figure out if this is actually useful. Not tested enough. '''
def reward_mentee_linkedin(matrix, mentors, mentees, score=50, **kwargs):
    with open('data/linkedin/linkedindata2.json') as f:
        json_object = json.load(f)
        for i, _ in enumerate(mentors):
            for j, mentee in enumerate(mentees):
                jobs = None
                for elem in json_object:
                    # We found the corresponding data to the mentee in the json
                    if elem["query"] == mentee.linkedin:
                        try:
                            jobs = elem["jobs"]
                        except KeyError:
                            break

                if jobs != None:
                    if len(jobs) > 5:
                        matrix[i][j] += score

    return matrix


''' If the mentee does not have a LinkedIn picture, we punish (usually means they are not active).'''


''' We still need to figure out if this is actually useful. Not tested enough. '''
def punish_missing_linkedin_picture(matrix, mentors, mentees, score=30, **kwargs):
    with open('data/linkedin/linkedindata2.json', encoding="utf8") as f:
        json_object = json.load(f)
        for i, _ in enumerate(mentors):
            for j, mentee in enumerate(mentees):
                data = None
                for elem in json_object:
                    if elem["query"] == mentee.linkedin:
                        try:
                            data = elem["general"]
                            break
                        except KeyError:
                            break

                if data != None:
                    if data["imgUrl"] == "":
                        matrix[i][j] -= score
    return matrix


# To finish this, we need a "remote okay" field for the mentees.
# Punishes remote mentor if the given mentee does not want a remote mentor
def punish_remote_mentor(matrix, mentors, mentees, score=10, **kwargs):
    with open('data/linkedin/linkedindata2.json') as f:
        json_object = json.load(f)
        for i, mentor in enumerate(mentors):
            if "linkedin" not in str(mentor.linkedin):
                continue

            data = None
            for elem in json_object:
                try:
                    if str(mentor.linkedin) == elem["query"]:
                        data = elem["general"]
                        break
                except KeyError:
                    data = None
                    break

            if data == None:
                continue

            try:
                if "denmark" in str(data["location"]).lower():
                    continue
            except KeyError:
                continue

            for j, mentee in enumerate(mentees):
                if mentee.remote.lower() == "no":
                    matrix[i][j] -= score
                elif mentee.remote.lower() != "yes":
                    matrix[i][j] -= score / 3

    return matrix

def punish_remote_mentee(matrix, mentors, mentees, score=10, **kwargs):
    not_remote = ["Karol Krzak", "Tinna Dofradottir", "Kyriakos Michailidis", "Stefanos Ntomalis", "Johannes Thiem", "Antonia Griesz", "Konstantinos Papamanolis", "Ludek Cizinsky", "Alejandro Manzano Azores"]
    
    with open('data/linkedin/linkedindata2.json') as f:
        json_object = json.load(f)
        for i, mentor in enumerate(mentors):
 
            for j, mentee in enumerate(mentees):
                data = None
                for elem in json_object:
                    try:
                        if str(mentee.linkedin) == elem["query"]:
                            data = elem["general"]
                            break
                    except KeyError:
                        data = None
                        break

                if data == None:
                    continue

                try:
                    location = data["location"].lower()

                    if mentee.name in not_remote:
                        continue

                    if "denmark" not in location and "copenhagen" not in location:
                        matrix[i][j] = 0
                except KeyError:
                    continue

    return matrix

''' I think we can decomission this function - it's not useful. '''
def punish_bad_mentee_grammar(matrix, mentors, mentees, score=10, **kwargs):
    tool_us = language_tool_python.LanguageTool('en-US')
    tool_uk = language_tool_python.LanguageTool('en-GB')
    grammar_scores = []

    for j, mentee in enumerate(mentees):
        no_grammar_mistakes_us = len(tool_us.check(str(mentee.content)))
        no_grammar_mistakes_uk = len(tool_uk.check(str(mentee.content)))
        no_words = len(re.findall(r'\w+', str(mentee.content)))
        if no_grammar_mistakes_uk == 0 or no_grammar_mistakes_us == 0:
            grammar_scores.append(0)
            continue
        content_relationship_us = (no_grammar_mistakes_us / no_words) * 400
        content_relationship_uk = (no_grammar_mistakes_uk / no_words) * 400
        grammar_score = min(content_relationship_uk, content_relationship_us)

        if grammar_score > 20:
            grammar_scores.append(score)

    for i, _ in enumerate(mentors):
        for j, _ in enumerate(mentees):
            matrix[i][j] -= grammar_scores[j]

    return matrix


''' I think we can decomission this function. It is not useful. '''
def reward_generalists(matrix, mentors, mentees, score=10, **kwargs):
    mentor_leadership_keywords = ['manager', 'head', 'CEO', 'chief', 'cio', 'cto', 'officer' 'director', 'president', 'vice' 'executive', 'leader', 'leadership', 'management', 'generalist', 'leading', 'managing', 'managerial']
    mentee_leadership_keywords = ['manager', 'leader', 'leadership', 'management', 'generalist', 'leading', 'managing', 'managerial']

    for i, mentor in enumerate(mentors):
        mentor_result = [True for val in mentor_leadership_keywords if val in str(mentor.role).lower()]
        if True in mentor_result:
            for j, mentee in enumerate(mentees):
                mentee_result = [True for val in mentee_leadership_keywords if val in str(mentee.content)]
            
                if len(mentee_result) > 1:
                    matrix[i][j] += score

    return matrix


def punish_bachelor_mentees(matrix, mentors, mentees, score=50, **kwargs):
    for i, _ in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            study = str(mentee.study_line).lower()
            sem = float(mentee.semester)
            if ("bsc" in study or "beng" in study or "diplom" in study) and sem < 5:
                matrix[i][j] -= score
    return matrix


def reward_clustered_mentors(matrix, mentors, mentees, person_dict, score=30):
    for j, mentee in enumerate(mentees):
        cluster_list = get_cluster_list(mentee, person_dict)
        for i, mentor in enumerate(mentors):
            if mentor.id in cluster_list and not mentor.id in mentee.priorities:
                matrix[i][j] += score

    return matrix


''' I think we can decomission this function - it's not useful. '''
def reward_high_lix(matrix, mentors, mentees, score=50, **kwargs):
    lix_numbers = []

    for i, _ in enumerate(mentors):
        lix_numbers = []
        for j, mentee in enumerate(mentees):
            content = str(mentee.content)
            no_words = len(re.findall(r'\w+', content))
            no_periods = content.count('.') + content.count(':')
            list_of_words = content.split()
            no_long_words = len([word for word in list_of_words if len(word) > 6])

            if float(no_words) != 0 and float(no_periods) != 0:
                lix = (float(no_words) / float(no_periods)) + (float(no_long_words * 100) / float(no_words))
            else:
                lix = 0
            lix_numbers.append(lix)

        lix_average = sum(lix_numbers)/len(lix_numbers)

        for j, mentee in enumerate(mentees):
            if lix_numbers[j] < 45:
                matrix[i][j] += score

                if lix > 34:
                    matrix[i][j] += score
    return matrix


def punish_lack_of_linkedin(matrix, mentors, mentees, score=50, **kwargs):
    count = 0
    for i, _ in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            if "linkedin" not in str(mentee.linkedin):
                count += 1
                matrix[i][j] -= score
    return matrix


def punish_short_mentee_profile(matrix, mentors, mentees, score=100, **kwargs):
    for i, _ in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            if len(re.findall(r'\w+', str(mentee.content))) < 250:
                matrix[i][j] -= score
    return matrix

def punish_short_mentor_profile(matrix, mentors, mentees, score=0, **kwargs):
    for i, mentor in enumerate(mentors):
        if len(re.findall(r'\w+', str(mentor.content))) < 70:
            for j, mentee in enumerate(mentees):
                matrix[i][j] = score

    return matrix

def punish_mentor_without_fullname(matrix, mentors, mentees, score=0, **kwargs):
    for i, mentor in enumerate(mentors):
        if len(re.findall(r'\w+', str(mentor.name))) < 2:
            for j, mentee in enumerate(mentees):
                matrix[i][j] = score

    return matrix

def reward_female_gender(matrix, mentors, mentees, score=40, **kwargs):
    for i, mentor in enumerate(mentors):
        mentor_gender = gender_detector.get_gender(str(mentor.name.split(' ', 1)[0]))
        if str(mentor_gender) == 'female' or str(mentor_gender) == 'mostly_female':
            for j, mentee in enumerate(mentees):
                mentee_gender = gender_detector.get_gender(str(mentee.name.split(' ', 1)[0]))
                if str(mentee_gender) == 'female' or str(mentee_gender) == 'mostly_female':
                    text = mentee.content + mentor.content

                    # If female has been mentioned in either mentor's og mentee's profile
                    # then it must be important to have a female counterpart
                    if "woman" in text or "female" in text or "girl" in text.lower():
                        matrix[i][j] += score

    return matrix


# This should take both priorities and something else into consideration
def reward_same_industry(matrix, mentors, mentees, score=100, **kwargs):
    no_mentors = len(mentors)
    no_mentees = len(mentees)

    for i, mentor in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            if mentee.industry in mentor.industry:
                matrix[i][j] += score

    return matrix


def reward_priorities(matrix, mentors, mentees, score=150, **kwargs):
    no_mentors = len(mentors)
    no_mentees = len(mentees)

    for i, mentor in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            if str(mentee.id) in mentor.priorities:
                matrix[i][j] += score

    return matrix


def set_lower_limit(matrix, mentors, mentees, score=150, **kwargs):
    no_mentors = len(mentors)
    no_mentees = len(mentees)

    for i, mentor in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            if str(mentee.id) in mentor.priorities:
                if matrix[i][j] < score:
                    matrix[i][j] = 0
                    continue

    return matrix



def get_cluster_list(mentee, person_dict):
    other_priorities = []

    for mentor_id in mentee.priorities:
        try:
            mentor_prioritized = person_dict[int(mentor_id)].priorities
        except:
            continue
        for mentee_id in mentor_prioritized:
            try:
                mentee_priority = person_dict[int(mentee_id)].priorities
                other_priorities.extend(mentee_priority)
            except:
                pass

    for elem in other_priorities:
        if other_priorities.count(elem) == 1:
            other_priorities.remove(elem)

    return set(other_priorities)


# Interface lemma tokenizer from nltk with sklearn
class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


def reward_similar_text(matrix, mentors, mentees, score=220, **kwargs):
    sc = SimilarityCalc()

    filename = "similarity_matrix_" + global_vars.ROUND + ".txt"

    binary_matrix = get_or_calculate_cache(filename, calculate_similarity, mentees=mentees, mentors=mentors, sc=sc)
    for i in range(len(binary_matrix)):
        for j in range(len(binary_matrix[0])):
            if binary_matrix[i][j] == 1:
                matrix[i][j] += score

    return matrix


def calculate_similarity(mentees, mentors, sc):
    binary_matrix = [[len(mentors)] * len(mentees) for i in range(len(mentors))]
    for i, mentor in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            similarity_score = sc.content_similarity(str(mentor.content), str(mentee.content))[0]
            if similarity_score > 0.2:
                binary_matrix[i][j] = 1
            else:
                binary_matrix[i][j] = 0
    return binary_matrix

def find_highest_scores_mentee_refactored(binary_matrix, similarity_score, mentees):
    priorities_list = []
    for i, this_mentee in enumerate(mentees):
        priorities = []

        indices = list(list(zip(*heapq.nlargest(5, enumerate(binary_matrix[i]), key=operator.itemgetter(1))))[0])
        for indice in indices:
            priorities.extend(mentees[int(indice)].priorities)

        priorities_list.append(priorities)

    return priorities_list


def find_highest_scores_mentor_refactored(binary_matrix, similarity_score, mentors):
    priorities_list = []
    for i, this_mentor in enumerate(mentors):
        priorities = []

        indices = list(list(zip(*heapq.nlargest(5, enumerate(binary_matrix[i]), key=operator.itemgetter(1))))[0])
        for indice in indices:
            priorities.extend(mentors[int(indice)].priorities)

        priorities_list.append(priorities)

    return priorities_list


def find_highest_scores_mentee(binary_matrix, similarity_score, mentees):
    priorities_list = []
    for i, this_mentee in enumerate(mentees):
        if not this_mentee.priorities:
            priorities_list.append([])
            continue
        priorities = []

        indices = list(list(zip(*heapq.nlargest(5, enumerate(binary_matrix[i]), key=operator.itemgetter(1))))[0])
        for indice in indices:
            priorities.extend(mentees[int(indice)].priorities)

        priorities_list.append(priorities)

    return priorities_list


def find_highest_scores_mentor(binary_matrix, similarity_score, mentors):
    priorities_list = []
    for i, this_mentor in enumerate(mentors):
        if not this_mentor.priorities:
            priorities_list.append([])
            continue
        priorities = []

        indices = list(list(zip(*heapq.nlargest(5, enumerate(binary_matrix[i]), key=operator.itemgetter(1))))[0])
        for indice in indices:
            priorities.extend(mentors[int(indice)].priorities)

        priorities_list.append(priorities)

    return priorities_list

def reward_potential_priorities_mentor_refactored(matrix, mentees, mentors, score=100, **kwargs):
    binary_matrix = mentor_similar_text(mentors)
    similarity_score = find_similarity_mentor(mentors)
    scores = find_highest_scores_mentor_refactored(binary_matrix, similarity_score, mentors)
    for j, mentee in enumerate(mentees):
        for i, mentor in enumerate(mentors):
            if str(mentee.id) in scores[i]:
                matrix[i][j] += score

    return matrix

def reward_potential_priorities_mentee_factor_refactored(matrix, mentees, mentors, score=100, **kwargs):
    binary_matrix = mentee_similar_text(mentees)
    similarity_score = find_similarity_mentee(mentees)
    scores = find_highest_scores_mentee_refactored(binary_matrix, similarity_score, mentees)
    for i, mentor in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            if str(mentor.id) in scores[j]:
                if len(mentee.priorities) == 0:
                    matrix[i][j] += score

    return matrix


def reward_potential_priorities_mentor(matrix, mentees, mentors, score=100, **kwargs):
    binary_matrix = mentor_similar_text(mentors)
    similarity_score = find_similarity_mentor(mentors)
    scores = find_highest_scores_mentor(binary_matrix, similarity_score, mentors)
    for j, mentee in enumerate(mentees):
        for i, mentor in enumerate(mentors):
            if str(mentee.id) in scores[i]:
                matrix[i][j] += score

    return matrix

def reward_potential_priorities_mentor_factor(matrix, mentees, mentors, score=100, **kwargs):
    # 1: Find similar values based on text
    binary_matrix = mentor_similar_text(mentors)
    # 2: Find similarities based on other things
    similarity_score = find_similarity_mentor(mentors)
    # 3: Find the mentees with highest total scores
    scores = find_highest_scores_mentor(binary_matrix, similarity_score, mentors)
    # 4: Choose the priorities of those mentees in a smart way
    # 5: Add those to the right columns/rows
    for j, mentee in enumerate(mentees):
        for i, mentor in enumerate(mentors):
            if str(mentee.id) in scores[i]:
                # Change to it more than 0 and maybe remove 0s
                if len(mentor.priorities) > 0:
                    matrix[i][j] += score
                else:
                    matrix[i][j] += score*4
                

    return matrix


def reward_potential_priorities_mentor_factor_refactored(matrix, mentees, mentors, score=100, **kwargs):
    # 1: Find similar values based on text
    binary_matrix = mentor_similar_text(mentors)
    # 2: Find similarities based on other things
    similarity_score = find_similarity_mentor(mentors)
    # 3: Find the mentees with highest total scores
    scores = find_highest_scores_mentor_refactored(binary_matrix, similarity_score, mentors)
    # 4: Choose the priorities of those mentees in a smart way
    # 5: Add those to the right columns/rows
    for j, mentee in enumerate(mentees):
        for i, mentor in enumerate(mentors):
            if str(mentee.id) in scores[i]:
                # Change to it more than 0 and maybe remove 0s
                if len(mentor.priorities) == 0:
                    matrix[i][j] += score

    return matrix


def reward_potential_priorities_mentee_factor(matrix, mentees, mentors, score=100, **kwargs):
    binary_matrix = mentee_similar_text(mentees)
    similarity_score = find_similarity_mentee(mentees)
    scores = find_highest_scores_mentee(binary_matrix, similarity_score, mentees)
    for i, mentor in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            if str(mentor.id) in scores[j]:
                if len(mentee.priorities) > 2:
                    matrix[i][j] += score
                else:
                    matrix[i][j] += score*4

    return matrix

def reward_potential_priorities_mentee_refactored(matrix, mentees, mentors, score=100, **kwargs):
    # 1: Find similar values based on text
    binary_matrix = mentee_similar_text(mentees)
    # 2: Find similarities based on other things
    similarity_score = find_similarity_mentee(mentees)
    # 3: Find the mentees with highest total scores
    scores = find_highest_scores_mentee_refactored(binary_matrix, similarity_score, mentees)
    # 4: Choose the priorities of those mentees in a smart way
    # 5: Add those to the right columns/rows
    for i, mentor in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            if str(mentor.id) in scores[j]:
                matrix[i][j] += score

    return matrix

def reward_potential_priorities_mentee(matrix, mentees, mentors, score=100, **kwargs):
    # 1: Find similar values based on text
    binary_matrix = mentee_similar_text(mentees)
    # 2: Find similarities based on other things
    similarity_score = find_similarity_mentee(mentees)
    # 3: Find the mentees with highest total scores
    scores = find_highest_scores_mentee(binary_matrix, similarity_score, mentees)
    # 4: Choose the priorities of those mentees in a smart way
    # 5: Add those to the right columns/rows
    for i, mentor in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            if str(mentor.id) in scores[j]:
                    matrix[i][j] += score



    return matrix


def mentee_similar_text(mentees, **kwargs):
    sc = SimilarityCalc()
    file_name = "mentee_similarity_{}.txt".format(global_vars.ROUND)
    binary_matrix = get_or_calculate_cache(file_name, calculate_mentee_similar_text, mentees=mentees, sc=sc)
    return binary_matrix

def mentor_similar_text(mentors, **kwargs):
    sc = SimilarityCalc()
    file_name = "mentor_similarity_{}.txt".format(global_vars.ROUND)
    binary_matrix = get_or_calculate_cache(file_name, calculate_mentor_similar_text, mentors=mentors, sc=sc)
    return binary_matrix

def calculate_mentee_similar_text(mentees, sc):
    binary_matrix = [[0] * len(mentees) for i in range(len(mentees))]
    for i, mentee_1 in enumerate(mentees):
        for j, mentee_2 in enumerate(mentees):
            if mentee_1 == mentee_2:
                continue

            similarity_score = sc.content_similarity(str(mentee_1.content), str(mentee_2.content))[0]
            binary_matrix[i][j] = similarity_score
    return binary_matrix

def calculate_mentor_similar_text(mentors, sc):
    binary_matrix = [[0] * len(mentors) for i in range(len(mentors))]
    for i, mentor_1 in enumerate(mentors):
        for j, mentor_2 in enumerate(mentors):
            if mentor_1 == mentor_2:
                continue

            similarity_score = sc.content_similarity(str(mentor_1.content), str(mentor_2.content))[0]
            binary_matrix[i][j] = similarity_score
    return binary_matrix

def find_similarity_mentee(mentees: List[Mentee]):
    binary_matrix = [[0] * len(mentees) for i in range(len(mentees))]
    for i, mentee_1 in enumerate(mentees):
        for j, mentee_2 in enumerate(mentees):
            if mentee_1.id == mentee_2.id:
                continue

            if mentee_1.study_line == mentee_2.study_line:
                binary_matrix[i][j] += 1

            if mentee_1.university == mentee_2.university:
                binary_matrix[i][j] += 0.2

    return binary_matrix

def find_similarity_mentor(mentors: List[Mentor]):
    binary_matrix = [[0] * len(mentors) for i in range(len(mentors))]
    for i, mentor_1 in enumerate(mentors):
        for j, mentor_2 in enumerate(mentors):
            if mentor_1.id == mentor_2.id:
                continue

            if any(x in mentor_1.industry for x in mentor_2.industry):
                binary_matrix[i][j] += 1

    return binary_matrix

def reward_similarity_text_to_keywords(matrix, mentees, mentors, score=0, **kwargs):
    get_or_calculate_cache("glove_cache_flipped.json", calculate_similar_score_flipped, mentees=mentees,
                           mentors=mentors)

    return matrix


def reward_similarity_keywords_to_keywords(matrix, mentees, mentors, score=0, **kwargs):
    get_or_calculate_cache("glove_cache_keywords.json", calculate_similar_score_keywords, mentees=mentees, mentors=mentors)

    return matrix


def reward_similarity_texts_to_texts(matrix, mentees, mentors, score=0, **kwargs):
    get_or_calculate_cache("glove_cache_text.json", calculate_similar_score_texts, mentees=mentees, mentors=mentors)

    return matrix

def reward_similarity_keywords_to_texts_new(matrix, mentees, mentors, score=0, **kwargs):
    result_matrix = get_or_calculate_cache("glove_cache_new_{}.json".format(global_vars.ROUND), calculate_similar_score_new, mentees=mentees, mentors=mentors)
    first = True
    mentee_keywords = get_keyword_list(mentees)

    for mentee_idx, (mentee_scores, keyword) in enumerate(zip(result_matrix, mentee_keywords)):
        if first:
            first = False
            continue

        if keyword == ['nan']:
            continue

        # print(keyword)

        if isinstance(mentee_scores, float):
            continue

        try:
            top_5 = sorted(range(len(mentee_scores)), key=lambda i: mentee_scores[i])[-6:]
        except TypeError as e:
            print("Something went wrong")
            continue

        for index in top_5:
            if index > len(mentors):
                continue
            matrix[index][mentee_idx] += score

    return matrix


def reward_similarity_keywords_to_texts(matrix, mentees, mentors, score=0, **kwargs):
    result_matrix = get_or_calculate_cache("glove_cache_{}.json".format(global_vars.ROUND), calculate_similar_score, mentees=mentees, mentors=mentors)
    first = True
    mentee_keywords = get_keyword_list(mentees)

    for mentee_idx, (mentee_scores, keyword) in enumerate(zip(result_matrix, mentee_keywords)):
        if first:
            first = False
            continue

        if keyword == ['nan']:
            continue

        # print(keyword)

        if isinstance(mentee_scores, float):
            continue

        top_5 = sorted(range(len(mentee_scores)), key=lambda i: mentee_scores[i])[-6:]
        for index in top_5:
            if index > len(mentors):
                continue
            matrix[index][mentee_idx] += score

    return matrix


def calculate_similar_score_texts(mentees, mentors):
    mentor_contents = [mentor.content for mentor in mentors]
    mentee_contents = [mentee.content for mentee in mentees]
    glove = Glove(mentor_contents)
    scores = []
    for mentee_content, mentee in zip(mentee_contents, mentees):
        glove_result = glove.get_score(mentee_content)
        scores.append(glove_result)

    return scores


def calculate_similar_score_keywords(mentees, mentors):
    mentor_keywords = get_keyword_list(mentors)
    mentee_keywords = get_keyword_list(mentees)
    glove = Glove(mentor_keywords + mentee_keywords)
    scores = []
    for mentee_keyword, mentee in zip(mentee_keywords, mentees):
        glove_result = glove.get_score(mentee_keyword)
        scores.append(glove_result)

    return scores


def calculate_similar_score_flipped(mentees, mentors):
    mentor_keywords = get_keyword_list(mentors)
    mentee_contents = [mentee.content for mentee in mentees]
    glove = Glove(mentee_contents)
    scores = []
    for mentee_keyword, mentee in zip(mentor_keywords, mentees):
        glove_result = glove.get_score(mentee_keyword)
        scores.append(glove_result)

    return scores


def calculate_similar_score_new(mentees, mentors):
    mentee_keywords = get_keyword_list(mentees)
    mentor_contents = [mentor.content for mentor in mentors]
    glove = Glove_new(mentor_contents + mentee_keywords)
    scores = []
    for mentee_keyword, mentee in zip(mentee_keywords, mentees):
        glove_result = glove.get_score(mentee_keyword)
        scores.append(glove_result)

    return scores

def calculate_similar_score(mentees, mentors):
    mentee_keywords = get_keyword_list(mentees)
    mentor_contents = [mentor.content for mentor in mentors]
    glove = Glove(mentor_contents)
    scores = []
    for mentee_keyword, mentee in zip(mentee_keywords, mentees):
        glove_result = glove.get_score(mentee_keyword)
        scores.append(glove_result)

    return scores


# Keywords to many keywords
def reward_similarity_keyword_multiple_keywords_glove(matrix, mentees, mentors, score=0, **kwargs):
    mentor_keywords = get_keyword_list(mentors)
    mentee_keywords = get_keyword_list(mentees)
    all_mentee_keywords = []

    for i, mentor in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            all_mentee_keywords.append(str(mentee_keywords))

        glove_result = glove_score_1v1(mentor_keywords, all_mentee_keywords)
        average = sum(glove_result) / len(glove_result)

        # Take result from glove and act accordingly
        for j, mentee in enumerate(mentees):
            if glove_result[j] > average:
                matrix[i][j] += score

    return matrix


# Keyword to many texts
def reward_similarity_keyword_multiple_texts_glove(matrix, mentees, mentors, score=0, **kwargs):
    mentor_keywords = get_keyword_list(mentors)
    all_mentee_content = []

    for i, mentor in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            all_mentee_content.append(str(mentee.content))

        glove_result = glove_score_1v1(mentor_keywords, all_mentee_content)
        average = sum(glove_result) / len(glove_result)

        # Take result from glove and act accordingly
        for j, mentee in enumerate(mentees):
            if glove_result[j] > average:
                matrix[i][j] += score

    return matrix


# Text to many texts
def reward_similar_text_multiple_texts_glove(matrix, mentees, mentors, score=0, **kwargs):
    all_mentee_content = []

    for i, mentor in enumerate(mentors):
        for j, mentee in enumerate(mentees):
            all_mentee_content.append(str(mentee.content))

        glove_result = glove_score_1v1(mentor.content, all_mentee_content)
        average = sum(glove_result) / len(glove_result)

        # Take result from glove and act accordingly
        for j, mentee in enumerate(mentees):
            if glove_result[j] > average:
                matrix[i][j] += score

    return matrix


def generate_random_score(matrix, mentees, mentors, score=0, **kwargs):
    for i,mentor in enumerate(mentors):
        for j,mentee in enumerate(mentees):
            score = random.randint(0,350)
            matrix[i][j] += score

    return matrix
