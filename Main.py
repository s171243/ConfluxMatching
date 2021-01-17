from DataStore import DataStore
from MatchGenerator import MatchGenerator, get_score_function_from_user
from MatchRater5000 import show_matches
from Reporting import Reporting, choose_file, choose_file_rating, calculate_all_ratings
from Scoring import *
import global_vars

def run_reporting():
    reporting = Reporting()
    reporting.analyze_2()


def re_match():
    # get global list
    TOGGLE_CMDLINE = 1
    ds = DataStore()
    mentors, mentees, person_dict = ds.load_data("initial_data_{}".format(global_vars.ROUND))

    # get rating list
    ratings = choose_file_rating()

    solo_mentors = []
    solo_mentees = []
    # get list of unmatched

    print("wow")

    for mentor, mentee in zip(mentors, mentees):
        mentee_value = ratings.query('mentee_id == "' + str(mentee.id) + '"')
        mentor_value = ratings.query('mentor_id == "' + str(mentor.id) + '"')

        if not mentee_value.empty:
            mentee_index = mentee_value.index
            is_matched = ratings["match_rating"][mentee_index].values[0]
            if is_matched != 1:
                if mentee.id in person_dict:
                    solo_mentees.append(mentee)
        else:
            solo_mentees.append(mentee)

        if not mentee_value.empty:
            mentor_index = mentor_value.index
            is_matched = ratings["match_rating"][mentor_index].values[0]
            if is_matched != 1:
                if mentor.id in person_dict:
                    solo_mentors.append(mentor)
        else:
            solo_mentors.append(mentor)

    print("fsfdsf")
    """
    for i, rating in ratings.iterrows():
        if rating["match_rating"] != 1:
            if rating["mentee_id"] in person_dict:
                solo_mentees.append(person_dict[rating["mentee_id"]])

            if rating["mentor_id"] in person_dict:
                solo_mentors.append(person_dict[rating["mentor_id"]])"""

    print(len(solo_mentees))
    print(len(solo_mentors))

    # create new Match Generator
    mg = MatchGenerator(mentors=solo_mentors, mentees=solo_mentees, person_dict=person_dict)
    mg.set_rematch(True)

    if TOGGLE_CMDLINE:
        score_function = get_score_function_from_user()
        if score_function is None:
            score_function = ScoreFunction5
    else:
        score_function = ScoreFunction5

    mg.calculate_score_object(score_function)

    filename = mg.run_hungarian()
    mg.save_hungarian_result()
    mg.save_matching_configuration()
    # should we save it in a different way?
    str_id = mg.get_str_id()
    re_ratings = ds.load_rating_file(filename, prefix="")

    matching_list = []

    for _, row in re_ratings.iterrows():
        mentor = person_dict[row["mentor_id"]]
        mentee = person_dict[row["mentee_id"]]
        score = row["score"]
        matching_list.append((mentor, mentee, score))

    show_matches(matching_list, str_id)


def run_rating():
    print("\nWhich round of the program do you want to work with?\n1) Q1\n2) Q3\n")
    round = input("Write a number:  ")
    if round == '1':
        global_vars.ROUND = "Q1"
    elif round == '2':
        global_vars.ROUND = "Q3"
        
    person_dict, results, str_id = choose_file()
    result_list = []
    for index, result in results.iterrows():
        mentee_id = result['mentee_id']
        mentor_id = result['mentor_id']
        score = result['score']
        if mentee_id in person_dict and mentor_id in person_dict:
            result_list.append((person_dict[mentor_id], person_dict[mentee_id], score))

    show_matches(result_list, str_id=str_id)


def main():
    running = True
    while (running):
        TOGGLE_CMDLINE = True
        print("\nWelcome to the match-generator\n\n")
        print("You have a few different options:")
        print("1) Run the matching algorithm")
        print("2) See reporting on a matching round")
        print("3) Manually rate the matchings")
        print("4) Retroactively calculate matching scores")
        print("5) Rematch")
        print("Write anything else to exit")

        choice = input("\nWhich one do you want?\n")

        if choice == '1':
            run_matching(TOGGLE_CMDLINE)
        elif choice == '2':
            run_reporting()
        elif choice == '3':
            run_rating()
        elif choice == '4':
            calculate_all_ratings()
        elif choice == '5':
            re_match()
        else:
            print("Command not recognized. Exiting.")
            running = False


def run_matching(TOGGLE_CMDLINE):

    print("\nWhich round of the program do you want to work with?\n1) Q1\n2) Q3\n")
    round = input("Write a number:  ")
    if round == '1':
        global_vars.ROUND = "Q1"
    elif round == '2':
        global_vars.ROUND = "Q3"

    match_gen = MatchGenerator()
    print("Started MatchGenerator")
    numbers = [0, 5, 10, 15, 25, 50, 70]
    highest_number = 0
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
    match_gen.calculate_percentage_rematched()
    print("Percentage calculated")
    # match_gen.analyze_hungarian_result()
    match_gen.save_hungarian_result()
    match_gen.save_matching_configuration()
    main()
    # print("Score: {}. That's: {:. 2f}".format(number_old_matches, ))"""


if __name__ == "__main__":
    main()
