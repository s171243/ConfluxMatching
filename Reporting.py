import math

from DataStore import DataStore
import os
import pandas as pd
import re
import global_vars

pd.set_option('display.max_columns', 21)

def choose_file_rating(folder="data/match_rating/"):
    ds = DataStore()
    files = os.listdir(folder)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    print("Welcome to the analyzer. Which file would you like to analyze?")

    for index, file in enumerate(files):
        print("({}) {}".format(index, file))

    file_index = input("Write the number to choose the file. Write anything else to cancel\n")

    if int(file_index) in range(1, len(files)):
        print("File {} chosen".format(files[int(file_index)]))
        results = ds.load_rating_file(files[int(file_index)])
        return results
    else:
        print("Command not recognized. Returning.")
        return


def calculate_all_ratings(folder="data/match_rating/"):
    ds = DataStore()
    files = os.listdir(folder)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    for file in files:
        print("File {} chosen".format(file))
        results = ds.load_rating_file(file)

        # calculate the relevant data
        count_maybe = 0
        count_yes = 0
        count_no = 0

        for i, result in results.iterrows():
            rating = result["match_rating"]
            if rating == 0:
                count_no += 1
            elif rating == 1:
                count_yes += 1
            elif rating == -1:
                count_maybe += 1

        # Create data
        data = {
            "bad": count_no,
            "good": count_yes,
            "maybe": count_maybe
        }

        output_path = "data/rating_result.csv"
        now_id = file.replace("rating_", "").replace(".csv", "")
        data["id"] = now_id

        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            df2 = pd.DataFrame([data])
            df3 = df.append(df2)
            df3.to_csv(output_path, index=False, mode='w')
        else:
            df = pd.DataFrame([data])
            df.to_csv(output_path, index=False, mode='w')

        print("Good: {}\nBad: {}\nMaybe: {}".format(count_yes, count_no, count_maybe))
        # append it to .csv-file

def choose_file(folder="data/matching_result/"):
    ds = DataStore()
    mentors, mentees, person_dict = ds.load_data("initial_data_{}".format(global_vars.ROUND))
    files = os.listdir(folder)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    print("Welcome to the analyzer. Which file would you like to analyze?")

    for index, file in enumerate(files):
        print("({}) {}".format(index, file))

    file_index = input("Write the number to choose the file. Write anything else to cancel\n")

    if int(file_index) in range(1, len(files)):
        print("File {} chosen".format(files[int(file_index)]))
        results = ds.load_result_file(files[int(file_index)])
        str_id = files[int(file_index)].replace("matching_result_", "").replace(".csv", "")
    else:
        print("Command not recognized. Returning.")
        return

    return person_dict, results, str_id


class Reporting:
    def __init__(self):
        pass

    def analyze(self):
        person_dict, results, _ = choose_file()

        number_old_matches = 0
        number_non_old_matches = 0

        for index, result in results.iterrows():
            mentee_id = result['mentee_id']
            mentor_id = result['mentor_id']
            if mentee_id in person_dict and mentor_id in person_dict:
                if mentee_id == person_dict[mentor_id].former_match:
                    number_old_matches += 1
                else:
                    number_non_old_matches += 1
            else:
                print("What happened here? {} {}, {} {}".format(mentee_id, mentor_id, (mentee_id in person_dict),
                                                                (mentor_id in person_dict)))

        print("RESULT:\nWe had {} matches that were the same\n"
              "We had {} new matches".format(number_old_matches, number_non_old_matches))

    def analyze_2(self):
        df = pd.read_csv("data/matching_configuration_2.csv")
        df2 = pd.read_csv("data/matching_results.csv")
        df3 = df.merge(df2, left_on="id", right_on="id")
        print("Which one of the following would you like to analyze?\n")
        for index, row in df3.iterrows():
            print("{}: {}".format(index, row["id"]))

        result = input("\nWrite a number.\n")
        if int(result) in range(df3.shape[0]):
            row = df3.iloc[[int(result)]]
            # print(row)
            print("     {:>10} {:>10} {:>10}".format("Q1", "Q2", "Q3"))
            print("OLD: {:>10} {:>10} {:>10}".format(row['old_q1'].values[0], row['old_q2'].values[0],
                                                     row['old_q3'].values[0]))
            print("NEW: {:>10} {:>10} {:>10}".format(row['new_q1'].values[0], row['new_q2'].values[0],
                                                     row['new_q3'].values[0]))

            print("\nRepeated matches: {}. Repeated mentees: {}".format(row['new_repeat'].values[0],
                                                                        int(row['matched_again'].values[0])))

            print("\n\nSCORE FUNCTIONS")
            for column in row.columns:
                if "reward" in column or "punish" in column:
                    print("{:>28}: {:>15}".format(column, int(row[column].values[0])))


if __name__ == "__main__":
    r = Reporting()
    r.analyze_2()
