import os
from datetime import datetime
from typing import *

import pandas as pd
from pandas import DataFrame

from objects import Mentee, Mentor, Person


class DataStore():

    def __init__(self):
        self.DEBUG = 0
        self.rematch = 0
        self.now = datetime.now()  # current date and time
        self.now_str = self.now.strftime("%Y%m%d_%H_%M_%S")

    def get_now(self):
        return self.now_str

    def save_to_disk(self, df: DataFrame, prefix: str = "match_rating/rating_", filename=""):
        if self.DEBUG:
            return
        if self.rematch:
            prefix = "match_rerating/rating_"
        if filename == "":
            filename = self.now_str
        df.to_csv("data/" + prefix + filename + ".csv", sep=";")

    def save_to_disk_custom(self, df: DataFrame, file_name: str = "match_rating/rating_"):
        if self.DEBUG:
            return
        df.to_csv("data/" + ".csv", sep=";")

    def save_matches(self, list_of_matches, prefix="matching_result/matching_result_"):
        if self.rematch:
            prefix = "match_rerating/rating_"
        match_dataframe = self.prepare_array(list_of_matches)
        self.save_to_disk(match_dataframe, prefix)
        self.save_to_disk_custom(match_dataframe, "matching_result/newest_matching_result")
        return "data/" + prefix + self.now_str + ".csv"

    def prepare_array(self, list_of_matches):
        df = pd.DataFrame()
        for mentor, mentee, score in list_of_matches:
            new_df = pd.DataFrame({"mentor_name": [mentor.name], "mentee_name": [mentee.name],
                                   "mentee_id": mentee.id, "mentor_id": mentor.id, "score": score})
            df = df.append(new_df, ignore_index=True)
        else:
            return df

    def load_data(self, filename):
        df = pd.read_csv("data/" + filename + ".csv", sep=";")

        mentors = []
        mentees = []

        for row_tuple in df.iterrows():
            row = row_tuple[1]

            if row["type"] == "mentee":
                mentee = self.initialize_mentee(row)

                mentees.append(mentee)

            if row["type"] == "mentor":
                mentor = self.initialize_mentor(row)
                mentors.append(mentor)

        person_dict = {}
        for mentor, mentee in zip(mentors, mentees):
            person_dict[mentee.id] = mentee
            person_dict[mentor.id] = mentor
        for mentee in mentees:
            person_dict[mentee.id] = mentee

        return mentors, mentees, person_dict

    def initialize_mentor(self, row):
        mentor = Mentor(name=row["name"], age=0, id=row["id"], gender=row["gender"],
                        linkedin=row["extp_linkedin"],
                        university=row["education_background_0_degree"], study_line="", company=row["company"],
                        company_type=row["company_type"], position=row["extp_position"], role=str(row["position"]),
                        content=row["content"])
        priority_1 = str(row["mentor_prioritized_1"]).split(",")
        priority_2 = str(row["mentor_prioritized_2"]).split(",")
        priority_3 = str(row["mentor_prioritized_3"]).split(",")
        mentor.former_match = row["match"]
        # mentor.position = row["match"]

        mentor.priorities = priority_1 + priority_2 + priority_3
        mentor.priorities = [x for x in mentor.priorities if x and x != 'nan' and x != ' ']

        mentor.set_industry(row["categories"])
        return mentor

    def initialize_mentee(self, row):
        mentee = Mentee(name=row["name"], age=0, id=row["id"], gender=row["gender"],
                        linkedin=row["extp_linkedin"], university=row["university"],
                        study_line=row["position"], content=row["content"], remote=str(row["remote"]),
                        semester=(row["semester"]))
        priority_1 = str(row["mentee_priority_1"]).split(",")
        priority_2 = str(row["mentee_priority_2"]).split(",")
        priority_3 = str(row["mentee_priority_3"]).split(",")

        mentee.priorities = priority_1 + priority_2 + priority_3
        mentee.priorities = [x for x in mentee.priorities if x and x != 'nan' and x != ' ']
        mentee.former_match = row["match"]

        mentee.set_field(row["position"])

        return mentee

    def load_result_file(self, filename: str = ""):
        if filename == "":
            filename = self.now_str

        return pd.read_csv("data/matching_result/" + filename, sep=";")

    def load_rating_file(self, filename: str = "", prefix: str = "data/match_rating/"):
        if filename == "":
            filename = self.now_str

        return pd.read_csv(prefix + filename, sep=None, engine='python')

    def append_matching_info(self, df, str_id):
        count_maybe = 0
        count_yes = 0
        count_no = 0

        for i, result in df.iterrows():
            rating = result["match_rating"]
            if rating == 0:
                count_no += 1
            elif rating == 1:
                count_yes += 1
            elif rating == -1:
                count_maybe += 1

        data = {
            "bad": count_no,
            "good": count_yes,
            "maybe": count_maybe,
            "id": str_id
        }

        output_path = "data/rating_result.csv"

        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            df2 = pd.DataFrame([data])
            df3 = df.append(df2)
            df3.to_csv(output_path, index=False, mode='w')
        else:
            df = pd.DataFrame([data])
            df.to_csv(output_path, index=False, mode='w')

        pass
