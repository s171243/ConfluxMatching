from tkinter import Tk, Text, BOTH, W, N, E, S
from tkinter.ttk import Frame, Button, Label, Style
from tkinter import ttk
from DataStore import DataStore

import pandas as pd
import os
from objects import Mentee, Mentor

LOREM = """Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Donec odio. Quisque volutpat mattis eros. Nullam malesuada erat ut turpis. Suspendisse urna nibh, viverra non, semper suscipit, posuere a, pede.

Donec nec justo eget felis facilisis fermentum. Aliquam porttitor mauris sit amet orci. Aenean dignissim pellentesque felis."""


class MatchRater:

    def __init__(self, str_id, df=None, rematch=False):
        self.df = df
        self.str_id = str_id
        self.initialize()
        self.ds = DataStore()
        if rematch:
            self.ds.rematch = True

    def initialize(self):
        self.root = Tk()
        self.root.title("MatchRater 5000")
        self.root.bind('<Return>', self.YES_onclick)
        self.root.bind('<BackSpace>', self.NO_onclick)
        self.root.bind('<space>', self.MAYBE_onclick)
        self.root.columnconfigure(0, pad=100, minsize=450)
        self.root.columnconfigure(1, pad=10, minsize=450)
        self.root.rowconfigure(0, minsize=50)

    def add_label(self, text, row, column, pady=10, padx=25, sticky=W):
        lbl = Label(self.root, text=text, wraplength=350)
        lbl.grid(row=row, column=column, pady=pady, padx=padx, sticky=sticky)

    def gui_test(self):
        self.add_label("Mentee name", 0, 0, padx=10, sticky=N)
        self.add_label("Position", 1, 0)
        self.add_label("University", 2, 0)
        self.add_label(LOREM, 3, 0)
        self.add_label("Suggestion 1", 4, 0)
        self.add_label("Suggestion 2", 5, 0)
        self.add_label("Suggestion 3", 6, 0)

        self.add_label("Mentee name", 0, 1, padx=10, sticky=N)
        self.add_label("Position", 1, 1)
        self.add_label("Company", 2, 1)
        self.add_label(LOREM, 3, 1)
        self.add_label("Suggestion 1", 4, 1)
        self.add_label("Suggestion 2", 5, 1)
        self.add_label("Suggestion 3", 6, 1)

        self.setup_bottom()

    def setup_bottom(self, match_score=""):
        lbl = Label(self.root, text=str(
            match_score) + "\nWas this match good/plausible? E.g. could it have happened when we were matching?")
        lbl.grid(row=7, column=0, columnspan=2, pady=(50, 5), sticky=S)
        btn = Button(self.root, text="Yes", command=self.YES_onclick)
        btn.grid(row=8, column=0, sticky=E, pady=(0, 50))
        btn = Button(self.root, text="No", command=self.NO_onclick)
        btn.grid(row=8, column=1, sticky=W, pady=(0, 50))

    def show_match(self, mentor, mentee, score):
        self.mentor = mentor
        self.mentee = mentee

        self.add_label(mentee.name, 0, 0, padx=10, sticky=N)
        self.add_label(mentee.position, 1, 0)
        self.add_label(mentee.university, 2, 0)
        self.add_label(mentee.content, 3, 0)
        """
        self.add_label(mentee.priorities[0], 4, 0)
        self.add_label(mentee.priorities[1], 5, 0)
        self.add_label(mentee.priorities[2], 6, 0)"""

        self.add_label(mentor.name, 0, 1, padx=10, sticky=N)
        self.add_label(mentor.position, 1, 1)
        self.add_label(mentor.company, 2, 1)
        self.add_label(mentor.content, 3, 1)

        """self.add_label(mentor.priorities[0], 4, 1)
        self.add_label(mentor.priorities[1], 5, 1)
        self.add_label(mentor.priorities[2], 6, 1)"""

        self.setup_bottom(match_score=score)

    def start(self):
        self.root.mainloop()

    def NO_onclick(self, event=None):
        print("The match between " + self.mentee.name + " and " + self.mentor.name + " was bad")
        self.df = self.df.append({"mentee_id": self.mentee.id, "mentor_id": self.mentor.id, "match_rating": 0},
                                 ignore_index=True)
        for child in self.root.winfo_children():
            child.destroy()

        self.root.quit()
        # self.root.destroy()

    def MAYBE_onclick(self, event=None):
        print("The match between " + self.mentee.name + " and " + self.mentor.name + " was good!")
        self.df = self.df.append({"mentee_id": self.mentee.id, "mentor_id": self.mentor.id, "match_rating": -1},
                                 ignore_index=True)
        for child in self.root.winfo_children():
            child.destroy()

        self.root.quit()
        # self.root.destroy()

    def YES_onclick(self, event=None):
        print("The match between " + self.mentee.name + " and " + self.mentor.name + " was good!")
        self.df = self.df.append({"mentee_id": self.mentee.id, "mentor_id": self.mentor.id, "match_rating": 1},
                                 ignore_index=True)
        for child in self.root.winfo_children():
            child.destroy()

        self.root.quit()
        # self.root.destroy()

    # TODO: Fix issues with saving (probably caused by issue in YES/NO_onclick)
    def export_csv(self):
        self.ds.save_to_disk(df=self.df)
        self.ds.append_matching_info(df=self.df, str_id=self.str_id)

    def kill(self):
        self.root.destroy()

    def get_cache(self):
        cached_matches = {}
        duplicated = {}
        for file in os.listdir("data/match_rating"):
            results = self.ds.load_rating_file(file)
            for i, result in results.iterrows():
                dict_key = str(result["mentor_id"]) + str(result["mentee_id"])

                # If the key already exists
                if dict_key in cached_matches and dict_key not in duplicated:
                    if result["match_rating"] != cached_matches[dict_key]:
                        duplicated[dict_key] = result["match_rating"]
                        del (cached_matches[dict_key])
                        continue

                # If the key does not exist
                cached_matches[dict_key] = result["match_rating"]

        return cached_matches

    def rate_match(self, mentor, mentee, rating):
        self.df = self.df.append({"mentee_id": mentee.id, "mentor_id": mentor.id, "match_rating": rating},
                                 ignore_index=True)


def show_matches(matching_list, str_id):
    df = pd.DataFrame(columns=["mentee_id", "mentor_id", "match_rating"])
    gt = MatchRater(df=df, str_id=str_id)
    cached_matches = gt.get_cache()
    for mentor, mentee, score in matching_list:
        dict_key = str(mentor.id) + str(mentee.id)
        #if dict_key in cached_matches and cached_matches[dict_key] == 1:
            # print("Cached: {}".format(str(mentor.id) + str(mentee.id)))
            # gt.rate_match(mentor, mentee, cached_matches[str(mentor.id) + str(mentee.id)])
           # continue

        gt.show_match(mentor=mentor, mentee=mentee, score=score)
        gt.start()
    else:
        gt.kill()
        gt.export_csv()


# TODO: Where should this be placed?
def report_match_rating():
    filename = "rating_20201005_10_57_08"
    result_df = ds.load_rating_file(filename)
    count_good_matches = 0
    count_bad_matches = 0
    for row in result_df.iterrows():
        actual_row = row[1]
        if actual_row["match_rating"] == 0:
            count_bad_matches += 1
        if actual_row["match_rating"] == 1:
            count_good_matches += 1
    print("MATCH REPORT:\nAnalyzing file " + filename + ".csv\n\nWe made a total of " + str(
        count_good_matches + count_bad_matches) + " of which " + str(count_good_matches) + " were good and " + str(
        count_bad_matches) + " were bad.")


if __name__ == '__main__':
    """
    mentees = []
    mentors = []
    mentee = Mentee(
        name="Mentee name",
        age=0,
        id=1234,
        gender="m",
        linkedin="https://www.linkedin.com/in/christian-bøgelund-92747799/",
        study_line="Cand. Alt",
        university="DTU",
        content=LOREM
    )
    mentees.append(mentee)

    mentor = Mentor(
        name="Nana Bule",
        age=0,
        id=2345,
        gender="m",
        linkedin="https://www.linkedin.com/in/christian-bøgelund-92747799/",
        study_line="Cand. Alt",
        company="Microsoft",
        university="",
        company_type="",
        position="CEO",
        content=LOREM
    )
    mentors.append(mentor)

    mentee = Mentee(
        name="Endnu en mentee",
        age=0,
        id=1234,
        gender="m",
        linkedin="https://www.linkedin.com/in/christian-bøgelund-92747799/",
        study_line="Cand. Alt",
        university="DTU",
        content=LOREM
    )
    mentees.append(mentee)

    mentor = Mentor(
        name="En helt anden mentor, der ikke er Nana",
        age=0,
        id=2345,
        gender="m",
        linkedin="https://www.linkedin.com/in/christian-bøgelund-92747799/",
        study_line="Cand. Alt",
        company="Microsoft",
        university="",
        company_type="",
        position="CEO",
        content=LOREM
    )
    
    mentors.append(mentor)
    """

    # mentees, mentors = zip(*hungarian_main())
    # show_matches(mentees, mentors)

    ds = DataStore()
