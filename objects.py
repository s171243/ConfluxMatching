class Person:
    def __init__(self, name, age, id, gender, linkedin, university, study_line,
                 content):
        # TODO: Add common variables that mentor and mentee shares
        self.name = name
        self.age = age
        self.id = id
        self.gender = gender
        self.linkedin = linkedin
        self.university = university
        self.study_line = study_line
        self.content = content
        self.priorities = []
        self.former_match = None

    def __str__(self):
        return str(self.id) + ": " + self.name

class Mentee(Person):
    def __init__(self, name, age, id, gender, linkedin, university, study_line,
                 content, remote, semester):
        super().__init__(name, age, id, gender, linkedin, university, study_line,
                         content)
        self.language = ""
        self.semester = ""
        self.industry = self.set_field(study_line)
        self.position = ""
        self.remote = str(remote)
        self.semester = semester


    def set_field(self, education):
        field_dict = {
            "Construction & Buildings": ["construction", "civil", "architectural", "structural", "environmental",
                                         "petroleum", "byggeri"],
            "Data Science, Analytics & AI": ["data", "artificial intelligence", "ai", "machine learning", "mathematical", "mathematics", "analytics"],
            "Energy & Wind Energy": ["sustainable", "energy", "wind"],
            "IT, Software & Electronics": ["software", "electronic", "electro", "math", "digital", "electrical",
                                           "computer", "game", "informatics",
                                           "acoustics", "automation", "games", "telecommunication", "information"],
            "Logistics & Supply Chain Management": ["strategic", "transport", "logistics", "management"],
            "Management Consulting, Business & Executives": ["business", "management", "sales", "economics",
                                                             "industrial"],
            "Manufacturing, Process & Production": ["innovation", "process", "material", "autonomous", "mechanical",
                                                    "polymer", "produktion", "aquatic", "maskin", "maritime",
                                                    "production",
                                                    "marine"],
            "Pharmaceuticals, Chemical, Healthcare & Life Science": ["biotechnology", "biochemistry", "healthcare", "life science", "chemical", "biochemical", "physics", "chemistry",
                                                                     "biomedical", "food", "bioinformatics", "biology",
                                                                     "medicine", "pharmaceutical", "environmental", "disease"],
            "Product Design, Development & UX / UI": ["design", "innovation"],
        }

        result = ""
        for key, array in field_dict.items():
            for value in array:
                if value in str(education).lower():
                    result = key

        self.industry = result


class Mentor(Person):
    def __init__(self, name, age, id, gender, linkedin, university, study_line, company, company_type, position,
                 content, role):
        super().__init__(name, age, id, gender, linkedin, university, study_line,
                         content)
        self.university_preference = ""
        self.language_preference = ""
        self.years_of_experience = ""
        self.industry = []
        self.former_match = ""
        self.position = position
        self.company = company
        self.role = str(role)

    def set_industry(self, categories: str):
        if type(categories) != str:
            return
        categories = categories.replace("&amp;", "&")
        category_list = categories.split(";")
        self.industry = category_list

