from re import sub
from gensim.utils import simple_preprocess
import numpy as np
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity


def preprocess(doc):
    # Tokenize, clean up input document string
    if type(doc) is float:
        return ""
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    stopwords = ['the', 'and', 'are', 'a']
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]


class Glove:
    def __init__(self, documents):
        print(type(documents[0]))
        if isinstance(documents[0], list):
            print("It is a list")
            documents = [[" ".join(document)] for document in documents if isinstance(document, list)]

        documents = [str(document) for document in documents]

        self.corpus = [preprocess(document) for document in documents if type(document) is str]
        self.documents = documents

        '''
        Then we create a similarity matrix, that contains the similarity between each pair of words, 
        weighted using the term frequency:
        '''
        # Load the model: this is a big file, can take a while to download and open
        glove = api.load("glove-wiki-gigaword-50")
        self.similarity_index = WordEmbeddingSimilarityIndex(glove)

    def get_score(self, query_string):
        if isinstance(query_string, list):
            query_string = " ".join(query_string)

        query = preprocess(query_string)
        print("Everything has been initialized")
        dictionary = Dictionary(self.corpus + [query])
        tfidf = TfidfModel(dictionary=dictionary)

        # Create the term similarity matrix.
        similarity_matrix = SparseTermSimilarityMatrix(self.similarity_index, dictionary, tfidf)

        '''
        Finally, we calculate the soft cosine similarity between the query and each of the documents. 
        Unlike the regular cosine similarity (which would return zero for vectors with no overlapping terms), 
        the soft cosine similarity considers word similarity as well.
        '''
        # Compute Soft Cosine Measure between the query and the documents.
        # From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
        query_tf = tfidf[dictionary.doc2bow(query)]

        index = SoftCosineSimilarity(
            tfidf[[dictionary.doc2bow(document) for document in self.corpus]],
            similarity_matrix)

        doc_similarity_scores = index[query_tf]

        # Output the sorted similarity scores and documents
        sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
        count = 0
        print("Mentee values: {}".format(query_string))
        for idx in sorted_indexes:
            count += 1
            if count > 10:
                break
            # print(f'{idx} \t {doc_similarity_scores[idx]:0.3f} \t {self.documents[idx]}')

        return doc_similarity_scores


class Glove_new:
    def __init__(self, documents):
        print("Initializing GloVe")
        if isinstance(documents[0], list):
            print("It is a list")
            documents = [[" ".join(document)] for document in documents if isinstance(document, list)]

        documents = [str(document) for document in documents]

        self.corpus = [preprocess(document) for document in documents if type(document) is str]
        self.documents = documents

        '''
        Then we create a similarity matrix, that contains the similarity between each pair of words, 
        weighted using the term frequency:
        '''
        # Load the model: this is a big file, can take a while to download and open
        glove = api.load("glove-wiki-gigaword-50")
        print("Document loaded")
        self.similarity_index = WordEmbeddingSimilarityIndex(glove)
        self.dictionary = Dictionary(self.corpus)
        self.tfidf = TfidfModel(dictionary=self.dictionary)
        print("Model is running")

        # Create the term similarity matrix.
        self.similarity_matrix = SparseTermSimilarityMatrix(self.similarity_index, self.dictionary, self.tfidf)
        print("Everything has been initialized")

    def get_score(self, query_string):
        if isinstance(query_string, list):
            query_string = " ".join(query_string)

        query = preprocess(query_string)

        '''
        Finally, we calculate the soft cosine similarity between the query and each of the documents. 
        Unlike the regular cosine similarity (which would return zero for vectors with no overlapping terms), 
        the soft cosine similarity considers word similarity as well.
        '''
        # Compute Soft Cosine Measure between the query and the documents.
        # From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
        query_tf = self.tfidf[self.dictionary.doc2bow(query)]

        index = SoftCosineSimilarity(
            self.tfidf[[self.dictionary.doc2bow(document) for document in self.corpus]],
            self.similarity_matrix)

        doc_similarity_scores = index[query_tf]

        # Output the sorted similarity scores and documents
        print("Mentee values: {}".format(query_string))

        return doc_similarity_scores


def glove_score_1v1(query_string, documents):
    # query_string = 'Leticia has 3+ years of experience in data science. She has a background in applied mathematics and computer science and currently works as a data scientist at Ørsted. In her work, she builds condition-based algorithms to predict when their offshore wind turbines are going to fail in order to optimize daily operations. Leticia has an international upbringing and has lived in 9 different countries, and she is driven by a great work environment with diversity in the workplace. Leticia wants to become a mentor to help students in their transition to professional life and share their own experiences of studying and working abroad and succeeding as a woman in a male-dominated field. Leticia would prefer a mentee that has ambition and drive, such that she has a better understanding of where he or she wants to go and how she can help in the best way.'
    # documents = ['I would describe myself as being dedicated and curious. I am very interested in data analytics and operations research, specially in connection with logistics and planning. For my Bachelor thesis I did a simulation project with Copenhagen Malmö Port on how to optimise the logistics operations at their container-terminal, which really sparked my interest in this area. I am always interesting in learning new things and I try to take advantage of the great opportunities offered through my studies at DTU - like this mentorship or having the opportunity to go abroad for a semester. Last year I spent a semester at Hong Kong University of Science and Technology which was a big experience both academically and personally. Currently, I am working as a student assistant in Danmarks Nationalbank, and even though it is interesting getting an insight into the financial world and having to apply my skills to a different area, at some time, I would like to try something more related to my studies. I would like to be part of the program to gain more knowledge of what it is like working in the industry as a data analyst or engineer - preferably working with logistics, data analytics or operations research. I know very few engineers outside the academic world at DTU, so I would appreciate a mentor who could share some of their experiences and tips on transitioning from student to professional. I am leaning towards specialising in prescriptive analytics, so I would also be very interested in learning more about how optimisation methods and simulation studies are actually applied to real-world problems. What I hope to achieve as a mentee is to be more prepared for working in the industry and get advice on how to make smart choices regarding my studies. I would also appreciate some advice on whether to take another semester abroad during my Masters or gain more work-experience.',
    # 'My greatest ambition is to leave the world in a better state for humans to experience the quality of life than it was when I entered it. This reason lead me to choose scientific studies - general engineering in Paris at first, and then Applied Mathematics in DTU - in the hope to use technologys leverage for maximum impact. Disclaimer: I am currently not looking for a position as I am to continue working for Tomorrow, the fantastic company I am already working for I nevertheless am very interested to get some insights, from a mentor that went through a similar line of study, into how they decided on starting to work straight away vs continue in the academic world by applying for a PhD. I am also eager to learn more about what it actually means to be a professional "data scientist". How much research/theory is actually useful in day-to-day operations and what level of freedom they can have in their decisions and organisation. I am also curious to learn more about career path for data scientist. The popularity of this position is fairly recent and for this reason, career evolution for a data scientist is still rather obscure to me.']
    # 'I would describe myself as focused, structured and vigorous. My main interest is overall concrete technology. It is from the mixing recipes to the maintaining of old structures to "cure" its sickness. The topic of my bachelor project was about testing the different national and international test methods for alkali silica reactions (ASR). To find out the most optimal methods, to catch that sand and stone which could develop ASR. My master thesis is about testing if mine tailings could be used as a substitute for fly ash, which soon not will be available at the same amount as earlier. In my free time, I have been doing a lot of volunteering. I have been a coach for a handball team for 11-12 year old girls for two years. I learned a lot about coaching, planning and taught the girls to be team players. Further I have been part of the organizing committee for the study start and the council for my study line for three years. Where I further developed my competencies planning, leading and get things done. I usually take the lead when things need to be done, but I dont know if Im suited for management. I hope to get a closer look at "the real life", to get ready when I finish my thesis in January. I want to a mentee to get knowledge about the "life" after university. I would prefer a mentor who works with civil engineering, but a mentor who can taught me difference between consulting and entrepreneur firms, so I can find out what is right for me, would be a nice. I still don\'t know what exactly I can be, but I would appreciate some advice. I hope to achieve a way into the business, which could help me find a job after my thesis.']

    # From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb

    # Preprocess the documents, including the query string
    corpus = [preprocess(document) for document in documents]
    query = preprocess(query_string)

    '''
    Then we create a similarity matrix, that contains the similarity between each pair of words, 
    weighted using the term frequency:
    '''
    # Load the model: this is a big file, can take a while to download and open
    glove = api.load("glove-wiki-gigaword-50")
    similarity_index = WordEmbeddingSimilarityIndex(glove)

    # Build the term dictionary, TF-idf model
    print("Everything has been initialized")
    dictionary = Dictionary(corpus + [query])
    tfidf = TfidfModel(dictionary=dictionary)

    # Create the term similarity matrix.  
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

    '''
    Finally, we calculate the soft cosine similarity between the query and each of the documents. 
    Unlike the regular cosine similarity (which would return zero for vectors with no overlapping terms), 
    the soft cosine similarity considers word similarity as well.
    '''
    # Compute Soft Cosine Measure between the query and the documents.
    # From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
    query_tf = tfidf[dictionary.doc2bow(query)]

    index = SoftCosineSimilarity(
        tfidf[[dictionary.doc2bow(document) for document in corpus]],
        similarity_matrix)

    doc_similarity_scores = index[query_tf]

    # Output the sorted similarity scores and documents
    sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
    count = 0
    print("Mentee values: {}".format(query_string))
    for idx in sorted_indexes:
        count += 1
        if count > 10:
            break
        print(f'{idx} \t {doc_similarity_scores[idx]:0.3f} \t {documents[idx]}')
    return doc_similarity_scores
