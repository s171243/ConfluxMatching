import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from sklearn.preprocessing import StandardScaler

from sklearn import metrics
import numpy as np
from sklearn.cluster import DBSCAN

import DataStore

class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = nltk.WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc) if t not in self.ignore_tokens]

class SimilarityCalc:
    def __init__(self):
        # Download stopwords list
        self.stop_words = set(stopwords.words('english'))

        # Lemmatize the stop words
        self.tokenizer = LemmaTokenizer()
        self.token_stop = self.tokenizer(' '.join(self.stop_words))

        # Create TF-idf model
        self.vectorizer = TfidfVectorizer(stop_words=self.token_stop,
                                          tokenizer=self.tokenizer)

    def content_similarity(self, string1, string2):
        doc_vectors = self.vectorizer.fit_transform([string1] + [string2])

        # Calculate similarity
        cosine_similarities = linear_kernel(doc_vectors[0:1], doc_vectors).flatten()
        document_scores = [item.item() for item in cosine_similarities[1:]]
        return document_scores
        # print("This is the doc score:")
