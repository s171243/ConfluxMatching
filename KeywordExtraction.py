import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords

def pre_process(text):
    # lowercase
    text = text.lower()

    # remove tags
    text = re.sub("</?.*?>", " <> ", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    return text


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = []
    for idx in range(len(feature_vals)):
        results.append(feature_vals[idx])

    return results

def get_keyword_list(mentees):
    docs = []

    for mentee in mentees:
        docs.append(str(mentee.content))

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

    # print(stopwords)
    stop_words = set(stopwords.words('english'))
    #print(stop_words)
    cv = CountVectorizer(max_df=0.85, stop_words=stop_words)
    word_count_vector = cv.fit_transform(docs)
    feature_names = cv.get_feature_names()
    tfidf_transformer.fit(word_count_vector)

    #print("Ready to parse profiles")

    all_keywords = []

    for doc, mentee in zip(docs, mentees):
        doc = pre_process(doc)

        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

        # sort the tf-idf vectors by descending order of scores
        sorted_items = sort_coo(tf_idf_vector.tocoo())

        # extract only the top n; n here is 10
        keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
        all_keywords.append(keywords)
        #print("Mentee {} has words {}".format(mentee.id, keywords))

    return all_keywords
