import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import string
nltk.download('punkt')
nltk.download('stopwords')

import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec