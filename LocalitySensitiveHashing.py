import hashlib
import re
import struct
from collections import defaultdict

import nltk
import numpy as np
from nltk.corpus import stopwords
from scipy.integrate import quad as integrate


class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = nltk.WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc) if t not in self.ignore_tokens]

def hash_algorithm(data, type="sha1"):
    valid_types = {"md5", "sha1"}
    if type not in valid_types:
        raise ValueError("Result: Type must be one of %s." % valid_types)

    if type == "sha1":
        hash_obj = hashlib.sha1(data).digest()[:4]
    if type == "md5":
        hash_obj = hashlib.md5(data).digest()[:4]

    hash_val = struct.unpack('<I', hash_obj)
    return hash_val[0]

class TextSimilarityWithLSH:
    def __init__(self, texts, baseline, bands, rows):
        self.texts = texts
        self.baseline = baseline
        self.bands = bands
        self.rows = rows

    def create_minhash_signatures(self, baseline, no_permutations):
        stop_words = set(stopwords.words('english'))
        lt = LemmaTokenizer()

        min_hashes = []
        for text in self.texts:
            text = re.sub(r'[^\w\s]','', str(text).lower())
            tokens = lt.__call__(text)

            min_hash = MinHash(no_permutations = no_permutations)

            for shingle in tokens:
                if shingle not in stop_words:
                    min_hash.update(shingle.encode('utf8'))

            min_hashes.append(min_hash)

        lsh = LSH(baseline=baseline, no_permutations=no_permutations, bands=self.bands, rows=self.rows) 

        for key, m in enumerate(min_hashes):
            lsh.add(key, m)

        return min_hashes, lsh

    def query_lsh(self, min_hashes, lsh_signature):
        results = []

        for idx, min_hash in enumerate(min_hashes):
            result = lsh_signature.query_candidates(min_hash)
            results.append(result)

        return results

    def text_similarity_with_lsh(self):
        # TODO: Test with different threshold values here

        min_hashes, lsh_signatures = self.create_minhash_signatures(self.baseline,no_permutations = 128)
        #min_hashes, lsh_signatures = self.create_minhash_signatures(baseline=0.337,no_permutations = 128)
        #min_hashes, lsh_signatures = self.create_minhash_signatures(baseline=0.337,no_permutations = 128)
        #min_hashes, lsh_signatures = self.create_minhash_signatures(baseline=0.337,no_permutations = 128)

        results = self.query_lsh(min_hashes, lsh_signatures)

        # Mentees average length for baseline = 0.3: 17.952853598014887
        # Mentors average length for baseline = 0.3: 3.93617021276595751

        # Mentees average length for baseline = 0.4: 2.518610421836228 
        # Mentors average length for baseline = 0.4: 1.4574468085106382

        # Mentees average length for baseline = 0.337: 7.774193548387097
        # Mentors average length for baseline = 0.337: 2.9893617021276597 
        return results

class MinHash:
    """ 
    This is the class for the MinHash part of the LSH Minhash algorithm.
      
    Attributes: 
        no_permutations (int): No. permutations for the hash functions.
        hash_function (function): Hash function such as md5 or sha.
        hash_values (list): Hash values for the document.
        permutations (list): Random numbers used for universal hashing.
    """
    def __init__(self, no_permutations=128, hash_function=hash_algorithm, hash_values=None, permutations=None):
        # Initialize to function - either md5 or sha1 hash function
        self.MERSENNE_PRIME = (1 << 61) - 1
        self.MAX_HASH = (1 << 32) - 1 
        self.no_permutations = no_permutations
        self.hash_function = hash_algorithm
        self.hash_values = np.ones(self.no_permutations, dtype=np.uint64)*self.MAX_HASH

        randomizer = np.random.RandomState(1)
        
        self.permutations = np.array([(randomizer.randint(1, self.MERSENNE_PRIME, dtype=np.uint64),
                                        randomizer.randint(0, self.MERSENNE_PRIME, dtype=np.uint64))
                                        for _ in range(no_permutations)], dtype=np.uint64).T

        self.hash_function = hash_function

    def update(self, shingle):
        hash_val = self.hash_function(shingle, "md5")
        x,y = self.permutations
        
        phv = np.bitwise_and((x * hash_val + y) % self.MERSENNE_PRIME, np.uint64(self.MAX_HASH))
        self.hash_values = np.minimum(phv, self.hash_values)

class LSH:
    """ 
    This is the class for the LSH part of the LSH Minhash algorithm.
      
    Attributes: 
        baseline (int): Baseline for Jaccard similarity. 
        no_permutations (int): No. permutations for the hash functions.
    """
    def __init__(self, baseline=0.337,no_permutations=128, bands=32, rows=4):
        self.baseline = baseline
        self.no_permutations = no_permutations
        
        #self.bands, self.rows = self.compute_parameters()

        # For n = 128, threshold = 0.2
        self.bands = bands
        self.rows = rows
        # For n = 128, threshold = 0.3
        #self.bands = 32 
        #self.rows = 4
        # For n = 128, threshold = 0.4
        #self.bands = 32
        #self.rows = 4
        # For n = 128, threshold = 0.5
        #self.bands = 16
        #self.rows = 8

        self._H = self.byteswap
        self.signature_matrix = [(i*self.rows, (i+1)*self.rows) for i in range(self.bands)]
        self.hash_tables = [ defaultdict(set) for i in range(self.bands)]

    def add(self, min_hash_key, min_hash):
        Hs = []
        for (start, end) in self.signature_matrix:
            Hs.append(self._H(min_hash.hash_values[start:end]))
        
        for H, bucket in zip(Hs, self.hash_tables):
            bucket[H].add(min_hash_key)

    def byteswap(self, hs):
        return bytes(hs.byteswap().data)

    def query_candidates(self, min_hash):
        candidates = set()

        for (start_range, end_range), bucket in zip(self.signature_matrix, self.hash_tables):
            H = self._H(min_hash.hash_values[start_range:end_range])
            for key in bucket.get(H):
                candidates.add(key)

        return list(candidates)


text1 = "I have always been to go beyond what is expected. This drive stems from my life-long excitement about the developments engineering and science have to offer towards society, and my desire to be at the forefront of such developments. In my BSc Aerospace Engineering (TU Delft), I joined the honours development programme. Herein, I got the opportunity to learn, by doing, how scientific work is conducted. I performed wind tunnel experiments and developed a numerical prediction code for turbulent transition together with a PhD candidate. After graduating from my bachelors, I had a rather unconventional gap year; joined the race team Formula Student Team Delft. In short, we were 70 full-time and part-time student engineers that formed a team to build and race a world championship winning electric race car. Throughout the year, I was responsible for the Aerodynamics Development, CFD, Aerodynamics Production & Chassis Production and Competition Season Strategy. The experience was amazing for numerous reasons besides winning trophies and setting records on world famous circuits. Most notably: I discovered I have a passion for CFD and turbulence modelling in an industrial context as well as using numerical tools and mathematical models to optimize output of a team or process. I chose to study wind energy in the international dual degree programme 'EWEM' (DTU and TU Delft) since it was the perfect blend between a socially relevant cause, my passion for CFD and numerics and an international high quality education. Currently I am preparing to start my MSc thesis on data driven improvements for CFD (RANS) turbulence models for wind turbine / wind farm applications. After graduating, I hope to pursue a career in renewable energy as an engineer specialized in aerodynamics and/or machine learning or optimization techniques. Furthermore, having a minor in management and course experience in mathematical modelling and optimization (next to my experience as strategist in Formula Student Team Delft), I am also very interested in using numerics as a project-/process-management tool."

text2 = "Mie has 5+ years of experience with data analysis, machine learning and mathematical modeling. She spent two years at Red Bull Racing in the UK, where she primarily worked on developing new tire models, data analysis for aerodynamics and various models for race strategy. The common denominator for all of these areas; machine learning. At Ørsted, where she has been for the past 2,5 years, Mie has continued working with machine learning, however, she has also had the opportunity to work with more classical statistical tools and optimization. The focus at Ørsted has been on tool and model development, as well as knowledge sharing, meaning she is also spending a lot of time talking about best practices and explaining complex models to people without the technical expertise. Mie wants to help students structure their last years at university and help them achieve their dreams. She wants to share her own ups and downs in the hope that someone else might avoid her pitfalls and learn from the mistakes she has made. Mie wants to inspire talented students to aim high and help them get the best possible start in their working life." 

text3 = "My interest areas have always spanned extremely broad; ranging from politics and volunteer work, to mathematics and cool innovations in science. This is not only reflected on my past experiences, but also what I see for my future. For me, it is important to gain a holistic set of competencies - I want to have an impact on the world, and since I believe there is an incredible amount of ways to do this, I want to explore and investigate as many options as possible - and with so many working years left, I value the ability to pivot in untraditional directions. Next to my studies, I work as a student developer at Microsoft, where my tasks include everything from hardcore development to management of smaller technical projects. I am also the co-founder of Conflux, which I have helped grow from 23 students to over 300. My passion for leadership and “grass-root” movements emerged in high school, where I was elected as chairman of the student council. A whole new world was later opened up to me when I, in my gap year, obtained a position as an Executive Assistant to the CEO of an IT company, Sonlinc (now acquired by EG), where I was in charge of running a tender response for a government bid, that we ended up winning. This, combined with a backpacking trip across South East Asia, where I met a rich community of remote developers, made me realize what huge possibilities and scalable affect the IT field has on the world and every kind of field; and I wanted to be a part of it. My motivation for joining the program is many-fold: I am close to finishing my bachelor's degree, and am starting to navigate in the many different paths that lie ahead professionally, both during and after my studies. My hope is to find a way to combine my passion for technology and leadership with my moral, political, and ethical compass. But do you sometimes need to put the latter aside, in order to gain the right skills to do so later on? I have considered going into management consulting, but I am still unsure if that’s an environment I would thrive in, or if the skills I could acquire in such an environment would be beneficial to where I want to go in the long-term. I’d like to expand my horizon in terms of what many other options that are out there that I might have missed, and therefore seek the help of a professional that has tried an unconventional path. From a here-and-now perspective, I am also keen on finding ways I can advance in my current position; and how I can emphasize my own strengths to do so, for instance by empowering my negotiation skills."

texts1 = ['It was a great movies', 
               'It was a good movie',
               'I like great and loyal dogs']

texts = texts1

#text_sim = TextSimilarityWithLSH(texts)
#text_sim.text_similarity_with_lsh()