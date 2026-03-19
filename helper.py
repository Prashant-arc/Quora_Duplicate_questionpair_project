import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import warnings
import pickle
import spacy
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import distance
from scipy.spatial.distance import cosine, cityblock
import warnings
nlp = spacy.load('en_core_web_md')

q_count_dict = pickle.load(open('q_count_dict.pkl', 'rb'))

warnings.filterwarnings('ignore') 

def preprocess(q):

    q = str(q).lower().strip()

    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()



    return q


def test_fetch_token_feature(q1,q2):
    safe_div = 0.001

    STOP_WORDS = stopwords.words("english")

    token_features = [0.0]*8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if(len(q1_tokens)==0 or len(q2_tokens)==0):
        return token_features
    
    #get the non stop words from question
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    #get the stopword in question 
    q1_stops = set([word for word in q1_tokens if word  in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word  in STOP_WORDS])

    #get the common nonstop word from question pair 
    common_word_count = len(q1_words & q2_words)

    #get the common stop words from questio pair
    common_stop_count = len(q1_stops & q2_stops)

    #common tokens
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words),len(q2_words)) + safe_div)
    token_features[1] = common_word_count / (max(len(q1_words),len(q2_words)) + safe_div)
    token_features[2] = common_stop_count / (min(len(q1_stops),len(q2_stops)) + safe_div)
    token_features[3] = common_stop_count / (max(len(q1_stops),len(q2_stops)) + safe_div)
    token_features[4] = common_token_count / (min(len(q1_tokens),len(q2_tokens)) + safe_div)
    token_features[5] = common_token_count / (max(len(q1_tokens),len(q2_tokens)) + safe_div)

    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])


    return token_features

def test_fetch_fuzzy_features(q1,q2):

    fuzzy_features = [0.0]*4

     # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features

def test_fetch_length_features(q1,q2):
    length_features = [0.0]*3

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if(len(q1_tokens)==0 or len(q2_tokens)==0):
        return length_features
    
    length_features[0] = abs( len(q1_tokens) - len(q2_tokens))

    length_features[1] = (len(q1_tokens) + len(q2_tokens)) /2.0
    #longest substring ratio
    strs = list(distance.lcsubstrings(q1, q2))
    if strs: # Check if strs is not empty
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    else:
        length_features[2] = 0.0 

    return length_features


def test_common_words(q1,q2):
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())

    common = w1 & w2 

    return len(common) 

def test_total_words(q1,q2):
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())

    return len(w1) + len(w2)


from scipy.spatial.distance import cosine,cityblock

def query_point_creator(q1, q2):
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    
    # 1.Frequency
    f1 = q_count_dict.get(q1, 0)
    f2 = q_count_dict.get(q2, 0)

    input_query = []

    # order matters 
    input_query.append(len(q1))           # q1len
    input_query.append(len(q2))           # q2len
    input_query.append(f1)                # freq_qid1
    input_query.append(f2)                # freq_qid2
    input_query.append(len(q1.split()))   # q1_n_words
    input_query.append(len(q2.split()))   # q2_n_words
    input_query.append(f1 + f2)           # freq_q1+q2
    input_query.append(abs(f1 - f2))      # freq_q1-q2
    
    common_words = test_common_words(q1, q2)
    total_words = test_total_words(q1, q2)
    input_query.append(total_words)       # word_Total
    input_query.append(round(common_words / (total_words + 0.001), 2)) # word_share

    # ADVANCED FEATURES
    input_query.extend(test_fetch_token_feature(q1, q2))
    input_query.extend(test_fetch_fuzzy_features(q1, q2))
    input_query.extend(test_fetch_length_features(q1, q2))

    #vectors 
    doc1 = nlp(q1)
    doc2 = nlp(q2)

    v1 = np.nan_to_num(doc1.vector)
    v2 = np.nan_to_num(doc2.vector)

    cos_dist = cosine(v1,v2)
    city_dist = cityblock(v1,v2)

    input_query.append(cos_dist)    
    input_query.append(city_dist) 

    input_query.extend(v1)
    input_query.extend(v2)

    return np.array(input_query).reshape(1, -1)