import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


def read_email(email):                   # function to read email and return list of words
  words = email.split()
  return words
  
def stemming(words):                    # function to return stem words for every word in the list of words
    ps = PorterStemmer()
    stem_words = []
    for w in words: 
        stem_words.append(ps.stem(w))
    return stem_words


def remove_stop_words(words):                       #removing stop words from the list of words
    stop_words = set(stopwords.words('english'))
    stem_no_stop_words = []
    for r in words:
        if not r in stop_words:
            stem_no_stop_words.append(r)
    return stem_no_stop_words

def build_vocabulary(list):                         # building the vocabulary from all the emails
  vocabulary = []
  current_vocabulary = set()
  for words in list:
    for w in words:
      if not w in current_vocabulary:
        current_vocabulary.add(w)
        vocabulary.append(w)
  return vocabulary