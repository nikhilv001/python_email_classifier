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

  
def get_bow(words,vocabulary):
  email_bow = {}
  for w in vocabulary:
    email_bow.update({w:0})
  for elements in words:
    if email_bow[elements] == 0:
      email_bow[elements] += 1
  return email_bow

def read_data():
    results = []
    with open('spam_or_not_spam.csv', 'r', encoding='utf8') as f:
        for line in f:
                words = line.split(',')
                results.append([words[0], words[1][0][0]])
    spam_or_not_spam = results
    spam_or_not_spam.pop(0)
    list_of_stem_words = []
    list_of_stem_words_with_label = []
    data = []
    for mail in spam_or_not_spam:
        stem_words = remove_stop_words(stemming(read_email(mail[0])))
        list_of_stem_words.append(stem_words)
        list_of_stem_words_with_label.append([stem_words,int(mail[1])])
    vocabulary = build_vocabulary(list_of_stem_words)
    for w in list_of_stem_words_with_label:
      data.append([get_bow(w[0],vocabulary),int(w[1])])
    return data,list_of_stem_words_with_label


# visuallze data distribution
def data_vis(data,list_of_stem_words_with_label):
  data_visualization_0 = {}
  data_visualization_1 = {}
  data_visualization = {}
  for elements in data:
    for k in elements[0]:
      data_visualization.update({k:0})
      data_visualization_0.update({k:0})
      data_visualization_1.update({k:0})
    break
  for elements in list_of_stem_words_with_label:
    for k in elements[0]:
      data_visualization[k] += int(1)
      if elements[1] == 0 :
        data_visualization_0[k] += int(1)
      if elements[1] == 1 :
        data_visualization_1[k] += int(1)
  plt.bar(list(data_visualization.keys()), data_visualization.values(), color='g')
  plt.show()
  plt.bar(list(data_visualization_0.keys()), data_visualization_0.values(), color='g')
  plt.show()
  plt.bar(list(data_visualization_1.keys()), data_visualization_1.values(), color='g')
  plt.show()
  return


def main():
  data = []
  (data,list_of_stem_words_with_label) = read_data()
  (a,b) = split(data)

if __name__ == "__main__":
  main()