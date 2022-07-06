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

def split(data):
  np.random.shuffle(data)
  size = len(data)
  train_len = (0.8*size)
  idx = 0
  train_data = []
  test_data = []
  for elements in data:
    if idx < train_len:
      train_data.append(elements)
    else :
      test_data.append(elements)
    idx += 1
  return train_data, test_data


def svm_classifier(train_data,test_data):
  #Create a svm Classifier
  clf = svm.SVC(kernel='linear') # Linear Kernel

  #Train the model using the training sets
  X_train = []
  y_train = []
  true_test_labels = []
  for lists in train_data:
    freq = []
    for k, v in lists[0].items():
      freq.append(v)
    y_train.append(lists[1])
    X_train.append(freq)
  clf.fit(X_train, y_train)
  #Predict the response for test dataset
  X_test = []
  for lists in test_data:
    freq = []
    for k, v in lists[0].items():
      freq.append(v)
    X_test.append(freq)
    true_test_labels.append(lists[1])
  y_pred = clf.predict(X_test)
  # return predict_labels
  return true_test_labels,y_pred


def knn_classifier(train_data,test_data):
  X_train = []
  y_train = []
  true_test_labels = []
  for lists in train_data:
    freq = []
    for k, v in lists[0].items():
      freq.append(v)
    y_train.append(lists[1])
    X_train.append(freq)
  
  #Predict the response for test dataset
  X_test = []
  for lists in test_data:
    freq = []
    for k, v in lists[0].items():
      freq.append(v)
    X_test.append(freq)
    true_test_labels.append(lists[1])

  knn = KNeighborsClassifier(n_neighbors=3)
    
  #Train the model using the training sets
  knn.fit(X_train, y_train)

  #Predict the response for test dataset
  y_pred = knn.predict(X_test)
  # return predict_labels
  return true_test_labels,y_pred

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

  (truelabelssvm,svmpredictedlabels) = svm_classifier(a,b)
  (truelabelsknn,knnpredictedlabels) = knn_classifier(a,b)
  
if __name__ == "__main__":
  main()