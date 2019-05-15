#!/usr/bin/env python

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
import gc
import numpy as np
import pandas as pd
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sys
import string
import gensim
from sklearn.svm import SVC

######################################################
######################################################

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


######################################################
######################################################

def removepunctuation(s):
    for c in string.punctuation:
        s= s.replace(c,"")
    return s

######################################################
######################################################

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        word = removepunctuation(word)
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)
    
######################################################
######################################################

reload(sys)
sys.setdefaultencoding('utf8')

porter = PorterStemmer()

datadf = pd.read_csv("../data/train_set.csv", sep='\t',encoding='utf-8')

contentdf = datadf['Content'].head(12000)
categorydf = datadf['Category'].head(12000)
contentdf.is_copy = False

print "Starting the Stemming Process... (might take a while) "

counter = 0
for t in contentdf: 
    contentdf[counter] = stemSentence(contentdf[counter])
    counter = counter + 1

wordlist = []

counter = 0
for t in contentdf:    
    wordlist.append(contentdf[counter].split())
    counter = counter + 1

print "Creating dictionary..."

model = gensim.models.Word2Vec(wordlist, size=100)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))


accuracy = 0
recall = 0
precision = 0

fold = 1

clfpipe = Pipeline([
    ("word2vec_vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("random_for", SVC(kernel='linear'))])

kf = KFold(n_splits=10)

print "Starting Classification with SVD (W2V + Stemming)"

for train_index, test_index in kf.split(wordlist):
    print "Fold Number: " + str(fold)
    X_train, X_test = [wordlist[i] for i in train_index], [wordlist[i] for i in test_index]
    y_train, y_test = categorydf[train_index].tolist(), categorydf[test_index].tolist()
    clfpipe.fit(X_train,y_train)
    y_pred=clfpipe.predict(X_test)
    accuracy = (metrics.accuracy_score(y_test, y_pred) / 10) + accuracy
    precision = (precision_score(y_test, y_pred, average='macro') / 10) + precision
    recall =  (recall_score(y_test, y_pred, average='macro') / 10) + recall
    
    fold = fold + 1
    
print y_pred

print "Accuracy: " + str(accuracy)
print "Precision: " + str(precision)
print "Recall: "  + str(recall)


