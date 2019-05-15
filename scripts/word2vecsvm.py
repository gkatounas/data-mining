#!/usr/bin/env python

import gensim
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
 
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


datadf = pd.read_csv("../data/train_set.csv", sep='\t')

print "Starting Classification with SVM (W2V) ..."
print "Creating Vectors..."

wordlist = []
counter = 0

contentdf2 = datadf['Content'].head(12000)
categorydf2 = datadf['Category'].head(12000)

for t in contentdf2:    
    wordlist.append(contentdf2[counter].split())
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
    ("svm", SVC(kernel='linear'))])

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(wordlist):
    print "Fold Number: " + str(fold)
    X_train, X_test = [wordlist[i] for i in train_index], [wordlist[i] for i in test_index]
    y_train, y_test = categorydf2[train_index].tolist(), categorydf2[test_index].tolist()
    clfpipe.fit(X_train,y_train)
    y_pred=clfpipe.predict(X_test)
    accuracy = (metrics.accuracy_score(y_test, y_pred) / 10) + accuracy
    precision = (precision_score(y_test, y_pred, average='macro') / 10) + precision
    recall =  (recall_score(y_test, y_pred, average='macro') / 10) + recall
    
    fold = fold + 1

print "Accuracy (SVM + W2V): " + str(accuracy)
print "Precision (SVM + W2V): " + str(precision)
print "Recall (SVM + W2V): "  + str(recall)
