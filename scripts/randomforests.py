#!/usr/bin/env python

import gensim

import numpy as np
import re
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import gc
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import sys
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
import string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.decomposition import TruncatedSVD
from sklearn import datasets
from gensim.test.utils import common_texts
from gensim.sklearn_api import W2VTransformer
 
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

contentdf = datadf['Content'].head(8000)
categorydf = datadf['Category'].head(8000)
contentdf.is_copy = False


print "Creating dictionary..."

vectorizer = CountVectorizer()
voc = vectorizer.fit_transform(contentdf)


print "Applying Bag of words algorithm..."

counter = 0
for t in contentdf:    
    bow = vectorizer.transform([t]).toarray()
    contentdf[counter] = bow[0]
    counter = counter + 1


kf = KFold(n_splits=10)


print "Starting Classification with Random Forrests (Bag of Words) ..."

accuracy = 0
recall = 0
precision = 0

fold = 1
clf=RandomForestClassifier(n_estimators=10, n_jobs = -1)
for train_index, test_index in kf.split(contentdf.tolist()):
    print "Fold Number: " + str(fold)
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = contentdf[train_index].tolist(), contentdf[test_index].tolist()
    y_train, y_test = categorydf[train_index].tolist(), categorydf[test_index].tolist()
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy = (metrics.accuracy_score(y_test, y_pred) / 10) + accuracy
    precision = (precision_score(y_test, y_pred, average='macro') / 10) + precision
    recall =  (recall_score(y_test, y_pred, average='macro') / 10) + recall
    
    fold = fold + 1

print "Accuracy (RF + BOW): " + str(accuracy)
print "Precision (RF + BOW): " + str(precision)
print "Recall (RF + BOW): "  + str(recall)


print "Starting Classification with Random Forrests (SVD) ..."


accuracy = 0
recall = 0
precision = 0

fold = 1


for train_index, test_index in kf.split(contentdf.tolist()):
    print "Fold Number: " + str(fold)
    svd=TruncatedSVD(n_components=1000, random_state=42)
    clf=RandomForestClassifier(n_estimators=10, n_jobs = -1)
    transformer=TfidfTransformer()
    pipelineRF = Pipeline([ ('tfidf', transformer), ('svd',svd), ('clf', clf)])
    X_train, X_test = contentdf[train_index].tolist(), contentdf[test_index].tolist()
    y_train, y_test = categorydf[train_index].tolist(), categorydf[test_index].tolist()
    pipelineRF.fit(X_train,y_train)
    y_pred=pipelineRF.predict(X_test)
    accuracy = (metrics.accuracy_score(y_test, y_pred) / 10) + accuracy
    precision = (precision_score(y_test, y_pred, average='macro') / 10) + precision
    recall =  (recall_score(y_test, y_pred, average='macro') / 10) + recall
    
    fold = fold + 1

print "Accuracy (RF + SVD): " + str(accuracy)
print "Precision (RF + SVD): " + str(precision)
print "Recall (RF + SVD): "  + str(recall)


