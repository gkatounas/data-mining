#!/usr/bin/env python

import numpy as np
import re
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
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.decomposition import TruncatedSVD
from sklearn import datasets
from sklearn.svm import SVC

datadf = pd.read_csv("../data/train_set.csv", sep='\t')

contentdf = datadf['Content'].head(3000)
categorydf = datadf['Category'].head(3000)
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


print "Starting Classification with SVM (Bag of Words) ..."

accuracy = 0
recall = 0
precision = 0

fold = 1
clf = SVC(kernel='linear') 
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

print "Accuracy (SVM  + BOW): " + str(accuracy)
print "Precision (SVM  + BOW): " + str(precision)
print "Recall (SVM  + BOW): "  + str(recall)


print "Starting Classification with SVM (SVD) ..."


accuracy = 0
recall = 0
precision = 0

fold = 1


for train_index, test_index in kf.split(contentdf.tolist()):
    print "Fold Number: " + str(fold)
    svd=TruncatedSVD(n_components=1000, random_state=42)
    clf=SVC(kernel='linear')
    #transformer=TfidfTransformer()
    pipelineRF = Pipeline([ ('svd',svd), ('clf', clf)])
    X_train, X_test = contentdf[train_index].tolist(), contentdf[test_index].tolist()
    y_train, y_test = categorydf[train_index].tolist(), categorydf[test_index].tolist()
    pipelineRF.fit(X_train,y_train)
    y_pred=pipelineRF.predict(X_test)
    accuracy = (metrics.accuracy_score(y_test, y_pred) / 10) + accuracy
    precision = (precision_score(y_test, y_pred, average='macro') / 10) + precision
    recall =  (recall_score(y_test, y_pred, average='macro') / 10) + recall
    
    fold = fold + 1

print "Accuracy (SVM + SVD): " + str(accuracy)
print "Precision (SVM + SVD): " + str(precision)
print "Recall (SVM + SVD): "  + str(recall)

