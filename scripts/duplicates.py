#!/usr/bin/env python

import gc
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import sys
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import svm


lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))



datadf = pd.read_csv("../data/train_set.csv", sep='\t')

contentdf = datadf['Content']
categorydf = datadf['Category']
titledf = datadf['Title']
iddf = datadf['Id']

csvfile = open('../output/duplicates.csv', 'w')
titles = "Document_ID1\tDocument_ID2\tSimilarity\n"
csvfile.write(titles)

listBusiness = []
listBusinessId = []
listFilm = []
listFilmId = []
listFootball = []
listFootballId = []
listTechnology = []
listTechnologyId = []
listPolitics = []
listPoliticsId = []

l = len(contentdf)-1
while(l >= 0):
    if(categorydf[l] == 'Business'):
	    listBusiness.append(str(contentdf[l]))
	    listBusinessId.append(str(iddf[l]))
	    
    elif(categorydf[l] == 'Film'):
	    listFilm.append(str(contentdf[l]))
	    listFilmId.append(str(iddf[l]))
	    
    elif(categorydf[l] == 'Technology'):
	    listTechnology.append(str(contentdf[l]))
	    listTechnologyId.append(str(iddf[l]))
	    
    elif(categorydf[l] == 'Politics'):
	    listPolitics.append(str(contentdf[l]))
	    listPoliticsId.append(str(iddf[l]))
	    
    else:
	    listFootball.append(str(contentdf[l]))
	    listFootballId.append(str(iddf[l]))
	    
    l=l-1
#end while


LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(listBusiness)

tf_matrix = LemVectorizer.transform(listBusiness).toarray()

tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(tf_matrix)

tfidf_matrix = tfidfTran.transform(tf_matrix)

cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

x = 0
y = 0

for lis in  cos_similarity_matrix:
    y = 0
    for value in lis:
        if value > 0.7 and x != y:
            res = str(listBusinessId[x]) + "\t" + str(listBusinessId[y]) + "\t" + str(value) + "\n"
            csvfile.write(res)
            
        y = y + 1
    x = x + 1
    #end_for
#end_for

LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(listFilm)

tf_matrix = LemVectorizer.transform(listFilm).toarray()

tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(tf_matrix)

tfidf_matrix = tfidfTran.transform(tf_matrix)

cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

x = 0
y = 0

for lis in  cos_similarity_matrix:
    y = 0
    for value in lis:
        if value > 0.7 and x != y:
            res = str(listFilmId[x]) + "\t" + str(listFilmId[y]) + "\t" + str(value) + "\n"
            csvfile.write(res)
            
        y = y + 1
    x = x + 1
    #end_for
#end_for

LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(listFootball)

tf_matrix = LemVectorizer.transform(listFootball).toarray()

tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(tf_matrix)

tfidf_matrix = tfidfTran.transform(tf_matrix)

cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

x = 0
y = 0

for lis in  cos_similarity_matrix:
    y = 0
    for value in lis:
        if value > 0.7 and x != y:
            res = str(listFootballId[x]) + "\t" + str(listFootballId[y]) + "\t" + str(value) + "\n"
            csvfile.write(res)
            
        y = y + 1
    x = x + 1
    #end_for
#end_for

LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(listPolitics)

tf_matrix = LemVectorizer.transform(listPolitics).toarray()

tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(tf_matrix)

tfidf_matrix = tfidfTran.transform(tf_matrix)

cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

x = 0
y = 0

for lis in  cos_similarity_matrix:
    y = 0
    for value in lis:
        if value > 0.7 and x != y:
            res = str(listPoliticsId[x]) + "\t" + str(listPoliticsId[y]) + "\t" + str(value) + "\n"
            csvfile.write(res)
            
        y = y + 1
    x = x + 1
    #end_for
#end_for


LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(listTechnology)

tf_matrix = LemVectorizer.transform(listTechnology).toarray()

tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(tf_matrix)

tfidf_matrix = tfidfTran.transform(tf_matrix)

cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

x = 0
y = 0

for lis in  cos_similarity_matrix:
    y = 0
    for value in lis:
        if value > 0.7 and x != y:
            res = str(listTechnologyId[x]) + "\t" + str(listTechnologyId[y]) + "\t" + str(value) + "\n"
            csvfile.write(res)
            
        y = y + 1
    x = x + 1
    #end_for
#end_for

csvfile.close()
#print cos_similarity_matrix


###############################################################
###############################################################
