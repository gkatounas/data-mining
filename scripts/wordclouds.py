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


###############################################################
###############################################################

def word_cloud(contentdf, categorydf):
    
    listBusiness = []
    listFilm = []
    listFootball = []
    listTechnology = []
    listPolitics = []
    
    l = len(contentdf)-1
    while(l >= 0):
	    if(categorydf[l] == 'Business'):
		    listBusiness.append(str(contentdf[l]))
		    
	    elif(categorydf[l] == 'Film'):
		    listFilm.append(str(contentdf[l]))
		    
	    elif(categorydf[l] == 'Technology'):
		    listTechnology.append(str(contentdf[l]))
		    
	    elif(categorydf[l] == 'Politics'):
		    listPolitics.append(str(contentdf[l]))
		    
	    else:
		    listFootball.append(str(contentdf[l]))
	    l=l-1
    #end while
    
    
    textBusiness = str(listBusiness)
    textPolitics = str(listPolitics)
    textTechnology = str(listTechnology)
    textFilm = str(listFilm)
    textFootball = str(listFootball)
    
    #create stopwords tuple
    stopfile = open("../data/stopwords.txt","r")
    more_stopwords = stopfile.readlines()
    more_stopwords = map(lambda s: s.strip(), more_stopwords)
    special_stopwords = ('x80' , 'xe2' , 'x9d','x93','x99','x99s','xc2','us','x99t','xa6','xa0','xa9','xc3','x9cThe','x9cWe','x9cI','x9cIt','x99re','ve','x9d\'','Su xa1rez','ll','i0')
    stopfile.close()
    more_stopwords.extend(special_stopwords)
    more_stopwords_tuple = tuple(more_stopwords)
    stop_words = STOPWORDS.union(more_stopwords_tuple)
    
    #create the wordclouds
    print "Creating Business WordCloud..."
    businessWordCloud = WordCloud(stopwords=stop_words, background_color='white', width=1200, height=1000).generate(textBusiness)
    name = '../output/wordclouds/BusinessWordcloud.png'
    plt.imshow(businessWordCloud)
    plt.title("BusinessWordcloud")
    businessWordCloud.to_file(name)
    plt.axis('off')
    
    print "Creating Politics WordCloud..."
    politicsWordCloud = WordCloud(stopwords=stop_words, background_color='white', width=1200, height=1000).generate(textPolitics)
    name = '../output/wordclouds/PoliticsWordcloud.png'
    plt.imshow(politicsWordCloud)
    plt.title("PoliticsWordcloud")
    politicsWordCloud.to_file(name)
    plt.axis('off')
    
    print "Creating Football WordCloud..."
    footballWordCloud = WordCloud(stopwords=stop_words, background_color='white', width=1200, height=1000).generate(textFootball)
    name = '../output/wordclouds/FootballWordcloud.png'
    plt.imshow(footballWordCloud)
    plt.title("FootballWordCloud")
    footballWordCloud.to_file(name)
    plt.axis('off')
    
    print "Creating Film WordCloud..."
    filmWordCloud = WordCloud(stopwords=stop_words, background_color='white', width=1200, height=1000).generate(textFilm)
    name = '../output/wordclouds/FilmWordCloud.png'
    plt.imshow(filmWordCloud)
    plt.title("FilmWordCloud")
    filmWordCloud.to_file(name)
    plt.axis('off')
    
    print "Creating Technology WordCloud..."
    technologyWordCloud = WordCloud(stopwords=stop_words, background_color='white', width=1200, height=1000).generate(textTechnology)
    name = '../output/wordclouds/TechnologyWordCloud.png'
    plt.imshow(technologyWordCloud)
    plt.title("TechnologyWordCloud")
    technologyWordCloud.to_file(name)
    plt.axis('off')

###############################################################
###############################################################

def main():

    reload(sys)
    sys.setdefaultencoding('utf8')

    datadf = pd.read_csv("../data/train_set.csv", sep='\t',encoding='utf-8')
    
    contentdf = datadf['Content']
    categorydf = datadf['Category']
    titledf = datadf['Title']
    iddf = datadf['Id']
    
    word_cloud(contentdf, categorydf)

if __name__== "__main__":
  main()
