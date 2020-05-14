#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:11:47 2020

@author: ornwipa
"""

import os
import re
import string
import nltk
# nltk.download('stopwords')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression #, SGDClassifier

def readText():
    ''' read text file, one file per one item in the list '''
    extractedText = []
    filepath = 'text'
    for filename in os.listdir(filepath):
        file = open(filepath + '/' + filename, 'r')
        extractedText.append(file.read())
        file.close()
    return extractedText 

def splitParagraphs(text):
    ''' \n is a line break as read (not new paragraph), so should be kept;
    \n\n is a new paragraph, so will be used as splitting indicator;
    \n\n\n is a page break in the book (not new paragraph), so will be together '''
    text = text.replace('\n\n\n', ' ') # connect the page break as entire text
    textList = []
    beg_index = 1
    for char in range(len(text) - 2):
        if text[char] == '\n' and text[char+1] == '\n':
            end_index = char
            textList.append(text[beg_index:end_index])
            beg_index = char + 2
        if beg_index > len(text):
            break        
    return textList # total of 221 paragraphs in the book

def processText(text):
    ''' text is a string type, make all characters lowercase, 
    remove '\n' of new paragraph (not necessary for one paragraph text),
    remove all punctuations, non-alphabets, and extra space '''
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('','',string.punctuation))
    text = re.sub(' +', ' ', text)
    return text

def extractFeaturesLabels(text):
    ''' this function is specific to the book L’ÉTRANGER,
    split between PREMIÈRE PARTIE et DEUXIÈME PARTIE to capture X,
    assign y = 0 for PREMIÈRE PARTIE, y = 1 for DEUXIÈME PARTIE,
    return pandas dataframe of X and y '''
    for i in range(len(text)-1):
        tokens = text[i].split()
        j = 0
        while j < len(tokens):
            if tokens[j] == 'première' and tokens[j+1] == 'partie':
                beg1 = i + 1
                break
            else:
                j += 1
        j = 0
        while j < len(tokens):
            if tokens[j] == 'deuxième' and tokens[j+1] == 'partie':
                end1 = i
                beg2 = i + 1
                break
            else:
                j += 1
        if text[i] == 'texte libre de droits':
            end2 = i - 1   
            break
    # print(beg1, end1, beg2, end2)
    splitedText = text[beg1:(end2+1)] # total of 204 paragraphs
    label = [0]*(beg2-beg1) + [1]*(end2-end1)
    out = pd.DataFrame([splitedText, label])
    # out = out.T
    # out.columnss = ['texte', 'partie']
    return out.T

def selectFeatures():
    ''' text here is a list of paragraph for creating ... 
    ... sparse matrix of (n_samples, n_features) '''
    arrêt = set(nltk.corpus.stopwords.words('french')) # 'stopwords' en français
    arrêt = arrêt.union({'avoir', 'être', 'ça', 'cela', 'cet', 'cette'})
    ''' 'buid ngram model: unigram, bigram, trigram '''
    vectorizer = TfidfVectorizer(ngram_range = (1, 3), 
                                 min_df = 0.0001, max_df = 0.95, 
                                 max_features = 1000, 
                                 strip_accents = None,                                                                                
                                 use_idf = True, smooth_idf = True, 
                                 sublinear_tf = True, 
                                 stop_words = arrêt)
    return vectorizer

def main():
    ''' pre-process input text '''
    textList = readText()
    for text in textList:
        paragraphs = splitParagraphs(text)
    processedText = []
    for paragraph in paragraphs:
        processedText.append(processText(paragraph)) # corpus without label
    ''' extract ngram features and labels '''
    dataset = extractFeaturesLabels(processedText)
    y = dataset[1] # dataset['partie']
    X = dataset[0] # selectFeatures(dataset['texte'])
    ''' split training and testing dataset '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    ''' select features and train model '''
    vectorizer = selectFeatures()
    X_train = vectorizer.fit_transform(X_train)
    print(vectorizer.get_feature_names())
    # print(vectorizer.get_stop_words())
    # print(X_train.shape) # dimension = (153, 300)
    clf = LogisticRegression() # SGDClassifier(loss = "hinge", penalty = "l2")
    clf.fit(X_train, list(y_train))
    ''' predict outcomes and test model '''
    X_test = vectorizer.transform(X_test)
    # y_predicted = clf.predict(X_test)
    print(clf.score(X_test, list(y_test)))
        
    
if __name__ == "__main__":
    main()
    