'''This code example and test was taken from https://www.geeksforgeeks.org/python-nlp-analysis-of-restaurant-reviews/?ref=lbp'''
from typing import Reversible
import numpy as np
import pandas as pd
#Clean data
import re
import nltk


# Read a TAB separated file, declare its separation in 'delimiter' property
dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t")

print(dataset.columns)

#Clean data for easy processing
nltk.download('stopwords')

#remove stop words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#initialize empty array to append clean text
corpus = []

#1000 reviews rows to clean (from dataset)
for i in range(0,1000):
    #get only letters from the review on pos i
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #convert to lower case
    review = review.lower()
    #split to array, default delimiter is space " "
    review = review.split()

    #Create PorterStemmer object, take main stem of each word
    pstem = PorterStemmer()

    #Loop for stemming each word in string array 
    review = [pstem.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    #join all string elements
    review =    ' '.join(review)

    #Append each string to create an array of clean text
    corpus.append(review)

#Creating Bag of words model
from sklearn.feature_extraction.text import CountVectorizer

#Extract 1500 features,make use of the 'max_features' attribute to experiment to get better results
countVector = CountVectorizer(max_features=1500)

#X contains corpus (dependen variable)
X = countVector.fit_transform(corpus).toarray()

#Y contains answer if ewview is positive or regative
Y = dataset.iloc[:,1].values


# ====== Preparing tests ======

#splitting dataset into TRAINING set and TEST set
from sklearn.model_selection import train_test_split

#Experiment with "test_size" for better results
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25)

#Fitting Random Forest Classifier to training set
from sklearn.ensemble import RandomForestClassifier

#n_estimators are the number of trees, experiment with n_estimators for better results
model  = RandomForestClassifier(n_estimators=501, criterion='entropy')
model.fit(x_train,y_train)

# ===== Predicting final results
y_pred = model.predict(x_test)

print("Prediction results")
print(y_pred)

#confusion Matrix
from sklearn.metrics import confusion_matrix

cMatrix = confusion_matrix(y_test, y_pred)
print("Confussion Matrix")
print(cMatrix)








