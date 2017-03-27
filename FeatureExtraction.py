__author__ = 'Alexander'
from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from Utility import *

data = load_dict("DataCleared"+str(0))
data.head()
train, test = train_test_split(data, test_size=0.2, random_state=42) #randomly divide full set to training and test parts
print('data loaded')

###Define parameters
n_features = 1500
use_hashing = False
select_chi2 = 500
use_variance = True

y_train, y_test =train['Label'], test['Label']
# train_text = []
# test_text = []
# for each in train['Cleared']:
#     train_text.append(to_words(each, {}))
# print(train_text)

# for each in test['Cleared']:
#     test_text.append(to_words(each, {}))
# print(test_text)
print("Extracting features from the training data using a sparse vectorizer")
if use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=n_features)
    X_train = vectorizer.transform(train['Cleared'])
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(train['Cleared'])
print("n_samples: %d, n_features: %d" % X_train.shape)
print()
# print(X_train)
print("Extracting features from the test data using the same vectorizer")
X_test = vectorizer.transform(test['Cleared'])
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

if use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()
len(feature_names)

# if use_variance:
#     print("Extracting features by Variance trashhold")
#     sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#     result = sel.fit_transform(trainheadlines)
if select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

if feature_names:
    feature_names = np.asarray(feature_names)
len(feature_names)
print(feature_names)