from ggplot import ggplot, aes, geom_line, geom_abline, ggtitle

__author__ = 'Alexander'
import pandas as pd
import numpy as np
import os
import pylab as pl
import sklearn
import matplotlib.pyplot as plt
import re
import nltk
import matplotlib

from sklearn.cross_validation import train_test_split
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from time import time

matplotlib.rcParams["figure.figsize"] = "8, 8"
print(os.getcwd())
os.chdir(os.getcwd()+"\TitlesAndStockMarket")

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import metrics

from sklearn.datasets import fetch_20newsgroups #look what is it

#load data
data = pd.read_csv('C:/Users/Alexander/Documents/GitHub/MyProjects/PythonProjects/TitlesAndStockMarket/resources/Combined_News_DJIA.csv')
data.head()

# train = data[data['Date'] < '2015-01-01']
# test = data[data['Date'] > '2014-12-31']
# print(train.head())

import pickle
def save_dict(obj, name):
    with open(name+'.pkl','wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name+'.pkl','rb') as file:
        return pickle.load(file)

#use this to load words
dic = load_dict("SortedWords")

#example to count words and make data frame
# example = []
# # t0 = time()
# tokenizer = CountVectorizer().build_tokenizer()
# for row in range(0, len(data.index)):
#     words = ' '.join(str(x) for x in data.iloc[row, 2:27])
#     # print("done in %0.3fs." % (time() - t0))
#     example += tokenizer(words)
# print("done for")
# words = set(example)
# print("done set")

#need 20 min to calculate for all data better to use pickle functions...
# dic = []
# t0 = time()
# for x in words:
#     dic.append([x, example.count(x)])
#     print("done in %0.3fs." % (time() - t0))
# print("done dict")
# save_dict(dic, "SortedWords")

df = pd.DataFrame(dic, columns=['Word', 'Count'])
print(df.head())
df = df.sort_values(by=['Count'], ascending=False)
df = df[df['Count'] > 5] #delete small variable

stops = set(stopwords.words("english"))
print(stops)
def to_words(content):
    letters_only = re.sub("[^a-zA-Z]", " ", content)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words ))

# print("Loading dataset...")
# t0 = time()
# dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
# data_samples = dataset.data[:n_samples]
# print("done in %0.3fs." % (time() - t0))

#train different models from kaggle
data = pd.read_csv('C:/Users/Alexander/Documents/GitHub/MyProjects/PythonProjects/TitlesAndStockMarket/resources/Combined_News_DJIA.csv')
data['Combined']=data.iloc[:, 2:27].apply(lambda row: ''.join(str(row.values)), axis=1)
train, test = train_test_split(data,test_size=0.2,random_state=42)
#how to get proportions 0 and 1
non_decrease = train[train['Label']==1]
decrease = train[train['Label']==0]
print(len(non_decrease)/len(df))
#full baskets
non_decrease_word=[]
decrease_word=[]
for each in non_decrease['Combined']:
    non_decrease_word.append(to_words(each))
for each in decrease['Combined']:
    decrease_word.append(to_words(each))
#create two clouds with positive and negative words
wordcloud1 = WordCloud(background_color='black',
                      width=3000,
                      height=2500
                     ).generate(decrease_word[0])
plt.figure(1,figsize=(8,8))
plt.imshow(wordcloud1)
plt.axis('off')
plt.show()

wordcloud2 = WordCloud(background_color='white',
                      width=3000,
                      height=2500
                     ).generate(non_decrease_word[0])
plt.figure(1,figsize=(8,8))
plt.imshow(wordcloud2)
plt.axis('off')
plt.show()

#train ml models
tfidf=TfidfVectorizer()
train_text = []
test_text = []
for each in train['Combined']:
    train_text.append(to_words(each))

for each in test['Combined']:
    test_text.append(to_words(each))
train_features = tfidf.fit_transform(train_text)
test_features = tfidf.transform(test_text)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from ggplot import *


Classifiers = [
    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    # KNeighborsClassifier(3),
    # SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]

dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['Label'])
        pred = fit.predict(test_features)
        prob = fit.predict_proba(test_features)[:,1]
    except Exception:
        fit = classifier.fit(dense_features,train['Label'])
        pred = fit.predict(dense_test)
        prob = fit.predict_proba(dense_test)[:,1]
    accuracy = accuracy_score(pred,test['Label'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))
    fpr, tpr, _ = roc_curve(test['Label'],prob)
    tmp = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    # plt.plot(tmp['fpr'], tmp['tpr'])
    # plt.title('Roc Curve of '+classifier.__class__.__name__)
    # plt.show()
    g = ggplot(tmp, aes(x='fpr', y='tpr')) +geom_line() +geom_abline(linetype='dashed')+ ggtitle('Roc Curve of '+classifier.__class__.__name__)
    print(g)

#basic concepts
trainheadlines = []
for row in range(0, len(data.index)):
    trainheadlines.append(' '.join(str(x) for x in data.iloc[row, 2:27]))
print(trainheadlines[:10])
# example = CountVectorizer().build_tokenizer()(trainheadlines)
basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
print(basictrain.shape)
basicmodel = LogisticRegression()
basicmodel = basicmodel.fit(basictrain[:len(train.index)], train["Label"])
testheadlines = []

for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
basictest = basicvectorizer.transform(testheadlines)
simplePredict = basicmodel.predict(basictest)
pd.crosstab(test["Label"], simplePredict, rownames=["Actual"], colnames=["Predicted"])
sklearn.metrics.accuracy_score(test["Label"], simplePredict)
basicwords = basicvectorizer.get_feature_names()
basiccoeffs = basicmodel.coef_.tolist()[0]
coeffdf = pd.DataFrame({'Word': basicwords,
                        'Coefficient': basiccoeffs})
coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
coeffdf.head(5)
coeffdf.tail(5)
#some statistics
example = CountVectorizer().build_tokenizer()(trainheadlines)
df = pd.DataFrame([[x, example.count(x)] for x in set(example)], columns=['Word', 'Count'])
df = df.sort_values(by=['Count'], ascending=False)

#more advanced with weights
transformer = TfidfVectorizer(min_df=1, smooth_idf=False)
tfidf = transformer.fit_transform(trainheadlines)
print(tfidf.shape)
model = LogisticRegression()
model = model.fit(tfidf[:len(train.index)], train["Label"])
testY = []
for row in range(0, len(test.index)):
    testY.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
basictest = transformer.transform(testY)
predictions = model.predict(basictest)
pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
sklearn.metrics.accuracy_score(test["Label"], predictions)

#advanced model
advancedvectorizer = CountVectorizer(ngram_range=(2, 2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = LogisticRegression()
advancedmodel = advancedmodel.fit(advancedtrain[:len(train.index)], train["Label"])
testheadlines = []
for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
advancedtest = advancedvectorizer.transform(testheadlines)
advpredictions = advancedmodel.predict(advancedtest)
pd.crosstab(test["Label"], advpredictions, rownames=["Actual"], colnames=["Predicted"])
sklearn.metrics.accuracy_score(test["Label"], advpredictions)
advwords = advancedvectorizer.get_feature_names()
advcoeffs = advancedmodel.coef_.tolist()[0]
advcoeffdf = pd.DataFrame({'Words' : advwords, 'Coefficient' : advcoeffs})
advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
advcoeffdf.head(5)
advcoeffdf.tail(5)

#select features by using variance treshhold
#don't know how to work with it
from Utility import *
from sklearn.feature_selection import VarianceThreshold

trainheadlines = []
for row in range(0, len(data.index)):
    trainheadlines.append(' '.join(str(x) for x in data.iloc[row, 2:27]))
print(trainheadlines[:10])
basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
print(basictrain.shape)
advancedvectorizer = CountVectorizer(ngram_range=(2, 2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
transformer = TfidfVectorizer(min_df=1, smooth_idf=False)
tfidf = transformer.fit_transform(trainheadlines)
print(tfidf.shape)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
result = sel.fit_transform(trainheadlines)