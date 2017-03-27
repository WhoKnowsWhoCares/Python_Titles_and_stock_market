__author__ = 'Alexander'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV

print(__doc__)
from pprint import pprint
from time import time
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

from Utility import *
print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# print(os.getcwd())
os.chdir('./TitlesAndStockMarket')

# df = pd.read_csv('./resources/Combined_News_DJIA.csv')
# df["Combined"] = df.iloc[:, 2:27].apply(lambda row: ''.join(str(row.values)), axis=1)
# df.head()
# df_train, df_test = train_test_split(df, test_size=0.2, random_state=42) #randomly divide full set to training and test parts

data = load_dict("DataCleared"+str(0))
data.head()
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42) #randomly divide full set to training and test parts

train_text = []
test_text = []
for each in data_train['Cleared']:
    train_text.append(each)
# print(train_text)

for each in data_test['Cleared']:
    test_text.append(each)

y_train, y_test =data_train['Label'], data_test['Label']
transformer = TfidfVectorizer(min_df=1, smooth_idf=False, sublinear_tf=True, max_df=0.5,
                             stop_words='english')
X_train = transformer.fit_transform(train_text)
X_test = transformer.transform(test_text)
print(X_train[:5])

feature_names = transformer.get_feature_names()
len(feature_names)

##################################################
#constants to save in
pipline_data = "PipelineRes"

###################################################
#pipeline usage
n_features = 1500
select_chi2 = 200

###############################################3#
#extract select_chi2 count of features
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
print(feature_names)
len(feature_names)
#################################################
#
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    # train_time = time() - t0
    # print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    # test_time = time() - t0
    # print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score #, train_time, test_time

results = []
for clf, name in (
        # (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        # (Perceptron(n_iter=50), "Perceptron"),
        # (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        # (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (LogisticRegressionCV(fit_intercept=True, solver='newton-cg',class_weight='balanced',refit=True), "Logistic"),
        (AdaBoostClassifier(), "AdaBoost"),
        (BernoulliNB(alpha=.01), "BernulliNB"),
        (DecisionTreeClassifier(), "Decision tree"),
        (RandomForestClassifier(n_estimators=200), "Random forest")):

    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

###################################################33
#plotting
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(2)]

clf_names, score = results

# plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.yticks(())
# plt.legend(loc='best')
# plt.subplots_adjust(left=.25)
# plt.subplots_adjust(top=.95)
# plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()