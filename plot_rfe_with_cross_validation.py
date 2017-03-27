"""
===================================================
Recursive feature elimination with cross-validation
===================================================

A recursive feature elimination example with automatic tuning of the
number of features selected with cross-validation.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV

print(__doc__)

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
# X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
#                            n_redundant=2, n_repeated=0, n_classes=8,
#                            n_clusters_per_class=1, random_state=0)

from Utility import *
import os
os.chdir('./TitlesAndStockMarket')

data = load_dict("DataCleared"+str(0))
data.head()
train, test = train_test_split(data, test_size=0.2, random_state=42) #randomly divide full set to training and test parts
print('data loaded')

full_text = []
train_text = []
test_text = []
for each in data['Cleared']:
    full_text.append(each)

for each in train['Cleared']:
    train_text.append(each)
# print(train_text)

for each in test['Cleared']:
    test_text.append(each)

basicvectorizer = CountVectorizer(stop_words='english')
basictrain = basicvectorizer.fit_transform(train_text)
print(basictrain.shape)
advancedvectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
advancedtrain = advancedvectorizer.fit_transform(train_text)
print(advancedtrain.shape)
transformer = TfidfVectorizer(min_df=1, smooth_idf=False)
tfidf = transformer.fit_transform(train_text)
print(tfidf.shape)

# Create the RFE object and compute a cross-validated score.
# svc = SVC(kernel="linear")
logit = LogisticRegressionCV(fit_intercept=True, solver='newton-cg',class_weight='balanced',refit=True)
# The "accuracy" scoring is proportional to the number of correct
# classifications

rfecv = RFECV(estimator=logit, step=10, cv=StratifiedKFold(2),
              scoring='accuracy', n_jobs=2)
rfecv.fit(tfidf[:len(train.index)], train["Label"])

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
