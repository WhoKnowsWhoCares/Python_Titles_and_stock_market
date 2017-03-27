from sklearn.metrics import accuracy_score

__author__ = 'Alexander'

from pprint import pprint
from time import time
import logging
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV, RandomizedLogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

from Utility import *

# os.chdir('./TitlesAndStockMarket')

# df = pd.read_csv('./resources/Combined_News_DJIA.csv')
# df["Combined"] = df.iloc[:, 2:27].apply(lambda row: ''.join(str(row.values)), axis=1)
# df.head()
# train, test = train_test_split(df, test_size=0.2, random_state=42) #randomly divide full set to training and test parts

data = load_dict("DataCleared"+str(0))
data.head()
train, test = train_test_split(data, test_size=0.2, random_state=42) #randomly divide full set to training and test parts

pipeline = Pipeline([
    ('logitCV', LogisticRegressionCV()),
    ('logitCVlin', LogisticRegressionCV(solver='linear')),
    # ('rlogit', RandomizedLogisticRegression())
    # ('vect', CountVectorizer()),
    # ('tfidf', TfidfTransformer()),
    # ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    # 'rlogit__scaling':(0.25, 0.5, 0.75),
    # 'rlogit__fit_intercept':(True, False),

    'logitCVlin__class_weight':('balanced',None),
    'logitCVlin__penalty':('l1','l2'),
    'logitCVlin__refit':(True, False),
    'logitCVlin__intercept_scaling':(1,2,5),

    'logitCV__fit_intercept':(True, False),
    'logitCV__solver':('lbfgs', 'sag', 'newton-cg'),
    'logitCV__class_weight':('balanced',None),
    'logitCV__refit':(True, False)

    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 50, 100, 500),
    # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'vect__stop_words': (None, 'english'),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__alpha': (0.00001, 0.000001),
    # 'clf__penalty': ('l2', 'elasticnet'),
    # 'clf__n_iter': (10, 50, 100),
}

# def score(model, test):
#     return accuracy_score(model.predict(test['Cleared']), test['Label'])

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=2, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data['Cleared'], data['Label'])
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Score = save_dict(grid_search, "Logistic")
# grid_search.best_score_