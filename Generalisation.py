from ggplot import *

__author__ = 'Alexander'

import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = "8, 8"

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, roc_curve
#get some classifier
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from Utility import *

# print(os.getcwd())
os.chdir('./TitlesAndStockMarket')

# df = pd.read_csv('./resources/Combined_News_DJIA.csv')
# df["Combined"] = df.iloc[:, 2:27].apply(lambda row: ''.join(str(row.values)), axis=1)
# df.head()

#set of the classifiers
Classifiers = [
    # [LogisticRegression(), CountVectorizer()],
    # [LogisticRegression(), TfidfVectorizer(min_df=1, smooth_idf=False)],
    LogisticRegressionCV(solver='newton-cg',class_weight='balanced',n_jobs=2),
    # LogisticRegressionCV(solver='sag',class_weight='balanced',n_jobs=2),
    LogisticRegressionCV(solver='lbfgs',class_weight='balanced',n_jobs=2),
    LogisticRegressionCV(solver='liblinear',penalty='l1',class_weight='balanced',n_jobs=2),
    LogisticRegressionCV(solver='liblinear',penalty='l2', dual=True, class_weight='balanced',n_jobs=2),
    # LogisticRegression(C=0.000000001, solver='liblinear', max_iter=100),
    # CountVectorizer(stop_words='english', ngram_range=(1, 1)),
    # CountVectorizer(stop_words='english', ngram_range=(1, 2)),
    # TfidfVectorizer(stop_words='english', ngram_range=(1, 1), norm='l1'),
    # TfidfVectorizer(stop_words='english', ngram_range=(1, 1), norm='l2'),
    # TfidfVectorizer(stop_words='english', ngram_range=(1, 2), norm='l1'),
    # TfidfVectorizer(stop_words='english', ngram_range=(1, 2), norm='l2'),
    SGDClassifier(loss='log', n_jobs=2),
    SGDClassifier(loss='modified_huber', n_jobs=2),
    SGDClassifier(loss='squared_hinge', n_jobs=2),
    SGDClassifier(loss='perceptron', n_jobs=2),
    SGDClassifier(penalty='l1', n_jobs=2),
    SGDClassifier(penalty='l2', n_jobs=2),
    SGDClassifier(penalty='elasticnet', n_jobs=2),
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    DecisionTreeClassifier(criterion='entropy'),
    RandomForestClassifier(n_estimators=200, n_jobs=2),
    AdaBoostClassifier(n_estimators=100),
    AdaBoostClassifier(algorithm='SAMME'),
    AdaBoostClassifier(),
    BernoulliNB(),
    GaussianNB(),
]
# to count all words and make data frame
# list = collect_words_and_save('UsefulWords')

#use this to load words
#full set of words saved in "SortedWords"
# list = load_dict("UsefulWords")
# len(list) #34567

#clear all data
# stops = set(stopwords.words("english"))
# data = manage_df_and_save("DataCleared"+str(0), df, stops)
#load cleared data
data = load_dict("DataCleared"+str(0))
# data.head()
# data_train, data_test = train_test_split(data, test_size=0.2, random_state=42) #randomly divide full set to training and test parts
#
# train_text = []
# test_text = []
# for each in data_train['Cleared']:
#     train_text.append(each)
# # print(train_text)
#
# for each in data_test['Cleared']:
#     test_text.append(each)
#
# y_train, y_test =data_train['Label'], data_test['Label']
# transformer = TfidfVectorizer(min_df=1, smooth_idf=False, sublinear_tf=True, max_df=0.5,
#                              stop_words='english')
# X_train = transformer.fit_transform(train_text)
# X_test = transformer.transform(test_text)
# print(X_train[:5])
#
# feature_names = transformer.get_feature_names()
# len(feature_names)


Results = []
n_features = 1500
select_chi2 = 500
# min=100
def Main(data, select_chi2):

    print("started classification"+str(select_chi2))
    # dic = [[k, v] for k, v in list if v > min] #apply conditions
    # print(len(dic))

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
    train_features = transformer.fit_transform(train_text)
    test_features = transformer.transform(test_text)
    # print(X_train[:5])

    feature_names = transformer.get_feature_names()
    len(feature_names)

    print("Extracting %d best features by a chi-squared test" %
      select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=select_chi2)
    train_features = ch2.fit_transform(train_features, y_train)
    test_features = ch2.transform(test_features)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

    #create data framte
    # useful = pd.DataFrame(dic, columns=['Word', 'Count'])
    # useful = useful.sort_values(by=['Count'], ascending=False)

    # words = useful['Word'].unique() #all our words
    # print(len(words))

# #get word histogram
# plt.hist(useful['Count'], 25, range=(1, 25))
# plt.title("Histogram count of words by count in text")
# plt.xlabel("Total count in text")
# plt.ylabel("Count of words")
# plt.savefig('count.png')
# plt.show()

    # train, test = train_test_split(data, test_size=0.2, random_state=42) #randomly divide full set to training and test parts

        #how to get proportions 0 and 1
    # non_decrease = data_train[data_train['Label']==1]
    # decrease = data_train[data_train['Label']==0]
    # proportions = len(non_decrease)/len(data)
    # print("part of the days when it's not decreasing", proportions)

# #to look at words affected to decrease and increase of the stock prices
# #full baskets - get only meaningul words
# non_decrease_word=[]
# decrease_word=[]
# for each in non_decrease['Cleared']:
#     non_decrease_word.append(each)
# for each in decrease['Cleared']:
#     decrease_word.append(each)
# #create two clouds with positive and negative words
# wordcloud1 = WordCloud(background_color='black',
#                       width=3000,
#                       height=2500
#                      ).generate(decrease_word[0])
# plt.figure(1,figsize=(8,8))
# plt.savefig('wordcloud1.png')
# # plt.imshow(wordcloud1)
# # plt.axis('off')
# # plt.show()
#
# wordcloud2 = WordCloud(background_color='white',
#                       width=3000,
#                       height=2500
#                      ).generate(non_decrease_word[0])
# plt.figure(1,figsize=(8, 8))
# plt.savefig('wordcloud2.png')
# # plt.imshow(wordcloud2)
# # plt.axis('off')
# # plt.show()


    #work with prediction models

    #clear everything
    Accuracy=[0]*len(Classifiers)
    # Crosstable=[]
    # Model=[]
    # Plots=[]
    # Coefficients=[]

    for i in range(len(Classifiers)):
        # vectorizer = Classifiers[i][1]
        model = Classifiers[i]
        print("started "+model.__class__.__name__)
        # train_features = vectorizer.fit_transform(train_text)
        # test_features = vectorizer.transform(test_text)
        try:
            fit = model.fit(train_features, y_train)
            pred = fit.predict(test_features)
            # prob = fit.predict_proba(test_features)[:,1]
        except Exception:
            continue
            # dense_features=train_features.toarray()
            # dense_test= test_features.toarray()
            # fit = model.fit(dense_features,y_train)
            # pred = fit.predict(dense_test)
            # prob = fit.predict_proba(dense_test)[:,1]
        accuracy = accuracy_score(pred, y_test) #count accuracy
        Accuracy[i] = accuracy
        # ct = pd.crosstab(y_test, pred, rownames=["Actual"], colnames=["Predicted"]).apply(lambda r: r/sum(r), axis=1) #get crosstable
        # Crosstable.append(ct)
        # Model.append(model.__class__.__name__)
        # look at coefficients works on logistic regr
        # basicwords = vectorizer.get_feature_names()
        # if model.coef_:
        #     basiccoeffs = model.coef_.tolist()[0] #depend on model
        #     coeffdf = pd.DataFrame({'Coefficient': basiccoeffs,
        #                             'Word': feature_names})
        #     coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=False)
        #     Coefficients.append(coeffdf) #to compare with count of words
        # print('Crosstable \n', ct)
        # print('Accuracy of '+model.__class__.__name__+' is '+str(accuracy))
        # print('Count of features ', len(feature_names))
        # fpr, tpr, _ = roc_curve(y_test, prob)
        # tmp = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
        # g = ggplot(tmp, aes(x='fpr', y='tpr')) \
        #     + geom_line() + geom_abline(linetype='dashed') + ggtitle('Roc Curve of ' + model.__class__.__name__)
        # Plots.append(g)
        # print(g)
    # import_in_file(feature_names, Model, Accuracy, Crosstable, Plots, proportions)
    print("done classification"+str(select_chi2))
    Results.append([select_chi2, Accuracy])
    return Results

#some results saved in "Results"

for i in range(100, 500):
    try:
        Main(data, i)
    except Exception:
        print("There is Error on "+str(i)+" step")
        continue
save_dict(Results, "ResultsAlot")
print("All iterations done!")
# print(len(data))

#load if not running
#some results saved in "Results"
#some in "Generalization"
Results = load_dict("Results3")
# print(Results[480][1])
res = pd.DataFrame(Results, columns=["Iter","Accuracy"])
res.tail()
len(res)

# print(res['Accuracy'].tolist())
plt.plot(res['Iter'].tolist(),res['Accuracy'].tolist())
# plt.legend(["Logit","Decision tree","Random forest","AdaBoost","GaussianNB"])
plt.title("Accurcy classifiers with different number of features")
plt.xlabel("Feature count")
plt.ylabel("Accuracy")
plt.legend(loc='best')
plt.show()
#import to file
import_in_file(Results)

