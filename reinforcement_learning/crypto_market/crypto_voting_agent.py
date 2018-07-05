import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.externals import joblib
from sklearn.metrics.classification import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

dir_data = 'data_btc_eur/'
dir_models = 'models_btc_eur/'
ticket = 'BTC_EUR'

print('loading data...')
x = np.load(dir_data + 'x.npy')
print(x.shape)
x = np.nan_to_num(x)

y_orig = np.load(dir_data + 'y.npy')

print('reshaping data...')
if len(x.shape) > 2:
    x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
    print(x.shape)

if x.shape[1] == 7:
    x = x[:, 2:7]

print(x.shape)

print('min:', np.min(x))
print('max:', np.max(x))

labels = ['ror', 'bench', 'lowerb', 'mean', 'median', 'higherb']

if len(y_orig.shape) > 1:
    y = y_orig.reshape(y_orig.shape[0] * y_orig.shape[1])
else:
    y = y_orig
print('y_orig:', y.shape)

unique, counts = np.unique(y_orig, return_counts=True)
print(dict(zip(unique, counts)))

print('spliting data...')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

print('getting model...Voting')
clf1 = RandomForestClassifier()
clf2 = KNeighborsClassifier(3)
clf3 = MLPClassifier(early_stopping=True,
                     activation='tanh',
                     max_iter=100,
                     shuffle=True,
                     validation_fraction=0.1,
                     tol=1e-6,)
clf4_1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
clf4_2 = AdaBoostClassifier(base_estimator=RandomForestClassifier())
clf5 = GaussianNB()
clf6 = DecisionTreeClassifier()

clf = VotingClassifier(
    estimators=[('c1', clf1), ('c2', clf2), ('c3', clf3), ('c4_1', clf4_1), ('c4_2', clf4_2), ('c5', clf5),
                ('c6', clf6)]
    , voting='soft')

print('training...')
clf.fit(x, y)

print('predicting...')
predicted = clf.predict(x_test)
print(classification_report(y_test, predicted))

id = len(os.listdir(dir_models))
joblib.dump(clf, dir_models + ticket + '_voting_' + str(id) + '.pkl')
