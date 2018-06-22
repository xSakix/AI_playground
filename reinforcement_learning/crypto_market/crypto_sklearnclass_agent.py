import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.metrics.classification import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

dir_data = 'data_ltc_btc/'
dir_models = 'models_ltc_btc/'
ticket = 'LTC_BTC'

factor = 1

print('loading data...')
x = np.load(dir_data + 'x.npy')
print(x.shape)

x = np.nan_to_num(x)
x = np.repeat(x, factor, axis=0)

y_orig = np.load(dir_data + 'y.npy')
y_orig = np.repeat(y_orig, factor, axis=0)

print('reshaping data...')
if len(x.shape) > 2:
    x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
    print(x.shape)

if x.shape[1] == 7:
    x = x[:, 2:7]

print(x.shape)
print('scaling data...')

print('min:', np.min(x))
print('max:', np.max(x))

labels = ['ror', 'bench', 'lowerb', 'mean', 'median', 'higherb']

if len(y_orig.shape) > 1:
    y = y_orig.reshape(y_orig.shape[0]*y_orig.shape[1])
else:
    y = y_orig
print('x:', x.shape)
print('y_orig:', y.shape)

unique, counts = np.unique(y_orig, return_counts=True)
print(dict(zip(unique, counts)))

print('spliting data...')
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

print('getting model...RandomForest')
clf = RandomForestClassifier(verbose=True)

print('training...')
clf.fit(x, y)

print('predicting...')
predicted = clf.predict(x_test)
print(classification_report(y_test, predicted))

id = len(os.listdir(dir_models))
joblib.dump(clf, dir_models + ticket + '_random_forrest_' + str(id) + '.pkl')

print('getting model...KNeighbors')
clf = KNeighborsClassifier(3)

print('training...')
clf.fit(x, y)

print('predicting...')
predicted = clf.predict(x_test)
print(classification_report(y_test, predicted))

id = len(os.listdir(dir_models))
joblib.dump(clf, dir_models + ticket + '_kneighbors_' + str(id) + '.pkl')

print('getting model...MLP')
clf = MLPClassifier(alpha=1)

print('training...')
clf.fit(x, y)

print('predicting...')
predicted = clf.predict(x_test)
print(classification_report(y_test, predicted))

id = len(os.listdir(dir_models))
joblib.dump(clf, dir_models + ticket + '_mlp_' + str(id) + '.pkl')

print('getting model...Ada')
clf = AdaBoostClassifier()

print('training...')
clf.fit(x, y)

print('predicting...')
predicted = clf.predict(x_test)
print(classification_report(y_test, predicted))

id = len(os.listdir(dir_models))
joblib.dump(clf, dir_models + ticket + '_adaboost_' + str(id) + '.pkl')

print('getting model...GaussianNB')
clf = GaussianNB()

print('training...')
clf.fit(x, y)

print('predicting...')
predicted = clf.predict(x_test)
print(classification_report(y_test, predicted))

id = len(os.listdir(dir_models))
joblib.dump(clf, dir_models + ticket + '_gausianNB_' + str(id) + '.pkl')

print('getting model...Decision tree classifier')
clf = DecisionTreeClassifier()

print('training...')
clf.fit(x, y)

print('predicting...')
predicted = clf.predict(x_test)
print(classification_report(y_test, predicted))

id = len(os.listdir(dir_models))

joblib.dump(clf, dir_models + ticket + '_decision_tree_' + str(id) + '.pkl')
