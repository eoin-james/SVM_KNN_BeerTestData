from sklearn import svm
import pandas as pd

import numpy as np
import data_pre_process as pp

train_path = 'BeerData/beer_training.csv'
test_path = 'BeerData/beer_test.csv'


def svm_():
    x_train, x_test, y_train, y_test, sc = pp.pre_process(train_path, test_path)
    clf = svm.SVC(kernel='poly', degree=6)
    clf.fit(x_train, y_train.ravel())
    y_pred = clf.predict(x_test)
    # print(y_pred, '\n', y_test)
    print(len(y_pred))
    match = len([True for i, j in zip(y_pred, y_test) if i == j])
    print(match)

svm_()