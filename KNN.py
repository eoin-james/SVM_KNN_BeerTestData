from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import data_pre_process as dpp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix

"""
K Nearest Neighbour Classifier for Beer Classification
"""

# Data Path
TRAIN_PATH = 'BeerData/beer_training.csv'
TEST_PATH = 'BeerData/beer_test.csv'

# HyperParam
K = list(range(9, 14, 2))  # K = 9 -> 13 odd only
P = range(1, 3)  # Manhattan / Euclidean
WEIGHTS = ['uniform', 'distance']  # Weight function

#
h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])


def knn_hp_testing(x_train, y_train, x_test, y_test):
    y_pred = None
    for k, p, w in itertools.product(K, P, WEIGHTS):
        classifier = KNeighborsClassifier(
            n_neighbors=k,
            weights=w,
            algorithm='auto',
            metric='minkowski',
            p=p
        )
        classifier.fit(x_train, y_train.ravel())

        y_pred = classifier.predict(x_test)
        match = len([True for i, j in zip(y_pred, y_test) if i == j])
        actual = len(y_test)

        cm = confusion_matrix(y_test, y_pred)

        print('{matches}/{total} matches for K={k}, P={p}, Weight={w}'.format(matches=match, total=actual, k=k, p=p, w=w))
        print(cm)
        print()


def knn():
    k, p, w = None, None, None
    classifier = KNeighborsClassifier(
        n_neighbors=k,
        weights=w,
        algorithm='auto',
        metric='minkowski',
        p=p
    )

def main():
    # Preprocess Data
    x_train, x_test, y_train, y_test, sc = dpp.pre_process(TRAIN_PATH, TEST_PATH)

    # Run Classifier
    knn_hp_testing(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    # predicted, actual = knn()
    main()
