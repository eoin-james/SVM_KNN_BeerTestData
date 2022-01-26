from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import data_pre_process as pp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

train_path = 'BeerData/beer_training.csv'
test_path = 'BeerData/beer_test.csv'

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])


def knn():
    y_pred = None
    x_train, x_test, y_train, y_test, sc = pp.pre_process(train_path, test_path)

    for k in range(1, 16):
        for p in range(1, 3):
            for weights in ['uniform', 'distance']:
                classifier = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=p, weights=weights)
                classifier.fit(x_train, y_train.ravel())

                # Predicting the Test set results
                y_pred = classifier.predict(x_test)
                match = len([True for i, j in zip(y_pred, y_test) if i == j])
                actual = len(y_test)
                print("k =", k, '; p =', p, '; Weight =', weights, "; Match = ", match)

    return y_pred, y_test


if __name__ == '__main__':
    predicted, actual = knn()
