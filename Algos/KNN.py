from sklearn.neighbors import KNeighborsClassifier

from DataHandling import pre_process

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

        print(
            '{matches}/{total} matches for K={k}, P={p}, Weight={w}'.format(matches=match, total=actual, k=k, p=p, w=w))
        print(cm)
        print()


def knn_classifier(x_train, y_train, x_test):
    """
    KNN Classifier for set Hyperparams with a plot of the results
    :param x_train:
    :param y_train:
    :param x_test:
    :return:
    """
    """Hyper Params"""
    k, p, w = 11, 1, 'distance'

    # Classifier
    classifier = KNeighborsClassifier(
        n_neighbors=k,
        weights=w,
        algorithm='auto',
        metric='minkowski',
        p=p
    )
    classifier.fit(x_train, y_train.ravel())

    return classifier.predict(x_test)


def main():
    # Preprocess Data
    x_train, x_test, y_train, y_test, sc = pre_process(TRAIN_PATH, TEST_PATH)

    # Run Classifier
    # knn_hp_testing(x_train, y_train, x_test, y_test)
    y_pred = knn_classifier(x_train, y_train, x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


if __name__ == '__main__':
    main()
