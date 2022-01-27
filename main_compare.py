from Algos import knn_classifier, svm_classifier
from DataHandling import pre_process

from sklearn.metrics import confusion_matrix

# Data Path
TRAIN_PATH = 'BeerData/beer_training.csv'
TEST_PATH = 'BeerData/beer_test.csv'


def main():
    x_train, x_test, y_train, y_test, sc = pre_process(TRAIN_PATH, TEST_PATH)

    knn_pred = knn_classifier(x_train, y_train, x_test)
    svm_pred = svm_classifier(x_train, y_train, x_test)

    knn_cm = confusion_matrix(y_test, knn_pred)
    svm_cm = confusion_matrix(y_test, svm_pred)

    knn_matches = len([True for i, j in zip(knn_pred, y_test) if i == j])
    svm_matches = len([True for i, j in zip(svm_pred, y_test) if i == j])

    print('KNN Results')
    print('{matches}/{total} matches'.format(matches=knn_matches, total=len(y_test)))
    print('Confusion matrix')
    print(knn_cm)
    print()
    print('SVM Results')
    print('{matches}/{total} matches'.format(matches=svm_matches, total=len(y_test)))
    print('Confusion matrix')
    print(svm_cm)
    print()


if __name__ == '__main__':
    main()
