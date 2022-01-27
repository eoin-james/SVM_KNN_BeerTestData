from sklearn import svm
from sklearn.metrics import confusion_matrix

from DataHandling import pre_process

# Data Path
TRAIN_PATH = '../BeerData/beer_training.csv'
TEST_PATH = '../BeerData/beer_test.csv'


def svm_classifier(x_train, x_test, y_train):
    clf = svm.SVC(kernel='poly', degree=6)
    clf.fit(x_train, y_train.ravel())
    y_pred = clf.predict(x_test)
    return y_pred


def main():
    # Preprocess Data
    x_train, x_test, y_train, y_test, sc = dpp.pre_process(TRAIN_PATH, TEST_PATH)

    y_pred = svm_classifier(x_train, x_test, y_train)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


if __name__ == '__main__':
    main()
