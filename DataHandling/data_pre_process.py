"""Data Pre-processing file"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

"""
Data files headings are 
calorific_value, nitrogen, turbidity, style, alcohol, sugars, bitterness, beer_id, colour, degree_of_fermentation
"""


def pre_process(train_file, test_file):
    # Extract training data to pandas file and add headings
    training_df = pd.read_csv(train_file, sep='\t', engine='python')
    training_df.columns = ["calorific_value", "nitrogen", "turbidity", "style", "alcohol", "sugars", "bitterness",
                           "beer_id", "colour", "degree_of_fermentation"]

    # Extract test data to pandas file and add headings
    test_df = pd.read_csv(test_file, sep='\t', engine='python')
    test_df.columns = ["calorific_value", "nitrogen", "turbidity", "style", "alcohol", "sugars", "bitterness",
                       "beer_id", "colour", "degree_of_fermentation"]

    # Matrix of features and dependant variable vector - style is the element to predict
    x_train = training_df.iloc[:, training_df.columns != 'style']
    y_train = training_df.iloc[:, training_df.columns == 'style']

    x_test = test_df.iloc[:, test_df.columns != 'style']
    y_test = test_df.iloc[:, test_df.columns == 'style']

    # Handle any missing data if any exists
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    imputer.fit(x_train)
    x_train = imputer.transform(x_train)

    imputer.fit(x_test)
    x_test = imputer.transform(x_test)

    # Label Encode the Dependant Variable
    lb = LabelEncoder()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    # Feature Scaling - Only do to training set
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test, y_train, y_test, sc
