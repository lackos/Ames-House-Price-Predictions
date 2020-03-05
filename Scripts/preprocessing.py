import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os

import scipy.stats as stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
import warnings

import model_evaluation as me

def data_prep(target):
    """
    Prepare the data in the training and testing sets. This function performs
    feature engineering. It does not impute or enocode the variables.
    """
    ## Load the datasets
    training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    training_data.dropna(subset= [target], inplace=True)
    test_data =  pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), index_col='Id')

    ##Drop outliers from training set which were found in the exploration
    training_data = training_data.drop(training_data[training_data['Id'] == 1299].index)
    training_data = training_data.drop(training_data[training_data['Id'] == 524].index)

    ## Set the index
    training_data.set_index('Id')

    ## log transform variables to normalize the distrbution for error reduction
    training_data['SalePrice'] = np.log(training_data['SalePrice'])

    training_data['GrLivArea'] = np.log(training_data['GrLivArea'])
    test_data['GrLivArea'] = np.log(test_data['GrLivArea'])

    training_data['LotArea'] = np.log(training_data['LotArea'])
    test_data['LotArea'] = np.log(test_data['LotArea'])

    training_data['TotRmsAbvGrd'] = np.log(training_data['TotRmsAbvGrd'])
    test_data['TotRmsAbvGrd'] = np.log(test_data['TotRmsAbvGrd'])

    ## Has a pool
    training_data['has_pool'] = training_data.apply(lambda x: 0 if x['PoolArea'] == 0 else 1, axis = 1)
    test_data['has_pool'] = test_data.apply(lambda x: 0 if x['PoolArea'] == 0 else 1, axis = 1)

    ## Create a new column for the house with basements and log transform non
    ## zero values.
    training_data['HasBsmt'] = pd.Series(len(training_data['TotalBsmtSF']), index=training_data.index)
    training_data['HasBsmt'] = 0
    training_data.loc[training_data['TotalBsmtSF']>0,'HasBsmt'] = 1
    training_data.loc[training_data['HasBsmt']==1,'TotalBsmtSF'] = np.log(training_data['TotalBsmtSF'])

    test_data['HasBsmt'] = pd.Series(len(test_data['TotalBsmtSF']), index=test_data.index)
    test_data['HasBsmt'] = 0
    test_data.loc[test_data['TotalBsmtSF']>0,'HasBsmt'] = 1
    test_data.loc[test_data['HasBsmt']==1,'TotalBsmtSF'] = np.log(test_data['TotalBsmtSF'])

    return training_data, test_data

def label_encode_objects(training, testing, objects):
    """
    Encoded categorical features (objects) with labels. Should only be used for ordinal
    features.
    """
    # Make copy to avoid changing original data
    label_X_train = training.copy()
    label_X_test = testing.copy()

    ## Apply label encoder to each column with categorical data
    for col in objects:
        le = LabelEncoder().fit(training[col])
        label_X_train[col] = le.transform(training[col])
        label_X_test[col] = le.transform(testing[col])

    training = label_X_train
    testing = label_X_test

    return training, testing

def replace_nan_with_none(dataframe, features):
    for col in features:
        dataframe[col] = dataframe[col].fillna('none')
    return dataframe

def onehot_encode_objects(training, testing, object_cols):
    """
    One-hot enocode the object variables (object_cols).
    """
    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(training[object_cols]))
    OH_cols_test = pd.DataFrame(OH_encoder.transform(testing[object_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = training.index
    OH_cols_test.index = testing.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_training = training.drop(object_cols, axis=1)
    num_X_testing = testing.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_training = pd.concat([num_X_training, OH_cols_train], axis=1)
    OH_X_testing = pd.concat([num_X_testing, OH_cols_test], axis=1)

    training = OH_X_training
    testing = OH_X_testing

    return training, testing
