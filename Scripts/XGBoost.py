import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os

import scipy.stats as stats

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

def score_model(y_valid, predictions, log=False):
    if log == False:
        score = np.sqrt(mean_squared_error(np.log(y_valid), np.log(predictions)))
    elif log == True:
        score = np.sqrt(mean_squared_error(y_valid, predictions))
    return score

def data_prep(target):
    training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    training_data.dropna(subset= [target], inplace=True)
    test_data =  pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), index_col='Id')

    ##Drop outliers from training set
    training_data = training_data.drop(training_data[training_data['Id'] == 1299].index)
    training_data = training_data.drop(training_data[training_data['Id'] == 524].index)

    ## Set the index
    training_data.set_index('Id')

    ## log transform variables to normalize the distrbution for error reduction
    training_data['SalePrice'] = np.log(training_data['SalePrice'])

    training_data['GrLivArea'] = np.log(training_data['GrLivArea'])
    test_data['GrLivArea'] = np.log(test_data['GrLivArea'])

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
    Encodes object classes with labels
    """

    le_dict = {}

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

def binary_pool(dataframe):
    dataframe['has_pool'] = dataframe.apply(lambda x: 0 if x['PoolArea'] == 0 else 1, axis = 1)
    return dataframe

def onehot_encode_objects(training, testing, object_cols):
    """
    One-hot enocode the object variables
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

def cv_XGB():
    pass

def simple_Regressor(X_train, y_train, X_val, y_val, params):
    xg_reg = xgb.XGBRegressor(param = params)
    xg_reg.fit(X_train,y_train)
    preds = xg_reg.predict(X_val)
    # print(preds)
    print(score_model(y_val, preds, log=True))

    return xg_reg

    # xgb.plot_importance(xg_reg)
    # plt.rcParams['figure.figsize'] = [5, 5]
    # plt.show()

def Rand_search_CV(X_train, y_train):
    one_to_left = stats.beta(10, 1)
    from_zero_positive = stats.expon(0, 50)

    params = {
        "n_estimators": stats.randint(3, 40),
        "max_depth": stats.randint(3, 40),
        "learning_rate": stats.uniform(0.05, 0.4),
        "colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": stats.uniform(0, 10),
        'reg_alpha': from_zero_positive,
        "min_child_weight": from_zero_positive,
    }

    xg_reg = xgb.XGBRegressor(nthreads=-1)
    gs = RandomizedSearchCV(xg_reg, params, n_jobs=1)
    gs.fit(X_train, y_train)
    # print(gs.cv_results_)
    print(gs.best_params_)
    print(gs.best_score_)

def create_submission(filename, preds_test, X_test):
    submission = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
    submission.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)


def main():

    params = {'colsample_bytree': 0.888987147635711, 'gamma': 2.7169303234894624, 'learning_rate': 0.42965239231146557, 'max_depth': 8, 'min_child_weight': 32.23306727399577, 'n_estimators': 22, 'reg_alpha': 3.2303668433984454, 'subsample': 0.9788542854030317}

    # features = ['GrLivArea', 'HasBsmt', 'TotalBsmtSF', 'YearBuilt', 'OverallQual', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'GarageCars', 'PoolArea', 'Fireplaces', 'YearRemodAdd', 'MiscVal', 'TotRmsAbvGrd']
    numeric_features = ['YearBuilt', 'OverallQual', 'OverallCond', 'LotArea',
    'BedroomAbvGr', 'FullBath', 'HalfBath', 'GarageCars', 'PoolArea', 'Fireplaces',
    'YearRemodAdd', 'MiscVal', 'GrLivArea', 'TotRmsAbvGrd', 'TotalBsmtSF']
    object_features = ['CentralAir', 'Heating', 'LandContour', 'BldgType',
    'HouseStyle', 'ExterCond', 'Street', 'GarageQual', 'PoolQC', 'LotShape',
    'LotConfig', 'LandSlope', 'Neighborhood', 'ExterQual']
    features = numeric_features + object_features

    ## Define the features will will need to be one-hot encoded and label encoded.
    oh_features = ['Heating', 'LandContour', 'BldgType', 'HouseStyle', 'ExterCond',
    'Street', 'LotConfig', 'Neighborhood',]
    label_enc_features = ['CentralAir', 'GarageQual', 'PoolQC', 'LotShape',
    'LandSlope', 'ExterQual']
    target = 'SalePrice'
    training, test = data_prep(target)

    y = training[target]
    X = training[features]

    X_test = test[features]

    X = replace_nan_with_none(X, label_enc_features)
    X_test = replace_nan_with_none(X_test, label_enc_features)
    X = binary_pool(X)
    X_test = binary_pool(X_test)

    X, X_test = label_encode_objects(X, X_test, label_enc_features)
    X, X_test = onehot_encode_objects(X, X_test, oh_features)

    print(X.shape)
    print(X_test.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 0)

    # Rand_search_CV(X, y)
    xg_reg = simple_Regressor(X_train, y_train, X_val, y_val, params)
    preds_test = xg_reg.predict(X_test)
    final_predictions = np.exp(preds_test)
    print(final_predictions)

    create_submission('XGBoost_regressor_1.csv', final_predictions, X_test)

    pass



if __name__ == "__main__":
    main()
