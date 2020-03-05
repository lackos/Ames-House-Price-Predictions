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
import preprocessing as pp

warnings.filterwarnings('ignore')

## Directory Locations

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

def simple_Regressor(X_train, y_train, X_val, y_val, params):
    """
    Basic implementation of the XGB regression. Returns the fitted model for
    further use.
    """
    ## Define and fit the model with default values
    xg_reg = xgb.XGBRegressor(param = params)
    xg_reg.fit(X_train,y_train)

    ## Predict the values of the validation set
    preds = xg_reg.predict(X_val)

    ## Score the validation set predictions
    print("Validation score: " + str(me.score_model(y_val, preds, log=True)))
    return xg_reg

    # xgb.plot_importance(xg_reg)
    # plt.rcParams['figure.figsize'] = [5, 5]
    # plt.show()

def Rand_search_CV(X_train, y_train):
    """
    Performs a cross validated randomized search to find the optimal XGBoost
    parameters.
    """
    one_to_left = stats.beta(10, 1)
    from_zero_positive = stats.expon(0, 50)

    ## Define the parameter space for random search
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
    gs = RandomizedSearchCV(xg_reg, params, n_jobs=1, n_iter=60)
    gs.fit(X_train, y_train)
    # print(gs.cv_results_)
    print("Best XGB parameters: " + str(gs.best_params_))
    print("Best XGB score: " + str(gs.best_score_))

def create_submission(filename, preds_test, X_test):
    """
    Creates a submission file for the Kaggle competition.
    """
    submission = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
    submission.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

def main():
    ### Best parameters found using randomized search.
    params = {'colsample_bytree': 0.9329929987362903, 'gamma': 0.7254978933056699, 'learning_rate': 0.2727602216620203, 'max_depth': 39, 'min_child_weight': 139.48848439142373, 'n_estimators': 30, 'reg_alpha': 2.773979167415611, 'subsample': 0.9634441486492169}


    ## Define the predictor features (both numeric and categorical).
    numeric_features = ['YearBuilt', 'OverallQual', 'OverallCond', 'LotArea',
    'BedroomAbvGr', 'FullBath', 'HalfBath', 'GarageCars', 'PoolArea', 'Fireplaces',
    'YearRemodAdd', 'GrLivArea', 'TotRmsAbvGrd', 'TotalBsmtSF', 'HasBsmt']
    object_features = ['CentralAir', 'LandContour', 'BldgType',
    'HouseStyle', 'ExterCond', 'Neighborhood']
    features = numeric_features + object_features

    ## Define the features will will need to be one-hot encoded and label encoded.
    oh_features = ['LandContour', 'BldgType', 'HouseStyle', 'ExterCond',
      'Neighborhood',]
    label_enc_features = ['CentralAir',  ]
    target = 'SalePrice'

    ## Prep the data
    training, test = pp.data_prep(target)

    y = training[target]
    X = training[features]
    X_test = test[features]

    ## Encode the categorical variables
    X = pp.replace_nan_with_none(X, label_enc_features)
    X_test = pp.replace_nan_with_none(X_test, label_enc_features)
    X, X_test = pp.label_encode_objects(X, X_test, label_enc_features)
    X, X_test = pp.onehot_encode_objects(X, X_test, oh_features)

    # print(X.shape)
    # print(X_test.shape)

    ## Split training set into a smaller training set and validation set.
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state = 0)

    ## Output the best parameters for XGBRegressor
    # Rand_search_CV(X, y)

    ## Train and fit the model
    xg_reg = simple_Regressor(X_train, y_train, X_val, y_val, params)

    # me.perm_import(xg_reg, features, X_val, y_val)
    # me.part_plot_1D(xg_reg, X_train.columns.to_list(), X_val, y_val, 'OverallQual')
    # me.part_plot_2D(xg_reg, X_train.columns.to_list(), X_val, y_val, 'OverallQual', 'CentralAir')
    # me.shap_values(X_val.iloc[0,:], xg_reg)


    # preds_test = xg_reg.predict(X_test)
    # final_predictions = np.exp(preds_test)
    # print(final_predictions)
    # ## Create submission for Kaggle competition.
    # create_submission('XGBoost_regressor_2.csv', final_predictions, X_test)

if __name__ == "__main__":
    main()
