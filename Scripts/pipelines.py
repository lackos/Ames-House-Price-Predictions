import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os

import scipy.stats as stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import preprocessing as pp

import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

from model_evaluation import simple_score, log_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

def auto_pipline():
    ## Target feature (Predictor)
    target = 'SalePrice'

    ## Features to train on
    numeric_features = ['YearBuilt', 'OverallQual', 'OverallCond', 'LotArea',
    'BedroomAbvGr', 'FullBath', 'HalfBath', 'GarageCars', 'PoolArea', 'Fireplaces',
    'YearRemodAdd', 'GrLivArea', 'TotRmsAbvGrd', 'TotalBsmtSF', 'HasBsmt']
    object_features = ['CentralAir', 'LandContour', 'BldgType',
    'HouseStyle', 'ExterCond', 'Neighborhood']
    features = numeric_features + object_features



    ## Set scoring
    # custom_score = make_scorer(simple_score, greater_is_better=False)
    custom_score = make_scorer(log_score, greater_is_better=False)

    ## Load training set and split in train and val sets
    training_data, test_data = pp.data_prep(target)

    # training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    # training_data.dropna(subset= [target], inplace=True)
    X_train, X_val, y_train, y_val = train_test_split(training_data[features], training_data[target], train_size=0.8, test_size=0.2, random_state = 0)
    print("Training and validation tests loaded")

    ###
    ### Define the preprocessing transforms using ColumnTransformer
    ###
    transformer = ColumnTransformer(transformers=[
                                   ('oh_encode', OneHotEncoder(handle_unknown='ignore'), object_features),
                                   ], remainder='passthrough')

    ###
    ### Set transform parameters
    ###
    n_features_to_test = np.arange(1, 5)
    gamma_to_test = np.arange(0, 5)
    one_to_left = stats.beta(10, 1)
    from_zero_positive = stats.expon(0, 50)

    params_list = []

    ## reduce_dim parameters (PCA)
    PCA_parameters = {'reduce_dim': [PCA()],
                    'reduce_dim__n_components': n_features_to_test,
                    }

    ## reduce_dim parameters (SelectKBest)
    Kbest_parameters = { 'reduce_dim': [SelectKBest(f_regression)],
                    'reduce_dim__k': n_features_to_test,
                    }

    ## XGBoost parameters
    XGB_params = {"XGboost__n_estimators": stats.randint(3, 40),
                "XGboost__max_depth": stats.randint(3, 40),
                "XGboost__learning_rate": stats.uniform(0.05, 0.4),
                "XGboost__colsample_bytree": one_to_left,
                "XGboost__subsample": one_to_left,
                "XGboost__gamma": stats.uniform(0, 10),
                'XGboost__reg_alpha': from_zero_positive,
                "XGboost__min_child_weight": from_zero_positive,
                }

    params_list.append(dict(**PCA_parameters, **XGB_params))
    params_list.append(dict(**Kbest_parameters, **XGB_params))

    # print(params_list)

    print("Pipeline parameters set")
    pipe = Pipeline([
        ('transform', transformer),
        ('reduce_dim', PCA()),
        ('XGboost', xgb.XGBRegressor(nthreads=-1, objective='reg:squarederror'))
        ])

    ###
    ### Default Parameter Pipe
    ###
    # pipe = pipe.fit(X_train, y_train)
    # print("pipeline fitted")
    # predictions = pipe.predict(X_val)
    # print(score_model(y_val, predictions))


    ###
    ### GridSearch
    ###
    # gridsearch = GridSearchCV(pipe, params, verbose=1, scoring=custom_score).fit(X_train, y_train)
    # print('Final score is: ', gridsearch.score(X_val, y_val))

    ###
    ### Random Grid Search
    ###
    randsearch = RandomizedSearchCV(pipe, params_list, verbose=1, scoring=custom_score, n_iter=500).fit(X_train, y_train)
    predictions = randsearch.predict(X_val)
    print("Score on validation set: " + round(log_score(y_val, predictions), 4))
    print("Best Parameters for XGBoost: " + randsearch.best_params_)



def main():
    auto_pipline()

if __name__ == "__main__":
    main()
