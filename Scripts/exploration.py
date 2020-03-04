import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt
import seaborn as sns
import jinja2

from XGBoost import data_prep, label_encode_objects, replace_nan_with_none

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesRegressor

from scipy.stats import norm
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

def object_cols(dataset):
    s = (dataset.dtypes == 'object')
    object_cols = list(s[s].index)
    return object_cols

def data_overview():
    """
    Outputs a .md file with a general overview of the data in the training set
    and the test set.
    """
    ## Load the training and testing data
    training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col = 'Id')
    test_data = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), index_col = 'Id')

    ## Count the number of numerical columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_numeric = len(training_data.select_dtypes(include=numerics).columns)

    ## Count the number of object Columns
    object_types = ['object']
    num_objects = len(training_data.select_dtypes(include=object_types).columns)

    ## number of rows of the training set
    num_rows = training_data.shape[0]

    ## List of columns with nan values
    na_col_list_train = [col for col in training_data.columns if training_data[col].isnull().any()]
    na_col_dict_train = {}
    for col in na_col_list_train:
        na_col_dict_train[col] = training_data[col].isna().sum()

    na_col_list_test = [col for col in test_data.columns if training_data[col].isnull().any()]
    na_col_dict_test = {}
    for col in na_col_list_test:
        na_col_dict_test[col] = test_data[col].isna().sum()

    ## Catagorical Data
    obj_col_dict_train = {}
    for col in object_cols(training_data):
        obj_col_dict_train[col] = training_data[col].nunique()

    obj_col_dict_test = {}
    for col in object_cols(test_data):
        obj_col_dict_test[col] = test_data[col].nunique()

    overview = open(os.path.join(OUTPUT_DIR, 'data_overview.md'), 'w')
    overview.write("# Data Overview \n")
    overview.write(" ## Training Data \n")
    overview.write("Total num of rows: "  + str(num_rows) + "\n \n")
    overview.write("Total num of columns: "  + str(len(training_data.columns)) + "\n")
    overview.write("Num numerical of columns: "  + str(num_numeric) + "\n")
    overview.write("Num object of columns: "  + str(num_objects) + "\n")
    overview.write("### Numerical Data \n")
    overview.write("Numerical columns: " + str(training_data.select_dtypes(include=numerics).columns) + '\n')
    overview.write(str(training_data.describe()) + "\n \n")
    overview.write("### Catagorical Data \n")
    overview.write("Number of unique objects \n" + str(obj_col_dict_train) + "\n \n")
    overview.write("### Columns with nan values \n")
    overview.write(str(na_col_dict_train) + "\n \n")
    overview.write("## Test data \n")
    overview.write("### Numerical Data \n")
    overview.write(str(test_data.describe()) + "\n \n")
    overview.write("### Catagorical Data \n")
    overview.write("Number of unique objects \n" + str(obj_col_dict_test) + "\n \n")
    overview.write("### Columns with nan values \n")
    overview.write(str(na_col_dict_test) + "\n \n")
    overview.close()

def feature_search(Training_data):
    """
    Given a training set with the outcome this function will output the most
    effective features.
    """
    pass

def numerical_correlation_matrix():
    training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col = 'Id')
    test_data = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), index_col = 'Id')
    corr = training_data.corr()
    corr.style.background_gradient(cmap='coolwarm')
    sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
    plt.show()

def target_corr_matrix(target):
    training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col = 'Id')
    k = 10 #number of variables for heatmap

    corr = training_data.corr()
    cols = corr.nlargest(k, target)[target].index
    cm = np.corrcoef(training_data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()

def scatterplot(feature, target):
    sns.set()
    training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col = 'Id')
    fig, ax = plt.subplots(figsize=(11, 9))

    ax = sns.scatterplot(x=training_data[feature], y=training_data[target])
    plt.show()
    pass

def feature_importance(features):
    training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col = 'Id')
    model = ExtraTreesRegressor
    pass

def histogram(feature):
    sns.set()
    training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col = 'Id')
    fig, ax = plt.subplots(figsize=(11, 9))
    ax = sns.distplot(training_data[feature])
    plt.show()

def feature_pair_plot(training_data, target):
    # training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col = 'Id')
    k = 10 #number of variables for heatmap

    corr = training_data.corr()
    cols = corr.nlargest(k, target)[target].index
    sns.pairplot(training_data[cols], size = 2.5)
    plt.show()

def feature_probplot(training_data, feature, log=False):

    # training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col = 'Id')
    if log == False:
        sns.distplot(training_data[feature], fit=norm);
        fig = plt.figure()
        res = stats.probplot(training_data[feature], plot=plt)
        plt.show()
    elif log == True:
        training_data[feature] = np.log(training_data[feature])
        sns.distplot(training_data[feature], fit=norm);
        fig = plt.figure()
        res = stats.probplot(training_data[feature], plot=plt)
        plt.show()

def single_feature_boxplot(training_data, feature):
    sns.set(style="whitegrid")
    fig = plt.figure()
    sns.boxplot(x=training_data[feature])
    plt.show()

def categorical_boxplot(training_data, feature, predictor):
    sns.set(style="whitegrid")
    fig = plt.figure()
    sns.boxplot(x=training_data[feature], y=training_data[predictor])
    plt.show()

def categorical_boxplot(training_data, feature, predictor):
    sns.set(style="whitegrid")
    fig = plt.figure()
    sns.boxplot(x=training_data[feature], y=training_data[predictor])
    plt.show()

def multi_feature_boxplot(training_data, features, predictor):
    sns.set(style="whitegrid")
    fig = plt.figure()
    sns.boxplot(data = training_data[features] y=training_data[predictor])
    plt.show()

def main():

    numeric_features = ['YearBuilt', 'OverallQual', 'OverallCond', 'LotArea',
    'BedroomAbvGr', 'FullBath', 'HalfBath', 'GarageCars', 'PoolArea', 'Fireplaces',
     'MiscVal', 'GrLivArea', 'TotRmsAbvGrd', 'TotalBsmtSF', 'HasBsmt']
    object_features = ['CentralAir', 'Heating', 'LandContour', 'BldgType',
    'HouseStyle', 'ExterCond', 'Street', 'GarageQual', 'PoolQC', 'LotShape',
    'LotConfig', 'LandSlope', 'Neighborhood', 'ExterQual']
    features = numeric_features + object_features
    label_enc_features = ['CentralAir', 'GarageQual', 'PoolQC', 'LotShape',
    'LandSlope', 'ExterQual']

    # data_overview()
    # correlation_matrix()
    # histogram('GarageArea')
    # scatterplot('GrLivArea', 'SalePrice')
    training, testing = data_prep('SalePrice')
    X = training[features + ['SalePrice']]
    X_test = testing[features]
    X = replace_nan_with_none(X, label_enc_features)
    X_test = replace_nan_with_none(X_test, label_enc_features)
    X, X_test = label_encode_objects(X, X_test, label_enc_features)

    ###
    ### PAIRPLOTS
    ###
    # feature_pair_plot(X, 'SalePrice')
    # feature_probplot(X, 'PoolQC', log=False)

    ###
    ### Correlaton matrix
    ###
    # target_corr_matrix('SalePrice')

    ###
    ### BOXPLOTS
    ###
    # single_feature_boxplot(X, 'SalePrice')
    # categorical_boxplot(X, 'CentralAir' ,'SalePrice')
    multi_feature_boxplot(X, numeric_features, 'SalePrice')





if __name__ == "__main__":
    main()
