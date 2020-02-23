import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt
import seaborn as sns
import jinja2

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

def impute_numerical(dataframe_dict, strategy):
    my_imputer = SimpleImputer(strategy=strategy)

    imputed_train_X = pd.DataFrame(my_imputer.fit_transform(dataframe_dict['X_train']))
    imputed_val_X = pd.DataFrame(my_imputer.transform(dataframe_dict['X_val']))

    # Imputation removed column names; put them back
    imputed_train_X.columns = dataframe_dict['X_train'].columns
    imputed_val_X.columns = dataframe_dict['X_val'].columns

    dataframe_dict['X_train'] = imputed_train_X
    dataframe_dict['X_val'] = imputed_val_X

    return dataframe_dict, my_imputer

def plot_score(training_dict):
    sns.set()
    fig, ax = plt.subplots(figsize=(11, 9))
    x = np.arange(1, 302, 5)
    print(x)
    y = []
    best_n = 0
    best_score = 1000
    for n in x:
        print("n-estimators: " + str(n))
        model = RandomForestRegressor(n_estimators=n)
        model.fit(training_dict['X_train'], training_dict['y_train'])
        preds = model.predict(training_dict['X_val'])
        score = score_model(training_dict['y_val'], preds, log=True)
        y.append(round(score, 4))
        if score < best_score:
            n_best = n
            best_score = score
    plt.plot(x, y)
    print("Best score: " + str(best_score) + " for " + str(n_best) + "n-estimators")
    plt.show()
    # print(y)

def label_encode_objects(data_dict, objects):
    """
    Encodes object classes with labels
    """

    le_dict = {}

    # Make copy to avoid changing original data
    label_X_train = data_dict['X_train'].copy()
    label_X_valid = data_dict['X_val'].copy()

    ## Apply label encoder to each column with categorical data
    for col in objects:
        le_dict[col] = LabelEncoder().fit(data_dict['X_train'][col])
        label_X_train[col] = le_dict[col].transform(data_dict['X_train'][col])
        label_X_valid[col] = le_dict[col].transform(data_dict['X_val'][col])

    data_dict['X_train'] = label_X_train
    data_dict['X_val'] = label_X_valid

    return data_dict, le_dict

def score_model(y_valid, predictions, log=False):
    if log == False:
        score = np.sqrt(mean_squared_error(np.log(y_valid), np.log(predictions)))
    elif log == True:
        score = np.sqrt(mean_squared_error(y_valid, predictions))
    return score

def load_data(features, target):
    """
    Returns a dictionary with all the data and the various components for the
    training set
    """
    training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    ## Drop the rows in the entire dataset where the target has no value
    training_data.dropna(subset= [target], inplace=True)

    ##Drop outliers from training set
    training_data = training_data.drop(training_data[training_data['Id'] == 1299].index)
    training_data = training_data.drop(training_data[training_data['Id'] == 524].index)

    training_data['SalePrice'] = np.log(training_data['SalePrice'])
    training_data['GrLivArea'] = np.log(training_data['GrLivArea'])

    ## Set index
    training_data.set_index('Id')

    y = training_data[target]
    X = training_data[features]

    X['HasBsmt'] = pd.Series(len(X['TotalBsmtSF']), index=X.index)
    X['HasBsmt'] = 0
    X.loc[X['TotalBsmtSF']>0,'HasBsmt'] = 1
    X.loc[X['HasBsmt']==1,'TotalBsmtSF'] = np.log(X['TotalBsmtSF'])

    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 0)

    training_dict = {}

    training_dict['X_total'] = X
    training_dict['y_total'] = y
    training_dict['X_train'] = train_X
    training_dict['X_val'] = val_X
    training_dict['y_train'] = train_y
    training_dict['y_val'] = val_y

    return training_dict

def create_submission(filename, preds_test, X_test):
    submission = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
    submission.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

def load_test():
    X_test =  pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), index_col='Id')
    return X_test

def onehot_encode_objects(data_dict, object_cols):
    """
    One-hot enocode the object variables
    """

    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(data_dict['X_train'][object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(data_dict['X_val'][object_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = data_dict['X_train'].index
    OH_cols_valid.index = data_dict['X_val'].index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = data_dict['X_train'].drop(object_cols, axis=1)
    num_X_valid = data_dict['X_val'].drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    data_dict['X_train'] = OH_X_train
    data_dict['X_val'] = OH_X_valid

    return data_dict, OH_encoder

def binary_pool(dataframe):
    dataframe['has_pool'] = dataframe.apply(lambda x: 0 if x['PoolArea'] == 0 else 1, axis = 1)
    return dataframe

def replace_nan_with_none(dataframe, features):
    for col in features:
        dataframe[col] = dataframe[col].fillna('none')
    return dataframe

def regression_forest():
    """
    Simple Regression forest using both numerical and categorical variables. In this implementation, there is no target contamination (not data on the sale of the houses) and no test pool contamination (only data in the test set is used to model the regression). This differs from many other analyses which combine the test and the trianing set to preprocess the data
    """

    ## Define the target.
    target = "SalePrice"

    ## Define the features to be used.
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

    ## Load the data of the training set, split into training and validation
    ## subsets. Store these in a dictionary.
    training_dict = load_data(features, target)

    ## Feature engineering
    training_dict['X_train'] = binary_pool(training_dict['X_train'])
    training_dict['X_val'] = binary_pool(training_dict['X_val'])
    training_dict['X_train'] = replace_nan_with_none(training_dict['X_train'], label_enc_features)
    training_dict['X_val'] = replace_nan_with_none(training_dict['X_val'], label_enc_features)

    print(training_dict['X_train'].columns)

    ## Preprocessing training data.
    training_dict, le_dict = label_encode_objects(training_dict, label_enc_features)
    training_dict, one_hot_encoder = onehot_encode_objects(training_dict, oh_features)

    # plot_score(training_dict)

    ## Define the model
    model = RandomForestRegressor(n_estimators=241)
    model.fit(training_dict['X_train'], training_dict['y_train'])

    ## Predict and score the validation set. Print the score.
    preds = model.predict(training_dict['X_val'])
    print(round(score_model(training_dict['y_val'], preds, log=True), 4))

    ## Load and process the test set with the same features as the training set.
    X_test = load_test()[features]
    X_test = binary_pool(X_test)
    X_test = replace_nan_with_none(X_test, label_enc_features)
    X_test['GarageCars'].fillna(0, inplace=True)
    X_test['GarageCars'] = X_test['GarageCars'].astype(int)
    X_test['GrLivArea'] = np.log(X_test['GrLivArea'])

    X_test.loc[X_test['TotalBsmtSF'].isna(), 'TotalBsmtSF'] = 0

    X_test['HasBsmt'] = pd.Series(len(X_test['TotalBsmtSF']), index=X_test.index)
    X_test['HasBsmt'] = 0
    X_test.loc[X_test['TotalBsmtSF']>0,'HasBsmt'] = 1
    X_test.loc[X_test['HasBsmt']==1,'TotalBsmtSF'] = np.log(X_test['TotalBsmtSF'])

    print(X_test.columns)

    label_X_test = X_test.copy()
    for col in label_enc_features:
        print(col)
        print(le_dict[col].classes_)
        label_X_test[col] = le_dict[col].transform(X_test[col])
    X_test = label_X_test

    oh_cols_test = pd.DataFrame(one_hot_encoder.transform(X_test[oh_features]))
    oh_cols_test.index = X_test.index
    num_X_test = X_test.drop(oh_features, axis=1)
    oh_X_test = pd.concat([num_X_test, oh_cols_test], axis=1)
    X_test = oh_X_test

    ## Predict the test set targets using the generated model.
    preds_test = model.predict(X_test)
    final_predicitons = np.exp(preds_test)
    print(final_predicitons)

    ## Create csv file for submission.
    # create_submission('forest_model_log_transformed.csv', final_predicitons, X_test)

def main():
    regression_forest()
    pass



if __name__ == "__main__":
    main()
