import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

def impute_numerical(dataframe_dict, strategy):
    my_imputer = SimpleImputer(strategy=strategy)

    imputed_train_X = pd.DataFrame(my_imputer.fit_transform(dataframe_dict['train_X']))
    imputed_val_X = pd.DataFrame(my_imputer.transform(dataframe_dict['val_X']))

    test_imputer = my_imputer
    # print("Imputed dataframe")
    # print(imputed_dataframe)

    # Imputation removed column names; put them back
    imputed_train_X.columns = dataframe_dict['train_X'].columns
    imputed_val_X.columns = dataframe_dict['train_X'].columns

    dataframe_dict['train_X'] = imputed_train_X
    dataframe_dict['val_X'] = imputed_val_X

    return dataframe_dict, test_imputer

def score_model(y_valid, predictions):
    return np.sqrt(mean_squared_error(np.log(y_valid), np.log(predictions)))

def load_data(features, target):
    """
    Returns a dictionary with all the data and the various components for the
    training set
    """
    training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col='Id')
    ## Drop the rows in the entire dataset where the target has no value
    training_data.dropna(subset= [target], inplace=True)
    y = training_data[target]
    X = training_data[features]

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

def one_stat_submission(statistic = "mean"):
    """
    Create submission which just returns a submission with a single statistic of the
    target in the training set.
    """
    ## Load the data and calculate the statitic for the SalePrice
    training = load_data([] , "SalePrice")
    statistic = training['y_total'].describe()[statistic]

    # predictions_val = pd.DataFrame(index=training['y_val'].index)
    # predictions_val['SalePrice'] = statistic


    X_test = load_test()
    test_preds = pd.DataFrame({'Id': X_test.index, 'SalePrice': statistic})
    test_preds.to_csv(os.path.join(OUTPUT_DIR, "mean_submission.csv"), index=False)

    # print(score_model(training['y_val'], predictions_val))


def drop_na_rows(data_dict, column):
    """
    Clean data for analysis by dropping all rows with nan values in a column. Takes
    in the split data dictionary and outputs the new one with dropped values.
    """
    ## Get a list of the columns with nan values.
    # na_col_list = [col for col in data_dict['train_X'].columns if data_dict['train_X'][col].isnull().any()]

    ## Remove these columns from the training and validation data.
    X_train_na_list = data_dict['X_train'][data_dict['X_train'][column].isnull()].index.tolist()
    X_val_na_list = data_dict['X_val'][data_dict['X_val'][column].isnull()].index.tolist()
    reduced_X_train = data_dict['X_train'].dropna(subset=[column], axis=0)
    reduced_X_val = data_dict['X_val'].dropna(subset=[column], axis=0)
    reduced_y_train = data_dict['y_train'].drop(X_train_na_list)
    reduced_y_val = data_dict['y_val'].drop(X_val_na_list)



    data_dict['X_train'] = reduced_X_train
    data_dict['y_train'] = reduced_y_train
    data_dict['X_val'] = reduced_X_val
    data_dict['y_val'] = reduced_y_val

    return data_dict

def linear_submission():
    """
    Creates a prediction based on only a single feature. That is, simple 1D linear regression.
    """
    training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col='Id')
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    list_numeric = training_data.select_dtypes(include=numerics).columns

    training = load_data(list_numeric, 'SalePrice')

    # print(training)
    regressor = LinearRegression()

    lin_regress = open(os.path.join(OUTPUT_DIR, 'lin_regress_results.md'), 'w')
    lin_regress.write("## This contains the results of the linear regression scores of all the numerical \
features \n")

    for feature in list_numeric:#["LotFrontage", "LotArea", 'OverallQual', "OverallCond"]:
        training = drop_na_rows(training, feature)
        print(feature)
        X = training['X_train'][feature].values.reshape(-1,1)
        y = training['y_train'].values.reshape(-1,1)
        X_val = training['X_val'][feature].values.reshape(-1,1)
        y_val = training['y_val'].values.reshape(-1,1)
        regressor.fit(X, y)
        y_pred = regressor.predict(X_val)
        print(score_model(training['y_val'], y_pred))
        lin_regress.write(feature + " : " + str(round(score_model(training['y_val'], y_pred), 4)) + ' \n')
    lin_regress.close()

    X_test = load_test()
    print(X_test['OverallQual'].unique())
    X = training['X_train']["OverallQual"].values.reshape(-1,1)
    y = training['y_train'].values.reshape(-1,1)
    X_test_reshaped = X_test['OverallQual'].values.reshape(-1,1)
    regressor.fit(X, y)
    predictions = regressor.predict(X_test_reshaped).flatten()

    print(type(predictions))
    print(predictions)


    test_preds = pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions})
    test_preds.to_csv(os.path.join(OUTPUT_DIR, "lin_Overval_submission.csv"), index=False)

def main():
    # one_stat_submission("50%")
    linear_submission()
    # training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col='Id')
    # training_data_index = training_data[training_data['LotFrontage'].isnull()].index.tolist()
    # print(len(training_data_index))
    pass



if __name__ == "__main__":
    main()
