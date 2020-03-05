import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os

import scipy.stats as stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

## Feature importance modules
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, get_dataset, info_plots
# import shap ## Not avaliable on python 3.8

import xgboost as xgb
import warnings

def log_score(y_valid, predictions):
    return np.abs(np.sqrt(mean_squared_error(y_valid, predictions)))

def simple_score(y_valid, predictions):
    score = np.abs(np.sqrt(mean_squared_error(np.log(y_valid), np.log(predictions))))
    return score

def score_model(y_valid, predictions, log=False):
    """
    Scores the predictions against the known values for the validation set. The
    metric is the RMSE of the logarithms. If the target has been tranasformed the
    function will not apply the log for scoring. Otherwise it will.
    """
    if log == False:
        score = np.sqrt(mean_squared_error(np.log(y_valid), np.log(predictions)))
    elif log == True:
        score = np.sqrt(mean_squared_error(y_valid, predictions))
    return score

def perm_import(model, features, X_val, y_val):
    perm = PermutationImportance(model, random_state=1).fit(X_val, y_val)
    # eli5.show_weights(perm, feature_names = features)
    print(eli5.format_as_text(eli5.explain_weights(model)))
    pass

def part_plot_1D(model, total_features, X_val, y_val, feature):
    pdp_dist = pdp.pdp_isolate(model=model, dataset=X_val, model_features=total_features, feature=feature)
    pdp.pdp_plot(pdp_dist, feature)
    plt.show()

def part_plot_2D(model, total_features, X_val, y_val, feature1, feature2):
    inter1  =  pdp.pdp_interact(model=model, dataset=X_val, model_features=total_features, features=[feature1, feature2])
    pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=[feature1, feature2], plot_type='grid')
    plt.show()

def shap_values(prediction, model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(prediction)
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0], prediction)
