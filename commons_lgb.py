import commons
import pandas as pd
import lightgbm as lgb
import numpy as np


def add_default_lgb_parameters(parameters: dict) -> dict:
    parameters["num_threads"] = 12
    parameters["bagging_freq"] = 1
    parameters["verbose"] = -1
    parameters["metric"] = "None"
    return parameters


def mean_correlation_by_era_lgb_loss(prediction: pd.Series, data_lgb: lgb.Dataset, era: pd.Series) -> (
        str, float, bool):
    labels = data_lgb.get_label()
    correlation = commons.mean_grouped_spearman_correlation(prediction, labels, era)
    return "mean_correlation_by_era", correlation, True


def mean_correlation_minus_ensemble_correlation_by_era_lgb_loss(
        prediction: pd.Series, ensemble_prediction: pd.Series, data_lgb: lgb.Dataset,
        era: pd.Series, weight_ensemble_correlation: int) -> (
        str, float, bool):
    labels = data_lgb.get_label()
    correlation = commons.mean_grouped_spearman_correlation(prediction, labels, era)
    correlation_ensemble_predictions = commons.mean_grouped_spearman_correlation(prediction, ensemble_prediction, era)
    mean_correlation_minus_ensemble_correlation_by_era = correlation - correlation_ensemble_predictions*weight_ensemble_correlation
    return "mean_correlation_minus_ensemble_correlation_by_era", mean_correlation_minus_ensemble_correlation_by_era, True


# unused - does not seem to perform well, unfortunately
# can't be used on first round
def pearson_correlation_loss(y_pred, dtrain):
    y_true = dtrain.get_label()
    N = len(y_pred)
    
    x_mean = np.mean(y_pred)
    y_mean = np.mean(y_true)
    sd_x = np.std(y_pred)
    sd_y = np.std(y_true)
    cov_xy = np.cov(y_pred, y_true, bias = True)[0, 1]
    
    gradient = -1/sd_y * ((y_true-y_mean) - cov_xy*(y_pred-x_mean)/sd_x**2)/sd_x
    hessian = -1/(N*sd_y) * (3*cov_xy*(y_pred-x_mean)**2/sd_x**2 - 2*(y_pred-x_mean)*(y_true-y_mean) - cov_xy*(N-1))/sd_x**3

    return gradient, hessian


# unused
def l2_loss(y, data):
    t = data.get_label()
    grad = y - t 
    hess = np.ones_like(y)
    return grad, hess
