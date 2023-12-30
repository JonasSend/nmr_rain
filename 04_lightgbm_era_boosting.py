"""
train, validate, and save lightGBM models with randomly selected hyperparameters
manually analyse and restrict hyperparameter space
regularly restrict the training set on the half (era-wise) for which the predictive performance is worse
"""

import pandas as pd
import lightgbm as lgb
from lightgbm import log_evaluation, record_evaluation
import numpy as np
import random
import commons
import commons_lgb

era_train = pd.read_csv("data/era_train.csv").squeeze("columns")
df_X_train = pd.read_parquet("data/X_train_selected_features.parquet")
y_train = pd.read_csv("data/y_train.csv").squeeze("columns")
era_validate = pd.read_csv("data/era_validate.csv").squeeze("columns")
df_X_validate = pd.read_parquet("data/X_validate_selected_features.parquet")
y_validate = pd.read_csv("data/y_validate.csv").squeeze("columns")


def mean_correlation_by_era_loss(_prediction: np.ndarray, _data_lgb: lgb.Dataset) -> tuple[str, float, bool]:
    return commons_lgb.mean_correlation_by_era_lgb_loss(pd.Series(_prediction), _data_lgb, era_validate)


spaces = [[.005, .01],
          list(range(4, 11)),
          [2 ** x - 1 for x in list(range(4, 10))],
          [.4, .5, .6, .7, .8, .9, 1],
          [.2, .3, .4, .5, .6, .7, .8, .9, 1],
          [10, 100, 500, 1000, 5000, 10000, 50000],
          [0, .00001, .0001, .001, .005, .01, .1, 1, 10, 100, 1000, 10000],
          [0, .00001, .0001, .001, .005, .01, .1, 1, 10, 100],
          [0, .00001, .0001, .001, .005, .01, .1, 1, 10, 100, 1000, 10000, 100000]]
hyperparameter_space = dict(zip(commons.LGB_PARAMETER_NAMES, spaces))
interval = 25

# result = []
# i = 0
# if already saved:
result = commons.load_pickle("data/lgb_era_boosting_result.pkl")
i = len(result)

while i < 105:
    parameters = commons.sample_parameters(hyperparameter_space)
    parameters = commons_lgb.add_default_lgb_parameters(parameters)

    df_X_subset = pd.DataFrame()
    y_subset = pd.Series(dtype="float64")
    best_score = 0
    best_iteration = 0
    counter = 0


    def train_model(_data_lgb: lgb.basic.Dataset, _model: lgb.Booster) -> lgb.Booster:
        callbacks = [
            log_evaluation(interval),
            record_evaluation(score)
        ]
        return lgb.train(parameters,
                         _data_lgb,
                         init_model=_model,
                         valid_sets=[data_validate_lgb],
                         num_boost_round=interval,
                         feval=mean_correlation_by_era_loss,
                         callbacks=callbacks)


    for j in range(0, round(10000 / interval)):
        data_validate_lgb = lgb.Dataset(df_X_validate, label=y_validate)
        score = {}

        if j == 0:
            data_lgb = lgb.Dataset(df_X_train, label=y_train)
            # noinspection PyTypeChecker
            model = train_model(data_lgb, None)
        else:
            data_lgb = lgb.Dataset(df_X_subset, label=y_subset)
            # noinspection PyUnboundLocalVariable
            model = train_model(data_lgb, model)

        scores = score["valid_0"].get("mean_correlation_by_era")
        current_best_score = max(scores)
        if current_best_score > best_score:
            best_iteration = j * interval + (scores.index(current_best_score) + 1)
            best_score = current_best_score
            counter = 0
            model.save_model("models/lgb_era_boosting_model_" + str(i) + ".txt")
        else:
            counter += 1

        if counter >= round(100 / interval):
            break

        prediction = model.predict(df_X_train)
        correlation_by_era = commons.grouped_spearman_correlation(prediction, y_train, era_train)
        correlation_by_era = correlation_by_era.apply(lambda x: random.uniform(-.01, .01) if np.isnan(x) else x)

        era_subset = list(correlation_by_era[correlation_by_era <= correlation_by_era.median()].index)
        df_X_subset = df_X_train[era_train.isin(era_subset)]
        y_subset = y_train[era_train.isin(era_subset)]

    result.append(list(parameters.values())[:-4] + [best_iteration, best_score, i])
    commons.save_as_pickle(result, "data/lgb_era_boosting_result.pkl")

    commons.print_time_name_and_number("model", i)
    i += 1

df_result = commons.get_result_as_dataframe(result, commons.LGB_PARAMETER_NAMES)
df_result.to_parquet("data/lgb_era_boosting_result.parquet")

df_result = commons.get_constrained_result_dataframe(df_result, hyperparameter_space)
commons.plot_score_by_parameter_spaces(df_result)
