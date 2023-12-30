"""
train, validate, and save lightGBM models with randomly selected hyperparameters
manually analyse and restrict hyperparameter space
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import commons
import commons_lgb

df_X_train = pd.read_parquet("data/X_train_selected_features.parquet")
y_train = pd.read_csv("data/y_train.csv").squeeze("columns")
era_validate = pd.read_csv("data/era_validate.csv").squeeze("columns")
df_X_validate = pd.read_parquet("data/X_validate_selected_features.parquet")
y_validate = pd.read_csv("data/y_validate.csv").squeeze("columns")


def mean_correlation_by_era_loss(prediction: np.ndarray, _data_lgb: lgb.Dataset) -> tuple[str, float, bool]:
    return commons_lgb.mean_correlation_by_era_lgb_loss(pd.Series(prediction), _data_lgb, era_validate)


spaces = [[.005, .01],
          list(range(3, 11)),
          [2 ** x - 1 for x in list(range(3, 9))],
          [.4, .5, .6, .7, .8, .9, 1],
          [.2, .3, .4, .5, .6, .7, .8, .9, 1],
          [100, 500, 1000, 5000, 10000, 50000],
          [0, .00001, .0001, .001, .005, .01, .1, 1, 10, 100, 1000, 10000],
          [0, .00001, .0001, .001, .005, .01, .1, 1, 10],
          [0, .00001, .0001, .001, .005, .01, .1, 1, 10, 100, 1000, 10000, 100000]]
hyperparameter_space = dict(zip(commons.LGB_PARAMETER_NAMES, spaces))

# result = []
# i = 0
# if already saved:
result = commons.load_pickle("data/lgb_result.pkl")
i = len(result)

while i < 150:
    parameters = commons.sample_parameters(hyperparameter_space)
    parameters = commons_lgb.add_default_lgb_parameters(parameters)

    data_lgb = lgb.Dataset(df_X_train, label=y_train)
    data_validate_lgb = lgb.Dataset(df_X_validate, label=y_validate)

    callbacks = [
        early_stopping(stopping_rounds=100, first_metric_only=True, verbose=False),
        log_evaluation(100)
    ]

    model = lgb.train(params=parameters,
                      train_set=data_lgb,
                      valid_sets=[data_validate_lgb],
                      num_boost_round=10000,
                      feval=mean_correlation_by_era_loss,
                      callbacks=callbacks
                      )

    best_score = model.best_score.get("valid_0")["mean_correlation_by_era"]
    result.append(list(parameters.values())[:-4] + [model.best_iteration, best_score, i])

    model.save_model("models/lgb_model_" + str(i) + ".txt")
    commons.save_as_pickle(result, "data/lgb_result.pkl")

    commons.print_time_name_and_number("model", i)
    i += 1

df_result = commons.get_result_as_dataframe(result, commons.LGB_PARAMETER_NAMES)
df_result.to_parquet("data/lgb_result.parquet")

df_result = commons.get_constrained_result_dataframe(df_result, hyperparameter_space)
commons.plot_score_by_parameter_spaces(df_result)
