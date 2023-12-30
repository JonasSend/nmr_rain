"""
train, validate, and save keras models with randomly selected hyperparameters
manually analyse and restrict hyperparameter space
regularly restrict the training set on the half (era-wise) for which the predictive performance is worse
"""

import pandas as pd
import numpy as np
import commons
import commons_tf
import random

era_train = pd.read_csv("data/era_train.csv").squeeze("columns")
df_X_train = pd.read_parquet("data/X_train_selected_features.parquet")
y_train = pd.read_csv("data/y_train.csv").squeeze("columns")
era_validate = pd.read_csv("data/era_validate.csv").squeeze("columns")
df_X_validate = pd.read_parquet("data/X_validate_selected_features.parquet")
y_validate = pd.read_csv("data/y_validate.csv").squeeze("columns")

spaces = [[.000001, .000005, .00001],
          [1, 2, 3, 4],
          [1000, 2000, 5000, 10000],
          [10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
          [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
          [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
          [100, 200, 500, 1000],
          [.1, .2, .5],
          [0, .05, .1, .2, .5],
          [0, .00001, .0001],
          [.00001, .0001, .001, .01, .1]]
hyperparameter_space = dict(zip(commons.KERAS_PARAMETER_NAMES, spaces))
interval = 2

# result = []
# i = 0
# if already saved:
result = commons.load_pickle("data/keras_era_boosting_result.pkl")
i = len(result)

while i < 75:
    parameters = commons.sample_parameters(hyperparameter_space)

    model = commons_tf.build_and_compile_keras_model(df_X_train.shape[1], parameters)
    # stopping handled via era boosting
    callbacks = [commons_tf.CustomCallback(df_X_validate, y_validate, era_validate,
                                           stop=1000, max_epochs=1000, index=i, description="era_boosting_")]


    def train_model(df_X: pd.DataFrame, y: pd.Series) -> None:
        model.fit(
            df_X, y,
            epochs=interval,
            batch_size=parameters["batch_size"],
            callbacks=callbacks,
            verbose=0
        )


    df_X_subset = pd.DataFrame()
    y_subset = pd.Series(dtype="float64")
    best_score = 0
    best_epoch = 0
    counter = 0

    for j in range(0, round(500 / interval)):
        if j == 0:
            train_model(df_X_train, y_train)
        else:
            train_model(df_X_subset, y_subset)

        current_best_score = callbacks[0].best_score
        if current_best_score > best_score:
            best_epoch = j * interval + (callbacks[0].best_epoch + 1)
            best_score = current_best_score
            counter = 0
        else:
            counter += 1

        if counter >= round(10 / interval):
            break

        prediction = pd.Series((model.predict(df_X_train, verbose=0).flatten()))
        correlation_by_era = commons.grouped_spearman_correlation(prediction, y_train, era_train)
        correlation_by_era = correlation_by_era.apply(lambda x: random.uniform(-.01, .01) if np.isnan(x) else x)

        era_subset = list(correlation_by_era[correlation_by_era <= correlation_by_era.median()].index)
        df_X_subset = df_X_train[era_train.isin(era_subset)]
        y_subset = y_train[era_train.isin(era_subset)]

        commons.print_time_name_and_number("interval", j)

    result.append(list(parameters.values()) + [best_epoch, best_score, i])
    commons.save_as_pickle(result, "data/keras_era_boosting_result.pkl")

    commons.print_time_name_and_number("model", i)
    i += 1

df_result = commons.get_result_as_dataframe(result, commons.KERAS_PARAMETER_NAMES)
df_result.to_parquet("data/keras_era_boosting_result.parquet")

df_result = commons.get_constrained_result_dataframe(df_result, hyperparameter_space)
commons.plot_score_by_parameter_spaces(df_result)
