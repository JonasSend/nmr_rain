"""
train, validate, and save keras/tensorflow models with randomly selected hyperparameters
manually analyse and restrict hyperparameter space
"""

import pandas as pd
import commons
import commons_tf

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
          [0, .00001, .0001, .001, .01, .1]]
hyperparameter_space = dict(zip(commons.KERAS_PARAMETER_NAMES, spaces))

# result = []
# i = 0
# if already saved:
result = commons.load_pickle("data/keras_result.pkl")
i = len(result)

while i < 150:
    parameters = commons.sample_parameters(hyperparameter_space)

    model = commons_tf.build_and_compile_keras_model(df_X_train.shape[1], parameters)
    callbacks = [commons_tf.CustomCallback(df_X_validate, y_validate, era_validate,
                                           stop=10, max_epochs=500, index=i, description="")]

    model.fit(
        df_X_train, y_train,
        epochs=1000,
        batch_size=parameters["batch_size"],
        callbacks=callbacks,
        verbose=0
    )

    result.append(list(parameters.values()) + [callbacks[0].best_epoch, callbacks[0].best_score, i])
    commons.save_as_pickle(result, "data/keras_result.pkl")

    commons.print_time_name_and_number("model", i)
    i += 1

df_result = commons.get_result_as_dataframe(result, commons.KERAS_PARAMETER_NAMES)
df_result.to_parquet("data/keras_result.parquet")

df_result = commons.get_constrained_result_dataframe(df_result, hyperparameter_space)
commons.plot_score_by_parameter_spaces(df_result)
