"""
use the best models from the [lightGBM, keras] x [regular, era-boosting] approaches
train an ensemble on the validation set and use the performance on the test set as stopping condition
remove the multiplied linear component from validation and test predictions (neutralising) and find best-performing
multiplier
"""

import pandas as pd
import lightgbm as lgb
from tensorflow import keras
import statsmodels.api as sm
import commons
from commons import normalise
from commons import mean_grouped_spearman_correlation
import commons_tf
import tensorflow as tf

era_validate = pd.read_csv("data/era_validate.csv").squeeze("columns")
df_X_validate = pd.read_parquet("data/X_validate_selected_features.parquet")
y_validate = pd.read_csv("data/y_validate.csv").squeeze("columns")
era_test = pd.read_csv("data/era_test.csv").squeeze("columns")
df_X_test = pd.read_parquet("data/X_test_selected_features.parquet")
y_test = pd.read_csv("data/y_test.csv").squeeze("columns")
era_set_validate = era_validate.unique()
era_set_test = era_test.unique()

number_of_models = 25


def get_model_indices_from_result_df(name: str) -> list[int]:
    _df = pd.read_parquet("data/" + name + "_result.parquet")
    return _df.sort_values("best_score", ascending=False)["index"].iloc[0:number_of_models].tolist()


lgb_models = get_model_indices_from_result_df("lgb")
lgb_era_boosting_models = get_model_indices_from_result_df("lgb_era_boosting")
keras_models = get_model_indices_from_result_df("keras")
keras_era_boosting_models = get_model_indices_from_result_df("keras_era_boosting")

prediction_validate = []
prediction_test = []


def load_lgb_models_and_predict(name: str, model_indices: list[int]) -> None:
    for j in model_indices:
        model = lgb.Booster(model_file="models/" + name + "_model_" + str(j) + ".txt")
        prediction_validate.append(pd.Series(model.predict(df_X_validate)))
        prediction_test.append(pd.Series((model.predict(df_X_test).flatten())))
        commons.print_time_name_and_number(name + " model", j)


load_lgb_models_and_predict("lgb", lgb_models)
load_lgb_models_and_predict("lgb_era_boosting", lgb_era_boosting_models)


def load_keras_models_and_predict(name: str, model_indices: list[int]) -> None:
    for j in model_indices:
        model = keras.models.load_model("models/" + name + "_model_" + str(j) + ".h5",
                                        custom_objects={
                                            "pearson_correlation_loss": commons_tf.pearson_correlation_loss})
        prediction_validate.append(pd.Series((model.predict(df_X_validate).flatten())))
        prediction_test.append(pd.Series((model.predict(df_X_test).flatten())))
        commons.print_time_name_and_number(name + " model", j)


load_keras_models_and_predict("keras", keras_models)
load_keras_models_and_predict("keras_era_boosting", keras_era_boosting_models)

df_prediction_validate = pd.DataFrame(prediction_validate).transpose()
df_prediction_test = pd.DataFrame(prediction_test).transpose()

df_prediction_validate = df_prediction_validate.groupby(era_validate).transform(lambda x: normalise(x))
df_prediction_test = df_prediction_test.groupby(era_test).transform(lambda x: normalise(x))

best_model = -1
best_score_validation = -1
for i in range(number_of_models * 4):
    _score_validation = mean_grouped_spearman_correlation(df_prediction_validate[i], y_validate, era_validate)
    if _score_validation > best_score_validation:
        best_score_validation = _score_validation
        best_model = i
    commons.print_time_name_and_number("model", i)

prediction_validate_ensemble = df_prediction_validate[best_model]
prediction_test_ensemble = df_prediction_test[best_model]
best_score_test = mean_grouped_spearman_correlation(prediction_test_ensemble, y_test, era_test)

stop_counter = 0
number_of_ensembled_models = 1
ensembled_models = [best_model]

while stop_counter < 5:
    _best_score_validation = 0
    _best_score_test = 0
    add = -1

    for i in range(number_of_models * 4):
        _prediction_validate_ensemble = (prediction_validate_ensemble * number_of_ensembled_models / (
                    number_of_ensembled_models + 1)) + (
                                                df_prediction_validate[i] / (number_of_ensembled_models + 1))
        _score_validation = mean_grouped_spearman_correlation(_prediction_validate_ensemble, y_validate, era_validate)
        if _score_validation > _best_score_validation:
            _best_score_validation = _score_validation
            add = i

    prediction_validate_ensemble = (prediction_validate_ensemble * number_of_ensembled_models / (
                number_of_ensembled_models + 1)) + (
                                           df_prediction_validate[add] / (number_of_ensembled_models + 1))
    prediction_test_ensemble = (prediction_test_ensemble * number_of_ensembled_models / (
                number_of_ensembled_models + 1)) + (
                                       df_prediction_test[add] / (number_of_ensembled_models + 1))
    _best_score_test = mean_grouped_spearman_correlation(prediction_test_ensemble, y_test, era_test)
    ensembled_models.append(add)
    number_of_ensembled_models = number_of_ensembled_models + 1

    if _best_score_test > best_score_test:
        best_score_test = _best_score_test
        stop_counter = 0
    else:
        stop_counter = stop_counter + 1

    commons.print_time_name_and_number("added model", add)

ensembled_models = ensembled_models[:-5]
number_of_ensembled_models = len(ensembled_models)
prediction_validate_ensemble = sum([df_prediction_validate[i] for i in ensembled_models]) / number_of_ensembled_models
prediction_test_ensemble = sum([df_prediction_test[i] for i in ensembled_models]) / number_of_ensembled_models

# save predictions for later use as meta-model proxy predictions
normalise(prediction_validate_ensemble).to_csv("data/ensemble_prediction_validate.csv", index=False)
normalise(prediction_test_ensemble).to_csv("data/ensemble_prediction_test.csv", index=False)

df_X_validate["prediction"] = prediction_validate_ensemble
df_X_test["prediction"] = prediction_test_ensemble

# delete quotes from column names to make regression syntax work
df_X_validate.columns = [x.replace("'", "") for x in df_X_validate.columns]
df_X_test.columns = [x.replace("'", "") for x in df_X_test.columns]

linear_component_validate = pd.Series(dtype="float16")
linear_component_test = pd.Series(dtype="float16")


# FIXME: move to commons
def get_linear_component(_df: pd.DataFrame) -> pd.Series:
    y = _df["prediction"]
    x = _df.drop("prediction", axis=1)
    x = sm.add_constant(x)
    lm = sm.OLS(y, x).fit()
    return pd.Series(lm.predict())


for era in era_set_validate:
    df = df_X_validate[era_validate == era]
    linear_component_validate = pd.concat([linear_component_validate, get_linear_component(df)])
    commons.print_time_name_and_number("era:", era)

linear_component_validate.reset_index(drop=True, inplace=True)
df_X_validate["linear_component"] = linear_component_validate

for era in era_set_test:
    df = df_X_test[era_test == era]
    linear_component_test = pd.concat([linear_component_test, get_linear_component(df)])
    commons.print_time_name_and_number("era:", era)

linear_component_test.reset_index(drop=True, inplace=True)
df_X_test["linear_component"] = linear_component_test


# FIXME: move to commons
def get_neutralised_predictions(_df: pd.DataFrame, _m: float) -> pd.Series:
    return normalise(normalise(_df["prediction"]) - (normalise(_df["linear_component"]) * _m))


result = []
for m in [x / 20 for x in range(0, 16)]:
    prediction_neutral_validate = pd.Series(dtype="float16")
    for era in era_set_validate:
        _df_X_validate = df_X_validate[era_validate == era]
        prediction_neutral_validate = pd.concat(
            [prediction_neutral_validate, get_neutralised_predictions(_df_X_validate, m)])

    prediction_neutral_test = pd.Series(dtype="float16")
    for era in era_set_test:
        _df_X_test = df_X_test[era_test == era]
        prediction_neutral_test = pd.concat([prediction_neutral_test, get_neutralised_predictions(_df_X_test, m)])

    result.append([m, mean_grouped_spearman_correlation(prediction_neutral_validate, y_validate, era_validate),
                   mean_grouped_spearman_correlation(prediction_neutral_test, y_test, era_test)])

    commons.print_time_name_and_number("multiplier:", m)

df_result = pd.DataFrame(result, columns=["multiplier", "score_on_val", "score_on_test"])
df_result["average_score"] = df_result[["score_on_val", "score_on_test"]].mean(axis=1)
# best for lgb only: .15


def print_model_indices(models: list[int], model_type: int) -> None:
    print([models[i] for i in [x - (number_of_models * model_type) for x in ensembled_models if
                               number_of_models * (model_type + 1) > x >= (number_of_models * model_type)]])


print_model_indices(lgb_models, 0)
print_model_indices(lgb_era_boosting_models, 1)
print_model_indices(keras_models, 2)
print_model_indices(keras_era_boosting_models, 3)
