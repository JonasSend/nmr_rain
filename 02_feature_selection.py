"""
train on features + shadow features
shuffle each feature and predict to measure predictive performance
only keep features that on average perform better than the best shadow feature
"""

import pandas as pd
import lightgbm as lgb
import commons
import commons_lgb

features = commons.load_pickle("data/feature_names.pkl")

df_X_train = pd.read_parquet("data/X_train_with_shadow_features.parquet")
y_train = pd.read_csv("data/y_train.csv").squeeze("columns")
era_validate = pd.read_csv("data/era_feature_selection.csv").squeeze("columns")
df_X_validate = pd.read_parquet("data/X_feature_selection_with_shadow_features.parquet")
y_validate = pd.read_csv("data/y_feature_selection.csv").squeeze("columns")


def mean_correlation_by_era_loss(_prediction: pd.Series, _data_lgb: lgb.Dataset) -> (str, float, bool):
    return commons_lgb.mean_correlation_by_era_lgb_loss(_prediction, _data_lgb, era_validate)


# hyperparameter space that has proven performant in previous experiments
spaces = [[.01, .02],
          list(range(10, 17)) + [-1],
          [2 ** x - 1 for x in list(range(8, 17))],
          [.5, .6, .7, .8, .9, 1],
          [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
          [200, 500, 1000, 2000, 5000],
          [0, .0001, .001, .005, .01, .1, 1, 10, 100, 1000],
          [0, .0001, .001, .01, .1, 1, 10],
          [0, .0001, .001, .01, .1, 1, 10, 100, 1000, 10000]]
hyperparameter_space = dict(zip(commons.LGB_PARAMETER_NAMES, spaces))

df_features = pd.DataFrame({"feature": features})

for i in range(25, 40):  # FIXME
    parameters = commons.sample_parameters(hyperparameter_space)
    parameters = commons.add_default_lgb_parameters(parameters)

    data_lgb = lgb.Dataset(df_X_train, label=y_train)
    data_validate_lgb = lgb.Dataset(df_X_validate, label=y_validate)

    model = lgb.train(parameters,
                      data_lgb,
                      valid_sets=data_validate_lgb,
                      num_boost_round=5000,
                      feval=mean_correlation_by_era_loss,
                      early_stopping_rounds=25,
                      verbose_eval=True
                      )

    result_name = "model_" + str(i)
    df_features[result_name] = -1

    for k in range(len(features)):
        feature = features[k]
        feature_copy = df_X_validate[feature].copy()
        df_X_validate[feature] = df_X_validate[feature].sample(frac=1).values

        prediction = pd.Series(model.predict(df_X_validate))

        score = commons.mean_grouped_spearman_correlation(prediction, y_validate, era_validate)
        df_features[result_name][df_features["feature"] == feature] = model.best_score.get("valid_0")[
                                                                          "mean_correlation_by_era"] - score

        df_X_validate[feature] = feature_copy
        if k % 100 == 0:
            commons.print_time_name_and_number("feature", k)

    df_features.to_pickle("data/feature_selection_result.pkl")

    commons.print_time_name_and_number("model", i)

df_features["avg"] = df_features.loc[:, df_features.columns != "feature"].mean(axis=1)
df_features["select"] = df_features["avg"] > max(df_features["avg"][-100:])
feature_selection_list = list(df_features["feature"][df_features["select"]])

commons.save_as_pickle(feature_selection_list, "data/feature_selection_list.pkl")

for name in ["train", "validate", "test"]:
    df_X = pd.read_parquet("data/X_" + name + ".parquet")
    df_X = df_X[feature_selection_list]
    df_X.to_parquet("data/X_" + name + "_selected_features.parquet")
