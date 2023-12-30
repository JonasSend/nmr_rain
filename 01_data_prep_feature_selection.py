"""
create shadow features for the train and respective validate sets
"""

import pandas as pd
import random
import commons

era_train = pd.read_csv("data/era_train.csv").squeeze("columns")
df_X_train = pd.read_parquet("data/X_train.parquet")
y_train = pd.read_csv("data/y_train.csv").squeeze("columns")

era_validate = pd.read_csv("data/era_feature_selection.csv").squeeze("columns")
df_X_validate = pd.read_parquet("data/X_feature_selection.parquet")
y_validate = pd.read_csv("data/y_feature_selection.csv").squeeze("columns")

feature_names = list(df_X_train.columns)


def add_and_save_shadow_features(name: str, _df: pd.DataFrame, _feature_names: list[str]) -> None:
    for i in _feature_names:
        _df[i + "_shadow"] = _df[i].copy().sample(frac=1).values
    
    _df.to_parquet("data/X_" + name + "_with_shadow_features.parquet")


# Usually, we would use the same number of shadow features as regular features,
# but we restrict the number of shadow features to 100 to save memory and time.
random_features = random.sample(feature_names, 100)
feature_names_with_shadow_features = feature_names + random_features
commons.save_as_pickle(feature_names_with_shadow_features, "data/feature_names.pkl")

add_and_save_shadow_features("train", df_X_train, random_features)
add_and_save_shadow_features("feature_selection", df_X_validate, random_features)
