"""
download training and validation data and merge the two datasets
save data as era, target, and feature matrix
"""

import pandas as pd
from numerapi import NumerAPI
import os
from typing import List

numeraiAPI = NumerAPI()
if os.path.exists("data/train.parquet"):
    os.remove("data/train.parquet")
if os.path.exists("data/validate.parquet"):
    os.remove("data/validate.parquet")

numeraiAPI.download_dataset("v4.2/train_int8.parquet", "data/train.parquet")
numeraiAPI.download_dataset("v4.2/validation_int8.parquet", "data/validate.parquet")

df_train = pd.read_parquet("data/train.parquet") # eras 1-574
df_validate = pd.read_parquet("data/validate.parquet") # eras 575-1080+
df_validate = df_validate[~df_validate["target"].isna()]

# split each "year" of the validation set into 3 and add to the separate validation sets
df_validate["era"] = df_validate["era"].astype('int16')
era_validate_unique = df_validate["era"].unique()
min_validation_era = min(era_validate_unique)
max_validation_era = max(era_validate_unique)
era_feature_selection = []
era_validate = []
era_test = []
for i in range(min_validation_era, max_validation_era, 51):
    era_feature_selection.extend(list(range(i, i+17)))
    era_validate.extend(list(range(i+17, i+34)))
    era_test.extend(list(range(i+34, i+51)))


def safe_data(name: str, _df: pd.DataFrame, era: List[int]) -> None:
    _df = _df[_df["era"].isin(era)]
    
    _era = _df["era"]
    _y = _df["target"]
    _df.drop(list(_df.filter(regex="target").columns) + ["data_type", "era"], axis=1, inplace=True)

    _era.to_csv("data/era_" + name + ".csv", index=False)
    _y.to_csv("data/y_" + name + ".csv", index=False)
    _df.to_parquet("data/X_" + name + ".parquet")


safe_data("train", df_train, df_train["era"].unique())
safe_data("feature_selection", df_validate, era_feature_selection)
safe_data("validate", df_validate, era_validate)
safe_data("test", df_validate, era_test)
