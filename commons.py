import pandas as pd
from typing import Any
import random
import math
import pickle
import numpy as np
from scipy.stats import norm
from datetime import datetime

NR_FOLDS: int = 3
ERAS_TO_PURGE: int = 3
LGB_PARAMETER_NAMES: list[str] = ["learning_rate", "max_depth", "num_leaves", "bagging_fraction",
                                  "feature_fraction", "min_data", "min_sum_hessian_in_leaf", "lambda_l1", "lambda_l2"]
KERAS_PARAMETER_NAMES: list[str] = ["learning_rate", "layers", "n1", "n2", "n3", "n4",
                                    "batch_size", "dropout_in", "dropout_hidden", "l_1", "l_2"]


def sample_parameters(space: dict[str, list[Any]]) -> dict[str, Any]:
    parameters: dict[str, Any] = dict()
    for key, value in space.items():
        if key == "num_leaves":
            parameters[key] = sample_leaves(parameters["max_depth"], value)
        elif key in ["n1", "n2", "n3", "n4"]:
            parameters[key] = sample_nodes(key, parameters, value)
        else:
            parameters[key] = sample_parameter(value)
    return parameters


def sample_parameter(space: list[Any]) -> Any:
    return random.choice(space)


def sample_leaves(depth: int, space: list[int]) -> int:
    if depth == -1:
        leaves = random.choice(space)
    elif depth == 2:
        leaves = 3
    else:
        leaves = random.choice(get_leaf_space(depth, space))
    return leaves


def get_leaf_space(depth: int, space: list[int]) -> list[int]:
    potential_leave_space: list[int] = [2 ** x - 1 for x in list(range(math.ceil(math.log(depth, 2)), depth + 1))]
    return list(set(space) & set(potential_leave_space))


def sample_nodes(key: str, parameters: dict[str, Any], space: list[int]) -> int:
    """results in funnel-shaped neural nets"""
    if parameters["layers"] < int(key[1]):
        return -1
    if key == "n1":
        return random.choice(space)
    return random.choice([x for x in space if x <= parameters["n" + str(int(key[1]) - 1)]])


def mean_grouped_spearman_correlation(prediction: pd.Series, target: pd.Series, era: pd.Series) -> float:
    return grouped_spearman_correlation(prediction, target, era).mean()


def grouped_spearman_correlation(prediction: pd.Series, target: pd.Series, era: pd.Series) -> pd.Series:
    data: pd.DataFrame = pd.DataFrame({"era": era, "prediction": prediction, "target": target})
    return pd.Series([numerai_corr_for_df(data[data["era"] == i]) for i in data["era"].unique()])


def numerai_corr_for_df(df: pd.DataFrame) -> pd.Series:
    return numerai_corr(df["prediction"], df["target"])


# correlation function that puts more weight on the tails
# provided by numerai (see https://forum.numer.ai/t/target-cyrus-new-primary-target/6303)
def numerai_corr(prediction: pd.Series, target: pd.Series):
    ranked_prediction = (prediction.rank(method="average").values - 0.5) / prediction.count()
    gauss_ranked_prediction = norm.ppf(ranked_prediction)  # gaussianise predictions
    centered_target = (normalise(target) - 0.5) * 4  # use same target definition as numerai

    prediction_heavy_tails = np.sign(gauss_ranked_prediction) * np.abs(gauss_ranked_prediction) ** 1.5
    target_heavy_tails = np.sign(centered_target) * np.abs(centered_target) ** 1.5

    return np.corrcoef(prediction_heavy_tails, target_heavy_tails)[0, 1]


def create_era_sets(nr_eras: int) -> (list[list[int]], list[list[int]]):
    eras_per_fold = round(nr_eras / NR_FOLDS)
    era_sets_train = []
    era_sets_validate = []

    for i in range(NR_FOLDS):
        if (i + 1) < NR_FOLDS:
            era_sets_validate.append(list(range(eras_per_fold * i + 1, eras_per_fold * (i + 1) + 1)))
        else:
            era_sets_validate.append(list(range(eras_per_fold * i + 1, nr_eras + 1)))
        era_sets_train.append(list(range(1, max(1, eras_per_fold * i + 1 - ERAS_TO_PURGE))) +
                              list(range(min(nr_eras + 1, eras_per_fold * (i + 1) + 1 + ERAS_TO_PURGE), nr_eras + 1)))

    return era_sets_train, era_sets_validate


def get_result_as_dataframe(result: list[Any], parameter_names: list[str]) -> pd.DataFrame:
    df_result = pd.DataFrame(result, columns=parameter_names + ["best_iteration", "best_score", "index"])
    df_result["best_score"].fillna(0, inplace=True)
    return df_result


def get_constrained_result_dataframe(df_result: pd.DataFrame, hyperparameter_space: dict) -> pd.DataFrame:
    for p in hyperparameter_space.keys():
        space = hyperparameter_space[p].copy()
        if p in ["n2", "n3", "n4"]:
            space.append(-1)
        df_result = df_result[df_result[p].isin(space)]
    return df_result


def plot_score_by_parameter_spaces(df_result: pd.DataFrame) -> None:
    for p in list(df_result.columns)[:-3]:
        df_result.boxplot(column="best_score", by=p)


def normalise(x: pd.Series) -> pd.Series:
    return (x - min(x)) / (max(x) - min(x))


def save_as_pickle(x: Any, path: str) -> None:
    open_file = open(path, "wb")
    pickle.dump(x, open_file)
    open_file.close()


def load_pickle(path: str) -> Any:
    open_file = open(path, "rb")
    x = pickle.load(open_file)
    open_file.close()
    return x


def print_time_name_and_number(name: str, number: float) -> None:
    print(datetime.now().strftime("%H:%M:%S") + " . . . " + name + " " + str(number))
