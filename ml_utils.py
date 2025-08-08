"""
Reusable data preprocessing and modeling utilities for the House Price project.

This module centralizes common helpers used across notebooks to keep
notebooks concise and ensure a single source of truth.
"""

from __future__ import annotations

from typing import List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# ------------------------------
# Column utilities
# ------------------------------
def grab_col_names(
    dataframe: pd.DataFrame,
    cat_th: int = 10,
    car_th: int = 20,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split columns into categorical, cardinal categorical and numerical groups.

    Returns (cat_cols, cat_but_car, num_cols)
    """
    categorical_columns = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    numeric_but_categorical = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtype != "O"
    ]
    categorical_but_cardinal = [
        col for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtype == "O"
    ]

    cat_cols = [col for col in categorical_columns + numeric_but_categorical if col not in categorical_but_cardinal]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    num_cols = [col for col in num_cols if col not in numeric_but_categorical]

    return cat_cols, categorical_but_cardinal, num_cols


# ------------------------------
# Outlier handling (IQR based)
# ------------------------------
def outlier_thresholds(
    dataframe: pd.DataFrame,
    variable: str,
    low_quantile: float = 0.10,
    up_quantile: float = 0.90,
) -> Tuple[float, float]:
    """
    Compute lower and upper bounds using an asymmetric IQR between the given quantiles.
    """
    q1 = dataframe[variable].quantile(low_quantile)
    q3 = dataframe[variable].quantile(up_quantile)
    iqr = q3 - q1
    low_limit = float(q1 - 1.5 * iqr)
    up_limit = float(q3 + 1.5 * iqr)
    return low_limit, up_limit


def replace_with_thresholds(
    dataframe: pd.DataFrame,
    variable: str,
    low_quantile: float = 0.10,
    up_quantile: float = 0.90,
) -> None:
    """
    Winsorize values outside the IQR-based thresholds in-place.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, low_quantile, up_quantile)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# ------------------------------
# Missing value imputation
# ------------------------------
def quick_missing_imp(
    data: pd.DataFrame,
    num_method: str = "median",
    cat_length: int = 17,
    target: str = "SalePrice",
) -> pd.DataFrame:
    """
    Fast imputation: categorical columns (<= cat_length unique) by mode, numerical by mean/median.
    Preserves target column if present.
    """
    data = data.copy()
    temp_target: Optional[pd.Series] = data[target] if target in data.columns else None

    # Categorical by mode (limited cardinality)
    data = data.apply(
        lambda s: s.fillna(s.mode()[0]) if (s.dtype == "O" and len(s.unique()) <= cat_length and s.isnull().any()) else s
    )

    # Numeric by mean/median
    if num_method == "mean":
        data = data.apply(lambda s: s.fillna(s.mean()) if (s.dtype != "O" and s.isnull().any()) else s)
    else:
        data = data.apply(lambda s: s.fillna(s.median()) if (s.dtype != "O" and s.isnull().any()) else s)

    if temp_target is not None:
        data[target] = temp_target

    return data


# ------------------------------
# Rare encoder
# ------------------------------
def rare_encoder(dataframe: pd.DataFrame, rare_perc: float = 0.01) -> pd.DataFrame:
    """
    Replace rare categories (frequency < rare_perc) with the label 'Rare' for object dtype columns.
    """
    temp_df = dataframe.copy()
    rare_columns = [
        col for col in temp_df.columns
        if temp_df[col].dtype == "O" and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any()
    ]

    for col in rare_columns:
        freqs = temp_df[col].value_counts() / len(temp_df)
        rare_labels = freqs[freqs < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(rare_labels), "Rare", temp_df[col])

    return temp_df


# ------------------------------
# Encoders
# ------------------------------
def label_encoder(dataframe: pd.DataFrame, binary_col: str) -> pd.DataFrame:
    """
    Label-encode a binary categorical column (in-place) and return the dataframe.
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(
    dataframe: pd.DataFrame,
    categorical_cols: Iterable[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encode selected categorical columns and return a new DataFrame.
    """
    categorical_cols = list(categorical_cols)
    if not categorical_cols:
        return dataframe
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)


# ------------------------------
# Target mean encoding (smoothed)
# ------------------------------
def target_mean_encode(
    frame: pd.DataFrame,
    column: str,
    target: str = "SalePrice",
    m: float = 100.0,
) -> pd.Series:
    """
    Smoothed target mean encoding for a single categorical column.
    """
    global_mean = frame[target].mean()
    stats = frame.groupby(column)[target].agg(["mean", "count"])  # type: ignore[index]
    smooth = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)
    return smooth


# ------------------------------
# Feature importance plot
# ------------------------------
def plot_importance(
    model,
    features: pd.DataFrame,
    num: Optional[int] = None,
    save: bool = False,
    save_path: str = "importances.png",
) -> None:
    """
    Plot feature importances for tree-based models exposing `feature_importances_`.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not have attribute 'feature_importances_'.")

    importances = getattr(model, "feature_importances_")
    feature_names = list(features.columns)

    if num is None:
        num = len(feature_names)

    feature_imp = pd.DataFrame({"Value": importances, "Feature": feature_names})
    feature_imp = feature_imp.sort_values(by="Value", ascending=False).head(num)

    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp)
    plt.title("Features")
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
    plt.show()


__all__ = [
    "grab_col_names",
    "outlier_thresholds",
    "replace_with_thresholds",
    "quick_missing_imp",
    "rare_encoder",
    "label_encoder",
    "one_hot_encoder",
    "target_mean_encode",
    "plot_importance",
]


