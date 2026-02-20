import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
from src.config import ALPHA

def numerical_tests(df, target):
    selected = []

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        if col == target:
            continue

        group_yes = df[df[target] == "Yes"][col]
        group_no = df[df[target] == "No"][col]

        stat, p = ttest_ind(group_yes, group_no, equal_var=False)

        if p < ALPHA:
            selected.append(col)

    return selected


def categorical_tests(df, target):
    selected = []

    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        if col == target:
            continue

        contingency = pd.crosstab(df[col], df[target])
        stat, p, dof, expected = chi2_contingency(contingency)

        if p < ALPHA:
            selected.append(col)

    return selected
