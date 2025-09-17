# Module to automate data processing
import pandas as pd
import numpy as np


def compute_log_returns(s: pd.Series, lag: int):
    s_log_returns = pd.Series(np.nan, index=s.index)
    for t in range(len(s)):
        try:
            s_log_returns.iloc[t] = np.log(s.iloc[t + lag + 1]) - np.log(
                s.iloc[t + 1]
            )  # Wait for both stock to close
        except Exception:
            s_log_returns.iloc[t] = np.nan
    return s_log_returns


def check_similar_vals(s1: pd.Series, s2: pd.Series, precision: int):
    """
    Check similarity between series elemts that are not null
    """
    n1, n2 = len(s1), len(s2)
    assert n1 == n2, f"Series with different length : {n1} vs {n2}"

    for i, (v1, v2) in enumerate(zip(s1.values, s2.values)):
        if (
            not np.isnan(v1)
            and not np.isnan(v2)
            and v1.round(precision) != v2.round(precision)
        ):
            print(f"Different values at row {i} : {v1} vs {v2}")
            return False

    return True


def split_lag(s: pd.Series, relevant_lags: list):
    res_df = pd.DataFrame()
    for lag in relevant_lags:
        res_df["L" + str(lag)] = s.shift(-lag)

    return res_df.dropna()
