# Module to automate data processing
import pandas as pd
import numpy as np
import plotly.graph_objects as go


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
        res_df["L" + str(lag)] = s.shift(lag)

    return res_df.dropna()


def save_plot_cv_scores(
    scores: dict, title: str, xaxis: str, yaxis: str, img_path: str
):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(scores.keys()), y=list(scores.values())))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
    )

    fig.write_image(img_path)


def save_plot_feature_importance(scores: dict, img_path: str):
    """
    Display feature importance box plots
    """

    feature_names = list(scores.keys())

    fig = go.Figure()

    for feature_name in feature_names:
        if feature_name != "baseline":
            fig.add_trace(
                go.Box(
                    y=scores[feature_name],
                    name=feature_name,
                    boxpoints="outliers",
                    marker_color="blue",
                    line=dict(color="blue"),
                    showlegend=False,
                )
            )

    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(feature_names) - 0.5,
        y0=scores["baseline"],
        y1=scores["baseline"],
        line=dict(color="red", width=2, dash="dash"),
        name="Baseline",
        showlegend=True,
    )

    fig.update_layout(
        title="Permutation Feature Importance (Random Shuffling)",
        xaxis_title="Feature score",
        yaxis_title="MSE",
        showlegend=True,
    )
    fig.write_image(img_path)
