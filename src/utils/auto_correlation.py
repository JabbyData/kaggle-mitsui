"""
Module defining auto-correlator
"""

import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from copy import deepcopy
import plotly.graph_objects as go


class AutoCorrelator:
    def __init__(self, alpha: float, nlags: int):
        self.alpha = alpha
        self.nlags = nlags

    def get_acf_stats(self, series: np.array):
        """
        Compute and return ACF values
        """
        acf_stats = acf(
            series,
            nlags=self.nlags if self.nlags else 50,
            alpha=self.alpha,
            qstat=True,
            adjusted=True,
        )
        return {
            "score": acf_stats[0],
            "confint": acf_stats[1],
            "qstat": acf_stats[2],  # excludes lag 0
            "p-val": acf_stats[3],  # excludes lag 0
        }

    def plot_acf_stats(self, series: np.array):
        assert (
            self.alpha is not None
        ), "Please assign a confidence level alpha to the instance"
        acf_scores = self.get_acf_stats(series)

        lags = np.arange(len(acf_scores["score"]))
        acf_vals = acf_scores["score"]
        confint = acf_scores["confint"]
        confint_centered = deepcopy(confint)
        for i, val in enumerate(acf_vals):
            confint_centered[i][0] -= val
            confint_centered[i][1] -= val

        fig_acf = go.Figure()

        fig_acf.add_trace(
            go.Scatter(x=lags, y=acf_vals, mode="markers+lines", name="ACF")
        )

        fig_acf.add_trace(
            go.Scatter(
                x=lags,
                y=confint_centered[:, 0],
                mode="lines",
                line=dict(color="lightgrey", dash="dash"),
                name="Lower CI",
            )
        )

        fig_acf.add_trace(
            go.Scatter(
                x=lags,
                y=confint_centered[:, 1],
                mode="lines",
                line=dict(color="lightgrey", dash="dash"),
                name="Upper CI",
            )
        )

        fig_acf.update_layout(
            title="ACF plot with Confidence Intervals",
            xaxis_title="Lag",
            yaxis_title="Coeff value",
            showlegend=True,
        )

        fig_acf.show()

    def get_pacf_stats(self, series: np.array):
        """
        Compute and return ACF values
        """
        acf_stats = pacf(
            series,
            nlags=self.nlags if self.nlags else 50,
            alpha=self.alpha,
        )
        return {
            "score": acf_stats[0],
            "confint": acf_stats[1],
        }

    def plot_pacf_stats(self, series: np.array):
        assert (
            self.alpha is not None
        ), "Please assign a confidence level alpha to the instance"
        pacf_scores = self.get_pacf_stats(series)

        lags = np.arange(len(pacf_scores["score"]))
        pacf_vals = pacf_scores["score"]
        confint = pacf_scores["confint"]
        confint_centered = deepcopy(confint)
        for i, val in enumerate(pacf_vals):
            confint_centered[i][0] -= val
            confint_centered[i][1] -= val

        fig_pacf = go.Figure()

        fig_pacf.add_trace(
            go.Scatter(x=lags, y=pacf_vals, mode="markers+lines", name="PACF")
        )

        fig_pacf.add_trace(
            go.Scatter(
                x=lags,
                y=confint_centered[:, 0],
                mode="lines",
                line=dict(color="lightgrey", dash="dash"),
                name="Lower CI",
            )
        )

        fig_pacf.add_trace(
            go.Scatter(
                x=lags,
                y=confint_centered[:, 1],
                mode="lines",
                line=dict(color="lightgrey", dash="dash"),
                name="Upper CI",
            )
        )

        fig_pacf.update_layout(
            title="PACF plot with Confidence Intervals",
            xaxis_title="Lag",
            yaxis_title="Coeff value",
            showlegend=True,
        )

        fig_pacf.show()

    def select_lags(self, series: np.array, mode: str = "indirect"):
        """
        Select relevant correlated values
        """
        assert mode in ["direct", "indirect"], f"unknown mode {mode}"
        print(f"Selecting significant lags with mode {mode}")
        index = []
        if mode == "direct":
            acf_stats = self.get_acf_stats(series)

            for i, (acf_val, acf_confint) in enumerate(
                zip(acf_stats["score"], acf_stats["confint"])
            ):
                if abs(acf_val) > acf_confint[1] - acf_val:
                    index.append(i)
        else:
            acf_stats = self.get_acf_stats(series)
            pacf_stats = self.get_pacf_stats(series)

            for i, (acf_val, acf_confint, pacf_val, pacf_confint) in enumerate(
                zip(
                    acf_stats["score"],
                    acf_stats["confint"],
                    pacf_stats["score"],
                    pacf_stats["confint"],
                )
            ):
                if (
                    abs(acf_val) > acf_confint[1] - acf_val
                    or abs(pacf_val) > pacf_confint[1] - pacf_val
                ):
                    index.append(i)

        return index
