"""
Module to define fractional differentiation
"""

import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller


class Fracdiff1D:
    """
    Fractional differentiator on 1D series
    """

    def __init__(self):
        self.fd_weights = None
        self.order = None

    def get_weights(self, order: float, threshold: float = 1e-2) -> np.array:
        """
        Computes and assign fractional differentiation weights
        """
        weights = [1.0]
        weight = 1.0
        i = 1
        while abs(weight) > threshold:
            new_weight = -weight * (order - i + 1) / i
            if abs(new_weight) <= threshold:
                break
            weights.append(new_weight)
            i += 1
            weight = new_weight

        self.order = order
        self.fd_weights = np.array(weights)

    def plot_weights(self):
        """
        Plots fractional differentiation weights
        """
        assert (
            self.fd_weights is not None
        ), "no weights assigned, please call get_weights"
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=np.arange(stop=len(self.fd_weights)), y=self.fd_weights)
        )

        fig.update_layout(
            xaxis_title="Index",
            yaxis_title="Weight value",
            width=800,
            title=f"Fractional differentiated weights (order {self.order})",
        )

        fig.show()

    def transform(self, series: np.array):
        n = len(series)
        n_weights = len(self.fd_weights)
        fd_series = []
        series = series.reshape(
            -1,
        )
        for i in range(n_weights - 1, n):
            fd_series.append(series[i - n_weights + 1 : i + 1] @ self.fd_weights[::-1])

        return np.array(fd_series)

    def plot_transformed_series(self, series: np.array):
        """
        Plots original vs fracdiff series
        """
        assert (
            self.fd_weights is not None
        ), "Weights not assigned, please call get_weights method first"

        fig = go.Figure()
        index = np.arange(len(series))
        fd_series = self.transform(series)

        fig.add_trace(
            go.Scatter(
                x=index,
                y=series.reshape(
                    -1,
                ),
                name="Original",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=index[len(self.fd_weights) - 1 :],
                y=fd_series,
                name="FracDiff",
            )
        )

        fig.update_layout(
            xaxis_title="Index",
            yaxis_title="Value",
            title="Original vs FracDiff series",
        )

        fig.show()

    def fit(self, series: np.array):
        """
        Find the smallest order of differentiation rendering the series stationary
        """
        print("Making the series stationary ...")
        num_trials = 50
        order_candidates = np.linspace(start=0, stop=2, num=num_trials)
        for i, order in enumerate(order_candidates, start=1):
            print(f"Trial {i}/{num_trials}")
            self.get_weights(order=order)
            fd_series = self.transform(series)
            test_res = adfuller(fd_series, autolag="AIC")
            if test_res[1] < 0.05:
                print(
                    f"Series stationary with order {order} and ADF p_value {test_res[1]} (level 5%)"
                )
                return fd_series

    def revert(self, series: np.array):
        """
        Revert series to its original space
        """
        raise NotImplementedError
