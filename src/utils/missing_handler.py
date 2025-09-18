"""
Module to handle missing values in series
"""

import pandas as pd
import plotly.graph_objects as go
import os


class MissingHandler:
    def __init__(self, method: str):
        self.method = method

    def get_completed(self, dataframe: pd.DataFrame, series_name: str, index_name: str):
        print("Handling missing values ...")
        series = dataframe[[index_name, series_name]].copy()
        series[index_name] = pd.to_datetime(series[index_name], unit="D", origin="unix")
        series = series.set_index(index_name)
        print(
            f"Fixing {series.isna().values.astype(int).sum()} missing values with {self.method} method"
        )
        series_completed = series.interpolate(method="time")
        return series_completed

    def save_plot_completed_series(
        self,
        dataframe: pd.DataFrame,
        series_name: str,
        index_name: str,
        img_path: str,
    ):
        series = dataframe[[index_name, series_name]].copy()
        series[index_name] = pd.to_datetime(series[index_name], unit="D", origin="unix")
        series = series.set_index(index_name)

        missing_dates = series[series[series_name].isna()].index

        series_completed = series.interpolate(method=self.method)

        series_missing = series_completed.loc[missing_dates]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=series.index, y=series[series_name], name="original")
        )

        fig.add_trace(
            go.Scatter(
                x=series_missing.index,
                y=series_missing[series_name],
                name="interpolated",
                mode="markers",
            )
        )

        fig.update_layout(
            xaxis_title=index_name,
            yaxis_title="values",
            title=f"Plot of completed series {series_name}",
        )

        fig.write_image(img_path)
