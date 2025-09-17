"""
Module implementing Random Forest Regressor
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import tabulate
import plotly.graph_objects as go
import math


class RandomForestEstimator:
    def __init__(
        self,
        model: str,
        cv_folds: int = 5,
        min_n_est: int = 100,
        max_n_est: int = 500,
        seed: int = 42,
    ):
        assert model in ["random_forest"], f"Uknown model {model}"
        self.model = model
        assert min_n_est <= max_n_est, "min_n_est should be lower or equal to max_n_est"
        self.min_n_est = min_n_est
        self.max_n_est = max_n_est
        self.cv_folds = cv_folds
        self.seed = seed
        self.rf_regressor = None

    def fine_tune(self, X_train: np.array, y_train: np.array, n_iter: int = 50):
        step = math.ceil((self.max_n_est - self.min_n_est) / n_iter)
        n_est_list = list(range(self.min_n_est, self.max_n_est + 1, step))
        tscv = tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = {}
        min_score = float("+inf")
        for n_est in n_est_list:
            print(f"Iterating with {n_est} estimators")
            cv_score = 0.0
            rf_regressor = RandomForestRegressor(
                random_state=self.seed, n_estimators=n_est, n_jobs=-1
            )
            for train_index, valid_index in tscv.split(X_train):
                rf_regressor.fit(X_train[train_index], y_train[train_index])
                valid_preds = rf_regressor.predict(X_train[valid_index])
                cv_score += mean_squared_error(y_train[valid_index], valid_preds)
            cv_score /= self.cv_folds
            scores[n_est] = cv_score
            if cv_score < min_score:
                min_score = cv_score
                n_est_min = n_est
                self.rf_regressor = rf_regressor

        self.n_est = n_est_min
        print(f"Minimal score found for {self.n_est} estimators : {min_score}")

        return scores

    def predict(self, X: np.array):
        assert (
            self.rf_regressor is not None
        ), "Please fine-tune the model first calling fine_tune"
        return self.rf_regressor.predict(X)

    def evaluate(self, X: np.array, targets: np.array):
        rf_reg_preds = self.predict(X)
        table = []
        headers = ["MSE", "MAE", "MAPE", "R2"]
        mse = mean_squared_error(targets, rf_reg_preds)
        mae = mean_absolute_error(targets, rf_reg_preds)
        mape = mean_absolute_percentage_error(targets, rf_reg_preds)
        r2 = r2_score(targets, rf_reg_preds)
        table.append([mse, mae, mape, r2])
        print(tabulate.tabulate(table, headers=headers, tablefmt="github"))

    def plot_feature_importance(self):
        assert (
            self.rf_regressor is not None
        ), "Please fine-tune the model first calling fine_tune"

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=list(range(len(self.rf_regressor.feature_importances_))),
                y=self.rf_regressor.feature_importances_,
            )
        )

        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Feature Index",
            yaxis_title="Importance",
        )

        fig.show()
