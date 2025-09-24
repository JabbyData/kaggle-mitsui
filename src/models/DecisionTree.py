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
import os

class RandomForestEstimator:
    def __init__(
        self,
        cv_folds: int = 5,
        min_n_est: int = 100,
        max_n_est: int = 500,
        seed: int = 42,
    ):
        assert min_n_est <= max_n_est, "min_n_est should be lower or equal to max_n_est"
        self.min_n_est = min_n_est
        self.max_n_est = max_n_est
        self.cv_folds = cv_folds
        self.seed = seed
        self.rf_regressor = None

    def fine_tune(self, X_train: np.array, y_train: np.array, n_iter: int = 50, save_distrib: bool=False, X_test: np.array=None, img_dir: str=None):
        print("Fine-tuning model ...")
        step = math.ceil((self.max_n_est - self.min_n_est) / n_iter)
        n_est_list = list(range(self.min_n_est, self.max_n_est + 1, step))
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = {}
        min_score = float("+inf")
        for c,n_est in enumerate(n_est_list):
            print(f"Iterating with {n_est} estimators")
            cv_score = 0.0
            rf_regressor = RandomForestRegressor(
                random_state=self.seed, n_estimators=n_est, n_jobs=-1
            )
            for i,(train_index, valid_index) in enumerate(tscv.split(X_train),start=1):
                if c == 0:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Histogram(
                            x=X_train[train_index][:, 0],
                            name="Train",
                            opacity=0.6,
                            marker_color="blue",
                        )
                    )
                    fig.add_trace(
                        go.Histogram(
                            x=X_train[valid_index][:, 0],
                            name="Valid",
                            opacity=0.6,
                            marker_color="orange",
                        )
                    )
                    if X_test is not None:
                        fig.add_trace(
                            go.Histogram(
                                x=X_test[:, 0],
                                name="Test",
                                opacity=0.6,
                                marker_color="green",
                            )
                        )
                    fig.update_layout(
                        barmode="overlay",
                        title="Distribution of First Feature: Train, Valid, Test",
                        xaxis_title="Value of First Feature",
                        yaxis_title="Count",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        template="plotly_white"
                    )
                    if img_dir is not None:
                        os.makedirs(img_dir,exist_ok=True)
                        img_path = os.path.join(img_dir,f"{i}.png")
                        fig.write_image(img_path)
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
        print("Evaluation scores on test set")
        print(tabulate.tabulate(table, headers=headers, tablefmt="github"))
        return {
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
        }

    def select_feature(
        self, X: np.array, y: np.array, feature_names: list, n_seed_iter: int = 10
    ):
        assert (
            self.rf_regressor is not None
        ), "No model assigned, please fine-tune the model first calling fine_tune"
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        baseline_score = 0.0
        for train_index, valid_index in tscv.split(X):
            self.rf_regressor.fit(X[train_index], y[train_index])
            preds = self.rf_regressor.predict(X[valid_index])
            baseline_score += mean_squared_error(y[valid_index], preds)
        baseline_score /= self.cv_folds

        scores = {"baseline": baseline_score}
        for feature_index in range(X.shape[1]):
            feature_name = feature_names[feature_index]
            print(
                f"Analyzing feature n° {feature_index + 1}/{X.shape[1]} : {feature_name}"
            )
            scores[feature_name] = []
            for seed_iter in np.random.randint(0, 10000, size=n_seed_iter):
                score = 0.0
                for train_index, valid_index in tscv.split(X):
                    X_train = X[train_index].copy()
                    y_train = y[train_index]
                    X_valid = X[valid_index].copy()
                    y_valid = y[valid_index]
                    X_train[:, feature_index] = np.random.RandomState(
                        seed_iter
                    ).permutation(X_train[:, feature_index])
                    self.rf_regressor.fit(X_train, y_train)
                    X_valid[:, feature_index] = np.random.RandomState(
                        seed_iter
                    ).permutation(X_valid[:, feature_index])
                    preds = self.rf_regressor.predict(X_valid)
                    score += mean_squared_error(y_valid, preds)
                scores[feature_name].append(score / self.cv_folds)

        return scores

class DecisionTreeEstimator():
    def __init__(self, n_est: int, min_samples_split: int, min_impurity_decrease: float, cv_folds: int, seed: float):
        self.cv_folds = cv_folds
        self.regressor = RandomForestRegressor(
            n_estimators=n_est,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            random_state=seed
        )

    def cross_validate(self, X_train: np.array, y_train: np.array):
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        score = 0
        for i, (train_index,valid_index) in enumerate(tscv.split(X_train)):
            self.regressor.fit(X_train[train_index],y_train[train_index])
            valid_preds = self.regressor.predict(X_train[valid_index])
            score += mean_absolute_error(y_train[valid_index],valid_preds) # Arbitrary MAE for interpretability
        return score / self.cv_folds
    
    def fit(self, X_train: np.array, y_train: np.array):
        self.regressor.fit(X_train, y_train)

    def evaluate(self, X_test: np.array, y_test: np.array):
        rf_reg_preds = self.regressor.predict(X_test)
        table = []
        headers = ["MSE", "MAE", "MAPE", "R2"]
        mse = mean_squared_error(y_test, rf_reg_preds)
        mae = mean_absolute_error(y_test, rf_reg_preds)
        mape = mean_absolute_percentage_error(y_test, rf_reg_preds)
        r2 = r2_score(y_test, rf_reg_preds)
        table.append([mse, mae, mape, r2])
        print("Evaluation scores on test set")
        print(tabulate.tabulate(table, headers=headers, tablefmt="github"))
        return {
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
        }

