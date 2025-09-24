"""
Module implementing optimization techniques
"""

import optuna
from models.DecisionTree import DecisionTreeEstimator
import numpy as np

def objective(trial, args: dict, X_train: np.array, y_train: np.array, seed: int):
    if args.model == "random_forest":
        h_params = {}
        h_params["cv_folds"] = args.cv_folds
        h_params["n_estimators"] = trial.suggest_int("n_estimators",args.min_est,args.max_est)
        h_params["min_samples_split"] = trial.suggest_int("min_samples_split", args.min_samples_split, args.max_samples_split)
        h_params["min_impurity_decrease"] = trial.suggest_float("min_impurity_decrease", args.min_impurity_decrease, args.max_impurity_decrease)
    else:
        raise NotImplementedError(f"Objective function not developped for model {args.model}")
    
    return cv_score(args.model, h_params, X_train, y_train, seed)

def cv_score(model_name: str, h_params: dict, X_train: np.array, y_train: np.array, seed: float):
    if model_name == "random_forest":
        model = DecisionTreeEstimator(
            n_est=h_params["n_estimators"],
            min_samples_split=h_params["min_samples_split"],
            min_impurity_decrease=h_params["min_impurity_decrease"],
            cv_folds=h_params["cv_folds"],
            seed=seed
        )
    else:
        raise NotImplementedError(f"Scoring function not developped for model {model_name}")
    return model.cross_validate(X_train=X_train,y_train=y_train)


# TODO : create study
# TODO : laucnh experiment
# TODO : parallelize search + pruning