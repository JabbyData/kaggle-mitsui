"""
Module to define parser
"""

import argparse


def init_parser(description: str = ""):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--series_name",
        type=str,
        required=True,
        default="LME_CA_Close_LR4",
        help="Series ro predict",
    )
    parser.add_argument(
        "--path_train_df",
        type=str,
        required=True,
        default="train.csv",
        help="Path to training dataset to extract info from",
    )
    parser.add_argument(
        "--path_test_df",
        type=str,
        required=True,
        default="test.csv",
        help="Path to test dataset to extract info from",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        required=True,
        default="date_id",
        help="Index of the dataframe",
    )
    parser.add_argument(
        "--interpolation_method",
        type=str,
        required=True,
        default="time",
        help="Interpolation method to deal with missing values",
    )
    parser.add_argument(
        "--lag",
        type=int,
        required=True,
        default=4,
        help="Lag to apply to the series (target)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        default=0.05,
        help="Statistical Test first species error level",
    )
    parser.add_argument(
        "--n_lags",
        type=int,
        required=True,
        default=100,
        help="Number of lags to include in autocorrelation analysis",
    )
    parser.add_argument(
        "--feat_selec_mode",
        type=str,
        required=True,
        default="indirect",
        help="Autocorrelation mode to select features",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="random_forest",
        help="Machine learning model",
    )

    # Optimization

    parser.add_argument(
        "--min_est",
        type=int,
        required=True,
        default=100,
        help="Minimum number of estimators",
    )
    parser.add_argument(
        "--max_est",
        type=int,
        required=True,
        default=300,
        help="Maximum number of estimators",
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        required=True,
        default=2,
        help="Minimum number of samples to split tree",
    )
    parser.add_argument(
        "--max_samples_split",
        type=int,
        required=True,
        default=20,
        help="Maximum number of minimum number of samples to split tree",
    )
    parser.add_argument(
        "--min_impurity_decrease",
        type=float,
        required=True,
        default=0.0,
        help="Minimum impurity decrease to split tree",
    )
    parser.add_argument(
        "--max_impurity_decrease",
        type=float,
        required=True,
        default=0.1,
        help="Maximum minimum impurity decrease to split tree",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        required=True,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        required=True,
        default=50,
        help="Number of OPTUNA trials",
    )
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        default="MSE",
        help="Optimization loss",
    )


    parser.add_argument(
        "--path_rf_config",
        type=str,
        required=True,
        default="src/res/rf_config.json",
        help="Path to Random Forest configs",
    )

    # Feature selection

    parser.add_argument(
    "--n_seed_iter",
    type=int,
    required=True,
    default=50,
    help="Number of different seeds used for feature selection",
    )

    return parser
