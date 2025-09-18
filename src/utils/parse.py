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
        "--cv_folds",
        type=int,
        required=True,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        default="MSE",
        help="Optimization loss",
    )
    return parser
