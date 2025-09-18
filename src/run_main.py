"""
Module to run experiments
"""

# Dependencies
## Data manip
import pandas as pd
from utils.tools import compute_log_returns, check_similar_vals
from utils.missing_handler import MissingHandler
from utils.tools import split_lag

## Files & System
import os

## Init
from utils.parse import init_parser

## Linear Algebra
import numpy as np

## Machine Learning
from models.DecisionTree import RandomForestEstimator
from sklearn.model_selection import train_test_split

## Stats
from utils import fracdiff
from utils.auto_correlation import AutoCorrelator

## Visualisation
from utils.tools import save_plot_cv_scores
from utils.tools import save_plot_feature_importance

## Reproducibility
seed = 42
np.random.seed(seed)

## DEBUG
import debugpy

debugpy.listen(5678)
print("Waiting for debugger attach...")
debugpy.wait_for_client()

parser = init_parser()
args = parser.parse_args()

train_df = pd.read_csv(os.path.join(os.getcwd(), args.path_train_df))[
    [args.index_name, args.series_name]
]

train_df_copy = train_df.copy(deep=True)  # for test extra input

target_name = args.series_name + "_LR" + str(args.lag)

train_df[target_name] = compute_log_returns(train_df[args.series_name], args.lag)


# Missing values
img_path = os.path.join("src/plots/missing_vals", target_name + ".png")
miss_handler = MissingHandler(method=args.interpolation_method)
miss_handler.save_plot_completed_series(
    dataframe=train_df,
    series_name=target_name,
    index_name=args.index_name,
    img_path=img_path,
)
target_completed = miss_handler.get_completed(
    dataframe=train_df, series_name=target_name, index_name=args.index_name
)

# Stationary series
f_differentiator = fracdiff.Fracdiff1D()
train_df[target_name] = f_differentiator.fit(target_completed.to_numpy())

# Feature selection
ac = AutoCorrelator(alpha=args.alpha, nlags=args.n_lags)
img_path = os.path.join("src/plots/acf", target_name + ".png")
ac.save_plot_acf_stats(train_df[target_name], img_path=img_path)
img_path = os.path.join("src/plots/pacf", target_name + ".png")
ac.save_plot_pacf_stats(train_df[target_name], img_path=img_path)
relevant_lags = ac.select_lags(train_df[target_name], mode=args.feat_selec_mode)

# Input feature
print("Spliting into train / valid / test set ...")
train_df = split_lag(s=train_df[target_name], relevant_lags=relevant_lags)

# Set Split
X = train_df.drop(columns=["L0"]).values
y = train_df["L0"].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False) # to remove (cf test set appart)

# Model
## Base
dt_regressor = RandomForestEstimator(
    model=args.model,
    cv_folds=args.cv_folds,
    min_n_est=args.min_est,
    max_n_est=args.max_est,
    seed=seed,
)

scores = dt_regressor.fine_tune(
    X_train=X, y_train=y, n_iter=args.max_est - args.min_est + 1
)
img_dir = os.path.join("src", "plots", "hyper_param_ft", target_name)
os.makedirs(img_dir, exist_ok=True)
img_path = os.path.join(img_dir, "base.png")
save_plot_cv_scores(
    scores,
    title="Plot of cross-validation score (TSCV) versus number of estimators",
    xaxis="Number of estimators",
    yaxis=args.loss,
    img_path=img_path,
)

# TODO : complete test set process
# Data process test set
# test_df = pd.read_csv(os.path.join(os.getcwd(), args.path_test_df))[
#     [args.index_name, args.series_name]
# ]

# max_lag = sorted(relevant_lags)[-1]
# extra_rows = train_df_copy.tail(max_lag).copy(deep=True)
# test_start_id = test_df[args.index_name].iloc[0]
# extra_rows[args.index_name] = np.arange(test_start_id - max_lag, test_start_id)
# test_df = pd.concat([extra_rows, test_df], ignore_index=True)

# test_df[target_name] = compute_log_returns(test_df[args.series_name], args.lag)

# test_target_completed = miss_handler.get_completed(
#     dataframe=test_df, series_name=target_name, index_name=args.index_name
# )

# test_df[target_name] = f_differentiator.transform(test_target_completed.to_numpy())


# eval_scores = dt_regressor.evaluate(X=X_test, targets=y_test)

# # Save evaluation scores to file
# results_dir = os.path.join("src", "res", target_name)
# os.makedirs(results_dir, exist_ok=True)
# results_file = os.path.join(results_dir, "base.txt")
# with open(results_file, "w") as f:
#     f.write(f"Evaluation Results for {target_name} base\n")
#     f.write("=" * 40 + "\n")
#     for metric, score in eval_scores.items():
#         f.write(f"{metric}: {score}\n")

# Feature selection
# train_df_copy = train_df.copy(deep=True)
### DEBUG
train_df_copy = train_df.copy(deep=True)[["L0", "L1", "L2", "L11", "L12", "L21", "L53"]]
###
stop = False
step = 0
while not stop:
    print(f"Removing irrelevant features (step {step+1})")
    X = train_df_copy.drop(columns=["L0"]).values
    feature_names = train_df_copy.drop(columns=["L0"]).columns
    print("Feature index: ", feature_names)
    scores = dt_regressor.select_feature(X, y, feature_names)
    feature_to_drop = []
    for feature_name, values in scores.items():
        if feature_name != "baseline" and np.median(values) < scores["baseline"]:
            feature_to_drop.append(feature_name)
    if len(feature_to_drop) == 0:
        print("No more feature to remove")
        stop = True
    else:
        print("Removing : ", feature_to_drop)
        train_df_copy = train_df_copy.drop(columns=feature_to_drop)
    img_dir = os.path.join("src", "plots", "feat_selec", target_name)
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, str(step + 1) + ".png")
    save_plot_feature_importance(scores=scores, img_path=img_path)

# Final Training
