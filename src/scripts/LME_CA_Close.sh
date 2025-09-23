# Script to run experiments on LME_CA_Close
# Shared
SERIES_NAME=LME_CA_Close
PATH_TRAIN_DF=train.csv
PATH_TEST_DF=test.csv
INDEX_NAME=date_id
INTERPOLATION_METHOD=time
ALPHA=0.05
N_LAGS=100
FEAT_SELEC_MODE=indirect

# Random Forest Lag 4
LAG=4
MIN_EST=100
MAX_EST=300
MODEL=random_forest
CV_FOLDS=5
LOSS=MSE
PATH_RF_CONFIG=src/res/rf_config.json

## DEBUG
# MIN_EST=10
# MAX_EST=20

python src/run_main.py \
    --series_name $SERIES_NAME \
    --path_train_df $PATH_TRAIN_DF \
    --path_test_df $PATH_TEST_DF \
    --index_name $INDEX_NAME \
    --interpolation_method $INTERPOLATION_METHOD \
    --lag $LAG \
    --alpha $ALPHA \
    --n_lags $N_LAGS \
    --feat_selec_mode $FEAT_SELEC_MODE \
    --model $MODEL \
    --min_est $MIN_EST \
    --max_est $MAX_EST \
    --cv_folds $CV_FOLDS \
    --loss $LOSS \
    --path_rf_config $PATH_RF_CONFIG \