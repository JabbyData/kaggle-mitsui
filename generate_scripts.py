#!/usr/bin/env python3
"""
Script to generate bash scripts for each asset based on target_pairs.csv
"""

import csv
import os
import stat
from pathlib import Path

def parse_target_pairs(csv_file):
    """Parse target_pairs.csv and extract unique assets"""
    assets = set()
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair = row['pair'].strip()
            
            # Handle both single assets and asset pairs
            if ' - ' in pair:
                # Split asset pairs
                asset1, asset2 = pair.split(' - ')
                assets.add(asset1.strip())
                assets.add(asset2.strip())
            else:
                # Single asset
                assets.add(pair)
    
    return sorted(assets)

def create_script_template(asset_name):
    """Create bash script template for a given asset"""
    script_content = f"""# Script to run experiments on {asset_name}
# Shared
SERIES_NAME={asset_name}
PATH_TRAIN_DF=train.csv
PATH_TEST_DF=test.csv
INDEX_NAME=date_id
INTERPOLATION_METHOD=time
ALPHA=0.05
N_LAGS=100
FEAT_SELEC_MODE=indirect
N_SEED_ITER=50

# Random Forest Lag 1
LAG=1
MIN_EST=100
MAX_EST=300
MODEL=random_forest
CV_FOLDS=5
LOSS=MSE
PATH_RF_CONFIG=src/res/rf_config.json

python src/run_main.py \\
    --series_name $SERIES_NAME \\
    --path_train_df $PATH_TRAIN_DF \\
    --path_test_df $PATH_TEST_DF \\
    --index_name $INDEX_NAME \\
    --interpolation_method $INTERPOLATION_METHOD \\
    --lag $LAG \\
    --alpha $ALPHA \\
    --n_lags $N_LAGS \\
    --feat_selec_mode $FEAT_SELEC_MODE \\
    --model $MODEL \\
    --min_est $MIN_EST \\
    --max_est $MAX_EST \\
    --cv_folds $CV_FOLDS \\
    --loss $LOSS \\
    --path_rf_config $PATH_RF_CONFIG \\
    --n_seed_iter $N_SEED_ITER \\

# Random Forest Lag 2
LAG=2
MIN_EST=100
MAX_EST=300
MODEL=random_forest
CV_FOLDS=5
LOSS=MSE
PATH_RF_CONFIG=src/res/rf_config.json

python src/run_main.py \\
    --series_name $SERIES_NAME \\
    --path_train_df $PATH_TRAIN_DF \\
    --path_test_df $PATH_TEST_DF \\
    --index_name $INDEX_NAME \\
    --interpolation_method $INTERPOLATION_METHOD \\
    --lag $LAG \\
    --alpha $ALPHA \\
    --n_lags $N_LAGS \\
    --feat_selec_mode $FEAT_SELEC_MODE \\
    --model $MODEL \\
    --min_est $MIN_EST \\
    --max_est $MAX_EST \\
    --cv_folds $CV_FOLDS \\
    --loss $LOSS \\
    --path_rf_config $PATH_RF_CONFIG \\
    --n_seed_iter $N_SEED_ITER \\

# Random Forest Lag 3
LAG=3
MIN_EST=100
MAX_EST=300
MODEL=random_forest
CV_FOLDS=5
LOSS=MSE
PATH_RF_CONFIG=src/res/rf_config.json

python src/run_main.py \\
    --series_name $SERIES_NAME \\
    --path_train_df $PATH_TRAIN_DF \\
    --path_test_df $PATH_TEST_DF \\
    --index_name $INDEX_NAME \\
    --interpolation_method $INTERPOLATION_METHOD \\
    --lag $LAG \\
    --alpha $ALPHA \\
    --n_lags $N_LAGS \\
    --feat_selec_mode $FEAT_SELEC_MODE \\
    --model $MODEL \\
    --min_est $MIN_EST \\
    --max_est $MAX_EST \\
    --cv_folds $CV_FOLDS \\
    --loss $LOSS \\
    --path_rf_config $PATH_RF_CONFIG \\
    --n_seed_iter $N_SEED_ITER \\

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

python src/run_main.py \\
    --series_name $SERIES_NAME \\
    --path_train_df $PATH_TRAIN_DF \\
    --path_test_df $PATH_TEST_DF \\
    --index_name $INDEX_NAME \\
    --interpolation_method $INTERPOLATION_METHOD \\
    --lag $LAG \\
    --alpha $ALPHA \\
    --n_lags $N_LAGS \\
    --feat_selec_mode $FEAT_SELEC_MODE \\
    --model $MODEL \\
    --min_est $MIN_EST \\
    --max_est $MAX_EST \\
    --cv_folds $CV_FOLDS \\
    --loss $LOSS \\
    --path_rf_config $PATH_RF_CONFIG \\
    --n_seed_iter $N_SEED_ITER \\
"""
    return script_content

def main():
    print("Parsing target_pairs.csv...")
    assets = parse_target_pairs('target_pairs.csv')
    print(f"Found {len(assets)} unique assets")
    
    scripts_dir = os.path.join("src","scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    
    created_scripts = []
    skipped_scripts = []
    
    for asset in assets:
        script_filename = f"{asset}.sh"
        script_path = os.path.join(scripts_dir,script_filename)
        
        if os.path.exists(script_filename):
            skipped_scripts.append(script_filename)
            continue   

        script_content = create_script_template(asset)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        Path(script_path).chmod(Path(script_path).stat().st_mode | stat.S_IEXEC)
        
        created_scripts.append(script_filename)
        print(f"Created: {script_filename}")
    
    print(f"\nSummary:")
    print(f"Created {len(created_scripts)} new scripts")
    print(f"Skipped {len(skipped_scripts)} existing scripts")
    
    if created_scripts:
        print(f"\nNew scripts created:")
        for script in created_scripts:
            print(f"  - {script}")
    
    if skipped_scripts:
        print(f"\nExisting scripts skipped:")
        for script in skipped_scripts:
            print(f"  - {script}")

if __name__ == "__main__":
    main()