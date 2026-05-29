"""
GENERATE TEST PREDICTIONS
Create predictions on test data with optimal threshold
Outputs: test_predictions.csv (CoilID, Y)
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import os
import sys

def generate_predictions(threshold=0.28):
    """
    Generate predictions on test data

    Parameters:
    -----------
    threshold : float
        Classification threshold (default: 0.28 from Part 7)

    Returns:
    --------
    str : Path to output CSV file
    """

    print("\n" + "="*70)
    print("TEST DATA PREDICTIONS")
    print("="*70)

    # Check if required files exist
    print("\nChecking for required files...")

    if not os.path.exists('train.csv'):
        print("✗ train.csv not found")
        return None

    if not os.path.exists('test.csv'):
        print("✗ test.csv not found")
        return None

    print("✓ train.csv found")
    print("✓ test.csv found")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"  Training data: {train_df.shape}")
    print(f"  Test data: {test_df.shape}")

    # Get CoilID from test
    coil_ids = test_df['CoilID'].values

    # Get features
    print("\nPreparing features...")

    # Try to use engineered features if available
    use_engineered = False
    if os.path.exists('X_engineered.csv') and os.path.exists('X_engineered_test.csv'):
        try:
            X_train = pd.read_csv('X_engineered.csv')
            X_test = pd.read_csv('X_engineered_test.csv')

            # Remove CoilID if present
            if 'CoilID' in X_train.columns:
                X_train = X_train.drop('CoilID', axis=1)
            if 'CoilID' in X_test.columns:
                X_test = X_test.drop('CoilID', axis=1)

            use_engineered = True
            print("✓ Using engineered features")

        except Exception as e:
            print(f"⚠ Could not load engineered features: {e}")

    # Fall back to raw features
    if not use_engineered:
        feature_cols = [col for col in train_df.columns if col.startswith('X')]
        X_train = train_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()
        print(f"✓ Using raw features: {len(feature_cols)} columns")

    y_train = train_df['Y'].values

    # Median imputation for NaN values (fit on train, applied to test)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    print(f"  ✓ Imputed missing values (median)")

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")

    # Train model
    # Config validated via 5-fold CV: regularized XGBoost achieves the best
    # out-of-fold AUC (0.8655) vs original config (0.8618). Heavier
    # regularization + shallower trees reduces overfitting on the 66-positive
    # imbalanced dataset.
    print("\nTraining regularized XGBoost model on full training data...")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print(f"✓ Model trained (scale_pos_weight={scale_pos_weight:.1f})")

    # Generate predictions
    print("\nGenerating predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    print(f"  Using threshold: {threshold:.4f}")
    print(f"  Predicted defects: {(y_pred == 1).sum()} ({100*(y_pred==1).sum()/len(y_pred):.1f}%)")
    print(f"  Predicted non-defects: {(y_pred == 0).sum()} ({100*(y_pred==0).sum()/len(y_pred):.1f}%)")

    # Create output DataFrame
    predictions_df = pd.DataFrame({
        'CoilID': coil_ids,
        'Y': y_pred
    })

    # Save to CSV
    output_file = 'test_predictions.csv'
    predictions_df.to_csv(output_file, index=False)

    print(f"\n✓ Predictions saved to: {output_file}")
    print(f"\nPredictions Preview (first 10 rows):")
    print(predictions_df.head(10).to_string(index=False))

    print("\n" + "="*70)
    print("READY FOR DOWNLOAD")
    print("="*70)
    print(f"\nFile: test_predictions.csv")
    print(f"Format: CoilID, Y (0=No Defect, 1=Defect)")
    print(f"Total rows: {len(predictions_df)}")

    return output_file

if __name__ == "__main__":
    # Get threshold from command line or use default
    threshold = 0.28
    if len(sys.argv) > 1:
        try:
            threshold = float(sys.argv[1])
            print(f"\nUsing custom threshold: {threshold}")
        except ValueError:
            print(f"Invalid threshold. Using default: {threshold}")

    output_file = generate_predictions(threshold=threshold)

    if output_file:
        print(f"\n✓ Success! File ready for download: {output_file}")
    else:
        print("\n✗ Failed to generate predictions")
