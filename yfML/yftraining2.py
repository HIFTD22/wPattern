#!/usr/bin/env python3
"""
Enhanced training for W-pattern recognition with small datasets.
Uses ensemble methods, proper temporal validation, and feature selection.

Usage:
  python yftrain_enhanced.py --csv w_with_enhanced_features.csv --target y_shape
  python yftrain_enhanced.py --csv w_with_enhanced_features.csv --target all
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from joblib import dump, load

# Core ML libraries
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.calibration import CalibratedClassifierCV

# Optional advanced models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# File paths
MODEL_SHAPE = "model_shape_enhanced.pkl"
MODEL_BREAKOUT = "model_breakout_enhanced.pkl"
MODEL_REACHTARGET = "model_reachtarget_enhanced.pkl"
FEATURES_FILE = "feature_list_enhanced.json"
SCALER_FILE = "scaler_enhanced.pkl"

# Enhanced feature set (all features from enhanced yfslope)
ENHANCED_FEATURES = [
    # Original geometry & timing features
    "drop1_norm", "rise1_norm", "drop2_norm",
    "neckline_slope_per_day", "trough_sim",
    "spread_t", "p1_to_l1_t", "l1_to_p2_t", "p2_to_l2_t",
    
    # Original indicators  
    "rsi_p2", "rsi_l2",
    "vol_ratio_p2", "vol_ratio_l2",
    "px_vs_ma20_p2", "px_vs_ma20_l2", "px_vs_ma50_l2",
    
    # Enhanced regime features
    "regime_trend", "regime_volatility", "regime_volume", "regime_strength",
    
    # Enhanced volume features
    "volume_accumulation_ratio", "volume_confirmation_ratio", 
    "volume_trend_slope", "volume_consistency", "volume_momentum",
    
    # Enhanced pattern quality features
    "pattern_duration_balance", "pattern_trough_consistency", 
    "pattern_peak_relationship", "pattern_clarity", "pattern_amplitude_balance",
    
    # Enhanced market context features
    "market_position", "trend_context", "volatility_context", "support_strength"
]

def create_temporal_splits(df, n_splits=5):
    """Create time-aware train/test splits."""
    # Sort by pattern completion date (low2_date as primary)
    date_cols = ['low2_date', 'p2_date', 'low1_date', 'p1_date']
    sort_col = None
    
    for col in date_cols:
        if col in df.columns:
            df[col + '_dt'] = pd.to_datetime(df[col], errors='coerce')
            if df[col + '_dt'].notna().sum() > len(df) * 0.8:  # At least 80% valid dates
                sort_col = col + '_dt'
                break
    
    if sort_col is None:
        # Fallback to regular stratified splits
        print("Warning: No valid date columns found, using stratified splits")
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(df, df.iloc[:, -1]))
    
    # Sort by time
    df_sorted = df.sort_values(sort_col).reset_index()
    indices = df_sorted['index'].values
    
    # Create expanding window splits (walk-forward)
    splits = []
    n = len(indices)
    
    for i in range(n_splits):
        # Start with 50% of data, expand by 10% each split
        train_end = int(n * (0.5 + i * 0.08))
        test_start = train_end
        test_end = min(n, int(n * (0.65 + i * 0.08)))
        
        if test_end > test_start and train_end > 10:  # Ensure minimum sizes
            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]
            splits.append((train_idx, test_idx))
    
    if len(splits) < 3:  # Fallback if too few splits
        return list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(df, df.iloc[:, -1]))
    
    return splits

def select_features(X, y, feature_names, max_features=15):
    """Select best features to avoid overfitting with small datasets."""
    if len(feature_names) <= max_features:
        return feature_names
    
    # Remove features with too many missing values
    missing_pct = np.isnan(X).mean(axis=0)
    valid_features = [f for i, f in enumerate(feature_names) if missing_pct[i] < 0.5]
    valid_X = X[:, [i for i in range(len(feature_names)) if missing_pct[i] < 0.5]]
    
    if len(valid_features) <= max_features:
        return valid_features
    
    # Use univariate selection + RFE
    # Fill remaining NaN with median
    for i in range(valid_X.shape[1]):
        col = valid_X[:, i]
        if np.any(np.isnan(col)):
            valid_X[np.isnan(col), i] = np.nanmedian(col)
    
    # Univariate selection (top 20)
    selector1 = SelectKBest(f_classif, k=min(20, len(valid_features)))
    selector1.fit(valid_X, y)
    
    # Get top features
    selected_indices = selector1.get_support(indices=True)
    selected_features = [valid_features[i] for i in selected_indices]
    selected_X = valid_X[:, selected_indices]
    
    # RFE for final selection
    if len(selected_features) > max_features:
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        rfe = RFE(estimator, n_features_to_select=max_features, step=1)
        rfe.fit(selected_X, y)
        
        final_features = [selected_features[i] for i, selected in enumerate(rfe.support_) if selected]
        return final_features
    
    return selected_features

def build_ensemble_model(random_state=42):
    """Build ensemble model optimized for small datasets."""
    models = []
    
    # Logistic regression (works well with small data)
    logistic = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            penalty='l2', C=1.0, class_weight='balanced', 
            solver='liblinear', random_state=random_state, max_iter=1000
        ))
    ])
    models.append(('logistic', logistic))
    
    # Random Forest (good for small data with proper settings)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced',
        random_state=random_state
    )
    models.append(('random_forest', rf))
    
    # Gradient Boosting (conservative settings)
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        min_samples_split=5, min_samples_leaf=2,
        random_state=random_state
    )
    models.append(('gradient_boosting', gb))
    
    # Add XGBoost if available
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, eval_metric='logloss'
        )
        models.append(('xgboost', xgb))
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft'
    )
    
    # Calibrate probabilities
    calibrated_ensemble = CalibratedClassifierCV(
        ensemble, method='isotonic', cv=3
    )
    
    return calibrated_ensemble

def evaluate_model(model, X, y, splits, feature_names):
    """Evaluate model with temporal validation."""
    scores = []
    feature_importance = np.zeros(len(feature_names))
    
    for i, (train_idx, test_idx) in enumerate(splits):
        # Safety checks for index bounds
        max_train_idx = np.max(train_idx) if len(train_idx) > 0 else -1
        max_test_idx = np.max(test_idx) if len(test_idx) > 0 else -1
        
        if max_train_idx >= len(X) or max_test_idx >= len(X):
            print(f"Warning: Split {i} has out-of-bounds indices. Skipping.")
            continue
            
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check for class imbalance
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"Warning: Split {i} lacks both classes. Skipping.")
            continue
            
        # Check minimum sizes
        if len(y_train) < 3 or len(y_test) < 2:
            print(f"Warning: Split {i} has insufficient data (train: {len(y_train)}, test: {len(y_test)}). Skipping.")
            continue
            
        # Fit and predict
        try:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate AUC
            auc = roc_auc_score(y_test, y_pred_proba)
            scores.append(auc)
            print(f"  Split {i+1}: AUC = {auc:.3f} (train: {len(y_train)}, test: {len(y_test)})")
            
            # Accumulate feature importance (if available)
            try:
                if hasattr(model.base_estimator, 'feature_importances_'):
                    feature_importance += model.base_estimator.feature_importances_
                elif hasattr(model, 'feature_importances_'):
                    feature_importance += model.feature_importances_
            except:
                pass
                
        except Exception as e:
            print(f"Warning: Split {i} failed with error: {e}")
            continue
    
    if len(scores) == 0:
        print("Warning: No valid splits for evaluation!")
        return 0.5, feature_importance
    
    return np.mean(scores), feature_importance / len(splits)

def train_target(df, target_col, filter_ws_for_outcomes=False):
    """Train model for specific target."""
    print(f"\n=== Training {target_col} ===")
    
    # Filter data if needed
    data = df.copy()
    if filter_ws_for_outcomes and "y_shape" in data.columns:
        before_count = len(data)
        data = data[data["y_shape"].fillna(0).astype(int) == 1].copy()
        print(f"Filtered to W-patterns only: {before_count} → {len(data)} rows")
    
    # Get target
    if target_col not in data.columns:
        print(f"Missing column '{target_col}'")
        return None
    
    data = data[data[target_col].isin([0, 1])].copy()
    if len(data) < 10:
        print(f"Insufficient data for {target_col}: {len(data)} rows")
        return None
    
    y = data[target_col].astype(int).values
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    print(f"Class distribution: {n_neg} negative, {n_pos} positive")
    
    if n_pos < 3 or n_neg < 3:
        print(f"Need at least 3 examples per class, got {n_neg}/{n_pos}")
        return None
    
    # Prepare features
    available_features = [f for f in ENHANCED_FEATURES if f in data.columns]
    if len(available_features) == 0:
        print("No features found! Run enhanced feature engineering first.")
        return None
    
    print(f"Available features: {len(available_features)}")
    
    # Extract and clean feature matrix
    X_raw = data[available_features].values.astype(float)
    
    # Handle missing values
    for i in range(X_raw.shape[1]):
        col = X_raw[:, i]
        if np.any(np.isnan(col)):
            median_val = np.nanmedian(col)
            X_raw[np.isnan(col), i] = median_val if not np.isnan(median_val) else 0
    
    # Feature selection
    selected_features = select_features(X_raw, y, available_features, max_features=12)
    feature_indices = [available_features.index(f) for f in selected_features]
    X = X_raw[:, feature_indices]
    
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    
    # Create temporal splits
    splits = create_temporal_splits(data)
    print(f"Created {len(splits)} temporal validation splits")
    
    # Build and evaluate model
    model = build_ensemble_model()
    mean_auc, feature_imp = evaluate_model(model, X, y, splits, selected_features)
    
    print(f"Cross-validation AUC: {mean_auc:.3f}")
    
    # Final training on all data
    model.fit(X, y)
    
    # Save model and metadata
    model_files = {
        'y_shape': MODEL_SHAPE,
        'y_breakout': MODEL_BREAKOUT, 
        'y_reachtarget': MODEL_REACHTARGET
    }
    
    model_path = model_files[target_col]
    dump(model, model_path)
    
    # Save features and scaler info
    metadata = {
        'features': selected_features,
        'target': target_col,
        'n_samples': len(data),
        'class_distribution': {'negative': int(n_neg), 'positive': int(n_pos)},
        'cv_auc': float(mean_auc),
        'feature_importance': {f: float(imp) for f, imp in zip(selected_features, feature_imp)}
    }
    
    features_path = f"{target_col}_features.json"
    with open(features_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {features_path}")
    
    # Top features by importance
    if np.sum(feature_imp) > 0:
        top_features = sorted(zip(selected_features, feature_imp), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 features:")
        for feat, imp in top_features:
            print(f"  {feat}: {imp:.3f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Enhanced W-pattern training")
    parser.add_argument('--csv', required=True, help='Path to enhanced features CSV')
    parser.add_argument('--target', choices=['y_shape', 'y_breakout', 'y_reachtarget', 'all'], 
                       default='all', help='Target(s) to train')
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.csv):
        print(f"Error: File {args.csv} not found")
        print("Run: python yfslope_enhanced.py --in w_data.csv --out w_with_enhanced_features.csv")
        sys.exit(1)
    
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} patterns from {args.csv}")
    
    # Check for enhanced features
    enhanced_count = sum(1 for f in ENHANCED_FEATURES if f in df.columns)
    print(f"Found {enhanced_count}/{len(ENHANCED_FEATURES)} enhanced features")
    
    if enhanced_count < len(ENHANCED_FEATURES) * 0.5:
        print("Warning: Many enhanced features missing. Did you run yfslope_enhanced.py?")
    
    # Train models
    targets_to_train = []
    if args.target == 'all':
        targets_to_train = [('y_shape', False), ('y_breakout', True), ('y_reachtarget', True)]
    else:
        filter_ws = args.target in ['y_breakout', 'y_reachtarget']
        targets_to_train = [(args.target, filter_ws)]
    
    trained_models = {}
    for target, filter_ws in targets_to_train:
        model = train_target(df, target, filter_ws_for_outcomes=filter_ws)
        if model is not None:
            trained_models[target] = model
    
    # Save global feature list for compatibility
    all_features = [f for f in ENHANCED_FEATURES if f in df.columns]
    with open(FEATURES_FILE, 'w') as f:
        json.dump(all_features, f)
    
    print(f"\n=== Training Complete ===")
    print(f"Trained {len(trained_models)} models")
    print(f"Global features saved to: {FEATURES_FILE}")
    
    if len(trained_models) == 0:
        print("Warning: No models were successfully trained!")
        print("Check your data quality and class distributions.")
    else:
        print("\nReady for scanning! Use:")
        print("python yfscan_enhanced.py --limit 50")

if __name__ == "__main__":
    main()