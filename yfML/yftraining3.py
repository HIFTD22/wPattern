#!/usr/bin/env python3
"""
STRICT Training for W-pattern recognition - High quality models only.
Uses additional quality filtering and conservative training approaches.

Usage:
  python yftraining3.py --csv w_with_enhanced_features.csv --target all
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.calibration import CalibratedClassifierCV

# Optional advanced models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# File paths - Strict versions
MODEL_SHAPE = "model_shape_strict.pkl"
MODEL_BREAKOUT = "model_breakout_strict.pkl"
MODEL_REACHTARGET = "model_reachtarget_strict.pkl"
FEATURES_FILE = "feature_list_strict.json"
SCALER_FILE = "scaler_strict.pkl"

# Enhanced feature set with strict quality metrics
STRICT_FEATURES = [
    # Core geometry & timing features
    "drop1_norm", "rise1_norm", "drop2_norm",
    "neckline_slope_per_day", "trough_sim",
    "spread_t", "p1_to_l1_t", "l1_to_p2_t", "p2_to_l2_t",
    
    # Core indicators  
    "rsi_p2", "rsi_l2",
    "vol_ratio_p2", "vol_ratio_l2",
    "px_vs_ma20_p2", "px_vs_ma20_l2", "px_vs_ma50_l2",
    
    # Enhanced regime features
    "regime_trend", "regime_volatility", "regime_volume", "regime_strength",
    
    # Enhanced volume features
    "volume_accumulation_ratio", "volume_confirmation_ratio", 
    "volume_trend_slope", "volume_consistency", "volume_momentum",
    
    # Enhanced pattern quality features (key for strict models)
    "pattern_duration_balance", "pattern_trough_consistency", 
    "pattern_peak_relationship", "pattern_clarity", "pattern_amplitude_balance",
    
    # Enhanced market context features
    "market_position", "trend_context", "volatility_context", "support_strength",
    
    # Additional strict quality features (if available)
    "pattern_symmetry", "recovery_strength", "quality_score", "overall_range"
]

def create_strict_temporal_splits(df, target_col, n_splits=3):
    """Create strict time-aware splits with quality filtering."""
    n = len(df)
    
    # For very small datasets
    if n < 15:
        split_point = max(5, int(n * 0.7))
        train_idx = np.arange(split_point)
        test_idx = np.arange(split_point, n)
        return [(train_idx, test_idx)]
    
    # Filter for high-quality patterns first if quality metrics available
    df_work = df.copy().reset_index(drop=True)
    
    # Quality filtering for strict training
    quality_mask = pd.Series(True, index=df_work.index)
    
    # Filter by pattern quality if metrics available
    if 'quality_score' in df_work.columns:
        quality_mask &= (df_work['quality_score'] >= 0.02)  # 2% minimum quality
    
    if 'overall_range' in df_work.columns:
        quality_mask &= (df_work['overall_range'] >= 0.03)  # 3% minimum range
        
    if 'trough_sim' in df_work.columns:
        quality_mask &= (df_work['trough_sim'] <= 0.15)  # Max 15% trough difference
    
    if 'pattern_clarity' in df_work.columns:
        quality_mask &= (df_work['pattern_clarity'] >= 0.5)  # Minimum clarity
    
    # Apply quality filter
    df_quality = df_work[quality_mask].copy()
    
    if len(df_quality) < 10:
        print(f"Warning: Only {len(df_quality)} high-quality patterns, using all data")
        df_quality = df_work
    else:
        print(f"Using {len(df_quality)}/{len(df_work)} high-quality patterns for training")
    
    # Sort by date for temporal splits
    date_cols = ['low2_date', 'p2_date', 'low1_date', 'p1_date']
    sort_col = None
    
    for col in date_cols:
        if col in df_quality.columns:
            df_quality[col + '_dt'] = pd.to_datetime(df_quality[col], errors='coerce')
            if df_quality[col + '_dt'].notna().sum() > len(df_quality) * 0.5:
                sort_col = col + '_dt'
                break
    
    if sort_col:
        df_quality = df_quality.sort_values(sort_col).reset_index(drop=True)
    
    # Create conservative expanding window splits
    n_quality = len(df_quality)
    splits = []
    
    for i in range(min(n_splits, 3)):
        # More conservative train sizes for strict models
        train_pct = 0.65 + i * 0.1  # Start at 65%, expand by 10%
        test_size = max(3, int(n_quality * 0.15))  # 15% for test
        
        train_size = min(n_quality - test_size, max(8, int(n_quality * train_pct)))
        
        if train_size >= 8 and (n_quality - train_size) >= 3:
            train_idx = np.arange(train_size)
            test_idx = np.arange(train_size, min(train_size + test_size, n_quality))
            
            if len(test_idx) >= 3:
                splits.append((train_idx, test_idx))
    
    # Fallback
    if len(splits) == 0:
        split_point = max(8, int(n_quality * 0.75))
        train_idx = np.arange(split_point)
        test_idx = np.arange(split_point, n_quality)
        splits = [(train_idx, test_idx)]
    
    return splits

def strict_feature_selection(X, y, feature_names, max_features=10):
    """Strict feature selection for high-quality models."""
    if len(feature_names) <= max_features:
        return feature_names
    
    # Remove features with excessive missing values
    missing_pct = np.isnan(X).mean(axis=0)
    valid_features = [f for i, f in enumerate(feature_names) if missing_pct[i] < 0.3]
    valid_X = X[:, [i for i in range(len(feature_names)) if missing_pct[i] < 0.3]]
    
    if len(valid_features) <= max_features:
        return valid_features
    
    # Fill NaN with median for feature selection
    for i in range(valid_X.shape[1]):
        col = valid_X[:, i]
        if np.any(np.isnan(col)):
            valid_X[np.isnan(col), i] = np.nanmedian(col)
    
    # Multi-stage selection for strict models
    
    # Stage 1: Statistical significance
    selector1 = SelectKBest(f_classif, k=min(15, len(valid_features)))
    selector1.fit(valid_X, y)
    
    stage1_indices = selector1.get_support(indices=True)
    stage1_features = [valid_features[i] for i in stage1_indices]
    stage1_X = valid_X[:, stage1_indices]
    
    # Stage 2: Recursive feature elimination with cross-validation
    if len(stage1_features) > max_features:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        
        # Use RFECV for optimal number of features
        rfecv = RFECV(estimator, step=1, cv=3, scoring='roc_auc', n_jobs=-1)
        rfecv.fit(stage1_X, y)
        
        # Select top features, capped at max_features
        n_features = min(max_features, rfecv.n_features_)
        rfe = RFE(estimator, n_features_to_select=n_features)
        rfe.fit(stage1_X, y)
        
        final_features = [stage1_features[i] for i, selected in enumerate(rfe.support_) if selected]
        return final_features
    
    return stage1_features

def build_strict_ensemble_model(n_samples, random_state=42):
    """Build conservative ensemble optimized for strict high-quality patterns."""
    models = []
    
    # Conservative Logistic Regression (excellent for small high-quality data)
    logistic = Pipeline([
        ('scaler', RobustScaler()),  # More robust to outliers
        ('lr', LogisticRegression(
            penalty='l1', C=0.5, class_weight='balanced', 
            solver='liblinear', random_state=random_state, max_iter=2000
        ))
    ])
    models.append(('logistic', logistic))
    
    # Conservative Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_split=max(2, n_samples//10),
        min_samples_leaf=max(1, n_samples//20), class_weight='balanced',
        random_state=random_state, max_features='sqrt'
    )
    models.append(('random_forest', rf))
    
    # Conservative Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3,
        min_samples_split=max(2, n_samples//10), min_samples_leaf=max(1, n_samples//20),
        subsample=0.8, random_state=random_state
    )
    models.append(('gradient_boosting', gb))
    
    # Add XGBoost if available (conservative settings)
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            min_child_weight=max(1, n_samples//15), subsample=0.8, 
            colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=1.0,
            random_state=random_state, eval_metric='logloss'
        )
        models.append(('xgboost', xgb))
    
    # Conservative voting ensemble
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft'
    )
    
    # Strict probability calibration
    calibrated_ensemble = CalibratedClassifierCV(
        ensemble, method='sigmoid', cv=3  # Sigmoid often better for small data
    )
    
    return calibrated_ensemble

def evaluate_strict_model(model, X, y, splits, feature_names):
    """Strict model evaluation with comprehensive metrics."""
    scores = []
    precision_scores = []
    recall_scores = []
    feature_importance = np.zeros(len(feature_names))
    
    for i, (train_idx, test_idx) in enumerate(splits):
        # Safety checks
        max_train_idx = np.max(train_idx) if len(train_idx) > 0 else -1
        max_test_idx = np.max(test_idx) if len(test_idx) > 0 else -1
        
        if max_train_idx >= len(X) or max_test_idx >= len(X):
            print(f"Warning: Split {i} has out-of-bounds indices. Skipping.")
            continue
            
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check for class balance
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"Warning: Split {i} lacks both classes. Skipping.")
            continue
            
        # Minimum size check
        if len(y_train) < 5 or len(y_test) < 2:
            print(f"Warning: Split {i} has insufficient data. Skipping.")
            continue
        
        try:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            scores.append(auc)
            
            # Precision and recall
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            print(f"  Split {i+1}: AUC={auc:.3f}, Precision={precision:.3f}, Recall={recall:.3f} (train:{len(y_train)}, test:{len(y_test)})")
            
            # Feature importance (if available)
            try:
                if hasattr(model, 'feature_importances_'):
                    feature_importance += model.feature_importances_
                elif hasattr(model, 'estimators_'):
                    # For ensemble models, try to extract importance
                    for estimator in model.estimators_:
                        if hasattr(estimator, 'feature_importances_'):
                            feature_importance += estimator.feature_importances_
                            break
            except:
                pass
                
        except Exception as e:
            print(f"Warning: Split {i} failed: {e}")
            continue
    
    if len(scores) == 0:
        print("Warning: No valid splits for evaluation!")
        return 0.5, 0.0, 0.0, feature_importance
    
    mean_auc = np.mean(scores)
    mean_precision = np.mean(precision_scores) if precision_scores else 0.0
    mean_recall = np.mean(recall_scores) if recall_scores else 0.0
    
    return mean_auc, mean_precision, mean_recall, feature_importance / len(splits)

def train_strict_target(df, target_col, filter_ws_for_outcomes=False):
    """Train strict model for specific target with quality filtering."""
    print(f"\n=== STRICT TRAINING: {target_col} ===")
    
    # Filter data
    data = df.copy()
    if filter_ws_for_outcomes and "y_shape" in data.columns:
        before_count = len(data)
        data = data[data["y_shape"].fillna(0).astype(int) == 1].copy()
        print(f"W-patterns only: {before_count} → {len(data)} rows")
    
    data = data.reset_index(drop=True)
    
    # Target validation
    if target_col not in data.columns:
        print(f"Missing column '{target_col}'")
        return None
    
    data = data[data[target_col].isin([0, 1])].copy()
    if len(data) < 8:  # Stricter minimum
        print(f"Insufficient data for strict training: {len(data)} rows (need ≥8)")
        return None
    
    y = data[target_col].astype(int).values
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    print(f"Class distribution: {n_neg} negative, {n_pos} positive")
    
    if n_pos < 2 or n_neg < 2:  # Stricter minimum
        print(f"Need ≥2 examples per class for strict training, got {n_neg}/{n_pos}")
        return None
    
    # Features
    available_features = [f for f in STRICT_FEATURES if f in data.columns]
    if len(available_features) == 0:
        print("No features found!")
        return None
    
    print(f"Available features: {len(available_features)}")
    
    # Feature matrix preparation
    X_raw = data[available_features].values.astype(float)
    
    # Robust missing value handling
    for i in range(X_raw.shape[1]):
        col = X_raw[:, i]
        if np.any(np.isnan(col)):
            # Use median, but fallback to 0 if all NaN
            median_val = np.nanmedian(col)
            X_raw[np.isnan(col), i] = median_val if not np.isnan(median_val) else 0
    
    # Strict feature selection
    selected_features = strict_feature_selection(X_raw, y, available_features, max_features=8)  # Fewer features for strict
    feature_indices = [available_features.index(f) for f in selected_features]
    X = X_raw[:, feature_indices]
    
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    
    # Strict temporal splits
    splits = create_strict_temporal_splits(data, target_col)
    print(f"Created {len(splits)} strict validation splits")
    
    # Build strict model
    model = build_strict_ensemble_model(len(data))
    mean_auc, mean_precision, mean_recall, feature_imp = evaluate_strict_model(
        model, X, y, splits, selected_features)
    
    print(f"Cross-validation Results:")
    print(f"  AUC: {mean_auc:.3f}")
    print(f"  Precision: {mean_precision:.3f}")
    print(f"  Recall: {mean_recall:.3f}")
    
    # Quality check - require minimum performance for strict models
    if mean_auc < 0.65:
        print(f"Warning: Low AUC ({mean_auc:.3f}) for strict model. Consider:")
        print("  1. More/better quality training data")
        print("  2. Different feature engineering")
        print("  3. Relaxing strict criteria")
    
    # Final training on all data
    model.fit(X, y)
    
    # Save strict model
    model_files = {
        'y_shape': MODEL_SHAPE,
        'y_breakout': MODEL_BREAKOUT, 
        'y_reachtarget': MODEL_REACHTARGET
    }
    
    model_path = model_files[target_col]
    dump(model, model_path)
    
    # Enhanced metadata for strict models
    metadata = {
        'model_type': 'strict_ensemble',
        'features': selected_features,
        'target': target_col,
        'n_samples': len(data),
        'class_distribution': {'negative': int(n_neg), 'positive': int(n_pos)},
        'cv_metrics': {
            'auc': float(mean_auc),
            'precision': float(mean_precision),
            'recall': float(mean_recall)
        },
        'feature_importance': {f: float(imp) for f, imp in zip(selected_features, feature_imp)},
        'training_config': {
            'strict_mode': True,
            'min_quality_score': 0.02,
            'min_overall_range': 0.03,
            'max_trough_similarity': 0.15,
            'calibration_method': 'sigmoid'
        },
        'quality_filters_applied': True,
        'timestamp': datetime.now().isoformat()
    }
    
    features_path = f"{target_col}_features_strict.json"
    with open(features_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved strict model: {model_path}")
    print(f"Saved metadata: {features_path}")
    
    # Feature importance analysis
    if np.sum(feature_imp) > 0:
        top_features = sorted(zip(selected_features, feature_imp), 
                            key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 features:")
        for feat, imp in top_features:
            print(f"  {feat}: {imp:.3f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Strict W-pattern training")
    parser.add_argument('--csv', required=True, help='Path to enhanced features CSV')
    parser.add_argument('--target', choices=['y_shape', 'y_breakout', 'y_reachtarget', 'all'], 
                       default='all', help='Target(s) to train')
    parser.add_argument('--min-quality', type=float, default=0.02, 
                       help='Minimum quality score for training data')
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.csv):
        print(f"Error: File {args.csv} not found")
        sys.exit(1)
    
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} patterns from {args.csv}")
    
    # Check for strict features
    strict_count = sum(1 for f in STRICT_FEATURES if f in df.columns)
    print(f"Found {strict_count}/{len(STRICT_FEATURES)} strict features")
    
    if strict_count < len(STRICT_FEATURES) * 0.7:
        print("Warning: Many strict features missing. Quality may be reduced.")
    
    # Apply global quality filtering if requested
    if args.min_quality > 0 and 'quality_score' in df.columns:
        before_count = len(df)
        df = df[df['quality_score'] >= args.min_quality].copy()
        print(f"Quality filtering: {before_count} → {len(df)} patterns")
    
    if df.empty:
        print("No patterns meet quality criteria!")
        sys.exit(1)
    
    # Train models
    targets_to_train = []
    if args.target == 'all':
        targets_to_train = [('y_shape', False), ('y_breakout', True), ('y_reachtarget', True)]
    else:
        filter_ws = args.target in ['y_breakout', 'y_reachtarget']
        targets_to_train = [(args.target, filter_ws)]
    
    trained_models = {}
    for target, filter_ws in targets_to_train:
        model = train_strict_target(df, target, filter_ws_for_outcomes=filter_ws)
        if model is not None:
            trained_models[target] = model
    
    # Save global feature list for compatibility
    all_features = [f for f in STRICT_FEATURES if f in df.columns]
    with open(FEATURES_FILE, 'w') as f:
        json.dump(all_features, f)
    
    print(f"\n=== STRICT TRAINING COMPLETE ===")
    print(f"Trained {len(trained_models)} strict models")
    print(f"Global features saved to: {FEATURES_FILE}")
    
    if len(trained_models) == 0:
        print("❌ No models were successfully trained!")
        print("Recommendations:")
        print("  1. Check data quality and class distributions")
        print("  2. Consider relaxing strict criteria")
        print("  3. Gather more high-quality training examples")
    else:
        print("✅ Strict models ready for high-quality scanning!")
        print("Usage:")
        print("  python yfscan3.py --limit 20 --min-score 0.7")
    
    # Model summary
    for target, model in trained_models.items():
        metadata_file = f"{target}_features_strict.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                metrics = metadata.get('cv_metrics', {})
                print(f"\n{target.upper()} Model Summary:")
                print(f"  AUC: {metrics.get('auc', 0):.3f}")
                print(f"  Precision: {metrics.get('precision', 0):.3f}")
                print(f"  Recall: {metrics.get('recall', 0):.3f}")
                print(f"  Features: {len(metadata.get('features', []))}")

if __name__ == "__main__":
    main()