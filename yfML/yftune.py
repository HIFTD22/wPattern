#!/usr/bin/env python3
import argparse, os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from joblib import dump
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Optional models (skipped if not installed)
HAS_XGB = HAS_LGBM = HAS_CAT = True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False
try:
    import lightgbm as lgb
except Exception:
    HAS_LGBM = False
try:
    from catboost import CatBoostClassifier
except Exception:
    HAS_CAT = False

FEATURES_FILE = "feature_list.json"

def load_features():
    if not os.path.exists(FEATURES_FILE):
        print(f"ERROR: {FEATURES_FILE} not found (train once to write it).", file=sys.stderr)
        sys.exit(1)
    with open(FEATURES_FILE, "r") as f:
        return json.load(f)

def parse_datecol(s):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

def make_splits_by_time(df, n_splits=3):
    # Sort by low2_trading_date -> low2_date -> p2_trading_date as fallbacks
    for col in ["low2_trading_date","low2_date","p2_trading_date","p2_date"]:
        if col in df.columns:
            d = pd.to_datetime(df[col], errors="coerce")
            if d.notna().any():
                df = df.assign(_sort_date=d).sort_values("_sort_date")
                break
    else:
        # if no dates, just deterministic split
        df = df.assign(_sort_date=np.arange(len(df)))
        df = df.sort_values("_sort_date")

    idx = df.index.to_numpy()
    n = len(idx)
    if n < 12:  # too small, still return one split
        cuts = [int(n*0.7)]
        splits = [ (idx[:cuts[0]], idx[cuts[0]:]) ]
        return splits

    # Walk-forward cut points ~60%, 75%, 90%
    cuts = sorted(set([int(n*0.6), int(n*0.75), int(n*0.9)]))
    splits = []
    for c in cuts:
        if c < 4 or (n-c) < 4: 
            continue
        train_idx = idx[:c]
        test_idx  = idx[c:]
        splits.append((train_idx, test_idx))
    if not splits:
        splits = [ (idx[:int(n*0.7)], idx[int(n*0.7):]) ]
    return splits

def evaluate_model(model, X, y, splits):
    scores = []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:,1]
        scores.append(roc_auc_score(y[te], p))
    return float(np.mean(scores)) if scores else np.nan

def build_candidates(target):
    # smaller grids to keep it quick
    cands = []

    # Logistic (great for small N, interpretable)
    cands += [
        ("logit_l2", Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(
            penalty="l2", C=C, class_weight="balanced", solver="liblinear", max_iter=200))]))
        for C in [0.1, 1.0, 10.0]
    ]
    cands += [
        ("logit_l1", Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(
            penalty="l1", C=C, class_weight="balanced", solver="liblinear", max_iter=200))]))
        for C in [0.1, 1.0, 10.0]
    ]

    # RandomForest
    for n_est in [200]:
        for md in [None, 5]:
            for msl in [1, 3]:
                cands.append((
                    f"rf_{n_est}_md{md}_msl{msl}",
                    RandomForestClassifier(
                        n_estimators=n_est, max_depth=md, min_samples_leaf=msl,
                        class_weight="balanced", random_state=42)
                ))

    # GradientBoosting (baseline)
    for lr in [0.05, 0.1]:
        for md in [2, 3]:
            cands.append((
                f"gb_lr{lr}_md{md}",
                GradientBoostingClassifier(random_state=42, learning_rate=lr, max_depth=md, n_estimators=200)
            ))

    # XGBoost (if available)
    if HAS_XGB:
        for lr in [0.05, 0.1]:
            cands.append((
                f"xgb_lr{lr}",
                XGBClassifier(
                    n_estimators=300, learning_rate=lr, max_depth=3,
                    subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                    eval_metric="auc", tree_method="hist", random_state=42)
            ))

    # LightGBM (if available)
    if HAS_LGBM:
        for lr in [0.05, 0.1]:
            cands.append((
                f"lgbm_lr{lr}",
                lgb.LGBMClassifier(
                    n_estimators=400, learning_rate=lr, num_leaves=31,
                    subsample=0.9, colsample_bytree=0.8, reg_lambda=0.0,
                    objective="binary", random_state=42)
            ))

    # CatBoost (if available)
    if HAS_CAT:
        for lr in [0.05, 0.1]:
            cands.append((
                f"cat_lr{lr}",
                CatBoostClassifier(
                    iterations=400, learning_rate=lr, depth=4,
                    loss_function="Logloss", eval_metric="AUC",
                    verbose=False, random_seed=42)
            ))
    return cands

def main():
    ap = argparse.ArgumentParser(description="Quick model tuner with walk-forward validation")
    ap.add_argument("--csv", required=True, help="Path to w_with_features.csv")
    ap.add_argument("--target", required=True, choices=["y_shape","y_breakout","y_reachtarget"])
    ap.add_argument("--save", required=True, help="Path to save best model (e.g., model_shape.pkl)")
    ap.add_argument("--report", default="tune_report.json", help="Path to write leaderboard JSON")
    args = ap.parse_args()

    feats = load_features()
    df = pd.read_csv(args.csv)

    # For outcome targets, restrict to Ws like in your training
    if args.target != "y_shape" and "y_shape" in df.columns:
        df = df[df["y_shape"].fillna(0).astype(int) == 1].copy()

    # Keep labeled rows only
    df = df[df[args.target].isin([0,1])].copy()
    if df.empty:
        print("No labeled rows for target.", file=sys.stderr); sys.exit(1)

    # Features
    for f in feats:
        if f not in df.columns:
            print(f"Missing feature '{f}'. Did you run yfslope1.py?", file=sys.stderr); sys.exit(1)

    X = df[feats].astype(float).fillna(0.0).to_numpy()
    y = df[args.target].astype(int).to_numpy()

    # Walk-forward splits
    splits = make_splits_by_time(df)

    # Candidates
    cands = build_candidates(args.target)

    # Evaluate
    board = []
    for name, model in cands:
        try:
            auc = evaluate_model(model, X, y, splits)
        except Exception as e:
            auc = np.nan
        board.append({"name": name, "auc": None if np.isnan(auc) else float(auc), "ok": not np.isnan(auc)})

    # Rank
    board = [b for b in board if b["ok"]]
    board.sort(key=lambda d: d["auc"], reverse=True)
    if not board:
        print("All candidates failed. Check data.", file=sys.stderr); sys.exit(1)

    best_name = board[0]["name"]
    print("Leaderboard (top 10):")
    for row in board[:10]:
        print(f"  {row['name']:20s}  AUC={row['auc']:.3f}")

    # Fit best on ALL data, save
    best_model = None
    for name, model in cands:
        if name == best_name:
            best_model = model
            break
    best_model.fit(X, y)
    dump(best_model, args.save)
    with open(args.report, "w") as f:
        json.dump({"target": args.target, "best": board[0], "leaderboard": board}, f, indent=2)
    print(f"Saved best model '{best_name}' → {args.save}")
    print(f"Report → {args.report}")

if __name__ == "__main__":
    main()
