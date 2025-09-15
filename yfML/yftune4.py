#!/usr/bin/env python3
"""
yftune.py — Quick tuner (time-aware or stratified CV) for W-shape pipeline.

Models tried:
  - Logistic (L1/L2), RandomForest, GradientBoosting
  - XGBoost / LightGBM / CatBoost (if installed)
  - LightGBM with monotone constraints (domain priors)
  - MiniROCKET ➜ Logistic (sequence view, OHLCV windows built per row)
  - Simple 2-view Stacker: [MiniROCKET➜Logit, LGBM(mono)] ➜ Logistic meta (with Platt calibration)

Usage (recommend stratified first for your current small N):
  python yftune.py --csv w_with_features.csv --target y_breakout --cv stratified --splits 3 \
    --save model_breakout.pkl --report tune_breakout.json
"""

import argparse, os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from joblib import dump
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Optional libs
HAS_XGB = HAS_LGBM = HAS_CAT = HAS_ROCKET = True
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
try:
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    from sktime.utils.data_processing import from_3d_numpy_to_nested
except Exception:
    HAS_ROCKET = False

import yfinance as yf

FEATURES_FILE = "feature_list.json"
RECENT_WINDOW = 90

# ---------- utilities ----------

def load_features():
    if not os.path.exists(FEATURES_FILE):
        print(f"ERROR: {FEATURES_FILE} not found (train once to write it).", file=sys.stderr); sys.exit(1)
    with open(FEATURES_FILE, "r") as f:
        return json.load(f)

def norm_symbol(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")

def parse_date(s: str):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

def robust_history_to(end_date: pd.Timestamp, ticker: str, length: int = RECENT_WINDOW) -> np.ndarray:
    """Return OHLCV window ending at end_date (inclusive), shape (5, length)."""
    tkr = norm_symbol(ticker)
    end = end_date.date() if isinstance(end_date, pd.Timestamp) else end_date
    start_pad = end - timedelta(days=365)
    try:
        df = yf.Ticker(tkr).history(start=start_pad.isoformat(),
                                    end=(end + timedelta(days=5)).isoformat(),
                                    interval="1d", auto_adjust=True)
        df = df[["Open","High","Low","Close","Volume"]].dropna().sort_index()
        # align end_date to nearest trading date within 3d
        dates = np.array([d.date() for d in df.index])
        diffs = np.array([abs((d - end).days) for d in dates])
        j = int(diffs.argmin())
        if diffs[j] > 3:
            return None
        # get last 'length' rows ending at j
        start = max(0, j - (length - 1))
        win = df.iloc[start:j+1].copy()
        if len(win) < length:
            return None
        arr = win[["Open","High","Low","Close","Volume"]].to_numpy(dtype=float).T  # (5, length)
        # per-channel zscore for stability
        arr = (arr - arr.mean(axis=1, keepdims=True)) / (arr.std(axis=1, keepdims=True) + 1e-9)
        return arr
    except Exception:
        return None

def build_seq_panel(df_rows: pd.DataFrame, target_col: str) -> (object, np.ndarray):
    """
    Build sktime nested panel X_seq for MiniROCKET:
      Returns X_nested (sktime format) and y (numpy).
    Uses each row's low2_trading_date (or low2_date) as the window endpoint.
    """
    X_list = []
    keep_idx = []
    for idx, r in df_rows.iterrows():
        end_str = r.get("low2_trading_date", r.get("low2_date", ""))
        end_dt = parse_date(end_str)
        tkr = r.get("ticker", "")
        arr = robust_history_to(end_dt, tkr, RECENT_WINDOW)
        if arr is None:
            continue
        # sktime expects shape (n_instances, n_timepoints, n_channels)
        X_list.append(arr.T)  # (len, 5)
        keep_idx.append(idx)
    if not keep_idx:
        return None, None
    X3d = np.stack(X_list, axis=0)  # (n, length, 5)
    X_nested = from_3d_numpy_to_nested(X3d)  # sktime nested panel
    y = df_rows.loc[keep_idx, target_col].astype(int).to_numpy()
    return (X_nested, y), np.array(keep_idx, dtype=int)

def make_time_splits(df, y, n_splits=3):
    # time order by low2_trading_date -> low2_date -> p2_trading_date
    sort_col = None
    for col in ["low2_trading_date","low2_date","p2_trading_date","p2_date"]:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce")
            if s.notna().any():
                df = df.assign(_sort=s).sort_values("_sort"); sort_col = col; break
    if sort_col is None:
        df = df.assign(_sort=np.arange(len(df))).sort_values("_sort")
    idx = df.index.to_numpy()
    n = len(idx)
    if n < 12: 
        return [(idx[:int(n*0.7)], idx[int(n*0.7):])]
    cuts = sorted(set([int(n*0.6), int(n*0.75), int(n*0.9)]))
    splits = []
    for c in cuts:
        tr, te = idx[:c], idx[c:]
        if len(te) >= 4:
            # ensure both classes exist in test
            if len(np.unique(y[np.searchsorted(idx, te)])) >= 2:
                splits.append((tr, te))
    if not splits:
        splits = [(idx[:int(n*0.7)], idx[int(n*0.7):])]
    return splits

def make_stratified_splits(y, n_splits=3):
    sk = StratifiedKFold(n_splits=min(n_splits, np.bincount(y).min(), 5), shuffle=True, random_state=42)
    idx = np.arange(len(y))
    return [(tr, te) for tr, te in sk.split(idx, y)]

def evaluate_tabular(model, X, y, splits):
    scores = []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2: continue
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:,1]
        scores.append(roc_auc_score(y[te], p))
    return float(np.mean(scores)) if scores else np.nan

def evaluate_sequence(pipe, X_nested, y, splits, keep_idx, full_length):
    scores = []
    # map splits from original df indices -> kept indices
    idx_map = {ix:i for i, ix in enumerate(keep_idx)}
    for tr, te in splits:
        tr_k = [idx_map[i] for i in tr if i in idx_map]
        te_k = [idx_map[i] for i in te if i in idx_map]
        if len(tr_k) < 5 or len(te_k) < 4: 
            continue
        if len(np.unique(y[tr_k])) < 2 or len(np.unique(y[te_k])) < 2:
            continue
        pipe.fit(X_nested.iloc[tr_k], y[tr_k])
        p = pipe.predict_proba(X_nested.iloc[te_k])[:,1]
        scores.append(roc_auc_score(y[te_k], p))
    return float(np.mean(scores)) if scores else np.nan

# ---------- model candidates ----------

def build_tabular_candidates():
    cands = []
    # Logistic
    for pen in ["l2","l1"]:
        for C in [0.1, 1.0, 10.0]:
            cands.append((
                f"logit_{pen}_C{C}",
                Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(penalty=pen, C=C, class_weight="balanced",
                                                    solver="liblinear", max_iter=800))])
            ))
    # RF
    for n_est in [300]:
        for md in [None, 5]:
            for msl in [1, 3]:
                cands.append((
                    f"rf_{n_est}_md{md}_msl{msl}",
                    RandomForestClassifier(n_estimators=n_est, max_depth=md,
                                          min_samples_leaf=msl, class_weight="balanced",
                                          random_state=42)
                ))
    # GB
    for lr in [0.05, 0.1]:
        for md in [2, 3]:
            cands.append((
                f"gb_lr{lr}_md{md}",
                GradientBoostingClassifier(learning_rate=lr, max_depth=md,
                                           n_estimators=300, random_state=42)
            ))
    # XGB/LGBM/CAT
    if HAS_XGB:
        for lr in [0.05, 0.1]:
            cands.append((
                f"xgb_lr{lr}",
                XGBClassifier(n_estimators=400, learning_rate=lr, max_depth=3,
                              subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                              eval_metric="auc", tree_method="hist", random_state=42)
            ))
    if HAS_LGBM:
        for lr in [0.05, 0.1]:
            cands.append((
                f"lgbm_lr{lr}",
                lgb.LGBMClassifier(n_estimators=500, learning_rate=lr, num_leaves=31,
                                   subsample=0.9, colsample_bytree=0.8, reg_lambda=0.0,
                                   objective="binary", random_state=42)
            ))
        # LGBM with monotone constraints (domain priors)
        # Map directions: +1 vol_ratio*, -1 rsi*, -1 px_vs_ma*, -1 dist_to_neckline_l2
        # Others 0 (unknown)
    return cands

def monotone_vector(feature_names):
    monos = []
    for f in feature_names:
        if f.startswith("vol_ratio_"):
            monos.append(1)
        elif f.startswith("rsi_") or f.startswith("px_vs_ma"):
            monos.append(-1)
        elif f == "dist_to_neckline_l2":
            monos.append(-1)
        else:
            monos.append(0)
    return monos

class TwoViewStacker(BaseEstimator, ClassifierMixin):
    """
    Simple two-view stacker:
      - Base A: MiniROCKET➜Logistic (sequence)
      - Base B: LightGBM (monotone constraints) on tabular features
      - Each base is calibrated (Platt)
      - Meta: Logistic on [pA, pB]
    Saves everything; predict_proba() returns meta probs.
    """
    def __init__(self, feature_names, random_state=42):
        self.feature_names = feature_names
        self.random_state = random_state
        self.base_seq = None
        self.base_tab = None
        self.cal_seq = None
        self.cal_tab = None
        self.meta = None
        self.keep_idx_ = None
        self.X_nested_ = None
        self.y_seq_ = None

    def fit(self, df, X_tab, y, splits=None):
        # Build sequence panel (subset)
        if not HAS_ROCKET or not HAS_LGBM:
            raise RuntimeError("MiniROCKET or LightGBM not available.")
        (X_nested, y_seq), keep_idx = build_seq_panel(df, df.columns[df.columns.str.fullmatch("y_.*")][0])
        # Use the SAME target values on kept rows
        y_seq = df.loc[keep_idx, df.columns[df.columns.str.fullmatch("y_.*")][0]].astype(int).to_numpy()
        self.keep_idx_ = keep_idx
        self.X_nested_ = X_nested
        self.y_seq_ = y_seq

        # Base models
        seq_pipe = make_pipeline(
            MiniRocketMultivariate(num_kernels=10000, random_state=self.random_state),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        )
        lgbm = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.8, objective="binary",
            random_state=self.random_state,
            monotone_constraints=monotone_vector(self.feature_names)
        )

        # Calibrated wrappers
        self.cal_seq = CalibratedClassifierCV(seq_pipe, method="sigmoid", cv=3)
        self.cal_tab = CalibratedClassifierCV(lgbm, method="sigmoid", cv=3)

        # Train calibrated bases
        self.cal_seq.fit(self.X_nested_, self.y_seq_)
        self.cal_tab.fit(X_tab, y)

        # Build meta training set (align rows)
        pA = np.full(len(df), np.nan)
        pA[self.keep_idx_] = self.cal_seq.predict_proba(self.X_nested_)[:,1]
        pB = self.cal_tab.predict_proba(X_tab)[:,1]
        mask = ~np.isnan(pA)
        Z = np.c_[pA[mask], pB[mask]]
        yZ = y[mask]

        self.meta = LogisticRegression(max_iter=1000)
        self.meta.fit(Z, yZ)
        return self

    def predict_proba(self, df, X_tab):
        pB = self.cal_tab.predict_proba(X_tab)[:,1]
        # For sequence, rebuild windows for incoming df
        (X_nested, _), keep_idx = build_seq_panel(df, df.columns[df.columns.str.fullmatch("y_.*")][0])
        pA_full = np.full(len(df), np.nan)
        if X_nested is not None:
            pA = self.cal_seq.predict_proba(X_nested)[:,1]
            pA_full[keep_idx] = pA
            # fallback: if missing sequence window, use tabular only
            pA_full = np.nan_to_num(pA_full, nan=pB)
        else:
            pA_full = pB
        Z = np.c_[pA_full, pB]
        pr = self.meta.predict_proba(Z)
        return pr

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Quick tuner (time-aware or stratified CV)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True, choices=["y_shape","y_breakout","y_reachtarget"])
    ap.add_argument("--save", required=True)
    ap.add_argument("--report", default="tune_report.json")
    ap.add_argument("--cv", choices=["time","stratified"], default="time")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--try-stack", action="store_true", help="Try 2-view stacker (MiniROCKET+LGBM)")
    args = ap.parse_args()

    feats = load_features()
    df = pd.read_csv(args.csv)

    # For outcomes: keep only Ws
    if args.target != "y_shape" and "y_shape" in df.columns:
        df = df[df["y_shape"].fillna(0).astype(int) == 1].copy()
    df = df[df[args.target].isin([0,1])].copy()
    if df.empty:
        print("No labeled rows for target.", file=sys.stderr); sys.exit(1)

    # ---- Tabular features
    for f in feats:
        if f not in df.columns:
            print(f"Missing feature '{f}'. Run yfslope1.py first.", file=sys.stderr); sys.exit(1)
    X_tab = df[feats].astype(float).fillna(0.0).to_numpy()
    y = df[args.target].astype(int).to_numpy()

    # CV splits
    if args.cv == "time":
        splits = make_time_splits(df.copy(), y, n_splits=args.splits)
        # Fallback to stratified if time splits end up single-class
        if all(len(np.unique(y[np.searchsorted(df.index.to_numpy(), te)])) < 2 for _, te in splits):
            splits = make_stratified_splits(y, n_splits=min(args.splits, 5))
    else:
        splits = make_stratified_splits(y, n_splits=min(args.splits, 5))

    # ---- Tabular candidates
    board = []
    tab_cands = build_tabular_candidates()
    # Add LGBM with monotone constraints candidate if available
    if HAS_LGBM:
        tab_cands.append((
            "lgbm_mono",
            lgb.LGBMClassifier(
                n_estimators=600, learning_rate=0.05, num_leaves=31,
                subsample=0.9, colsample_bytree=0.8, objective="binary",
                random_state=42,
                monotone_constraints=monotone_vector(feats)
            )
        ))

    for name, model in tab_cands:
        try:
            auc = evaluate_tabular(model, X_tab, y, splits)
        except Exception:
            auc = np.nan
        if not np.isnan(auc):
            board.append({"name": name, "auc": float(auc), "view": "tabular"})

    # ---- Sequence candidate (MiniROCKET➜Logit)
    if HAS_ROCKET:
        try:
            (X_seq, y_seq), keep_idx = build_seq_panel(df.assign(**{args.target: y}), args.target)
            if X_seq is not None:
                seq_pipe = make_pipeline(
                    MiniRocketMultivariate(num_kernels=10000, random_state=42),
                    LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
                )
                auc_seq = evaluate_sequence(seq_pipe, X_seq, y_seq, splits, keep_idx, RECENT_WINDOW)
                if not np.isnan(auc_seq):
                    board.append({"name": "minirocket_logit", "auc": float(auc_seq), "view": "sequence"})
        except Exception:
            pass

    if args.try_stack and HAS_ROCKET and HAS_LGBM and len(df) >= 15:
        try:
            stack = TwoViewStacker(feature_names=feats, random_state=42)
            # Use stratified folds for oof-like meta training if time splits are too small
            stack.fit(df.assign(**{args.target: y}), X_tab, y, splits=splits)
            # quick in-sample AUC (sanity only)
            pr = stack.predict_proba(df.assign(**{args.target: y}), X_tab)[:,1]
            auc_s = roc_auc_score(y, pr)
            board.append({"name": "stack2", "auc": float(auc_s), "view": "stack"})
        except Exception:
            pass

    if not board:
        print("All candidates failed. Check data.", file=sys.stderr); sys.exit(1)

    # Rank and save best
    board.sort(key=lambda d: d["auc"], reverse=True)
    print("Leaderboard (top 10):")
    for r in board[:10]:
        print(f"  {r['name']:18s}  AUC={r['auc']:.3f}  view={r['view']}")

    best = board[0]["name"]
    print(f"Best: {best}")

    # Fit best on ALL data and save
    if best == "minirocket_logit" and HAS_ROCKET:
        (X_seq, y_seq), keep_idx = build_seq_panel(df.assign(**{args.target: y}), args.target)
        seq_pipe = make_pipeline(
            MiniRocketMultivariate(num_kernels=10000, random_state=42),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        )
        seq_pipe.fit(X_seq, y_seq)
        dump(seq_pipe, args.save)
    elif best == "stack2" and HAS_ROCKET and HAS_LGBM:
        stack = TwoViewStacker(feature_names=feats, random_state=42)
        stack.fit(df.assign(**{args.target: y}), X_tab, y, splits=splits)
        dump(stack, args.save)
    else:
        # find model by name and fit on all data
        mdl = None
        for name, model in tab_cands:
            if name == best:
                mdl = model; break
        if mdl is None:
            # fallback to logistic
            mdl = Pipeline([("sc", StandardScaler()),
                            ("lr", LogisticRegression(max_iter=800, class_weight="balanced", solver="liblinear"))])
        mdl.fit(X_tab, y)
        dump(mdl, args.save)

    with open(args.report, "w") as f:
        json.dump({"target": args.target, "best": board[0], "leaderboard": board}, f, indent=2)
    print(f"Saved best model → {args.save}")
    print(f"Report → {args.report}")

if __name__ == "__main__":
    main()
