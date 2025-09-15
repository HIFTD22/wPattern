#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wml4.py — W-Pattern labeling, training, and scanning (S&P500 / Nasdaq-100)

Menu:
1) Train more data (≈5 candidates to label; auto-retrain -> model_shape.pkl)
2) Check one ticker (90d) and plot
3) Scan S&P 500 (targets via your formula; auto-use model if present)
4) Scan Nasdaq-100 (targets via your formula; auto-use model if present)
q) Exit

Key behaviors:
- Peaks/troughs use Open/Close preference (peak=max(Open,Close), trough=min(Open,Close)).
- Tilt filter (P1 ≥ P2 * (1 - p1_ge_tol)).
- Labels saved to labels_shape.csv; Option 1 auto-retrains RandomForest and saves model_shape.pkl.
- Scans auto-load model_shape.pkl if present (else heuristics).
"""

import os
import io
import json
import joblib
import math
import random
import warnings
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ----------------------------
# Config
# ----------------------------
LOOKBACK_DAYS_DEFAULT = 90
LABEL_CSV = "labels_shape.csv"         # unified label store
MODEL_PATH = "model_shape.pkl"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# Only train on these features (we also store extra fields, but model uses these)
FEATURE_COLUMNS = [
    "dur_p1_l2", "dur_l1_l2", "dur_p1_l1", "dur_l1_p2", "dur_p2_l2",
    "p1_price", "p2_price", "l1_price", "l2_price",
    "depth_symmetry", "height", "width",
    "neckline_slope", "trough_midpoint", "target_magnitude",
    "close_now", "close_vs_neck",
    "vol_mean", "vol_now",
]

# ----------------------------
# Symbol loaders
# ----------------------------
def _normalize_symbol(sym: str) -> str:
    # yfinance uses '-' instead of '.' for tickers like BRK.B
    return sym.strip().upper().replace(".", "-")

def load_sp500_symbols() -> List[str]:
    if os.path.exists("sp500.txt"):
        with open("sp500.txt", "r") as f:
            syms = [_normalize_symbol(s) for s in f.read().splitlines() if s.strip()]
        print(f"📊 Loaded {len(syms)} from sp500.txt")
        return syms
    try:
        print("⚠️ sp500.txt missing; attempting Wikipedia…")
        resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), flavor="lxml")
        df = tables[0]
        syms = [_normalize_symbol(s) for s in df["Symbol"].astype(str).tolist()]
        with open("sp500.txt", "w") as f:
            f.write("\n".join(syms))
        print(f"📊 Loaded {len(syms)} from Wikipedia → cached to sp500.txt")
        return syms
    except Exception as e:
        print(f"❌ Could not load S&P 500 list: {e}")
        return []

def load_nasdaq100_symbols() -> List[str]:
    if os.path.exists("nasdaq100.txt"):
        with open("nasdaq100.txt", "r") as f:
            syms = [_normalize_symbol(s) for s in f.read().splitlines() if s.strip()]
        print(f"📊 Loaded {len(syms)} from nasdaq100.txt")
        return syms
    try:
        print("⚠️ nasdaq100.txt missing; attempting Wikipedia…")
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), flavor="lxml")
        candidates = [t for t in tables if any(str(c).lower() in ("symbol", "ticker") for c in t.columns)]
        if not candidates:
            candidates = [tables[0]]
        df = candidates[0]
        col = None
        for c in df.columns:
            if str(c).strip().lower() in ("symbol", "ticker"):
                col = c; break
        if col is None:
            col = df.columns[0]
        syms = [_normalize_symbol(s) for s in df[col].astype(str).tolist()]
        with open("nasdaq100.txt", "w") as f:
            f.write("\n".join(syms))
        print(f"📊 Loaded {len(syms)} from Wikipedia → cached to nasdaq100.txt")
        return syms
    except Exception as e:
        print(f"❌ Could not load Nasdaq-100 list: {e}")
        return []

# ----------------------------
# Data fetch
# ----------------------------
def fetch_history(symbol: str, days: int = LOOKBACK_DAYS_DEFAULT) -> Optional[pd.DataFrame]:
    try:
        end = datetime.now()
        start = end - timedelta(days=days * 1.6)  # pad to include at least 90 trading days
        data = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=True)
        if data is None or data.empty:
            return None
        data = data.sort_index()
        if len(data) > days:
            data = data.iloc[-days:].copy()
        need = {"Open","Close","High","Low","Volume"}
        if not need.issubset(set(data.columns)):
            return None
        data = data.dropna(subset=["Open","Close","High","Low"])
        return data
    except Exception:
        return None

# ----------------------------
# Utility: price selection per day
# ----------------------------
def idx_to_date(df: pd.DataFrame, idx: int) -> str:
    idx = max(0, min(idx, len(df) - 1))
    return pd.to_datetime(df.index[idx]).date().isoformat()

def oc_peak_price(df: pd.DataFrame, i: int) -> float:
    row = df.iloc[i]
    return float(max(row["Open"], row["Close"]))

def oc_trough_price(df: pd.DataFrame, i: int) -> float:
    row = df.iloc[i]
    return float(min(row["Open"], row["Close"]))

# ----------------------------
# W detection (heuristic)
# ----------------------------
def detect_w_patterns(df: pd.DataFrame,
                      p1_ge_tol: float = 0.0,
                      prominence_factor: float = 0.5) -> List[Dict]:
    """
    Find W candidates using Close to locate pivots, but evaluate P/L with Open/Close preference.
      - P1: last peak before L1
      - P2: highest peak between L1 and L2 (neckline)
      - Tilt: P1 >= P2 * (1 - p1_ge_tol)
      - Trough similarity: |L2-L1|/max(L1,L2) <= 0.30
      - Target: per your formula
    """
    if len(df) < 20:
        return []

    close = df["Close"].values
    n = len(close)
    dist = max(3, n // 12)
    prom = np.std(close) * prominence_factor

    peaks_idx, _ = find_peaks(close, prominence=prom, distance=dist)
    troughs_idx, _ = find_peaks(-close, prominence=prom, distance=dist)
    if len(troughs_idx) < 2 or len(peaks_idx) < 1:
        return []

    candidates = []
    for i in range(len(troughs_idx) - 1):
        l1 = troughs_idx[i]
        l2 = troughs_idx[i + 1]
        if l2 - l1 < 3 or l2 - l1 > 85:
            continue

        peaks_between = [p for p in peaks_idx if l1 < p < l2]
        if not peaks_between:
            continue
        p2 = max(peaks_between, key=lambda x: oc_peak_price(df, x))

        peaks_before = [p for p in peaks_idx if p < l1]
        if not peaks_before:
            continue
        p1 = max(peaks_before)

        P1 = oc_peak_price(df, p1)
        P2 = oc_peak_price(df, p2)
        L1 = oc_trough_price(df, l1)
        L2 = oc_trough_price(df, l2)

        # tilt filter
        if P1 < P2 * (1.0 - p1_ge_tol):
            continue

        # trough similarity
        sim = abs(L2 - L1) / max(L1, L2, 1e-12)
        if sim > 0.30:
            continue

        # target formula (your spec)
        days_p1_p2 = max(1, p2 - p1)
        neckline_slope = (P2 - P1) / days_p1_p2
        trough_midpoint = (L1 + L2) / 2.0
        target_magnitude = P2 - trough_midpoint
        breakout_estimate_days = days_p1_p2
        p3_idx = min(n - 1, p2 + breakout_estimate_days)
        breakout_estimate_date = idx_to_date(df, p3_idx)
        breakout_estimate_price = neckline_slope * breakout_estimate_days + P2
        target_price = breakout_estimate_price + target_magnitude

        height = P2 - min(L1, L2)
        width = l2 - l1
        depth_sym = 1.0 - sim
        size_factor = min(1.0, (width * max(height, 1e-6)) / (n * np.std(close) + 1e-9))
        proportion = min(1.0, width / (max(height, 1e-6) * 2.0))
        proba = float(np.clip(depth_sym * 0.5 + size_factor * 0.3 + proportion * 0.2, 0, 1))

        candidates.append({
            "p1_idx": int(p1), "l1_idx": int(l1), "p2_idx": int(p2), "l2_idx": int(l2),
            "p1_date": idx_to_date(df, p1), "l1_date": idx_to_date(df, l1),
            "p2_date": idx_to_date(df, p2), "l2_date": idx_to_date(df, l2),
            "p1_price": float(P1), "l1_price": float(L1),
            "p2_price": float(P2), "l2_price": float(L2),
            "neckline_slope": float(neckline_slope),
            "trough_midpoint": float(trough_midpoint),
            "target_magnitude": float(target_magnitude),
            "breakout_estimate_days": int(breakout_estimate_days),
            "breakout_estimate_date": breakout_estimate_date,
            "breakout_estimate_price": float(breakout_estimate_price),
            "target_price": float(target_price),
            "width": int(width),
            "height": float(height),
            "depth_symmetry": float(depth_sym),
            "proba": float(proba),
        })

    candidates.sort(key=lambda x: x["proba"], reverse=True)
    return candidates

# ----------------------------
# Feature building (for model)
# ----------------------------
def build_features(df: pd.DataFrame, pat: Dict) -> Dict:
    i_p1 = int(pat["p1_idx"]); i_l1 = int(pat["l1_idx"])
    i_p2 = int(pat["p2_idx"]); i_l2 = int(pat["l2_idx"])
    P1 = float(pat["p1_price"]); P2 = float(pat["p2_price"])
    L1 = float(pat["l1_price"]); L2 = float(pat["l2_price"])

    width = i_l2 - i_l1
    height = P2 - min(L1, L2)
    depth_sym = float(pat.get("depth_symmetry", 1.0))

    now = float(df["Close"].iloc[-1])
    close_vs_neck = now - P2

    vol_mean = float(df["Volume"].iloc[max(0, i_p1-20):i_l2+1].mean())
    vol_now  = float(df["Volume"].iloc[-1])

    return {
        "dur_p1_l2": i_l2 - i_p1,
        "dur_l1_l2": i_l2 - i_l1,
        "dur_p1_l1": i_l1 - i_p1,
        "dur_l1_p2": i_p2 - i_l1,
        "dur_p2_l2": i_l2 - i_p2,
        "p1_price": P1, "p2_price": P2, "l1_price": L1, "l2_price": L2,
        "depth_symmetry": depth_sym,
        "height": height,
        "width": width,
        "neckline_slope": float(pat.get("neckline_slope", 0.0)),
        "trough_midpoint": float(pat.get("trough_midpoint", (L1+L2)/2.0)),
        "target_magnitude": float(pat.get("target_magnitude", P2 - (L1+L2)/2.0)),
        "close_now": now,
        "close_vs_neck": close_vs_neck,
        "vol_mean": vol_mean,
        "vol_now": vol_now,
    }

# ----------------------------
# Model scoring
# ----------------------------
_missing_warned = False
def score_with_model(bundle, df: pd.DataFrame, pat: Dict) -> float:
    global _missing_warned
    feats = build_features(df, pat)
    order = bundle["features"]
    missing = [k for k in order if k not in feats]
    if missing and not _missing_warned:
        print(f"⚠️ Model expects {len(missing)} unknown feature(s). "
              f"Filling 0.0: {', '.join(missing[:8])}{'…' if len(missing)>8 else ''}")
        _missing_warned = True

    X = [[feats.get(k, 0.0) for k in order]]
    m = bundle["model"]
    if hasattr(m, "predict_proba"):
        P = m.predict_proba(X)
        # if both classes present during training
        if P.shape[1] == 2:
            idx1 = int(np.where(m.classes_ == 1)[0][0])
            return float(P[0, idx1])
        # one-class model fallback
        return 1.0 if m.classes_[0] == 1 else 0.0
    # no proba available
    return float(m.predict(X)[0])


def load_model_bundle_or_none(path: str = MODEL_PATH):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception as e:
        warnings.warn(f"Could not load model from {path}: {e}")
    return None

# ----------------------------
# Plotting
# ----------------------------
def plot_pattern(df: pd.DataFrame, pat: Dict, symbol: str, out_path: Optional[str] = None, title_suffix: str = ""):
    dates = pd.to_datetime(df.index)
    close = df["Close"].values
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, close, label="Close", lw=1.5)

    for key, color, marker in [("p1_idx","tab:orange","^"),
                               ("l1_idx","tab:blue","v"),
                               ("p2_idx","tab:orange","^"),
                               ("l2_idx","tab:blue","v")]:
        i = int(pat[key])
        ax.scatter(dates[i], close[i], color=color, marker=marker, s=80, zorder=5)

    p1 = int(pat["p1_idx"]); p2 = int(pat["p2_idx"])
    ax.plot([dates[p1], dates[p2]], [close[p1], close[p2]], "--", color="tab:red", lw=1.2, label="Neckline (P1→P2)")

    try:
        be_date = pd.to_datetime(pat["breakout_estimate_date"])
        ax.axvline(be_date, color="gray", ls=":", lw=1, label="Est. breakout date")
    except Exception:
        pass

    tgt = float(pat.get("target_price", np.nan))
    if not np.isnan(tgt):
        ax.axhline(tgt, color="tab:green", ls="--", lw=1.2, label=f"Target ≈ {tgt:.2f}")

    ax.set_title(f"{symbol} — W candidate {title_suffix}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
    else:
        plt.show()

# ----------------------------
# Label writing (robust, unified)
# ----------------------------
def write_label_row(csv_path: str, symbol: str, df: pd.DataFrame, pat: Dict, y_shape: int):
    """
    Append one labeled row. We store the feature vector + important context fields.
    """
    feat = build_features(df, pat)
    row = {
        **feat,
        "symbol": symbol,
        "p1_idx": pat["p1_idx"], "l1_idx": pat["l1_idx"],
        "p2_idx": pat["p2_idx"], "l2_idx": pat["l2_idx"],
        "p1_date": pat["p1_date"], "l1_date": pat["l1_date"],
        "p2_date": pat["p2_date"], "l2_date": pat["l2_date"],
        "p1_price": pat["p1_price"], "l1_price": pat["l1_price"],
        "p2_price": pat["p2_price"], "l2_price": pat["l2_price"],
        "neckline_slope": pat.get("neckline_slope", 0.0),
        "trough_midpoint": pat.get("trough_midpoint", 0.0),
        "target_magnitude": pat.get("target_magnitude", 0.0),
        "breakout_estimate_days": pat.get("breakout_estimate_days", 0),
        "breakout_estimate_date": pat.get("breakout_estimate_date", ""),
        "breakout_estimate_price": pat.get("breakout_estimate_price", 0.0),
        "target_price": pat.get("target_price", 0.0),
        "y_shape": int(y_shape),
    }
    df_row = pd.DataFrame([row])
    # safe append
    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)

# ----------------------------
# Training
# ----------------------------
def train_and_save_model(label_csv: str, out_path: str = MODEL_PATH) -> dict:
    # robust read
    df = pd.read_csv(label_csv, engine="python", on_bad_lines="skip")

    # normalize target column
    if "y_shape" not in df.columns and "y" in df.columns:
        df["y_shape"] = df["y"].map(lambda v: 1 if str(v).strip().lower() in {"1","y","yes","true"} else 0)

    if "y_shape" not in df.columns:
        raise ValueError("labels CSV must contain 'y_shape' (0/1).")

    df = df[df["y_shape"].isin([0,1])].copy()
    if len(df) < 5 or df["y_shape"].nunique() < 2:
        raise ValueError("Not enough labeled data or only one class present.")

    # features → numeric
    X = df[[c for c in FEATURE_COLUMNS if c in df.columns]].copy()
    for c in FEATURE_COLUMNS:
        if c not in X.columns:
            X[c] = 0.0
    X = X[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)
    y = df["y_shape"].astype(int).values

    # fit final model on all data
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)

    # choose safe CV folds: ≤ minority class count (and at least 2, at most 5)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    minority = max(1, min(n_pos, n_neg))
    n_splits = max(2, min(5, minority))

    aucs = []
    if n_pos == 0 or n_neg == 0:
        aucs = [float("nan")]
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for tr, va in skf.split(X, y):
            clf_cv = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
            clf_cv.fit(X.iloc[tr], y[tr])
            P = clf_cv.predict_proba(X.iloc[va])
            if P.shape[1] == 2:
                idx1 = int(np.where(clf_cv.classes_ == 1)[0][0])
                proba = P[:, idx1]
            else:
                # one-class fold: constant proba
                proba = np.ones(len(va)) if clf_cv.classes_[0] == 1 else np.zeros(len(va))
            aucs.append(roc_auc_score(y[va], proba))

    bundle = {
        "model": clf,
        "features": FEATURE_COLUMNS,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "n_rows": int(len(df)),
    }
    joblib.dump(bundle, out_path)
    return {
        "saved_to": out_path,
        "n_rows": int(len(df)),
        "n_features": len(FEATURE_COLUMNS),
        "cv_auc_mean": (float(np.nanmean(aucs)) if len(aucs) else float("nan")),
        "cv_auc_std": (float(np.nanstd(aucs)) if len(aucs) else float("nan")),
    }

def load_model_bundle_or_none(path: str = MODEL_PATH):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception as e:
        warnings.warn(f"Could not load model from {path}: {e}")
    return None

# ----------------------------
# Menu options
# ----------------------------
def option_train_more(p1_ge_tol: float = 0.05, min_proba: float = 0.55, n_ask: int = 5):
    """
    Suggest ≈5 candidates; label yes/no/skip; auto-retrain to model_shape.pkl
    """
    print("\n[train] Proposing candidates to label…")
    symbols = load_sp500_symbols()
    random.shuffle(symbols)
    proposed = []

    for sym in symbols:
        df = fetch_history(sym, LOOKBACK_DAYS_DEFAULT)
        if df is None:
            continue
        pats = detect_w_patterns(df, p1_ge_tol=p1_ge_tol)
        pats = [p for p in pats if p["proba"] >= min_proba]
        if not pats:
            continue
        proposed.append((sym, df, pats[0]))
        if len(proposed) >= n_ask:
            break

    if not proposed:
        print("No candidates met the threshold. Try lowering min_proba or changing tilt tolerance.")
        return

    print(f"\n[label] Saving to: {LABEL_CSV}")
    print("Type 'y' if this is a W (by your rule), 'n' if not, 's' to skip.\n")

    for k, (sym, df, pat) in enumerate(proposed, 1):
        print(f"[{k}/{len(proposed)}] {sym}  "
              f"P1={pat['p1_date']} L1={pat['l1_date']}  "
              f"P2={pat['p2_date']} L2={pat['l2_date']}  "
              f"Target≈${pat['target_price']:.2f}  EstBreakout={pat['breakout_estimate_date']}  "
              f"HeurP={pat['proba']:.2f}")
        chart_path = os.path.join(CHART_DIR, f"label_{sym}_{pat['l2_date']}.png")
        plot_pattern(df, pat, sym, out_path=chart_path, title_suffix="(labeling)")
        print(f"  Chart saved → {chart_path}")
        ans = input("  Is this a W? [y/n/s]: ").strip().lower()
        if ans == "y":
            write_label_row(LABEL_CSV, sym, df, pat, 1)
            print("  → saved y_shape=1")
        elif ans == "n":
            write_label_row(LABEL_CSV, sym, df, pat, 0)
            print("  → saved y_shape=0")
        else:
            print("  → skipped")

    # auto-retrain
    print("\n[train] Retraining model on all accumulated labels…")
    try:
        summary = train_and_save_model(LABEL_CSV, MODEL_PATH)
        print("[train] Model updated ✅")
        print(f"  Saved to       : {summary['saved_to']}")
        print(f"  Labeled rows   : {summary['n_rows']}")
        print(f"  Features       : {summary['n_features']}")
        print(f"  5-fold AUC     : {summary['cv_auc_mean']:.3f} ± {summary['cv_auc_std']:.3f}")
    except Exception as e:
        print(f"[train] Skipped model update: {e}")

def option_check_one():
    sym = input("Ticker (e.g., COST): ").strip().upper()
    tol = float(input("Tilt tolerance (P1≥P2) e.g. 0.02 [default 0.05]: ") or "0.05")
    thr = float(input("Min heuristic probability (0-1) [default 0.6]: ") or "0.6")

    df = fetch_history(_normalize_symbol(sym), LOOKBACK_DAYS_DEFAULT)
    if df is None or df.empty:
        print("No data.")
        return
    pats = detect_w_patterns(df, p1_ge_tol=tol)
    if not pats:
        print("No W candidates found.")
        return
    kept = [p for p in pats if p["proba"] >= thr] or pats[:1]
    for i, p in enumerate(kept, 1):
        chart_path = os.path.join(CHART_DIR, f"{sym}_single_{i}.png")
        plot_pattern(df, p, sym, out_path=chart_path, title_suffix=f"(#{i})")
        print(f"Saved plot → {chart_path}")
        print(f"  P(W)={p['proba']:.2f}  P1={p['p1_date']} L1={p['l1_date']}  "
              f"P2={p['p2_date']} L2={p['l2_date']}  Target≈${p['target_price']:.2f}  "
              f"EstBreakout={p['breakout_estimate_date']}")

def scan_universe(symbols: List[str], min_proba: float, p1_ge_tol: float,
                  use_model: bool, bundle, plot_top: int = 25, tag: str = "scan"):
    hits = []
    for sym in symbols:
        df = fetch_history(sym, LOOKBACK_DAYS_DEFAULT)
        if df is None:
            continue
        pats = detect_w_patterns(df, p1_ge_tol=p1_ge_tol)
        if not pats:
            continue
        for pat in pats:
            proba = score_with_model(bundle, df, pat) if use_model else float(pat["proba"])
            if proba >= min_proba:
                hit = {**pat, "symbol": sym, "proba": float(proba),
                       "current": float(df["Close"].iloc[-1])}
                hits.append((sym, df, hit))

    if not hits:
        print("No candidates met threshold.")
        return

    hits.sort(key=lambda t: t[2]["proba"], reverse=True)
    top = hits[:max(1, plot_top)]
    for k, (sym, df, pat) in enumerate(top, 1):
        out = os.path.join(CHART_DIR, f"{tag}_{sym}_{k}.png")
        plot_pattern(df, pat, sym, out_path=out, title_suffix=f"(rank {k}, P={pat['proba']:.2f})")
        print(f"[{k}] {sym:6}  P={pat['proba']:.2f}  "
              f"P1={pat['p1_date']} L1={pat['l1_date']}  "
              f"P2={pat['p2_date']} L2={pat['l2_date']}  "
              f"Target≈${pat['target_price']:.2f}  EstBreakout={pat['breakout_estimate_date']}  → {out}")

def option_scan_sp500():
    tol = float(input("Tilt tolerance (P1≥P2) e.g. 0.02 [default 0.05]: ") or "0.05")
    thr = float(input("Min probability/confidence (0-1) [default 0.6]: ") or "0.6")
    symbols = load_sp500_symbols()
    bundle = load_model_bundle_or_none(MODEL_PATH)
    use_model = bundle is not None
    if use_model:
        print(f"🔮 Using trained model: {MODEL_PATH} (features={len(bundle['features'])}, rows={bundle.get('n_rows','?')})")
    else:
        print("ℹ️ No trained model found. Using heuristics.")
    print(f"\nScanning {len(symbols)} symbols in S&P 500…")
    scan_universe(symbols, min_proba=thr, p1_ge_tol=tol, use_model=use_model, bundle=bundle,
                  plot_top=25, tag="sp500")

def option_scan_nasdaq100():
    tol = float(input("Tilt tolerance (P1≥P2) e.g. 0.02 [default 0.05]: ") or "0.05")
    thr = float(input("Min probability/confidence (0-1) [default 0.6]: ") or "0.6")
    symbols = load_nasdaq100_symbols()
    bundle = load_model_bundle_or_none(MODEL_PATH)
    use_model = bundle is not None
    if use_model:
        print(f"🔮 Using trained model: {MODEL_PATH} (features={len(bundle['features'])}, rows={bundle.get('n_rows','?')})")
    else:
        print("ℹ️ No trained model found. Using heuristics.")
    print(f"\nScanning {len(symbols)} symbols in NASDAQ-100…")
    scan_universe(symbols, min_proba=thr, p1_ge_tol=tol, use_model=use_model, bundle=bundle,
                  plot_top=25, tag="nas100")

# ----------------------------
# Menu loop
# ----------------------------
def menu_loop():
    while True:
        print("\nW-Pattern Labeling + Scanning")
        print("============================================================\n")
        print("Options:")
        print("1) Train more data (≈5 candidates to label)")
        print("2) Check one ticker (90d) and plot")
        print("3) Scan S&P 500 (targets via your formula)")
        print("4) Scan Nasdaq-100 (targets via your formula)")
        print("q) Exit\n")
        sel = input("Select: ").strip().lower()
        if sel == "1":
            try:
                tol = float(input("Tilt tolerance (P1≥P2) e.g. 0.02 [default 0.05]: ") or "0.05")
                thr = float(input("Min heuristic probability (0-1) [default 0.55]: ") or "0.55")
                option_train_more(p1_ge_tol=tol, min_proba=thr, n_ask=5)
            except Exception as e:
                print(f"⚠️ Error: {e}")
            input("\nPress Enter to return to the menu…")
        elif sel == "2":
            try:
                option_check_one()
            except Exception as e:
                print(f"⚠️ Error: {e}")
            input("\nPress Enter to return to the menu…")
        elif sel == "3":
            try:
                option_scan_sp500()
            except Exception as e:
                print(f"⚠️ Error: {e}")
            input("\nPress Enter to return to the menu…")
        elif sel == "4":
            try:
                option_scan_nasdaq100()
            except Exception as e:
                print(f"⚠️ Error: {e}")
            input("\nPress Enter to return to the menu…")
        elif sel == "q":
            print("👋 Goodbye!")
            break
        else:
            print("Please choose 1/2/3/4 or q.")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    menu_loop()
