#!/usr/bin/env python3
"""
w_ml.py — Interactive ML loop for W-shaped pattern detection
- Candidate generation from Yahoo Finance (last 90 trading days)
- Interactive labeling (Yes/No) with plots
- Feature extraction (price structure + indicators)
- Model training (Gradient Boosting by default)
- Scoring S&P 500 universe

Install:
  pip install yfinance pandas numpy scipy scikit-learn matplotlib requests

Usage examples:
  # Step 1: propose candidates and label them (press y/n/s)
  python w_ml.py propose --limit 120 --per-ticker 2 --plot

  # Step 2: train a model from your labels
  python w_ml.py train --csv w_labels.csv --save model_shape.pkl

  # Step 3: scan all S&P 500 with the trained model
  python w_ml.py scan --model model_shape.pkl --min-proba 0.6 --top 25 --plot-top 5
"""

import os, io, json, math, argparse, warnings, joblib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.signal import find_peaks

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------- Ticker universe loader (cache to sp500.txt) ----------

def load_sp500_symbols() -> List[str]:
    def _norm(s: str) -> str:
        return s.strip().upper().replace('.', '-')
    cache = "sp500.txt"
    if os.path.exists(cache):
        with open(cache, "r") as f:
            syms = [_norm(x) for x in f if x.strip()]
        # de-dupe preserve order
        seen, out = set(), []
        for s in syms:
            if s and s not in seen:
                seen.add(s); out.append(s)
        if len(out) >= 480:
            print(f"📊 Loaded {len(out)} tickers from sp500.txt")
            return out
        else:
            print(f"⚠️ sp500.txt only had {len(out)} symbols. Rebuilding from Wikipedia…")
    # scrape wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    syms = [_norm(s) for s in df["Symbol"].astype(str).tolist() if s.strip()]
    seen, out = set(), []
    for s in syms:
        if s and s not in seen:
            seen.add(s); out.append(s)
    if len(out) < 480:
        raise RuntimeError("Wikipedia returned too few S&P 500 tickers.")
    with open(cache, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"📊 Loaded {len(out)} from Wikipedia and cached to sp500.txt")
    return out

# ---------- Indicators ----------

def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    s = np.asarray(series, float)
    delta = np.diff(s, prepend=s[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    # Wilder's smoothing
    def smooth(x):
        out = np.zeros_like(x)
        out[0] = x[1:period+1].mean() if len(x) > period else x.mean()
        alpha = 1.0/period
        for i in range(1, len(x)):
            out[i] = alpha*x[i] + (1-alpha)*out[i-1]
        return out
    roll_up = smooth(up)
    roll_dn = smooth(dn) + 1e-12
    rs = roll_up / roll_dn
    return 100.0 - (100.0 / (1.0 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    prev_close = np.roll(c, 1); prev_close[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    # Wilder's ATR
    out = np.zeros_like(tr)
    out[:period] = tr[:period].mean() if len(tr) >= period else tr.mean()
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        out[i] = alpha * tr[i] + (1 - alpha) * out[i-1]
    return out

def sma(series: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(series)
    return s.rolling(n, min_periods=1).mean().values

def vol_zscore(vol: np.ndarray, n: int = 20) -> np.ndarray:
    v = pd.Series(vol)
    mean = v.rolling(n, min_periods=1).mean()
    std = v.rolling(n, min_periods=1).std(ddof=0).replace(0, np.nan)
    z = (v - mean) / std
    return z.fillna(0).values

# ---------- Yahoo fetch ----------

def fetch_history(symbol: str, lookback_days: int = 90) -> Optional[pd.DataFrame]:
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    try:
        df = yf.Ticker(symbol).history(start=start, end=end, interval="1d")
        if isinstance(df, pd.DataFrame) and len(df) >= 30:
            return df
    except Exception:
        pass
    return None

# ---------- Candidate generation (lenient; then ML will learn) ----------

def find_w_candidates_from_ohlc(df: pd.DataFrame,
                                min_conf: float = 0.30,
                                p1_ge_tol: float = 0.00,
                                max_per_symbol: int = 5) -> List[Dict]:
    """
    Uses Open/Close extremes per your rule:
      peaks from oc_high = max(Open, Close)
      troughs from oc_low = min(Open, Close)
    Returns lenient candidates; ML will separate good/bad later.
    """
    opens = df['Open'].values.astype(float)
    closes = df['Close'].values.astype(float)
    highs = df['High'].values.astype(float)
    lows  = df['Low'].values.astype(float)
    oc_high = np.maximum(opens, closes)
    oc_low  = np.minimum(opens, closes)
    n = len(oc_high)
    if n < 30:
        return []

    # extrema params (lenient)
    prom_low  = max(1e-9, 0.35 * np.std(oc_low))
    prom_high = max(1e-9, 0.35 * np.std(oc_high))
    dist_val  = max(1, n // 12)  # allow closer valleys
    dist_peak = max(1, n // 14)

    valleys_idx, _ = find_peaks(-oc_low,  prominence=prom_low,  distance=dist_val)
    peaks_idx,   _ = find_peaks( oc_high, prominence=prom_high, distance=dist_peak)

    x = np.arange(n)
    valleys = [(int(x[i]), float(oc_low[i]), int(i)) for i in valleys_idx]
    peaks   = [(int(x[i]), float(oc_high[i]), int(i)) for i in peaks_idx]

    out = []
    for i in range(len(valleys) - 1):
        for j in range(i + 1, len(valleys)):
            x1, y1, idx1 = valleys[i]
            x2, y2, idx2 = valleys[j]
            series_len = n
            # checks
            height_tol = 0.35  # allow 35% diff for leniency
            min_sep = series_len * 0.06  # ~5–6 bars on 90-day window
            height_diff = abs(y1 - y2) / max(y1, y2, 1e-12)
            if height_diff > height_tol or abs(x2 - x1) < min_sep:
                continue
            # peaks between → P2
            peaks_between = [p for p in peaks if x1 < p[0] < x2]
            if not peaks_between:
                continue
            # last peak before L1 → P1
            peaks_before = [p for p in peaks if p[0] < x1]
            if not peaks_before:
                continue
            left_peak  = max(peaks_before, key=lambda p: p[0])  # by position
            right_peak = max(peaks_between, key=lambda p: p[1])  # highest

            # tilt: P1 >= P2*(1 - tol)
            if left_peak[1] < right_peak[1] * (1.0 - p1_ge_tol):
                continue

            # neckline above both valleys
            neck_y = right_peak[1]
            if neck_y <= max(y1, y2):
                continue

            # simple confidence (same recipe)
            pattern_width = abs(x2 - x1)
            pattern_height = neck_y - min(y1, y2)
            depth_sym = 1.0 - height_diff
            conf = 0.5 * depth_sym + 0.3 * min(1.0, (pattern_width * pattern_height) / (series_len * 10.0)) \
                   + 0.2 * min(1.0, pattern_width / max(pattern_height * 2.0, 1e-12))
            if conf < min_conf:
                continue

            out.append({
                'left_valley':  {'x': x1, 'y': y1, 'index': idx1},
                'right_valley': {'x': x2, 'y': y2, 'index': idx2},
                'left_peak':    {'x': left_peak[0],  'y': left_peak[1],  'index': left_peak[2]},
                'right_peak':   {'x': right_peak[0], 'y': right_peak[1], 'index': right_peak[2]},
                'neckline':     {'x': right_peak[0], 'y': right_peak[1], 'index': right_peak[2]},
                'confidence': float(conf),
                'depth_symmetry': float(depth_sym)
            })

    # keep top by confidence
    out = sorted(out, key=lambda d: d['confidence'], reverse=True)[:max_per_symbol]
    return out

# ---------- Feature engineering ----------

FEATURE_COLUMNS = [
    # structure (prices, deltas, ratios)
    "p1_price","l1_price","p2_price","l2_price",
    "p1_l1_drop_pct","l1_p2_rise_pct","p2_l2_drop_pct",
    "trough_similarity","p1_ge_p2", "neckline_slope_per_bar",
    "bars_p1_p2","bars_l1_l2","bars_span_total",
    # indicators at L2
    "rsi14_l2","atr14_l2","close_sma20_ratio_l2","close_sma50_ratio_l2",
    "vol_z20_l2",
    # distances
    "nearline_gap_pct_l2",
    # misc scale
    "range_pct_window"
]

def build_features(df: pd.DataFrame, pat: Dict) -> Dict:
    """Compute a consistent feature dict for one candidate pattern."""
    opens  = df['Open'].values.astype(float)
    closes = df['Close'].values.astype(float)
    highs  = df['High'].values.astype(float)
    lows   = df['Low'].values.astype(float)
    oc_high = np.maximum(opens, closes)
    oc_low  = np.minimum(opens, closes)
    close   = df['Close'].values.astype(float)
    vol     = df['Volume'].values.astype(float) if 'Volume' in df.columns else np.zeros_like(close)

    idx_p1 = pat['left_peak']['index']
    idx_l1 = pat['left_valley']['index']
    idx_p2 = pat['right_peak']['index']
    idx_l2 = pat['right_valley']['index']

    # prices per your rule
    p1_price = oc_high[idx_p1]
    p2_price = oc_high[idx_p2]
    l1_price = oc_low[idx_l1]
    l2_price = oc_low[idx_l2]

    # basic spans
    bars_p1_p2 = max(1, idx_p2 - idx_p1)
    bars_l1_l2 = max(1, idx_l2 - idx_l1)
    bars_total = max(1, idx_l2 - idx_p1)

    # changes
    p1_l1_drop_pct = (l1_price - p1_price) / p1_price
    l1_p2_rise_pct = (p2_price - l1_price) / max(l1_price, 1e-12)
    p2_l2_drop_pct = (l2_price - p2_price) / max(p2_price, 1e-12)

    trough_similarity = 1.0 - (abs(l2_price - l1_price) / max(l1_price, l2_price, 1e-12))
    p1_ge_p2 = 1.0 if p1_price >= p2_price else 0.0

    # neckline slope (bars)
    neckline_slope = (p2_price - p1_price) / bars_p1_p2

    # indicators
    rsi14 = rsi(close, 14)
    atr14 = atr(df, 14)
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    vz20  = vol_zscore(vol, 20)

    # at L2
    rsi14_l2 = rsi14[idx_l2] if idx_l2 < len(rsi14) else np.nan
    atr14_l2 = atr14[idx_l2] if idx_l2 < len(atr14) else np.nan
    c_l2 = close[idx_l2]
    sma20_l2 = sma20[idx_l2] if idx_l2 < len(sma20) else np.nan
    sma50_l2 = sma50[idx_l2] if idx_l2 < len(sma50) else np.nan
    close_sma20_ratio = c_l2 / max(sma20_l2, 1e-12)
    close_sma50_ratio = c_l2 / max(sma50_l2, 1e-12)
    vol_z20_l2 = vz20[idx_l2] if idx_l2 < len(vz20) else 0.0

    # nearline distance at L2 (how far below neckline)
    # line through P1→P2, evaluate at x = idx_l2
    y_at_l2 = p1_price + neckline_slope * (idx_l2 - idx_p1)
    nearline_gap_pct = (y_at_l2 - c_l2) / max(c_l2, 1e-12)

    # window scale
    window_range_pct = (np.nanmax(highs) - np.nanmin(lows)) / max(np.nanmin(lows), 1e-12)

    feats = {
        "p1_price": p1_price, "l1_price": l1_price, "p2_price": p2_price, "l2_price": l2_price,
        "p1_l1_drop_pct": p1_l1_drop_pct, "l1_p2_rise_pct": l1_p2_rise_pct, "p2_l2_drop_pct": p2_l2_drop_pct,
        "trough_similarity": trough_similarity, "p1_ge_p2": p1_ge_p2,
        "neckline_slope_per_bar": neckline_slope,
        "bars_p1_p2": bars_p1_p2, "bars_l1_l2": bars_l1_l2, "bars_span_total": bars_total,
        "rsi14_l2": rsi14_l2, "atr14_l2": atr14_l2,
        "close_sma20_ratio_l2": close_sma20_ratio, "close_sma50_ratio_l2": close_sma50_ratio,
        "vol_z20_l2": vol_z20_l2,
        "nearline_gap_pct_l2": nearline_gap_pct,
        "range_pct_window": window_range_pct
    }
    return feats

# ---------- Target math (for display) ----------

def compute_target_math(df: pd.DataFrame, pat: Dict) -> Dict:
    opens, closes = df['Open'].values, df['Close'].values
    oc_high = np.maximum(opens, closes)
    oc_low  = np.minimum(opens, closes)
    dates = df.index

    p1 = pat['left_peak']['index']
    p2 = pat['right_peak']['index']
    l1 = pat['left_valley']['index']
    l2 = pat['right_valley']['index']

    p1_price = oc_high[p1]; p2_price = oc_high[p2]
    l1_price = oc_low[l1];  l2_price = oc_low[l2]

    trough_midpoint = (l1_price + l2_price) / 2.0
    target_magnitude = p2_price - trough_midpoint
    bars_between = max(1, p2 - p1)
    neckline_slope = (p2_price - p1_price) / bars_between
    breakout_est_price = p2_price + neckline_slope * bars_between
    target_price = breakout_est_price + target_magnitude

    if p2 + bars_between < len(dates):
        breakout_est_date = str(dates[p2 + bars_between].date())
    else:
        breakout_est_date = str(dates[-1].date())

    return {
        "trough_midpoint": float(trough_midpoint),
        "target_magnitude": float(target_magnitude),
        "neckline_slope": float(neckline_slope),
        "breakout_estimate_price": float(breakout_est_price),
        "breakout_estimate_date": breakout_est_date,
        "target_price": float(target_price)
    }

# ---------- Plotting ----------

def plot_candidate(symbol: str, df: pd.DataFrame, pat: Dict, target_info: Dict):
    dates = df.index
    close = df['Close'].values
    opens = df['Open'].values
    oc_high = np.maximum(opens, close)
    oc_low  = np.minimum(opens, close)

    p1 = pat['left_peak']['index']; p2 = pat['right_peak']['index']
    l1 = pat['left_valley']['index']; l2 = pat['right_valley']['index']

    plt.figure(figsize=(12,6))
    plt.plot(dates, close, '-', alpha=0.8, label='Close')

    # draw neckline line across window using P1->P2 slope
    x0, y0 = p1, oc_high[p1]
    slope = (oc_high[p2] - oc_high[p1]) / max(1, p2 - p1)
    xs = np.arange(len(dates))
    neck = y0 + slope*(xs - x0)
    plt.plot(dates, neck, '--', alpha=0.7, label='Neckline (P1→P2)')

    # marks
    plt.scatter(dates[p1], oc_high[p1], c='tab:orange', s=90, zorder=3, label='P1 (peak)')
    plt.scatter(dates[l1], oc_low[l1],  c='tab:red',    s=90, zorder=3, label='L1 (trough)')
    plt.scatter(dates[p2], oc_high[p2], c='tab:orange', s=90, zorder=3, label='P2 (peak)')
    plt.scatter(dates[l2], oc_low[l2],  c='tab:red',    s=90, zorder=3, label='L2 (trough)')

    # target price horizontal
    plt.axhline(target_info["target_price"], color='tab:green', linestyle=':', alpha=0.8,
                label=f"Target ${target_info['target_price']:.2f}")

    plt.title(f"{symbol} — W candidate (conf={pat['confidence']:.0%})")
    plt.legend(loc='best')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

# ---------- Label store ----------

LABEL_CSV = "w_labels.csv"

def append_label_row(row: Dict, csv_path: str = LABEL_CSV):
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

# ---------- CLI Commands ----------

def cmd_propose(args):
    symbols = load_sp500_symbols()
    if args.limit and args.limit > 0:
        symbols = symbols[:args.limit]

    labeled = 0
    for i, sym in enumerate(symbols, 1):
        df = fetch_history(sym, lookback_days=args.lookback)   # ✅ correct keyword
        if df is None:
            continue

        pats = find_w_candidates_from_ohlc(df,
                                           min_conf=args.min_conf,
                                           p1_ge_tol=args.p1_ge_tol,
                                           max_per_symbol=args.per_ticker)
        if not pats:
            continue

        for pat in pats:
            # compute features & target math
            feats = build_features(df, pat)
            tgt = compute_target_math(df, pat)

            # print summary
            p1 = pat['left_peak']['index']; p2 = pat['right_peak']['index']
            l1 = pat['left_valley']['index']; l2 = pat['right_valley']['index']
            row = {
                "ticker": sym,
                "p1_date": str(df.index[p1].date()),
                "p1_price": feats["p1_price"],
                "l1_date": str(df.index[l1].date()),
                "l1_price": feats["l1_price"],
                "p2_date": str(df.index[p2].date()),
                "p2_price": feats["p2_price"],
                "l2_date": str(df.index[l2].date()),
                "l2_price": feats["l2_price"],
                "neckline_slope": feats["neckline_slope_per_bar"],
                # learning target (your answer)
                "y_shape": None
            }
            # merge features for training later
            for k in FEATURE_COLUMNS:
                row[k] = feats.get(k, np.nan)

            if args.plot:
                try:
                    plot_candidate(sym, df, pat, tgt)
                except Exception:
                    pass

            # interactive label
            print(f"\n[{i}/{len(symbols)}] {sym} candidate (conf={pat['confidence']:.0%})")
            print(f"  P1: {row['p1_date']} ${row['p1_price']:.2f} | L1: {row['l1_date']} ${row['l1_price']:.2f}")
            print(f"  P2: {row['p2_date']} ${row['p2_price']:.2f} | L2: {row['l2_date']} ${row['l2_price']:.2f}")
            ans = input("Is this a (forming or complete) W?  [y]es / [n]o / [s]kip: ").strip().lower()
            if ans in ('y','yes'):
                row["y_shape"] = 1
                append_label_row(row, csv_path=args.out)
                labeled += 1
                print("  ✓ saved with y_shape=1")
            elif ans in ('n','no'):
                row["y_shape"] = 0
                append_label_row(row, csv_path=args.out)
                labeled += 1
                print("  ✓ saved with y_shape=0")
            else:
                print("  ↷ skipped")

    print(f"\nDone. Labeled rows appended: {labeled}")
    print(f"Label file: {args.out}")


def _load_training_table(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No label CSV found at {csv_path}")
    df = pd.read_csv(csv_path)
    # keep only rows with y_shape in {0,1}
    df = df[df['y_shape'].isin([0,1])]
    # drop obviously non-feature columns if present (but keep dates/prices if you want)
    return df

def _Xy_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X = df[FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df['y_shape'].values.astype(int)
    return X.values.astype(float), y, FEATURE_COLUMNS

def cmd_train(args):
    df = _load_training_table(args.csv)
    if len(df) < 10:
        raise ValueError("Need at least 10 labeled rows to train something meaningful.")
    X, y, feat_names = _Xy_from_df(df)

    model = GradientBoostingClassifier(random_state=42)
    # quick CV report
    cv = StratifiedKFold(n_splits=min(5, sum(y==1), sum(y==0), 5), shuffle=True, random_state=42)
    aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"[train] {len(df)} rows | CV AUC: {aucs.mean():.3f} ± {aucs.std():.3f}")
    model.fit(X, y)
    joblib.dump({"model": model, "features": feat_names}, args.save)
    print(f"[train] saved → {args.save}")

    # simple feature importances (if supported)
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_})
        print("\nTop features:")
        print(fi.sort_values("importance", ascending=False).head(10).to_string(index=False))

def score_symbol_with_model(sym: str, model_bundle: Dict, lookback: int = 90,
                            min_conf: float = 0.3, p1_ge_tol: float = 0.0) -> List[Dict]:
    df = fetch_history(sym, lookback)
    if df is None:
        return []
    pats = find_w_candidates_from_ohlc(df, min_conf=min_conf, p1_ge_tol=p1_ge_tol, max_per_symbol=5)
    if not pats:
        return []
    out = []
    for pat in pats:
        feats = build_features(df, pat)
        X = np.array([[feats.get(k, 0.0) for k in model_bundle["features"]]], dtype=float)
        proba = model_bundle["model"].predict_proba(X)[0,1]
        tgt = compute_target_math(df, pat)
        out.append({
            "ticker": sym,
            "proba": float(proba),
            "confidence": float(pat["confidence"]),
            "p1_date": str(df.index[pat['left_peak']['index']].date()),
            "l1_date": str(df.index[pat['left_valley']['index']].date()),
            "p2_date": str(df.index[pat['right_peak']['index']].date()),
            "l2_date": str(df.index[pat['right_valley']['index']].date()),
            "target_price": tgt["target_price"],
            "breakout_estimate_date": tgt["breakout_estimate_date"]
        })
    return out

def cmd_scan(args):
    # load model
    bundle = joblib.load(args.model)
    symbols = load_sp500_symbols()
    if args.limit and args.limit > 0:
        symbols = symbols[:args.limit]

    hits = []
    for i, sym in enumerate(symbols, 1):
        print(f"scanning {i}/{len(symbols)}: {sym}...", end=' ')
        scored = score_symbol_with_model(sym, bundle, lookback=args.lookback,
                                         min_conf=args.min_conf, p1_ge_tol=args.p1_ge_tol)
        if scored:
            cnt = sum(1 for r in scored if r["proba"] >= args.min_proba)
            print(f"found {cnt} ≥ threshold")
            hits.extend([r for r in scored if r["proba"] >= args.min_proba])
        else:
            print("no candidates")

    if not hits:
        print("\nNo hits ≥ threshold. Try lowering --min-proba or increasing lookback.")
        return

    hits = sorted(hits, key=lambda d: d["proba"], reverse=True)
    print(f"\nTop {min(args.top, len(hits))} hits:")
    for row in hits[:args.top]:
        print(f"  {row['ticker']:<6}  P(W)={row['proba']:.2f}  "
              f"P1={row['p1_date']}  L1={row['l1_date']}  P2={row['p2_date']}  L2={row['l2_date']}  "
              f"Target≈${row['target_price']:.2f}  EstBreakout={row['breakout_estimate_date']}")

    # Optional plotting of first k
    if args.plot_top > 0:
        for row in hits[:args.plot_top]:
            sym = row["ticker"]
            df = fetch_history(sym, lookback_days=args.lookback)   # ✅ correct keyword
            # Recompute the best pattern for plotting (same scoring path)
            pats = find_w_candidates_from_ohlc(df, min_conf=args.min_conf, p1_ge_tol=args.p1_ge_tol, max_per_symbol=5)
            if not pats:
                continue
            # pick the highest model prob among pats
            best_pat, best_prob, best_tgt = None, -1.0, None
            for pat in pats:
                feats = build_features(df, pat)
                X = np.array([[feats.get(k, 0.0) for k in bundle["features"]]], dtype=float)
                proba = bundle["model"].predict_proba(X)[0,1]
                if proba > best_prob:
                    best_prob = proba
                    best_pat = pat
            if best_pat:
                target_info = compute_target_math(df, best_pat)
                try:
                    plot_candidate(sym, df, best_pat, target_info)
                except Exception:
                    pass

# ---------- Main ----------

def build_arg_parser():
    p = argparse.ArgumentParser(description="Interactive ML pipeline for W-shaped pattern detection")
    sub = p.add_subparsers(dest="cmd", required=True)

    # propose (interactive labeling)
    pp = sub.add_parser("propose", help="Suggest candidates and interactively label y/n")
    pp.add_argument("--limit", type=int, default=100, help="Number of tickers to scan (from S&P 500)")
    pp.add_argument("--per-ticker", type=int, default=2, help="Max candidates per ticker")
    pp.add_argument("--lookback", type=int, default=90, help="Lookback days")
    pp.add_argument("--min-conf", type=float, default=0.30, help="Minimum heuristic confidence to propose")
    pp.add_argument("--p1-ge-tol", type=float, default=0.00, help="Allow P1 below P2 by this fraction (e.g. 0.02)")
    pp.add_argument("--plot", action="store_true", help="Show plot for each candidate")
    pp.add_argument("--out", default=LABEL_CSV, help="Output label CSV (append)")
    pp.set_defaults(func=cmd_propose)

    # train
    tr = sub.add_parser("train", help="Train a model from labeled CSV")
    tr.add_argument("--csv", default=LABEL_CSV, help="Input labeled CSV (with y_shape)")
    tr.add_argument("--save", default="model_shape.pkl", help="Output model path")
    tr.set_defaults(func=cmd_train)

    # scan
    sc = sub.add_parser("scan", help="Score S&P500 with trained model")
    sc.add_argument("--model", required=True, help="Trained model .pkl from 'train'")
    sc.add_argument("--limit", type=int, default=0, help="Limit tickers (0=all)")
    sc.add_argument("--lookback", type=int, default=90, help="Lookback days")
    sc.add_argument("--min-proba", type=float, default=0.6, help="Min predicted probability to show")
    sc.add_argument("--min-conf", type=float, default=0.30, help="Heuristic min confidence for candidate generation")
    sc.add_argument("--p1-ge-tol", type=float, default=0.00, help="Allow P1 below P2 by this fraction")
    sc.add_argument("--top", type=int, default=25, help="Show top N")
    sc.add_argument("--plot-top", type=int, default=0, help="Plot this many top hits")
    sc.set_defaults(func=cmd_scan)

    return p

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)
