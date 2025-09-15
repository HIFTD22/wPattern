#!/usr/bin/env python3
"""
wml1.py — Menu-driven W-shaped pattern project

Menu:
  1) Train more data (propose ~10 candidates; you label y/n/skip)
  2) Check one ticker (last 90 trading days) + plot
  3) Scan S&P 500 (targets via your formula)
  4) Scan Nasdaq-100 (targets via your formula)
  q) Exit

Notes
- Peaks = max(Open, Close); Troughs = min(Open, Close)  (your rule)
- Tilt filter: P1 >= P2 (with tolerance)
- Targets use your formula:
    trough_midpoint = (L1 + L2)/2
    target_magnitude = P2 - trough_midpoint
    breakout_estimate_days = bars between P1 and P2
    neckline_slope = (P2 - P1)/bars
    breakout_estimate_price = P2 + neckline_slope * bars
    target_price = breakout_estimate_price + target_magnitude
- If a model .pkl is provided, we show P(W). Otherwise heuristics only.

Install:
  pip install yfinance pandas numpy scipy scikit-learn matplotlib requests joblib
"""

import os, io, json, math, argparse, warnings, joblib, sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import re
from scipy.signal import find_peaks

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------- Universe loaders -------------------------------

def _norm_ticker(s: str) -> str:
    return s.strip().upper().replace('.', '-')

def load_sp500_symbols() -> List[str]:
    cache = "sp500.txt"
    if os.path.exists(cache):
        with open(cache, "r") as f:
            syms = [_norm_ticker(x) for x in f if x.strip()]
        seen, out = set(), []
        for s in syms:
            if s and s not in seen:
                seen.add(s); out.append(s)
        if len(out) >= 480:
            print(f"📊 Loaded {len(out)} S&P 500 tickers from sp500.txt")
            return out
        else:
            print(f"⚠️ sp500.txt had only {len(out)} symbols; fetching from Wikipedia…")

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    syms = [_norm_ticker(s) for s in df["Symbol"].astype(str).tolist() if s.strip()]
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

import re  # <-- add this near the top with your other imports

def load_nasdaq100_symbols() -> List[str]:
    """
    Robustly load the Nasdaq-100 universe.
    1) Prefer local nasdaq100.txt (one ticker per line)
    2) Else scrape Wikipedia and detect the 'Ticker/Symbol' column even if headers are ints
    Saves a cleaned list back to nasdaq100.txt for future runs.
    """
    cache = "nasdaq100.txt"
    # 1) Local cache first
    if os.path.exists(cache):
        with open(cache, "r") as f:
            raw = [x.strip() for x in f if x.strip()]
        seen, out = set(), []
        for s in raw:
            s = _norm_ticker(s)
            if s and s not in seen:
                seen.add(s); out.append(s)
        if 80 <= len(out) <= 200:
            print(f"📊 Loaded {len(out)} Nasdaq-100 tickers from {cache}")
            return out
        else:
            print(f"⚠️ {cache} had {len(out)} symbols; attempting Wikipedia scrape…")

    # 2) Wikipedia scrape (robust column detection)
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text), flavor="bs4")

    tickers: List[str] = []

    def looks_like_ticker_series(s: pd.Series) -> bool:
        # Many rows resemble ticker tokens like AAPL, GOOG, BRK.B, etc.
        # Avoid columns that are mostly numeric or long text.
        vals = s.astype(str).str.strip()
        if len(vals) == 0:
            return False
        # A simple heuristic: >=50% match small token pattern, and <15% contain digits
        pat = re.compile(r"^[A-Za-z]{1,5}(?:[.\-][A-Za-z]{1,3})?$")
        matches = vals.apply(lambda x: bool(pat.fullmatch(x)))
        digit_rate = vals.str.contains(r"\d").mean()
        return matches.mean() >= 0.5 and digit_rate < 0.15

    for t in tables:
        # Normalize header names to strings
        cols = [str(c).strip().lower() for c in t.columns]

        # Try explicit ticker/symbol columns first
        target_col_name = None
        for i, c in enumerate(cols):
            if "ticker" in c or "symbol" in c:
                target_col_name = t.columns[i]
                break

        series = None
        if target_col_name is not None:
            series = t[target_col_name]
        else:
            # Heuristic: pick a column that *looks* like a ticker list
            for col in t.columns:
                s = t[col]
                if looks_like_ticker_series(s):
                    series = s
                    break

        if series is None:
            continue

        # Clean and collect
        for raw in series.astype(str).tolist():
            tok = _norm_ticker(raw)
            # Filter obvious non-tickers (empty, long descriptions, caret indices, etc.)
            if not tok or len(tok) > 10 or "^" in tok or " " in tok:
                continue
            tickers.append(tok)

    # Deduplicate, keep order
    seen, out = set(), []
    for s in tickers:
        if s and s not in seen:
            seen.add(s); out.append(s)

    if len(out) < 80:
        raise RuntimeError("Could not parse a reliable Nasdaq-100 list from Wikipedia. "
                           "Create nasdaq100.txt (one ticker per line) and rerun.")

    # Cache it
    with open(cache, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"📊 Loaded {len(out)} from Wikipedia and cached to {cache}")
    return out

# ------------------------------- Indicators -------------------------------------

def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    s = np.asarray(series, float)
    delta = np.diff(s, prepend=s[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)

    # Wilder smoothing
    alpha = 1.0 / period
    roll_up = np.zeros_like(up)
    roll_dn = np.zeros_like(dn)
    roll_up[0] = up[:period+1].mean() if len(up) > period else up.mean()
    roll_dn[0] = dn[:period+1].mean() if len(dn) > period else dn.mean()
    for i in range(1, len(up)):
        roll_up[i] = alpha * up[i] + (1 - alpha) * roll_up[i-1]
        roll_dn[i] = alpha * dn[i] + (1 - alpha) * roll_dn[i-1]
    roll_dn = np.where(roll_dn == 0, 1e-12, roll_dn)
    rs = roll_up / roll_dn
    return 100.0 - (100.0 / (1.0 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    alpha = 1.0 / period
    out = np.zeros_like(tr)
    out[:period] = tr[:period].mean() if len(tr) >= period else tr.mean()
    for i in range(period, len(tr)):
        out[i] = alpha * tr[i] + (1 - alpha) * out[i-1]
    return out

def sma(series: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(series).rolling(n, min_periods=1).mean().values

def vol_zscore(vol: np.ndarray, n: int = 20) -> np.ndarray:
    v = pd.Series(vol)
    mean = v.rolling(n, min_periods=1).mean()
    std = v.rolling(n, min_periods=1).std(ddof=0).replace(0, np.nan)
    z = (v - mean) / std
    return z.fillna(0).values

# ------------------------------- Data fetch -------------------------------------

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

# ------------------------------- Candidate finder -------------------------------

def find_w_candidates_from_ohlc(df: pd.DataFrame,
                                min_conf: float = 0.30,
                                p1_ge_tol: float = 0.00,
                                max_per_symbol: int = 5) -> List[Dict]:
    """
    Uses Open/Close extremes:
      peaks from oc_high = max(Open, Close)
      troughs from oc_low = min(Open, Close)
    Returns lenient candidates; you (or the model) separate good/bad.
    """
    opens = df['Open'].values.astype(float)
    closes = df['Close'].values.astype(float)
    oc_high = np.maximum(opens, closes)
    oc_low  = np.minimum(opens, closes)
    n = len(oc_high)
    if n < 30:
        return []

    # lenient extrema params
    prom_low  = max(1e-9, 0.35 * np.std(oc_low))
    prom_high = max(1e-9, 0.35 * np.std(oc_high))
    dist_val  = max(1, n // 12)
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

            height_tol = 0.35  # 35% diff allowed
            min_sep = series_len * 0.06
            height_diff = abs(y1 - y2) / max(y1, y2, 1e-12)
            if height_diff > height_tol or abs(x2 - x1) < min_sep:
                continue

            peaks_between = [p for p in peaks if x1 < p[0] < x2]
            if not peaks_between:
                continue
            peaks_before = [p for p in peaks if p[0] < x1]
            if not peaks_before:
                continue

            left_peak  = max(peaks_before, key=lambda p: p[0])  # last before L1
            right_peak = max(peaks_between, key=lambda p: p[1])  # highest between

            # tilt: P1 >= P2*(1 - tol)
            if left_peak[1] < right_peak[1] * (1.0 - p1_ge_tol):
                continue

            neck_y = right_peak[1]
            if neck_y <= max(y1, y2):
                continue

            # confidence (symmetry + size)
            pattern_width = abs(x2 - x1)
            pattern_height = neck_y - min(y1, y2)
            depth_sym = 1.0 - height_diff
            conf = 0.5 * depth_sym \
                   + 0.3 * min(1.0, (pattern_width * pattern_height) / (series_len * 10.0)) \
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

    return sorted(out, key=lambda d: d['confidence'], reverse=True)[:max_per_symbol]

# ------------------------------- Target math ------------------------------------

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

# ------------------------------- Plotting ---------------------------------------

def plot_candidate(symbol: str, df: pd.DataFrame, pat: Dict, tgt: Dict):
    dates = df.index
    close = df['Close'].values
    opens = df['Open'].values
    oc_high = np.maximum(opens, close)
    oc_low  = np.minimum(opens, close)

    p1 = pat['left_peak']['index']; p2 = pat['right_peak']['index']
    l1 = pat['left_valley']['index']; l2 = pat['right_valley']['index']

    plt.figure(figsize=(12,6))
    plt.plot(dates, close, '-', alpha=0.85, label='Close')

    # neckline (P1->P2)
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

    # target price
    plt.axhline(tgt["target_price"], color='tab:green', linestyle=':', alpha=0.9,
                label=f"Target ${tgt['target_price']:.2f}")

    plt.title(f"{symbol} — W candidate (conf={pat['confidence']:.0%})  "
              f"Est breakout {tgt['breakout_estimate_date']}")
    plt.legend(loc='best')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

# ------------------------------- Optional model ---------------------------------

FEATURE_COLUMNS = [
    "p1_price","l1_price","p2_price","l2_price",
    "p1_l1_drop_pct","l1_p2_rise_pct","p2_l2_drop_pct",
    "trough_similarity","p1_ge_p2","neckline_slope_per_bar",
    "bars_p1_p2","bars_l1_l2","bars_span_total",
    "rsi14_l2","atr14_l2","close_sma20_ratio_l2","close_sma50_ratio_l2","vol_z20_l2",
    "nearline_gap_pct_l2","range_pct_window"
]

def build_features(df: pd.DataFrame, pat: Dict) -> Dict:
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

    p1_price = oc_high[idx_p1]
    p2_price = oc_high[idx_p2]
    l1_price = oc_low[idx_l1]
    l2_price = oc_low[idx_l2]

    bars_p1_p2 = max(1, idx_p2 - idx_p1)
    bars_l1_l2 = max(1, idx_l2 - idx_l1)
    bars_total = max(1, idx_l2 - idx_p1)

    p1_l1_drop_pct = (l1_price - p1_price) / max(p1_price, 1e-12)
    l1_p2_rise_pct = (p2_price - l1_price) / max(l1_price, 1e-12)
    p2_l2_drop_pct = (l2_price - p2_price) / max(p2_price, 1e-12)

    trough_similarity = 1.0 - (abs(l2_price - l1_price) / max(l1_price, l2_price, 1e-12))
    p1_ge_p2 = 1.0 if p1_price >= p2_price else 0.0

    neckline_slope = (p2_price - p1_price) / bars_p1_p2

    rsi14 = rsi(close, 14)
    atr14 = atr(df, 14)
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    vz20  = vol_zscore(vol, 20)

    rsi14_l2 = rsi14[idx_l2] if idx_l2 < len(rsi14) else np.nan
    atr14_l2 = atr14[idx_l2] if idx_l2 < len(atr14) else np.nan
    c_l2 = close[idx_l2]
    sma20_l2 = sma20[idx_l2] if idx_l2 < len(sma20) else np.nan
    sma50_l2 = sma50[idx_l2] if idx_l2 < len(sma50) else np.nan
    close_sma20_ratio = c_l2 / max(sma20_l2, 1e-12)
    close_sma50_ratio = c_l2 / max(sma50_l2, 1e-12)
    vol_z20_l2 = vz20[idx_l2] if idx_l2 < len(vz20) else 0.0

    y_at_l2 = p1_price + neckline_slope * (idx_l2 - idx_p1)
    nearline_gap_pct = (y_at_l2 - c_l2) / max(c_l2, 1e-12)

    window_range_pct = (np.nanmax(highs) - np.nanmin(lows)) / max(np.nanmin(lows), 1e-12)

    return {
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

# ------------------------------- Label store ------------------------------------

LABEL_CSV = "w_labels.csv"

def append_label_row(row: Dict, csv_path: str = LABEL_CSV):
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

# ------------------------------- Option 1: Propose batch ------------------------

def option_train_more(batch_labels:int=10, per_ticker:int=2, lookback:int=90,
                      min_conf:float=0.30, p1_ge_tol:float=0.00, out_path:str=LABEL_CSV,
                      universe:str="sp500"):
    symbols = load_sp500_symbols() if universe=="sp500" else load_nasdaq100_symbols()
    labeled = 0
    for i, sym in enumerate(symbols, 1):
        if labeled >= batch_labels:
            break
        df = fetch_history(sym, lookback_days=lookback)
        if df is None:
            continue
        pats = find_w_candidates_from_ohlc(df, min_conf=min_conf, p1_ge_tol=p1_ge_tol, max_per_symbol=per_ticker)
        if not pats:
            continue
        for pat in pats:
            if labeled >= batch_labels:
                break
            feats = build_features(df, pat)
            tgt = compute_target_math(df, pat)
            p1 = pat['left_peak']['index']; p2 = pat['right_peak']['index']
            l1 = pat['left_valley']['index']; l2 = pat['right_valley']['index']
            row = {
                "ticker": sym,
                "p1_date": str(df.index[p1].date()), "p1_price": feats["p1_price"],
                "l1_date": str(df.index[l1].date()), "l1_price": feats["l1_price"],
                "p2_date": str(df.index[p2].date()), "p2_price": feats["p2_price"],
                "l2_date": str(df.index[l2].date()), "l2_price": feats["l2_price"],
                "neckline_slope": feats["neckline_slope_per_bar"], "y_shape": None
            }
            for k in FEATURE_COLUMNS:
                row[k] = feats.get(k, np.nan)

            # quick plot
            try:
                plot_candidate(sym, df, pat, tgt)
            except Exception:
                pass

            print(f"\n[{i}] {sym} candidate (conf={pat['confidence']:.0%})")
            print(f"  P1 {row['p1_date']} ${row['p1_price']:.2f} | L1 {row['l1_date']} ${row['l1_price']:.2f}")
            print(f"  P2 {row['p2_date']} ${row['p2_price']:.2f} | L2 {row['l2_date']} ${row['l2_price']:.2f}")
            ans = input("Is this a W (forming/completed)? [y]es / [n]o / [s]kip: ").strip().lower()
            if ans in ('y','yes'):
                row["y_shape"] = 1; append_label_row(row, out_path); labeled += 1; print("  ✓ saved y=1")
            elif ans in ('n','no'):
                row["y_shape"] = 0; append_label_row(row, out_path); labeled += 1; print("  ✓ saved y=0")
            else:
                print("  ↷ skipped")

    print(f"\nDone. Labeled this session: {labeled}. Saved to {out_path}")

# ------------------------------- Option 2: One ticker ---------------------------

def option_check_ticker(ticker: str, lookback:int=90, min_conf:float=0.30, p1_ge_tol:float=0.00,
                        model_path:str=""):
    df = fetch_history(ticker, lookback_days=lookback)
    if df is None:
        print("❌ No data.")
        return
    pats = find_w_candidates_from_ohlc(df, min_conf=min_conf, p1_ge_tol=p1_ge_tol, max_per_symbol=3)
    if not pats:
        print("📊 No W-like candidates found with current heuristics.")
        return

    model_bundle = None
    if model_path and os.path.exists(model_path):
        try:
            model_bundle = joblib.load(model_path)
        except Exception:
            model_bundle = None

    best_pat = None
    best_score = -1.0
    best_tgt = None
    for pat in pats:
        score = pat["confidence"]
        if model_bundle:
            feats = build_features(df, pat)
            X = np.array([[feats.get(k, 0.0) for k in model_bundle["features"]]], dtype=float)
            score = model_bundle["model"].predict_proba(X)[0,1]
        if score > best_score:
            best_score = score
            best_pat = pat
            best_tgt = compute_target_math(df, pat)

    print(f"\nBest candidate for {ticker}: score={best_score:.2f} ({'P(W)' if model_bundle else 'heuristic conf'})")
    p1 = best_pat['left_peak']['index']; p2 = best_pat['right_peak']['index']
    l1 = best_pat['left_valley']['index']; l2 = best_pat['right_valley']['index']
    print(f"  P1 {str(df.index[p1].date())}  P2 {str(df.index[p2].date())}  "
          f"L1 {str(df.index[l1].date())}  L2 {str(df.index[l2].date())}")
    print(f"  Est breakout {best_tgt['breakout_estimate_date']}  Target ≈ ${best_tgt['target_price']:.2f}")

    # plot it
    try:
        plot_candidate(ticker, df, best_pat, best_tgt)
    except Exception as e:
        print(f"(plot error: {e})")

# ------------------------------- Option 3/4: Scan universes --------------------

def score_symbol(sym: str, model_bundle, lookback:int, min_conf:float, p1_ge_tol:float):
    df = fetch_history(sym, lookback_days=lookback)
    if df is None:
        return []
    pats = find_w_candidates_from_ohlc(df, min_conf=min_conf, p1_ge_tol=p1_ge_tol, max_per_symbol=3)
    if not pats:
        return []
    out = []
    for pat in pats:
        feats = build_features(df, pat)
        if model_bundle:
            X = np.array([[feats.get(k, 0.0) for k in model_bundle["features"]]], dtype=float)
            proba = float(model_bundle["model"].predict_proba(X)[0,1])
        else:
            proba = float(pat["confidence"])  # heuristic fallback
        tgt = compute_target_math(df, pat)
        out.append({
            "ticker": sym,
            "score": proba,
            "p1_date": str(df.index[pat['left_peak']['index']].date()),
            "l1_date": str(df.index[pat['left_valley']['index']].date()),
            "p2_date": str(df.index[pat['right_peak']['index']].date()),
            "l2_date": str(df.index[pat['right_valley']['index']].date()),
            "target_price": tgt["target_price"],
            "breakout_estimate_date": tgt["breakout_estimate_date"]
        })
    return out

def option_scan_universe(universe:str="sp500", lookback:int=90, min_proba:float=0.6,
                         min_conf:float=0.30, p1_ge_tol:float=0.00, top:int=25,
                         model_path:str="", plot_top:int=0):
    symbols = load_sp500_symbols() if universe=="sp500" else load_nasdaq100_symbols()
    model_bundle = None
    if model_path and os.path.exists(model_path):
        try:
            model_bundle = joblib.load(model_path)
            print(f"✔️ Using model: {model_path}")
        except Exception:
            print("⚠️ Could not load model; using heuristics only.")

    hits = []
    for i, sym in enumerate(symbols, 1):
        print(f"scanning {i}/{len(symbols)}: {sym}...", end=' ')
        scored = score_symbol(sym, model_bundle, lookback, min_conf, p1_ge_tol)
        if scored:
            cnt = sum(1 for r in scored if r["score"] >= min_proba)
            print(f"found {cnt} ≥ threshold")
            hits.extend([r for r in scored if r["score"] >= min_proba])
        else:
            print("no candidates")

    if not hits:
        print("\nNo hits ≥ threshold.")
        return

    hits = sorted(hits, key=lambda d: d["score"], reverse=True)
    print(f"\nTop {min(top, len(hits))} hits:")
    for row in hits[:top]:
        print(f"  {row['ticker']:<6}  {'P(W)' if model_bundle else 'Conf'}={row['score']:.2f}  "
              f"P1={row['p1_date']}  L1={row['l1_date']}  P2={row['p2_date']}  L2={row['l2_date']}  "
              f"Target≈${row['target_price']:.2f}  EstBreakout={row['breakout_estimate_date']}")

    # Optional plots for the first few
    if plot_top > 0:
        k = min(plot_top, len(hits))
        print(f"\nPlotting first {k}…")
        for row in hits[:k]:
            sym = row["ticker"]
            try:
                df = fetch_history(sym, lookback_days=lookback)
                pats = find_w_candidates_from_ohlc(df, min_conf=min_conf, p1_ge_tol=p1_ge_tol, max_per_symbol=3)
                # choose best for plotting by score
                best_pat, best_score = None, -1.0
                for pat in pats:
                    feats = build_features(df, pat)
                    if model_bundle:
                        X = np.array([[feats.get(k, 0.0) for k in model_bundle["features"]]], dtype=float)
                        score = model_bundle["model"].predict_proba(X)[0,1]
                    else:
                        score = pat["confidence"]
                    if score > best_score:
                        best_score = score; best_pat = pat
                if best_pat:
                    tgt = compute_target_math(df, best_pat)
                    plot_candidate(sym, df, best_pat, tgt)
            except Exception as e:
                print(f"(plot skip {sym}: {e})")

# ------------------------------- Menu loop --------------------------------------

def menu_loop():
    print("🎯 W Pattern Menu")
    print("="*60)
    while True:
        print("\nOptions:")
        print("1) Train more data (≈10 candidates to label)")
        print("2) Check one ticker (90d) and plot")
        print("3) Scan S&P 500 (targets via your formula)")
        print("4) Scan Nasdaq-100 (targets via your formula)")
        print("q) Exit")
        choice = input("\nSelect: ").strip().lower()
        if choice == '1':
            try:
                batch = input("How many to label this round? [default 10]: ").strip()
                batch = int(batch) if batch else 10
                per_ticker = input("Max candidates per ticker? [default 2]: ").strip()
                per_ticker = int(per_ticker) if per_ticker else 2
                tol = input("Tilt tolerance (P1≥P2) e.g. 0.02 for 2% [default 0.00]: ").strip()
                tol = float(tol) if tol else 0.00
                option_train_more(batch_labels=batch, per_ticker=per_ticker,
                                  p1_ge_tol=tol, out_path=LABEL_CSV, universe="sp500")
            except Exception as e:
                print(f"⚠️ {e}")
            input("\nPress Enter to return to menu…")
        elif choice == '2':
            t = input("Ticker (e.g., COST): ").strip().upper().replace('.','-')
            if not t:
                print("No ticker entered.")
                continue
            tol = input("Tilt tolerance (P1≥P2) e.g. 0.02 [default 0.00]: ").strip()
            tol = float(tol) if tol else 0.00
            model = input("Optional model path (.pkl) [Enter to skip]: ").strip()
            option_check_ticker(t, p1_ge_tol=tol, model_path=model)
            input("\nPress Enter to return to menu…")
        elif choice == '3':
            tol = input("Tilt tolerance (P1≥P2) e.g. 0.02 [default 0.00]: ").strip()
            tol = float(tol) if tol else 0.00
            thr = input("Min probability/confidence (0-1) [default 0.6]: ").strip()
            thr = float(thr) if thr else 0.6
            model = input("Optional model path (.pkl) [Enter to use heuristics]: ").strip()
            option_scan_universe("sp500", min_proba=thr, p1_ge_tol=tol, model_path=model, plot_top=0)
            input("\nPress Enter to return to menu…")
        elif choice == '4':
            tol = input("Tilt tolerance (P1≥P2) e.g. 0.02 [default 0.00]: ").strip()
            tol = float(tol) if tol else 0.00
            thr = input("Min probability/confidence (0-1) [default 0.6]: ").strip()
            thr = float(thr) if thr else 0.6
            model = input("Optional model path (.pkl) [Enter to use heuristics]: ").strip()
            option_scan_universe("nasdaq100", min_proba=thr, p1_ge_tol=tol, model_path=model, plot_top=0)
            input("\nPress Enter to return to menu…")
        elif choice == 'q':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid option.")

# ------------------------------- Entry point ------------------------------------

if __name__ == "__main__":
    # If you still want CLI subcommands, you can add them; for now we go straight to menu.
    menu_loop()
