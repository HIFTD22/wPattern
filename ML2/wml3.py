#!/usr/bin/env python3
"""
W-Pattern Labeling + Scanning (Heuristics + Optional ML)

Features:
- Load S&P 500 / Nasdaq-100 symbols (file cache first; Wikipedia scrape with lxml fallback)
- Fetch 90d daily OHLCV from Yahoo Finance
- Detect W candidates using Open/Close extremes (P1/L1/P2/L2 order, P1 ≥ P2 tilt)
- Compute target using user's formula (neckline slope, trough midpoint)
- Optional ML scoring if a joblib bundle is supplied
- Save plots for proposals, single checks, and universe scans
- Interactive menu that does not auto-exit

Files used/created:
- sp500.txt / nasdaq100.txt (optional caches; one ticker per line)
- w_labels.csv (appended as you label proposals)
- plots/ (charts)
- results/*.csv (scan outputs)

Dependencies (pip or conda):
  yfinance, pandas, numpy, matplotlib, scikit-learn, joblib, requests, lxml
"""

from __future__ import annotations
import os
import io
import re
import sys
import time
import math
import json
import joblib
import queue
import random
import string
import signal
import typing as T
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

import yfinance as yf
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score  # only used if you wire training later

# ----------------------------
# Globals / paths
# ----------------------------
PLOTS_DIR = "plots"
RESULTS_DIR = "results"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------
# Utility
# ----------------------------
def _norm_ticker(s: str) -> str:
    s = str(s).strip().upper()
    s = s.replace(" ", "")
    s = s.replace(".", "-")  # Yahoo uses - instead of .
    return s

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _input_default(prompt: str, default: str) -> str:
    s = input(prompt).strip()
    return s if s else default

# ----------------------------
# Universe loaders (robust)
# ----------------------------
def _dedupe_keep_order(items: T.Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def load_sp500_symbols() -> list[str]:
    """
    Load S&P 500.
    1) sp500.txt if present (one ticker per line)
    2) Wikipedia scrape with lxml (no html5lib requirement)
    """
    cache = "sp500.txt"
    if os.path.exists(cache):
        with open(cache, "r") as f:
            raw = [x.strip() for x in f if x.strip()]
        syms = [_norm_ticker(x) for x in raw]
        syms = _dedupe_keep_order(syms)
        if len(syms) >= 450:
            print(f"📊 Loaded {len(syms)} S&P 500 symbols from {cache}")
            return syms
        else:
            print(f"⚠️ {cache} has only {len(syms)} symbols; attempting Wikipedia…")

    # Wikipedia scrape
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    resp.raise_for_status()

    # Prefer lxml (avoids html5lib)
    tables = pd.read_html(resp.text, flavor="lxml")
    if not tables:
        raise RuntimeError("Could not parse S&P 500 tables from Wikipedia.")

    df = tables[0]
    # The symbol column is usually 'Symbol'
    col = None
    for c in df.columns:
        if str(c).strip().lower() in ("symbol", "ticker"):
            col = c
            break
    if col is None:
        # Fallback: heuristic
        candidates = []
        for c in df.columns:
            s = df[c].astype(str).str.strip()
            pat = re.compile(r"^[A-Za-z]{1,5}(?:[.\-][A-Za-z]{1,3})?$")
            m = s.apply(lambda x: bool(pat.fullmatch(x))).mean()
            if m >= 0.5:
                candidates.append(c)
        col = candidates[0] if candidates else df.columns[0]

    syms = df[col].astype(str).map(_norm_ticker).tolist()
    syms = _dedupe_keep_order(syms)
    if len(syms) < 450:
        raise RuntimeError("Wikipedia returned too few S&P symbols; create sp500.txt and rerun.")
    with open(cache, "w") as f:
        f.write("\n".join(syms) + "\n")
    print(f"📊 Loaded {len(syms)} S&P 500 symbols from Wikipedia → cached to {cache}")
    return syms

def load_nasdaq100_symbols() -> list[str]:
    """
    Load Nasdaq-100.
    1) nasdaq100.txt if present
    2) Wikipedia scrape (lxml-first, no html5lib)
    """
    cache = "nasdaq100.txt"
    if os.path.exists(cache):
        with open(cache, "r") as f:
            raw = [x.strip() for x in f if x.strip()]
        syms = [_norm_ticker(x) for x in raw]
        syms = _dedupe_keep_order(syms)
        if 80 <= len(syms) <= 200:
            print(f"📊 Loaded {len(syms)} Nasdaq-100 tickers from {cache}")
            return syms
        else:
            print(f"⚠️ {cache} has {len(syms)} symbols; attempting Wikipedia…")

    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    resp.raise_for_status()

    tables = []
    try:
        tables = pd.read_html(resp.text, flavor="lxml")
    except Exception:
        # Best-effort fallback: use BeautifulSoup+ lxml to slice tables
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")
        for tbl in soup.find_all("table"):
            try:
                tdf = pd.read_html(str(tbl), flavor="lxml")[0]
                tables.append(tdf)
            except Exception:
                continue

    if not tables:
        raise RuntimeError("Could not parse Nasdaq-100 tables. Create nasdaq100.txt and rerun.")

    tickers = []
    pat = re.compile(r"^[A-Za-z]{1,5}(?:[.\-][A-Za-z]{1,3})?$")

    def looks_like_ticker_series(s: pd.Series) -> bool:
        vals = s.astype(str).str.strip()
        if len(vals) == 0:
            return False
        matches = vals.apply(lambda x: bool(pat.fullmatch(x)))
        digit_rate = vals.str.contains(r"\d").mean()
        return matches.mean() >= 0.5 and digit_rate < 0.15

    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        target = None
        for i, c in enumerate(cols):
            if "ticker" in c or "symbol" in c:
                target = t.columns[i]
                break
        series = t[target] if target is not None else None
        if series is None:
            for col in t.columns:
                s = t[col]
                if looks_like_ticker_series(s):
                    series = s
                    break
        if series is None:
            continue
        for raw in series.astype(str).tolist():
            tok = _norm_ticker(raw)
            if tok and len(tok) <= 10 and "^" not in tok and " " not in tok:
                tickers.append(tok)

    syms = _dedupe_keep_order(tickers)
    if len(syms) < 80:
        raise RuntimeError("Wikipedia returned too few Nasdaq-100; create nasdaq100.txt and rerun.")
    with open(cache, "w") as f:
        f.write("\n".join(syms) + "\n")
    print(f"📊 Loaded {len(syms)} from Wikipedia → cached to {cache}")
    return syms

# ----------------------------
# Data fetch
# ----------------------------
def fetch_history(symbol: str, lookback_days: int = 90, min_bars: int = 45) -> pd.DataFrame | None:
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 10)  # buffer for holidays
    try:
        df = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(df) < min_bars:
        return None
    df.index = pd.to_datetime(df.index)
    return df.iloc[-lookback_days:] if len(df) > lookback_days else df

# ----------------------------
# Open/Close extremes helpers
# ----------------------------
def oc_high(df: pd.DataFrame, i: int) -> float:
    o = float(df["Open"].iloc[i]); c = float(df["Close"].iloc[i])
    return max(o, c)

def oc_low(df: pd.DataFrame, i: int) -> float:
    o = float(df["Open"].iloc[i]); c = float(df["Close"].iloc[i])
    return min(o, c)

# ----------------------------
# Candidate detection (heuristics)
# ----------------------------
def find_w_candidates_from_ohlc(
    df: pd.DataFrame,
    p1_ge_tol: float = 0.00,
    min_conf: float = 0.40,
    max_per_symbol: int = 3,
) -> list[dict]:
    """
    Detect W candidates using OC extremes.

    Steps:
    - Build OC-high series (for peaks), OC-low series (for troughs).
    - Find trough indices by peaking on negative OC-low.
    - Find peak indices by peaking on OC-high.
    - For every pair of troughs L1<L2 with >= one peak in between:
        P2 = highest peak between L1..L2 (neckline)
        P1 = last peak before L1
        Require P1 >= P2*(1-p1_ge_tol)
        Compute confidence, target, etc.
    """
    if len(df) < 30:
        return []

    oc_high_series = df[["Open", "Close"]].max(axis=1).to_numpy()
    oc_low_series  = df[["Open", "Close"]].min(axis=1).to_numpy()

    # Prominences scale with volatility
    vol = np.nanstd(df["Close"].pct_change().dropna().values) or 0.01
    prom_peak = max(oc_high_series.std() * 0.20, float(np.nanmean(oc_high_series)) * 0.002)
    prom_trough = max(oc_low_series.std() * 0.20, float(np.nanmean(oc_low_series)) * 0.002)

    # Peaks (P candidates)
    pk_idx, _ = find_peaks(oc_high_series, prominence=prom_peak, distance=max(3, len(df)//30))
    # Troughs (L candidates) via inverted series
    tr_idx, _ = find_peaks(-oc_low_series, prominence=prom_trough, distance=3)

    if len(tr_idx) < 2 or len(pk_idx) < 1:
        return []

    cands: list[dict] = []
    N = len(df)

    for a in range(len(tr_idx)-1):
        i_l1 = int(tr_idx[a])
        for b in range(a+1, len(tr_idx)):
            i_l2 = int(tr_idx[b])
            if i_l2 - i_l1 < 3:  # need some spacing
                continue
            # peaks between
            between = [i for i in pk_idx if i_l1 < i < i_l2]
            if not between:
                continue
            i_p2 = int(max(between, key=lambda i: oc_high(df, i)))  # neckline peak
            # peak before L1 as P1
            left_peaks = [i for i in pk_idx if i < i_l1]
            if not left_peaks:
                continue
            i_p1 = int(max(left_peaks))  # last by index

            P1 = oc_high(df, i_p1)
            P2 = oc_high(df, i_p2)
            L1 = oc_low(df,  i_l1)
            L2 = oc_low(df,  i_l2)

            # tilt: P1 >= P2*(1 - p1_ge_tol)
            if P1 < P2 * (1.0 - float(p1_ge_tol)):
                continue

            # neckline above troughs
            if P2 <= max(L1, L2):
                continue

            # Compute basic metrics
            pattern_width = i_l2 - i_l1
            pattern_height = P2 - min(L1, L2)
            if pattern_height <= 0:
                continue

            # symmetry (valley depths closeness)
            depth_sym = 1.0 - abs(L2 - L1) / max(L1, L2, 1e-9)
            # proportion – prefer some width vs height balance
            prop = min(1.0, pattern_width / max(1.0, pattern_height / max(1e-6, np.mean(df["Close"]))))
            # proximity factor – how close current price is to neckline (forming preference)
            curr = float(df["Close"].iloc[-1])
            neck_val = P2  # neckline at P2 (by definition)
            prox = 1.0 - max(0.0, (neck_val - curr) / max(1e-9, neck_val))  # 1 if at/above, smaller if below

            # heuristic probability
            proba = 0.45*depth_sym + 0.25*min(1.0, pattern_height/max(1e-6, np.std(df['Close']))) + 0.30*max(0.0, min(1.0, prox))
            proba = float(max(0.0, min(1.0, proba)))

            if proba < min_conf:
                continue

            # Your target formula
            tgt = target_from_formula(df, i_p1, i_l1, i_p2, i_l2)

            cands.append({
                "p1_idx": i_p1, "l1_idx": i_l1, "p2_idx": i_p2, "l2_idx": i_l2,
                "p1_date": df.index[i_p1], "l1_date": df.index[i_l1],
                "p2_date": df.index[i_p2], "l2_date": df.index[i_l2],
                "p1_price": P1, "l1_price": L1, "p2_price": P2, "l2_price": L2,
                "proba": proba,
                **tgt
            })

    # sort by probability descending
    cands.sort(key=lambda d: d.get("proba", 0.0), reverse=True)
    return cands[:max_per_symbol] if max_per_symbol else cands

# ----------------------------
# Target formula (user spec)
# ----------------------------
def target_from_formula(df: pd.DataFrame, i_p1: int, i_l1: int, i_p2: int, i_l2: int) -> dict:
    """
    trough_midpoint = (L1+L2)/2
    target_magnitude = P2 - trough_midpoint
    breakout_estimate_days = i_p2 - i_p1
    neckline_slope = (P2 - P1) / breakout_estimate_days
    breakout_estimate_price = P2 + neckline_slope * breakout_estimate_days
    target_price = breakout_estimate_price + target_magnitude
    """
    P1 = oc_high(df, i_p1)
    P2 = oc_high(df, i_p2)
    L1 = oc_low(df,  i_l1)
    L2 = oc_low(df,  i_l2)

    trough_mid = 0.5*(L1 + L2)
    target_mag = P2 - trough_mid
    days = max(1, int(i_p2 - i_p1))
    neck_slope = (P2 - P1) / float(days)
    breakout_est_price = P2 + neck_slope * days
    target_price = breakout_est_price + target_mag

    i_breakout_est = min(len(df)-1, i_p2 + days)
    dt_breakout_est = df.index[i_breakout_est]

    return dict(
        target_price=float(target_price),
        trough_midpoint=float(trough_mid),
        target_magnitude=float(target_mag),
        neckline_slope=float(neck_slope),
        breakout_estimate_days=int(days),
        breakout_estimate_price=float(breakout_est_price),
        breakout_estimate_date=pd.to_datetime(dt_breakout_est),
    )

# ----------------------------
# Plotting
# ----------------------------
def _idx_from_pat(df: pd.DataFrame, pat: dict, idx_keys, date_keys):
    for k in idx_keys:
        if k in pat and pat[k] is not None:
            return int(pat[k])
    for k in date_keys:
        if k in pat and pat[k] is not None:
            dt = pd.to_datetime(pat[k])
            if dt in df.index:
                return int(df.index.get_loc(dt))
            loc = df.index.get_indexer([dt], method="nearest")[0]
            return int(loc)
    raise KeyError("Cannot resolve index from pattern.")

def plot_w_candidate(df: pd.DataFrame, pat: dict, symbol: str, outdir=PLOTS_DIR, title_note: str = "") -> str:
    _ensure_dir(outdir)
    i_p1 = _idx_from_pat(df, pat, ["p1_idx"], ["p1_date"])
    i_l1 = _idx_from_pat(df, pat, ["l1_idx", "low1_idx"], ["l1_date", "low1_date"])
    i_p2 = _idx_from_pat(df, pat, ["p2_idx"], ["p2_date"])
    i_l2 = _idx_from_pat(df, pat, ["l2_idx", "low2_idx"], ["l2_date", "low2_date"])

    # neckline via P1→P2 using OC-highs
    P1 = oc_high(df, i_p1)
    P2 = oc_high(df, i_p2)
    x = np.arange(len(df))
    m = (P2 - P1) / max(1, (i_p2 - i_p1))
    b = P1 - m * i_p1
    neckline = m * x + b

    # target info
    tgt = target_from_formula(df, i_p1, i_l1, i_p2, i_l2)

    date0 = df.index[0].strftime("%Y%m%d")
    date1 = df.index[-1].strftime("%Y%m%d")
    fname = f"{symbol}_{date0}_{date1}.png"
    path = os.path.join(outdir, fname)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df.index, df["Close"].values, lw=1.2, label="Close")
    ax.plot(df.index, neckline, ls="--", lw=1.1, label="Neckline (P1→P2)")

    # pivot markers
    ax.scatter(df.index[i_p1], oc_high(df, i_p1), s=55, zorder=5, label="P1 (OC-high)")
    ax.scatter(df.index[i_p2], oc_high(df, i_p2), s=55, zorder=5, label="P2 (OC-high)")
    ax.scatter(df.index[i_l1], oc_low(df, i_l1),  s=55, zorder=5, label="L1 (OC-low)")
    ax.scatter(df.index[i_l2], oc_low(df, i_l2),  s=55, zorder=5, label="L2 (OC-low)")

    ax.axhline(tgt["target_price"], color="tab:green", lw=1.1, ls=":", label=f"Target ≈ {tgt['target_price']:.2f}")
    ax.axvline(tgt["breakout_estimate_date"], color="tab:orange", lw=1, ls=":", label=f"Est. Breakout ~ {tgt['breakout_estimate_date'].date()}")

    conf = pat.get("proba", pat.get("confidence"))
    conf_txt = f" | P(W)={conf:.2f}" if isinstance(conf, (int, float)) else ""
    ax.set_title(f"{symbol} W-candidate{conf_txt}{(' | ' + title_note) if title_note else ''}")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path

# ----------------------------
# Feature builder + model scoring (optional)
# ----------------------------


# ----------------------------
# Feature builder + model scoring (with legacy compatibility)
# ----------------------------

FEATURE_COLUMNS = [
    # current heuristic features
    "dur_p1_l2", "dur_l1_l2", "dur_p1_l1", "dur_l1_p2", "dur_p2_l2",
    "p1_price", "p2_price", "l1_price", "l2_price",
    "depth_symmetry", "height", "width",
    "neckline_slope", "trough_midpoint", "target_magnitude",
    "close_now", "close_vs_neck",
    "vol_mean", "vol_now",
]

def build_features(df: pd.DataFrame, pat: dict) -> dict:
    i_p1 = int(pat["p1_idx"]); i_l1 = int(pat["l1_idx"])
    i_p2 = int(pat["p2_idx"]); i_l2 = int(pat["l2_idx"])
    P1 = float(pat["p1_price"]); P2 = float(pat["p2_price"])
    L1 = float(pat["l1_price"]); L2 = float(pat["l2_price"])

    width = i_l2 - i_l1
    height = P2 - min(L1, L2)
    depth_sym = 1.0 - abs(L2 - L1) / max(L1, L2, 1e-9)

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

def compute_legacy_feature(name: str, df: pd.DataFrame, pat: dict):
    """
    Compute common legacy features your older model might expect.
    If a name isn’t recognized, return None (we’ll fill 0.0).
    """
    try:
        i_p1 = int(pat["p1_idx"]); i_l1 = int(pat["l1_idx"])
        i_p2 = int(pat["p2_idx"]); i_l2 = int(pat["l2_idx"])
        P1 = float(pat["p1_price"]); P2 = float(pat["p2_price"])
        L1 = float(pat["l1_price"]); L2 = float(pat["l2_price"])
        close_now = float(df["Close"].iloc[-1])
    except Exception:
        return None

    if name == "p1_l1_drop_pct":
        return (L1 - P1) / P1  # negative number (drop)
    if name == "l1_p2_rise_pct":
        return (P2 - L1) / max(1e-12, L1)
    if name == "p2_l2_drop_pct":
        return (L2 - P2) / max(1e-12, P2)  # negative
    if name == "width_bars":
        return i_l2 - i_l1
    if name == "height_abs":
        return P2 - min(L1, L2)
    if name == "near_neckline_pct":
        return (close_now - P2) / max(1e-12, P2)
    if name == "vol_mean_20":
        return float(df["Volume"].iloc[max(0, i_p1-20):i_l2+1].mean())
    if name == "vol_now":
        return float(df["Volume"].iloc[-1])
    if name == "close_now":
        return close_now
    if name == "close_vs_neck":
        return close_now - P2
    if name == "p1_to_p2_days":
        return i_p2 - i_p1
    if name == "l1_to_l2_days":
        return i_l2 - i_l1

    # Add more mappings here if your old model used other names.
    return None

_missing_warned = False

def score_with_model(bundle, df: pd.DataFrame, pat: dict) -> float:
    """
    Build current features, then backfill any legacy feature names
    required by the loaded model. Unknowns are filled with 0.0 (once-warn).
    """
    global _missing_warned
    feats = build_features(df, pat)
    order = bundle["features"]
    missing = []
    # backfill legacy keys
    for k in order:
        if k not in feats:
            val = compute_legacy_feature(k, df, pat)
            if val is None:
                missing.append(k)
                feats[k] = 0.0  # safe default
            else:
                feats[k] = float(val)

    if missing and not _missing_warned:
        print(f"⚠️ Model expects {len(missing)} unknown feature(s). "
              f"Filled with 0.0: {', '.join(missing[:8])}{'…' if len(missing)>8 else ''}")
        _missing_warned = True

    X = [[feats.get(k, 0.0) for k in order]]
    if hasattr(bundle["model"], "predict_proba"):
        return float(bundle["model"].predict_proba(X)[0][1])
    else:
        return float(bundle["model"].predict(X)[0])
    i_p1 = int(pat["p1_idx"]); i_l1 = int(pat["l1_idx"])
    i_p2 = int(pat["p2_idx"]); i_l2 = int(pat["l2_idx"])
    P1 = float(pat["p1_price"]); P2 = float(pat["p2_price"])
    L1 = float(pat["l1_price"]); L2 = float(pat["l2_price"])

    width = i_l2 - i_l1
    height = P2 - min(L1, L2)
    depth_sym = 1.0 - abs(L2 - L1) / max(L1, L2, 1e-9)

    now = float(df["Close"].iloc[-1])
    close_vs_neck = now - P2

    # volumes
    vol_mean = float(df["Volume"].iloc[max(0, i_p1-20):i_l2+1].mean())
    vol_now = float(df["Volume"].iloc[-1])

    feats = {
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
    return feats

def load_model_bundle(model_path: str | None):
    if not model_path:
        return None
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("Model file must be a dict with keys: 'model', 'features'.")
    return bundle



# ----------------------------
# Option 1 — Train more data (proposals)
# ----------------------------
def option_train_more(min_conf=0.25, p1_ge_tol=0.02, batch_size=10):
    syms = load_sp500_symbols()
    random.shuffle(syms)
    labeled_rows = []
    proposals_dir = os.path.join(PLOTS_DIR, "proposals")
    _ensure_dir(proposals_dir)

    print(f"\nProposing ≈{batch_size} candidates for labeling (Y/N/Skip). Charts saved to {proposals_dir}/\n")

    count = 0
    for sym in syms:
        if count >= batch_size:
            break
        df = fetch_history(sym, 90)
        if df is None: 
            continue
        cands = find_w_candidates_from_ohlc(df, p1_ge_tol=p1_ge_tol, min_conf=min_conf, max_per_symbol=2)
        if not cands:
            continue
        for pat in cands:
            png = None
            try:
                png = plot_w_candidate(df, pat, sym, proposals_dir, title_note="proposal")
            except Exception as e:
                print(f"   (plot skipped: {e})")

            print(f"\n{sym}: P(W)~{pat['proba']:.2f}  "
                  f"P1={pat['p1_date'].date()}  L1={pat['l1_date'].date()}  "
                  f"P2={pat['p2_date'].date()}  L2={pat['l2_date'].date()}  "
                  f"Target≈${pat['target_price']:.2f}")
            if png:
                print(f"   chart: {png}")

            ans = input("   Label as W? [y]es / [n]o / [s]kip: ").strip().lower()
            if ans not in ("y", "n"):
                print("   skipped.")
                continue

            y = 1 if ans == "y" else 0
            feats = build_features(df, pat)
            row = {
                "symbol": sym,
                **{k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in pat.items()},
                **feats,
                "y_shape": y
            }
            labeled_rows.append(row)
            count += 1
            if count >= batch_size:
                break

    if labeled_rows:
        out_csv = "w_labels.csv"
        df_out = pd.DataFrame(labeled_rows)
        if os.path.exists(out_csv):
            df_out.to_csv(out_csv, mode="a", header=False, index=False)
        else:
            df_out.to_csv(out_csv, index=False)
        print(f"\n✅ Appended {len(labeled_rows)} rows → {out_csv}")
    else:
        print("\nNo labels added.")

# ----------------------------
# Option 2 — Check one ticker & plot
# ----------------------------
def option_check_one_ticker():
    sym = _norm_ticker(_input_default("Ticker (e.g., COST): ", "COST"))
    tol = float(_input_default("Tilt tolerance P1≥P2 (e.g., 0.02): ", "0.00"))
    thr = float(_input_default("Min probability/confidence (0-1): ", "0.60"))
    model_path = input("Optional model path (.pkl) [Enter to use heuristics]: ").strip()
    bundle = load_model_bundle(model_path) if model_path else None

    df = fetch_history(sym, 90)
    if df is None:
        print("No data.")
        return

    cands = find_w_candidates_from_ohlc(df, p1_ge_tol=tol, min_conf=0.0, max_per_symbol=5)
    if not cands:
        print("No W candidates found.")
        return

    # score and filter
    hits = []
    for pat in cands:
        proba = score_with_model(bundle, df, pat) if bundle else float(pat["proba"])
        pat["proba"] = proba
        if proba >= thr:
            hits.append(pat)

    if not hits:
        print("No candidates above threshold.")
        return

    hits.sort(key=lambda d: d["proba"], reverse=True)
    outdir = os.path.join(PLOTS_DIR, "single")
    _ensure_dir(outdir)
    for i, pat in enumerate(hits, 1):
        png = plot_w_candidate(df, pat, sym, outdir)
        print(f"[{i}] {sym}  P(W)={pat['proba']:.2f}  Target≈${pat['target_price']:.2f}  → {png}")

# ----------------------------
# Option 3/4 — Universe scans
# ----------------------------
def option_scan_universe(universe: str, min_proba=0.6, p1_ge_tol=0.0, model_path=None, plot_top=25):
    symbols = load_sp500_symbols() if universe == "sp500" else load_nasdaq100_symbols()
    bundle = load_model_bundle(model_path) if model_path else None
    t0 = time.time()
    hits: list[tuple[str, dict, pd.DataFrame]] = []

    print(f"\nScanning {len(symbols)} symbols in {universe.upper()}…")
    for i, sym in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {sym} …", end="")
        df = fetch_history(sym, 90)
        if df is None:
            print(" no data")
            continue
        cands = find_w_candidates_from_ohlc(df, p1_ge_tol=p1_ge_tol, min_conf=0.0, max_per_symbol=3)
        if not cands:
            print(" none")
            continue
        kept = []
        for pat in cands:
            proba = score_with_model(bundle, df, pat) if bundle else float(pat["proba"])
            pat["proba"] = proba
            if proba >= min_proba:
                kept.append(pat)
        if kept:
            print(f" {len(kept)} hit(s)")
            for pat in kept:
                hits.append((sym, pat, df.copy()))
        else:
            print(" below threshold")

    if not hits:
        print("No candidates found.")
        return

    # sort by probability
    hits.sort(key=lambda t: t[1]["proba"], reverse=True)

    # save CSV
    rows = []
    for sym, pat, df in hits:
        rows.append({
            "symbol": sym,
            "proba": pat["proba"],
            "p1_date": pat["p1_date"],
            "l1_date": pat["l1_date"],
            "p2_date": pat["p2_date"],
            "l2_date": pat["l2_date"],
            "p1_price": pat["p1_price"],
            "l1_price": pat["l1_price"],
            "p2_price": pat["p2_price"],
            "l2_price": pat["l2_price"],
            "target_price": pat["target_price"],
            "breakout_estimate_date": pat["breakout_estimate_date"],
            "neckline_slope": pat["neckline_slope"],
            "trough_midpoint": pat["trough_midpoint"],
            "target_magnitude": pat["target_magnitude"],
        })
    out_csv = os.path.join(RESULTS_DIR, f"scan_{universe}_{_ts()}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n✅ Saved results → {out_csv}")

    # plots
    outdir = os.path.join(PLOTS_DIR, universe)
    _ensure_dir(outdir)
    count = 0
    for sym, pat, df in hits:
        if plot_top > 0 and count >= plot_top:
            break
        try:
            png = plot_w_candidate(df, pat, sym, outdir)
            print(f"📈 {sym}: {png}")
            count += 1
        except Exception as e:
            print(f"(plot skipped for {sym}: {e})")

    dt = time.time() - t0
    print(f"\nDone. {len(hits)} candidates in {dt:.1f}s.")

# ----------------------------
# Menu loop
# ----------------------------
def menu_loop():
    while True:
        print("\nOptions:")
        print("1) Train more data (≈10 candidates to label)")
        print("2) Check one ticker (90d) and plot")
        print("3) Scan S&P 500 (targets via your formula)")
        print("4) Scan Nasdaq-100 (targets via your formula)")
        print("q) Exit\n")
        sel = input("Select: ").strip().lower()

        if sel == "1":
            tol = float(_input_default("Tilt tolerance (P1≥P2) e.g. 0.02 [default 0.00]: ", "0.00"))
            thr = float(_input_default("Min confidence for proposal (0-1) [default 0.25]: ", "0.25"))
            bs = int(_input_default("Batch size [default 10]: ", "10"))
            option_train_more(min_conf=thr, p1_ge_tol=tol, batch_size=bs)
            input("\nPress Enter to return to menu…")
        elif sel == "2":
            option_check_one_ticker()
            input("\nPress Enter to return to menu…")
        elif sel == "3":
            tol = float(_input_default("Tilt tolerance (P1≥P2) e.g. 0.02 [default 0.00]: ", "0.00"))
            thr = float(_input_default("Min probability/confidence (0-1) [default 0.6]: ", "0.60"))
            model = input("Optional model path (.pkl) [Enter to use heuristics]: ").strip()
            option_scan_universe("sp500", min_proba=thr, p1_ge_tol=tol, model_path=model or None, plot_top=25)
            input("\nPress Enter to return to menu…")
        elif sel == "4":
            tol = float(_input_default("Tilt tolerance (P1≥P2) e.g. 0.02 [default 0.00]: ", "0.00"))
            thr = float(_input_default("Min probability/confidence (0-1) [default 0.6]: ", "0.60"))
            model = input("Optional model path (.pkl) [Enter to use heuristics]: ").strip()
            option_scan_universe("nasdaq100", min_proba=thr, p1_ge_tol=tol, model_path=model or None, plot_top=25)
            input("\nPress Enter to return to menu…")
        elif sel in ("q", "quit", "exit"):
            print("👋 Goodbye!")
            break
        else:
            print("Invalid selection.")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("W-Pattern Labeling + Scanning")
    print("=" * 60)
    menu_loop()
