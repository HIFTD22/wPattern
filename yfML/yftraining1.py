#!/usr/bin/env python3
"""
Two-stage W-pattern pipeline (daily OHLCV, last 90 trading days)

Stage 1 (Shape/Condition Recognition):
  - train --target shape         # trains a W vs not-W recognizer from your labeled CSV (w_with_features.csv)
  - scan-shape                   # finds candidates and scores P(W) with the shape model

Stage 2 (Outcomes on Ws only):
  - train --target breakout      # trains breakout success model on labeled Ws
  - train --target reachtarget   # trains "reach price target" model on labeled Ws
  - scan                         # applies shape filter, then scores breakout/reach-target

Also available:
  - train --target all           # trains shape + breakout + reachtarget in one go
  - propose                      # auto-generate pattern candidates for you to label
  - augment is done by yfslope1.py (your separate script) to add slope/indicators/features

Install (once):
  pip install yfinance pandas numpy scikit-learn requests lxml matplotlib joblib
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import argparse
import sys
import time
from datetime import datetime, timedelta, date
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import dump, load

# ------------------------- Global config -------------------------

RECENT_WINDOW = 90
MAX_WORKERS = 8
TIMEOUT_SECS = 12
MAX_RETRIES = 3
RETRY_SLEEP = 0.7

# Relaxed W-structure thresholds (for candidate generation)
PCT_ZZ = 0.015       # zigzag reversal baseline %
ATR_MULT = 0.5
ATR_LEN = 14
MIN_SEP = 3
MAX_SEP = 85
TROUGH_SIM_LO = 0.00
TROUGH_SIM_HI = 0.20
PREDROP_MIN = 0.02
POST_L2_LIFT_MIN = 0.00
BREAKOUT_BUFFER = 0.001
FORMING_NEARLINE_TOL = 0.05

PRICE_MODE_PRIORITY = ["OCMID", "CLOSE", "OPEN", "HL2", "HLC3"]

# Models / files
MODEL_SHAPE        = "model_shape.pkl"
MODEL_BREAKOUT     = "model_breakout.pkl"
MODEL_REACHTARGET  = "model_reachtarget.pkl"
FEATURES_FILE      = "feature_list.json"
PLOTS_DIR          = "plots"

# ------------------------- Feature Set -------------------------
# These are computed by your yfslope1.py (augment) and also by this script for proposals/scan
FEATURE_COLUMNS = [
    # geometry & timing (pre-breakout only; uses your labeled P/L prices)
    "drop1_norm","rise1_norm","drop2_norm",
    "neckline_slope_per_day","trough_sim",
    "spread_t","p1_to_l1_t","l1_to_p2_t","p2_to_l2_t",
    # indicators (Yahoo-derived, no leakage)
    "rsi_p2","rsi_l2",
    "vol_ratio_p2","vol_ratio_l2",
    "px_vs_ma20_p2","px_vs_ma20_l2","px_vs_ma50_l2",
]

# ------------------------- Utilities -------------------------

def norm_symbol(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")

def parse_ymd(s: str) -> Optional[date]:
    if pd.isna(s) or str(s).strip() == "":
        return None
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()
    except Exception:
        return None

def get_sp500_tickers(limit: int) -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url, timeout=TIMEOUT_SECS)
        table = next((t for t in tables if "Symbol" in t.columns), None)
        if table is not None:
            syms = (
                table["Symbol"].astype(str)
                .str.replace(".", "-", regex=False)
                .str.strip().str.upper().tolist()
            )
            syms = sorted(list({s for s in syms if s and s != "NAN"}))
            if len(syms) >= limit:
                return syms[:limit]
    except Exception:
        pass
    fallback_50 = [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","LLY","AVGO",
        "TSLA","JPM","V","WMT","XOM","UNH","MA","PG","ORCL","COST",
        "HD","MRK","KO","PEP","BAC","ABBV","CRM","PFE","TMO","DIS",
        "ACN","CSCO","NFLX","ABT","DHR","AMD","QCOM","LIN","MCD","TXN",
        "VZ","NKE","UPS","PM","CAT","IBM","HON","RTX","LOW","ADP"
    ]
    return fallback_50[:limit]

def atr(df: pd.DataFrame, n: int = ATR_LEN) -> pd.Series:
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def robust_history(ticker: str) -> Optional[pd.DataFrame]:
    for attempt in range(1, MAX_RETRIES+1):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="12mo", interval="1d", auto_adjust=True)
            if df is None or df.empty:
                raise RuntimeError("empty")
            need = ["Open","High","Low","Close","Volume"]
            if not all(c in df.columns for c in need):
                raise RuntimeError("missing OHLCV")
            df = df[need].dropna().sort_index()
            if df.shape[0] > RECENT_WINDOW:
                df = df.tail(RECENT_WINDOW).copy()
            return df
        except Exception:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_SLEEP)
            continue

def build_series(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "OCMID":
        s = (df["Open"].astype(float) + df["Close"].astype(float)) / 2.0
    elif mode == "CLOSE":
        s = df["Close"].astype(float)
    elif mode == "OPEN":
        s = df["Open"].astype(float)
    elif mode == "HL2":
        s = (df["High"].astype(float) + df["Low"].astype(float)) / 2.0
    elif mode == "HLC3":
        s = (df["High"].astype(float) + df["Low"].astype(float) + df["Close"].astype(float)) / 3.0
    else:
        raise ValueError(mode)
    return np.asarray(s.to_numpy(), dtype=float).reshape(-1)

# Effective prices per your rule
def effective_peak_price(df: pd.DataFrame, idx: int) -> float:
    o = float(df["Open"].iloc[idx]); c = float(df["Close"].iloc[idx]); h = float(df["High"].iloc[idx])
    return float(max(h, o, c))

def effective_trough_price(df: pd.DataFrame, idx: int) -> float:
    o = float(df["Open"].iloc[idx]); c = float(df["Close"].iloc[idx]); l = float(df["Low"].iloc[idx])
    return float(min(l, o, c))

# ------------------------- Indicators -------------------------

def rsi_ewm(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["VolMA20"] = df["Volume"].rolling(20, min_periods=1).mean()
    df["RSI14"] = rsi_ewm(df["Close"], 14)
    return df

def indicator_features_at(df: pd.DataFrame, idx: int, tag: str) -> Dict[str, float]:
    if idx is None or idx < 0 or idx >= len(df):
        return {f"rsi_{tag}": np.nan, f"vol_ratio_{tag}": np.nan,
                f"px_vs_ma20_{tag}": np.nan, f"px_vs_ma50_{tag}": np.nan}
    c = float(df["Close"].iloc[idx])
    rsi = float(df["RSI14"].iloc[idx]) if not pd.isna(df["RSI14"].iloc[idx]) else np.nan
    vol = float(df["Volume"].iloc[idx])
    volma20 = float(df["VolMA20"].iloc[idx]) if not pd.isna(df["VolMA20"].iloc[idx]) else np.nan
    ma20 = float(df["MA20"].iloc[idx]) if not pd.isna(df["MA20"].iloc[idx]) else np.nan
    ma50 = float(df["MA50"].iloc[idx]) if not pd.isna(df["MA50"].iloc[idx]) else np.nan
    return {
        f"rsi_{tag}": rsi,
        f"vol_ratio_{tag}": (vol / volma20) if (volma20 and not np.isnan(volma20) and volma20 != 0) else np.nan,
        f"px_vs_ma20_{tag}": (c / ma20 - 1.0) if (ma20 and not np.isnan(ma20) and ma20 != 0) else np.nan,
        f"px_vs_ma50_{tag}": (c / ma50 - 1.0) if (ma50 and not np.isnan(ma50) and ma50 != 0) else np.nan,
    }

# ------------------------- ZigZag pivots to LOCATE candidates ------------------

def atr_series(df: pd.DataFrame, n: int = ATR_LEN) -> np.ndarray:
    return atr(df, n).to_numpy()

def zigzag_pivots_hl(df: pd.DataFrame,
                     pct_th: float = PCT_ZZ,
                     atr_mult: float = ATR_MULT) -> List[Tuple[int, float, str]]:
    highs = df["High"].astype(float).to_numpy()
    lows  = df["Low"].astype(float).to_numpy()
    n = highs.size
    if n < 5:
        return []
    a = atr_series(df, ATR_LEN)

    def thr(i_ref: int, ref_price: float) -> float:
        return max(pct_th * max(1e-9, ref_price), atr_mult * a[i_ref])

    piv, mode = [], None
    cur_ext_idx, cur_ext_price = 0, highs[0]
    for i in range(1, n):
        if highs[i] > cur_ext_price:
            cur_ext_price, cur_ext_idx = highs[i], i
        if (cur_ext_price - lows[i]) >= thr(cur_ext_idx, cur_ext_price):
            piv.append((cur_ext_idx, highs[cur_ext_idx], "peak")); mode = "down"; break
    if mode is None:
        cur_ext_idx, cur_ext_price = 0, lows[0]
        for i in range(1, n):
            if lows[i] < cur_ext_price:
                cur_ext_price, cur_ext_idx = lows[i], i
            if (highs[i] - cur_ext_price) >= thr(cur_ext_idx, cur_ext_price):
                piv.append((cur_ext_idx, lows[cur_ext_idx], "trough")); mode = "up"; break
    if mode is None: return []

    if mode == "down":
        ext_idx, ext_price = cur_ext_idx, lows[cur_ext_idx]
        for i in range(cur_ext_idx+1, n):
            if lows[i] < ext_price:
                ext_price, ext_idx = lows[i], i
            elif (highs[i] - ext_price) >= thr(ext_idx, ext_price):
                piv.append((ext_idx, lows[ext_idx], "trough")); mode = "up"; ext_idx, ext_price = i, highs[i]
    else:
        ext_idx, ext_price = cur_ext_idx, highs[cur_ext_idx]
        for i in range(cur_ext_idx+1, n):
            if highs[i] > ext_price:
                ext_price, ext_idx = highs[i], i
            elif (ext_price - lows[i]) >= thr(ext_idx, ext_price):
                piv.append((ext_idx, highs[ext_idx], "peak")); mode = "down"; ext_idx, ext_price = i, lows[i]

    terminal_label = "peak" if mode == "up" else "trough"
    term_price = df["High"].iloc[ext_idx] if terminal_label == "peak" else df["Low"].iloc[ext_idx]
    piv.append((ext_idx, float(term_price), terminal_label))

    piv = sorted(piv, key=lambda x: x[0])
    cleaned = []
    for idx, price, lab in piv:
        if not cleaned: cleaned.append((idx, price, lab)); continue
        pidx, pprice, plab = cleaned[-1]
        if lab == plab:
            if (lab == "peak" and price >= pprice) or (lab == "trough" and price <= pprice):
                cleaned[-1] = (idx, price, lab)
        else:
            cleaned.append((idx, price, lab))
    return cleaned

def neckline_value_at(k: int, p1_idx: int, p1_val: float, p2_idx: int, p2_val: float) -> float:
    if p2_idx == p1_idx: return p2_val
    return p1_val + (p2_val - p1_val) * (k - p1_idx) / (p2_idx - p1_idx)

# ------------------------- Candidate detection (relaxed geometry) ---------------

def detect_w_candidates(df_in: pd.DataFrame, series: np.ndarray) -> List[Dict]:
    df = add_indicators(df_in)
    dates = df.index.to_list()
    n = series.size
    if n < 30: return []

    piv = zigzag_pivots_hl(df, PCT_ZZ, ATR_MULT)
    if len(piv) < 5: return []

    trough_idx = [i for i,_,lab in piv if lab == "trough"]
    peak_idx   = [i for i,_,lab in piv if lab == "peak"]
    pivd = {i:(i,px,lab) for i,px,lab in piv}

    out = []
    for i1 in trough_idx:
        for i2 in trough_idx:
            if i2 <= i1: continue
            sep = i2 - i1
            if not (MIN_SEP <= sep <= MAX_SEP): continue

            mids = [p for p in peak_idx if i1 < p < i2]
            if not mids: continue
            p2 = max(mids, key=lambda x: pivd[x][1])

            befores = [p for p in peak_idx if p < i1]
            if not befores: continue
            p1 = max(befores)
            if not (p1 < i1 < p2 < i2): continue

            # Effective pivot prices
            l1_eff  = effective_trough_price(df, i1)
            l2_eff  = effective_trough_price(df, i2)
            pk1_eff = effective_peak_price(df, p1)
            pk2_eff = effective_peak_price(df, p2)

            # Net-direction checks (very relaxed)
            if (l1_eff / max(1e-9, pk1_eff) - 1.0) > -PREDROP_MIN:  # ≥ ~2% drop into L1
                continue
            if (pk2_eff / max(1e-9, l1_eff) - 1.0) <= 0:            # rise to P2
                continue
            if (l2_eff / max(1e-9, pk2_eff) - 1.0) >= 0:            # drop to L2
                continue

            # Trough similarity
            trough_diff = abs(l2_eff - l1_eff) / max(1e-9, l1_eff)
            if not (TROUGH_SIM_LO <= trough_diff <= TROUGH_SIM_HI): continue

            # Neckline slope
            slope_per_day = (pk2_eff - pk1_eff) / max(1, (p2 - p1))

            # Breakout check for status (optional)
            after_peaks = [p for p in peak_idx if p > i2]
            breakout_idx = None
            for p3 in after_peaks:
                line_val = neckline_value_at(p3, p1, pk1_eff, p2, pk2_eff)
                p3_eff   = effective_peak_price(df, p3)
                if series[p3] >= line_val * (1.0 + BREAKOUT_BUFFER) or \
                   p3_eff     >= line_val * (1.0 + BREAKOUT_BUFFER):
                    breakout_idx = p3
                    break
            status = "COMPLETED" if breakout_idx is not None else "FORMING"

            # Geometry/timing features (normalized)
            base = max(1e-9, min(l1_eff, l2_eff))
            drop1 = (l1_eff/base) - (pk1_eff/base)
            rise1 = (pk2_eff/base) - (l1_eff/base)
            drop2 = (l2_eff/base) - (pk2_eff/base)
            spread_t = (i2 - i1) / float(RECENT_WINDOW)
            p1_to_l1 = (i1 - p1) / float(RECENT_WINDOW)
            l1_to_p2 = (p2 - i1) / float(RECENT_WINDOW)
            p2_to_l2 = (i2 - p2) / float(RECENT_WINDOW)

            # Indicator features at P2 and L2
            feats_p2 = indicator_features_at(df, p2, "p2")
            feats_l2 = indicator_features_at(df, i2, "l2")

            out.append({
                "ticker": None, "price_mode": None,
                "p1_idx": p1, "low1_idx": i1, "p2_idx": p2, "low2_idx": i2, "p3_idx": breakout_idx,
                "p1_date": dates[p1].date().isoformat(),
                "low1_date": dates[i1].date().isoformat(),
                "p2_date": dates[p2].date().isoformat(),
                "low2_date": dates[i2].date().isoformat(),
                "p3_date": dates[breakout_idx].date().isoformat() if breakout_idx is not None else "",
                "p1_price": float(pk1_eff), "low1_price": float(l1_eff),
                "p2_price": float(pk2_eff), "low2_price": float(l2_eff),
                "p3_price": float(effective_peak_price(df, breakout_idx)) if breakout_idx is not None else np.nan,
                "status": status,

                # default labels (to be filled if exporting)
                "y_shape": "", "y_breakout": "", "y_reachtarget": "",

                # features
                "drop1_norm": float(drop1),
                "rise1_norm": float(rise1),
                "drop2_norm": float(drop2),
                "neckline_slope_per_day": float(slope_per_day),
                "trough_sim": float(trough_diff),
                "spread_t": float(spread_t),
                "p1_to_l1_t": float(p1_to_l1),
                "l1_to_p2_t": float(l1_to_p2),
                "p2_to_l2_t": float(p2_to_l2),
                **feats_p2,
                **feats_l2,
            })
    return out

# ------------------------- Plotting (optional) ---------------------------------

def plot_candidate(df: pd.DataFrame, rec: Dict, ticker: str, outdir: str, series: np.ndarray, title_suffix: str):
    os.makedirs(outdir, exist_ok=True)
    dates = pd.to_datetime(df.index)
    n = series.size

    p1 = rec["p1_idx"]; l1 = rec["low1_idx"]; p2 = rec["p2_idx"]; l2 = rec["low2_idx"]; p3 = rec.get("p3_idx")
    p1_eff = rec["p1_price"]; p2_eff = rec["p2_price"]
    neck_vals = np.array([neckline_value_at(k, p1, p1_eff, p2, p2_eff) for k in range(n)])

    plt.figure(figsize=(10,5))
    plt.plot(dates, df["Close"].to_numpy(), linewidth=1.0, alpha=0.5, label="Close")
    plt.plot(dates, series, linewidth=1.8, label="Detection series")
    plt.plot(dates, neck_vals, "--", linewidth=1.2, label="Neckline (effective P1→P2)")

    plt.scatter(dates[[p1, l1, p2, l2]], series[[p1, l1, p2, l2]], s=48, marker="D", label="P1/L1/P2/L2")
    plt.annotate("P1", (dates[p1], series[p1]), xytext=(5, 8), textcoords="offset points")
    plt.annotate("L1", (dates[l1], series[l1]), xytext=(5, -12), textcoords="offset points")
    plt.annotate("P2", (dates[p2], series[p2]), xytext=(5, 8), textcoords="offset points")
    plt.annotate("L2", (dates[l2], series[l2]), xytext=(5, -12), textcoords="offset points")

    if p3 is not None and not np.isnan(p3):
        plt.scatter(dates[p3], series[p3], s=64, marker="^", label="P3")
        plt.annotate("P3", (dates[p3], series[p3]), xytext=(5, 8), textcoords="offset points")

    title = f"{ticker} — {rec['status']} {title_suffix}"
    plt.title(title)
    plt.legend(loc="best"); plt.grid(True, alpha=0.3)
    fname = f"{ticker}_{rec['low1_date']}_{rec['low2_date']}_{rec['status']}.png".replace(":","-")
    fpath = os.path.join(outdir, fname)
    plt.tight_layout(); plt.savefig(fpath, dpi=160); plt.close()
    return fpath

# ------------------------- TRAINING --------------------------------------------

def _train_model(df: pd.DataFrame, target_col: str, feature_cols: List[str], model_path: str,
                 filter_ws_for_outcomes: bool) -> bool:
    data = df.copy()

    # For outcome models, keep only rows that are Ws per label (y_shape==1)
    if filter_ws_for_outcomes and "y_shape" in data.columns:
        data = data[data["y_shape"].fillna(0).astype(int) == 1].copy()

    # Target labels must be 0/1
    if target_col not in data.columns:
        print(f"[train:{target_col}] Missing column '{target_col}'. Skipping.")
        return False
    data = data[data[target_col].isin([0,1])].copy()
    if data.empty:
        print(f"[train:{target_col}] No rows with {target_col} in {{0,1}}. Skipping.")
        return False

    # Features presence
    for col in feature_cols:
        if col not in data.columns:
            raise ValueError(f"[train:{target_col}] Missing feature '{col}'. Did you run yfslope1.py augment?")

    X = data[feature_cols].astype(float).fillna(0.0).to_numpy()
    y = data[target_col].astype(int).to_numpy()

    n_pos = int(y.sum()); n_neg = int(len(y) - y.sum())
    if n_pos < 5 or n_neg < 5:
        print(f"[train:{target_col}] Need >=5 of each class. Got {n_pos} / {n_neg}. Skipping.")
        return False

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in kf.split(X, y):
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:,1]
        auc = roc_auc_score(y[te], proba)
        aucs.append(auc)
    print(f"[train:{target_col}] 5-fold AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f} (N={len(y)})")

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    dump(model, model_path)
    with open(FEATURES_FILE, "w") as f:
        json.dump(feature_cols, f)
    print(f"[train:{target_col}] saved model → {model_path}")
    return True

def subcmd_train(args):
    df = pd.read_csv(args.csv)

    targets = []
    if args.target in ("shape", "all"):
        targets.append(("y_shape", MODEL_SHAPE, False))
    if args.target in ("breakout", "all"):
        targets.append(("y_breakout", MODEL_BREAKOUT, True))
    if args.target in ("reachtarget", "all"):
        targets.append(("y_reachtarget", MODEL_REACHTARGET, True))

    if not targets:
        print("No targets selected.")
        return

    any_trained = False
    for col, path, filter_ws in targets:
        ok = _train_model(df, col, FEATURE_COLUMNS, path, filter_ws_for_outcomes=filter_ws)
        any_trained = any_trained or ok
    if not any_trained:
        print("[train] Nothing was trained (insufficient labeled data?).")

# ------------------------- SCANNING --------------------------------------------

def subcmd_scan_shape(args):
    if not os.path.exists(MODEL_SHAPE):
        print("Missing shape model. Train first:\n  python yfcsv1.py train --csv w_with_features.csv --target shape")
        return
    model_s = load(MODEL_SHAPE)
    with open(FEATURES_FILE, "r") as f:
        feats = json.load(f)

    tickers = get_sp500_tickers(args.limit)
    rows = []
    print(f"[scan-shape] scoring P(W) for {len(tickers)} tickers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(robust_history, t): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]; df0 = fut.result()
            if df0 is None or df0.empty:
                print(f"[{t}] no data"); continue
            df = add_indicators(df0)

            any_hit = False
            for mode in PRICE_MODE_PRIORITY:
                series = build_series(df, mode)
                cand = detect_w_candidates(df, series)
                if not cand:
                    continue
                for c in cand:
                    c["ticker"] = t; c["price_mode"] = mode
                    # compute model inputs
                    X = np.array([[float(c.get(k, 0.0)) for k in feats]])
                    p_shape = float(model_s.predict_proba(X)[0,1])
                    c["ml_shape_proba"] = p_shape
                    if p_shape >= args.shape_threshold:
                        rows.append(c)
                        any_hit = True
                        if args.plot:
                            try:
                                plot_candidate(df, c, t, args.plot_dir, series, f"(P(W)={p_shape:.2f}, mode {mode})")
                            except Exception:
                                pass
                if any_hit:
                    break

    if not rows:
        print("[scan-shape] no W candidates above threshold.")
        return

    cols = [
        "ticker","price_mode","ml_shape_proba",
        "p1_date","p1_price","low1_date","low1_price","p2_date","p2_price","low2_date","low2_price","p3_date","p3_price",
        "status", *FEATURE_COLUMNS
    ]
    df_out = pd.DataFrame(rows)
    for c in cols:
        if c not in df_out.columns: df_out[c] = np.nan
    df_out = df_out[cols].sort_values("ml_shape_proba", ascending=False).reset_index(drop=True)
    out_name = args.out_csv or "w_shape_hits.csv"
    df_out.to_csv(out_name, index=False)
    print(f"[scan-shape] wrote {out_name} ({len(df_out)} rows)")
    if args.plot:
        print(f"[scan-shape] plots in: {os.path.abspath(args.plot_dir)}")

def subcmd_scan(args):
    # Load models if available
    have_shape    = os.path.exists(MODEL_SHAPE)
    have_breakout = os.path.exists(MODEL_BREAKOUT)
    have_target   = os.path.exists(MODEL_REACHTARGET)
    if not (have_breakout or have_target or have_shape):
        print("No trained models found. Train at least one target first.")
        return

    model_s = load(MODEL_SHAPE) if have_shape else None
    model_b = load(MODEL_BREAKOUT) if have_breakout else None
    model_t = load(MODEL_REACHTARGET) if have_target else None
    with open(FEATURES_FILE, "r") as f:
        feats = json.load(f)

    tickers = get_sp500_tickers(args.limit)
    rows = []
    print(f"[scan] scoring {len(tickers)} tickers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(robust_history, t): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]; df0 = fut.result()
            if df0 is None or df0.empty:
                print(f"[{t}] no data"); continue
            df = add_indicators(df0)

            for mode in PRICE_MODE_PRIORITY:
                series = build_series(df, mode)
                cand = detect_w_candidates(df, series)
                if not cand:
                    continue
                for c in cand:
                    c["ticker"] = t; c["price_mode"] = mode
                    X = np.array([[float(c.get(k, 0.0)) for k in feats]])

                    # Stage 1 filter (if model present and enabled)
                    if model_s is not None:
                        p_shape = float(model_s.predict_proba(X)[0,1])
                        c["ml_shape_proba"] = p_shape
                        if args.shape_filter and p_shape < args.shape_threshold:
                            continue

                    # Stage 2 predictions
                    if model_b is not None:
                        c["ml_breakout_proba"] = float(model_b.predict_proba(X)[0,1])
                    if model_t is not None:
                        c["ml_reachtarget_proba"] = float(model_t.predict_proba(X)[0,1])

                    rows.append(c)
                    if args.plot:
                        pmax = max(
                            c.get("ml_reachtarget_proba", 0.0),
                            c.get("ml_breakout_proba", 0.0),
                            c.get("ml_shape_proba", 0.0),
                        )
                        if pmax >= args.plot_threshold:
                            try:
                                plot_candidate(df, c, t, args.plot_dir, series, f"(p≈{pmax:.2f}, mode {mode})")
                            except Exception:
                                pass
                break  # keep first mode with any hits

    if not rows:
        print("[scan] no candidates to score.")
        return

    base_cols = [
        "ticker","price_mode",
        "p1_date","p1_price","low1_date","low1_price","p2_date","p2_price","low2_date","low2_price","p3_date","p3_price",
        "status", *FEATURE_COLUMNS
    ]
    proba_cols = []
    if have_shape:    proba_cols.append("ml_shape_proba")
    if have_breakout: proba_cols.append("ml_breakout_proba")
    if have_target:   proba_cols.append("ml_reachtarget_proba")

    cols = base_cols + proba_cols
    df_out = pd.DataFrame(rows)
    for c in cols:
        if c not in df_out.columns: df_out[c] = np.nan
    df_out = df_out[cols]

    # Sort by best available signal
    sort_cols = []
    if have_shape:    sort_cols.append("ml_shape_proba")
    if have_breakout: sort_cols.append("ml_breakout_proba")
    if have_target:   sort_cols.append("ml_reachtarget_proba")
    if sort_cols:
        df_out = df_out.sort_values(sort_cols, ascending=False)
    df_out = df_out.reset_index(drop=True)

    out_name = args.out_csv or "w_model_detections.csv"
    df_out.to_csv(out_name, index=False)
    print(f"[scan] wrote {out_name} ({len(df_out)} rows)")
    if args.plot:
        print(f"[scan] plots in: {os.path.abspath(args.plot_dir)}")

# ------------------------- propose (optional labeling set) ---------------------

def subcmd_propose(args):
    tickers = get_sp500_tickers(args.limit)
    rows = []
    print(f"[propose] scanning {len(tickers)} tickers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(robust_history, t): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]; df0 = fut.result()
            if df0 is None or df0.empty:
                print(f"[{t}] no data"); continue
            df = add_indicators(df0)

            hit_any = False
            for mode in PRICE_MODE_PRIORITY:
                series = build_series(df, mode)
                cand = detect_w_candidates(df, series)
                if cand:
                    for c in cand:
                        c["ticker"] = t; c["price_mode"] = mode
                        rows.append(c)
                        if args.plot:
                            try:
                                plot_candidate(df, c, t, args.plot_dir, series, f"(mode {mode})")
                            except Exception:
                                pass
                    hit_any = True
                    break
            if not hit_any and args.debug:
                print(f"[{t}] no W-like candidates")

    if not rows:
        print("No candidates found."); return

    cols = [
        "ticker","price_mode",
        "p1_date","p1_price","low1_date","low1_price","p2_date","p2_price","low2_date","low2_price","p3_date","p3_price",
        "status","y_shape","y_breakout","y_reachtarget",
        *FEATURE_COLUMNS,
        "p1_idx","low1_idx","p2_idx","low2_idx","p3_idx"
    ]
    df_out = pd.DataFrame(rows)
    for c in cols:
        if c not in df_out.columns: df_out[c] = np.nan
    df_out = df_out[cols]
    out_name = args.out_csv or "w_candidates.csv"
    df_out.to_csv(out_name, index=False)
    print(f"[propose] wrote {out_name} ({len(df_out)} rows)")
    if args.plot:
        print(f"[propose] plots in: {os.path.abspath(args.plot_dir)}")

# ------------------------- CLI -------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Two-stage W-pattern pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prop = sub.add_parser("propose", help="Generate candidate CSV for labeling (geometry+indicators)")
    p_prop.add_argument("--limit", type=int, default=50)
    p_prop.add_argument("--out", dest="out_csv", default="w_candidates.csv")
    p_prop.add_argument("--plot", action="store_true")
    p_prop.add_argument("--plot-dir", default=PLOTS_DIR)
    p_prop.add_argument("--debug", action="store_true")
    p_prop.set_defaults(func=subcmd_propose)

    p_tr = sub.add_parser("train", help="Train model(s) from labeled CSV (uses pre-breakout features)")
    p_tr.add_argument("--csv", required=True, help="Path to labeled CSV (e.g., w_with_features.csv)")
    p_tr.add_argument("--target", choices=["shape","breakout","reachtarget","all"], default="all",
                      help="Which target(s) to train")
    p_tr.set_defaults(func=subcmd_train)

    p_ss = sub.add_parser("scan-shape", help="Score P(W) and list W-shaped candidates")
    p_ss.add_argument("--limit", type=int, default=50)
    p_ss.add_argument("--out", dest="out_csv", default="w_shape_hits.csv")
    p_ss.add_argument("--shape-threshold", type=float, default=0.50, help="Keep if P(W) >= threshold")
    p_ss.add_argument("--plot", action="store_true")
    p_ss.add_argument("--plot-dir", default=PLOTS_DIR)
    p_ss.set_defaults(func=subcmd_scan_shape)

    p_sc = sub.add_parser("scan", help="Full scan (optionally gated by shape model) and outcome scoring")
    p_sc.add_argument("--limit", type=int, default=50)
    p_sc.add_argument("--out", dest="out_csv", default="w_model_detections.csv")
    p_sc.add_argument("--shape-filter", action="store_true", help="Use shape model to filter candidates")
    p_sc.add_argument("--shape-threshold", type=float, default=0.50)
    p_sc.add_argument("--plot", action="store_true")
    p_sc.add_argument("--plot-dir", default=PLOTS_DIR)
    p_sc.add_argument("--plot-threshold", type=float, default=0.65)
    p_sc.set_defaults(func=subcmd_scan)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
