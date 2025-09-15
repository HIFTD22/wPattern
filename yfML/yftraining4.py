#!/usr/bin/env python3
"""
yfcsv1.py — Two-stage W-pattern pipeline (with extended features)

Subcommands:
  - propose     : auto-generate candidates + plots (for labeling)
  - train       : (kept for convenience) train 'shape'/'breakout'/'reachtarget' using GradientBoosting
                  NOTE: for best models, use yftune.py and its saved .pkl instead.
  - scan-shape  : score P(W) using model_shape.pkl (or detections only if not present)
  - scan        : full scan with optional shape gate + outcome predictions

Install:
  pip install yfinance pandas numpy scikit-learn requests lxml matplotlib joblib
"""

import warnings
warnings.filterwarnings("ignore")

import os, json, argparse, sys, time
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import dump, load

# --------------------- Config ---------------------

RECENT_WINDOW = 90
MAX_WORKERS = 8
TIMEOUT_SECS = 12
MAX_RETRIES = 3
RETRY_SLEEP = 0.7

# Relaxed W thresholds
PCT_ZZ = 0.015
ATR_MULT = 0.5
ATR_LEN = 14
MIN_SEP = 3
MAX_SEP = 85
TROUGH_SIM_LO = 0.00
TROUGH_SIM_HI = 0.30   # slightly looser
PREDROP_MIN = 0.02
POST_L2_LIFT_MIN = -0.01
BREAKOUT_BUFFER = 0.001
FORMING_NEARLINE_TOL = 0.12

PRICE_MODE_PRIORITY = ["OCMID","CLOSE","OPEN","HL2","HLC3"]

MODEL_SHAPE       = "model_shape.pkl"
MODEL_BREAKOUT    = "model_breakout.pkl"
MODEL_REACHTARGET = "model_reachtarget.pkl"
FEATURES_FILE     = "feature_list.json"
PLOTS_DIR         = "plots"

# --------------------- Features ---------------------

FEATURE_COLUMNS = [
    # geometry + timing
    "drop1_norm","rise1_norm","drop2_norm",
    "neckline_slope_per_day","trough_sim",
    "spread_t","p1_to_l1_t","l1_to_p2_t","p2_to_l2_t",
    # indicators
    "rsi_p2","rsi_l2",
    "vol_ratio_p2","vol_ratio_l2",
    "px_vs_ma20_p2","px_vs_ma20_l2","px_vs_ma50_l2",
    # new extras
    "atr_pct_l2","dist_to_neckline_l2","mom10_to_l2","mp_min_dist",
]

# --------------------- Utils ---------------------

def norm_symbol(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")

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

def robust_history(ticker: str) -> Optional[pd.DataFrame]:
    for attempt in range(1, MAX_RETRIES+1):
        try:
            df = yf.Ticker(ticker).history(period="12mo", interval="1d", auto_adjust=True)
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
            time.sleep(RETRY_SLEEP); continue

def rsi_ewm(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def atr_series(df: pd.DataFrame, n: int = ATR_LEN) -> pd.Series:
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["VolMA20"] = df["Volume"].rolling(20, min_periods=1).mean()
    df["RSI14"] = rsi_ewm(df["Close"], 14)
    df["ATR14"] = atr_series(df, 14)
    return df

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

def effective_peak_price(df: pd.DataFrame, idx: int) -> float:
    o = float(df["Open"].iloc[idx]); c = float(df["Close"].iloc[idx]); h = float(df["High"].iloc[idx])
    return float(max(h, o, c))

def effective_trough_price(df: pd.DataFrame, idx: int) -> float:
    o = float(df["Open"].iloc[idx]); c = float(df["Close"].iloc[idx]); l = float(df["Low"].iloc[idx])
    return float(min(l, o, c))

def neckline_value_at(k: int, p1_idx: int, p1_val: float, p2_idx: int, p2_val: float) -> float:
    if p2_idx == p1_idx: return p2_val
    return p1_val + (p2_val - p1_val) * (k - p1_idx) / (p2_idx - p1_idx)

# --------------------- ZigZag pivots ---------------------

def atr(df: pd.DataFrame, n: int = ATR_LEN) -> pd.Series:
    return atr_series(df, n)

def zigzag_pivots_hl(df: pd.DataFrame, pct_th: float = PCT_ZZ, atr_mult: float = ATR_MULT):
    highs = df["High"].astype(float).to_numpy()
    lows  = df["Low"].astype(float).to_numpy()
    n = highs.size
    if n < 5: return []
    a = atr(df, ATR_LEN).to_numpy()

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

# --------------------- Candidate detection ---------------------

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

            # effective pivot prices
            l1_eff  = effective_trough_price(df, i1)
            l2_eff  = effective_trough_price(df, i2)
            pk1_eff = effective_peak_price(df, p1)
            pk2_eff = effective_peak_price(df, p2)

            if (l1_eff / max(1e-9, pk1_eff) - 1.0) > -PREDROP_MIN:  # drop into L1
                continue
            if (pk2_eff / max(1e-9, l1_eff) - 1.0) <= 0:            # rise to P2
                continue
            if (l2_eff / max(1e-9, pk2_eff) - 1.0) >= 0:            # drop to L2
                continue

            trough_diff = abs(l2_eff - l1_eff) / max(1e-9, l1_eff)
            if not (TROUGH_SIM_LO <= trough_diff <= TROUGH_SIM_HI): continue

            slope_per_day = (pk2_eff - pk1_eff) / max(1, (p2 - p1))

            # breakout check
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

            base = max(1e-9, min(l1_eff, l2_eff))
            drop1 = (l1_eff/base) - (pk1_eff/base)
            rise1 = (pk2_eff/base) - (l1_eff/base)
            drop2 = (l2_eff/base) - (pk2_eff/base)
            spread_t = (i2 - i1) / float(RECENT_WINDOW)
            p1_to_l1 = (i1 - p1) / float(RECENT_WINDOW)
            l1_to_p2 = (p2 - i1) / float(RECENT_WINDOW)
            p2_to_l2 = (i2 - p2) / float(RECENT_WINDOW)

            # indicator features
            feats_p2 = {
                "rsi_p2": float(df["RSI14"].iloc[p2]),
                "vol_ratio_p2": float(df["Volume"].iloc[p2] / max(1e-9, df["VolMA20"].iloc[p2] or np.nan)),
                "px_vs_ma20_p2": float(df["Close"].iloc[p2] / max(1e-9, df["MA20"].iloc[p2]) - 1.0) if df["MA20"].iloc[p2] else np.nan,
            }
            feats_l2_full = {
                "rsi_l2": float(df["RSI14"].iloc[i2]),
                "vol_ratio_l2": float(df["Volume"].iloc[i2] / max(1e-9, df["VolMA20"].iloc[i2] or np.nan)),
                "px_vs_ma20_l2": float(df["Close"].iloc[i2] / max(1e-9, df["MA20"].iloc[i2]) - 1.0) if df["MA20"].iloc[i2] else np.nan,
                "px_vs_ma50_l2": float(df["Close"].iloc[i2] / max(1e-9, df["MA50"].iloc[i2]) - 1.0) if df["MA50"].iloc[i2] else np.nan,
                "atr_pct_l2": float(df["ATR14"].iloc[i2] / max(1e-9, df["Close"].iloc[i2])),
            }

            # distance to neckline at L2
            neck_l2 = neckline_value_at(i2, p1, pk1_eff, p2, pk2_eff)
            dist_neck_l2 = (neck_l2 - l2_eff) / max(1e-9, l2_eff)

            # momentum into L2 (10 bars)
            if i2 >= 10:
                c_prev = float(df["Close"].iloc[i2-10])
                c_l2 = float(df["Close"].iloc[i2])
                mom10 = (c_l2 / c_prev - 1.0) if c_prev else np.nan
            else:
                mom10 = np.nan

            # matrix profile motif distance (Close)
            try:
                import stumpy
                start_idx = max(0, i2 - (RECENT_WINDOW - 1))
                close_win = df["Close"].iloc[start_idx:i2+1].to_numpy(dtype=float)
                mp_min = float(stumpy.stump(close_win, 30)[:,0].min()) if len(close_win) >= 32 else np.nan
            except Exception:
                mp_min = np.nan

            rec = {
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
                "y_shape": "", "y_breakout": "", "y_reachtarget": "",
                "drop1_norm": float(drop1),
                "rise1_norm": float(rise1),
                "drop2_norm": float(drop2),
                "neckline_slope_per_day": float(slope_per_day),
                "trough_sim": float(trough_diff),
                "spread_t": float(spread_t),
                "p1_to_l1_t": float(p1_to_l1),
                "l1_to_p2_t": float(l1_to_p2),
                "p2_to_l2_t": float(p2_to_l2),
                # indicators
                "rsi_p2": feats_p2["rsi_p2"],
                "rsi_l2": feats_l2_full["rsi_l2"],
                "vol_ratio_p2": feats_p2["vol_ratio_p2"],
                "vol_ratio_l2": feats_l2_full["vol_ratio_l2"],
                "px_vs_ma20_p2": feats_p2["px_vs_ma20_p2"],
                "px_vs_ma20_l2": feats_l2_full["px_vs_ma20_l2"],
                "px_vs_ma50_l2": feats_l2_full["px_vs_ma50_l2"],
                # new extras
                "atr_pct_l2": feats_l2_full["atr_pct_l2"],
                "dist_to_neckline_l2": float(dist_neck_l2),
                "mom10_to_l2": float(mom10),
                "mp_min_dist": float(mp_min),
            }
            out.append(rec)
    return out

# --------------------- Plotting ---------------------

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
    plt.plot(dates, neck_vals, "--", linewidth=1.2, label="Neckline (P1→P2 eff.)")
    plt.scatter(dates[[p1, l1, p2, l2]], series[[p1, l1, p2, l2]], s=48, marker="D", label="P1/L1/P2/L2")
    if p3 is not None and not np.isnan(p3):
        plt.scatter(dates[p3], series[p3], s=64, marker="^", label="P3")
    title = f"{ticker} — {rec['status']} {title_suffix}"
    plt.title(title); plt.legend(loc="best"); plt.grid(True, alpha=0.3)
    fname = f"{ticker}_{rec['low1_date']}_{rec['low2_date']}_{rec['status']}.png".replace(":","-")
    fpath = os.path.join(outdir, fname)
    plt.tight_layout(); plt.savefig(fpath, dpi=160); plt.close()
    return fpath

# --------------------- TRAIN (baseline GB) ---------------------

def _train_model(df: pd.DataFrame, target_col: str, model_path: str):
    data = df.copy()
    if target_col in ("y_breakout","y_reachtarget") and "y_shape" in data.columns:
        data = data[data["y_shape"].fillna(0).astype(int) == 1].copy()
    data = data[data[target_col].isin([0,1])].copy()
    if data.empty:
        print(f"[train:{target_col}] No rows with {target_col} in {{0,1}}."); return False
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            raise ValueError(f"[train:{target_col}] Missing feature '{col}'. Run yfslope1.py first.")

    X = data[FEATURE_COLUMNS].astype(float).fillna(0.0).to_numpy()
    y = data[target_col].astype(int).to_numpy()

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in kf.split(X, y):
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], proba))
    print(f"[train:{target_col}] 5-fold AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f} (N={len(y)})")

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    dump(model, model_path)
    with open(FEATURES_FILE, "w") as f:
        json.dump(FEATURE_COLUMNS, f)
    print(f"[train:{target_col}] saved → {model_path}")
    return True

# --------------------- SCAN / PROPOSE ---------------------

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
            hit = False
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
                    hit = True; break
            if not hit and args.debug: print(f"[{t}] no W-like candidates")
    if not rows:
        print("No candidates found."); return
    cols = ["ticker","price_mode",
            "p1_date","p1_price","low1_date","low1_price","p2_date","p2_price","low2_date","low2_price","p3_date","p3_price",
            "status","y_shape","y_breakout","y_reachtarget", *FEATURE_COLUMNS,
            "p1_idx","low1_idx","p2_idx","low2_idx","p3_idx"]
    df_out = pd.DataFrame(rows)
    for c in cols:
        if c not in df_out.columns: df_out[c] = np.nan
    df_out = df_out[cols]
    out_name = args.out_csv or "w_candidates.csv"
    df_out.to_csv(out_name, index=False)
    print(f"[propose] wrote {out_name} ({len(df_out)} rows)")
    if args.plot: print(f"[propose] plots in: {os.path.abspath(args.plot_dir)}")

def subcmd_train(args):
    df = pd.read_csv(args.csv)
    any_trained = False
    if args.target in ("shape","all"):
        any_trained |= _train_model(df.copy(), "y_shape", MODEL_SHAPE)
    if args.target in ("breakout","all"):
        any_trained |= _train_model(df.copy(), "y_breakout", MODEL_BREAKOUT)
    if args.target in ("reachtarget","all"):
        any_trained |= _train_model(df.copy(), "y_reachtarget", MODEL_REACHTARGET)
    if not any_trained:
        print("[train] Nothing trained.")

def subcmd_scan_shape(args):
    have_shape = os.path.exists(MODEL_SHAPE)
    model_s = load(MODEL_SHAPE) if have_shape else None
    feats = FEATURE_COLUMNS
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
            for mode in PRICE_MODE_PRIORITY:
                series = build_series(df, mode)
                cand = detect_w_candidates(df, series)
                if not cand: continue
                for c in cand:
                    c["ticker"] = t; c["price_mode"] = mode
                    if model_s is not None:
                        X = np.array([[float(c.get(k, 0.0)) for k in feats]])
                        p_shape = float(getattr(model_s, "predict_proba", lambda X: np.array([[0,0]]))(X)[:,1][0])
                        c["ml_shape_proba"] = p_shape
                        if p_shape >= args.shape_threshold:
                            rows.append(c)
                            if args.plot:
                                try: plot_candidate(df, c, t, args.plot_dir, series, f"(P(W)={p_shape:.2f}, {mode})")
                                except Exception: pass
                    else:
                        rows.append(c)
                break
    if not rows:
        print("[scan-shape] no W candidates above threshold."); return
    cols = ["ticker","price_mode","ml_shape_proba",
            "p1_date","p1_price","low1_date","low1_price","p2_date","p2_price","low2_date","low2_price","p3_date","p3_price",
            "status", *FEATURE_COLUMNS]
    df_out = pd.DataFrame(rows)
    for c in cols:
        if c not in df_out.columns: df_out[c] = np.nan
    df_out = df_out[cols].sort_values("ml_shape_proba", ascending=False).reset_index(drop=True)
    out_name = args.out_csv or "w_shape_hits.csv"
    df_out.to_csv(out_name, index=False)
    print(f"[scan-shape] wrote {out_name} ({len(df_out)} rows)")
    if args.plot: print(f"[scan-shape] plots in: {os.path.abspath(args.plot_dir)}")

def subcmd_scan(args):
    have_shape    = os.path.exists(MODEL_SHAPE)
    have_breakout = os.path.exists(MODEL_BREAKOUT)
    have_target   = os.path.exists(MODEL_REACHTARGET)

    model_s = load(MODEL_SHAPE) if have_shape else None
    model_b = load(MODEL_BREAKOUT) if have_breakout else None
    model_t = load(MODEL_REACHTARGET) if have_target else None
    feats = FEATURE_COLUMNS

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
                if not cand: continue
                for c in cand:
                    c["ticker"] = t; c["price_mode"] = mode
                    X = np.array([[float(c.get(k, 0.0)) for k in feats]])
                    if model_s is not None and args.shape_filter:
                        p_shape = float(getattr(model_s, "predict_proba", lambda X: np.array([[0,0]]))(X)[:,1][0])
                        c["ml_shape_proba"] = p_shape
                        if p_shape < args.shape_threshold:
                            continue
                    if model_b is not None:
                        c["ml_breakout_proba"] = float(getattr(model_b, "predict_proba")(X)[0,1])
                    if model_t is not None:
                        c["ml_reachtarget_proba"] = float(getattr(model_t, "predict_proba")(X)[0,1])
                    rows.append(c)
                    if args.plot:
                        pmax = max(c.get("ml_reachtarget_proba",0), c.get("ml_breakout_proba",0), c.get("ml_shape_proba",0))
                        if pmax >= args.plot_threshold:
                            try: plot_candidate(df, c, t, args.plot_dir, series, f"(p≈{pmax:.2f}, {mode})")
                            except Exception: pass
                break

    if not rows:
        print("[scan] no candidates to score."); return
    base_cols = ["ticker","price_mode",
                 "p1_date","p1_price","low1_date","low1_price","p2_date","p2_price","low2_date","low2_price","p3_date","p3_price",
                 "status", *FEATURE_COLUMNS]
    proba_cols = []
    if have_shape:    proba_cols.append("ml_shape_proba")
    if have_breakout: proba_cols.append("ml_breakout_proba")
    if have_target:   proba_cols.append("ml_reachtarget_proba")
    cols = base_cols + proba_cols

    df_out = pd.DataFrame(rows)
    for c in cols:
        if c not in df_out.columns: df_out[c] = np.nan
    df_out = df_out[cols]
    sort_cols = [c for c in ["ml_shape_proba","ml_breakout_proba","ml_reachtarget_proba"] if c in df_out.columns]
    if sort_cols: df_out = df_out.sort_values(sort_cols, ascending=False)
    df_out = df_out.reset_index(drop=True)

    out_name = args.out_csv or "w_model_detections.csv"
    df_out.to_csv(out_name, index=False)
    print(f"[scan] wrote {out_name} ({len(df_out)} rows)")
    if args.plot: print(f"[scan] plots in: {os.path.abspath(args.plot_dir)}")

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description="Two-stage W-pattern pipeline (extended features)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prop = sub.add_parser("propose", help="Generate candidates (for labeling)")
    p_prop.add_argument("--limit", type=int, default=50)
    p_prop.add_argument("--out", dest="out_csv", default="w_candidates.csv")
    p_prop.add_argument("--plot", action="store_true")
    p_prop.add_argument("--plot-dir", default=PLOTS_DIR)
    p_prop.add_argument("--debug", action="store_true")
    p_prop.set_defaults(func=subcmd_propose)

    p_tr = sub.add_parser("train", help="(baseline) train shape/breakout/reachtarget with GB")
    p_tr.add_argument("--csv", required=True)
    p_tr.add_argument("--target", choices=["shape","breakout","reachtarget","all"], default="all")
    p_tr.set_defaults(func=subcmd_train)

    p_ss = sub.add_parser("scan-shape", help="Score P(W) from model_shape.pkl")
    p_ss.add_argument("--limit", type=int, default=50)
    p_ss.add_argument("--out", dest="out_csv", default="w_shape_hits.csv")
    p_ss.add_argument("--shape-threshold", type=float, default=0.50)
    p_ss.add_argument("--plot", action="store_true")
    p_ss.add_argument("--plot-dir", default=PLOTS_DIR)
    p_ss.set_defaults(func=subcmd_scan_shape)

    p_sc = sub.add_parser("scan", help="Full scan (use shape gate if available)")
    p_sc.add_argument("--limit", type=int, default=50)
    p_sc.add_argument("--out", dest="out_csv", default="w_model_detections.csv")
    p_sc.add_argument("--shape-filter", action="store_true")
    p_sc.add_argument("--shape-threshold", type=float, default=0.50)
    p_sc.add_argument("--plot", action="store_true")
    p_sc.add_argument("--plot-dir", default=PLOTS_DIR)
    p_sc.add_argument("--plot-threshold", type=float, default=0.65)
    p_sc.set_defaults(func=subcmd_scan)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
