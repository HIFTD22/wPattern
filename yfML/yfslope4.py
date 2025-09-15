#!/usr/bin/env python3
"""
yfslope1.py  —  Append neckline slope + indicators + motif/extra features to a labeled W-pattern CSV.

Input CSV (minimum):
  ticker,
  p1_date,p1_price,
  low1_date,low1_price,
  p2_date,p2_price,
  low2_date,low2_price,
  y_shape, y_breakout (optional), y_reachtarget (optional)

Output: original columns + the following new ones:
  p1_trading_date,p2_trading_date,low1_trading_date,low2_trading_date
  p1_idx,p2_idx,low1_idx,low2_idx,p1p2_trading_days
  neckline_slope_per_day
  drop1_norm,rise1_norm,drop2_norm,trough_sim,spread_t,p1_to_l1_t,l1_to_p2_t,p2_to_l2_t
  rsi_p2,rsi_l2,vol_ratio_p2,vol_ratio_l2,px_vs_ma20_p2,px_vs_ma20_l2,px_vs_ma50_l2
  atr_pct_l2, dist_to_neckline_l2, mom10_to_l2
  mp_min_dist  (Matrix Profile motif distance, lower ⇒ more W-like)

Usage:
  python yfslope1.py --in w_data.csv --out w_with_features.csv --tolerance-days 3
"""

import argparse
import sys
from datetime import datetime, timedelta, date
from typing import Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf

# Optional deps for motif and RSI
try:
    import stumpy
    HAS_STUMPY = True
except Exception:
    HAS_STUMPY = False

RECENT_WINDOW = 90  # used to normalize timing features

# --------------------- Utilities ---------------------

def norm_symbol(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")

def parse_ymd(s: str) -> Optional[date]:
    if pd.isna(s) or str(s).strip() == "":
        return None
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()
    except Exception:
        return None

def robust_history_range(ticker: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    # Pad for MA/RSI/ATR history
    start_pad = (start_date or end_date) - timedelta(days=220)
    end_pad   = (end_date or start_date) + timedelta(days=5)
    try:
        df = yf.Ticker(ticker).history(
            start=start_pad.isoformat(),
            end=(end_pad + timedelta(days=6)).isoformat(),
            interval="1d",
            auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        need = ["Open","High","Low","Close","Volume"]
        if not all(c in df.columns for c in need):
            return None
        return df[need].dropna().sort_index()
    except Exception:
        return None

def nearest_trading_index(df: pd.DataFrame, target: date, tolerance_days: int = 3) -> Optional[int]:
    if df is None or df.empty or target is None:
        return None
    dates = np.array([d.date() for d in df.index])
    diffs = np.array([abs((d - target).days) for d in dates])
    j = int(diffs.argmin())
    return j if diffs[j] <= tolerance_days else None

def rsi_ewm(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def atr_series(df: pd.DataFrame, n: int = 14) -> pd.Series:
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

def effective_peak_price(df: pd.DataFrame, idx: int) -> float:
    o = float(df["Open"].iloc[idx]); c = float(df["Close"].iloc[idx]); h = float(df["High"].iloc[idx])
    return float(max(h, o, c))

def effective_trough_price(df: pd.DataFrame, idx: int) -> float:
    o = float(df["Open"].iloc[idx]); c = float(df["Close"].iloc[idx]); l = float(df["Low"].iloc[idx])
    return float(min(l, o, c))

def neckline_value_at(k: int, p1_idx: int, p1_val: float, p2_idx: int, p2_val: float) -> float:
    if p2_idx == p1_idx: return p2_val
    return p1_val + (p2_val - p1_val) * (k - p1_idx) / (p2_idx - p1_idx)

def matrix_profile_min_distance(series: np.ndarray, m: int = 30) -> float:
    """Lower ⇒ there exists a strong repeated motif of length m in the window.
    Acts as a generic 'W-ish motif' proxy without needing a template."""
    if not HAS_STUMPY:
        return np.nan
    a = np.asarray(series, dtype=float)
    if a.size < m + 2:
        return np.nan
    try:
        mp = stumpy.stump(a, m)[:, 0]  # profile values (z-norm distances)
        return float(np.nanmin(mp))
    except Exception:
        return np.nan

# --------------------- Row computation ---------------------

def compute_row(row: pd.Series, tolerance_days: int) -> pd.Series:
    out = row.copy()

    ticker = norm_symbol(row.get("ticker", ""))
    p1_date = parse_ymd(row.get("p1_date")); p2_date = parse_ymd(row.get("p2_date"))
    l1_date = parse_ymd(row.get("low1_date")); l2_date = parse_ymd(row.get("low2_date"))

    try:
        p1_price = float(row.get("p1_price")); p2_price = float(row.get("p2_price"))
        l1_price = float(row.get("low1_price")); l2_price = float(row.get("low2_price"))
    except Exception:
        return out

    # init outputs
    for k in ["p1_trading_date","p2_trading_date","low1_trading_date","low2_trading_date"]:
        out[k] = ""
    for k in ["p1_idx","p2_idx","low1_idx","low2_idx","p1p2_trading_days",
              "neckline_slope_per_day","drop1_norm","rise1_norm","drop2_norm",
              "trough_sim","spread_t","p1_to_l1_t","l1_to_p2_t","p2_to_l2_t",
              "rsi_p2","rsi_l2","vol_ratio_p2","vol_ratio_l2","px_vs_ma20_p2","px_vs_ma20_l2","px_vs_ma50_l2",
              "atr_pct_l2","dist_to_neckline_l2","mom10_to_l2","mp_min_dist"]:
        out[k] = np.nan

    if not ticker or None in (p1_date,p2_date,l1_date,l2_date):
        return out

    d0 = min(p1_date,p2_date,l1_date,l2_date); d1 = max(p1_date,p2_date,l1_date,l2_date)
    df0 = robust_history_range(ticker, d0, d1)
    if df0 is None or df0.empty:
        return out

    df = add_indicators(df0)

    i_p1 = nearest_trading_index(df, p1_date, tolerance_days)
    i_p2 = nearest_trading_index(df, p2_date, tolerance_days)
    i_l1 = nearest_trading_index(df, l1_date, tolerance_days)
    i_l2 = nearest_trading_index(df, l2_date, tolerance_days)
    if None in (i_p1,i_p2,i_l1,i_l2):
        return out

    out["p1_trading_date"] = df.index[i_p1].date().isoformat()
    out["p2_trading_date"] = df.index[i_p2].date().isoformat()
    out["low1_trading_date"] = df.index[i_l1].date().isoformat()
    out["low2_trading_date"] = df.index[i_l2].date().isoformat()

    out["p1_idx"] = int(i_p1); out["p2_idx"] = int(i_p2)
    out["low1_idx"] = int(i_l1); out["low2_idx"] = int(i_l2)

    bars = abs(i_p2 - i_p1)
    out["p1p2_trading_days"] = int(bars)
    if bars > 0:
        out["neckline_slope_per_day"] = float((p2_price - p1_price) / bars)

    # geometry/timing (using YOUR labeled pivot prices; no post-breakout info)
    base = max(1e-9, min(l1_price, l2_price))
    out["drop1_norm"] = float((l1_price/base) - (p1_price/base))
    out["rise1_norm"] = float((p2_price/base) - (l1_price/base))
    out["drop2_norm"] = float((l2_price/base) - (p2_price/base))
    out["trough_sim"] = float(abs(l2_price - l1_price) / max(1e-9, l1_price))
    out["spread_t"] = float((i_l2 - i_l1) / RECENT_WINDOW)
    out["p1_to_l1_t"] = float((i_l1 - i_p1) / RECENT_WINDOW)
    out["l1_to_p2_t"] = float((i_p2 - i_l1) / RECENT_WINDOW)
    out["p2_to_l2_t"] = float((i_l2 - i_p2) / RECENT_WINDOW)

    # indicator features at P2 & L2
    feats_p2 = indicator_features_at(df, i_p2, "p2")
    feats_l2 = indicator_features_at(df, i_l2, "l2")
    out.update(feats_p2); out.update(feats_l2)

    # ATR% at L2
    try:
        atr_l2 = float(df["ATR14"].iloc[i_l2])
        c_l2 = float(df["Close"].iloc[i_l2])
        out["atr_pct_l2"] = (atr_l2 / c_l2) if c_l2 else np.nan
    except Exception:
        pass

    # Distance to neckline at L2 (relative)
    try:
        neck_l2 = neckline_value_at(i_l2, i_p1, p1_price, i_p2, p2_price)
        l2_eff = effective_trough_price(df, i_l2)  # per your rule
        out["dist_to_neckline_l2"] = (neck_l2 - l2_eff) / max(1e-9, l2_eff)
    except Exception:
        pass

    # Momentum into L2 (10 bars)
    try:
        if i_l2 >= 10:
            c_prev = float(df["Close"].iloc[i_l2-10])
            c_l2 = float(df["Close"].iloc[i_l2])
            out["mom10_to_l2"] = (c_l2 / c_prev - 1.0) if c_prev else np.nan
    except Exception:
        pass

    # Matrix Profile motif distance on Close over the whole window [last 90 bars around L2]
    try:
        end_idx = i_l2
        start_idx = max(0, end_idx - (RECENT_WINDOW - 1))
        close_win = df["Close"].iloc[start_idx:end_idx+1].to_numpy(dtype=float)
        out["mp_min_dist"] = matrix_profile_min_distance(close_win, m=30)
    except Exception:
        pass

    return out

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description="Append slope + indicators + motif features to W-pattern CSV")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV with P1/L1/P2/L2 (+ labels)")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output CSV path")
    ap.add_argument("--tolerance-days", type=int, default=3,
                    help="Max calendar-day distance to match your dates to trading days (default: 3)")
    args = ap.parse_args()

    df_in = pd.read_csv(args.in_csv)
    required = ["ticker","p1_date","p1_price","p2_date","p2_price","low1_date","low1_price","low2_date","low2_price"]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        print(f"ERROR: Input CSV missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for _, r in df_in.iterrows():
        rows.append(compute_row(r, args.tolerance_days))
    df_out = pd.DataFrame(rows)

    user_cols = list(df_in.columns)
    new_cols = [c for c in df_out.columns if c not in user_cols]
    df_out = df_out[user_cols + new_cols]
    df_out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} (rows: {len(df_out)})")

if __name__ == "__main__":
    main()
