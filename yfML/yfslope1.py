#!/usr/bin/env python3
"""
Append neckline slope + indicator features to a labeled W-pattern CSV.

Input CSV (minimum):
  ticker,
  p1_date,p1_price,
  low1_date,low1_price,
  p2_date,p2_price,
  low2_date,low2_price,
  y_shape,y_breakout
Optional extras (passed through):
  y_reachtarget

Output: original columns + the following new ones:
  p1_trading_date,p2_trading_date,low1_trading_date,low2_trading_date
  p1_idx,p2_idx,low1_idx,low2_idx,p1p2_trading_days
  neckline_slope_per_day
  drop1_norm,rise1_norm,drop2_norm,trough_sim,spread_t,p1_to_l1_t,l1_to_p2_t,p2_to_l2_t
  rsi_p2,rsi_l2,vol_ratio_p2,vol_ratio_l2,px_vs_ma20_p2,px_vs_ma20_l2,px_vs_ma50_l2

Usage:
  pip install yfinance pandas numpy
  python add_neckline_slope.py --in w_labeled.csv --out w_with_features.csv
"""

import argparse
import sys
from datetime import datetime, timedelta, date
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

RECENT_WINDOW = 90  # for time-normalizing features

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
    # Pad so MA50/RSI have enough history
    start_pad = start_date - timedelta(days=200)
    end_pad   = end_date + timedelta(days=5)
    try:
        df = yf.Ticker(ticker).history(
            start=start_pad.isoformat(),
            end=(end_date + timedelta(days=6)).isoformat(),
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
    if df is None or df.empty:
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

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["VolMA20"] = df["Volume"].rolling(20, min_periods=1).mean()
    df["RSI14"] = rsi_ewm(df["Close"], 14)
    return df

def indicator_features_at(df: pd.DataFrame, idx: int, tag: str) -> dict:
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

    # Init outputs
    for k in ["p1_trading_date","p2_trading_date","low1_trading_date","low2_trading_date"]:
        out[k] = ""
    for k in ["p1_idx","p2_idx","low1_idx","low2_idx","p1p2_trading_days",
              "neckline_slope_per_day","drop1_norm","rise1_norm","drop2_norm",
              "trough_sim","spread_t","p1_to_l1_t","l1_to_p2_t","p2_to_l2_t",
              "rsi_p2","rsi_l2","vol_ratio_p2","vol_ratio_l2","px_vs_ma20_p2","px_vs_ma20_l2","px_vs_ma50_l2"]:
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

    # Geometry/timing features (use YOUR labeled prices, no leakage)
    base = max(1e-9, min(l1_price, l2_price))
    out["drop1_norm"] = float((l1_price/base) - (p1_price/base))
    out["rise1_norm"] = float((p2_price/base) - (l1_price/base))
    out["drop2_norm"] = float((l2_price/base) - (p2_price/base))
    out["trough_sim"] = float(abs(l2_price - l1_price) / max(1e-9, l1_price))
    out["spread_t"] = float((i_l2 - i_l1) / RECENT_WINDOW)
    out["p1_to_l1_t"] = float((i_l1 - i_p1) / RECENT_WINDOW)
    out["l1_to_p2_t"] = float((i_p2 - i_l1) / RECENT_WINDOW)
    out["p2_to_l2_t"] = float((i_l2 - i_p2) / RECENT_WINDOW)

    # Indicator features at P2 & L2
    feats_p2 = indicator_features_at(df, i_p2, "p2")
    feats_l2 = indicator_features_at(df, i_l2, "l2")
    out.update(feats_p2); out.update(feats_l2)

    return out

def main():
    ap = argparse.ArgumentParser(description="Append neckline slope + indicators to W-pattern CSV")
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
