#!/usr/bin/env python3
"""
Scan 50 S&P 500 stocks for W-shaped (double-bottom) patterns
STRICTLY within the last 90 trading days using DAILY OHLC bars.

Key behaviors:
- Uses Yahoo Finance (Ticker.history) with auto-adjusted OHLC.
- Peaks/troughs are derived from each day's HIGH/LOW (not averages).
- Pattern detection prioritizes Open/Close dynamics for breakout/near-line checks,
  but will accept HIGH crossing the neckline as a fallback if OC-based checks don't show it.
- Relaxed, trend-aware W (Peak1→Low1→Peak2→Low2→Peak3) with diagonal neckline (P1→P2).
- Optional plotting (--plot) to save labeled PNGs.

Install:
    pip install yfinance pandas numpy scipy requests lxml matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------- General Config -------------------------

NUM_TICKERS = 50            # scan 50 names
RECENT_WINDOW = 90          # analyze exactly last 90 trading days
MAX_WORKERS = 8             # modest to reduce Yahoo throttling
TIMEOUT_SECS = 12

# Robust download knobs
MAX_RETRIES = 3
RETRY_SLEEP = 0.7  # seconds

# ------------------------- ZigZag / Sensitivity (Permissive) -------------------------

# Aimed to "count it if it resembles a W"
PCT_ZZ = 0.015              # 1.5% minimum reversal threshold
ATR_MULT = 0.5              # 0.5 * ATR(14) reversal
ATR_LEN = 14

# Geometric / structural constraints (relaxed per your guidance)
MIN_SEP = 3                 # min bars between Low1 and Low2
MAX_SEP = 85                # max bars between Low1 and Low2
TROUGH_SIM_LO = 0.00        # |L2-L1|/L1 >= 0%
TROUGH_SIM_HI = 0.20        # ... <= 20%
PREDROP_MIN = 0.02          # Peak1 -> Low1 drop >= 2%
POST_L2_LIFT_MIN = 0.00     # forming allowed even with tiny lift

# Neckline / breakout (permissive near-line)
BREAKOUT_BUFFER = 0.001     # +0.1% above neckline confirms breakout
FORMING_NEARLINE_TOL = 0.05 # within 5% of neckline counts as "near breakout"

# Preferred price-mode for detection/breakout checks.
# We try these in order, keeping hits from the first mode that yields patterns.
PRICE_MODE_PRIORITY = ["OCMID", "CLOSE", "OPEN", "HL2", "HLC3"]

# ------------------------- Tickers -------------------------

def get_sp500_tickers(limit: int = NUM_TICKERS) -> List[str]:
    """Try scraping S&P 500 tickers; fallback to 50 large names."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url, timeout=TIMEOUT_SECS)
        table = next((t for t in tables if "Symbol" in t.columns), None)
        if table is not None:
            syms = (
                table["Symbol"].astype(str)
                .str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
                .str.strip().str.upper().tolist()
            )
            syms = sorted(list({s for s in syms if s and s != "NAN"}))
            if len(syms) >= limit:
                return syms[:limit]
    except Exception:
        pass

    # Fallback: 50 widely-followed S&P 500 tickers
    return [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","LLY","AVGO",
        "TSLA","JPM","V","WMT","XOM","UNH","MA","PG","ORCL","COST",
        "HD","MRK","KO","PEP","BAC","ABBV","CRM","PFE","TMO","DIS",
        "ACN","CSCO","NFLX","ABT","DHR","AMD","QCOM","LIN","MCD","TXN",
        "VZ","NKE","UPS","PM","CAT","IBM","HON","RTX","LOW","ADP"
    ][:limit]

# ------------------------- Data / Indicators -------------------------

def download_last_90_trading_days(ticker: str) -> Optional[pd.DataFrame]:
    """
    Robust daily OHLC download:
      - auto_adjust=True so OHLC are split/dividend adjusted
      - retries to dodge transient rate limits
      - returns last up to 90 trading days with columns: Open, High, Low, Close
    """
    for attempt in range(1, MAX_RETRIES+1):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="9mo", interval="1d", auto_adjust=True)
            if df is None or df.empty:
                raise RuntimeError("empty frame")
            # Normalize and keep only what we need
            need = ["Open","High","Low","Close"]
            if not all(c in df.columns for c in need):
                raise RuntimeError(f"missing column(s)")
            df = df[need].dropna().sort_index()
            if df.shape[0] < 60:  # too few bars, likely throttling or new listing
                raise RuntimeError(f"only {df.shape[0]} rows")
            # Slice to last 90 if available
            if df.shape[0] > RECENT_WINDOW:
                df = df.tail(RECENT_WINDOW).copy()
            return df
        except Exception:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_SLEEP)
            continue

def atr(df: pd.DataFrame, n: int = ATR_LEN) -> pd.Series:
    """Simple ATR on daily bars (Wilder-like moving average)."""
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

# ------------------------- Price Series Modes -------------------------

def build_price_series(df: pd.DataFrame, mode: str) -> np.ndarray:
    """
    Return a 1-D price vector for detection according to the chosen mode.
    Modes (in order of preference):
      - 'OCMID'  : (Open + Close) / 2
      - 'CLOSE'  : Close
      - 'OPEN'   : Open
      - 'HL2'    : (High + Low) / 2
      - 'HLC3'   : (High + Low + Close) / 3
    """
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
        raise ValueError(f"unknown mode {mode}")
    return np.asarray(s.to_numpy(), dtype=float).reshape(-1)

# ------------------------- ZigZag Pivots from High/Low -------------------------

def zigzag_pivots_hl(df: pd.DataFrame,
                     pct_th: float = PCT_ZZ,
                     atr_mult: float = ATR_MULT) -> List[Tuple[int, float, str]]:
    """
    ZigZag pivots built from intraday extremes:
      - Peaks are picked from the day's HIGH
      - Troughs are picked from the day's LOW
    Reversal threshold per step = max(pct_th * ref_price, atr_mult * ATR(14)).
    Returns: [(index, price, 'peak'|'trough'), ...] sorted by index, alternating labels.
    """
    highs = df["High"].astype(float).to_numpy()
    lows  = df["Low"].astype(float).to_numpy()
    n = highs.size
    if n < 5:
        return []

    # ATR for dynamic thresholding
    atr_series = atr(df, ATR_LEN).to_numpy()

    def rev_thr(i_ref: int, ref_price: float) -> float:
        return max(pct_th * max(1e-9, ref_price), atr_mult * atr_series[i_ref])

    pivots = []
    mode = None  # 'up' or 'down'

    # Initialize by finding the first meaningful reversal (start with a candidate peak leg)
    cur_ext_idx = 0
    cur_ext_price = highs[0]  # track extreme high for a potential peak
    for i in range(1, n):
        if highs[i] > cur_ext_price:
            cur_ext_price = highs[i]
            cur_ext_idx = i
        # reversal down: enough drop from the highest high, measured vs next lows
        if (cur_ext_price - lows[i]) >= rev_thr(cur_ext_idx, cur_ext_price):
            pivots.append((cur_ext_idx, highs[cur_ext_idx], "peak"))
            mode = "down"
            break

    if mode is None:
        # Try starting with a trough leg
        cur_ext_idx = 0
        cur_ext_price = lows[0]
        for i in range(1, n):
            if lows[i] < cur_ext_price:
                cur_ext_price = lows[i]
                cur_ext_idx = i
            # reversal up: enough rise from the lowest low, measured vs next highs
            if (highs[i] - cur_ext_price) >= rev_thr(cur_ext_idx, cur_ext_price):
                pivots.append((cur_ext_idx, lows[cur_ext_idx], "trough"))
                mode = "up"
                break

    if mode is None:
        return []

    # Continue zigzag using H/L extremes
    if mode == "down":
        # we are dropping from a recorded peak; track lowest low
        ext_idx = cur_ext_idx
        ext_price = lows[ext_idx]
        for i in range(cur_ext_idx + 1, n):
            if lows[i] < ext_price:
                ext_price = lows[i]; ext_idx = i
            elif (highs[i] - ext_price) >= rev_thr(ext_idx, ext_price):
                # reversal up -> record trough at its LOW
                pivots.append((ext_idx, lows[ext_idx], "trough"))
                mode = "up"
                # new leg up: start tracking highest high
                ext_idx = i
                ext_price = highs[i]
    else:
        # mode == 'up' — we are rising from a recorded trough; track highest high
        ext_idx = cur_ext_idx
        ext_price = highs[ext_idx]
        for i in range(cur_ext_idx + 1, n):
            if highs[i] > ext_price:
                ext_price = highs[i]; ext_idx = i
            elif (ext_price - lows[i]) >= rev_thr(ext_idx, ext_price):
                # reversal down -> record peak at its HIGH
                pivots.append((ext_idx, highs[ext_idx], "peak"))
                mode = "down"
                # new leg down: start tracking lowest low
                ext_idx = i
                ext_price = lows[i]

    # Terminal pivot: whichever leg we're in now
    terminal_label = "peak" if mode == "up" else "trough"
    term_price = highs[ext_idx] if terminal_label == "peak" else lows[ext_idx]
    pivots.append((ext_idx, term_price, terminal_label))

    # Cleanup: ensure alternation; if same label repeats, keep the more extreme
    pivots = sorted(pivots, key=lambda x: x[0])
    cleaned = []
    for idx, price, lab in pivots:
        if not cleaned:
            cleaned.append((idx, price, lab))
        else:
            pidx, pprice, plab = cleaned[-1]
            if lab == plab:
                if (lab == "peak" and price >= pprice) or (lab == "trough" and price <= pprice):
                    cleaned[-1] = (idx, price, lab)
            else:
                cleaned.append((idx, price, lab))
    return cleaned

# ------------------------- Pattern Detection -------------------------

def neckline_value_at(k: int, p1_idx: int, p1_val: float, p2_idx: int, p2_val: float) -> float:
    if p2_idx == p1_idx:
        return p2_val
    slope = (p2_val - p1_val) / (p2_idx - p1_idx)
    return p1_val + slope * (k - p1_idx)

def detect_w_on_series(df: pd.DataFrame, series: np.ndarray) -> List[Dict]:
    """
    Detect relaxed 5-point W strictly in last bars, using:
      - Pivots from High/Low (peaks=HIGH, troughs=LOW)
      - Breakout/near-line checks on preferred 'series' (OCMID/CLOSE/OPEN),
        with a fallback that accepts HIGH crossing the neckline.
    """
    n = series.size
    if n < 60:
        return []

    dates = df.index.to_list()
    highs = df["High"].astype(float).to_numpy()
    lows  = df["Low"].astype(float).to_numpy()

    # Build pivots from High/Low only
    piv = zigzag_pivots_hl(df, PCT_ZZ, ATR_MULT)
    if len(piv) < 5:
        return []

    results = []
    trough_indices = [i for i,_,lab in piv if lab == "trough"]
    peak_indices   = [i for i,_,lab in piv if lab == "peak"]
    piv_dict = {i:(i,price,lab) for i,price,lab in piv}

    for i1 in trough_indices:
        for i2 in trough_indices:
            if i2 <= i1:
                continue
            sep = i2 - i1
            if not (MIN_SEP <= sep <= MAX_SEP):
                continue

            between_peaks = [p for p in peak_indices if i1 < p < i2]
            if not between_peaks:
                continue
            p2 = max(between_peaks, key=lambda x: piv_dict[x][1])

            before_peaks = [p for p in peak_indices if p < i1]
            if not before_peaks:
                continue
            p1 = max(before_peaks)

            if not (p1 < i1 < p2 < i2):
                continue

            # Prices: peaks from HIGH, troughs from LOW
            l1 = piv_dict[i1][1]   # Low1 (LOW)
            l2 = piv_dict[i2][1]   # Low2 (LOW)
            pk1 = piv_dict[p1][1]  # Peak1 (HIGH)
            pk2 = piv_dict[p2][1]  # Peak2 (HIGH)

            # Net-direction checks (relaxed magnitudes)
            if (l1 / max(1e-9, pk1) - 1.0) > -PREDROP_MIN:  # Peak1->Low1 drop ≥ 2%
                continue
            if (pk2 / max(1e-9, l1) - 1.0) <= 0:            # Low1->Peak2 up
                continue
            if (l2 / max(1e-9, pk2) - 1.0) >= 0:            # Peak2->Low2 down
                continue

            # Trough similarity (very permissive 0..20%)
            trough_diff = abs(l2 - l1) / max(1e-9, l1)
            if not (TROUGH_SIM_LO <= trough_diff <= TROUGH_SIM_HI):
                continue

            # Diagonal neckline from (p1, pk1) to (p2, pk2)
            breakout_idx = None
            after_peaks = [p for p in peak_indices if p > i2]
            for p3 in after_peaks:
                line_val = neckline_value_at(p3, p1, pk1, p2, pk2)
                # Prefer OC/Close/Open 'series', accept High as fallback
                if series[p3] >= line_val * (1.0 + BREAKOUT_BUFFER) or \
                   highs[p3]  >= line_val * (1.0 + BREAKOUT_BUFFER):
                    breakout_idx = p3
                    break

            # Forming (near-line) check
            status = "FORMING"
            nearline_ok = False
            if breakout_idx is not None:
                status = "COMPLETED"
            else:
                latest_idx = n - 1
                latest_line = neckline_value_at(latest_idx, p1, pk1, p2, pk2)
                # near on chosen series OR high
                near_latest = (
                    abs(series[latest_idx] - latest_line) / max(1e-9, latest_line) <= FORMING_NEARLINE_TOL or
                    abs(highs[latest_idx]  - latest_line) / max(1e-9, latest_line) <= FORMING_NEARLINE_TOL
                )
                # small post-L2 lift (relaxed)
                j_end = min(n, i2 + 15)
                post_l2_hi_series = np.max(series[i2:j_end]) if j_end > i2 else series[i2]
                lifted = (post_l2_hi_series / max(1e-9, min(l1, l2)) - 1.0) >= POST_L2_LIFT_MIN

                near_any_peak = False
                for rp in after_peaks:
                    if rp > latest_idx:
                        continue
                    lv = neckline_value_at(rp, p1, pk1, p2, pk2)
                    if (abs(series[rp] - lv) / max(1e-9, lv) <= FORMING_NEARLINE_TOL) or \
                       (abs(highs[rp]  - lv) / max(1e-9, lv) <= FORMING_NEARLINE_TOL):
                        near_any_peak = True
                        break

                nearline_ok = near_latest or near_any_peak
                if not (lifted and nearline_ok):
                    continue

            # Score (kept for ranking)
            sep_mid = (MIN_SEP + MAX_SEP) / 2.0
            c1 = max(0.0, 1.0 - (trough_diff - TROUGH_SIM_LO) / max(1e-9, (TROUGH_SIM_HI - TROUGH_SIM_LO)))
            c2 = 1.0
            c3 = 1.0 - (abs(sep - sep_mid) / max(1e-9, (MAX_SEP - MIN_SEP)))
            c4 = min(1.0, max(0.0, (pk1 - l1) / max(1e-9, 0.10 * pk1)))  # more pre-drop → slightly higher
            if breakout_idx is not None:
                line_at_b = neckline_value_at(breakout_idx, p1, pk1, p2, pk2)
                margin = (series[breakout_idx] / max(1e-9, line_at_b)) - 1.0
                # if series didn't break but High did, give margin credit from High
                if margin < 0 and highs[breakout_idx] >= line_at_b:
                    margin = (highs[breakout_idx] / max(1e-9, line_at_b)) - 1.0
                c5 = max(0.0, min(1.0, margin / 0.05))
            else:
                c5 = 0.7 if nearline_ok else 0.4

            results.append({
                "p1_idx": p1, "low1_idx": i1, "p2_idx": p2, "low2_idx": i2, "p3_idx": breakout_idx,
                "p1_date": dates[p1].date().isoformat(),
                "low1_date": dates[i1].date().isoformat(),
                "p2_date": dates[p2].date().isoformat(),
                "low2_date": dates[i2].date().isoformat(),
                "breakout_date": dates[breakout_idx].date().isoformat() if breakout_idx is not None else None,
                "p1_price": float(pk1),  # HIGH at P1
                "low1_price": float(l1), # LOW at L1
                "p2_price": float(pk2),  # HIGH at P2
                "low2_price": float(l2), # LOW at L2
                "status": "COMPLETED" if breakout_idx is not None else "FORMING",
                "score": int(round(100 * (0.25*c1 + 0.20*c2 + 0.25*c3 + 0.10*c4 + 0.20*c5))),
            })

    # Deduplicate: keep best per Low2 vicinity
    results.sort(key=lambda x: (-x["score"], x["low2_idx"]))
    pruned, used = [], set()
    for r in results:
        key = r["low2_idx"]
        if any(abs(key - u) <= 3 for u in used):
            continue
        used.add(key)
        pruned.append(r)
    return pruned

# ------------------------- Plotting -------------------------

def plot_candidate(df: pd.DataFrame, record: Dict, ticker: str, outdir: str,
                   series: np.ndarray, price_mode: str) -> Optional[str]:
    """
    Save a PNG plot for a single detected candidate:
      - the chosen detection series (OCMID/CLOSE/OPEN/HL2/HLC3)
      - all ZigZag pivots (from HL)
      - labeled P1/L1/P2/L2/P3
      - diagonal neckline (extended)
      - overlay Close line for context
    """
    os.makedirs(outdir, exist_ok=True)

    dates = pd.to_datetime(df.index)
    n = series.size

    # Recompute pivots for plotting consistency (HL-based)
    piv = zigzag_pivots_hl(df, PCT_ZZ, ATR_MULT)

    p1 = record["p1_idx"]; l1 = record["low1_idx"]; p2 = record["p2_idx"]; l2 = record["low2_idx"]
    p3 = record.get("p3_idx", None)

    # Neckline values across the plot (extend from p1->p2) on chosen series
    # Note: neckline is defined by HIGH values at P1/P2, so evaluate using those anchors
    series_p1_val = series[p1]  # visual reference on chosen series
    series_p2_val = series[p2]
    # But for the actual line, anchor at HL-true highs for P1/P2:
    hl_p1_val = df["High"].iloc[p1]
    hl_p2_val = df["High"].iloc[p2]
    x_idx = np.arange(n)
    neck_vals = np.array([neckline_value_at(k, p1, hl_p1_val, p2, hl_p2_val) for k in x_idx])

    plt.figure(figsize=(10, 5))
    # Context: Close
    plt.plot(dates, df["Close"].astype(float).to_numpy(), linewidth=1.0, alpha=0.5, label="Close (context)")
    # Detection series
    plt.plot(dates, series, linewidth=1.8, label=f"Detection series ({price_mode})")
    # Neckline
    plt.plot(dates, neck_vals, linestyle="--", linewidth=1.2, label="Diagonal neckline (P1→P2)")

    # Mark pivots
    if piv:
        piv_idx = [i for i,_,_ in piv]
        piv_px  = [px for _,px,_ in piv]
        plt.scatter(dates[piv_idx], piv_px, s=24, marker="o", label="Pivots (HL-based)")

    # Label the points using the series values for placement
    plt.scatter(dates[[p1, l1, p2, l2]], series[[p1, l1, p2, l2]], s=48, marker="D", zorder=5,
                label="P1/L1/P2/L2")
    plt.annotate("P1", (dates[p1], series[p1]), xytext=(5, 8), textcoords="offset points")
    plt.annotate("L1", (dates[l1], series[l1]), xytext=(5, -12), textcoords="offset points")
    plt.annotate("P2", (dates[p2], series[p2]), xytext=(5, 8), textcoords="offset points")
    plt.annotate("L2", (dates[l2], series[l2]), xytext=(5, -12), textcoords="offset points")

    if p3 is not None:
        plt.scatter(dates[p3], series[p3], s=64, marker="^", zorder=6, label="P3 (breakout)")
        plt.annotate("P3", (dates[p3], series[p3]), xytext=(5, 8), textcoords="offset points")

    title = f"{ticker} — W Pattern ({record['status']}, score {record['score']}, mode {price_mode})"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    fname = f"{ticker}_{record['low1_date']}_{record['low2_date']}_{record['status']}_{price_mode}.png".replace(":", "-")
    fpath = os.path.join(outdir, fname)
    plt.tight_layout()
    plt.savefig(fpath, dpi=160)
    plt.close()
    return fpath

# ------------------------- Per-ticker wrapper -------------------------

def analyze_ticker(ticker: str, make_plots: bool = False, plot_dir: str = "plots", debug: bool = False) -> List[Dict]:
    df = download_last_90_trading_days(ticker)
    if df is None or df.empty:
        if debug:
            print(f"[{ticker}] no data")
        return []
    if debug:
        print(f"[{ticker}] rows={df.shape[0]} last={df.index[-1].date()} "
              f"O={df['Open'].iloc[-1]:.2f} C={df['Close'].iloc[-1]:.2f}")

    all_records = []
    # Try price modes in priority until we find patterns; keep all hits from the first successful mode.
    for mode in PRICE_MODE_PRIORITY:
        series = build_price_series(df, mode)
        recs = detect_w_on_series(df, series)
        if recs:
            for r in recs:
                r["ticker"] = ticker
                r["price_mode"] = mode
                if make_plots:
                    try:
                        png = plot_candidate(df, r, ticker, plot_dir, series, mode)
                        r["plot_path"] = png
                    except Exception:
                        r["plot_path"] = None
            all_records.extend(recs)
            break  # stop after first mode with hits (prioritize OC, then fallbacks)

    return all_records

# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Save PNGs for each detected candidate")
    parser.add_argument("--plot-dir", default="plots", help="Directory to save plots")
    parser.add_argument("--limit", type=int, default=NUM_TICKERS, help="How many tickers to scan")
    parser.add_argument("--debug", action="store_true", help="Print per-ticker data stats")
    args = parser.parse_args()

    tickers = get_sp500_tickers(args.limit)
    print(f"Scanning {len(tickers)} S&P 500 symbols (daily OHLC, last {RECENT_WINDOW} trading days)...")

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(analyze_ticker, t, args.plot, args.plot_dir, args.debug): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                res = fut.result()
                rows.extend(res)
                if res:
                    best = max(res, key=lambda r: r["score"])
                    msg = f"[{t}] {len(res)} pattern(s). Best={best['status']} score={best['score']} (L2 {best['low2_date']}, mode {best['price_mode']})"
                    if args.plot and best.get("plot_path"):
                        msg += f" | {best['plot_path']}"
                    print(msg)
                elif args.debug:
                    print(f"[{t}] no patterns (all modes)")
            except Exception as e:
                print(f"[{t}] error: {e}")

    if not rows:
        print("No eligible W (double-bottom) patterns strictly within the last 90 trading days.")
        return

    out = pd.DataFrame(rows)
    cols_order = [
        "ticker",
        "price_mode",
        "status","score",
        "p1_date","low1_date","p2_date","low2_date","breakout_date",
        "p1_price","low1_price","p2_price","low2_price",
        "plot_path"
    ]
    for c in cols_order:
        if c not in out.columns:
            out[c] = np.nan

    out = out[cols_order].sort_values(
        ["status","score","ticker"], ascending=[True, False, True]
    ).reset_index(drop=True)

    fname = "w_patterns_sp500.csv"
    out.to_csv(fname, index=False)

    completed = (out["status"] == "COMPLETED").sum()
    forming = (out["status"] == "FORMING").sum()
    elapsed = time.time() - t0

    print("\n=== SUMMARY ===")
    print(f"Completed: {completed} | Forming: {forming} | Total: {len(out)}")
    print(f"Saved: {fname}")
    if args.plot:
        print(f"Plots saved to: {os.path.abspath(args.plot_dir)}")
    print(f"Elapsed: {elapsed:.1f}s")

    print("\nTop candidates:")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
