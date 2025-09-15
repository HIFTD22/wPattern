#!/usr/bin/env python3
"""
rules1.py — Rules-based W-pattern (double-bottom) scanner for S&P 500

Key rules (as specified):
- Window: last 90 daily bars (Yahoo Finance).
- Pivots use only Open/Close (peaks=max(OC), troughs=min(OC)); High/Low not used for pivot prices.
- ZigZag pathing on High/Low (lenient) to sketch general swings.
- Structure: P1 → L1 → P2 → L2. Spacing constraint applies to P1–P2 (default 3–85 bars).
- P1 should almost always >= P2 (tolerance tunable; default 2%).
- Direction checks are lenient (allow small counter-moves); can be disabled with --no-direction-checks.
- Neckline: straight line through P1 High → P2 High.
- Breakout: after L2, bar crosses above neckline by small buffer (default 0.1%).
  Use High for breakout if --use-high-for-breakout (otherwise Close).
- FORMING kept if either:
    (a) last bar within nearline_tol below the neckline, OR
    (b) last Close is at least (1+forming_rise_min) × L2 (default any rise since L2).

CLI examples:
  python rules1.py scan --limit 150 --symbols-file sp500.txt --use-high-for-breakout --plot
  python rules1.py scan --limit 200 --zigzag-pct 0.008 --atr-mult 0.25 --trough-sim-hi 0.40 --nearline-tol 0.18 --p1-ge-tol 0.05 --forming-rise-min 0.00 --use-high-for-breakout --plot --debug
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -------------------- Tunables / defaults --------------------

RECENT_WINDOW = 90
ATR_LEN = 14

# ZigZag sensitivity (lenient shape)
DEFAULT_ZZ_PCT = 0.010      # 1% move from extreme
DEFAULT_ATR_MULT = 0.40     # or 0.4 * ATR14, whichever is larger

# Spacing on P1–P2
DEFAULT_MIN_SEP = 3
DEFAULT_MAX_SEP = 85

# Trough similarity band
DEFAULT_TROUGH_SIM_LO = 0.00
DEFAULT_TROUGH_SIM_HI = 0.30

# Direction checks (sign-only); can be disabled
MIN_DROP_INTO_L1 = 0.0
MIN_RISE_TO_P2  = 0.0
MIN_DROP_TO_L2  = 0.0

# P1 >= P2 tolerance (allow tiny deviation). Set higher (e.g., 0.05) to be more permissive.
DEFAULT_P1_GE_TOL = 0.02

# Breakout / forming
DEFAULT_NECKLINE_BUFFER = 0.001  # 0.1% above neckline
DEFAULT_NEARLINE_TOL    = 0.12   # FORMING if within 12% below neckline
DEFAULT_FORMING_RISE_MIN = 0.00  # keep forming if last close >= (1+X) * L2 (default: any rise)

# Misc
MAX_WORKERS = 8
TIMEOUT_SECS = 12
RETRIES = 3
RETRY_SLEEP = 0.7

# -------------------- Helpers --------------------

def norm_symbol(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")

def read_symbols_file(path: str) -> List[str]:
    syms: List[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            syms.append(norm_symbol(s))
    # dedupe, keep order
    seen = set(); out = []
    for s in syms:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def get_sp500_tickers(limit: int, symbols_file: Optional[str], debug: bool=False) -> List[str]:
    if symbols_file and os.path.exists(symbols_file):
        syms = read_symbols_file(symbols_file)
        if debug:
            print(f"[tickers] loaded {len(syms)} symbols from file: {symbols_file}")
        return syms[:limit]

    # Try Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url, timeout=TIMEOUT_SECS)
        t = next((x for x in tables if "Symbol" in x.columns), None)
        if t is not None:
            syms = (
                t["Symbol"].astype(str)
                .str.replace(".", "-", regex=False)
                .str.strip().str.upper().tolist()
            )
            syms = [s for s in syms if s and s != "NAN"]
            # dedupe while keeping order
            seen=set(); ordered=[]
            for s in syms:
                if s not in seen:
                    seen.add(s); ordered.append(s)
            if debug:
                print(f"[tickers] fetched {len(ordered)} from Wikipedia")
            return ordered[:limit]
    except Exception:
        pass

    # Fallback 50
    fallback_50 = [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","COST","META","BRK-B","LLY","AVGO",
        "TSLA","JPM","V","WMT","XOM","UNH","MA","PG","ORCL","COST",
        "HD","MRK","KO","PEP","BAC","ABBV","CRM","PFE","TMO","DIS",
        "ACN","CSCO","NFLX","ABT","DHR","AMD","QCOM","LIN","MCD","TXN",
        "VZ","NKE","UPS","PM","CAT","IBM","HON","RTX","LOW","ADP"
    ]
    if debug:
        print(f"[tickers] Wikipedia fetch failed; using fallback list of {len(fallback_50)} symbols.")
        if limit > len(fallback_50):
            print("[tickers] To scan >50, pass --symbols-file with your full S&P list.")
    return fallback_50[:limit]

def fetch_hist(ticker: str) -> Optional[pd.DataFrame]:
    t = norm_symbol(ticker)
    for attempt in range(1, RETRIES+1):
        try:
            df = yf.Ticker(t).history(period="12mo", interval="1d", auto_adjust=True)
            if df is None or df.empty:
                raise RuntimeError("empty")
            need = ["Open","High","Low","Close","Volume"]
            if not all(c in df.columns for c in need):
                raise RuntimeError("missing cols")
            df = df[need].dropna().sort_index()
            if df.shape[0] < 40:
                raise RuntimeError("too few bars")
            return df.tail(RECENT_WINDOW).copy()
        except Exception:
            if attempt == RETRIES:
                return None
            time.sleep(RETRY_SLEEP)

def atr_series(df: pd.DataFrame, n: int = ATR_LEN) -> pd.Series:
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def neckline_value_at(k: int, p1_idx: int, p1_high: float, p2_idx: int, p2_high: float) -> float:
    if p2_idx == p1_idx:
        return p2_high
    return p1_high + (p2_high - p1_high) * (k - p1_idx) / (p2_idx - p1_idx)

# -------------------- Effective peak/trough (Open/Close only) --------------------

def eff_peak_price(df: pd.DataFrame, i: int) -> float:
    o = float(df["Open"].iloc[i]); c = float(df["Close"].iloc[i])
    return max(o, c)

def eff_trough_price(df: pd.DataFrame, i: int) -> float:
    o = float(df["Open"].iloc[i]); c = float(df["Close"].iloc[i])
    return min(o, c)

# -------------------- ZigZag on High/Low (lenient pathing) --------------------

def zigzag_hl(df: pd.DataFrame, pct_th: float, atr_mult: float) -> List[Tuple[int,float,str]]:
    highs = df["High"].to_numpy(float)
    lows  = df["Low"].to_numpy(float)
    n = len(highs)
    if n < 8: return []
    atr = atr_series(df, ATR_LEN).to_numpy()

    def thr(i_ref: int, ref_price: float) -> float:
        return max(pct_th * max(1e-9, ref_price), atr_mult * atr[i_ref])

    piv: List[Tuple[int,float,str]] = []
    mode = None
    cur_ext_idx, cur_ext_price = 0, highs[0]
    # find first swing
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

    # clean duplicates
    piv = sorted(piv, key=lambda x: x[0])
    cleaned: List[Tuple[int,float,str]] = []
    for idx, price, lab in piv:
        if not cleaned:
            cleaned.append((idx, price, lab)); continue
        pidx, pprice, plab = cleaned[-1]
        if lab == plab:
            if (lab == "peak" and price >= pprice) or (lab == "trough" and price <= pprice):
                cleaned[-1] = (idx, price, lab)
        else:
            cleaned.append((idx, price, lab))
    return cleaned

# -------------------- W detection (rules) --------------------

def detect_w_rules(
    df: pd.DataFrame,
    zz_pct: float,
    atr_mult: float,
    trough_sim_lo: float,
    trough_sim_hi: float,
    nearline_tol: float,
    neckline_buffer: float,
    use_high_for_breakout: bool,
    p1_ge_tol: float,
    min_sep: int,
    max_sep: int,
    forming_rise_min: float,
    direction_checks: bool,
    debug: bool=False
) -> List[Dict]:
    n = len(df)
    if n < 40: return []
    piv = zigzag_hl(df, zz_pct, atr_mult)
    if len(piv) < 4:
        if debug: print("  [debug] pivots < 4, skip")
        return []

    highs  = df["High"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    dates  = df.index.to_list()

    out: List[Dict] = []
    trough_idx = [i for i,_,lab in piv if lab == "trough"]
    peak_idx   = [i for i,_,lab in piv if lab == "peak"]

    for l1 in trough_idx:
        p1_cands = [p for p in peak_idx if p < l1]
        if not p1_cands: 
            continue
        p1 = max(p1_cands)  # latest peak before L1

        for l2 in trough_idx:
            if l2 <= l1: 
                continue
            p2_cands = [p for p in peak_idx if l1 < p < l2]
            if not p2_cands:
                continue
            p2 = max(p2_cands, key=lambda x: highs[x])

            if not (p1 < l1 < p2 < l2):
                continue

            # spacing on P1–P2
            if not (min_sep <= (p2 - p1) <= max_sep):
                continue

            # OC-based pivots
            p1_eff = eff_peak_price(df, p1)
            p2_eff = eff_peak_price(df, p2)
            l1_eff = eff_trough_price(df, l1)
            l2_eff = eff_trough_price(df, l2)

            # P1 >= P2 (tolerance)
            if p1_eff < p2_eff * (1.0 - p1_ge_tol):
                continue

            # leg directions (optional)
            if direction_checks:
                if not (l1_eff < p1_eff - MIN_DROP_INTO_L1 * abs(p1_eff)):  # drop into L1
                    continue
                if not (p2_eff > l1_eff + MIN_RISE_TO_P2  * abs(l1_eff)):   # rise to P2
                    continue
                if not (l2_eff < p2_eff - MIN_DROP_TO_L2  * abs(p2_eff)):   # drop to L2
                    continue

            # trough similarity
            trough_diff = abs(l2_eff - l1_eff) / max(1e-9, l1_eff)
            if not (trough_sim_lo <= trough_diff <= trough_sim_hi):
                continue

            # neckline via P1 High → P2 High
            slope_per_day = (highs[p2] - highs[p1]) / max(1, (p2 - p1))

            # breakout after L2
            breakout_idx: Optional[int] = None
            for k in range(l2 + 1, n):
                neck_k = neckline_value_at(k, p1, highs[p1], p2, highs[p2])
                px = float(df["High"].iloc[k]) if use_high_for_breakout else closes[k]
                if px >= neck_k * (1.0 + neckline_buffer):
                    breakout_idx = k
                    break

            status = "COMPLETED" if breakout_idx is not None else "FORMING"

            # forming rule: near neckline OR risen since L2 by forming_rise_min
            keep_forming = False
            if status == "FORMING":
                k = n - 1
                neck_now = neckline_value_at(k, p1, highs[p1], p2, highs[p2])
                near = (closes[k] >= neck_now * (1.0 - nearline_tol))
                risen = (closes[k] >= l2_eff * (1.0 + forming_rise_min))
                keep_forming = near or risen
                if not keep_forming:
                    continue

            rec = {
                "p1_idx": int(p1), "low1_idx": int(l1), "p2_idx": int(p2), "low2_idx": int(l2),
                "p3_idx": int(breakout_idx) if breakout_idx is not None else np.nan,
                "p1_date": dates[p1].date().isoformat(),
                "low1_date": dates[l1].date().isoformat(),
                "p2_date": dates[p2].date().isoformat(),
                "low2_date": dates[l2].date().isoformat(),
                "p3_date": dates[breakout_idx].date().isoformat() if breakout_idx is not None else "",
                # OC-based pivot prices
                "p1_price": float(p1_eff),
                "low1_price": float(l1_eff),
                "p2_price": float(p2_eff),
                "low2_price": float(l2_eff),
                "p3_price": float(df["High"].iloc[breakout_idx]) if breakout_idx is not None else np.nan,
                "neckline_slope_per_day": float(slope_per_day),
                "trough_sim": float(trough_diff),
                "status": status,
            }
            out.append(rec)

    if debug:
        print(f"  [debug] pivots={len(piv)} peaks={len([1 for _,_,l in piv if l=='peak'])} troughs={len([1 for _,_,l in piv if l=='trough'])} candidates={len(out)}")
    return out

# -------------------- Plotting --------------------

def plot_candidate(df: pd.DataFrame, rec: Dict, ticker: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    dates = pd.to_datetime(df.index)
    n = len(df)

    p1, l1, p2, l2 = rec["p1_idx"], rec["low1_idx"], rec["p2_idx"], rec["low2_idx"]
    p3 = rec.get("p3_idx")
    highs = df["High"].to_numpy(float)
    close = df["Close"].to_numpy(float)

    neck_vals = np.array([neckline_value_at(k, p1, highs[p1], p2, highs[p2]) for k in range(n)])

    plt.figure(figsize=(10,5))
    plt.plot(dates, close, lw=1.3, alpha=0.7, label="Close")
    plt.plot(dates, df["High"].to_numpy(float), lw=0.9, alpha=0.35, label="High")
    plt.plot(dates, df["Low"].to_numpy(float),  lw=0.9, alpha=0.35, label="Low")
    plt.plot(dates, neck_vals, "--", lw=1.4, label="Neckline (P1→P2 Highs)")

    plt.scatter(dates[[p1,l1,p2,l2]], close[[p1,l1,p2,l2]], s=52, marker="D", label="P1/L1/P2/L2 (OC-based)")
    if not (p3 is None or (isinstance(p3,float) and np.isnan(p3))):
        plt.scatter(dates[int(p3)], close[int(p3)], s=70, marker="^", label="P3 (breakout)")
    title = f"{ticker} — {rec['status']} (W)"
    plt.title(title); plt.legend(loc="best"); plt.grid(alpha=0.3)
    fname = f"{ticker}_{rec['low1_date']}_{rec['low2_date']}_{rec['status']}.png".replace(":","-")
    outpath = os.path.join(outdir, fname)
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()
    return outpath

# -------------------- Scan S&P --------------------

def scan_ticker(t: str, args) -> List[Dict]:
    df = fetch_hist(t)
    if df is None or df.empty:
        return []
    cands = detect_w_rules(
        df,
        zz_pct=args.zigzag_pct,
        atr_mult=args.atr_mult,
        trough_sim_lo=args.trough_sim_lo,
        trough_sim_hi=args.trough_sim_hi,
        nearline_tol=args.nearline_tol,
        neckline_buffer=args.neck_buffer,
        use_high_for_breakout=args.use_high_for_breakout,
        p1_ge_tol=args.p1_ge_tol,
        min_sep=args.min_sep,
        max_sep=args.max_sep,
        forming_rise_min=args.forming_rise_min,
        direction_checks=(not args.no_direction_checks),
        debug=args.debug
    )
    for c in cands:
        c["ticker"] = t
    if args.plot and cands:
        for c in cands[: args.max_plots_per_ticker]:
            try:
                plot_candidate(df, c, t, args.plot_dir)
            except Exception:
                pass
    return cands

def run_scan(args):
    syms = get_sp500_tickers(args.limit, args.symbols_file, args.debug)
    print(f"[scan] scanning {len(syms)} symbols (daily, last {RECENT_WINDOW} bars)...")

    rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(scan_ticker, s, args): s for s in syms}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                hits = fut.result()
            except Exception:
                hits = []
            if hits:
                print(f"[{s}] {len(hits)} candidate(s)")
                rows.extend(hits)

    if not rows:
        print("[scan] No W-shaped candidates found with current criteria.")
        if not args.symbols_file:
            print("[hint] If you intended to scan all 500, pass --symbols-file <path_to_list.txt>.")
        return

    cols = ["ticker","status",
            "p1_date","p1_price","low1_date","low1_price","p2_date","p2_price","low2_date","low2_price",
            "p3_date","p3_price",
            "neckline_slope_per_day","trough_sim",
            "p1_idx","low1_idx","p2_idx","low2_idx","p3_idx"]
    df_out = pd.DataFrame(rows)
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out = df_out[cols].sort_values(["status","low2_date","ticker"]).reset_index(drop=True)

    out_csv = args.out
    df_out.to_csv(out_csv, index=False)
    print(f"[scan] Wrote {out_csv}  (hits: {len(df_out)})")
    if args.plot:
        print(f"[scan] Plots in: {os.path.abspath(args.plot_dir)}")

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Rules-based W-pattern scanner (S&P 500, OC-based pivots)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scan", help="Scan symbols for W patterns (rules-based)")
    sc.add_argument("--limit", type=int, default=100, help="Number of symbols to scan")
    sc.add_argument("--out", default="w_rules_hits.csv", help="Output CSV")
    sc.add_argument("--symbols-file", help="Path to a text file with one ticker per line")

    # Sensitivity / shape knobs (ZigZag on High/Low)
    sc.add_argument("--zigzag-pct", type=float, default=DEFAULT_ZZ_PCT, help="ZigZag percent threshold (e.g., 0.010=1%)")
    sc.add_argument("--atr-mult", type=float, default=DEFAULT_ATR_MULT, help="ZigZag ATR multiple")
    sc.add_argument("--min-sep", type=int, default=DEFAULT_MIN_SEP, help="Minimum bars between P1 and P2")
    sc.add_argument("--max-sep", type=int, default=DEFAULT_MAX_SEP, help="Maximum bars between P1 and P2")

    # Trough similarity band
    sc.add_argument("--trough-sim-lo", type=float, default=DEFAULT_TROUGH_SIM_LO)
    sc.add_argument("--trough-sim-hi", type=float, default=DEFAULT_TROUGH_SIM_HI)

    # P1 ≥ P2 tolerance (0 = strict)
    sc.add_argument("--p1-ge-tol", type=float, default=DEFAULT_P1_GE_TOL,
                    help="Tolerance for P1≥P2 (e.g., 0.05 allows P1 up to 5% below P2)")

    # Breakout/FORMING
    sc.add_argument("--neck-buffer", type=float, default=DEFAULT_NECKLINE_BUFFER,
                    help="Required fraction above neckline to call breakout")
    sc.add_argument("--nearline-tol", type=float, default=DEFAULT_NEARLINE_TOL,
                    help="Keep FORMING if last bar within this fraction below neckline")
    sc.add_argument("--forming-rise-min", type=float, default=DEFAULT_FORMING_RISE_MIN,
                    help="Also keep FORMING if last close >= (1+X) * L2 (default 0.0 => any rise)")
    sc.add_argument("--use-high-for-breakout", action="store_true",
                    help="Use daily High for breakout check (default uses Close unless this flag is set)")

    sc.add_argument("--no-direction-checks", action="store_true", help="Disable leg direction checks (P1→L1→P2→L2 order only)")
    sc.add_argument("--plot", action="store_true", help="Save plots for each candidate")
    sc.add_argument("--plot-dir", default="plots", help="Folder for plots")
    sc.add_argument("--max-plots-per-ticker", type=int, default=2)
    sc.add_argument("--debug", action="store_true")

    sc.set_defaults(func=lambda a: run_scan(a))

    args = ap.parse_args()
    if args.cmd == "scan":
        run_scan(args)

if __name__ == "__main__":
    main()
