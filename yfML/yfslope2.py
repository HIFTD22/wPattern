#!/usr/bin/env python3
"""
Enhanced feature engineering for W-pattern recognition.
Adds market context, regime detection, and temporal dynamics.

Usage:
  python yfslope_enhanced.py --in w_data.csv --out w_with_enhanced_features.csv
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta, date
from typing import Optional, Dict
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

RECENT_WINDOW = 90

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
    start_pad = start_date - timedelta(days=200)
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
    
    # Add ATR for regime detection
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14, min_periods=1).mean()
    
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

def get_market_regime_features(df: pd.DataFrame, current_idx: int) -> Dict[str, float]:
    """Enhanced market regime detection."""
    lookback = min(50, current_idx)
    if lookback < 10:
        return {'regime_trend': 0, 'regime_volatility': 0, 'regime_volume': 0, 'regime_strength': 0}
    
    recent_data = df.iloc[current_idx-lookback:current_idx+1]
    
    # Trend regime with R² weighting
    x = np.arange(len(recent_data))
    slope, _, r_val, _, _ = stats.linregress(x, recent_data['Close'])
    trend_strength = abs(slope) * (r_val ** 2)
    
    # Volatility regime (normalized ATR)
    atr_current = recent_data['ATR14'].iloc[-1] if 'ATR14' in recent_data.columns else 0
    atr_long = df['ATR14'].rolling(50).mean().iloc[current_idx] if 'ATR14' in df.columns else 0
    vol_regime = (atr_current / max(atr_long, 1e-10) - 1.0) if atr_long > 0 else 0
    
    # Volume regime
    vol_ratio = recent_data['Volume'].mean() / max(df['Volume'].rolling(50).mean().iloc[current_idx], 1)
    
    # Market strength (combining trend consistency + volume)
    trend_consistency = r_val ** 2  # How linear is the trend
    vol_support = min(vol_ratio, 2.0)  # Cap extreme volume
    regime_strength = trend_consistency * vol_support
    
    return {
        'regime_trend': float(trend_strength),
        'regime_volatility': float(vol_regime), 
        'regime_volume': float(vol_ratio - 1.0),
        'regime_strength': float(regime_strength)
    }

def get_volume_profile_features(df: pd.DataFrame, p1: int, l1: int, p2: int, l2: int) -> Dict[str, float]:
    """Enhanced volume analysis during W formation."""
    
    # Volume during each phase
    vol_p1_l1 = df['Volume'].iloc[p1:l1+1].mean() if l1 > p1 else 0
    vol_l1_p2 = df['Volume'].iloc[l1:p2+1].mean() if p2 > l1 else 0
    vol_p2_l2 = df['Volume'].iloc[p2:l2+1].mean() if l2 > p2 else 0
    
    # Baseline volume (extended lookback for stability)
    baseline_start = max(0, p1 - 30)
    baseline_vol = df['Volume'].iloc[baseline_start:p1].mean()
    
    # Volume ratios
    vol_accumulation = vol_l1_p2 / max(baseline_vol, 1)
    vol_confirmation = vol_p2_l2 / max(baseline_vol, 1)
    
    # Volume trend and consistency
    pattern_vols = df['Volume'].iloc[p1:l2+1].values
    if len(pattern_vols) > 1:
        vol_trend_slope = stats.linregress(range(len(pattern_vols)), pattern_vols)[0] / max(baseline_vol, 1)
        vol_consistency = 1.0 / (1.0 + np.std(pattern_vols) / max(baseline_vol, 1))
    else:
        vol_trend_slope = 0
        vol_consistency = 0.5
    
    # Volume momentum (recent vs baseline)
    recent_vol = df['Volume'].iloc[l2-5:l2+1].mean() if l2 >= 5 else vol_p2_l2
    vol_momentum = recent_vol / max(baseline_vol, 1) - 1.0
    
    return {
        'volume_accumulation_ratio': float(vol_accumulation),
        'volume_confirmation_ratio': float(vol_confirmation), 
        'volume_trend_slope': float(vol_trend_slope),
        'volume_consistency': float(vol_consistency),
        'volume_momentum': float(vol_momentum)
    }

def get_pattern_quality_features(df: pd.DataFrame, p1: int, l1: int, p2: int, l2: int,
                                p1_price: float, l1_price: float, p2_price: float, l2_price: float) -> Dict[str, float]:
    """Assess technical quality of the W pattern."""
    
    # Duration analysis
    left_duration = l1 - p1
    middle_duration = p2 - l1  
    right_duration = l2 - p2
    total_duration = l2 - p1
    
    if total_duration > 0:
        # Ideal W has ~20% left, 60% middle, 20% right
        left_pct = left_duration / total_duration
        middle_pct = middle_duration / total_duration
        right_pct = right_duration / total_duration
        
        # Penalty for extreme imbalance
        duration_balance = 1.0 - abs(left_pct - 0.2) - abs(middle_pct - 0.6) - abs(right_pct - 0.2)
        duration_balance = max(0, duration_balance)
    else:
        duration_balance = 0
    
    # Price level consistency
    trough_consistency = 1.0 - abs(l2_price - l1_price) / max(l1_price, l2_price)
    
    # Peak relationship (P2 should be lower than P1 for classic W)
    peak_relationship = min(p2_price / max(p1_price, 1e-10), 1.5)  # Cap at 1.5
    
    # Pattern clarity (measure noise between key points)
    price_range = max(p1_price, p2_price) - min(l1_price, l2_price)
    if price_range > 0:
        # Check for false breaks or noise
        l1_p2_high = df['High'].iloc[l1:p2].max() if p2 > l1 else p2_price
        l1_p2_low = df['Low'].iloc[l1:p2].min() if p2 > l1 else l1_price
        noise_ratio = (l1_p2_high - l1_p2_low) / price_range
        pattern_clarity = max(0, 1.0 - noise_ratio)
    else:
        pattern_clarity = 0
    
    # Amplitude analysis (how deep/sharp are the moves)
    drop1_pct = abs(l1_price - p1_price) / max(p1_price, 1e-10)
    rise1_pct = abs(p2_price - l1_price) / max(l1_price, 1e-10)
    drop2_pct = abs(l2_price - p2_price) / max(p2_price, 1e-10)
    
    amplitude_balance = 1.0 - abs(drop1_pct - drop2_pct) / max(drop1_pct + drop2_pct, 1e-10)
    
    return {
        'pattern_duration_balance': float(duration_balance),
        'pattern_trough_consistency': float(trough_consistency),
        'pattern_peak_relationship': float(peak_relationship), 
        'pattern_clarity': float(pattern_clarity),
        'pattern_amplitude_balance': float(amplitude_balance)
    }

def get_market_context_features(df: pd.DataFrame, p1: int, l2: int) -> Dict[str, float]:
    """Market context around the pattern."""
    
    # Historical context
    lookback_period = min(100, p1)
    if lookback_period < 20:
        return {'market_position': 0.5, 'trend_context': 0, 'volatility_context': 0, 'support_strength': 0}
    
    historical_data = df.iloc[p1-lookback_period:p1]
    pattern_low = min(df['Low'].iloc[l2-2:l2+3]) if l2 >= 2 else df['Low'].iloc[l2]
    
    # Market position
    hist_high = historical_data['High'].max()
    hist_low = historical_data['Low'].min()
    hist_range = hist_high - hist_low
    
    market_position = (pattern_low - hist_low) / max(hist_range, 1e-10) if hist_range > 0 else 0.5
    
    # Trend context
    trend_slope = stats.linregress(range(len(historical_data)), historical_data['Close'])[0]
    trend_context = trend_slope / max(historical_data['Close'].mean(), 1)
    
    # Volatility context
    hist_vol = historical_data['Close'].std()
    recent_vol = df['Close'].iloc[p1:l2+1].std()
    volatility_context = recent_vol / max(hist_vol, 1e-10) - 1.0
    
    # Support strength (how many times this level was tested)
    support_level = pattern_low
    tolerance = support_level * 0.03  # 3% tolerance
    
    support_tests = 0
    for i in range(max(0, p1-50), p1):
        if abs(df['Low'].iloc[i] - support_level) < tolerance:
            support_tests += 1
    
    support_strength = min(support_tests / 5.0, 1.0)  # Normalize to [0,1]
    
    return {
        'market_position': float(market_position),
        'trend_context': float(trend_context),
        'volatility_context': float(volatility_context),
        'support_strength': float(support_strength)
    }

def compute_row(row: pd.Series, tolerance_days: int) -> pd.Series:
    out = row.copy()
    
    ticker = norm_symbol(row.get("ticker", ""))
    p1_date = parse_ymd(row.get("p1_date"))
    p2_date = parse_ymd(row.get("p2_date"))
    l1_date = parse_ymd(row.get("low1_date"))
    l2_date = parse_ymd(row.get("low2_date"))
    
    try:
        p1_price = float(row.get("p1_price"))
        p2_price = float(row.get("p2_price"))
        l1_price = float(row.get("low1_price"))
        l2_price = float(row.get("low2_price"))
    except Exception:
        print(f"Error parsing prices for {ticker}")
        return out
    
    # Initialize all new features
    new_features = [
        "p1_trading_date", "p2_trading_date", "low1_trading_date", "low2_trading_date",
        "p1_idx", "p2_idx", "low1_idx", "low2_idx", "p1p2_trading_days",
        "neckline_slope_per_day", "drop1_norm", "rise1_norm", "drop2_norm",
        "trough_sim", "spread_t", "p1_to_l1_t", "l1_to_p2_t", "p2_to_l2_t",
        "rsi_p2", "rsi_l2", "vol_ratio_p2", "vol_ratio_l2", 
        "px_vs_ma20_p2", "px_vs_ma20_l2", "px_vs_ma50_l2",
        # Enhanced features
        "regime_trend", "regime_volatility", "regime_volume", "regime_strength",
        "volume_accumulation_ratio", "volume_confirmation_ratio", "volume_trend_slope", 
        "volume_consistency", "volume_momentum",
        "pattern_duration_balance", "pattern_trough_consistency", "pattern_peak_relationship",
        "pattern_clarity", "pattern_amplitude_balance",
        "market_position", "trend_context", "volatility_context", "support_strength"
    ]
    
    for feature in new_features:
        if feature.endswith("_date"):
            out[feature] = ""
        else:
            out[feature] = np.nan
    
    if not ticker or None in (p1_date, p2_date, l1_date, l2_date):
        return out
    
    # Get market data
    d0 = min(p1_date, p2_date, l1_date, l2_date)
    d1 = max(p1_date, p2_date, l1_date, l2_date)
    df0 = robust_history_range(ticker, d0, d1)
    if df0 is None or df0.empty:
        print(f"No data for {ticker}")
        return out
    
    df = add_indicators(df0)
    
    # Find trading indices
    i_p1 = nearest_trading_index(df, p1_date, tolerance_days)
    i_p2 = nearest_trading_index(df, p2_date, tolerance_days)
    i_l1 = nearest_trading_index(df, l1_date, tolerance_days)
    i_l2 = nearest_trading_index(df, l2_date, tolerance_days)
    
    if None in (i_p1, i_p2, i_l1, i_l2):
        print(f"Could not find trading dates for {ticker}")
        return out
    
    # Basic features (original)
    out["p1_trading_date"] = df.index[i_p1].date().isoformat()
    out["p2_trading_date"] = df.index[i_p2].date().isoformat()
    out["low1_trading_date"] = df.index[i_l1].date().isoformat()
    out["low2_trading_date"] = df.index[i_l2].date().isoformat()
    
    out["p1_idx"] = int(i_p1)
    out["p2_idx"] = int(i_p2)
    out["low1_idx"] = int(i_l1)
    out["low2_idx"] = int(i_l2)
    
    bars = abs(i_p2 - i_p1)
    out["p1p2_trading_days"] = int(bars)
    if bars > 0:
        out["neckline_slope_per_day"] = float((p2_price - p1_price) / bars)
    
    # Geometry features (using labeled prices, no leakage)
    base = max(1e-9, min(l1_price, l2_price))
    out["drop1_norm"] = float((l1_price/base) - (p1_price/base))
    out["rise1_norm"] = float((p2_price/base) - (l1_price/base))
    out["drop2_norm"] = float((l2_price/base) - (p2_price/base))
    out["trough_sim"] = float(abs(l2_price - l1_price) / max(1e-9, l1_price))
    out["spread_t"] = float((i_l2 - i_l1) / RECENT_WINDOW)
    out["p1_to_l1_t"] = float((i_l1 - i_p1) / RECENT_WINDOW)
    out["l1_to_p2_t"] = float((i_p2 - i_l1) / RECENT_WINDOW)
    out["p2_to_l2_t"] = float((i_l2 - i_p2) / RECENT_WINDOW)
    
    # Indicator features
    feats_p2 = indicator_features_at(df, i_p2, "p2")
    feats_l2 = indicator_features_at(df, i_l2, "l2")
    out.update(feats_p2)
    out.update(feats_l2)
    
    # Enhanced features
    try:
        regime_feats = get_market_regime_features(df, i_l2)
        volume_feats = get_volume_profile_features(df, i_p1, i_l1, i_p2, i_l2)
        quality_feats = get_pattern_quality_features(df, i_p1, i_l1, i_p2, i_l2, p1_price, l1_price, p2_price, l2_price)
        context_feats = get_market_context_features(df, i_p1, i_l2)
        
        out.update(regime_feats)
        out.update(volume_feats)
        out.update(quality_feats)
        out.update(context_feats)
        
    except Exception as e:
        print(f"Error computing enhanced features for {ticker}: {e}")
    
    return out

def main():
    ap = argparse.ArgumentParser(description="Enhanced feature engineering for W-patterns")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output CSV")
    ap.add_argument("--tolerance-days", type=int, default=3, help="Date tolerance")
    args = ap.parse_args()
    
    df_in = pd.read_csv(args.in_csv)
    required = ["ticker", "p1_date", "p1_price", "p2_date", "p2_price", 
                "low1_date", "low1_price", "low2_date", "low2_price"]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing {len(df_in)} patterns...")
    rows = []
    for i, (_, r) in enumerate(df_in.iterrows()):
        print(f"Processing {i+1}/{len(df_in)}: {r['ticker']}")
        rows.append(compute_row(r, args.tolerance_days))
    
    df_out = pd.DataFrame(rows)
    
    # Preserve original column order, add new columns
    user_cols = list(df_in.columns)
    new_cols = [c for c in df_out.columns if c not in user_cols]
    df_out = df_out[user_cols + new_cols]
    
    df_out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} (rows: {len(df_out)}, columns: {len(df_out.columns)})")

if __name__ == "__main__":
    main()