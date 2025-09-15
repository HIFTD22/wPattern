#!/usr/bin/env python3
"""
STRICT Feature Engineering for W-pattern recognition.
Adds quality-focused features and enhanced validation for high-quality patterns.

Usage:
  python yfslope3.py --in w_data.csv --out w_with_strict_features.csv
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
    """Get extended history for strict quality analysis."""
    start_pad = start_date - timedelta(days=300)  # More history for better indicators
    try:
        df = yf.Ticker(ticker).history(
            start=start_pad.isoformat(),
            end=(end_date + timedelta(days=10)).isoformat(),  # Extra buffer
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

def nearest_trading_index(df: pd.DataFrame, target: date, tolerance_days: int = 2) -> Optional[int]:
    """Stricter tolerance for exact date matching."""
    if df is None or df.empty:
        return None
    dates = np.array([d.date() for d in df.index])
    diffs = np.array([abs((d - target).days) for d in dates])
    j = int(diffs.argmin())
    return j if diffs[j] <= tolerance_days else None

def rsi_ewm(close: pd.Series, n: int = 14) -> pd.Series:
    """Enhanced RSI calculation."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def add_strict_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators for strict analysis."""
    df = df.copy()
    
    # Basic indicators
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["MA100"] = df["Close"].rolling(100, min_periods=1).mean()  # Long-term trend
    df["VolMA20"] = df["Volume"].rolling(20, min_periods=1).mean()
    df["RSI14"] = rsi_ewm(df["Close"], 14)
    
    # Enhanced volatility measures
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14, min_periods=1).mean()
    df["ATR20"] = tr.rolling(20, min_periods=1).mean()
    
    # Price volatility
    df["PriceStd20"] = df["Close"].rolling(20).std()
    df["PriceStd50"] = df["Close"].rolling(50).std()
    
    # Volume analysis
    df["VolStd20"] = df["Volume"].rolling(20).std()
    df["VolRatio"] = df["Volume"] / df["VolMA20"]
    
    # Momentum indicators
    df["ROC10"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10) * 100).fillna(0)
    df["ROC20"] = ((df["Close"] - df["Close"].shift(20)) / df["Close"].shift(20) * 100).fillna(0)
    
    # Bollinger Bands for pattern context
    bb_mean = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BBUpper"] = bb_mean + (2 * bb_std)
    df["BBLower"] = bb_mean - (2 * bb_std)
    df["BBPosition"] = ((df["Close"] - bb_mean) / (2 * bb_std)).fillna(0)
    
    return df

def strict_indicator_features_at(df: pd.DataFrame, idx: int, tag: str) -> Dict[str, float]:
    """Enhanced indicator features with quality metrics."""
    if idx is None or idx < 0 or idx >= len(df):
        return {f"rsi_{tag}": np.nan, f"vol_ratio_{tag}": np.nan,
                f"px_vs_ma20_{tag}": np.nan, f"px_vs_ma50_{tag}": np.nan}
    
    c = float(df["Close"].iloc[idx])
    
    # Core indicators
    rsi = float(df["RSI14"].iloc[idx]) if not pd.isna(df["RSI14"].iloc[idx]) else np.nan
    vol = float(df["Volume"].iloc[idx])
    volma20 = float(df["VolMA20"].iloc[idx]) if not pd.isna(df["VolMA20"].iloc[idx]) else np.nan
    ma20 = float(df["MA20"].iloc[idx]) if not pd.isna(df["MA20"].iloc[idx]) else np.nan
    ma50 = float(df["MA50"].iloc[idx]) if not pd.isna(df["MA50"].iloc[idx]) else np.nan
    
    # Enhanced features
    atr14 = float(df["ATR14"].iloc[idx]) if not pd.isna(df["ATR14"].iloc[idx]) else np.nan
    bb_pos = float(df["BBPosition"].iloc[idx]) if not pd.isna(df["BBPosition"].iloc[idx]) else np.nan
    
    return {
        f"rsi_{tag}": rsi,
        f"vol_ratio_{tag}": (vol / volma20) if (volma20 and not np.isnan(volma20) and volma20 != 0) else np.nan,
        f"px_vs_ma20_{tag}": (c / ma20 - 1.0) if (ma20 and not np.isnan(ma20) and ma20 != 0) else np.nan,
        f"px_vs_ma50_{tag}": (c / ma50 - 1.0) if (ma50 and not np.isnan(ma50) and ma50 != 0) else np.nan,
        f"atr_normalized_{tag}": (atr14 / c) if (atr14 and not np.isnan(atr14) and c > 0) else np.nan,
        f"bb_position_{tag}": bb_pos,
    }

def get_strict_market_regime_features(df: pd.DataFrame, current_idx: int) -> Dict[str, float]:
    """Enhanced market regime detection with quality metrics."""
    lookback = min(60, current_idx)  # Extended lookback
    if lookback < 20:
        return {'regime_trend': 0, 'regime_volatility': 0, 'regime_volume': 0, 'regime_strength': 0,
                'regime_quality': 0}
    
    recent_data = df.iloc[current_idx-lookback:current_idx+1]
    
    # Enhanced trend analysis
    x = np.arange(len(recent_data))
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, recent_data['Close'])
    
    # Trend strength with statistical significance
    trend_strength = abs(slope) * (r_val ** 2)
    trend_significance = 1.0 - p_val if not np.isnan(p_val) else 0.0
    
    # Volatility regime with multiple measures
    atr_current = recent_data['ATR14'].iloc[-1] if 'ATR14' in recent_data.columns else 0
    atr_long = df['ATR14'].rolling(100).mean().iloc[current_idx] if 'ATR14' in df.columns else 0
    vol_regime = (atr_current / max(atr_long, 1e-10) - 1.0) if atr_long > 0 else 0
    
    # Price volatility
    price_vol = recent_data['Close'].std() / recent_data['Close'].mean() if recent_data['Close'].mean() > 0 else 0
    
    # Volume regime with consistency
    vol_ratio = recent_data['Volume'].mean() / max(df['Volume'].rolling(100).mean().iloc[current_idx], 1)
    vol_consistency = 1.0 / (1.0 + recent_data['Volume'].std() / max(recent_data['Volume'].mean(), 1))
    
    # Market quality score
    regime_quality = (trend_significance * 0.4 + 
                     min(vol_consistency, 1.0) * 0.3 + 
                     min(abs(r_val), 1.0) * 0.3)
    
    return {
        'regime_trend': float(trend_strength),
        'regime_volatility': float(vol_regime), 
        'regime_volume': float(vol_ratio - 1.0),
        'regime_strength': float((r_val ** 2) * min(vol_ratio, 2.0)),
        'regime_quality': float(regime_quality),
        'trend_significance': float(trend_significance),
        'price_volatility': float(price_vol)
    }

def get_strict_volume_profile_features(df: pd.DataFrame, p1: int, l1: int, p2: int, l2: int) -> Dict[str, float]:
    """Enhanced volume analysis with quality validation."""
    
    # Extended baseline for better comparison
    baseline_start = max(0, p1 - 40)
    baseline_vol = df['Volume'].iloc[baseline_start:p1].mean()
    baseline_std = df['Volume'].iloc[baseline_start:p1].std()
    
    # Volume during each phase
    vol_p1_l1 = df['Volume'].iloc[p1:l1+1].mean() if l1 > p1 else baseline_vol
    vol_l1_p2 = df['Volume'].iloc[l1:p2+1].mean() if p2 > l1 else baseline_vol
    vol_p2_l2 = df['Volume'].iloc[p2:l2+1].mean() if l2 > p2 else baseline_vol
    
    # Volume quality metrics
    vol_accumulation = vol_l1_p2 / max(baseline_vol, 1)
    vol_confirmation = vol_p2_l2 / max(baseline_vol, 1)
    
    # Volume trend analysis
    pattern_vols = df['Volume'].iloc[p1:l2+1].values
    vol_trend_slope = 0
    vol_consistency = 0.5
    vol_spike_quality = 0
    
    if len(pattern_vols) > 1:
        vol_trend_slope = stats.linregress(range(len(pattern_vols)), pattern_vols)[0] / max(baseline_vol, 1)
        vol_consistency = 1.0 / (1.0 + np.std(pattern_vols) / max(baseline_vol, 1))
        
        # Volume spike quality (high volume during drops is bullish)
        drop1_vol = df['Volume'].iloc[p1:l1+1].mean() if l1 > p1 else baseline_vol
        drop2_vol = df['Volume'].iloc[p2:l2+1].mean() if l2 > p2 else baseline_vol
        vol_spike_quality = min(drop1_vol / max(baseline_vol, 1), 
                               drop2_vol / max(baseline_vol, 1)) - 1.0
    
    # Recent volume momentum
    recent_vol = df['Volume'].iloc[max(0, l2-5):l2+1].mean()
    vol_momentum = recent_vol / max(baseline_vol, 1) - 1.0
    
    # Volume distribution quality
    vol_distribution_score = 0
    if baseline_std > 0:
        vol_distribution_score = 1.0 - min(baseline_std / baseline_vol, 2.0) / 2.0
    
    return {
        'volume_accumulation_ratio': float(vol_accumulation),
        'volume_confirmation_ratio': float(vol_confirmation), 
        'volume_trend_slope': float(vol_trend_slope),
        'volume_consistency': float(vol_consistency),
        'volume_momentum': float(vol_momentum),
        'volume_spike_quality': float(vol_spike_quality),
        'volume_distribution_quality': float(vol_distribution_score)
    }

def get_strict_pattern_quality_features(df: pd.DataFrame, p1: int, l1: int, p2: int, l2: int,
                                       p1_price: float, l1_price: float, p2_price: float, l2_price: float) -> Dict[str, float]:
    """Comprehensive pattern quality assessment."""
    
    # Enhanced duration analysis
    left_duration = l1 - p1
    middle_duration = p2 - l1  
    right_duration = l2 - p2
    total_duration = l2 - p1
    
    # Optimal W proportions: ~25% decline, 50% recovery, 25% second decline
    duration_balance = 0
    if total_duration > 0:
        left_pct = left_duration / total_duration
        middle_pct = middle_duration / total_duration
        right_pct = right_duration / total_duration
        
        # Score based on deviation from ideal proportions
        ideal_deviation = (abs(left_pct - 0.25) + 
                          abs(middle_pct - 0.5) + 
                          abs(right_pct - 0.25)) / 3
        duration_balance = max(0, 1.0 - ideal_deviation * 2)
    
    # Enhanced trough consistency
    trough_consistency = 1.0 - abs(l2_price - l1_price) / max(l1_price, l2_price)
    
    # Peak relationship quality
    peak_relationship = min(p2_price / max(p1_price, 1e-10), 1.5)
    peak_decline_quality = max(0, 1.0 - p2_price / p1_price)  # P2 should be lower
    
    # Pattern clarity with noise analysis
    price_range = max(p1_price, p2_price) - min(l1_price, l2_price)
    pattern_clarity = 0.5
    noise_level = 0
    
    if price_range > 0 and p2 > l1:
        # Measure price noise in recovery phase
        recovery_highs = df['High'].iloc[l1:p2].values
        recovery_lows = df['Low'].iloc[l1:p2].values
        
        # Clean recovery should show progressive improvement
        recovery_noise = np.std(recovery_highs) / np.mean(recovery_highs) if len(recovery_highs) > 0 else 0
        noise_level = min(recovery_noise, 1.0)
        pattern_clarity = max(0, 1.0 - noise_level)
    
    # Amplitude relationships
    drop1_pct = abs(l1_price - p1_price) / max(p1_price, 1e-10)
    recovery_pct = abs(p2_price - l1_price) / max(l1_price, 1e-10)
    drop2_pct = abs(l2_price - p2_price) / max(p2_price, 1e-10)
    
    amplitude_balance = 1.0 - abs(drop1_pct - drop2_pct) / max(drop1_pct + drop2_pct, 1e-10)
    recovery_quality = min(recovery_pct / ((drop1_pct + drop2_pct) / 2), 2.0) / 2.0
    
    # Pattern symmetry (some asymmetry is natural)
    time_symmetry = 1.0 - abs(left_duration - right_duration) / max(left_duration + right_duration, 1)
    
    # Overall pattern quality score
    quality_components = [
        duration_balance * 0.2,
        trough_consistency * 0.25,
        pattern_clarity * 0.25,
        amplitude_balance * 0.15,
        recovery_quality * 0.15
    ]
    overall_quality = sum(quality_components)
    
    return {
        'pattern_duration_balance': float(duration_balance),
        'pattern_trough_consistency': float(trough_consistency),
        'pattern_peak_relationship': float(peak_relationship), 
        'pattern_clarity': float(pattern_clarity),
        'pattern_amplitude_balance': float(amplitude_balance),
        'pattern_noise_level': float(noise_level),
        'pattern_recovery_quality': float(recovery_quality),
        'pattern_time_symmetry': float(time_symmetry),
        'pattern_overall_quality': float(overall_quality),
        'peak_decline_quality': float(peak_decline_quality)
    }

def get_strict_market_context_features(df: pd.DataFrame, p1: int, l2: int) -> Dict[str, float]:
    """Enhanced market context with trend and support analysis."""
    
    lookback_period = min(120, p1)  # Extended lookback
    if lookback_period < 30:
        return {'market_position': 0.5, 'trend_context': 0, 'volatility_context': 0, 
                'support_strength': 0, 'trend_quality': 0}
    
    historical_data = df.iloc[p1-lookback_period:p1]
    pattern_low = min(df['Low'].iloc[max(0, l2-3):l2+4])
    
    # Enhanced market positioning
    hist_high = historical_data['High'].max()
    hist_low = historical_data['Low'].min()
    hist_range = hist_high - hist_low
    
    market_position = (pattern_low - hist_low) / max(hist_range, 1e-10) if hist_range > 0 else 0.5
    
    # Multi-timeframe trend analysis
    short_trend = stats.linregress(range(30), df['Close'].iloc[p1-30:p1])[0] if p1 >= 30 else 0
    long_trend = stats.linregress(range(lookback_period), historical_data['Close'])[0]
    
    trend_context = long_trend / max(historical_data['Close'].mean(), 1)
    trend_consistency = 1.0 if (short_trend * long_trend) >= 0 else 0.0  # Same direction
    
    # Volatility context with regime detection
    hist_vol = historical_data['Close'].std()
    pattern_vol = df['Close'].iloc[p1:l2+1].std()
    volatility_context = pattern_vol / max(hist_vol, 1e-10) - 1.0
    
    # Enhanced support strength analysis
    support_level = pattern_low
    tolerance = support_level * 0.025  # 2.5% tolerance
    
    support_tests = 0
    support_quality = 0
    
    # Count support tests and measure quality
    for i in range(max(0, p1-80), p1):
        if abs(df['Low'].iloc[i] - support_level) < tolerance:
            support_tests += 1
            # Higher quality if volume was elevated during test
            if 'Volume' in df.columns:
                test_vol = df['Volume'].iloc[i]
                baseline_vol = df['Volume'].iloc[max(0, i-5):i].mean()
                if test_vol > baseline_vol * 1.2:  # Elevated volume
                    support_quality += 1
    
    support_strength = min(support_tests / 8.0, 1.0)  # Normalize
    support_quality_ratio = support_quality / max(support_tests, 1)
    
    # Trend quality assessment
    trend_r_squared = stats.linregress(range(lookback_period), historical_data['Close'])[2] ** 2
    trend_quality = trend_r_squared * trend_consistency
    
    return {
        'market_position': float(market_position),
        'trend_context': float(trend_context),
        'volatility_context': float(volatility_context),
        'support_strength': float(support_strength),
        'support_quality_ratio': float(support_quality_ratio),
        'trend_consistency': float(trend_consistency),
        'trend_quality': float(trend_quality),
        'short_long_trend_ratio': float(short_trend / max(abs(long_trend), 1e-10))
    }

def calculate_strict_quality_scores(p1_price: float, l1_price: float, p2_price: float, l2_price: float,
                                  pattern_days: int) -> Dict[str, float]:
    """Calculate comprehensive quality scores for the pattern."""
    
    # Movement quality
    drop1_pct = (p1_price - l1_price) / p1_price
    recovery_pct = (p2_price - l1_price) / l1_price
    drop2_pct = (p2_price - l2_price) / p2_price
    
    # Overall range
    max_price = max(p1_price, p2_price)
    min_price = min(l1_price, l2_price)
    overall_range = (max_price - min_price) / max_price
    
    # Pattern symmetry
    pattern_symmetry = 1.0 - abs(drop1_pct - drop2_pct) / max(drop1_pct + drop2_pct, 1e-10)
    
    # Recovery strength relative to drops
    recovery_strength = recovery_pct / max((drop1_pct + drop2_pct) / 2, 1e-10)
    
    # Time quality (not too fast, not too slow)
    time_quality = 1.0 - abs(pattern_days - 20) / 40.0  # Optimal around 20 days
    time_quality = max(0, min(time_quality, 1.0))
    
    # Combined quality score
    quality_score = (
        drop1_pct * 0.25 +      # Significant initial drop
        recovery_pct * 0.25 +   # Good recovery
        drop2_pct * 0.20 +      # Second drop
        pattern_symmetry * 0.15 + # Balance
        min(recovery_strength, 1.0) * 0.15  # Recovery quality
    )
    
    return {
        'quality_score': float(quality_score),
        'overall_range': float(overall_range),
        'pattern_symmetry': float(pattern_symmetry),
        'recovery_strength': float(min(recovery_strength, 2.0)),
        'time_quality': float(time_quality),
        'drop1_pct': float(drop1_pct),
        'recovery_pct': float(recovery_pct),
        'drop2_pct': float(drop2_pct)
    }

def compute_strict_row(row: pd.Series, tolerance_days: int) -> pd.Series:
    """Enhanced feature computation with strict quality validation."""
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
    strict_features = [
        # Original features
        "p1_trading_date", "p2_trading_date", "low1_trading_date", "low2_trading_date",
        "p1_idx", "p2_idx", "low1_idx", "low2_idx", "p1p2_trading_days",
        "neckline_slope_per_day", "drop1_norm", "rise1_norm", "drop2_norm",
        "trough_sim", "spread_t", "p1_to_l1_t", "l1_to_p2_t", "p2_to_l2_t",
        "rsi_p2", "rsi_l2", "vol_ratio_p2", "vol_ratio_l2", 
        "px_vs_ma20_p2", "px_vs_ma20_l2", "px_vs_ma50_l2",
        
        # Enhanced regime features
        "regime_trend", "regime_volatility", "regime_volume", "regime_strength",
        "regime_quality", "trend_significance", "price_volatility",
        
        # Enhanced volume features
        "volume_accumulation_ratio", "volume_confirmation_ratio", "volume_trend_slope", 
        "volume_consistency", "volume_momentum", "volume_spike_quality", "volume_distribution_quality",
        
        # Enhanced pattern quality
        "pattern_duration_balance", "pattern_trough_consistency", "pattern_peak_relationship",
        "pattern_clarity", "pattern_amplitude_balance", "pattern_noise_level",
        "pattern_recovery_quality", "pattern_time_symmetry", "pattern_overall_quality", "peak_decline_quality",
        
        # Enhanced market context
        "market_position", "trend_context", "volatility_context", "support_strength",
        "support_quality_ratio", "trend_consistency", "trend_quality", "short_long_trend_ratio",
        
        # Quality scores
        "quality_score", "overall_range", "pattern_symmetry", "recovery_strength", "time_quality",
        "drop1_pct", "recovery_pct", "drop2_pct"
    ]
    
    for feature in strict_features:
        if feature.endswith("_date"):
            out[feature] = ""
        else:
            out[feature] = np.nan
    
    if not ticker or None in (p1_date, p2_date, l1_date, l2_date):
        return out
    
    # Get market data with extended history
    d0 = min(p1_date, p2_date, l1_date, l2_date)
    d1 = max(p1_date, p2_date, l1_date, l2_date)
    df0 = robust_history_range(ticker, d0, d1)
    if df0 is None or df0.empty:
        print(f"No data for {ticker}")
        return out
    
    df = add_strict_indicators(df0)
    
    # Find trading indices with strict tolerance
    i_p1 = nearest_trading_index(df, p1_date, tolerance_days)
    i_p2 = nearest_trading_index(df, p2_date, tolerance_days)
    i_l1 = nearest_trading_index(df, l1_date, tolerance_days)
    i_l2 = nearest_trading_index(df, l2_date, tolerance_days)
    
    if None in (i_p1, i_p2, i_l1, i_l2):
        print(f"Could not find exact trading dates for {ticker}")
        return out
    
    # Validate sequence
    if not (i_p1 < i_l1 < i_p2 < i_l2):
        print(f"Invalid sequence for {ticker}: P1({i_p1}) L1({i_l1}) P2({i_p2}) L2({i_l2})")
        return out
    
    # Basic features
    out["p1_trading_date"] = df.index[i_p1].date().isoformat()
    out["p2_trading_date"] = df.index[i_p2].date().isoformat()
    out["low1_trading_date"] = df.index[i_l1].date().isoformat()
    out["low2_trading_date"] = df.index[i_l2].date().isoformat()
    
    out["p1_idx"] = int(i_p1)
    out["p2_idx"] = int(i_p2)
    out["low1_idx"] = int(i_l1)
    out["low2_idx"] = int(i_l2)
    
    bars = abs(i_p2 - i_p1)
    pattern_days = i_l2 - i_p1
    out["p1p2_trading_days"] = int(bars)
    if bars > 0:
        out["neckline_slope_per_day"] = float((p2_price - p1_price) / bars)
    
    # Geometry features
    base = max(1e-9, min(l1_price, l2_price))
    out["drop1_norm"] = float((l1_price/base) - (p1_price/base))
    out["rise1_norm"] = float((p2_price/base) - (l1_price/base))
    out["drop2_norm"] = float((l2_price/base) - (p2_price/base))
    out["trough_sim"] = float(abs(l2_price - l1_price) / max(1e-9, l1_price))
    out["spread_t"] = float((i_l2 - i_l1) / RECENT_WINDOW)
    out["p1_to_l1_t"] = float((i_l1 - i_p1) / RECENT_WINDOW)
    out["l1_to_p2_t"] = float((i_p2 - i_l1) / RECENT_WINDOW)
    out["p2_to_l2_t"] = float((i_l2 - i_p2) / RECENT_WINDOW)
    
    # Enhanced indicator features
    feats_p2 = strict_indicator_features_at(df, i_p2, "p2")
    feats_l2 = strict_indicator_features_at(df, i_l2, "l2")
    out.update(feats_p2)
    out.update(feats_l2)
    
    # All enhanced features
    try:
        regime_feats = get_strict_market_regime_features(df, i_l2)
        volume_feats = get_strict_volume_profile_features(df, i_p1, i_l1, i_p2, i_l2)
        quality_feats = get_strict_pattern_quality_features(df, i_p1, i_l1, i_p2, i_l2, 
                                                           p1_price, l1_price, p2_price, l2_price)
        context_feats = get_strict_market_context_features(df, i_p1, i_l2)
        score_feats = calculate_strict_quality_scores(p1_price, l1_price, p2_price, l2_price, pattern_days)
        
        out.update(regime_feats)
        out.update(volume_feats)
        out.update(quality_feats)
        out.update(context_feats)
        out.update(score_feats)
        
        # Final validation - mark patterns that meet strict criteria
        quality_passed = (
            score_feats.get('quality_score', 0) >= 0.025 and  # 2.5% minimum
            score_feats.get('overall_range', 0) >= 0.03 and   # 3% minimum range
            quality_feats.get('pattern_overall_quality', 0) >= 0.6 and  # 60% pattern quality
            out['trough_sim'] <= 0.12  # Max 12% trough difference
        )
        
        out['strict_quality_passed'] = float(quality_passed)
        
        if quality_passed:
            print(f"✓ HIGH QUALITY: {ticker} - Quality: {score_feats.get('quality_score', 0):.1%}, "
                  f"Range: {score_feats.get('overall_range', 0):.1%}")
        
    except Exception as e:
        print(f"Error computing strict features for {ticker}: {e}")
    
    return out

def main():
    ap = argparse.ArgumentParser(description="Strict feature engineering for W-patterns")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output CSV")
    ap.add_argument("--tolerance-days", type=int, default=2, help="Date matching tolerance (strict: 2)")
    args = ap.parse_args()
    
    df_in = pd.read_csv(args.in_csv)
    required = ["ticker", "p1_date", "p1_price", "p2_date", "p2_price", 
                "low1_date", "low1_price", "low2_date", "low2_price"]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)
    
    print(f"=== STRICT FEATURE ENGINEERING ===")
    print(f"Processing {len(df_in)} patterns with strict quality criteria...")
    print(f"Date matching tolerance: {args.tolerance_days} days")
    
    rows = []
    high_quality_count = 0
    
    for i, (_, r) in enumerate(df_in.iterrows()):
        print(f"Processing {i+1}/{len(df_in)}: {r['ticker']}")
        result = compute_strict_row(r, args.tolerance_days)
        rows.append(result)
        
        if result.get('strict_quality_passed', 0) == 1.0:
            high_quality_count += 1
    
    df_out = pd.DataFrame(rows)
    
    # Preserve original column order, add new columns
    user_cols = list(df_in.columns)
    new_cols = [c for c in df_out.columns if c not in user_cols]
    df_out = df_out[user_cols + new_cols]
    
    df_out.to_csv(args.out_csv, index=False)
    
    print(f"\n=== STRICT PROCESSING COMPLETE ===")
    print(f"Saved: {args.out_csv}")
    print(f"Total patterns: {len(df_out)}")
    print(f"Features added: {len(new_cols)}")
    print(f"High-quality patterns: {high_quality_count}/{len(df_out)} ({high_quality_count/len(df_out)*100:.1f}%)")
    
    if high_quality_count > 0:
        print(f"\n✅ Ready for strict training:")
        print(f"  python yftraining3.py --csv {args.out_csv} --target all")
        print(f"  python yfscan3.py --limit 20 --min-score 0.7")
    else:
        print(f"\n⚠️  No patterns met strict quality criteria.")
        print(f"Consider adjusting thresholds or reviewing input data quality.")

if __name__ == "__main__":
    main()