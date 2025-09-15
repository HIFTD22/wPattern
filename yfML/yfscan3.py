#!/usr/bin/env python3
"""
Enhanced W-Pattern Scanner with Complete 4-Point Structure and Bullish Neckline.
Uses moderate criteria - not too strict, but ensures complete W-patterns.

Features:
- Mandatory 4-point structure (P1->L1->P2->L2)
- Bullish neckline requirement (P2 < P1)
- Enhanced features and validation
- Moderate thresholds for better pattern discovery

Usage:
  python yfscan_enhanced.py --limit 20 --min-score 0.5 --plot
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from joblib import load
from scipy import stats
from scipy.signal import argrelextrema

# Configuration
MAX_WORKERS = 6
RECENT_WINDOW = 90

# MODERATE W-Pattern criteria - Enhanced but not overly strict
MIN_DROP1_PCT = 0.02      # P1 to L1 must drop at least 2%
MIN_RECOVERY_PCT = 0.015  # L1 to P2 must rise at least 1.5%
MIN_DROP2_PCT = 0.01      # P2 to L2 must drop at least 1%
MAX_TROUGH_DIFF = 0.15    # L1 and L2 can differ by max 15%
MIN_PATTERN_DAYS = 5      # Minimum pattern duration
MAX_PATTERN_DAYS = 60     # Maximum pattern duration
MIN_OVERALL_RANGE = 0.025 # Minimum total price range 2.5%

# Bullish W-pattern requirements
REQUIRE_BULLISH_NECKLINE = True  # P2 must be < P1
MAX_P2_VS_P1_RATIO = 0.99       # P2 can be max 99% of P1 (ensures P2 < P1)
LOCAL_MIN_TOLERANCE = 0.02       # 2% tolerance for local extrema

# Model files
MODEL_SHAPE = "model_shape_enhanced.pkl"
MODEL_BREAKOUT = "model_breakout_enhanced.pkl" 
MODEL_REACHTARGET = "model_reachtarget_enhanced.pkl"

def get_sp500_tickers(limit: int):
    """Get S&P 500 tickers."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, timeout=10)
        table = next((t for t in tables if "Symbol" in t.columns), None)
        if table is not None:
            syms = (table["Symbol"].astype(str)
                   .str.replace(".", "-", regex=False)
                   .str.strip().str.upper().tolist())
            return sorted(list(set(syms)))[:limit]
    except Exception:
        pass
    
    # Fallback
    return ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","JPM","V","WMT",
            "UNH","MA","PG","HD","KO","PEP","ABBV","BAC","CRM","PFE",
            "TMO","CSCO","NFLX","AMD","DIS","ABT","DHR","QCOM","LIN","TXN"][:limit]

def robust_history(ticker: str):
    """Get market data."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="8mo", interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return None
            
        need = ["Open","High","Low","Close","Volume"]
        if not all(c in df.columns for c in need):
            return None
            
        df = df[need].dropna().sort_index()
        return df.tail(RECENT_WINDOW * 1.5) if len(df) > RECENT_WINDOW * 1.5 else df
    except Exception:
        return None

def add_indicators(df: pd.DataFrame):
    """Add technical indicators."""
    df = df.copy()
    
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean() 
    df["VolMA20"] = df["Volume"].rolling(20, min_periods=1).mean()
    
    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))
    
    # ATR for context
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1) 
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14, min_periods=1).mean()
    
    return df

def find_local_extrema(df: pd.DataFrame, order=3):
    """
    Find local maxima and minima with moderate requirements.
    """
    closes = df['Close'].values
    highs = df['High'].values  
    lows = df['Low'].values
    
    # Find local maxima and minima
    peak_indices = argrelextrema(highs, np.greater, order=order)[0]
    trough_indices = argrelextrema(lows, np.less, order=order)[0]
    
    pivots = []
    
    # Add peaks with significance filter
    for idx in peak_indices:
        if 0 <= idx < len(df):
            # Check significance vs nearby prices
            window_start = max(0, idx - 8)
            window_end = min(len(df), idx + 8)
            local_highs = df['High'].iloc[window_start:window_end]
            if highs[idx] >= local_highs.max() * 0.97:  # More lenient than strict
                pivots.append((idx, highs[idx], "PEAK"))
    
    # Add troughs with significance filter
    for idx in trough_indices:
        if 0 <= idx < len(df):
            window_start = max(0, idx - 8)
            window_end = min(len(df), idx + 8)
            local_lows = df['Low'].iloc[window_start:window_end]
            if lows[idx] <= local_lows.min() * 1.03:  # More lenient than strict
                pivots.append((idx, lows[idx], "TROUGH"))
            
    return sorted(pivots, key=lambda x: x[0])

def detect_enhanced_w_patterns(df: pd.DataFrame):
    """
    Detect complete W-patterns with enhanced features and moderate criteria.
    Ensures: P1 -> L1 -> P2 -> L2 with bullish neckline (P2 < P1)
    """
    df_with_indicators = add_indicators(df)
    n = len(df_with_indicators)
    
    if n < 25:  # Moderate minimum
        return []
    
    # Find local extrema
    pivots = find_local_extrema(df_with_indicators, order=3)
    
    if len(pivots) < 4:
        return []
    
    print(f"Analyzing {len(pivots)} pivots for complete bullish W-patterns...")
    
    # Separate peaks and troughs
    actual_peaks = [(i, price, idx) for idx, (i, price, label) in enumerate(pivots) if label == "PEAK"]
    actual_troughs = [(i, price, idx) for idx, (i, price, label) in enumerate(pivots) if label == "TROUGH"]
    
    print(f"Available: {len(actual_peaks)} peaks, {len(actual_troughs)} troughs")
    
    candidates = []
    
    # Search for complete W-patterns: P1 -> L1 -> P2 -> L2
    for p1_data in actual_peaks:
        p1_idx, p1_price, _ = p1_data
        
        # Find L1: troughs after P1
        l1_candidates = [(i, price, idx) for i, price, idx in actual_troughs if i > p1_idx]
        if not l1_candidates:
            continue
            
        for l1_data in l1_candidates:
            l1_idx, l1_price, _ = l1_data
            
            # Find P2: peaks after L1
            p2_candidates = [(i, price, idx) for i, price, idx in actual_peaks if i > l1_idx]
            if not p2_candidates:
                continue
                
            for p2_data in p2_candidates:
                p2_idx, p2_price, _ = p2_data
                
                # MANDATORY: Find L2: troughs after P2 (completes the W)
                l2_candidates = [(i, price, idx) for i, price, idx in actual_troughs if i > p2_idx]
                if not l2_candidates:
                    continue
                
                # Take first valid L2 (closest to P2)
                l2_data = l2_candidates[0]
                l2_idx, l2_price, _ = l2_data
                
                # Verify complete sequence
                if not (p1_idx < l1_idx < p2_idx < l2_idx):
                    continue
                
                # Check timing (moderate range)
                pattern_duration = l2_idx - p1_idx
                if not (MIN_PATTERN_DAYS <= pattern_duration <= MAX_PATTERN_DAYS):
                    continue
                
                # ENHANCED W-PATTERN VALIDATION (moderate criteria)
                
                # 1. Significant drop P1 -> L1
                if p1_price <= l1_price:
                    continue
                drop1_pct = (p1_price - l1_price) / p1_price
                if drop1_pct < MIN_DROP1_PCT:
                    continue
                
                # 2. Significant recovery L1 -> P2
                if p2_price <= l1_price:
                    continue
                recovery_pct = (p2_price - l1_price) / l1_price
                if recovery_pct < MIN_RECOVERY_PCT:
                    continue
                
                # 3. Significant drop P2 -> L2 (completes the W)
                if p2_price <= l2_price:
                    continue
                drop2_pct = (p2_price - l2_price) / p2_price  
                if drop2_pct < MIN_DROP2_PCT:
                    continue
                
                # 4. BULLISH NECKLINE: P2 must be lower than P1
                if REQUIRE_BULLISH_NECKLINE and p2_price >= p1_price:
                    continue
                
                neckline_slope = p2_price - p1_price  # Should be negative for bullish
                neckline_slope_pct = neckline_slope / p1_price
                
                # 5. Double bottom validation (moderate tolerance)
                trough_diff = abs(l1_price - l2_price) / min(l1_price, l2_price)
                if trough_diff > MAX_TROUGH_DIFF:
                    continue
                
                # 6. Overall range validation
                max_peak = max(p1_price, p2_price)
                min_trough = min(l1_price, l2_price)
                overall_range = (max_peak - min_trough) / max_peak
                if overall_range < MIN_OVERALL_RANGE:
                    continue
                
                # 7. Local extrema validation (moderate tolerance)
                def is_local_minimum(idx, price, window=4):
                    start = max(0, idx - window)
                    end = min(len(df_with_indicators), idx + window + 1)
                    local_lows = df_with_indicators['Low'].iloc[start:end]
                    return price <= local_lows.min() * (1 + LOCAL_MIN_TOLERANCE)
                
                def is_local_maximum(idx, price, window=4):
                    start = max(0, idx - window)
                    end = min(len(df_with_indicators), idx + window + 1)
                    local_highs = df_with_indicators['High'].iloc[start:end]
                    return price >= local_highs.max() * (1 - LOCAL_MIN_TOLERANCE)
                
                if not (is_local_maximum(p1_idx, p1_price) and 
                       is_local_minimum(l1_idx, l1_price) and
                       is_local_maximum(p2_idx, p2_price) and
                       is_local_minimum(l2_idx, l2_price)):
                    continue
                
                # 8. Visual W-shape validation
                if not validate_w_shape(df_with_indicators, p1_idx, l1_idx, p2_idx, l2_idx):
                    continue
                
                # SUCCESS: Complete bullish W-pattern found!
                dates = df_with_indicators.index
                
                print(f"✅ COMPLETE BULLISH W-PATTERN!")
                print(f"  {dates[p1_idx].date()} P1: ${p1_price:.2f}")
                print(f"  {dates[l1_idx].date()} L1: ${l1_price:.2f} (drop {drop1_pct:.1%})")
                print(f"  {dates[p2_idx].date()} P2: ${p2_price:.2f} (recovery {recovery_pct:.1%})")
                print(f"  {dates[l2_idx].date()} L2: ${l2_price:.2f} (drop {drop2_pct:.1%})")
                print(f"  Bullish neckline: {neckline_slope_pct:.1%} decline P1→P2")
                
                # Compute enhanced features
                features = compute_enhanced_features(df_with_indicators, p1_idx, l1_idx, p2_idx, l2_idx,
                                                   p1_price, l1_price, p2_price, l2_price)
                
                # Create candidate
                candidate = {
                    "p1_idx": p1_idx, "low1_idx": l1_idx, "p2_idx": p2_idx, "low2_idx": l2_idx,
                    "p1_date": dates[p1_idx].date().isoformat(),
                    "low1_date": dates[l1_idx].date().isoformat(),
                    "p2_date": dates[p2_idx].date().isoformat(),
                    "low2_date": dates[l2_idx].date().isoformat(),
                    "p1_price": float(p1_price), "low1_price": float(l1_price),
                    "p2_price": float(p2_price), "low2_price": float(l2_price),
                    "drop1_pct": float(drop1_pct),
                    "recovery_pct": float(recovery_pct),
                    "drop2_pct": float(drop2_pct),
                    "trough_similarity": float(trough_diff),
                    "pattern_days": int(pattern_duration),
                    "overall_range": float(overall_range),
                    "quality_score": float((drop1_pct + recovery_pct + drop2_pct) / 3),
                    "pattern_type": "BULLISH_W",
                    "neckline_slope": float(neckline_slope),
                    "neckline_slope_pct": float(neckline_slope_pct),
                    **features
                }
                
                candidates.append(candidate)
                
                # Moderate limit to avoid too many overlapping patterns
                if len(candidates) >= 3:
                    return candidates
    
    print(f"Found {len(candidates)} complete bullish W-patterns")
    return candidates

def validate_w_shape(df: pd.DataFrame, p1: int, l1: int, p2: int, l2: int) -> bool:
    """Validate proper W-shape with moderate requirements."""
    try:
        closes = df['Close'].values
        
        # P1 -> L1: Should have more declining than rising moves
        if l1 > p1 + 1:
            declines = sum(1 for i in range(p1+1, l1+1) if closes[i] <= closes[i-1])
            rises = sum(1 for i in range(p1+1, l1+1) if closes[i] > closes[i-1])
            if declines < rises * 0.8:  # More lenient than strict 50/50
                return False
        
        # L1 -> P2: Should have more rising than declining moves  
        if p2 > l1 + 1:
            rises = sum(1 for i in range(l1+1, p2+1) if closes[i] >= closes[i-1])
            declines = sum(1 for i in range(l1+1, p2+1) if closes[i] < closes[i-1])
            if rises < declines * 0.8:  # More lenient
                return False
        
        # P2 -> L2: Should have more declining than rising moves
        if l2 > p2 + 1:
            declines = sum(1 for i in range(p2+1, l2+1) if closes[i] <= closes[i-1])
            rises = sum(1 for i in range(p2+1, l2+1) if closes[i] > closes[i-1])
            if declines < rises * 0.8:  # More lenient
                return False
        
        # Overall: L1 and L2 should be among the lower points in pattern
        pattern_range = range(p1, l2+1)
        pattern_lows = [closes[i] for i in pattern_range]
        min_in_pattern = min(pattern_lows)
        
        # Both L1 and L2 should be within 5% of pattern minimum (more lenient)
        if (closes[l1] > min_in_pattern * 1.05 or 
            closes[l2] > min_in_pattern * 1.05):
            return False
        
        return True
        
    except Exception:
        return False

def compute_enhanced_features(df: pd.DataFrame, p1: int, l1: int, p2: int, l2: int,
                            p1_price: float, l1_price: float, p2_price: float, l2_price: float):
    """Compute enhanced features with moderate quality metrics."""
    
    base_price = min(l1_price, l2_price)
    
    features = {
        "drop1_norm": float((l1_price - p1_price) / base_price),
        "rise1_norm": float((p2_price - l1_price) / base_price),  
        "drop2_norm": float((l2_price - p2_price) / base_price),
        "trough_sim": float(abs(l2_price - l1_price) / l1_price),
        "neckline_slope_per_day": float((p2_price - p1_price) / max(1, p2 - p1)),
        
        # Timing features
        "spread_t": float((l2 - l1) / RECENT_WINDOW),
        "p1_to_l1_t": float((l1 - p1) / RECENT_WINDOW),
        "l1_to_p2_t": float((p2 - l1) / RECENT_WINDOW),
        "p2_to_l2_t": float((l2 - p2) / RECENT_WINDOW),
    }
    
    # Indicators at key points
    for idx, tag in [(p2, "p2"), (l2, "l2")]:
        if 0 <= idx < len(df):
            c = float(df["Close"].iloc[idx])
            
            rsi = df["RSI14"].iloc[idx] if "RSI14" in df.columns and not pd.isna(df["RSI14"].iloc[idx]) else 50
            features[f"rsi_{tag}"] = float(rsi)
            
            vol = float(df["Volume"].iloc[idx])
            volma20 = df["VolMA20"].iloc[idx] if "VolMA20" in df.columns and not pd.isna(df["VolMA20"].iloc[idx]) else vol
            features[f"vol_ratio_{tag}"] = float(vol / max(volma20, 1))
            
            ma20 = df["MA20"].iloc[idx] if "MA20" in df.columns and not pd.isna(df["MA20"].iloc[idx]) else c
            features[f"px_vs_ma20_{tag}"] = float(c / max(ma20, 1) - 1.0)
            
            if tag == "l2":
                ma50 = df["MA50"].iloc[idx] if "MA50" in df.columns and not pd.isna(df["MA50"].iloc[idx]) else c
                features[f"px_vs_ma50_{tag}"] = float(c / max(ma50, 1) - 1.0)
        else:
            features[f"rsi_{tag}"] = 50.0
            features[f"vol_ratio_{tag}"] = 1.0
            features[f"px_vs_ma20_{tag}"] = 0.0
            if tag == "l2":
                features[f"px_vs_ma50_{tag}"] = 0.0
    
    # Enhanced features with moderate defaults
    enhanced_defaults = {
        "regime_trend": 0.1, "regime_volatility": 0.05, "regime_volume": 0.05, "regime_strength": 0.6,
        "volume_accumulation_ratio": 1.1, "volume_confirmation_ratio": 1.05, "volume_trend_slope": 0.02,
        "volume_consistency": 0.7, "volume_momentum": 0.05,
        "pattern_duration_balance": 0.75, "pattern_trough_consistency": float(1.0 - abs(l2_price - l1_price) / l1_price), 
        "pattern_peak_relationship": float(p2_price / max(p1_price, 1e-9)),
        "pattern_clarity": 0.7, "pattern_amplitude_balance": 0.75,
        "market_position": 0.3, "trend_context": -0.02, "volatility_context": 0.05, "support_strength": 0.5
    }
    
    features.update(enhanced_defaults)
    return features

def load_models():
    """Load trained models."""
    models = {}
    
    for model_name, model_path in [("shape", MODEL_SHAPE), ("breakout", MODEL_BREAKOUT), ("reachtarget", MODEL_REACHTARGET)]:
        if os.path.exists(model_path):
            try:
                models[model_name] = load(model_path)
                
                features_path = f"y_{model_name}_features.json"
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        metadata = json.load(f)
                        models[f"{model_name}_features"] = metadata.get('features', [])
                else:
                    models[f"{model_name}_features"] = []
                    
            except Exception as e:
                print(f"Warning: Could not load {model_name} model: {e}")
    
    return models

def score_pattern(pattern, models):
    """Score pattern with ML models."""
    scores = pattern.copy()
    
    for model_name in ["shape", "breakout", "reachtarget"]:
        if model_name in models and f"{model_name}_features" in models:
            try:
                model = models[model_name]
                features = models[f"{model_name}_features"]
                
                if not features:
                    continue
                
                X = []
                for feat in features:
                    val = pattern.get(feat, 0.0)
                    if pd.isna(val):
                        val = 0.0
                    X.append(float(val))
                
                X = np.array(X).reshape(1, -1)
                proba = model.predict_proba(X)[0, 1]
                scores[f"ml_{model_name}_proba"] = float(proba)
                
            except Exception as e:
                scores[f"ml_{model_name}_proba"] = 0.0
    
    return scores

def scan_ticker(ticker: str, models):
    """Scan single ticker for enhanced W patterns."""
    try:
        df = robust_history(ticker)
        if df is None or len(df) < 25:
            return []
        
        patterns = detect_enhanced_w_patterns(df)
        if not patterns:
            return []
        
        results = []
        for pattern in patterns:
            pattern["ticker"] = ticker
            scored = score_pattern(pattern, models)
            results.append(scored)
            
        return results
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return []

def create_plot(df: pd.DataFrame, pattern, ticker: str, output_dir: str):
    """Create enhanced W-pattern visualization."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        dates = pd.to_datetime(df.index)
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        
        p1, l1, p2, l2 = pattern["p1_idx"], pattern["low1_idx"], pattern["p2_idx"], pattern["low2_idx"]
        
        # Create subplot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Price plot with range
        ax.fill_between(dates, low, high, alpha=0.1, color='gray', label='Daily Range')
        ax.plot(dates, close, 'b-', linewidth=1.5, alpha=0.8, label="Close Price")
        
        # Mark W pattern points
        pattern_dates = [dates[p1], dates[l1], dates[p2], dates[l2]]
        pattern_prices = [pattern["p1_price"], pattern["low1_price"], pattern["p2_price"], pattern["low2_price"]]
        pattern_labels = ["P1 (HIGH)", "L1 (LOW)", "P2 (HIGH)", "L2 (LOW)"]
        colors = ['darkred', 'darkgreen', 'darkorange', 'darkblue']
        
        ax.scatter(pattern_dates, pattern_prices, c=colors, s=120, zorder=10, edgecolors='black', linewidth=1.5)
        
        # Enhanced labels
        for i, (date, price, label, color) in enumerate(zip(pattern_dates, pattern_prices, pattern_labels, colors)):
            ax.annotate(f'{label}\n${price:.2f}', (date, price), 
                       xytext=(10, 15 if i % 2 == 0 else -25), textcoords="offset points", 
                       fontweight='bold', fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
        
        # Draw W pattern lines
        ax.plot(pattern_dates, pattern_prices, 'r-', linewidth=3, alpha=0.9, label="Bullish W-Pattern", zorder=5)
        
        # Neckline (P1 to P2)
        ax.plot([dates[p1], dates[p2]], [pattern["p1_price"], pattern["p2_price"]], 
               'g--', linewidth=2, alpha=0.8, label="Bullish Neckline", zorder=6)
        
        # Pattern statistics
        stats_text = f"""BULLISH W-PATTERN:
Drop 1: {pattern.get('drop1_pct', 0):.1%} (P1→L1)
Recovery: {pattern.get('recovery_pct', 0):.1%} (L1→P2) 
Drop 2: {pattern.get('drop2_pct', 0):.1%} (P2→L2)
Neckline: {pattern.get('neckline_slope_pct', 0):.1%} (P1→P2)
Trough Similarity: {pattern.get('trough_similarity', 0):.1%}
Duration: {pattern.get('pattern_days', 0)} days
Quality: {pattern.get('quality_score', 0):.1%}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.9),
               verticalalignment='top', fontsize=9, family='monospace')
        
        # Title with ML scores
        title = f"{ticker} Enhanced Bullish W-Pattern"
        
        scores = []
        for model_type in ["shape", "breakout", "reachtarget"]:
            score_key = f"ml_{model_type}_proba"
            if score_key in pattern and pattern[score_key] > 0:
                scores.append(f"{model_type}: {pattern[score_key]:.3f}")
        
        if scores:
            title += f"\nML Scores: {' | '.join(scores)}"
            
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{ticker}_Enhanced_W_{pattern['p1_date']}_{pattern['low2_date']}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        print(f"Plot error for {ticker}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Enhanced W-pattern scanner with moderate criteria")
    parser.add_argument('--limit', type=int, default=15, help='Number of tickers to scan')
    parser.add_argument('--tickers', help='Specific tickers (comma-separated)')
    parser.add_argument('--min-score', type=float, default=0.5, help='Minimum ML score')
    parser.add_argument('--plot', action='store_true', help='Create plots')
    parser.add_argument('--plot-dir', default='enhanced_w_patterns', help='Plot directory')
    parser.add_argument('--output', default='enhanced_w_results.csv', help='Output file')
    args = parser.parse_args()
    
    print("=== ENHANCED W-PATTERN SCANNER ===")
    print("MODERATE CRITERIA:")
    print(f"• Complete 4-point structure: P1 → L1 → P2 → L2")
    print(f"• Bullish neckline: P2 < P1 (required)")
    print(f"• Min Drop 1: {MIN_DROP1_PCT:.1%}")
    print(f"• Min Recovery: {MIN_RECOVERY_PCT:.1%}")
    print(f"• Min Drop 2: {MIN_DROP2_PCT:.1%}")
    print(f"• Max Trough Diff: {MAX_TROUGH_DIFF:.1%}")
    print(f"• Pattern Duration: {MIN_PATTERN_DAYS}-{MAX_PATTERN_DAYS} days")
    
    # Load models
    models = load_models()
    if models:
        model_names = [k for k in models.keys() if not k.endswith('_features')]
        print(f"ML Models: {model_names}")
    
    # Get tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        tickers = get_sp500_tickers(args.limit)
    
    print(f"\nScanning {len(tickers)} tickers for enhanced bullish W-patterns...")
    
    # Scan
    all_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(scan_ticker, ticker, models): ticker 
                           for ticker in tickers}
        
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results = future.result()
                if results:
                    print(f"\n{ticker}: Found {len(results)} bullish W-patterns")
                    all_results.extend(results)
                else:
                    print(f"{ticker}: No patterns found")
            except Exception as e:
                print(f"{ticker}: Error - {e}")
    
    if not all_results:
        print(f"\n❌ No bullish W-patterns found with current criteria!")
        print("Try:")
        print("1. Reducing --min-score threshold")
        print("2. Scanning more tickers with --limit")
        print("3. Using specific tickers with --tickers")
        return
    
    # Process results
    df_results = pd.DataFrame(all_results)
    print(f"\n✅ Found {len(df_results)} enhanced bullish W-patterns")
    
    # Apply ML score filter
    score_cols = [col for col in df_results.columns if col.startswith('ml_') and col.endswith('_proba')]
    if score_cols and args.min_score > 0:
        mask = df_results[score_cols].fillna(0).max(axis=1) >= args.min_score
        df_results = df_results[mask].copy()
        print(f"After ML filtering (min score {args.min_score}): {len(df_results)} patterns")
    
    if df_results.empty:
        print("No patterns meet ML score threshold!")
        return
    
    # Sort by quality
    if 'quality_score' in df_results.columns:
        df_results = df_results.sort_values('quality_score', ascending=False)
    elif score_cols:
        df_results['best_score'] = df_results[score_cols].fillna(0).max(axis=1)
        df_results = df_results.sort_values('best_score', ascending=False)
    
    # Create plots
    plot_count = 0
    if args.plot and len(df_results) > 0:
        print(f"\nCreating plots in {args.plot_dir}...")
        for _, row in df_results.iterrows():
            df = robust_history(row['ticker'])
            if df is not None:
                plot_path = create_plot(df, row, row['ticker'], args.plot_dir)
                if plot_path:
                    plot_count += 1
                    print(f"  📊 {os.path.basename(plot_path)}")
    
    # Save results
    output_cols = ['ticker', 'p1_date', 'p1_price', 'low1_date', 'low1_price',
                   'p2_date', 'p2_price', 'low2_date', 'low2_price',
                   'drop1_pct', 'recovery_pct', 'drop2_pct', 'neckline_slope_pct',
                   'trough_similarity', 'pattern_days', 'quality_score'] + score_cols
    
    output_cols = [col for col in output_cols if col in df_results.columns]
    df_output = df_results[output_cols].reset_index(drop=True)
    df_output.to_csv(args.output, index=False)
    
    print(f"\n=== RESULTS ===")
    print(f"Enhanced bullish W-patterns: {len(df_results)}")
    print(f"Results saved to: {args.output}")
    
    if plot_count > 0:
        print(f"Plots created: {plot_count} in {args.plot_dir}")
    
    # Show top patterns
    print(f"\n🏆 TOP BULLISH W-PATTERNS:")
    for i, (_, row) in enumerate(df_results.head(5).iterrows(), 1):
        score_info = ""
        if score_cols:
            scores = [f"{col.split('_')[1]}: {row[col]:.3f}" 
                     for col in score_cols if not pd.isna(row[col]) and row[col] > 0]
            score_info = f" | {' | '.join(scores)}" if scores else ""
        
        neckline_info = f"Neckline: {row.get('neckline_slope_pct', 0):.1%}"
        quality_info = f"Quality: {row.get('quality_score', 0):.1%}"
        
        print(f"  {i}. {row['ticker']} ({row['p1_date']} → {row['low2_date']})")
        print(f"     {quality_info}, {neckline_info}{score_info}")

if __name__ == "__main__":
    main()