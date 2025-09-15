#!/usr/bin/env python3
"""
FRESH W-Pattern Scanner - Complete rewrite with correct logic.
Finds patterns that go: HIGH → LOW → HIGH → LOW (true W-shape)
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

# W-Pattern criteria
MIN_DROP1_PCT = 0.02      # P1 to L1 must drop at least 2%
MIN_RECOVERY_PCT = 0.015  # L1 to P2 must rise at least 1.5%
MIN_DROP2_PCT = 0.01      # P2 to L2 must drop at least 1%
MAX_TROUGH_DIFF = 0.15    # L1 and L2 can differ by max 15%
MIN_PATTERN_DAYS = 5      # Minimum pattern duration
MAX_PATTERN_DAYS = 60     # Maximum pattern duration

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
    return ["AAPL","MSFT","GOOGL","AMZN","COST", "NVDA","TSLA","META","JPM","V","WMT",
            "UNH","MA","PG","HD","KO","PEP","ABBV","BAC","CRM","PFE"][:limit]

def robust_history(ticker: str):
    """Get market data."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="6mo", interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return None
            
        need = ["Open","High","Low","Close","Volume"]
        if not all(c in df.columns for c in need):
            return None
            
        df = df[need].dropna().sort_index()
        return df.tail(RECENT_WINDOW) if len(df) > RECENT_WINDOW else df
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
    
    return df

def find_local_extrema(df: pd.DataFrame, order=5):
    """
    Find local maxima and minima using scipy.
    Returns actual high points as peaks and low points as troughs.
    """
    closes = df['Close'].values
    highs = df['High'].values  
    lows = df['Low'].values
    
    # Find local maxima (peaks) - actual high points
    peak_indices = argrelextrema(highs, np.greater, order=order)[0]
    
    # Find local minima (troughs) - actual low points  
    trough_indices = argrelextrema(lows, np.less, order=order)[0]
    
    # Create pivot list with actual prices
    pivots = []
    
    for idx in peak_indices:
        if 0 <= idx < len(df):
            pivots.append((idx, highs[idx], "PEAK"))  # Actual high points
    
    for idx in trough_indices:
        if 0 <= idx < len(df):
            pivots.append((idx, lows[idx], "TROUGH"))  # Actual low points
            
    return sorted(pivots, key=lambda x: x[0])

def detect_true_w_patterns(df: pd.DataFrame):
    """
    Detect TRUE W-patterns using price movement logic.
    W-pattern: HIGH price → LOW price → HIGH price → LOW price
    """
    df_with_indicators = add_indicators(df)
    n = len(df_with_indicators)
    
    if n < 20:
        return []
    
    # Find all local extrema
    pivots = find_local_extrema(df_with_indicators, order=3)
    
    if len(pivots) < 4:
        return []
    
    # Separate actual peaks (high points) and troughs (low points)
    actual_peaks = [(i, price, idx) for idx, (i, price, label) in enumerate(pivots) if label == "PEAK"]
    actual_troughs = [(i, price, idx) for idx, (i, price, label) in enumerate(pivots) if label == "TROUGH"]
    
    candidates = []
    
    # Look for W-pattern: PEAK → TROUGH → PEAK → TROUGH
    for p1_data in actual_peaks:
        p1_idx, p1_price, p1_pivot_idx = p1_data
        
        # Find troughs after this peak
        later_troughs = [t for t in actual_troughs if t[0] > p1_idx]
        if not later_troughs:
            continue
            
        for l1_data in later_troughs:
            l1_idx, l1_price, l1_pivot_idx = l1_data
            
            # Find peaks after first trough
            later_peaks = [p for p in actual_peaks if p[0] > l1_idx]
            if not later_peaks:
                continue
                
            for p2_data in later_peaks:
                p2_idx, p2_price, p2_pivot_idx = p2_data
                
                # Find troughs after second peak
                final_troughs = [t for t in actual_troughs if t[0] > p2_idx]
                if not final_troughs:
                    continue
                    
                for l2_data in final_troughs:
                    l2_idx, l2_price, l2_pivot_idx = l2_data
                    
                    # Check sequence: P1 → L1 → P2 → L2
                    if not (p1_idx < l1_idx < p2_idx < l2_idx):
                        continue
                    
                    # Check timing
                    pattern_duration = l2_idx - p1_idx
                    if not (MIN_PATTERN_DAYS <= pattern_duration <= MAX_PATTERN_DAYS):
                        continue
                    
                    # VALIDATE TRUE W-PATTERN
                    
                    # 1. Must start HIGH and drop significantly
                    if p1_price <= l1_price:  # Peak must be above trough
                        continue
                    drop1_pct = (p1_price - l1_price) / p1_price
                    if drop1_pct < MIN_DROP1_PCT:
                        continue
                    
                    # 2. Must recover from first trough  
                    if p2_price <= l1_price:  # Recovery peak must be above first trough
                        continue
                    recovery_pct = (p2_price - l1_price) / l1_price
                    if recovery_pct < MIN_RECOVERY_PCT:
                        continue
                    
                    # 3. Must drop from second peak
                    if p2_price <= l2_price:  # Second peak must be above second trough
                        continue
                    drop2_pct = (p2_price - l2_price) / p2_price  
                    if drop2_pct < MIN_DROP2_PCT:
                        continue
                    
                    # 4. Double bottom: troughs should be similar
                    trough_diff = abs(l1_price - l2_price) / min(l1_price, l2_price)
                    if trough_diff > MAX_TROUGH_DIFF:
                        continue
                    
                    # 5. P2 should generally be lower than P1 (weakening trend)
                    if p2_price > p1_price * 1.2:  # Allow some tolerance
                        continue
                    
                    # 6. Verify this is lowest points in vicinity
                    def is_local_minimum(idx, price, window=3):
                        start = max(0, idx - window)
                        end = min(len(df_with_indicators), idx + window + 1)
                        local_lows = df_with_indicators['Low'].iloc[start:end]
                        return price <= local_lows.min() * 1.02  # 2% tolerance
                    
                    if not (is_local_minimum(l1_idx, l1_price) and is_local_minimum(l2_idx, l2_price)):
                        continue
                        
                    # 7. Final validation - check the actual shape makes sense
                    dates = df_with_indicators.index
                    
                    print(f"✓ TRUE W-PATTERN FOUND!")
                    print(f"  {dates[p1_idx].date()} P1: HIGH ${p1_price:.2f}")
                    print(f"  {dates[l1_idx].date()} L1: LOW  ${l1_price:.2f} (drop {drop1_pct:.1%})")
                    print(f"  {dates[p2_idx].date()} P2: HIGH ${p2_price:.2f} (recovery {recovery_pct:.1%})")
                    print(f"  {dates[l2_idx].date()} L2: LOW  ${l2_price:.2f} (drop {drop2_pct:.1%})")
                    print(f"  Trough similarity: {trough_diff:.1%}")
                    
                    # Compute features for ML models
                    features = compute_w_features(df_with_indicators, p1_idx, l1_idx, p2_idx, l2_idx,
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
                        **features
                    }
                    
                    candidates.append(candidate)
                    
                    # Limit to avoid too many overlapping patterns
                    if len(candidates) >= 5:
                        return candidates
    
    return candidates

def compute_w_features(df: pd.DataFrame, p1: int, l1: int, p2: int, l2: int,
                      p1_price: float, l1_price: float, p2_price: float, l2_price: float):
    """Compute features for ML scoring."""
    
    # Basic geometry
    base_price = min(l1_price, l2_price)
    
    features = {
        "drop1_norm": float((l1_price - p1_price) / base_price),  # Negative value
        "rise1_norm": float((p2_price - l1_price) / base_price),  # Positive value  
        "drop2_norm": float((l2_price - p2_price) / base_price),  # Negative value
        "trough_sim": float(abs(l2_price - l1_price) / l1_price),
        "neckline_slope_per_day": float((p2_price - p1_price) / max(1, p2 - p1)),
        
        # Timing
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
            # Defaults
            features[f"rsi_{tag}"] = 50.0
            features[f"vol_ratio_{tag}"] = 1.0
            features[f"px_vs_ma20_{tag}"] = 0.0
            if tag == "l2":
                features[f"px_vs_ma50_{tag}"] = 0.0
    
    # Add enhanced feature defaults (for compatibility with trained models)
    enhanced_defaults = {
        "regime_trend": 0.1, "regime_volatility": 0.0, "regime_volume": 0.0, "regime_strength": 0.5,
        "volume_accumulation_ratio": 1.0, "volume_confirmation_ratio": 1.0, "volume_trend_slope": 0.0,
        "volume_consistency": 0.5, "volume_momentum": 0.0,
        "pattern_duration_balance": 0.7, "pattern_trough_consistency": float(1.0 - abs(l2_price - l1_price) / l1_price), 
        "pattern_peak_relationship": float(p2_price / max(p1_price, 1e-9)),
        "pattern_clarity": 0.6, "pattern_amplitude_balance": 0.7,
        "market_position": 0.3, "trend_context": 0.0, "volatility_context": 0.0, "support_strength": 0.4
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
                
                # Extract features
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
    """Scan single ticker for W patterns."""
    try:
        df = robust_history(ticker)
        if df is None or len(df) < 20:
            return []
        
        patterns = detect_true_w_patterns(df)
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
    """Create W-pattern visualization."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        dates = pd.to_datetime(df.index)
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        
        p1, l1, p2, l2 = pattern["p1_idx"], pattern["low1_idx"], pattern["p2_idx"], pattern["low2_idx"]
        
        plt.figure(figsize=(12, 8))
        
        # Plot price with high/low bands
        plt.fill_between(dates, low, high, alpha=0.1, color='gray', label='Daily Range')
        plt.plot(dates, close, 'b-', linewidth=1.5, alpha=0.8, label="Close Price")
        
        # Mark W pattern points
        pattern_dates = [dates[p1], dates[l1], dates[p2], dates[l2]]
        pattern_prices = [pattern["p1_price"], pattern["low1_price"], pattern["p2_price"], pattern["low2_price"]]
        pattern_labels = ["P1 (HIGH)", "L1 (LOW)", "P2 (HIGH)", "L2 (LOW)"]
        colors = ['red', 'green', 'orange', 'blue']
        
        plt.scatter(pattern_dates, pattern_prices, c=colors, s=120, zorder=10, edgecolors='black', linewidth=1)
        
        # Add labels with price info
        for i, (date, price, label, color) in enumerate(zip(pattern_dates, pattern_prices, pattern_labels, colors)):
            plt.annotate(f'{label}\n${price:.2f}', (date, price), 
                        xytext=(10, 15 if i % 2 == 0 else -25), textcoords="offset points", 
                        fontweight='bold', fontsize=9, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        # Draw W pattern lines
        w_dates = pattern_dates
        w_prices = pattern_prices
        plt.plot(w_dates, w_prices, 'r--', linewidth=3, alpha=0.8, label="W-Pattern", zorder=5)
        
        # Add pattern statistics
        stats_text = f"""W-PATTERN ANALYSIS:
Drop 1: {pattern.get('drop1_pct', 0):.1%} (P1→L1)
Recovery: {pattern.get('recovery_pct', 0):.1%} (L1→P2) 
Drop 2: {pattern.get('drop2_pct', 0):.1%} (P2→L2)
Trough Similarity: {pattern.get('trough_similarity', 0):.1%}
Pattern Duration: {pattern.get('pattern_days', 0)} days"""
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8),
                verticalalignment='top', fontsize=8, family='monospace')
        
        # Title with ML scores
        title = f"{ticker} W-Pattern Detection"
        
        scores = []
        for model_type in ["shape", "breakout", "reachtarget"]:
            score_key = f"ml_{model_type}_proba"
            if score_key in pattern and pattern[score_key] > 0:
                scores.append(f"{model_type}: {pattern[score_key]:.2f}")
        
        if scores:
            title += f"\nML Scores: {' | '.join(scores)}"
            
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price ($)", fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        filename = f"{ticker}_W_{pattern['p1_date']}_{pattern['low2_date']}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        print(f"Plot error for {ticker}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Fresh W-pattern scanner with correct logic")
    parser.add_argument('--limit', type=int, default=10, help='Number of tickers to scan')
    parser.add_argument('--tickers', help='Specific tickers (comma-separated)')
    parser.add_argument('--min-score', type=float, default=0.5, help='Minimum ML score')
    parser.add_argument('--plot', action='store_true', help='Create plots')
    parser.add_argument('--plot-dir', default='true_w_patterns', help='Plot directory')
    parser.add_argument('--output', default='true_w_results.csv', help='Output file')
    args = parser.parse_args()
    
    print("=== FRESH W-PATTERN SCANNER ===")
    print(f"Looking for patterns: HIGH → LOW → HIGH → LOW")
    print(f"Min drops: {MIN_DROP1_PCT:.1%}, {MIN_DROP2_PCT:.1%}")
    print(f"Min recovery: {MIN_RECOVERY_PCT:.1%}")
    print(f"Max trough difference: {MAX_TROUGH_DIFF:.1%}")
    
    # Load models
    models = load_models()
    if models:
        model_names = [k for k in models.keys() if not k.endswith('_features')]
        print(f"Loaded ML models: {model_names}")
    else:
        print("No ML models found - pattern detection only")
    
    # Get tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        tickers = get_sp500_tickers(args.limit)
    
    print(f"\nScanning {len(tickers)} tickers...")
    
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
                    print(f"\n{ticker}: Found {len(results)} W-patterns")
                    all_results.extend(results)
                else:
                    print(f"{ticker}: No W-patterns")
            except Exception as e:
                print(f"{ticker}: Error - {e}")
    
    if not all_results:
        print("\n❌ No true W-patterns found!")
        print("This could mean:")
        print("1. Current market doesn't have clear W-patterns")  
        print("2. Criteria are too strict - try lowering MIN_DROP1_PCT")
        print("3. Time window is too short - try longer history")
        return
    
    # Convert to DataFrame and filter
    df_results = pd.DataFrame(all_results)
    print(f"\n✅ Found {len(df_results)} total W-patterns")
    
    # Apply ML score filter
    score_cols = [col for col in df_results.columns if col.startswith('ml_') and col.endswith('_proba')]
    if score_cols and args.min_score > 0:
        mask = df_results[score_cols].fillna(0).max(axis=1) >= args.min_score
        df_results = df_results[mask].copy()
        print(f"After ML filtering (min score {args.min_score}): {len(df_results)} patterns")
    
    if df_results.empty:
        print("No patterns meet ML score threshold!")
        return
    
    # Sort by best score or pattern quality
    if score_cols:
        df_results['best_score'] = df_results[score_cols].fillna(0).max(axis=1)
        df_results = df_results.sort_values('best_score', ascending=False)
    else:
        df_results = df_results.sort_values(['drop1_pct', 'recovery_pct'], ascending=[False, False])
    
    # Create plots
    plot_count = 0
    if args.plot and len(df_results) > 0:
        print(f"\nCreating plots in {args.plot_dir}...")
        for _, row in df_results.head(10).iterrows():
            df = robust_history(row['ticker'])
            if df is not None:
                plot_path = create_plot(df, row, row['ticker'], args.plot_dir)
                if plot_path:
                    plot_count += 1
                    print(f"  Created: {os.path.basename(plot_path)}")
    
    # Save results
    output_cols = ['ticker', 'p1_date', 'p1_price', 'low1_date', 'low1_price',
                   'p2_date', 'p2_price', 'low2_date', 'low2_price',
                   'drop1_pct', 'recovery_pct', 'drop2_pct', 'trough_similarity', 'pattern_days'] + score_cols
    
    output_cols = [col for col in output_cols if col in df_results.columns]
    df_output = df_results[output_cols].reset_index(drop=True)
    df_output.to_csv(args.output, index=False)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"True W-patterns found: {len(df_results)}")
    print(f"Results saved to: {args.output}")
    
    if plot_count > 0:
        print(f"Created {plot_count} plots in: {args.plot_dir}")
    
    # Show top patterns
    print(f"\n🏆 TOP W-PATTERNS:")
    for i, (_, row) in enumerate(df_results.head(5).iterrows(), 1):
        score_info = ""
        if score_cols:
            scores = [f"{col.split('_')[1]}: {row[col]:.2f}" 
                     for col in score_cols if not pd.isna(row[col]) and row[col] > 0]
            score_info = f" | {' | '.join(scores)}" if scores else ""
        
        pattern_info = f"Drop1: {row.get('drop1_pct', 0):.1%}, Recovery: {row.get('recovery_pct', 0):.1%}, Drop2: {row.get('drop2_pct', 0):.1%}"
        
        print(f"  {i}. {row['ticker']} ({row['p1_date']} → {row['low2_date']})")
        print(f"     {pattern_info}{score_info}")

if __name__ == "__main__":
    main()