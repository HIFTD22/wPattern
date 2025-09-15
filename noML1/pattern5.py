#!/usr/bin/env python3
"""
S&P 500 W Pattern Scanner - Data Driven Version
Pure mathematical analysis using Yahoo Finance data
No image training - direct pattern detection on price series
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from datetime import datetime, timedelta
import json
import os
import warnings
from typing import List, Dict, Tuple, Optional
import requests
import io

warnings.filterwarnings('ignore')

class SP500WPatternScanner:
    def __init__(self, lookback_days=90, min_confidence=0.6):
        """
        Initialize the W Pattern Scanner
        
        Args:
            lookback_days (int): Days of historical data to analyze
            min_confidence (float): Minimum confidence threshold for patterns
        """
        print("🚀 Initializing S&P 500 W Pattern Scanner (Data-Driven)")
        print("=" * 60)
        
        self.lookback_days = lookback_days
        self.min_confidence = min_confidence
        self.results_folder = "w_pattern_results"
        
        # W Pattern Criteria
        self.valley_height_tolerance = 0.20  # 20% max difference between valleys
        self.min_valley_separation = 10      # Minimum days between valleys
        self.min_pattern_width = 20          # Minimum pattern width in days
        self.p1_ge_tolerance = 0.02          # Allow P1 up to 2% below P2
        
        # Create results folder
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
            
        # Load S&P 500 symbols
        self.sp500_symbols = self._load_sp500_symbols()
        print(f"📊 Loaded {len(self.sp500_symbols)} S&P 500 symbols")
        print(f"📅 Analysis period: {lookback_days} days")
        print(f"🎯 Confidence threshold: {min_confidence:.1%}")
        print("✅ Scanner ready!\n")

    def _load_sp500_symbols(self) -> List[str]:
        """Load S&P 500 symbols from Wikipedia with caching"""
        cache_file = "sp500_symbols.txt"
        
        # Try cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    symbols = [line.strip().upper().replace('.', '-') for line in f if line.strip()]
                if len(symbols) >= 480:
                    print(f"📁 Loaded {len(symbols)} symbols from cache")
                    return symbols
            except Exception as e:
                print(f"⚠️ Cache read error: {e}")
        
        # Fetch from Wikipedia
        try:
            print("🌐 Fetching S&P 500 list from Wikipedia...")
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            tables = pd.read_html(io.StringIO(response.text))
            df = tables[0]
            symbols = df["Symbol"].astype(str).str.strip().str.upper().str.replace('.', '-').tolist()
            symbols = [s for s in symbols if s and s != 'NAN']
            
            # Cache the results
            with open(cache_file, 'w') as f:
                f.write('\n'.join(symbols))
                
            print(f"✅ Fetched and cached {len(symbols)} symbols")
            return symbols
            
        except Exception as e:
            print(f"❌ Failed to fetch S&P 500 list: {e}")
            # Fallback to major stocks
            fallback = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 'UNH', 'JNJ',
                'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE', 'AVGO', 'KO',
                'MRK', 'COST', 'PEP', 'TMO', 'WMT', 'ABT', 'MCD', 'VZ', 'ACN', 'CSCO', 'DHR',
                'ADBE', 'BMY', 'LIN', 'NFLX', 'CRM', 'NKE', 'TXN', 'QCOM', 'RTX', 'WFC', 'UPS'
            ]
            print(f"📊 Using fallback list: {len(fallback)} symbols")
            return fallback

    def scan_sp500_for_w_patterns(self) -> List[Dict]:
        """
        Scan all S&P 500 stocks for W patterns
        
        Returns:
            List of dictionaries containing detected W patterns
        """
        print(f"🔍 SCANNING S&P 500 FOR W PATTERNS")
        print("=" * 60)
        print(f"📊 Analyzing {len(self.sp500_symbols)} stocks...")
        print(f"📅 Period: {self.lookback_days} days")
        print(f"🎯 Min confidence: {self.min_confidence:.1%}\n")
        
        results = []
        scan_stats = {
            'total_scanned': 0,
            'data_available': 0,
            'patterns_found': 0,
            'high_confidence': 0
        }
        
        for i, symbol in enumerate(self.sp500_symbols, 1):
            try:
                print(f"🔍 [{i:3d}/{len(self.sp500_symbols)}] {symbol:6s}...", end=' ')
                
                # Download stock data
                stock_data = self._download_stock_data(symbol)
                scan_stats['total_scanned'] += 1
                
                if stock_data is None or len(stock_data) < 30:
                    print("❌ Insufficient data")
                    continue
                    
                scan_stats['data_available'] += 1
                
                # Analyze for W patterns
                patterns = self._detect_w_patterns(stock_data, symbol)
                
                if patterns:
                    scan_stats['patterns_found'] += len(patterns)
                    high_conf_patterns = [p for p in patterns if p['confidence'] >= self.min_confidence]
                    scan_stats['high_confidence'] += len(high_conf_patterns)
                    
                    if high_conf_patterns:
                        print(f"✅ {len(high_conf_patterns)} pattern(s) found!")
                        results.extend(high_conf_patterns)
                        
                        # Show best pattern details
                        best = max(high_conf_patterns, key=lambda x: x['confidence'])
                        current_price = float(stock_data['Close'].iloc[-1])
                        upside = ((best['target_price'] - current_price) / current_price) * 100
                        print(f"     Best: {best['confidence']:.1%} confidence, "
                              f"${current_price:.2f} → ${best['target_price']:.2f} ({upside:+.1f}%)")
                    else:
                        print(f"📊 Pattern found (conf < {self.min_confidence:.1%})")
                else:
                    print("📊 No patterns")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:40]}")
                continue
        
        # Save results
        self._save_scan_results(results, scan_stats)
        
        # Display summary
        self._display_scan_summary(results, scan_stats)
        
        return results

    def _download_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Download stock data from Yahoo Finance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 30)  # Extra buffer
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if len(data) < 30:
                return None
                
            # Clean and prepare data
            data = data.dropna()
            data['Date'] = data.index
            
            return data
            
        except Exception:
            return None

    def _detect_w_patterns(self, data: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        Detect W patterns in stock price data
        
        Args:
            data: Stock price DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            List of detected W patterns
        """
        try:
            prices = data['Close'].values
            dates = data.index
            
            if len(prices) < self.min_pattern_width:
                return []
            
            # Find peaks and valleys
            valleys_idx, _ = find_peaks(-prices, 
                                      prominence=np.std(prices) * 0.3,
                                      distance=self.min_valley_separation)
            
            peaks_idx, _ = find_peaks(prices, 
                                    prominence=np.std(prices) * 0.3,
                                    distance=self.min_valley_separation // 2)
            
            if len(valleys_idx) < 2 or len(peaks_idx) < 2:
                return []
            
            patterns = []
            
            # Check all valley pairs for W patterns
            for i in range(len(valleys_idx) - 1):
                for j in range(i + 1, len(valleys_idx)):
                    l1_idx, l2_idx = valleys_idx[i], valleys_idx[j]
                    
                    # Check minimum separation
                    if l2_idx - l1_idx < self.min_pattern_width:
                        continue
                    
                    pattern = self._validate_w_pattern(
                        l1_idx, l2_idx, peaks_idx, prices, dates, symbol
                    )
                    
                    if pattern and pattern['confidence'] >= 0.3:  # Lower threshold for detection
                        patterns.append(pattern)
            
            # Return only the best patterns (avoid duplicates)
            if patterns:
                patterns.sort(key=lambda x: x['confidence'], reverse=True)
                return patterns[:3]  # Top 3 patterns max per stock
            
            return []
            
        except Exception:
            return []

    def _validate_w_pattern(self, l1_idx: int, l2_idx: int, peaks_idx: np.ndarray, 
                           prices: np.ndarray, dates: pd.DatetimeIndex, symbol: str) -> Optional[Dict]:
        """
        Validate if the formation qualifies as a W pattern
        
        Returns:
            Dictionary with pattern details if valid, None otherwise
        """
        try:
            l1_price = prices[l1_idx]
            l2_price = prices[l2_idx]
            
            # Check valley height similarity
            height_diff = abs(l1_price - l2_price) / max(l1_price, l2_price)
            if height_diff > self.valley_height_tolerance:
                return None
            
            # Find peaks between and around valleys
            peaks_between = peaks_idx[(peaks_idx > l1_idx) & (peaks_idx < l2_idx)]
            peaks_before = peaks_idx[peaks_idx < l1_idx]
            
            if len(peaks_between) == 0 or len(peaks_before) == 0:
                return None
            
            # Get the relevant peaks
            p1_idx = peaks_before[-1]  # Last peak before L1
            p2_idx = peaks_between[np.argmax(prices[peaks_between])]  # Highest peak between valleys
            
            p1_price = prices[p1_idx]
            p2_price = prices[p2_idx]
            
            # Apply tilt rule: P1 >= P2 (with tolerance)
            if p1_price < p2_price * (1.0 - self.p1_ge_tolerance):
                return None
            
            # Check that neckline (P2) is above valleys
            if p2_price <= max(l1_price, l2_price):
                return None
            
            # Calculate pattern metrics
            pattern_width_days = l2_idx - l1_idx
            pattern_height = p2_price - min(l1_price, l2_price)
            valley_symmetry = 1.0 - height_diff
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                valley_symmetry, pattern_width_days, pattern_height, 
                p1_price, p2_price, l1_price, l2_price
            )
            
            if confidence < 0.3:
                return None
            
            # Calculate price target using the provided formula
            target_data = self._calculate_price_target(
                l1_price, l2_price, p1_price, p2_price, 
                p1_idx, p2_idx, prices, dates
            )
            
            # Current stock metrics
            current_price = prices[-1]
            current_date = dates[-1]
            upside_potential = ((target_data['target_price'] - current_price) / current_price) * 100
            
            # Pattern completion status
            pattern_completion = min(1.0, (len(prices) - l2_idx) / (pattern_width_days * 0.5))
            
            return {
                'symbol': symbol,
                'confidence': confidence,
                'pattern_completion': pattern_completion,
                
                # Key points
                'l1_date': dates[l1_idx].strftime('%Y-%m-%d'),
                'l1_price': float(l1_price),
                'l2_date': dates[l2_idx].strftime('%Y-%m-%d'), 
                'l2_price': float(l2_price),
                'p1_date': dates[p1_idx].strftime('%Y-%m-%d'),
                'p1_price': float(p1_price),
                'p2_date': dates[p2_idx].strftime('%Y-%m-%d'),
                'p2_price': float(p2_price),
                
                # Pattern characteristics
                'pattern_width_days': int(pattern_width_days),
                'pattern_height': float(pattern_height),
                'valley_symmetry': float(valley_symmetry),
                'height_difference_pct': float(height_diff * 100),
                
                # Target calculation
                **target_data,
                
                # Current status
                'current_price': float(current_price),
                'current_date': current_date.strftime('%Y-%m-%d'),
                'upside_potential_pct': float(upside_potential),
                'risk_reward_ratio': float(abs(upside_potential) / 8.0),  # Assuming 8% stop loss
                
                # Technical indicators
                'volume_confirmation': self._check_volume_confirmation(l1_idx, l2_idx, p2_idx, prices),
                'trend_alignment': self._check_trend_alignment(prices[-20:]),
                
                'scan_timestamp': datetime.now().isoformat()
            }
            
        except Exception:
            return None

    def _calculate_confidence(self, valley_symmetry: float, width_days: int, 
                            height: float, p1: float, p2: float, l1: float, l2: float) -> float:
        """Calculate pattern confidence score"""
        
        # Valley symmetry factor (0-1)
        symmetry_score = valley_symmetry
        
        # Pattern proportions
        avg_valley = (l1 + l2) / 2
        height_ratio = height / avg_valley if avg_valley > 0 else 0
        proportion_score = min(1.0, height_ratio / 0.15)  # Target ~15% pattern height
        
        # Width appropriateness (prefer 20-60 day patterns)
        width_score = 1.0
        if width_days < 20:
            width_score = width_days / 20
        elif width_days > 60:
            width_score = max(0.3, 60 / width_days)
        
        # Peak alignment (slight preference for P1 > P2)
        peak_ratio = p1 / p2 if p2 > 0 else 1
        peak_score = min(1.0, peak_ratio)
        
        # Weighted combination
        confidence = (
            symmetry_score * 0.35 +      # Valley similarity most important
            proportion_score * 0.25 +     # Pattern size
            width_score * 0.25 +          # Time duration
            peak_score * 0.15            # Peak alignment
        )
        
        return max(0, min(1, confidence))

    def _calculate_price_target(self, l1_price: float, l2_price: float, 
                              p1_price: float, p2_price: float,
                              p1_idx: int, p2_idx: int, 
                              prices: np.ndarray, dates: pd.DatetimeIndex) -> Dict:
        """Calculate price target using the specific W pattern formula"""
        
        # Trough midpoint
        trough_midpoint = (l1_price + l2_price) / 2.0
        
        # Target magnitude
        target_magnitude = p2_price - trough_midpoint
        
        # Neckline slope calculation
        bars_diff = max(1, p2_idx - p1_idx)
        neckline_slope_per_bar = (p2_price - p1_price) / bars_diff
        
        # Breakout estimate
        breakout_estimate_price = p2_price + neckline_slope_per_bar * bars_diff
        
        # Final target
        target_price = breakout_estimate_price + target_magnitude
        
        # Estimate timing
        breakout_estimate_days = bars_diff
        try:
            breakout_estimate_date = (dates[p2_idx] + pd.Timedelta(days=breakout_estimate_days)).strftime('%Y-%m-%d')
        except:
            breakout_estimate_date = "TBD"
        
        return {
            'trough_midpoint': float(trough_midpoint),
            'target_magnitude': float(target_magnitude),
            'neckline_slope_per_day': float(neckline_slope_per_bar),
            'breakout_estimate_price': float(breakout_estimate_price),
            'breakout_estimate_days': int(breakout_estimate_days),
            'breakout_estimate_date': breakout_estimate_date,
            'target_price': float(target_price)
        }

    def _check_volume_confirmation(self, l1_idx: int, l2_idx: int, p2_idx: int, prices: np.ndarray) -> str:
        """Check volume confirmation (placeholder - would need volume data)"""
        return "Not analyzed"  # Would implement with volume data

    def _check_trend_alignment(self, recent_prices: np.ndarray) -> str:
        """Check overall trend alignment"""
        if len(recent_prices) < 10:
            return "Insufficient data"
        
        start_price = recent_prices[0]
        end_price = recent_prices[-1]
        change_pct = ((end_price - start_price) / start_price) * 100
        
        if change_pct > 5:
            return "Uptrend"
        elif change_pct < -5:
            return "Downtrend"
        else:
            return "Sideways"

    def _save_scan_results(self, results: List[Dict], stats: Dict):
        """Save scan results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"w_pattern_scan_{timestamp}.json"
        filepath = os.path.join(self.results_folder, filename)
        
        scan_data = {
            'scan_metadata': {
                'timestamp': datetime.now().isoformat(),
                'lookback_days': self.lookback_days,
                'min_confidence': self.min_confidence,
                'statistics': stats
            },
            'patterns_found': results
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(scan_data, f, indent=2, default=str)
            print(f"\n💾 Results saved: {filepath}")
        except Exception as e:
            print(f"\n⚠️ Failed to save results: {e}")

    def _display_scan_summary(self, results: List[Dict], stats: Dict):
        """Display formatted scan results summary"""
        print(f"\n" + "=" * 70)
        print("📊 SCAN RESULTS SUMMARY")
        print("=" * 70)
        print(f"📈 Stocks scanned: {stats['total_scanned']}")
        print(f"📊 Data available: {stats['data_available']}")
        print(f"🔍 Patterns found: {stats['patterns_found']}")
        print(f"⭐ High confidence: {stats['high_confidence']}")
        
        if results:
            print(f"\n🎯 TOP W PATTERN CANDIDATES:")
            print("-" * 70)
            print(f"{'#':>2} {'Symbol':>6} {'Conf':>5} {'Current':>8} {'Target':>8} {'Upside':>7} {'Days':>5}")
            print("-" * 70)
            
            # Sort by confidence, then by upside potential
            sorted_results = sorted(results, 
                                  key=lambda x: (x['confidence'], x['upside_potential_pct']), 
                                  reverse=True)
            
            for i, pattern in enumerate(sorted_results[:15], 1):
                print(f"{i:2d} {pattern['symbol']:>6} "
                      f"{pattern['confidence']:5.1%} "
                      f"${pattern['current_price']:7.2f} "
                      f"${pattern['target_price']:7.2f} "
                      f"{pattern['upside_potential_pct']:6.1f}% "
                      f"{pattern['pattern_width_days']:4d}")
                      
            if len(results) > 15:
                print(f"\n... and {len(results) - 15} more patterns")
                
            # Quick stats
            avg_upside = np.mean([p['upside_potential_pct'] for p in results])
            avg_confidence = np.mean([p['confidence'] for p in results])
            print(f"\n📊 Average upside potential: {avg_upside:.1f}%")
            print(f"📊 Average confidence: {avg_confidence:.1%}")
            
        else:
            print(f"\n📊 No high-confidence W patterns found in current scan")
            print(f"💡 Try lowering confidence threshold or increasing lookback period")

    def analyze_specific_stock(self, symbol: str) -> List[Dict]:
        """Analyze a specific stock for W patterns"""
        print(f"\n🔍 ANALYZING {symbol.upper()} FOR W PATTERNS")
        print("=" * 50)
        
        data = self._download_stock_data(symbol.upper())
        if data is None:
            print(f"❌ Could not fetch data for {symbol}")
            return []
        
        patterns = self._detect_w_patterns(data, symbol.upper())
        
        if patterns:
            print(f"✅ Found {len(patterns)} pattern(s)")
            for i, pattern in enumerate(patterns, 1):
                print(f"\n📊 Pattern {i}:")
                print(f"   Confidence: {pattern['confidence']:.1%}")
                print(f"   Current Price: ${pattern['current_price']:.2f}")
                print(f"   Target Price: ${pattern['target_price']:.2f}")
                print(f"   Upside: {pattern['upside_potential_pct']:+.1f}%")
                print(f"   Pattern Width: {pattern['pattern_width_days']} days")
                print(f"   Valley Dates: {pattern['l1_date']} to {pattern['l2_date']}")
        else:
            print(f"📊 No W patterns detected in {symbol}")
        
        return patterns

def main():
    """Main execution function"""
    print("🎯 S&P 500 W Pattern Scanner - Data Driven")
    print("=" * 60)
    
    # Initialize scanner
    scanner = SP500WPatternScanner(lookback_days=90, min_confidence=0.6)
    
    while True:
        print("\n📋 SCANNER OPTIONS:")
        print("1. Scan entire S&P 500")
        print("2. Analyze specific stock")
        print("3. Adjust settings")
        print("4. Exit")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break
        
        if choice == "1":
            candidates = scanner.scan_sp500_for_w_patterns()
            print(f"\n🎯 Scan complete! Found {len(candidates)} high-confidence patterns")
            
        elif choice == "2":
            symbol = input("Enter stock symbol: ").strip().upper()
            if symbol:
                scanner.analyze_specific_stock(symbol)
            
        elif choice == "3":
            print(f"\nCurrent settings:")
            print(f"  Lookback days: {scanner.lookback_days}")
            print(f"  Min confidence: {scanner.min_confidence:.1%}")
            print(f"  Valley tolerance: {scanner.valley_height_tolerance:.1%}")
            print(f"  P1≥P2 tolerance: {scanner.p1_ge_tolerance:.1%}")
            
            try:
                new_confidence = float(input(f"New min confidence (current {scanner.min_confidence:.1%}): ") or scanner.min_confidence)
                scanner.min_confidence = max(0.1, min(1.0, new_confidence))
                print(f"✅ Updated confidence threshold to {scanner.min_confidence:.1%}")
            except ValueError:
                print("⚠️ Invalid input, keeping current settings")
                
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❓ Invalid choice, please select 1-4")

if __name__ == "__main__":
    main()