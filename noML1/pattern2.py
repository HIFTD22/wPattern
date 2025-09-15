import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class AAPLFlagDetector:
    def __init__(self, period="10d", interval="30m"):
        """
        Initialize with AAPL 30-minute data using HIGH/LOW prices
        """
        self.symbol = "AAPL"
        self.interval = interval
        
        # Download AAPL data
        print(f"Downloading {self.symbol} data for {period} at {interval} intervals...")
        self.data = yf.download(self.symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        
        if self.data.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        # Clean the data
        self.data = self.data.dropna()
        
        if self.data.empty:
            raise ValueError(f"No valid data after cleaning for {self.symbol}")
        
        # Filter to regular market hours only (9:30 AM - 4:00 PM ET)
        self.data = self.filter_regular_hours(self.data)
        
        if self.data.empty:
            raise ValueError(f"No data during regular market hours for {self.symbol}")
        
        # Extract prices
        self.opens  = self.data['Open'].values
        self.highs  = self.data['High'].values
        self.lows   = self.data['Low'].values
        self.closes = self.data['Close'].values
        self.volumes = self.data['Volume'].values if 'Volume' in self.data.columns else None
        self.dates = self.data.index
        
        # Ensure arrays are 1-D
        for attr in ["opens","highs","lows","closes"]:
            arr = getattr(self, attr)
            if arr.ndim > 1:
                setattr(self, attr, arr.flatten())

        # NEW: OC-based effective prices for W pivots
        self.oc_peak = np.maximum(self.opens, self.closes)  # use for P1/P2 price
        self.oc_trough = np.minimum(self.opens, self.closes)  # use for L1/L2 price
        
        print(f"Loaded {len(self.highs)} candles for {self.symbol} (regular hours only)")
        print(f"Date range: {self.dates[0].strftime('%Y-%m-%d %H:%M')} to {self.dates[-1].strftime('%Y-%m-%d %H:%M')}")
    
    def filter_regular_hours(self, data):
        """
        Filter data to include only regular market hours (9:30 AM - 4:00 PM ET)
        """
        # Convert index to Eastern Time if not already
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert('US/Eastern')
        elif str(data.index.tz) != 'US/Eastern':
            data.index = data.index.tz_convert('US/Eastern')
        
        # Filter to regular market hours: 9:30 AM to 4:00 PM
        regular_hours = data.between_time('09:30', '16:00')
        
        print(f"Filtered from {len(data)} total candles to {len(regular_hours)} regular market hours candles")
        return regular_hours
    
    def find_flag_patterns(self):
        """
        Simplified approach: Find flag-like channels without requiring flagpoles first
        """
        print("Finding flag-like channels...")
        
        flag_patterns = []
        min_periods = 8
        max_periods = 20
        
        # Try different window sizes
        for window_size in range(min_periods, max_periods + 1):
            for start_idx in range(len(self.closes) - window_size):
                end_idx = start_idx + window_size
                
                # Simple channel analysis
                if end_idx - start_idx >= 8:  # Need minimum 8 periods
                    window_highs = self.highs[start_idx:end_idx + 1]
                    window_lows = self.lows[start_idx:end_idx + 1]
                    window_closes = self.closes[start_idx:end_idx + 1]
                    
                    # Create time indices
                    time_indices = np.arange(len(window_highs))
                    
                    # Fit lines to highs and lows
                    try:
                        high_slope, high_intercept, high_r_value, _, _ = stats.linregress(time_indices, window_highs)
                        low_slope, low_intercept, low_r_value, _, _ = stats.linregress(time_indices, window_lows)
                        
                        # Calculate how well the channel contains price action
                        upper_line = high_slope * time_indices + high_intercept
                        lower_line = low_slope * time_indices + low_intercept
                        
                        # Check containment
                        closes_in_channel = sum(1 for i, close in enumerate(window_closes) 
                                              if lower_line[i] <= close <= upper_line[i])
                        containment_ratio = closes_in_channel / len(window_closes)
                        
                        # Check if slopes are roughly parallel
                        if abs(high_slope) > 0.001 and abs(low_slope) > 0.001:
                            slope_similarity = min(abs(high_slope), abs(low_slope)) / max(abs(high_slope), abs(low_slope))
                        else:
                            slope_similarity = 1.0 if abs(high_slope) < 0.001 and abs(low_slope) < 0.001 else 0.7
                        
                        # Calculate average channel width
                        avg_price = np.mean(window_closes)
                        channel_width_start = (high_intercept - low_intercept) / avg_price
                        channel_width_end = ((high_slope * (len(time_indices) - 1) + high_intercept) - 
                                           (low_slope * (len(time_indices) - 1) + low_intercept)) / avg_price
                        avg_channel_width = (channel_width_start + channel_width_end) / 2
                        
                        # R-squared average
                        r_squared_avg = (high_r_value**2 + low_r_value**2) / 2
                        
                        # Quality score
                        quality_score = (containment_ratio * 0.5 + 
                                       slope_similarity * 0.3 + 
                                       r_squared_avg * 0.2)
                        
                        # Only keep high-quality channels with good containment
                        if (containment_ratio > 0.75 and 
                            quality_score > 0.7 and
                            slope_similarity > 0.6 and
                            0.005 <= avg_channel_width <= 0.1):
                            
                            avg_slope = (high_slope + low_slope) / 2
                            
                            if avg_slope < -0.5:  # Significant downward slope
                                pattern_type = 'Bullish Flag'
                                channel_direction = 'down'
                            elif avg_slope > 0.5:  # Significant upward slope
                                pattern_type = 'Bearish Flag'
                                channel_direction = 'up'
                            else:
                                continue  # Skip sideways patterns
                            
                            # Check for overlaps with existing patterns
                            overlaps = False
                            for existing in flag_patterns:
                                overlap_start = max(start_idx, existing['start_idx'])
                                overlap_end = min(end_idx, existing['end_idx'])
                                if overlap_end > overlap_start:
                                    overlap_length = overlap_end - overlap_start
                                    current_length = end_idx - start_idx
                                    if overlap_length / current_length > 0.5:
                                        overlaps = True
                                        break
                            
                            if not overlaps:
                                print(f"Found {pattern_type}: {self.dates[start_idx].strftime('%m-%d %H:%M')} to {self.dates[end_idx].strftime('%m-%d %H:%M')}")
                                print(f"  Containment: {containment_ratio:.1%}, Quality: {quality_score:.1%}")
                                print(f"  Channel width: {avg_channel_width:.1%}, Slope similarity: {slope_similarity:.1%}")
                                
                                flag_patterns.append({
                                    'type': pattern_type,
                                    'start_idx': start_idx,
                                    'end_idx': end_idx,
                                    'channel_direction': channel_direction,
                                    'containment_ratio': containment_ratio,
                                    'quality_score': quality_score,
                                    'avg_slope': avg_slope,
                                    'high_slope': high_slope,
                                    'low_slope': low_slope,
                                    'high_intercept': high_intercept,
                                    'low_intercept': low_intercept,
                                    'channel_width': avg_channel_width,
                                    'slope_similarity': slope_similarity,
                                    'r_squared_avg': r_squared_avg
                                })
                            
                    except Exception as e:
                        continue
        
        # Sort by quality score
        flag_patterns.sort(key=lambda x: x['quality_score'], reverse=True)
        
        print(f"Found {len(flag_patterns)} high-quality flag channels")
        return flag_patterns
    
    def detect_channel_breakout(self, pattern, lookout_periods=6):
        """
        Detect breakout from the flag channel using the trendlines
        """
        end_idx = pattern['end_idx']
        
        if end_idx + lookout_periods >= len(self.closes):
            return {
                'direction': 'none',
                'magnitude': 0,
                'is_success': False,
                'type': 'insufficient_data'
            }
        
        # Get the channel trendlines at the end of the pattern
        pattern_length = pattern['end_idx'] - pattern['start_idx']
        
        # Calculate future trendline projections
        high_slope = pattern['high_slope']
        low_slope = pattern['low_slope']
        high_intercept = pattern['high_intercept']
        low_intercept = pattern['low_intercept']
        
        # Look for breakouts in the next periods
        for i in range(1, lookout_periods + 1):
            check_idx = end_idx + i
            if check_idx >= len(self.highs):
                break
            
            # Project trendlines forward
            time_from_pattern_start = pattern_length + i
            projected_upper = high_slope * time_from_pattern_start + high_intercept
            projected_lower = low_slope * time_from_pattern_start + low_intercept
            
            actual_high = self.highs[check_idx]
            actual_low = self.lows[check_idx]
            
            # Check for breakout above upper trendline (bullish)
            if actual_high > projected_upper:
                breakout_magnitude = (actual_high - projected_upper) / projected_upper
                return {
                    'direction': 'up',
                    'magnitude': breakout_magnitude,
                    'breakout_price': actual_high,
                    'expected_resistance': projected_upper,
                    'breakout_time': self.dates[check_idx],
                    'is_success': True,
                    'type': 'channel_breakout_up'
                }
            
            # Check for breakout below lower trendline (bearish)
            if actual_low < projected_lower:
                breakout_magnitude = (actual_low - projected_lower) / projected_lower
                return {
                    'direction': 'down',
                    'magnitude': breakout_magnitude,
                    'breakout_price': actual_low,
                    'expected_support': projected_lower,
                    'breakout_time': self.dates[check_idx],
                    'is_success': True,
                    'type': 'channel_breakout_down'
                }
        
        # No breakout found
        return {
            'direction': 'none',
            'magnitude': 0,
            'is_success': False,
            'type': 'no_channel_breakout'
        }
    
    def print_pattern_summary(self, patterns):
        """
        Print a summary of all found flag patterns
        """
        if not patterns:
            print("No flag patterns found.")
            return
        
        print(f"\n" + "="*80)
        print(f"FOUND {len(patterns)} FLAG CHANNEL PATTERNS:")
        print("="*80)
        
        successful_patterns = 0
        
        for i, pattern in enumerate(patterns, 1):
            # Check for breakout
            breakout = self.detect_channel_breakout(pattern)
            
            status = "✓ SUCCESS" if breakout['is_success'] else "✗ NO BREAKOUT"
            if breakout['is_success']:
                successful_patterns += 1
            
            print(f"\nPattern {i}: {pattern['type']} - {status}")
            
            start_time = self.dates[pattern['start_idx']]
            end_time = self.dates[pattern['end_idx']]
            duration = pattern['end_idx'] - pattern['start_idx']
            
            print(f"  Time period: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Duration: {duration} periods ({duration * 0.5:.1f} hours)")
            print(f"  Channel direction: {pattern['channel_direction']} slope")
            print(f"  Quality score: {pattern['quality_score']:.1%}")
            print(f"  Containment ratio: {pattern['containment_ratio']:.1%}")
            print(f"  Channel width: {pattern['channel_width']:.1%}")
            print(f"  Slope similarity: {pattern['slope_similarity']:.1%}")
            
            if breakout['is_success']:
                print(f"  Breakout: {breakout['magnitude']:.1%} {breakout['direction']} at {breakout['breakout_time'].strftime('%Y-%m-%d %H:%M')}")
        
        success_rate = successful_patterns / len(patterns) if patterns else 0
        print(f"\nOVERALL SUCCESS RATE: {successful_patterns}/{len(patterns)} ({success_rate:.1%})")
        
        if patterns:
            avg_duration = np.mean([p['end_idx'] - p['start_idx'] for p in patterns])
            avg_quality = np.mean([p['quality_score'] for p in patterns])
            avg_containment = np.mean([p['containment_ratio'] for p in patterns])
            avg_width = np.mean([p['channel_width'] for p in patterns])
            
            print(f"\nPATTERN STATISTICS:")
            print(f"  Average duration: {avg_duration:.1f} periods ({avg_duration * 0.5:.1f} hours)")
            print(f"  Average quality score: {avg_quality:.1%}")
            print(f"  Average containment ratio: {avg_containment:.1%}")
            print(f"  Average channel width: {avg_width:.1%}")
    
    def visualize_patterns(self, patterns=None, show_details=True):
        """
        Visualize the flag patterns with channel lines
        """
        if patterns is None:
            patterns = self.find_flag_patterns()
        
        if not patterns:
            print("No patterns to visualize")
            return
        
        plt.figure(figsize=(20, 12))
        
        # Plot price chart
        plt.plot(self.dates, self.closes, 'k-', alpha=0.6, linewidth=1, label='Close Price')
        
        # Plot each flag pattern
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for i, pattern in enumerate(patterns):
            color = colors[i % len(colors)]
            
            # Get pattern data
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            
            pattern_times = self.dates[start_idx:end_idx + 1]
            pattern_length = len(pattern_times)
            
            # Calculate channel lines
            time_indices = np.arange(pattern_length)
            upper_line = pattern['high_slope'] * time_indices + pattern['high_intercept']
            lower_line = pattern['low_slope'] * time_indices + pattern['low_intercept']
            
            # Draw channel lines
            plt.plot(pattern_times, upper_line, color=color, linestyle='--', linewidth=2, alpha=0.8,
                    label=f'Pattern {i+1}: {pattern["type"]}')
            plt.plot(pattern_times, lower_line, color=color, linestyle='--', linewidth=2, alpha=0.8)
            
            # Fill channel
            plt.fill_between(pattern_times, upper_line, lower_line, color=color, alpha=0.1)
            
            # Mark pattern boundaries
            plt.axvline(x=pattern_times[0], color=color, linestyle=':', alpha=0.5)
            plt.axvline(x=pattern_times[-1], color=color, linestyle=':', alpha=0.5)
            
            # Check for breakout and mark it
            breakout = self.detect_channel_breakout(pattern)
            if breakout['is_success']:
                plt.scatter(breakout['breakout_time'], breakout['breakout_price'], 
                          color=color, s=100, marker='*', zorder=10, 
                          edgecolor='black', linewidth=1)
        
        plt.title(f'AAPL Flag Channel Patterns - 30-Minute Chart ({len(patterns)} patterns found)')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # =========================
    # NEW: W pattern detection
    # =========================
    def _local_extrema_oc(self, win=3):
        """
        Find local maxima of oc_peak and minima of oc_trough using a +/- win neighborhood.
        Returns two sorted index lists: peaks_idx, troughs_idx
        """
        n = len(self.closes)
        if n < 2*win + 1:
            return [], []
        peaks, troughs = [], []
        for i in range(win, n - win):
            seg_hi = self.oc_peak[i-win:i+win+1]
            seg_lo = self.oc_trough[i-win:i+win+1]
            if self.oc_peak[i] == seg_hi.max() and (self.oc_peak[i] > seg_hi[:win].max() or self.oc_peak[i] > seg_hi[win+1:].max()):
                peaks.append(i)
            if self.oc_trough[i] == seg_lo.min() and (self.oc_trough[i] < seg_lo[:win].min() or self.oc_trough[i] < seg_lo[win+1:].min()):
                troughs.append(i)
        return peaks, troughs

    def find_w_patterns(self, win=3, min_sep=3, max_sep=85, p1_ge_tol=0.0,
                        neckline_buffer=0.001, use_high_for_breakout=True, keep_forming=True):
        """
        Detect W sequences P1→L1→P2→L2 using OC-based prices, with your filter P1 >= P2.
        Returns list of dicts with pivots, status (FORMING/COMPLETED), and neckline slope.
        """
        peaks, troughs = self._local_extrema_oc(win=win)
        n = len(self.closes)
        if len(peaks) < 2 or len(troughs) < 2:
            return []

        out = []
        for l1 in troughs:
            # choose last peak before l1
            p1_cands = [p for p in peaks if p < l1]
            if not p1_cands: 
                continue
            p1 = max(p1_cands)
            for l2 in troughs:
                if l2 <= l1:
                    continue
                p2_cands = [p for p in peaks if l1 < p < l2]
                if not p2_cands:
                    continue
                # pick the tallest peak between l1 and l2
                p2 = max(p2_cands, key=lambda idx: self.oc_peak[idx])

                # order & spacing on P1–P2
                if not (p1 < l1 < p2 < l2):
                    continue
                if not (min_sep <= (p2 - p1) <= max_sep):
                    continue

                # P1 >= P2 filter (tolerance)
                p1_eff = float(self.oc_peak[p1])
                p2_eff = float(self.oc_peak[p2])
                if p1_eff < p2_eff * (1.0 - p1_ge_tol):
                    continue

                # Neckline: use Highs at P1 and P2 for the line (your prior convention)
                p1_high = float(self.highs[p1])
                p2_high = float(self.highs[p2])
                if p2 == p1:
                    slope_per_bar = 0.0
                else:
                    slope_per_bar = (p2_high - p1_high) / (p2 - p1)

                # Breakout after L2
                breakout_idx = None
                for k in range(l2 + 1, n):
                    if p2 == p1:
                        neck_k = p2_high
                    else:
                        neck_k = p1_high + slope_per_bar * (k - p1)
                    px = float(self.highs[k]) if use_high_for_breakout else float(self.closes[k])
                    if px >= neck_k * (1.0 + neckline_buffer):
                        breakout_idx = k
                        break

                status = "COMPLETED" if breakout_idx is not None else "FORMING"
                if status == "FORMING" and not keep_forming:
                    # drop forming setups if not requested
                    continue

                out.append({
                    "p1_idx": p1, "l1_idx": l1, "p2_idx": p2, "l2_idx": l2,
                    "p1_date": self.dates[p1], "l1_date": self.dates[l1],
                    "p2_date": self.dates[p2], "l2_date": self.dates[l2],
                    "p1_price": p1_eff, "p2_price": p2_eff,
                    "l1_price": float(self.oc_trough[l1]), "l2_price": float(self.oc_trough[l2]),
                    "neckline_slope_per_bar": slope_per_bar,
                    "p3_idx": breakout_idx,
                    "p3_date": (self.dates[breakout_idx] if breakout_idx is not None else None),
                    "p3_price": (float(self.highs[breakout_idx]) if breakout_idx is not None else None),
                    "status": status
                })
        # Sort: completed first, then by l2 date
        out.sort(key=lambda d: (d["status"] != "COMPLETED", d["l2_idx"]))
        return out

    def print_w_summary(self, w_list):
        if not w_list:
            print("No W patterns found with current settings.")
            return
        print("\n" + "="*80)
        print(f"FOUND {len(w_list)} W PATTERN(S) [P1≥P2 filter enforced]")
        print("="*80)
        for i, w in enumerate(w_list[:10], 1):
            print(f"\nW {i}: {w['status']}")
            print(f"  P1: {w['p1_date'].strftime('%Y-%m-%d %H:%M')}  price={w['p1_price']:.2f}")
            print(f"  L1: {w['l1_date'].strftime('%Y-%m-%d %H:%M')}  price={w['l1_price']:.2f}")
            print(f"  P2: {w['p2_date'].strftime('%Y-%m-%d %H:%M')}  price={w['p2_price']:.2f}")
            print(f"  L2: {w['l2_date'].strftime('%Y-%m-%d %H:%M')}  price={w['l2_price']:.2f}")
            if w["p3_idx"] is not None:
                print(f"  P3(breakout): {w['p3_date'].strftime('%Y-%m-%d %H:%M')}  price={w['p3_price']:.2f}")
            print(f"  Neckline slope/bar: {w['neckline_slope_per_bar']:.4f}")

    def visualize_w(self, w_list, which=0):
        if not w_list:
            print("No W patterns to visualize.")
            return
        w = w_list[min(which, len(w_list)-1)]
        p1, l1, p2, l2 = w["p1_idx"], w["l1_idx"], w["p2_idx"], w["l2_idx"]
        p3 = w["p3_idx"]
        idx0 = max(0, p1 - 10)
        idx1 = min(len(self.closes)-1, (p3 if p3 is not None else l2) + 10)

        dates = self.dates[idx0:idx1+1]
        closes = self.closes[idx0:idx1+1]
        highs  = self.highs [idx0:idx1+1]

        # Neckline values across view window
        def neck_at(k):
            if p2 == p1: 
                return self.highs[p2]
            return self.highs[p1] + (self.highs[p2]-self.highs[p1]) * ((k - p1) / (p2 - p1))
        neck_vals = np.array([neck_at(k) for k in range(idx0, idx1+1)])

        plt.figure(figsize=(14,7))
        plt.plot(dates, closes, lw=1.2, label="Close")
        plt.plot(dates, highs, lw=0.8, alpha=0.4, label="High")
        plt.plot(dates, neck_vals, "--", lw=1.4, label="Neckline (P1→P2 Highs)")

        for idx, lbl in [(p1,"P1"), (l1,"L1"), (p2,"P2"), (l2,"L2")]:
            plt.scatter(self.dates[idx], self.closes[idx], s=60, marker="D", label=lbl)
        if p3 is not None:
            plt.scatter(self.dates[p3], self.closes[p3], s=80, marker="^", label="P3 (breakout)")

        plt.title(f"{self.symbol} — W pattern ({w['status']})")
        plt.legend(); plt.grid(alpha=0.3)
        plt.xticks(rotation=45); plt.tight_layout(); plt.show()


# Main execution
if __name__ == "__main__":
    print("AAPL Pattern Detection - Channels + W Filter")
    print("=" * 60)
    
    # Initialize detector
    detector = AAPLFlagDetector(period="10d", interval="30m")
    
    # Find all flag patterns (existing functionality)
    patterns = detector.find_flag_patterns()
    detector.print_pattern_summary(patterns)
    
    # NEW: Find W patterns with P1 >= P2 filter enforced
    print("\nScanning for W patterns (P1≥P2)...")
    w_list = detector.find_w_patterns(
        win=3,              # local extrema radius
        min_sep=3,          # min bars between P1 and P2
        max_sep=85,         # max bars between P1 and P2
        p1_ge_tol=0.0,      # strict: P1 must be >= P2 (set 0.02 to allow 2% grace)
        neckline_buffer=0.001,
        use_high_for_breakout=True,
        keep_forming=True
    )
    detector.print_w_summary(w_list)

    # Visualize first W (if found)
    if w_list:
        print("\nVisualizing first W pattern...")
        detector.visualize_w(w_list, which=0)
    else:
        print("\nNo W patterns found to visualize.")
