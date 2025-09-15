import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class StockPatternTester:
    def __init__(self, symbol, period="60d", interval="5m"):
        """
        Initialize with stock symbol, time period, and interval
        For 5-minute data, period is limited to 60 days max by yfinance
        """
        self.symbol = symbol
        self.interval = interval
        
        # Download data with specified interval
        self.data = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        
        if self.data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Clean the data by removing any rows with NaN values
        self.data = self.data.dropna()
        
        if self.data.empty:
            raise ValueError(f"No valid data after cleaning for symbol {symbol}")
        
        self.prices = self.data['Close'].values
        self.dates = self.data.index
        
        # Ensure prices is a 1-D array (sometimes yfinance returns multi-dimensional)
        if self.prices.ndim > 1:
            self.prices = self.prices.flatten()
        
        print(f"Loaded {len(self.prices)} data points for {symbol}")
        print(f"Price array shape: {self.prices.shape}")
        
    def find_peaks_and_troughs(self, prominence=0.01, distance=12):
        """
        Find peaks and troughs in the stock price data
        prominence: minimum prominence for a peak/trough (as fraction of price range)
        distance: minimum number of data points between peaks (12 = ~1 hour for 5min data)
        """
        if len(self.prices) < 10:
            return np.array([]), np.array([])
            
        price_range = np.max(self.prices) - np.min(self.prices)
        if price_range == 0:
            return np.array([]), np.array([])
            
        min_prominence = prominence * price_range
        
        # For 5-minute data, we need distance parameter to avoid too many nearby peaks
        # Find peaks (local maxima)
        peaks, _ = find_peaks(self.prices, prominence=min_prominence, distance=distance)
        
        # Find troughs (local minima) by finding peaks in inverted data
        troughs, _ = find_peaks(-self.prices, prominence=min_prominence, distance=distance)
        
        return peaks, troughs
    
    def test_up_down_up_pattern(self, peaks, troughs, tolerance=0.015):
        """
        Test the up-down-up pattern: after going up-down-up, 
        does it go back down to form a line with previous troughs?
        """
        successful_patterns = 0
        total_patterns = 0
        pattern_details = []
        
        # Need at least 2 troughs to form a trendline
        if len(troughs) < 2:
            return 0, 0, []
        
        # Look for up-down-up patterns
        all_points = sorted([(idx, 'trough') for idx in troughs] + [(idx, 'peak') for idx in peaks])
        
        for i in range(len(all_points) - 3):
            # Check if we have trough-peak-trough-peak pattern (up-down-up)
            if (all_points[i][1] == 'trough' and 
                all_points[i+1][1] == 'peak' and 
                all_points[i+2][1] == 'trough' and 
                all_points[i+3][1] == 'peak'):
                
                trough1_idx = all_points[i][0]
                peak1_idx = all_points[i+1][0]
                trough2_idx = all_points[i+2][0]
                peak2_idx = all_points[i+3][0]
                
                # Calculate trendline from the two troughs
                x1, y1 = trough1_idx, self.prices[trough1_idx]
                x2, y2 = trough2_idx, self.prices[trough2_idx]
                
                # Find the next significant low after peak2
                next_low_idx = None
                min_price = float('inf')
                
                # Look ahead for the next 24 data points (~2 hours for 5min data)
                search_end = min(peak2_idx + 24, len(self.prices))
                for j in range(peak2_idx + 1, search_end):
                    if self.prices[j] < min_price:
                        min_price = self.prices[j]
                        next_low_idx = j
                
                if next_low_idx is not None:
                    total_patterns += 1
                    
                    # Calculate expected price based on trendline
                    if x2 != x1:  # Avoid division by zero
                        slope = (y2 - y1) / (x2 - x1)
                        expected_price = y1 + slope * (next_low_idx - x1)
                        
                        # Check if actual price is close to trendline (within tolerance)
                        actual_price = self.prices[next_low_idx]
                        if expected_price > 0:  # Avoid division by zero
                            price_diff = abs(actual_price - expected_price) / expected_price
                            
                            if price_diff <= tolerance:
                                successful_patterns += 1
                            
                            pattern_details.append({
                                'pattern_start': self.dates[trough1_idx],
                                'trough1_price': y1,
                                'peak1_price': self.prices[peak1_idx],
                                'trough2_price': y2,
                                'peak2_price': self.prices[peak2_idx],
                                'expected_price': expected_price,
                                'actual_price': actual_price,
                                'price_diff_pct': price_diff * 100,
                                'success': price_diff <= tolerance
                            })
        
        return successful_patterns, total_patterns, pattern_details
    
    def test_down_up_down_pattern(self, peaks, troughs, tolerance=0.015):
        """
        Test the down-up-down pattern: after going down-up-down,
        does it go back up to form a line with previous peaks?
        """
        successful_patterns = 0
        total_patterns = 0
        pattern_details = []
        
        if len(peaks) < 2:
            return 0, 0, []
        
        all_points = sorted([(idx, 'trough') for idx in troughs] + [(idx, 'peak') for idx in peaks])
        
        for i in range(len(all_points) - 3):
            # Check if we have peak-trough-peak-trough pattern (down-up-down)
            if (all_points[i][1] == 'peak' and 
                all_points[i+1][1] == 'trough' and 
                all_points[i+2][1] == 'peak' and 
                all_points[i+3][1] == 'trough'):
                
                peak1_idx = all_points[i][0]
                trough1_idx = all_points[i+1][0]
                peak2_idx = all_points[i+2][0]
                trough2_idx = all_points[i+3][0]
                
                # Calculate trendline from the two peaks
                x1, y1 = peak1_idx, self.prices[peak1_idx]
                x2, y2 = peak2_idx, self.prices[peak2_idx]
                
                # Find the next significant high after trough2
                next_high_idx = None
                max_price = 0
                
                # Look ahead for the next 24 data points (~2 hours for 5min data)
                search_end = min(trough2_idx + 24, len(self.prices))
                for j in range(trough2_idx + 1, search_end):
                    if self.prices[j] > max_price:
                        max_price = self.prices[j]
                        next_high_idx = j
                
                if next_high_idx is not None:
                    total_patterns += 1
                    
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                        expected_price = y1 + slope * (next_high_idx - x1)
                        
                        actual_price = self.prices[next_high_idx]
                        if expected_price > 0:
                            price_diff = abs(actual_price - expected_price) / expected_price
                            
                            if price_diff <= tolerance:
                                successful_patterns += 1
                            
                            pattern_details.append({
                                'pattern_start': self.dates[peak1_idx],
                                'peak1_price': y1,
                                'trough1_price': self.prices[trough1_idx],
                                'peak2_price': y2,
                                'trough2_price': self.prices[trough2_idx],
                                'expected_price': expected_price,
                                'actual_price': actual_price,
                                'price_diff_pct': price_diff * 100,
                                'success': price_diff <= tolerance
                            })
        
        return successful_patterns, total_patterns, pattern_details
    
    def run_full_test(self, prominence=0.01, distance=12, tolerance=0.015):
        """
        Run the complete test for both patterns
        Adjusted defaults for 5-minute data:
        - prominence: 1% (lower than daily due to smaller price movements)
        - distance: 12 periods (~1 hour minimum between peaks)
        - tolerance: 1.5% (tighter tolerance for shorter timeframes)
        """
        peaks, troughs = self.find_peaks_and_troughs(prominence, distance)
        
        print(f"Found {len(peaks)} peaks and {len(troughs)} troughs")
        
        # Test both patterns
        up_success, up_total, up_details = self.test_up_down_up_pattern(peaks, troughs, tolerance)
        down_success, down_total, down_details = self.test_down_up_down_pattern(peaks, troughs, tolerance)
        
        return {
            'symbol': self.symbol,
            'interval': self.interval,
            'total_periods': len(self.prices),
            'peaks_found': len(peaks),
            'troughs_found': len(troughs),
            'up_down_up': {
                'successful': up_success,
                'total': up_total,
                'success_rate': up_success / up_total if up_total > 0 else 0,
                'details': up_details
            },
            'down_up_down': {
                'successful': down_success,
                'total': down_total,
                'success_rate': down_success / down_total if down_total > 0 else 0,
                'details': down_details
            }
        }
    
    def visualize_patterns(self, prominence=0.01, distance=12):
        """
        Create a visualization showing the patterns
        """
        peaks, troughs = self.find_peaks_and_troughs(prominence, distance)
        
        plt.figure(figsize=(20, 10))  # Larger figure for 5-minute data
        plt.plot(self.dates, self.prices, 'b-', alpha=0.7, linewidth=0.8, label='Stock Price')
        plt.scatter(self.dates[peaks], self.prices[peaks], color='red', s=30, label='Peaks', zorder=5)
        plt.scatter(self.dates[troughs], self.prices[troughs], color='green', s=30, label='Troughs', zorder=5)
        
        plt.title(f'{self.symbol} - 5-Minute Peaks and Troughs Analysis')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Example usage and testing
def test_multiple_stocks(symbols, period="60d", interval="5m", prominence=0.01, distance=12, tolerance=0.015):
    """
    Test the theory on multiple stocks using 5-minute data
    Note: yfinance limits 5-minute data to 60 days maximum
    """
    results = []
    
    for symbol in symbols:
        print(f"\nTesting {symbol} with 5-minute data...")
        try:
            tester = StockPatternTester(symbol, period, interval)
            result = tester.run_full_test(prominence, distance, tolerance)
            results.append(result)
            
            print(f"{symbol} Results:")
            print(f"  Data points: {result['total_periods']} (5-minute intervals)")
            print(f"  Peaks found: {result['peaks_found']}, Troughs found: {result['troughs_found']}")
            print(f"  Up-Down-Up patterns: {result['up_down_up']['successful']}/{result['up_down_up']['total']} "
                  f"({result['up_down_up']['success_rate']:.1%})")
            print(f"  Down-Up-Down patterns: {result['down_up_down']['successful']}/{result['down_up_down']['total']} "
                  f"({result['down_up_down']['success_rate']:.1%})")
            
        except Exception as e:
            print(f"Error testing {symbol}: {e}")
    
    return results

# Additional function for comparing different timeframes
def compare_timeframes(symbol, daily_period="1y", intraday_period="60d"):
    """
    Compare the same pattern analysis between daily and 5-minute data
    """
    print(f"\nComparing timeframes for {symbol}")
    print("=" * 50)
    
    # Daily analysis
    try:
        print("Daily Analysis:")
        daily_tester = StockPatternTester(symbol, daily_period, "1d")
        daily_results = daily_tester.run_full_test(prominence=0.02, distance=1, tolerance=0.02)
        
        print(f"  Data points: {daily_results['total_periods']} days")
        print(f"  Up-Down-Up: {daily_results['up_down_up']['successful']}/{daily_results['up_down_up']['total']} "
              f"({daily_results['up_down_up']['success_rate']:.1%})")
        print(f"  Down-Up-Down: {daily_results['down_up_down']['successful']}/{daily_results['down_up_down']['total']} "
              f"({daily_results['down_up_down']['success_rate']:.1%})")
        
    except Exception as e:
        print(f"  Daily analysis failed: {e}")
        daily_results = None
    
    # 5-minute analysis
    try:
        print("\n5-Minute Analysis:")
        intraday_tester = StockPatternTester(symbol, intraday_period, "5m")
        intraday_results = intraday_tester.run_full_test()
        
        print(f"  Data points: {intraday_results['total_periods']} (5-minute intervals)")
        print(f"  Up-Down-Up: {intraday_results['up_down_up']['successful']}/{intraday_results['up_down_up']['total']} "
              f"({intraday_results['up_down_up']['success_rate']:.1%})")
        print(f"  Down-Up-Down: {intraday_results['down_up_down']['successful']}/{intraday_results['down_up_down']['total']} "
              f"({intraday_results['down_up_down']['success_rate']:.1%})")
        
        return daily_results, intraday_results
        
    except Exception as e:
        print(f"  5-minute analysis failed: {e}")
        return daily_results, None

# Run the test
if __name__ == "__main__":
    # Test on popular stocks with 5-minute data
    test_symbols = ['AAPL', 'MSFT']
    
    print("Testing Stock Pattern Theory on 5-Minute Data")
    print("=" * 60)
    print("Theory: After up-down-up, price returns to trough trendline")
    print("Theory: After down-up-down, price returns to peak trendline")
    print("Timeframe: 5-minute candlesticks")
    print("Period: 60 days (maximum for 5-minute data)")
    print("Tolerance: ±1.5% from expected price")
    print("Minimum distance between peaks: ~1 hour")
    
    # Test with 5-minute data (more sensitive parameters to find patterns)
    results = test_multiple_stocks(test_symbols, prominence=0.005, distance=6)
    
    # Calculate overall statistics
    total_up_patterns = sum(r['up_down_up']['total'] for r in results)
    successful_up_patterns = sum(r['up_down_up']['successful'] for r in results)
    
    total_down_patterns = sum(r['down_up_down']['total'] for r in results)
    successful_down_patterns = sum(r['down_up_down']['successful'] for r in results)
    
    print("\n" + "="*60)
    print("OVERALL RESULTS (5-MINUTE DATA):")
    if total_up_patterns > 0:
        print(f"Up-Down-Up patterns: {successful_up_patterns}/{total_up_patterns} "
              f"({successful_up_patterns/total_up_patterns:.1%} success rate)")
    else:
        print("No up-down-up patterns found")
        
    if total_down_patterns > 0:
        print(f"Down-Up-Down patterns: {successful_down_patterns}/{total_down_patterns} "
              f"({successful_down_patterns/total_down_patterns:.1%} success rate)")
    else:
        print("No down-up-down patterns found")
    
    print("\n" + "="*60)
    print("TIMEFRAME COMPARISON FOR AAPL:")
    daily_results, intraday_results = compare_timeframes('AAPL')
    
    print("\nKey Insights:")
    print("- 5-minute data provides more pattern occurrences")
    print("- Intraday patterns may be less influenced by news/sentiment")
    print("- Technical analysis might be more pure at shorter timeframes")
    
    # Example of detailed analysis for one stock
    print("\nDetailed 5-minute analysis for AAPL:")
    try:
        aapl_tester = StockPatternTester('AAPL', period="60d", interval="5m")
        aapl_tester.visualize_patterns()
    except Exception as e:
        print(f"Visualization failed: {e}")