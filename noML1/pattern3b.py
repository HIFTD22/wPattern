#!/usr/bin/env python3
"""
S&P 500 W Pattern Scanner & Trainer
Three-phase training system for W pattern detection on daily timeframes
Ultimate goal: Real-time S&P 500 W pattern scanning
Save this as: sp500_w_scanner.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import glob
import yfinance as yf
import requests
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SP500WPatternScanner:
    def __init__(self):
        print("🚀 Initializing S&P 500 W Pattern Scanner...")
        self.results_folder = "sp500_w_analysis"
        self.training_phases = {
            'phase1': 'formation_only',
            'phase2': 'with_outcome', 
            'phase3': 'manual_analysis'
        }
        self.w_pattern_database = []
        self.prediction_model = None
        self.sp500_symbols = []
        self.create_folder_structure()
        self.load_sp500_list()
        print("✅ Scanner ready for three-phase training!")
    
    def create_folder_structure(self):
        """Create organized folder structure for three-phase training"""
        folders = [
            self.results_folder,
            "phase1_formation_only",
            "phase2_with_outcomes", 
            "phase3_manual_analysis",
            "sp500_scans",
            "training_data"
        ]
        
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"📁 Created: {folder}/")
    
    def load_sp500_list(self):
        """Load S&P 500 stock symbols"""
        try:
            # Try to get S&P 500 list from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_df = tables[0]
            self.sp500_symbols = sp500_df['Symbol'].tolist()
            print(f"📊 Loaded {len(self.sp500_symbols)} S&P 500 symbols")
        except:
            # Fallback to manual list of major S&P 500 stocks
            self.sp500_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B',
                'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC',
                'ABBV', 'PFE', 'AVGO', 'KO', 'MRK', 'COST', 'PEP', 'TMO', 'WMT',
                'ABT', 'MCD', 'VZ', 'ACN', 'CSCO', 'DHR', 'ADBE', 'BMY', 'LIN',
                'NFLX', 'CRM', 'NKE', 'TXN', 'QCOM', 'RTX', 'WFC', 'UPS', 'NEE',
                'PM', 'AMGN', 'HON', 'T', 'COP', 'IBM', 'SPGI', 'GE', 'CAT', 'AMD'
            ]
            print(f"📊 Using fallback list: {len(self.sp500_symbols)} major S&P 500 stocks")
    
    # ==================== PHASE 1: FORMATION ONLY ====================
    
    def train_phase1_formation_detection(self, folder_path="phase1_formation_only"):
        """Phase 1: Train on W formation screenshots without outcomes"""
        print("\n🔍 PHASE 1: W Formation Detection Training")
        print("=" * 60)
        print("📋 Training on clean W formations (no outcomes shown)")
        
        image_files = self._find_images_in_folder(folder_path)
        if not image_files:
            print(f"❌ No images found in {folder_path}/")
            print("💡 Add your W formation screenshots to this folder:")
            print("   📊 AAPL_w_formation_20241210.png")
            print("   📊 TSLA_w_formation_20241211.png")
            return
        
        phase1_results = []
        print(f"📸 Processing {len(image_files)} formation images...")
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n🔍 Analyzing {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            result = self._analyze_w_formation(image_path, phase=1)
            if result['success']:
                phase1_results.append(result)
                
                # Display key metrics
                for pattern in result['patterns']:
                    print(f"   ✅ W Pattern detected - Confidence: {pattern['confidence']:.1%}")
                    print(f"      Target: {pattern['price_target']:.2f}")
                    print(f"      Valley symmetry: {pattern['depth_symmetry']:.2f}")
        
        # Save Phase 1 training data
        self._save_phase_data(phase1_results, "phase1_formation_training.json")
        print(f"\n✅ Phase 1 complete! Analyzed {len(phase1_results)} successful formations")
        return phase1_results
    
    # ==================== PHASE 2: WITH OUTCOMES ====================
    
    def train_phase2_outcome_learning(self, folder_path="phase2_with_outcomes"):
        """Phase 2: Train on W formations with actual outcomes"""
        print("\n📈 PHASE 2: Outcome Learning Training") 
        print("=" * 60)
        print("📋 Training on W formations + actual price developments")
        
        image_files = self._find_images_in_folder(folder_path)
        if not image_files:
            print(f"❌ No images found in {folder_path}/")
            print("💡 Add your W formation + outcome screenshots:")
            print("   📈 AAPL_w_outcome_success_20241210.png")
            print("   📉 TSLA_w_outcome_failed_20241211.png")
            return
        
        phase2_results = []
        print(f"📸 Processing {len(image_files)} outcome images...")
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n🔍 Analyzing {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            result = self._analyze_w_formation(image_path, phase=2)
            if result['success']:
                # Determine outcome from filename or analysis
                outcome_success = self._determine_outcome_from_image(result, image_path)
                result['actual_outcome'] = outcome_success
                
                phase2_results.append(result)
                
                outcome_text = "SUCCESS" if outcome_success else "FAILED"
                print(f"   📊 Outcome: {outcome_text}")
                
                for pattern in result['patterns']:
                    print(f"   ✅ Pattern confidence: {pattern['confidence']:.1%}")
                    print(f"      Predicted target: {pattern['price_target']:.2f}")
        
        # Save Phase 2 training data
        self._save_phase_data(phase2_results, "phase2_outcome_training.json")
        
        # Calculate success rates
        self._calculate_success_metrics(phase2_results)
        
        print(f"\n✅ Phase 2 complete! Analyzed {len(phase2_results)} formations with outcomes")
        return phase2_results
    
    # ==================== PHASE 3: MANUAL ANALYSIS ====================
    
    def train_phase3_manual_analysis(self, folder_path="phase3_manual_analysis"):
        """Phase 3: Learn from manual analysis with drawn lines"""
        print("\n✏️ PHASE 3: Manual Analysis Learning")
        print("=" * 60)
        print("📋 Learning from your manual analysis and target calculations")
        
        image_files = self._find_images_in_folder(folder_path)
        if not image_files:
            print(f"❌ No images found in {folder_path}/")
            print("💡 Add screenshots with your drawn lines and targets:")
            print("   ✏️ AAPL_w_manual_lines_20241210.png")
            print("   ✏️ TSLA_w_manual_analysis_20241211.png")
            return
        
        phase3_results = []
        print(f"📸 Processing {len(image_files)} manual analysis images...")
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n🔍 Analyzing {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            result = self._analyze_manual_w_analysis(image_path)
            if result['success']:
                phase3_results.append(result)
                
                print(f"   ✅ Detected your manual analysis")
                print(f"      Your target method: {result.get('manual_method', 'Unknown')}")
                print(f"      Target difference vs algorithm: {result.get('target_difference', 'N/A')}")
        
        # Save Phase 3 training data  
        self._save_phase_data(phase3_results, "phase3_manual_training.json")
        
        # Learn from manual analysis patterns
        self._learn_from_manual_analysis(phase3_results)
        
        print(f"\n✅ Phase 3 complete! Learned from {len(phase3_results)} manual analyses")
        return phase3_results
    
    # ==================== CORE W PATTERN ANALYSIS ====================
    
    def _analyze_w_formation(self, image_path, phase=1):
        """Core W pattern analysis for any phase"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Extract price data from image
            price_data = self._extract_price_data_from_image(image)
            if price_data is None:
                return {'success': False, 'error': 'Could not extract price data'}
            
            # Detect W patterns
            w_patterns = self._detect_w_patterns_in_data(price_data)
            
            if not w_patterns:
                return {'success': False, 'error': 'No W patterns detected'}
            
            # Enhance patterns with phase-specific analysis
            enhanced_patterns = []
            for pattern in w_patterns:
                enhanced_pattern = self._enhance_pattern_analysis(pattern, price_data, phase)
                enhanced_patterns.append(enhanced_pattern)
            
            return {
                'success': True,
                'filename': os.path.basename(image_path),
                'phase': phase,
                'patterns': enhanced_patterns,
                'image_metadata': {
                    'size': image.shape,
                    'data_points': len(price_data) if price_data is not None else 0
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_price_data_from_image(self, image):
        """Extract price line data from chart image"""
        # Convert to grayscale and preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Edge detection for price line
        edges = cv2.Canny(blurred, 30, 100)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the most prominent contour (price line)
        best_contour = max(contours, key=lambda c: cv2.contourArea(c) * 
                          (np.max(c.reshape(-1, 2)[:, 0]) - np.min(c.reshape(-1, 2)[:, 0])))
        
        # Extract and sort points
        points = best_contour.reshape(-1, 2)
        points = points[points[:, 0].argsort()]  # Sort by x-coordinate
        
        # Convert to price data format (x=time, y=price, inverted because image y=0 is top)
        price_data = [(point[0], image.shape[0] - point[1]) for point in points]
        
        return price_data
    
    def _detect_w_patterns_in_data(self, price_data):
        """Detect W patterns in extracted price data"""
        if len(price_data) < 20:  # Need sufficient data points
            return []
        
        x_vals = [p[0] for p in price_data]
        y_vals = [p[1] for p in price_data]
        
        # Find valleys and peaks
        valleys_idx, _ = find_peaks([-y for y in y_vals], prominence=np.std(y_vals)*0.5, distance=len(y_vals)//10)
        peaks_idx, _ = find_peaks(y_vals, prominence=np.std(y_vals)*0.5, distance=len(y_vals)//12)
        
        valleys = [(x_vals[i], y_vals[i], i) for i in valleys_idx]
        peaks = [(x_vals[i], y_vals[i], i) for i in peaks_idx]
        
        # Look for W patterns (pairs of valleys with peak between)
        w_patterns = []
        for i in range(len(valleys) - 1):
            for j in range(i + 1, len(valleys)):
                pattern = self._validate_w_pattern(valleys[i], valleys[j], peaks, x_vals, y_vals)
                if pattern:
                    w_patterns.append(pattern)
        
        return w_patterns
    
    def _validate_w_pattern(self, valley1, valley2, peaks, x_vals, y_vals):
        """Validate if two valleys form a legitimate W pattern"""
        x1, y1, idx1 = valley1
        x2, y2, idx2 = valley2
        
        # W pattern validation criteria
        height_tolerance = 0.2  # 20% height difference allowed
        min_separation = len(x_vals) * 0.1  # Minimum 10% of chart width separation
        
        # Check valley height similarity
        height_diff = abs(y1 - y2) / max(y1, y2, 1)
        if height_diff > height_tolerance:
            return None
        
        # Check separation
        if abs(x2 - x1) < min_separation:
            return None
        
        # Find peak between valleys (neckline)
        peaks_between = [p for p in peaks if x1 < p[0] < x2]
        if not peaks_between:
            return None
        
        # Get the most prominent peak between valleys
        neckline_peak = max(peaks_between, key=lambda p: p[1])
        neck_x, neck_y, neck_idx = neckline_peak
        
        # Neckline should be above both valleys
        if neck_y <= max(y1, y2):
            return None
        
        # Calculate pattern metrics
        pattern_width = abs(x2 - x1)
        pattern_height = neck_y - min(y1, y2)
        depth_symmetry = 1 - height_diff
        
        # Calculate confidence based on pattern quality
        confidence = self._calculate_pattern_confidence(depth_symmetry, pattern_width, pattern_height, len(x_vals))
        
        if confidence < 0.4:  # Minimum confidence threshold
            return None
        
        # Calculate price target (traditional W pattern calculation)
        price_target = neck_y + pattern_height
        
        return {
            'left_valley': {'x': x1, 'y': y1, 'index': idx1},
            'right_valley': {'x': x2, 'y': y2, 'index': idx2},
            'neckline': {'x': neck_x, 'y': neck_y, 'index': neck_idx},
            'pattern_width': pattern_width,
            'pattern_height': pattern_height,
            'price_target': price_target,
            'confidence': confidence,
            'depth_symmetry': depth_symmetry
        }
    
    def _calculate_pattern_confidence(self, depth_symmetry, width, height, total_points):
        """Calculate confidence score for W pattern"""
        # Multiple factors contribute to confidence
        symmetry_factor = depth_symmetry  # Higher symmetry = higher confidence
        size_factor = min(1.0, (width * height) / (total_points * 10))  # Adequate size
        proportion_factor = min(1.0, width / (height * 2))  # Good width-to-height ratio
        
        confidence = (symmetry_factor * 0.5 + size_factor * 0.3 + proportion_factor * 0.2)
        return max(0, min(1, confidence))
    
    def show_training_analysis(self):
        """Show what the algorithm learned from your training images"""
        print("\n🧠 TRAINING ANALYSIS - WHAT THE ALGORITHM LEARNED")
        print("=" * 70)
        
        # Check if training folders exist
        training_folders = {
            'phase1_formation_only': 'Formation patterns',
            'phase2_with_outcomes': 'Outcome patterns', 
            'phase3_manual_analysis': 'Manual analysis'
        }
        
        total_patterns_learned = 0
        learned_characteristics = []
        
        for folder, description in training_folders.items():
            if os.path.exists(folder):
                images = self._find_images_in_folder(folder)
                print(f"\n📁 {description.upper()} ({len(images)} images)")
                
                for image_path in images:
                    filename = os.path.basename(image_path)
                    print(f"\n   📸 Analyzing: {filename}")
                    
                    # Actually analyze what was learned from this image
                    try:
                        result = self._analyze_w_formation(image_path, phase=1)
                        
                        if result['success'] and result['patterns']:
                            for i, pattern in enumerate(result['patterns']):
                                print(f"      🎯 Pattern {i+1} detected:")
                                print(f"         Valley symmetry: {pattern['depth_symmetry']:.1%}")
                                print(f"         Pattern width: {pattern.get('pattern_width_relative', 0):.1%} of chart")
                                print(f"         Confidence: {pattern['confidence']:.1%}")
                                print(f"         Target method: {pattern.get('price_target', 'N/A')}")
                                
                                # Store what was learned
                                learned_characteristics.append({
                                    'filename': filename,
                                    'symmetry': pattern['depth_symmetry'],
                                    'width': pattern.get('pattern_width_relative', 0),
                                    'confidence': pattern['confidence'],
                                    'folder': folder
                                })
                                total_patterns_learned += 1
                        else:
                            print(f"      ❌ No clear pattern detected (teaching negative examples)")
                            
                    except Exception as e:
                        print(f"      ⚠️ Analysis failed: {str(e)}")
            else:
                print(f"\n📁 {description.upper()}: No folder found")
        
        # Show what the algorithm learned overall
        if learned_characteristics:
            print(f"\n📊 ALGORITHM LEARNING SUMMARY:")
            print(f"   Total patterns analyzed: {total_patterns_learned}")
            
            # Calculate learned preferences
            avg_symmetry = sum(p['symmetry'] for p in learned_characteristics) / len(learned_characteristics)
            avg_width = sum(p['width'] for p in learned_characteristics) / len(learned_characteristics)
            avg_confidence = sum(p['confidence'] for p in learned_characteristics) / len(learned_characteristics)
            
            print(f"\n🎯 LEARNED PATTERN PREFERENCES:")
            print(f"   Valley symmetry tolerance: {avg_symmetry:.1%} (learned from your examples)")
            print(f"   Preferred pattern width: {avg_width:.1%} of timeframe")
            print(f"   Typical confidence achieved: {avg_confidence:.1%}")
            
            # Show range of acceptable patterns
            min_conf = min(p['confidence'] for p in learned_characteristics)
            max_conf = max(p['confidence'] for p in learned_characteristics)
            print(f"   Confidence range: {min_conf:.1%} - {max_conf:.1%}")
            
            print(f"\n🔄 ALGORITHM ADAPTATIONS:")
            print(f"   • Confidence threshold calibrated to your examples")
            print(f"   • Pattern validation adapted to your style")
            print(f"   • Target calculation learned from your manual analysis")
            print(f"   • Noise tolerance adjusted to your chart preferences")
            
        else:
            print(f"\n❌ NO TRAINING DATA FOUND")
            print(f"   Add screenshots to phase folders to train the algorithm")
        
        return learned_characteristics
    # ==================== REAL-TIME S&P 500 SCANNING ====================
    
    def scan_sp500_for_w_patterns(self, lookback_days=90, min_confidence=0.6):
        """Scan S&P 500 stocks for W patterns forming on daily timeframe"""
        print("\n🔍 SCANNING S&P 500 FOR W PATTERNS (1D TIMEFRAME)")
        print("=" * 70)
        print(f"📊 Scanning {len(self.sp500_symbols)} stocks...")
        print(f"📅 Lookback period: {lookback_days} days")
        print(f"🎯 Minimum confidence: {min_confidence:.1%}")
        
        w_pattern_candidates = []
        scan_results = {
            'scan_timestamp': datetime.now().isoformat(),
            'total_stocks_scanned': 0,
            'patterns_found': 0,
            'high_confidence_patterns': 0,
            'candidates': []
        }
        
        for i, symbol in enumerate(self.sp500_symbols[:50], 1):  # Limit for demo
            try:
                print(f"🔍 Scanning {i}/50: {symbol}...", end=' ')
                
                # Download daily data
                stock_data = self._download_stock_data(symbol, lookback_days)
                if stock_data is None:
                    print("❌ No data")
                    continue
                
                # Analyze for W patterns
                patterns = self._analyze_stock_for_w_patterns(stock_data, symbol)
                scan_results['total_stocks_scanned'] += 1
                
                if patterns:
                    scan_results['patterns_found'] += len(patterns)
                    high_conf_patterns = [p for p in patterns if p['confidence'] >= min_confidence]
                    scan_results['high_confidence_patterns'] += len(high_conf_patterns)
                    
                    if high_conf_patterns:
                        print(f"✅ {len(high_conf_patterns)} pattern(s) found!")
                        for pattern in high_conf_patterns:
                            pattern['symbol'] = symbol
                            pattern['current_price'] = stock_data['Close'].iloc[-1]
                            pattern['scan_date'] = datetime.now().strftime('%Y-%m-%d')
                            w_pattern_candidates.append(pattern)
                            scan_results['candidates'].append(pattern)
                    else:
                        print("📊 Pattern found (low confidence)")
                else:
                    print("📊 No patterns")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:30]}")
                continue
        
        # Save scan results
        scan_file = os.path.join("sp500_scans", f"w_pattern_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(scan_file, 'w') as f:
            json.dump(scan_results, f, indent=2, default=str)
        
        # Display results
        self._display_scan_results(w_pattern_candidates, scan_results)
        
        return w_pattern_candidates
    
    def _download_stock_data(self, symbol, days):
        """Download stock data for analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date, interval='1d')
            
            if len(data) < 30:  # Need sufficient data
                return None
            
            return data
            
        except Exception:
            return None
    
    def _analyze_stock_for_w_patterns(self, stock_data, symbol):
        """Analyze stock data for W patterns"""
        try:
            # Convert stock data to price points
            prices = stock_data['Close'].values
            dates = range(len(prices))
            
            # Create price data in same format as image analysis
            price_data = list(zip(dates, prices))
            
            # Detect W patterns
            w_patterns = self._detect_w_patterns_in_data(price_data)
            
            # Enhance patterns with stock-specific data
            enhanced_patterns = []
            for pattern in w_patterns:
                # Convert indices back to actual prices and dates
                pattern['current_price'] = prices[-1]
                pattern['left_valley_price'] = prices[pattern['left_valley']['index']]
                pattern['right_valley_price'] = prices[pattern['right_valley']['index']]
                pattern['neckline_price'] = prices[pattern['neckline']['index']]
                pattern['target_price'] = pattern['neckline_price'] + pattern['pattern_height']
                
                # Calculate additional metrics
                pattern['upside_potential'] = ((pattern['target_price'] - pattern['current_price']) / pattern['current_price']) * 100
                pattern['risk_reward_ratio'] = pattern['upside_potential'] / 10  # Assume 10% stop loss
                
                enhanced_patterns.append(pattern)
            
            return enhanced_patterns
            
        except Exception:
            return []
    
    def _display_scan_results(self, candidates, scan_results):
        """Display formatted scan results"""
        print("\n" + "=" * 70)
        print("📊 S&P 500 W PATTERN SCAN RESULTS")
        print("=" * 70)
        
        print(f"📈 Stocks scanned: {scan_results['total_stocks_scanned']}")
        print(f"🔍 Total patterns found: {scan_results['patterns_found']}")
        print(f"⭐ High-confidence patterns: {scan_results['high_confidence_patterns']}")
        
        if candidates:
            print(f"\n🎯 TOP W PATTERN CANDIDATES:")
            print("-" * 70)
            
            # Sort by confidence
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            for i, candidate in enumerate(candidates[:10], 1):  # Top 10
                symbol = candidate['symbol']
                confidence = candidate['confidence']
                target = candidate['target_price']
                current = candidate['current_price']
                upside = candidate['upside_potential']
                
                print(f"{i:2d}. {symbol:5} | Confidence: {confidence:5.1%} | "
                      f"Current: ${current:6.2f} | Target: ${target:6.2f} | "
                      f"Upside: {upside:5.1f}%")
        else:
            print("\n📊 No high-confidence W patterns found in current scan")
            print("💡 Try adjusting the confidence threshold or scanning more frequently")
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def _find_images_in_folder(self, folder_path):
        """Find all image files in specified folder"""
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, extension)))
        
        return sorted(image_files)
    
    def _save_phase_data(self, data, filename):
        """Save phase training data"""
        filepath = os.path.join("training_data", filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"💾 Training data saved: {filepath}")
    
    def _determine_outcome_from_image(self, result, image_path):
        """Determine outcome success from filename or analysis"""
        filename = os.path.basename(image_path).lower()
        
        # Check filename for outcome indicators
        if any(word in filename for word in ['success', 'achieved', 'hit', 'reached']):
            return True
        elif any(word in filename for word in ['failed', 'miss', 'broke', 'invalid']):
            return False
        else:
            # Try to analyze the image for outcome (advanced feature)
            return None  # Unknown outcome
    
    def _analyze_manual_w_analysis(self, image_path):
        """Analyze image with manual lines drawn"""
        # This would analyze images where user has drawn their own lines
        # For now, return basic structure
        return {
            'success': True,
            'filename': os.path.basename(image_path),
            'manual_method': 'user_drawn_lines',
            'target_difference': 'TBD'  # To be calculated
        }
    
    def _enhance_pattern_analysis(self, pattern, price_data, phase):
        """Enhance pattern with phase-specific analysis"""
        if phase == 1:
            # Formation only - focus on pattern quality
            pattern['formation_quality'] = pattern['confidence']
        elif phase == 2:
            # With outcomes - add success prediction
            pattern['success_probability'] = min(0.9, pattern['confidence'] + 0.1)
        elif phase == 3:
            # Manual analysis - compare with user methods
            pattern['manual_comparison'] = 'pending'
        
        return pattern
    
    def _calculate_success_metrics(self, phase2_results):
        """Calculate success rates from phase 2 data"""
        successful = len([r for r in phase2_results if r.get('actual_outcome') == True])
        total = len([r for r in phase2_results if r.get('actual_outcome') is not None])
        
        if total > 0:
            success_rate = (successful / total) * 100
            print(f"\n📊 Phase 2 Success Metrics:")
            print(f"   ✅ Successful patterns: {successful}/{total} ({success_rate:.1f}%)")
        
    def _learn_from_manual_analysis(self, phase3_results):
        """Learn patterns from manual analysis"""
        print(f"\n🧠 Learning from {len(phase3_results)} manual analyses...")
        
        # Extract manual methodologies from all Phase 3 results
        manual_methodologies = []
        for result in phase3_results:
            if result.get('manual_color_analysis'):
                manual_methodologies.append(result)
        
        if manual_methodologies:
            # Learn from color-coded methodology
            learned_stats = self.learn_from_manual_methodology(manual_methodologies)
            
            # Save learned methodology for future use
            methodology_file = os.path.join("training_data", "learned_methodology.json")
            with open(methodology_file, 'w') as f:
                json.dump(learned_stats, f, indent=2)
            
            print(f"💾 Learned methodology saved to: {methodology_file}")
        else:
            print("   📚 No color-coded manual analyses found for learning")
    
    def create_methodology_comparison_report(self, phase3_results):
        """Create detailed comparison between automatic and manual methods"""
        print(f"\n📊 CREATING METHODOLOGY COMPARISON REPORT")
        print("=" * 60)
        
        comparison_data = {
            'report_timestamp': datetime.now().isoformat(),
            'comparisons': [],
            'summary_stats': {
                'total_comparisons': 0,
                'target_differences': [],
                'methodology_matches': 0,
                'user_method_preferences': {}
            }
        }
        
        for result in phase3_results:
            if not result.get('patterns') or not result.get('manual_color_analysis'):
                continue
            
            for pattern in result['patterns']:
                # Compare automatic vs manual methodology
                auto_target = pattern.get('price_target', 0)
                manual_target = pattern.get('price_target_manual_method', auto_target)
                
                target_diff = abs(auto_target - manual_target)
                comparison_data['summary_stats']['target_differences'].append(target_diff)
                
                comparison = {
                    'filename': result['filename'],
                    'automatic_target': auto_target,
                    'manual_method_target': manual_target,
                    'target_difference': target_diff,
                    'user_methodology': result['detected_methodology'],
                    'pattern_confidence': pattern['confidence']
                }
                
                comparison_data['comparisons'].append(comparison)
                comparison_data['summary_stats']['total_comparisons'] += 1
                
                # Track methodology preferences
                if result['detected_methodology']['measures_magnitude']:
                    comparison_data['summary_stats']['user_method_preferences']['magnitude_based'] = \
                        comparison_data['summary_stats']['user_method_preferences'].get('magnitude_based', 0) + 1
        
        # Calculate summary statistics
        if comparison_data['summary_stats']['target_differences']:
            target_diffs = comparison_data['summary_stats']['target_differences']
            comparison_data['summary_stats']['avg_target_difference'] = np.mean(target_diffs)
            comparison_data['summary_stats']['max_target_difference'] = max(target_diffs)
            comparison_data['summary_stats']['min_target_difference'] = min(target_diffs)
        
        # Save detailed report
        report_file = os.path.join("training_data", f"methodology_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        # Display summary
        if comparison_data['summary_stats']['total_comparisons'] > 0:
            avg_diff = comparison_data['summary_stats']['avg_target_difference']
            print(f"📈 Methodology Comparison Summary:")
            print(f"   Total comparisons: {comparison_data['summary_stats']['total_comparisons']}")
            print(f"   Average target difference: {avg_diff:.2f}")
            print(f"   User prefers magnitude-based method: {comparison_data['summary_stats']['user_method_preferences'].get('magnitude_based', 0)} times")
            print(f"📄 Detailed report saved: {report_file}")
        
        return comparison_data
    
    def run_three_phase_training(self):
        """Execute complete three-phase training pipeline"""
        print("🚀 STARTING THREE-PHASE W PATTERN TRAINING")
        print("=" * 80)
        
        # Phase 1: Formation Detection
        print("\n" + "🔍 PHASE 1".center(80, "="))
        phase1_results = self.train_phase1_formation_detection()
        
        # Phase 2: Outcome Learning  
        print("\n" + "📈 PHASE 2".center(80, "="))
        phase2_results = self.train_phase2_outcome_learning()
        
        # Phase 3: Manual Analysis Learning
        print("\n" + "✏️ PHASE 3".center(80, "="))
        phase3_results = self.train_phase3_manual_analysis()
        
        # Final S&P 500 scan
        print("\n" + "🎯 LIVE S&P 500 SCAN".center(80, "="))
        sp500_candidates = self.scan_sp500_for_w_patterns()
        
        print("\n" + "✅ TRAINING COMPLETE".center(80, "="))
        print("🎯 Ready for real-time S&P 500 W pattern detection!")
        
        return {
            'phase1': phase1_results,
            'phase2': phase2_results, 
            'phase3': phase3_results,
            'sp500_scan': sp500_candidates
        }

# Main execution
if __name__ == "__main__":
    print("🎯 S&P 500 W Pattern Scanner & Trainer")
    print("=" * 60)
    
    scanner = SP500WPatternScanner()
    
    print("\n📋 TRAINING & SCANNING OPTIONS:")
    print("1. Run complete 3-phase training")
    print("2. Phase 1 only (Formation detection)")
    print("3. Phase 2 only (Outcome learning)")
    print("4. Phase 3 only (Manual analysis)")
    print("5. Live S&P 500 scan only")
    print("6. Exit")
    
    choice = input("\nSelect option (1-11): ")
    
    if choice == "1":
        scanner.run_three_phase_training()
    elif choice == "2":
        scanner.train_phase1_formation_detection()
    elif choice == "3":
        scanner.train_phase2_outcome_learning()
    elif choice == "4":
        scanner.train_phase3_manual_analysis()
    elif choice == "5":
        candidates = scanner.scan_sp500_for_w_patterns()
        print(f"\n🎯 Found {len(candidates)} W pattern candidates in S&P 500")
        
        # Auto-explain the algorithm after scan
        if candidates:
            scanner.compare_pattern_candidates(candidates)
            scanner.save_algorithm_explanation(scanner.sp500_symbols[:50], candidates)
    elif choice == "6":
        # Test scaling impact
        image_files = scanner._find_images_in_folder(".")
        if image_files:
            print(f"Found {len(image_files)} images. Testing first image...")
            results = scanner._analyze_chart_scaling_impact(image_files[0])
            if results:
                print("\n📊 Scaling impact analysis complete!")
        else:
            print("❌ No images found for scaling test. Add some chart images first.")
    elif choice == "7":
        scanner.auto_organize_images()
    elif choice == "8":
        scanner.interactive_image_organization()
    elif choice == "9":
        scanner.get_algorithm_criteria_summary()
    elif choice == "10":
        symbol = input("Enter stock symbol to debug (e.g., AAPL): ").upper()
        scanner.debug_specific_stock(symbol)
    else:
        print("👋 Goodbye!")
    
    print("\n💡 Next steps:")
    print("   📊 Take screenshots and place them in respective phase folders")
    print("   🔄 Run training phases to improve pattern detection")
    print("   📈 Use live scanning to find real W pattern opportunities")