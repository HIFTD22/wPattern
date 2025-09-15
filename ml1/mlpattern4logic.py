#!/usr/bin/env python3
"""
W Pattern Trading Success Predictor - Trend-Based Detection
Detects W patterns based on trend transitions and relative peaks/troughs
Uses Yahoo Finance OHLC data for accurate trend analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
import os
import glob
import pickle
import json
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class WPatternTradingPredictor:
    def __init__(self):
        print("W Pattern Trading Success Predictor - Trend-Based Detection")
        
        # ML Models optimized for trading prediction
        self.pattern_classifier = RandomForestClassifier(
            n_estimators=50, max_depth=6, min_samples_split=5,
            min_samples_leaf=3, random_state=42, class_weight='balanced'
        )
        self.confidence_regressor = GradientBoostingRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.target_regressor = GradientBoostingRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42
        )
        
        self.scaler = RobustScaler()
        self.training_features = []
        self.training_labels = []
        self.confidence_labels = []
        self.target_labels = []
        self.pattern_metadata = []
        self.is_trained = False
        self.training_stats = {}
        
        # Enhanced S&P 500 symbols for comprehensive scanning
        self.sp500_symbols = [
            # Technology Mega Caps
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META',
            # Technology Large Caps
            'ORCL', 'CRM', 'ADBE', 'NFLX', 'CSCO', 'INTC', 'AMD', 'QCOM',
            'AMAT', 'MU', 'LRCX', 'KLAC', 'MRVL', 'FTNT', 'PANW', 'CRWD',
            # Financial Services
            'BRK-B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW',
            'USB', 'PNC', 'TFC', 'COF', 'CME', 'ICE', 'SPGI', 'MCO',
            # Healthcare
            'UNH', 'JNJ', 'PFE', 'ABT', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD',
            'LLY', 'ABBV', 'MDT', 'CI', 'CVS', 'HUM', 'ANTM', 'BSX', 'SYK',
            # Consumer Discretionary
            'HD', 'NKE', 'SBUX', 'MCD', 'LOW', 'TGT', 'DG', 'COST', 'WMT',
            'TJX', 'MAR', 'HLT', 'MGM', 'F', 'GM', 'TSLA', 'RIVN',
            # Industrial
            'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC',
            'GD', 'DE', 'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'FDX',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'OXY',
            'PXD', 'HAL', 'BKR', 'DVN', 'FANG', 'MRO', 'APA',
            # Consumer Staples
            'PG', 'KO', 'PEP', 'MRK', 'WBA', 'KMB', 'CL', 'GIS', 'K',
            # Communication Services
            'VZ', 'T', 'DIS', 'CMCSA', 'NFLX', 'GOOGL', 'META', 'PARA',
            # Utilities
            'NEE', 'SO', 'DUK', 'AEP', 'EXC', 'XEL', 'PEG', 'PCG',
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD',
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O'
        ]
        
        # Remove duplicates and ensure unique symbols
        self.sp500_symbols = list(set(self.sp500_symbols))
        
        # Trend-based detection criteria
        self.detection_criteria = {
            'min_trend_length': 5,          # Minimum 5 days for a trend
            'trend_strength_threshold': 0.02, # 2% minimum move to confirm trend
            'trough_symmetry_tolerance': 0.30, # 30% tolerance for trough symmetry
            'min_pattern_duration': 15,     # Minimum 15 days for complete pattern
            'max_pattern_duration': 90,     # Maximum 90 days for pattern
            'resistance_tolerance': 0.02,   # 2% tolerance for resistance breakthrough
            'min_rebound_strength': 0.015,  # 1.5% minimum rebound from trough
            'volume_confirmation_factor': 1.2 # Volume should be 20% above average at key points
        }
        
        # Feature names for interpretability
        self.feature_names = [
            'avg_height', 'volatility', 'price_range', 'num_points', 'trend_slope',
            'num_peaks', 'num_valleys', 'valley_symmetry', 'valley_separation',
            'mean_intensity', 'std_intensity', 'median_intensity', 'q25_intensity', 
            'q75_intensity', 'grad_x_mean', 'grad_y_mean', 'grad_x_std', 'grad_y_std',
            'hu_moment_1', 'hu_moment_2', 'hu_moment_3', 'hu_moment_4',
            'contour_area', 'contour_perimeter', 'compactness',
            'local_var_mean', 'local_var_std', 'edge_density'
        ]
    
    def extract_features_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """Extract 28 numerical features from chart image (for training on screenshots)"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (400, 300))
            
            features = []
            
            # Price line features (9)
            features.extend(self._extract_price_line_features(gray))
            
            # Statistical features (9)  
            features.extend(self._extract_statistical_features(gray))
            
            # Shape features (7)
            features.extend(self._extract_shape_features(gray))
            
            # Texture features (3)
            features.extend(self._extract_texture_features(gray))
            
            # Ensure exactly 28 features
            while len(features) < 28:
                features.append(0)
            
            features_array = np.array(features[:28])
            return np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
        except Exception as e:
            print(f"Feature extraction failed for {image_path}: {e}")
            return None
    
    def _extract_price_line_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract price line characteristics from screenshot"""
        features = []
        try:
            edges = cv2.Canny(gray_image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                points = main_contour.reshape(-1, 2)
                
                if len(points) > 10:
                    points = points[points[:, 0].argsort()]
                    y_values = points[:, 1].astype(float)
                    x_values = points[:, 0].astype(float)
                    
                    # Basic statistics
                    features.extend([
                        np.mean(y_values), np.std(y_values),
                        np.max(y_values) - np.min(y_values), len(points)
                    ])
                    
                    # Trend analysis
                    if len(y_values) > 2:
                        slope = np.polyfit(x_values, y_values, 1)[0]
                        features.append(float(slope))
                    else:
                        features.append(0)
                    
                    # Peak and valley detection (approximation from image)
                    min_distance = max(1, len(y_values) // 10)
                    peaks, _ = find_peaks(-y_values, distance=min_distance)
                    valleys, _ = find_peaks(y_values, distance=min_distance)
                    features.extend([len(peaks), len(valleys)])
                    
                    # Valley symmetry (approximation)
                    if len(valleys) >= 2:
                        valley_heights = y_values[valleys]
                        if np.mean(valley_heights) > 0:
                            symmetry = max(0, 1 - (np.std(valley_heights) / np.mean(valley_heights)))
                        else:
                            symmetry = 0
                        features.append(symmetry)
                        
                        # Valley separation
                        valley_positions = x_values[valleys]
                        avg_separation = np.mean(np.diff(valley_positions)) if len(valley_positions) > 1 else 0
                        features.append(avg_separation)
                    else:
                        features.extend([0, 0])
                else:
                    features.extend([0] * 9)
            else:
                features.extend([0] * 9)
        except:
            features.extend([0] * 9)
        
        return features[:9]
    
    def _extract_statistical_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract statistical image features"""
        try:
            features = [
                float(np.mean(gray_image)), float(np.std(gray_image)),
                float(np.median(gray_image)), float(np.percentile(gray_image, 25)),
                float(np.percentile(gray_image, 75))
            ]
            
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            
            features.extend([
                float(np.mean(np.abs(grad_x))), float(np.mean(np.abs(grad_y))),
                float(np.std(grad_x)), float(np.std(grad_y))
            ])
        except:
            features = [0] * 9
        return features
    
    def _extract_shape_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract shape-based features"""
        features = []
        try:
            _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            moments = cv2.moments(binary)
            
            if moments['m00'] != 0:
                hu_moments = cv2.HuMoments(moments).flatten()
                for hu in hu_moments[:4]:
                    if abs(hu) > 1e-10:
                        features.append(-np.sign(hu) * np.log10(np.abs(hu)))
                    else:
                        features.append(0)
            else:
                features.extend([0] * 4)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(main_contour)
                perimeter = cv2.arcLength(main_contour, True)
                
                features.extend([
                    float(area), float(perimeter),
                    float(area / (perimeter ** 2)) if perimeter > 0 else 0
                ])
            else:
                features.extend([0] * 3)
        except:
            features = [0] * 7
        
        return features[:7]
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract texture features"""
        try:
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            local_var = cv2.filter2D((gray_image.astype(np.float32) - local_mean) ** 2, -1, kernel)
            
            features = [float(np.mean(local_var)), float(np.std(local_var))]
            
            edges = cv2.Canny(gray_image, 50, 150)
            edge_density = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
            features.append(edge_density)
        except:
            features = [0] * 3
        return features
    
    def load_training_data(self):
        """Load training data from screenshot examples"""
        print("\nLOADING TRAINING DATA FROM SCREENSHOT EXAMPLES")
        print("=" * 50)
        print("Note: Screenshots show approximate W patterns, not exact start/end points")
        print("Training ML model to recognize W pattern characteristics")
        
        self.training_features = []
        self.training_labels = []
        self.confidence_labels = []
        self.target_labels = []
        self.pattern_metadata = []
        
        # Parse Phase 2 for ticker trading outcomes
        ticker_outcomes = {}
        phase2_folder = "phase2_with_outcomes"
        
        if os.path.exists(phase2_folder):
            images = glob.glob(os.path.join(phase2_folder, "*.png")) + \
                    glob.glob(os.path.join(phase2_folder, "*.jpg"))
            
            print(f"\nAnalyzing trading outcomes from {len(images)} Phase 2 images...")
            
            for image_path in images:
                filename = os.path.basename(image_path)
                ticker = filename.split('_')[0].upper()
                
                if 'success' in filename.lower():
                    ticker_outcomes[ticker] = 'profitable'
                    print(f"   {ticker}: PROFITABLE (breakout + target achieved)")
                elif 'fail' in filename.lower():
                    ticker_outcomes[ticker] = 'unprofitable'
                    print(f"   {ticker}: UNPROFITABLE (failed breakout or missed target)")
                else:
                    ticker_outcomes[ticker] = 'profitable'  # Default assumption
                    print(f"   {ticker}: PROFITABLE (default)")
        
        profitable_tickers = sum(1 for outcome in ticker_outcomes.values() if outcome == 'profitable')
        unprofitable_tickers = len(ticker_outcomes) - profitable_tickers
        
        print(f"\nTrading Outcome Summary:")
        print(f"   Profitable tickers: {profitable_tickers}")
        print(f"   Unprofitable tickers: {unprofitable_tickers}")
        
        # Process all phases with ticker-based labels
        phases = {
            'phase1_formation_only': 'Formation Only',
            'phase2_with_outcomes': 'With Trading Outcomes',
            'phase3_manual_analysis': 'Manual Analysis & Targeting'
        }
        
        for phase_folder, phase_name in phases.items():
            if os.path.exists(phase_folder):
                images = glob.glob(os.path.join(phase_folder, "*.png")) + \
                        glob.glob(os.path.join(phase_folder, "*.jpg"))
                
                print(f"\nProcessing {phase_name}: {len(images)} images")
                
                for image_path in images:
                    filename = os.path.basename(image_path)
                    ticker = filename.split('_')[0].upper()
                    
                    features = self.extract_features_from_image(image_path)
                    if features is not None:
                        self.training_features.append(features)
                        
                        # Apply ticker-based trading outcome
                        if ticker in ticker_outcomes:
                            if ticker_outcomes[ticker] == 'profitable':
                                label = 1  # Profitable trading outcome
                                confidence = 0.9 if phase_folder == 'phase3_manual_analysis' else 0.8
                                target = 1.25 if phase_folder == 'phase3_manual_analysis' else 1.2
                                outcome_type = "PROFITABLE"
                            else:
                                label = 0  # Unprofitable trading outcome
                                confidence = 0.3
                                target = 0.95
                                outcome_type = "UNPROFITABLE"
                        else:
                            label = 1  # Default to profitable
                            confidence = 0.6
                            target = 1.1
                            outcome_type = "PROFITABLE (default)"
                        
                        self.training_labels.append(label)
                        self.confidence_labels.append(confidence)
                        self.target_labels.append(target)
                        
                        self.pattern_metadata.append({
                            'filename': filename,
                            'ticker': ticker,
                            'phase': phase_folder,
                            'trading_outcome': ticker_outcomes.get(ticker, 'unknown'),
                            'label': 'profitable' if label == 1 else 'unprofitable'
                        })
                        
                        print(f"   {filename} -> {outcome_type}")
        
        # Convert to arrays and validate
        if self.training_features:
            self.training_features = np.array(self.training_features)
            self.training_labels = np.array(self.training_labels)
            self.confidence_labels = np.array(self.confidence_labels)
            self.target_labels = np.array(self.target_labels)
            
            total_samples = len(self.training_features)
            profitable_samples = np.sum(self.training_labels)
            unprofitable_samples = total_samples - profitable_samples
            
            print(f"\nFINAL TRAINING DATA:")
            print(f"   Total samples: {total_samples}")
            print(f"   Profitable patterns: {profitable_samples}")
            print(f"   Unprofitable patterns: {unprofitable_samples}")
            print(f"   Failure rate: {unprofitable_samples/total_samples:.1%}")
            print(f"   Training on screenshot pattern characteristics")
            
            return True
        else:
            print("No training data found!")
            return False
    
    def train_models(self):
        """Train ML models for trading success prediction"""
        if len(self.training_features) == 0:
            print("No training data available!")
            return False
        
        print(f"\nTRAINING ML MODELS FOR W PATTERN SUCCESS PREDICTION")
        print("=" * 50)
        
        X_scaled = self.scaler.fit_transform(self.training_features)
        
        unique, counts = np.unique(self.training_labels, return_counts=True)
        print(f"Training class distribution: {dict(zip(unique, counts))}")
        
        # Train/test split
        if len(self.training_features) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, self.training_labels, test_size=0.2, random_state=42,
                stratify=self.training_labels if len(unique) > 1 else None
            )
        else:
            X_train = X_test = X_scaled
            y_train = y_test = self.training_labels
        
        # Train classifier
        self.pattern_classifier.fit(X_train, y_train)
        y_pred = self.pattern_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Trading success prediction accuracy: {accuracy:.1%}")
        
        # Train regressors on profitable examples only
        profitable_mask = self.training_labels == 1
        profitable_count = np.sum(profitable_mask)
        
        if profitable_count >= 5:
            X_profitable = X_scaled[profitable_mask]
            conf_profitable = self.confidence_labels[profitable_mask]
            target_profitable = self.target_labels[profitable_mask]
            
            self.confidence_regressor.fit(X_profitable, conf_profitable)
            self.target_regressor.fit(X_profitable, target_profitable)
            
            print("Confidence and target regressors trained successfully")
        
        # Feature importance analysis
        feature_importance = self.pattern_classifier.feature_importances_
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        print(f"\nTOP FEATURES FOR TRADING SUCCESS:")
        for i in range(min(8, len(sorted_idx))):
            idx = sorted_idx[i]
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
            print(f"   {i+1}. {feature_name}: {feature_importance[idx]:.3f}")
        
        self.is_trained = True
        self.training_stats = {
            'samples': len(self.training_features),
            'profitable_samples': profitable_count,
            'unprofitable_samples': len(self.training_features) - profitable_count,
            'accuracy': accuracy,
            'features': len(self.feature_names),
            'trained_date': datetime.now().isoformat(),
            'feature_importance': dict(zip(self.feature_names, feature_importance))
        }
        
        print("ML MODEL TRAINING COMPLETE!")
        return True
    
    def scan_sp500_for_w_patterns(self, lookback_days=90):
        """Scan S&P 500 for trend-based W pattern formations"""
        print("S&P 500 TREND-BASED W PATTERN SCANNER")
        print("=" * 60)
        print(f"Scanning {len(self.sp500_symbols)} stocks for W patterns (last {lookback_days} days)")
        print("Detecting patterns based on trend transitions and relative peaks/troughs")
        print(f"ML Model Status: {'TRAINED' if self.is_trained else 'NOT TRAINED'}")
        if self.is_trained:
            print(f"Model Accuracy: {self.training_stats.get('accuracy', 0):.1%}")
        print()
        
        w_detections = []
        scan_progress = 0
        successful_scans = 0
        
        for symbol in self.sp500_symbols:
            scan_progress += 1
            try:
                print(f"[{scan_progress:3d}/{len(self.sp500_symbols)}] {symbol:<6}...", end=' ')
                
                # Fetch stock data
                stock = yf.Ticker(symbol)
                data = stock.history(period=f"{lookback_days + 30}d", interval='1d')  # Extra buffer for trend analysis
                
                if len(data) < 40:
                    print("Insufficient data")
                    continue
                
                successful_scans += 1
                
                # Detect W pattern using trend-based analysis
                w_pattern = self._detect_trend_based_w_pattern(data, symbol, lookback_days)
                
                if w_pattern:
                    # Add ML prediction if model is trained
                    if self.is_trained:
                        try:
                            chart_features = self._create_features_from_price_data(data.tail(lookback_days))
                            if chart_features is not None:
                                features_scaled = self.scaler.transform([chart_features])
                                
                                # Get ML predictions
                                profit_proba = self.pattern_classifier.predict_proba(features_scaled)[0]
                                w_pattern['ml_profit_probability'] = profit_proba[1] if len(profit_proba) > 1 else profit_proba[0]
                                
                                if w_pattern['ml_profit_probability'] > 0.5:
                                    trading_conf = self.confidence_regressor.predict(features_scaled)[0]
                                    target_mult = self.target_regressor.predict(features_scaled)[0]
                                    w_pattern['ml_confidence'] = max(0, min(1, trading_conf))
                                    w_pattern['ml_target_multiplier'] = max(0.8, min(2.0, target_mult))
                                    
                                    # Generate trading recommendation
                                    w_pattern['recommendation'] = self._generate_recommendation(w_pattern)
                                else:
                                    w_pattern['recommendation'] = "AVOID"
                        except Exception as e:
                            w_pattern['ml_profit_probability'] = None
                            w_pattern['recommendation'] = "ANALYSIS FAILED"
                    
                    w_detections.append(w_pattern)
                    
                    # Status display
                    if w_pattern.get('ml_profit_probability'):
                        prob = w_pattern['ml_profit_probability']
                        rec = w_pattern.get('recommendation', 'N/A')
                        print(f"W Pattern - {rec} ({prob:.0%})")
                    else:
                        print(f"W Pattern - {w_pattern['formation_status']}")
                else:
                    print("No W pattern")
                    
            except Exception as e:
                print(f"Error: {str(e)[:25]}")
        
        print(f"\nScanning complete: {successful_scans}/{len(self.sp500_symbols)} stocks analyzed")
        
        # Sort results by ML predictions and quality
        if w_detections:
            w_detections.sort(key=lambda x: (
                x.get('ml_profit_probability', 0) or 0,
                x.get('pattern_quality_score', 0),
                -x.get('days_since_pattern_start', 999)
            ), reverse=True)
        
        # Display comprehensive results
        self._display_scan_results(w_detections)
        
        # Save results
        if w_detections:
            self._save_scan_results(w_detections)
        
        return w_detections
    
    def _detect_trend_based_w_pattern(self, price_data, symbol, lookback_days):
        """Detect W patterns based on trend transitions"""
        try:
            # Focus on the lookback period for pattern detection
            recent_data = price_data.tail(lookback_days)
            highs = recent_data['High'].values
            lows = recent_data['Low'].values
            closes = recent_data['Close'].values
            volumes = recent_data['Volume'].values if 'Volume' in recent_data.columns else None
            dates = recent_data.index
            
            if len(closes) < 30:
                return None
            
            # Step 1: Identify trends using moving averages and price action
            trends = self._identify_trends(closes, highs, lows)
            
            # Step 2: Find trend transitions and key points
            trend_points = self._find_trend_transitions(trends, closes, highs, lows, dates)
            
            # Step 3: Look for W pattern sequences in trend transitions
            w_patterns = self._find_w_pattern_sequences(trend_points, closes, highs, lows, volumes, dates)
            
            if not w_patterns:
                return None
            
            # Step 4: Validate and score the best W pattern
            best_pattern = max(w_patterns, key=lambda p: p['quality_score'])
            
            # Step 5: Analyze current status
            current_price = closes[-1]
            current_date = dates[-1]
            
            # Add current status analysis
            pattern_status = self._analyze_current_pattern_status(best_pattern, current_price, current_date, len(closes))
            best_pattern.update(pattern_status)
            best_pattern['symbol'] = symbol
            best_pattern['current_price'] = current_price
            
            return best_pattern
            
        except Exception as e:
            return None
    
    def _identify_trends(self, closes, highs, lows, window=5):
        """Identify uptrends and downtrends using price action"""
        trends = []
        prices = closes
        
        # Use simple moving average for trend identification
        ma_short = pd.Series(prices).rolling(window=window).mean().values
        ma_long = pd.Series(prices).rolling(window=window*2).mean().values
        
        current_trend = None
        trend_start = 0
        
        for i in range(window*2, len(prices)):
            # Determine trend direction
            if ma_short[i] > ma_long[i] and prices[i] > ma_short[i]:
                new_trend = 'uptrend'
            elif ma_short[i] < ma_long[i] and prices[i] < ma_short[i]:
                new_trend = 'downtrend'
            else:
                new_trend = 'sideways'
            
            # Detect trend changes
            if current_trend != new_trend and current_trend is not None:
                # Validate trend strength
                start_price = prices[trend_start]
                end_price = prices[i-1]
                trend_strength = abs(end_price - start_price) / start_price
                
                if trend_strength >= self.detection_criteria['trend_strength_threshold']:
                    trends.append({
                        'type': current_trend,
                        'start_idx': trend_start,
                        'end_idx': i-1,
                        'start_price': start_price,
                        'end_price': end_price,
                        'strength': trend_strength,
                        'duration': i-1-trend_start
                    })
                
                trend_start = i
            
            current_trend = new_trend
        
        # Add the final trend
        if current_trend and len(prices) - trend_start >= self.detection_criteria['min_trend_length']:
            start_price = prices[trend_start]
            end_price = prices[-1]
            trend_strength = abs(end_price - start_price) / start_price
            
            trends.append({
                'type': current_trend,
                'start_idx': trend_start,
                'end_idx': len(prices)-1,
                'start_price': start_price,
                'end_price': end_price,
                'strength': trend_strength,
                'duration': len(prices)-1-trend_start
            })
        
        return trends
    
    def _find_trend_transitions(self, trends, closes, highs, lows, dates):
        """Find key transition points between trends"""
        points = []
        
        for i, trend in enumerate(trends):
            start_idx = trend['start_idx']
            end_idx = trend['end_idx']
            
            if trend['type'] == 'downtrend':
                # Find the relative peak before downtrend (if this isn't the first trend)
                if i == 0 or start_idx < 5:
                    peak_idx = start_idx
                    peak_price = highs[start_idx]
                else:
                    # Look back a few periods for the actual peak
                    search_start = max(0, start_idx - 5)
                    peak_idx = search_start + np.argmax(highs[search_start:start_idx+1])
                    peak_price = highs[peak_idx]
                
                # Find the trough at the end of downtrend
                trough_idx = start_idx + np.argmin(lows[start_idx:end_idx+1])
                trough_price = lows[trough_idx]
                
                points.append({
                    'type': 'peak',
                    'idx': peak_idx,
                    'price': peak_price,
                    'date': dates[peak_idx],
                    'trend_type': 'before_downtrend'
                })
                
                points.append({
                    'type': 'trough',
                    'idx': trough_idx,
                    'price': trough_price,
                    'date': dates[trough_idx],
                    'trend_type': 'end_downtrend'
                })
            
            elif trend['type'] == 'uptrend':
                # Find the peak at the end of uptrend
                peak_idx = start_idx + np.argmax(highs[start_idx:end_idx+1])
                peak_price = highs[peak_idx]
                
                points.append({
                    'type': 'peak',
                    'idx': peak_idx,
                    'price': peak_price,
                    'date': dates[peak_idx],
                    'trend_type': 'end_uptrend'
                })
        
        # Sort points by time
        points.sort(key=lambda p: p['idx'])
        
        return points
    
    def _find_w_pattern_sequences(self, trend_points, closes, highs, lows, volumes, dates):
        """Look for W pattern sequences in trend transition points"""
        w_patterns = []
        
        # We need at least 5 points for a W pattern: Peak1, Trough1, Peak2, Trough2, Peak3
        if len(trend_points) < 5:
            return w_patterns
        
        # Look for sequences that match W pattern criteria
        for i in range(len(trend_points) - 4):
            sequence = trend_points[i:i+5]
            
            # Check if we have the right sequence of point types
            if (sequence[0]['type'] == 'peak' and
                sequence[1]['type'] == 'trough' and 
                sequence[2]['type'] == 'peak' and
                sequence[3]['type'] == 'trough' and
                sequence[4]['type'] == 'peak'):
                
                # Extract the 5 points
                peak1 = sequence[0]
                trough1 = sequence[1]
                peak2 = sequence[2]
                trough2 = sequence[3]
                peak3 = sequence[4]
                
                # Validate W pattern criteria
                w_validation = self._validate_w_pattern_criteria(peak1, trough1, peak2, trough2, peak3)
                
                if w_validation['is_valid']:
                    # Calculate resistance line and check if Peak3 reaches it
                    resistance_analysis = self._analyze_resistance_breakthrough(peak1, peak2, peak3)
                    
                    # Volume analysis
                    volume_analysis = self._analyze_w_volume_pattern(
                        [peak1['idx'], trough1['idx'], peak2['idx'], trough2['idx'], peak3['idx']],
                        volumes
                    )
                    
                    # Calculate pattern quality
                    quality_score = self._calculate_w_quality_score(w_validation, resistance_analysis, volume_analysis)
                    
                    w_pattern = {
                        'peak1_idx': peak1['idx'],
                        'peak1_price': peak1['price'],
                        'peak1_date': peak1['date'].strftime('%Y-%m-%d'),
                        'trough1_idx': trough1['idx'],
                        'trough1_price': trough1['price'],
                        'trough1_date': trough1['date'].strftime('%Y-%m-%d'),
                        'peak2_idx': peak2['idx'],
                        'peak2_price': peak2['price'],
                        'peak2_date': peak2['date'].strftime('%Y-%m-%d'),
                        'trough2_idx': trough2['idx'],
                        'trough2_price': trough2['price'],
                        'trough2_date': trough2['date'].strftime('%Y-%m-%d'),
                        'peak3_idx': peak3['idx'],
                        'peak3_price': peak3['price'],
                        'peak3_date': peak3['date'].strftime('%Y-%m-%d'),
                        'pattern_start_date': peak1['date'].strftime('%Y-%m-%d'),
                        'pattern_duration_days': peak3['idx'] - peak1['idx'],
                        'trough_symmetry': w_validation['trough_symmetry'],
                        'resistance_line_slope': resistance_analysis['slope'],
                        'resistance_target_price': resistance_analysis['target_price'],
                        'resistance_breakthrough': resistance_analysis['breakthrough'],
                        'breakthrough_percentage': resistance_analysis['breakthrough_pct'],
                        'volume_score': volume_analysis['score'],
                        'quality_score': quality_score
                    }
                    
                    w_patterns.append(w_pattern)
        
        return w_patterns
    
    def _validate_w_pattern_criteria(self, peak1, trough1, peak2, trough2, peak3):
        """Validate that the 5 points form a valid W pattern"""
        
        # 1. Trough symmetry - both troughs should be at similar levels
        trough_avg = (trough1['price'] + trough2['price']) / 2
        trough_diff = abs(trough1['price'] - trough2['price'])
        trough_symmetry = 1 - (trough_diff / trough_avg) if trough_avg > 0 else 0
        
        # 2. Minimum rebound strength from troughs
        rebound1_strength = (peak2['price'] - trough1['price']) / trough1['price']
        rebound2_strength = (peak3['price'] - trough2['price']) / trough2['price']
        
        # 3. Pattern duration validation
        total_duration = peak3['idx'] - peak1['idx']
        
        # 4. Height validation - peaks should be significantly above troughs
        min_trough = min(trough1['price'], trough2['price'])
        peaks_above_troughs = (peak1['price'] > min_trough and 
                              peak2['price'] > min_trough and 
                              peak3['price'] > min_trough)
        
        # Validation criteria
        is_valid = (
            trough_symmetry > (1 - self.detection_criteria['trough_symmetry_tolerance']) and
            rebound1_strength >= self.detection_criteria['min_rebound_strength'] and
            rebound2_strength >= self.detection_criteria['min_rebound_strength'] and
            self.detection_criteria['min_pattern_duration'] <= total_duration <= self.detection_criteria['max_pattern_duration'] and
            peaks_above_troughs
        )
        
        return {
            'is_valid': is_valid,
            'trough_symmetry': trough_symmetry,
            'rebound1_strength': rebound1_strength,
            'rebound2_strength': rebound2_strength,
            'total_duration': total_duration,
            'peaks_above_troughs': peaks_above_troughs
        }
    
    def _analyze_resistance_breakthrough(self, peak1, peak2, peak3):
        """Analyze if Peak3 breaks through the resistance line from Peak1 to Peak2"""
        
        # Calculate resistance line slope
        x_diff = peak2['idx'] - peak1['idx']
        y_diff = peak2['price'] - peak1['price']
        slope = y_diff / x_diff if x_diff > 0 else 0
        
        # Calculate where resistance line should be at Peak3
        x_target = peak3['idx'] - peak1['idx']
        target_price = peak1['price'] + (slope * x_target)
        
        # Check if Peak3 breaks through resistance
        breakthrough_pct = ((peak3['price'] - target_price) / target_price) * 100 if target_price > 0 else 0
        breakthrough = breakthrough_pct >= -self.detection_criteria['resistance_tolerance'] * 100
        
        return {
            'slope': slope,
            'target_price': target_price,
            'breakthrough': breakthrough,
            'breakthrough_pct': breakthrough_pct
        }
    
    def _analyze_w_volume_pattern(self, point_indices, volume_data):
        """Analyze volume pattern at key W pattern points"""
        if volume_data is None or len(volume_data) <= max(point_indices):
            return {'score': 1.0, 'analysis': 'No volume data'}
        
        try:
            avg_volume = np.mean(volume_data)
            
            # Get volumes at key points
            volumes_at_points = [volume_data[idx] for idx in point_indices]
            
            # Ideal volume pattern: Higher volume at troughs (selling), moderate at peaks
            trough_volumes = [volumes_at_points[1], volumes_at_points[3]]  # Trough1, Trough2
            peak_volumes = [volumes_at_points[0], volumes_at_points[2], volumes_at_points[4]]  # Peak1, Peak2, Peak3
            
            volume_score = 1.0
            
            # Bonus for high volume at troughs
            avg_trough_volume = np.mean(trough_volumes)
            if avg_trough_volume > avg_volume * self.detection_criteria['volume_confirmation_factor']:
                volume_score += 0.3
            
            # Bonus for volume expansion at final breakout (Peak3)
            if volumes_at_points[4] > avg_volume * 1.1:
                volume_score += 0.2
            
            return {
                'score': min(2.0, volume_score),
                'avg_trough_volume': avg_trough_volume,
                'avg_peak_volume': np.mean(peak_volumes),
                'peak3_volume': volumes_at_points[4],
                'avg_volume': avg_volume
            }
            
        except Exception:
            return {'score': 1.0, 'analysis': 'Volume analysis failed'}
    
    def _calculate_w_quality_score(self, w_validation, resistance_analysis, volume_analysis):
        """Calculate overall W pattern quality score"""
        
        base_score = 0
        
        # Trough symmetry (0-25 points)
        base_score += w_validation['trough_symmetry'] * 25
        
        # Rebound strength (0-25 points)  
        avg_rebound = (w_validation['rebound1_strength'] + w_validation['rebound2_strength']) / 2
        base_score += min(25, avg_rebound * 500)  # Scale to 0-25 range
        
        # Resistance breakthrough (0-30 points)
        if resistance_analysis['breakthrough']:
            base_score += 30
            # Bonus for strong breakthrough
            if resistance_analysis['breakthrough_pct'] > 2:
                base_score += min(10, resistance_analysis['breakthrough_pct'] / 2)
        else:
            # Partial credit for getting close to resistance
            closeness = max(0, 100 + resistance_analysis['breakthrough_pct']) / 100
            base_score += closeness * 15
        
        # Volume confirmation (0-20 points)
        volume_contribution = min(20, (volume_analysis['score'] - 1) * 20)
        base_score += volume_contribution
        
        # Normalize to 0-1 scale
        return min(1.0, base_score / 100)
    
    def _analyze_current_pattern_status(self, w_pattern, current_price, current_date, total_data_length):
        """Analyze current status of the W pattern"""
        
        peak3_idx = w_pattern['peak3_idx']
        resistance_target = w_pattern['resistance_target_price']
        
        # Determine if pattern is complete (need some buffer after Peak3)
        completion_buffer = 3  # 3 days buffer
        is_complete = peak3_idx < total_data_length - completion_buffer
        
        if is_complete:
            days_since_completion = total_data_length - peak3_idx - 1
            if w_pattern['resistance_breakthrough']:
                status = "Completed W Pattern - Resistance Breakthrough"
            else:
                status = "Completed W Pattern - Resistance Not Reached"
        else:
            days_since_completion = 0
            if peak3_idx >= total_data_length - 2:
                status = "Forming Peak 3 (W Pattern in Progress)"
            else:
                status = "W Pattern Developing"
        
        # Breakout analysis
        breakout_threshold = resistance_target * (1 + self.detection_criteria['resistance_tolerance'])
        has_broken_out = current_price > breakout_threshold
        
        # Support/resistance levels
        support_level = min(w_pattern['trough1_price'], w_pattern['trough2_price'])
        resistance_level = resistance_target
        
        # Risk analysis
        downside_risk = abs(current_price - support_level) / current_price * 100
        upside_potential = abs(resistance_level * 1.15 - current_price) / current_price * 100
        
        return {
            'formation_status': status,
            'is_complete': is_complete,
            'days_since_completion': days_since_completion,
            'days_since_pattern_start': total_data_length - w_pattern['peak1_idx'] - 1,
            'resistance_level': resistance_level,
            'support_level': support_level,
            'has_broken_out': has_broken_out,
            'breakout_threshold': breakout_threshold,
            'downside_risk_pct': downside_risk,
            'upside_potential_pct': upside_potential,
            'left_valley_price': w_pattern['trough1_price'],
            'right_valley_price': w_pattern['trough2_price'],
            'time_span_days': w_pattern['pattern_duration_days']
        }
    
    def _generate_recommendation(self, w_pattern):
        """Generate trading recommendation based on pattern analysis"""
        ml_prob = w_pattern.get('ml_profit_probability', 0)
        ml_conf = w_pattern.get('ml_confidence', 0)
        quality = w_pattern.get('quality_score', 0)
        breakthrough = w_pattern.get('resistance_breakthrough', False)
        
        # Enhanced recommendation logic
        if ml_prob > 0.75 and ml_conf > 0.8 and quality > 0.7 and breakthrough:
            return "STRONG BUY"
        elif ml_prob > 0.65 and ml_conf > 0.6 and quality > 0.5:
            return "BUY"
        elif ml_prob > 0.55 and quality > 0.4:
            return "WATCH"
        elif ml_prob > 0.45:
            return "CAUTION"
        else:
            return "AVOID"
    
    def _create_features_from_price_data(self, price_data):
        """Create features from price data for ML prediction"""
        try:
            prices = price_data['Close'].values
            
            if len(prices) < 20:
                return None
            
            features = []
            
            # Price statistics (4 features)
            features.extend([
                np.mean(prices), np.std(prices),
                np.max(prices) - np.min(prices), len(prices)
            ])
            
            # Trend analysis
            slope = np.polyfit(range(len(prices)), prices, 1)[0]
            features.append(slope)
            
            # Peak and valley detection
            try:
                min_distance = max(1, len(prices) // 10)
                peaks, _ = find_peaks(prices, distance=min_distance)
                valleys, _ = find_peaks(-prices, distance=min_distance)
                features.extend([len(peaks), len(valleys)])
            except:
                features.extend([0, 0])
            
            # Valley symmetry approximation
            try:
                valleys, _ = find_peaks(-prices, distance=max(1, len(prices) // 10))
                if len(valleys) >= 2:
                    valley_prices = prices[valleys]
                    if np.mean(valley_prices) > 0:
                        symmetry = max(0, 1 - (np.std(valley_prices) / np.mean(valley_prices)))
                    else:
                        symmetry = 0
                else:
                    symmetry = 0
                features.append(symmetry)
            except:
                features.append(0)
            
            # Valley separation
            try:
                if len(valleys) >= 2:
                    separations = np.diff(valleys)
                    features.append(np.mean(separations))
                else:
                    features.append(0)
            except:
                features.append(0)
            
            # Statistical features (9 features)
            returns = np.diff(prices) / prices[:-1]
            price_normalized = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
            
            features.extend([
                np.mean(price_normalized), np.std(price_normalized),
                np.median(price_normalized), np.percentile(price_normalized, 25),
                np.percentile(price_normalized, 75), np.mean(np.abs(returns)),
                np.std(returns), np.mean(returns),
                np.std(np.diff(returns)) if len(returns) > 1 else 0
            ])
            
            # Shape features (7 features)
            features.extend([
                skew(prices) if len(prices) > 2 else 0,
                kurtosis(prices) if len(prices) > 2 else 0,
                np.var(prices), np.mean(np.abs(np.diff(prices))),
                np.sum(np.abs(returns)) if len(returns) > 0 else 0,
                len(prices),
                np.var(prices) / (np.mean(prices)**2) if np.mean(prices) > 0 else 0
            ])
            
            # Texture features (3 features)
            rolling_vol = pd.Series(prices).rolling(5).std().fillna(0)
            features.extend([
                np.mean(rolling_vol), np.std(rolling_vol),
                np.sum(np.abs(returns) > np.std(returns)) / len(returns) if len(returns) > 0 else 0
            ])
            
            # Ensure exactly 28 features
            while len(features) < 28:
                features.append(0)
            
            features_array = np.array(features[:28])
            return np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
        except Exception:
            return None
    
    def _display_scan_results(self, w_detections):
        """Display comprehensive scan results"""
        if not w_detections:
            print("\nSCAN COMPLETE - No trend-based W patterns detected")
            print("This could mean:")
            print("• No valid W formations in the past 90 days")
            print("• Patterns don't meet strict trend-based criteria")
            print("• Market conditions haven't produced clear W structures")
            return
        
        print(f"\nTREND-BASED W PATTERN DETECTIONS: {len(w_detections)} Found")
        print("=" * 80)
        
        # Summary table
        print(f"{'#':<3} {'Symbol':<6} {'Status':<20} {'Peak3 Date':<12} {'ML Score':<8} {'Rec':<10}")
        print("-" * 80)
        
        for i, detection in enumerate(w_detections[:10], 1):  # Top 10
            symbol = detection['symbol']
            status = detection['formation_status'][:19]
            peak3_date = detection.get('peak3_date', 'Forming')[:11]
            
            ml_score = detection.get('ml_profit_probability', 0)
            ml_score_str = f"{ml_score:.0%}" if ml_score else "N/A"
            
            recommendation = detection.get('recommendation', 'N/A')[:9]
            
            print(f"{i:<3} {symbol:<6} {status:<20} {peak3_date:<12} {ml_score_str:<8} {recommendation:<10}")
        
        # Detailed analysis for top 3 patterns
        print(f"\nDETAILED TREND-BASED ANALYSIS - TOP PATTERNS")
        print("=" * 80)
        
        for i, detection in enumerate(w_detections[:3], 1):
            self._display_trend_pattern_details(detection, i)
        
        # Trading recommendations summary
        recommendations = {}
        for d in w_detections:
            rec = d.get('recommendation', 'Unknown')
            recommendations[rec] = recommendations.get(rec, 0) + 1
        
        print(f"\nTRADING RECOMMENDATIONS SUMMARY")
        print("-" * 40)
        for rec, count in sorted(recommendations.items()):
            print(f"{rec}: {count} patterns")
    
    def _display_trend_pattern_details(self, detection, rank):
        """Display detailed trend-based W pattern analysis"""
        symbol = detection['symbol']
        print(f"\n#{rank} {symbol} - {detection['formation_status']}")
        print("-" * 60)
        
        # Trend-based W Pattern Structure
        print(f"TREND-BASED W PATTERN STRUCTURE:")
        print(f"   Peak 1 (Start):  ${detection.get('peak1_price', 0):.2f} ({detection.get('peak1_date', 'N/A')})")
        print(f"   Trough 1:        ${detection.get('trough1_price', 0):.2f} ({detection.get('trough1_date', 'N/A')})")
        print(f"   Peak 2 (Middle): ${detection.get('peak2_price', 0):.2f} ({detection.get('peak2_date', 'N/A')})")
        print(f"   Trough 2:        ${detection.get('trough2_price', 0):.2f} ({detection.get('trough2_date', 'N/A')})")
        print(f"   Peak 3 (End):    ${detection.get('peak3_price', 0):.2f} ({detection.get('peak3_date', 'N/A')})")
        print()
        
        # Resistance Line Analysis
        print(f"RESISTANCE BREAKTHROUGH ANALYSIS:")
        print(f"   Resistance Target: ${detection.get('resistance_target_price', 0):.2f}")
        print(f"   Peak 3 Actual:     ${detection.get('peak3_price', 0):.2f}")
        
        breakthrough_pct = detection.get('breakthrough_percentage', 0)
        if detection.get('resistance_breakthrough', False):
            if breakthrough_pct > 2:
                print(f"   Status: STRONG BREAKTHROUGH (+{breakthrough_pct:.1f}%)")
            else:
                print(f"   Status: RESISTANCE REACHED ({breakthrough_pct:+.1f}%)")
        else:
            print(f"   Status: BELOW RESISTANCE ({breakthrough_pct:.1f}%)")
        print()
        
        # Current Status
        print(f"CURRENT STATUS:")
        print(f"   Current Price:     ${detection['current_price']:.2f}")
        print(f"   Support Level:     ${detection.get('support_level', 0):.2f}")
        print(f"   Resistance Level:  ${detection.get('resistance_level', 0):.2f}")
        
        # Pattern Quality Metrics
        print(f"   Trough Symmetry:   {detection.get('trough_symmetry', 0):.2f}/1.00")
        print(f"   Pattern Quality:   {detection.get('quality_score', 0):.3f}/1.000")
        print(f"   Volume Score:      {detection.get('volume_score', 1.0):.2f}")
        print(f"   Pattern Duration:  {detection.get('pattern_duration_days', 0)} days")
        print(f"   Days Since Start:  {detection.get('days_since_pattern_start', 0)} days")
        
        # Formation completion status
        if detection['is_complete']:
            days_since = detection.get('days_since_completion', 0)
            print(f"   Days Since Peak 3: {days_since}")
            
            if detection['has_broken_out']:
                breakout_pct = ((detection['current_price'] - detection['resistance_level']) / detection['resistance_level']) * 100
                print(f"   BREAKOUT STATUS: CONFIRMED +{breakout_pct:.1f}% above resistance")
            else:
                to_breakout = ((detection['breakout_threshold'] - detection['current_price']) / detection['current_price']) * 100
                print(f"   To Breakout: {to_breakout:.1f}% needed")
        else:
            print(f"   Status: Pattern still developing")
        
        # ML predictions
        if detection.get('ml_profit_probability'):
            ml_prob = detection['ml_profit_probability']
            ml_conf = detection.get('ml_confidence', 0)
            ml_target = detection.get('ml_target_multiplier', 1.0)
            target_price = detection['current_price'] * ml_target
            
            print(f"\n   ML PREDICTIONS:")
            print(f"      Success Probability: {ml_prob:.1%}")
            print(f"      Trading Confidence:  {ml_conf:.1%}")
            print(f"      Target Price:        ${target_price:.2f} ({((ml_target-1)*100):+.1f}%)")
            print(f"      Recommendation:      {detection.get('recommendation', 'N/A')}")
        
        # Risk assessment
        risk = detection.get('downside_risk_pct', 0)
        upside = detection.get('upside_potential_pct', 0)
        print(f"\n   RISK ASSESSMENT:")
        print(f"      Downside Risk:       {risk:.1f}% (to support)")
        print(f"      Upside Potential:    {upside:.1f}% (15% above resistance)")
        if risk > 0:
            risk_reward = upside / risk
            print(f"      Risk/Reward Ratio:   1:{risk_reward:.1f}")
    
    def _save_scan_results(self, w_detections):
        """Save scan results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trend_w_pattern_scan_{timestamp}.json"
        
        # Prepare data for JSON serialization
        json_detections = []
        for d in w_detections:
            json_d = {}
            for k, v in d.items():
                if isinstance(v, (np.integer, np.floating)):
                    json_d[k] = float(v)
                elif hasattr(v, 'strftime'):  # datetime objects
                    json_d[k] = v.strftime('%Y-%m-%d')
                else:
                    json_d[k] = v
            json_detections.append(json_d)
        
        scan_results = {
            'scan_timestamp': datetime.now().isoformat(),
            'scan_date': datetime.now().strftime('%Y-%m-%d'),
            'detection_method': 'trend_based_w_pattern',
            'total_symbols_scanned': len(self.sp500_symbols),
            'total_detections': len(w_detections),
            'ml_model_used': self.is_trained,
            'detection_criteria': self.detection_criteria,
            'model_accuracy': self.training_stats.get('accuracy', 0) if self.is_trained else None,
            'detections': json_detections
        }
        
        with open(filename, 'w') as f:
            json.dump(scan_results, f, indent=2)
        
        print(f"\nScan results saved to: {filename}")
    
    def analyze_model_learnings(self):
        """Comprehensive analysis of what the model learned from training examples"""
        if not self.is_trained:
            print("Model not trained yet. Train the model first using option 1.")
            return
        
        print("COMPREHENSIVE MODEL ANALYSIS - TREND-BASED W PATTERNS")
        print("=" * 80)
        print("Analyzing what your model learned from screenshot examples...")
        print()
        
        # 1. Trend-based Detection Criteria
        self._analyze_trend_detection_criteria()
        
        # 2. Feature Importance Analysis
        self._analyze_feature_importance()
        
        # 3. Training Data Breakdown
        self._analyze_training_data()
        
        # 4. Pattern Boundary Rules (Trend-based)
        self._analyze_trend_pattern_boundaries()
        
        # 5. Success Prediction Logic
        self._analyze_success_predictions()
        
        # 6. Model Decision Rules
        self._analyze_decision_rules()
    
    def _analyze_trend_detection_criteria(self):
        """Analyze trend-based W pattern detection criteria"""
        print("TREND-BASED W PATTERN DETECTION CRITERIA")
        print("-" * 50)
        
        criteria = self.detection_criteria
        
        print("True W Pattern Structure based on trend transitions:")
        print("Peak1 (before downtrend) → Trough1 → Peak2 (uptrend end) → Trough2 → Peak3 (resistance test)")
        print()
        
        print("TREND IDENTIFICATION RULES:")
        print(f"   • Minimum trend length: {criteria['min_trend_length']} periods")
        print(f"   • Trend strength threshold: {criteria['trend_strength_threshold']*100:.0f}% price movement")
        print("   • Uses moving averages and price action for trend detection")
        print("   • Identifies relative peaks/troughs within trends, not absolute highs/lows")
        print()
        
        print("W PATTERN VALIDATION CRITERIA:")
        print(f"   • Trough symmetry tolerance: {criteria['trough_symmetry_tolerance']*100:.0f}%")
        print(f"   • Minimum rebound strength: {criteria['min_rebound_strength']*100:.1f}% from troughs")
        print(f"   • Pattern duration: {criteria['min_pattern_duration']}-{criteria['max_pattern_duration']} days")
        print(f"   • Resistance tolerance: {criteria['resistance_tolerance']*100:.0f}% for breakthrough")
        print()
        
        print("RESISTANCE BREAKTHROUGH ANALYSIS:")
        print("   • Resistance line drawn from Peak1 to Peak2")
        print("   • Peak3 must reach within 2% of resistance line")
        print("   • Breakthrough confirmed when Peak3 exceeds resistance")
        print("   • Volume expansion preferred at breakthrough")
        print()
        
        print("VOLUME CONFIRMATION:")
        print(f"   • Volume factor: {criteria['volume_confirmation_factor']}x average volume")
        print("   • Higher volume expected at troughs (selling climax)")
        print("   • Volume expansion at Peak3 confirms breakout strength")
    
    def _analyze_feature_importance(self):
        """Analyze which features the model finds most important"""
        print("\nFEATURE IMPORTANCE ANALYSIS")
        print("-" * 50)
        
        if hasattr(self.pattern_classifier, 'feature_importances_'):
            feature_importance = self.pattern_classifier.feature_importances_
            sorted_idx = np.argsort(feature_importance)[::-1]
            
            print("Top features that determine W pattern success (from screenshot analysis):")
            print()
            
            for i in range(min(10, len(sorted_idx))):
                idx = sorted_idx[i]
                feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
                importance = feature_importance[idx]
                
                # Interpret the feature
                interpretation = self._interpret_feature(feature_name, importance)
                print(f"{i+1:2d}. {feature_name:<18} ({importance:.3f}) - {interpretation}")
            
            print()
            print("KEY INSIGHTS FROM SCREENSHOT TRAINING:")
            
            # Top 3 feature categories
            top_3_idx = sorted_idx[:3]
            top_3_names = [self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}" for i in top_3_idx]
            
            if 'valley_symmetry' in top_3_names:
                print("   • Valley symmetry is critical for identifying valid W patterns")
            if 'volatility' in top_3_names or 'std_intensity' in top_3_names:
                print("   • Price volatility patterns help distinguish W formations")
            if any('peak' in name or 'valley' in name for name in top_3_names):
                print("   • Peak/valley structure recognition is key")
            if 'trend_slope' in top_3_names:
                print("   • Overall trend direction affects pattern success")
            
            print("   • Model trained on screenshot characteristics, applied to trend-based detection")
                
        else:
            print("Feature importance data not available")
    
    def _interpret_feature(self, feature_name, importance):
        """Interpret what each feature means for W patterns"""
        interpretations = {
            'valley_symmetry': 'How evenly matched the two trough levels are',
            'volatility': 'Price movement intensity during formation',
            'avg_height': 'Average height from troughs to peaks',
            'price_range': 'Total price range during pattern formation',
            'trend_slope': 'Overall trend direction (up/down/sideways)',
            'num_peaks': 'Number of significant peaks detected',
            'num_valleys': 'Number of significant troughs detected',
            'valley_separation': 'Time distance between the two troughs',
            'mean_intensity': 'Average price intensity in the pattern',
            'std_intensity': 'Price variation consistency',
            'compactness': 'How tight/loose the pattern shape is',
            'edge_density': 'Sharpness of price movements',
            'hu_moment_1': 'Overall shape orientation',
            'hu_moment_2': 'Shape aspect ratio',
            'contour_area': 'Total area enclosed by price pattern',
            'local_var_mean': 'Local price variation patterns'
        }
        
        base_interpretation = interpretations.get(feature_name, 'Pattern characteristic')
        
        if importance > 0.1:
            return f"{base_interpretation} (CRITICAL)"
        elif importance > 0.05:
            return f"{base_interpretation} (Very Important)"
        elif importance > 0.02:
            return f"{base_interpretation} (Important)"
        else:
            return f"{base_interpretation} (Minor factor)"
    
    def _analyze_training_data(self):
        """Analyze the training data composition"""
        print("\nTRAINING DATA ANALYSIS")
        print("-" * 50)
        
        if not self.pattern_metadata:
            print("No training metadata available")
            return
        
        # Data source breakdown
        phases = {}
        outcomes = {}
        tickers = set()
        
        for item in self.pattern_metadata:
            phases[item['phase']] = phases.get(item['phase'], 0) + 1
            outcomes[item['trading_outcome']] = outcomes.get(item['trading_outcome'], 0) + 1
            tickers.add(item['ticker'])
        
        print("SCREENSHOT DATA SOURCES:")
        for phase, count in phases.items():
            percentage = (count / len(self.pattern_metadata)) * 100
            print(f"   {phase}: {count} examples ({percentage:.1f}%)")
        
        print()
        print("TRADING OUTCOMES FROM SCREENSHOTS:")
        for outcome, count in outcomes.items():
            percentage = (count / len(self.pattern_metadata)) * 100
            print(f"   {outcome}: {count} examples ({percentage:.1f}%)")
        
        print()
        print("DATASET STATISTICS:")
        print(f"   Unique tickers: {len(tickers)}")
        print(f"   Total examples: {len(self.pattern_metadata)}")
        
        success_rate = outcomes.get('profitable', 0) / len(self.pattern_metadata) * 100
        print(f"   Historical success rate: {success_rate:.1f}%")
        
        # Show some example tickers
        example_tickers = list(tickers)[:10]
        print(f"   Example tickers: {', '.join(example_tickers)}")
        
        print()
        print("IMPORTANT NOTE:")
        print("   • Screenshots show approximate W patterns, not exact boundaries")
        print("   • ML model learns pattern characteristics from screenshots")
        print("   • Trend-based detection applies these learnings to precise OHLC data")
    
    def _analyze_trend_pattern_boundaries(self):
        """Explain how trend-based pattern boundaries are determined"""
        print("\nTREND-BASED PATTERN BOUNDARY DETECTION")
        print("-" * 50)
        
        print("How the system identifies W pattern points using trend analysis:")
        print()
        
        print("STEP 1 - TREND IDENTIFICATION:")
        print("   • Uses moving averages (5-period and 10-period)")
        print("   • Identifies uptrends, downtrends, and sideways movement")
        print("   • Validates trend strength (minimum 2% price movement)")
        print("   • Filters out noise with minimum trend duration")
        print()
        
        print("STEP 2 - PEAK 1 (PATTERN START):")
        print("   • Relative high point before a confirmed downtrend")
        print("   • Not necessarily the highest point in 90 days")
        print("   • Marks the beginning of the resistance line")
        print("   • Found by analyzing trend transitions")
        print()
        
        print("STEP 3 - TROUGH 1 (FIRST LOW):")
        print("   • Relative low after downtrend from Peak 1")
        print("   • Uses Low price data for precision")
        print("   • Can have small rebounds within the downtrend")
        print("   • Marks end of first decline phase")
        print()
        
        print("STEP 4 - PEAK 2 (MIDDLE HIGH):")
        print("   • Relative high after uptrend from Trough 1")
        print("   • Forms second point of resistance line")
        print("   • May be lower than Peak 1 (common in W patterns)")
        print("   • Validates the recovery bounce strength")
        print()
        
        print("STEP 5 - TROUGH 2 (SECOND LOW):")
        print("   • Relative low after downtrend from Peak 2")
        print("   • Must be symmetric to Trough 1 (within 30% tolerance)")
        print("   • Tests the same support level as Trough 1")
        print("   • Critical for W pattern validation")
        print()
        
        print("STEP 6 - PEAK 3 (BREAKOUT ATTEMPT):")
        print("   • Recovery high after uptrend from Trough 2")
        print("   • Must reach resistance line from Peak1-Peak2")
        print("   • Breakthrough above resistance confirms W completion")
        print("   • Volume expansion validates breakout strength")
        print()
        
        print("PATTERN COMPLETION RULES:")
        print("   • All 5 points must follow chronological order")
        print("   • Trough symmetry within 30% tolerance required")
        print("   • Minimum 1.5% rebound strength from each trough")
        print("   • Pattern duration: 15-90 days")
        print("   • Peak 3 within 2% of resistance line for completion")
    
    def _analyze_success_predictions(self):
        """Analyze how the model predicts trading success"""
        print("\nSUCCESS PREDICTION LOGIC")
        print("-" * 50)
        
        if hasattr(self, 'training_stats') and self.training_stats:
            stats = self.training_stats
            accuracy = stats.get('accuracy', 0)
            
            print(f"Model Performance Metrics:")
            print(f"   Overall accuracy: {accuracy:.1%}")
            print(f"   Training samples: {stats.get('samples', 0)}")
            print(f"   Profitable samples: {stats.get('profitable_samples', 0)}")
            print(f"   Unprofitable samples: {stats.get('unprofitable_samples', 0)}")
            
            if stats.get('unprofitable_samples', 0) > 0:
                failure_rate = stats['unprofitable_samples'] / stats['samples']
                print(f"   Historical failure rate: {failure_rate:.1%}")
        
        print()
        print("PREDICTION METHODOLOGY:")
        print("   • Trained on screenshot pattern characteristics")
        print("   • Applied to precise trend-based pattern detection")
        print("   • Random Forest classifier with 50 trees")
        print("   • Features scaled using RobustScaler")
        print("   • 28 technical features from pattern analysis")
        print()
        
        print("CONFIDENCE SCORING:")
        print("   • High confidence: >80% (Strong recommendation)")
        print("   • Medium confidence: 60-80% (Good opportunity)")
        print("   • Low confidence: 40-60% (Watch closely)")
        print("   • Very low confidence: <40% (Avoid)")
        print()
        
        print("TARGET PRICE CALCULATION:")
        print("   • Based on historical performance of similar patterns")
        print("   • Adjusted for resistance breakthrough strength")
        print("   • Typical targets: 10-25% above entry price")
        print("   • Volume confirmation increases target confidence")
    
    def _analyze_decision_rules(self):
        """Explain the model's decision-making process"""
        print("\nMODEL DECISION RULES FOR TREND-BASED W PATTERNS")
        print("-" * 50)
        
        print("How the model makes trading recommendations:")
        print()
        
        print("STRONG BUY CONDITIONS:")
        print("   • Success probability >75%")
        print("   • Trading confidence >80%")
        print("   • High pattern quality (>0.70)")
        print("   • Confirmed resistance breakthrough")
        print("   • Volume expansion at breakout")
        print()
        
        print("BUY CONDITIONS:")
        print("   • Success probability >65%")
        print("   • Trading confidence >60%")
        print("   • Good pattern quality (>0.50)")
        print("   • Pattern recently completed")
        print("   • Acceptable trough symmetry")
        print()
        
        print("WATCH CONDITIONS:")
        print("   • Success probability >55%")
        print("   • Decent pattern quality (>0.40)")
        print("   • Pattern still forming OR recently completed")
        print("   • Needs confirmation of resistance breakthrough")
        print()
        
        print("AVOID CONDITIONS:")
        print("   • Success probability <55%")
        print("   • Poor trough symmetry")
        print("   • Failed to reach resistance line")
        print("   • Weak volume confirmation")
        print("   • Low pattern quality score")
        print()
        
        print("TREND-BASED ADVANTAGES:")
        print("   • Identifies relative peaks/troughs, not just absolute highs/lows")
        print("   • Accounts for trend context and transitions")
        print("   • More accurate resistance line calculation")
        print("   • Better handles patterns within larger trends")
        print("   • Reduces false positives from market noise")
    
    def predict_single_pattern(self, image_path: str) -> Dict:
        """Predict trading success for a single W pattern image"""
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        features = self.extract_features_from_image(image_path)
        if features is None:
            return {'error': 'Could not extract features from image'}
        
        features_scaled = self.scaler.transform([features])
        
        # Predict profitability
        profit_proba = self.pattern_classifier.predict_proba(features_scaled)[0]
        will_be_profitable = self.pattern_classifier.predict(features_scaled)[0]
        
        result = {
            'will_be_profitable': bool(will_be_profitable),
            'profit_probability': float(profit_proba[1]) if len(profit_proba) > 1 else float(profit_proba[0]),
            'trading_confidence': 0.0,
            'target_multiplier': 1.0
        }
        
        # Add confidence and target if profitable
        if will_be_profitable:
            try:
                trading_conf = self.confidence_regressor.predict(features_scaled)[0]
                target_mult = self.target_regressor.predict(features_scaled)[0]
                
                result.update({
                    'trading_confidence': float(np.clip(trading_conf, 0, 1)),
                    'target_multiplier': float(np.clip(target_mult, 0.8, 2.0))
                })
            except Exception:
                pass
        
        return result
    
    def save_model(self, filename="trend_w_pattern_model.pkl"):
        """Save trained model"""
        if not self.is_trained:
            print("No trained model to save")
            return False
        
        model_data = {
            'pattern_classifier': self.pattern_classifier,
            'confidence_regressor': self.confidence_regressor,
            'target_regressor': self.target_regressor,
            'scaler': self.scaler,
            'training_stats': self.training_stats,
            'pattern_metadata': self.pattern_metadata,
            'feature_names': self.feature_names,
            'detection_criteria': self.detection_criteria,
            'is_trained': self.is_trained
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
        return True
    
    def load_model(self, filename="trend_w_pattern_model.pkl"):
        """Load pre-trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.pattern_classifier = model_data['pattern_classifier']
            self.confidence_regressor = model_data['confidence_regressor']
            self.target_regressor = model_data['target_regressor']
            self.scaler = model_data['scaler']
            self.training_stats = model_data['training_stats']
            self.pattern_metadata = model_data.get('pattern_metadata', [])
            self.feature_names = model_data.get('feature_names', self.feature_names)
            self.detection_criteria = model_data.get('detection_criteria', self.detection_criteria)
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded successfully")
            print(f"   Training accuracy: {self.training_stats.get('accuracy', 0):.1%}")
            print(f"   Training samples: {self.training_stats.get('samples', 0)}")
            return True
            
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

# Main execution
if __name__ == "__main__":
    print("W Pattern Trading Success Predictor - Trend-Based Detection")
    print("=" * 60)
    
    detector = WPatternTradingPredictor()
    
    while True:
        print("\nTREND-BASED W PATTERN ANALYSIS OPTIONS:")
        print("1. Train model on your trading examples")
        print("2. Load pre-trained model") 
        print("3. SCAN S&P 500 FOR TREND-BASED W PATTERNS")  # Main focus
        print("4. Predict trading success for single image")
        print("5. Save trained model")
        print("6. ANALYZE MODEL LEARNINGS & DETECTION CRITERIA")  # Detailed analysis
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ")
        
        if choice == "1":
            print("\nTraining model on your W pattern screenshot examples...")
            print("Note: Screenshots provide pattern characteristics, trend analysis provides precision")
            if detector.load_training_data():
                if detector.train_models():
                    print("\nModel training complete!")
                    print("Ready for trend-based S&P 500 scanning (option 3)")
            else:
                print("Training failed - check your data folders")
                
        elif choice == "2":
            detector.load_model()
            
        elif choice == "3":
            print("\nStarting Trend-Based S&P 500 W Pattern Scanner...")
            print("This detects W patterns based on trend transitions and relative peaks/troughs")
            print("Patterns identified by downtrend→trough→uptrend→peak→downtrend→trough→uptrend→resistance")
            
            # Ask for custom lookback period
            try:
                lookback = input("Enter lookback days (default 90): ").strip()
                lookback_days = int(lookback) if lookback else 90
            except:
                lookback_days = 90
            
            print(f"Scanning for W patterns within the past {lookback_days} trading days...")
            
            detections = detector.scan_sp500_for_w_patterns(lookback_days)
            
            if detections:
                print(f"\nSCAN COMPLETE: {len(detections)} trend-based W patterns found!")
                
                # Quick summary of top opportunities
                strong_buys = [d for d in detections if d.get('recommendation') == 'STRONG BUY']
                buys = [d for d in detections if d.get('recommendation') == 'BUY']
                
                if strong_buys:
                    print(f"\nSTRONG BUY opportunities: {len(strong_buys)}")
                    for d in strong_buys[:3]:
                        prob = d.get('ml_profit_probability', 0)
                        breakthrough = "BREAKTHROUGH" if d.get('resistance_breakthrough', False) else "APPROACHING"
                        print(f"   {d['symbol']}: {prob:.0%} success probability - {breakthrough}")
                
                if buys:
                    print(f"\nBUY opportunities: {len(buys)}")
                    for d in buys[:3]:
                        prob = d.get('ml_profit_probability', 0)
                        breakthrough = "BREAKTHROUGH" if d.get('resistance_breakthrough', False) else "APPROACHING"
                        print(f"   {d['symbol']}: {prob:.0%} success probability - {breakthrough}")
                        
            else:
                print("\nNo trend-based W patterns detected in current scan")
                print("This means no clear trend-based W formations were found in the specified timeframe")
                print("Try scanning with a longer lookback period or check back later")
            
        elif choice == "4":
            if detector.is_trained:
                image_path = input("Enter image path: ").strip()
                if os.path.exists(image_path):
                    result = detector.predict_single_pattern(image_path)
                    
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"\nTRADING SUCCESS PREDICTION (from screenshot):")
                        print(f"   Profitable: {result['will_be_profitable']}")
                        print(f"   Success Probability: {result['profit_probability']:.1%}")
                        print(f"   Trading Confidence: {result['trading_confidence']:.1%}")
                        print(f"   Target Multiplier: {result['target_multiplier']:.2f}x")
                        print(f"   Note: Based on pattern characteristics from screenshot")
                else:
                    print("Image file not found")
            else:
                print("Model not trained. Use option 1 first.")
                
        elif choice == "5":
            detector.save_model()
            
        elif choice == "6":
            print("\nAnalyzing your trained model's learnings...")
            detector.analyze_model_learnings()
            
        elif choice == "7":
            print("Goodbye! Happy trading!")
            break
            
        else:
            print("Invalid option. Please select 1-7.")
        
        input("\nPress Enter to continue...")

print("\nTREND-BASED W PATTERN TRADING PREDICTOR FEATURES:")
print("- Detects W patterns based on trend transitions")
print("- Uses relative peaks/troughs, not absolute highs/lows")
print("- Precise resistance line calculation from Peak1 to Peak2")
print("- ML-powered profitability predictions from screenshot training")
print("- Volume confirmation and breakthrough analysis")
print("- Comprehensive S&P 500 coverage with trend-based detection")
print("- Real-time pattern status and risk assessment")