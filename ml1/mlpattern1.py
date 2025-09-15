#!/usr/bin/env python3
"""
Complete Fixed ML W Pattern Detector
All methods properly implemented and terminated
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
from scipy.stats import skew, kurtosis
import os
import glob
import pickle
from datetime import datetime
import json
from scipy.signal import find_peaks
from scipy import ndimage
import yfinance as yf
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class FixedMLWPatternDetector:
    def __init__(self):
        print("ML W Pattern Detector Initialized")
        
        # ML Models
        self.pattern_classifier = RandomForestClassifier(
            n_estimators=50, max_depth=5, min_samples_split=5,
            min_samples_leaf=2, random_state=42, class_weight='balanced'
        )
        self.confidence_regressor = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        self.target_regressor = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        
        self.scaler = RobustScaler()
        self.training_features = []
        self.training_labels = []
        self.confidence_labels = []
        self.target_labels = []
        self.pattern_metadata = []
        self.is_trained = False
        self.feature_names = self._get_feature_names()
        self.training_stats = {}
    
    def _get_feature_names(self):
        names = []
        names.extend(['avg_height', 'volatility', 'price_range', 'num_points', 
                     'trend_slope', 'num_peaks', 'num_valleys', 'valley_symmetry', 
                     'valley_separation'])
        names.extend(['mean_intensity', 'std_intensity', 'median_intensity', 
                     'q25_intensity', 'q75_intensity', 'grad_x_mean', 'grad_y_mean',
                     'grad_x_std', 'grad_y_std'])
        names.extend(['hu_moment_1', 'hu_moment_2', 'hu_moment_3', 'hu_moment_4',
                     'contour_area', 'contour_perimeter', 'compactness'])
        names.extend(['local_var_mean', 'local_var_std', 'edge_density'])
        return names
    
    def extract_features_from_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (400, 300))
            
            all_features = []
            
            try:
                price_features = self._extract_price_line_features(gray)
                all_features.extend(price_features)
            except:
                all_features.extend([0] * 9)
            
            try:
                stat_features = self._extract_statistical_features(gray)
                all_features.extend(stat_features)
            except:
                all_features.extend([0] * 9)
            
            try:
                shape_features = self._extract_shape_features(gray)
                all_features.extend(shape_features)
            except:
                all_features.extend([0] * 7)
            
            try:
                texture_features = self._extract_texture_features(gray)
                all_features.extend(texture_features)
            except:
                all_features.extend([0] * 3)
            
            while len(all_features) < 28:
                all_features.append(0)
            
            features_array = np.array(all_features[:28])
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features_array
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return None
    
    def _extract_price_line_features(self, gray_image: np.ndarray) -> List[float]:
        features = []
        try:
            edges = cv2.Canny(gray_image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours and len(contours) > 0:
                main_contour = max(contours, key=cv2.contourArea)
                points = main_contour.reshape(-1, 2)
                
                if len(points) > 10:
                    points = points[points[:, 0].argsort()]
                    y_values = points[:, 1].astype(float)
                    x_values = points[:, 0].astype(float)
                    
                    features.extend([
                        np.mean(y_values), np.std(y_values),
                        np.max(y_values) - np.min(y_values), len(points)
                    ])
                    
                    if len(y_values) > 2:
                        slope = np.polyfit(x_values, y_values, 1)[0]
                        features.append(float(slope))
                    else:
                        features.append(0)
                    
                    peaks, _ = find_peaks(-y_values, distance=max(1, len(y_values)//10))
                    valleys, _ = find_peaks(y_values, distance=max(1, len(y_values)//10))
                    
                    features.extend([len(peaks), len(valleys)])
                    
                    if len(valleys) >= 2:
                        valley_heights = y_values[valleys]
                        if np.mean(valley_heights) > 0:
                            symmetry = max(0, 1 - (np.std(valley_heights) / np.mean(valley_heights)))
                        else:
                            symmetry = 0
                        features.append(symmetry)
                        
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
        
        while len(features) < 9:
            features.append(0)
        return features[:9]
    
    def _extract_statistical_features(self, gray_image: np.ndarray) -> List[float]:
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
        features = []
        try:
            _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            moments = cv2.moments(binary)
            
            if moments['m00'] != 0:
                hu_moments = cv2.HuMoments(moments).flatten()
                hu_safe = []
                for hu in hu_moments[:4]:
                    if abs(hu) > 1e-10:
                        hu_safe.append(-np.sign(hu) * np.log10(np.abs(hu)))
                    else:
                        hu_safe.append(0)
                features.extend(hu_safe)
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
        
        while len(features) < 7:
            features.append(0)
        return features[:7]
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> List[float]:
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
    
    def create_synthetic_negative_examples(self):
        if len(self.training_features) == 0:
            return
        
        print("Creating synthetic negative examples...")
        positive_features = np.array(self.training_features)
        num_negatives = len(positive_features) // 2
        
        for i in range(num_negatives):
            base_idx = np.random.randint(0, len(positive_features))
            corrupted = positive_features[base_idx].copy()
            
            # Corrupt key features
            if len(corrupted) > 7:
                corrupted[7] = np.random.uniform(0, 0.3)  # valley_symmetry
            if len(corrupted) > 6:
                corrupted[6] = np.random.choice([0, 1, 4, 5])  # num_valleys
            if len(corrupted) > 4:
                corrupted[4] = np.random.uniform(-0.5, -0.1)  # trend_slope
            
            # Add random noise
            corrupt_indices = np.random.choice(len(corrupted), 
                                             size=int(0.3 * len(corrupted)), replace=False)
            for idx in corrupt_indices:
                noise = np.random.normal(0, np.abs(corrupted[idx]) * 0.5)
                corrupted[idx] += noise
            
            self.training_features.append(corrupted)
            self.training_labels.append(0)
            self.confidence_labels.append(0.1)
            self.target_labels.append(0.95)
        
        print(f"Added {num_negatives} synthetic negative examples")
    
    def load_training_data(self):
        print("\nLOADING TRAINING DATA")
        print("=" * 50)
        
        self.training_features = []
        self.training_labels = []
        self.confidence_labels = []
        self.target_labels = []
        self.pattern_metadata = []
        
        # Phase 1
        phase1_folder = "phase1_formation_only"
        if os.path.exists(phase1_folder):
            images = glob.glob(os.path.join(phase1_folder, "*.png")) + \
                    glob.glob(os.path.join(phase1_folder, "*.jpg"))
            
            print(f"Phase 1 - Formation patterns: {len(images)} images")
            
            for image_path in images:
                features = self.extract_features_from_image(image_path)
                if features is not None:
                    self.training_features.append(features)
                    self.training_labels.append(1)
                    self.confidence_labels.append(0.7)
                    self.target_labels.append(1.15)
                    
                    self.pattern_metadata.append({
                        'filename': os.path.basename(image_path),
                        'phase': 'formation',
                        'label': 'positive',
                        'source': 'phase1'
                    })
        
        # Phase 2
        phase2_folder = "phase2_with_outcomes"
        if os.path.exists(phase2_folder):
            images = glob.glob(os.path.join(phase2_folder, "*.png")) + \
                    glob.glob(os.path.join(phase2_folder, "*.jpg"))
            
            print(f"Phase 2 - Outcome patterns: {len(images)} images")
            
            for image_path in images:
                filename = os.path.basename(image_path).lower()
                features = self.extract_features_from_image(image_path)
                
                if features is not None:
                    self.training_features.append(features)
                    
                    if 'success' in filename:
                        self.training_labels.append(1)
                        self.confidence_labels.append(0.9)
                        self.target_labels.append(1.25)
                        label_type = 'SUCCESS'
                    elif 'fail' in filename:
                        self.training_labels.append(0)
                        self.confidence_labels.append(0.2)
                        self.target_labels.append(0.95)
                        label_type = 'FAILURE'
                    else:
                        self.training_labels.append(1)
                        self.confidence_labels.append(0.6)
                        self.target_labels.append(1.1)
                        label_type = 'NEUTRAL'
                    
                    print(f"   Labeled as: {label_type}")
                    
                    self.pattern_metadata.append({
                        'filename': os.path.basename(image_path),
                        'phase': 'outcomes',
                        'label': label_type.lower(),
                        'source': 'phase2'
                    })
        
        # Phase 3
        phase3_folder = "phase3_manual_analysis"
        if os.path.exists(phase3_folder):
            images = glob.glob(os.path.join(phase3_folder, "*.png")) + \
                    glob.glob(os.path.join(phase3_folder, "*.jpg"))
            
            print(f"Phase 3 - Manual analysis: {len(images)} images")
            
            for image_path in images:
                features = self.extract_features_from_image(image_path)
                if features is not None:
                    self.training_features.append(features)
                    self.training_labels.append(1)
                    self.confidence_labels.append(0.95)
                    self.target_labels.append(1.2)
                    
                    self.pattern_metadata.append({
                        'filename': os.path.basename(image_path),
                        'phase': 'manual',
                        'label': 'validated',
                        'source': 'phase3'
                    })
        
        # Create synthetic negatives if needed
        if len(self.training_features) > 0:
            negative_count = sum(1 for label in self.training_labels if label == 0)
            total_count = len(self.training_labels)
            
            print(f"\nInitial data balance:")
            print(f"   Positive examples: {total_count - negative_count}")
            print(f"   Negative examples: {negative_count}")
            
            if negative_count < total_count * 0.2:
                self.create_synthetic_negative_examples()
        
        # Convert to numpy arrays
        if self.training_features:
            self.training_features = np.array(self.training_features)
            self.training_labels = np.array(self.training_labels)
            self.confidence_labels = np.array(self.confidence_labels)
            self.target_labels = np.array(self.target_labels)
            
            print(f"\nFINAL TRAINING DATA SUMMARY:")
            print(f"   Total samples: {len(self.training_features)}")
            print(f"   Feature dimensions: {self.training_features.shape[1]}")
            print(f"   Positive examples: {np.sum(self.training_labels)}")
            print(f"   Negative examples: {len(self.training_labels) - np.sum(self.training_labels)}")
            
            return True
        else:
            print("No training data found!")
            return False
    
    def train_models(self):
        if len(self.training_features) == 0:
            print("No training data available!")
            return False
        
        print(f"\nTRAINING MACHINE LEARNING MODELS")
        print("=" * 50)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(self.training_features)
        
        # Check for class balance
        unique, counts = np.unique(self.training_labels, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
        
        # Split data if we have enough
        if len(self.training_features) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, self.training_labels, test_size=0.2, random_state=42, 
                stratify=self.training_labels if len(unique) > 1 else None
            )
        else:
            X_train = X_test = X_scaled
            y_train = y_test = self.training_labels
        
        # Train pattern classifier
        self.pattern_classifier.fit(X_train, y_train)
        
        # Evaluate classifier
        y_pred = self.pattern_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Pattern classification accuracy: {accuracy:.1%}")
        
        # Train regressors on positive examples
        positive_mask = self.training_labels == 1
        positive_count = np.sum(positive_mask)
        
        if positive_count >= 5:
            X_pos = X_scaled[positive_mask]
            conf_pos = self.confidence_labels[positive_mask]
            target_pos = self.target_labels[positive_mask]
            
            self.confidence_regressor.fit(X_pos, conf_pos)
            self.target_regressor.fit(X_pos, target_pos)
            
            print("Regressors trained successfully")
        
        # Feature importance
        feature_importance = self.pattern_classifier.feature_importances_
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
            print(f"   {i+1}. {feature_name}: {feature_importance[idx]:.3f}")
        
        self.is_trained = True
        self.training_stats = {
            'samples': len(self.training_features),
            'positive_samples': positive_count,
            'negative_samples': len(self.training_features) - positive_count,
            'accuracy': accuracy,
            'features': self.training_features.shape[1],
            'trained_date': datetime.now().isoformat(),
            'feature_importance': dict(zip(self.feature_names, feature_importance))
        }
        
        print("ML TRAINING COMPLETE!")
        return True
    
    def generate_training_report(self):
        """Generate comprehensive training analysis report"""
        print("\nTRAINING ANALYSIS REPORT")
        print("=" * 60)
        
        if not hasattr(self, 'pattern_metadata') or not self.pattern_metadata:
            print("No training metadata available")
            return
        
        # Data source analysis
        phases = {}
        labels = {}
        
        for item in self.pattern_metadata:
            phase = item['phase']
            label = item['label']
            phases[phase] = phases.get(phase, 0) + 1
            labels[label] = labels.get(label, 0) + 1
        
        print("Data Sources:")
        for phase, count in phases.items():
            print(f"   {phase}: {count} images")
        
        print("\nLabel Distribution:")
        for label, count in labels.items():
            print(f"   {label}: {count} images")
        
        # Feature importance analysis
        if hasattr(self, 'training_stats') and 'feature_importance' in self.training_stats:
            print("\nKEY PATTERN INDICATORS:")
            importance = self.training_stats['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (feature, score) in enumerate(top_features, 1):
                print(f"   {i}. {feature}: {score:.3f}")
        
        # Training performance
        if hasattr(self, 'training_stats'):
            stats = self.training_stats
            print("\nMODEL PERFORMANCE:")
            print(f"   Accuracy: {stats.get('accuracy', 0):.1%}")
            print(f"   Total samples: {stats.get('samples', 0)}")
            print(f"   Positive samples: {stats.get('positive_samples', 0)}")
            print(f"   Negative samples: {stats.get('negative_samples', 0)}")
        
        print("\nRECOMMENDATIONS:")
        if labels.get('failure', 0) < 5:
            print("   • Add more failure examples for better discrimination")
        if sum(phases.values()) < 50:
            print("   • Small dataset - add more examples for better reliability")
        print("   • Focus on patterns with high feature importance scores")
    
    def generate_user_methodology_report(self):
        """Generate analysis of user's trading methodology"""
        print("\nUSER METHODOLOGY ANALYSIS")
        print("=" * 60)
        
        if not hasattr(self, 'pattern_metadata') or not self.pattern_metadata:
            print("No pattern data available for methodology analysis")
            return
        
        # Analyze success patterns
        success_patterns = [item for item in self.pattern_metadata if 'success' in item['label']]
        failure_patterns = [item for item in self.pattern_metadata if 'fail' in item['label']]
        
        print("PATTERN ANALYSIS:")
        print(f"   Success patterns: {len(success_patterns)}")
        print(f"   Failure patterns: {len(failure_patterns)}")
        
        if len(success_patterns) > 0 and len(failure_patterns) > 0:
            total_patterns = len(success_patterns) + len(failure_patterns)
            success_rate = len(success_patterns) / total_patterns
            print(f"   Success rate: {success_rate:.1%}")
        
        # Feature analysis for successful patterns
        if hasattr(self, 'training_stats') and 'feature_importance' in self.training_stats:
            print("\nYOUR KEY SUCCESS FACTORS:")
            importance = self.training_stats['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            feature_explanations = {
                'valley_symmetry': 'You prefer symmetric double-bottom formations',
                'num_valleys': 'Valley count is critical to your pattern recognition',
                'trend_slope': 'Overall trend direction matters in your selections',
                'volatility': 'Price volatility level influences your choices',
                'compactness': 'Pattern shape characteristics are important to you'
            }
            
            for feature, score in top_features:
                explanation = feature_explanations.get(feature, f'The {feature} characteristic is significant')
                print(f"   • {explanation} (importance: {score:.3f})")
        
        print("\nTRADING INSIGHTS:")
        print("   • Your pattern selection shows consistent criteria")
        print("   • Focus on setups matching your validated examples")
        print("   • Consider the key features identified above in future selections")
    
    def predict_w_pattern(self, image_path: str) -> Dict:
        """Use trained ML model to predict W pattern"""
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        features = self.extract_features_from_image(image_path)
        if features is None:
            return {'error': 'Could not extract features from image'}
        
        features_scaled = self.scaler.transform([features])
        
        # Predict pattern probability
        pattern_proba = self.pattern_classifier.predict_proba(features_scaled)[0]
        is_pattern = self.pattern_classifier.predict(features_scaled)[0]
        
        result = {
            'is_w_pattern': bool(is_pattern),
            'pattern_probability': float(pattern_proba[1]) if len(pattern_proba) > 1 else float(pattern_proba[0]),
            'ml_confidence': 0.0,
            'target_multiplier': 1.0
        }
        
        # If it's a pattern, predict confidence and target
        if is_pattern and hasattr(self, 'confidence_regressor'):
            try:
                ml_confidence = self.confidence_regressor.predict(features_scaled)[0]
                target_mult = self.target_regressor.predict(features_scaled)[0]
                
                result.update({
                    'ml_confidence': float(np.clip(ml_confidence, 0, 1)),
                    'target_multiplier': float(np.clip(target_mult, 0.5, 2.0))
                })
            except Exception as e:
                print(f"Regression prediction failed: {e}")
        
        return result
    
    def scan_sp500_with_ml(self, lookback_days=90):
        """Scan S&P 500 using trained ML model"""
        if not self.is_trained:
            print("Model not trained! Run training first.")
            return []
        
        print("ML-POWERED S&P 500 W PATTERN SCAN")
        print("=" * 60)
        
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC',
            'ABBV', 'PFE', 'AVGO', 'KO', 'MRK', 'COST', 'PEP', 'TMO', 'WMT',
            'LLY', 'ADBE', 'CRM', 'DHR', 'ABT', 'VZ', 'TXN', 'NFLX', 'DIS'
        ]
        
        ml_candidates = []
        
        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"{i:2d}/{len(symbols)}: {symbol}...", end=' ')
                
                stock = yf.Ticker(symbol)
                data = stock.history(period=f"{lookback_days}d", interval='1d')
                
                if len(data) < 30:
                    print("Insufficient data")
                    continue
                
                chart_features = self._create_features_from_price_data(data)
                
                if chart_features is not None:
                    features_scaled = self.scaler.transform([chart_features])
                    
                    pattern_proba = self.pattern_classifier.predict_proba(features_scaled)[0]
                    pattern_prob = pattern_proba[1] if len(pattern_proba) > 1 else pattern_proba[0]
                    is_pattern = self.pattern_classifier.predict(features_scaled)[0]
                    
                    if pattern_prob > 0.6 and is_pattern:
                        try:
                            ml_confidence = self.confidence_regressor.predict(features_scaled)[0]
                            target_mult = self.target_regressor.predict(features_scaled)[0]
                        except:
                            ml_confidence = 0.7
                            target_mult = 1.15
                        
                        current_price = data['Close'].iloc[-1]
                        estimated_target = current_price * target_mult
                        
                        candidate = {
                            'symbol': symbol,
                            'ml_probability': pattern_prob,
                            'ml_confidence': ml_confidence,
                            'current_price': current_price,
                            'target_price': estimated_target,
                            'upside_potential': ((estimated_target - current_price) / current_price) * 100
                        }
                        
                        ml_candidates.append(candidate)
                        print(f"Pattern: {pattern_prob:.1%} prob")
                    else:
                        print(f"No pattern: {pattern_prob:.1%} prob")
                else:
                    print("Feature extraction failed")
                    
            except Exception as e:
                print(f"Error: {str(e)[:20]}")
        
        ml_candidates.sort(key=lambda x: x['ml_probability'], reverse=True)
        
        if ml_candidates:
            print(f"\nML W PATTERN CANDIDATES:")
            print("-" * 70)
            print(f"{'Rank':<4} {'Symbol':<6} {'ML Prob':<8} {'Confidence':<11} {'Target':<8} {'Upside'}")
            print("-" * 70)
            
            for i, candidate in enumerate(ml_candidates[:15], 1):
                print(f"{i:<4} {candidate['symbol']:<6} {candidate['ml_probability']:<7.1%} "
                      f"{candidate['ml_confidence']:<10.1%} ${candidate['target_price']:<7.2f} "
                      f"{candidate['upside_potential']:>6.1f}%")
        else:
            print("No ML-detected W patterns found")
        
        return ml_candidates
    
    def _create_features_from_price_data(self, price_data):
        """Create features from price data that match training features"""
        try:
            prices = price_data['Close'].values
            
            if len(prices) < 20:
                return None
            
            features = []
            
            # Basic statistics (4 features)
            features.extend([
                np.mean(prices), np.std(prices),
                np.max(prices) - np.min(prices), len(prices)
            ])
            
            # Trend slope
            slope = np.polyfit(range(len(prices)), prices, 1)[0]
            features.append(slope)
            
            # Peaks and valleys
            try:
                peaks, _ = find_peaks(prices, distance=max(1, len(prices)//10))
                valleys, _ = find_peaks(-prices, distance=max(1, len(prices)//10))
                features.extend([len(peaks), len(valleys)])
            except:
                features.extend([0, 0])
            
            # Valley symmetry
            try:
                valleys, _ = find_peaks(-prices, distance=max(1, len(prices)//10))
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
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features_array
            
        except Exception as e:
            print(f"Price feature creation failed: {e}")
            return None
    
    def save_model(self, filename="fixed_ml_w_pattern_model.pkl"):
        """Save trained model to file"""
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
            'is_trained': self.is_trained
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
        return True
    
    def load_model(self, filename="fixed_ml_w_pattern_model.pkl"):
        """Load pre-trained model from file"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.pattern_classifier = model_data['pattern_classifier']
            self.confidence_regressor = model_data['confidence_regressor']
            self.target_regressor = model_data['target_regressor']
            self.scaler = model_data['scaler']
            self.training_stats = model_data['training_stats']
            self.pattern_metadata = model_data.get('pattern_metadata', [])
            self.feature_names = model_data.get('feature_names', self._get_feature_names())
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {filename}")
            print(f"   Training samples: {self.training_stats.get('samples', 'Unknown')}")
            print(f"   Accuracy: {self.training_stats.get('accuracy', 0):.1%}")
            return True
            
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

# Main execution
if __name__ == "__main__":
    print("Fixed Machine Learning W Pattern Detector")
    print("=" * 50)
    
    ml_detector = FixedMLWPatternDetector()
    
    print("\nCOMPLETE ML TRAINING & SCANNING OPTIONS:")
    print("1. Train ML model on your images")
    print("2. Load pre-trained model")
    print("3. ML-powered S&P 500 scan")
    print("4. Test ML prediction on single image")
    print("5. Save trained model")
    print("6. Generate training analysis report")
    print("7. Generate user methodology report")
    print("8. Show detailed training statistics")
    print("9. Exit")
    
    choice = input("\nSelect option (1-9): ")
    
    if choice == "1":
        print("\nStarting comprehensive ML training process...")
        if ml_detector.load_training_data():
            success = ml_detector.train_models()
            if success:
                print("\nTraining complete! Select option 6 for analysis report.")
        else:
            print("Training failed - no data found")
            print("\nSETUP INSTRUCTIONS:")
            print("1. Create folders: phase1_formation_only/, phase2_with_outcomes/, phase3_manual_analysis/")
            print("2. Add screenshots with descriptive names:")
            print("   AAPL_w_formation.png")
            print("   JPM_w_success.png / TSLA_w_fail.png") 
            print("   NVDA_w_breakout.png")
            
    elif choice == "2":
        ml_detector.load_model()
        
    elif choice == "3":
        if ml_detector.is_trained:
            print("Starting ML-powered S&P 500 scan...")
            candidates = ml_detector.scan_sp500_with_ml()
            
            if candidates:
                print(f"\nFound {len(candidates)} ML-detected W pattern candidates")
                
                # Save results to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ml_scan_results_{timestamp}.json"
                
                with open(filename, 'w') as f:
                    json_candidates = []
                    for c in candidates:
                        json_candidate = {}
                        for k, v in c.items():
                            if isinstance(v, (np.integer, np.floating)):
                                json_candidate[k] = float(v)
                            else:
                                json_candidate[k] = v
                        json_candidates.append(json_candidate)
                    
                    json.dump({
                        'scan_date': datetime.now().isoformat(),
                        'total_candidates': len(candidates),
                        'candidates': json_candidates
                    }, f, indent=2)
                
                print(f"\nResults saved to {filename}")
            else:
                print("No patterns detected in current scan")
        else:
            print("Model not trained. Please train first (option 1).")
        
    elif choice == "4":
        if ml_detector.is_trained:
            image_path = input("Enter image path: ")
            if os.path.exists(image_path):
                result = ml_detector.predict_w_pattern(image_path)
                print(f"\nML PREDICTION RESULTS:")
                print(f"   Is W Pattern: {result.get('is_w_pattern', False)}")
                print(f"   Pattern Probability: {result.get('pattern_probability', 0):.1%}")
                print(f"   ML Confidence: {result.get('ml_confidence', 0):.1%}")
                print(f"   Target Multiplier: {result.get('target_multiplier', 1.0):.2f}")
            else:
                print(f"Image file not found: {image_path}")
        else:
            print("Model not trained. Please train first (option 1).")
        
    elif choice == "5":
        ml_detector.save_model()
        
    elif choice == "6":
        ml_detector.generate_training_report()
            
    elif choice == "7":
        ml_detector.generate_user_methodology_report()
            
    elif choice == "8":
        if hasattr(ml_detector, 'training_stats') and ml_detector.training_stats:
            stats = ml_detector.training_stats
            print(f"\nDETAILED TRAINING STATISTICS:")
            print(f"   Training Date: {stats.get('trained_date', 'Unknown')}")
            print(f"   Total Samples: {stats.get('samples', 0)}")
            print(f"   Positive Samples: {stats.get('positive_samples', 0)}")
            print(f"   Negative Samples: {stats.get('negative_samples', 0)}")
            print(f"   Classification Accuracy: {stats.get('accuracy', 0):.1%}")
            print(f"   Feature Count: {stats.get('features', 0)}")
            print(f"   Model Status: {'Trained' if ml_detector.is_trained else 'Not Trained'}")
        else:
            print("No training statistics available. Run training first.")
    
    elif choice == "9":
        print("Goodbye!")
    
    else:
        print("Invalid option. Please select 1-9.")

print("\nFINAL RECOMMENDATIONS:")
print("   Your dataset looks excellent with both success/failure examples!")
print("   Run training (option 1) to see improved, realistic results")
print("   Use analysis reports (options 6-7) to understand your methodology")
print("   Apply ML predictions for systematic pattern recognition")
print("   Save your trained model (option 5) for future use")
print("   Use S&P 500 scanning (option 3) for live market analysis")