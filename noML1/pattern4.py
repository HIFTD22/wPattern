#!/usr/bin/env python3
"""
S&P 500 W Pattern Scanner (Rule-based, no ML)
- Scans all S&P 500 stocks for W-shaped patterns on daily timeframe
- Uses Open/Close extremes (peaks=max(Open,Close), troughs=min(Open,Close))
- Enforces tilt: P1 >= P2 (configurable tolerance)
- Target price = breakout_estimate_price + (P2 - trough_midpoint)
- Includes image-based analysis utilities (optional) for "training" folders

Requirements:
  pip install yfinance pandas numpy scipy matplotlib opencv-python (opencv optional for image parts)
"""

import os
import io
import json
import glob
import math
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# 3rd party
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.signal import find_peaks

# Optional: OpenCV (only used for image-based analysis)
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False


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
        self.prediction_model = None  # not used (no ML)
        self.sp500_symbols: List[str] = []

        # Config knobs
        self.lookback_days_default = 90
        self.min_conf_default = 0.60
        self.p1_ge_tol = 0.00  # allow P1 to be X% below P2 (e.g., 0.02 = 2%)

        self.create_folder_structure()
        self.load_sp500_list()
        print("✅ Scanner ready for three-phase training!")

    # ------------------- Files / loading -------------------

    def create_folder_structure(self):
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
        """Load the full S&P 500 list with NO short fallback.
        Prefers local cache sp500.txt; if absent, scrapes Wikipedia and caches it.
        """
        def _norm(s: str) -> str:
            # Yahoo style: BRK.B -> BRK-B
            return s.strip().upper().replace('.', '-')

        cache_path = os.path.join(os.getcwd(), "sp500.txt")

        # 1) Try local cache first
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                syms = [_norm(ln) for ln in f if ln.strip()]
            seen, full = set(), []
            for s in syms:
                if s and s not in seen:
                    seen.add(s)
                    full.append(s)
            if len(full) >= 480:
                self.sp500_symbols = full
                print(f"📊 Loaded {len(full)} S&P 500 symbols from sp500.txt")
                return
            else:
                print(f"⚠️ sp500.txt had only {len(full)} usable symbols; rebuilding from web…")

        # 2) Scrape Wikipedia and cache
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            df = tables[0]
            syms = df["Symbol"].astype(str).tolist()
            syms = [_norm(s) for s in syms if s.strip()]
            seen, full = set(), []
            for s in syms:
                if s and s not in seen:
                    seen.add(s)
                    full.append(s)
            if len(full) < 480:
                raise RuntimeError(f"Wikipedia parse returned only {len(full)} symbols (<480).")

            with open(cache_path, "w") as f:
                f.write("\n".join(full) + "\n")

            self.sp500_symbols = full
            print(f"📊 Loaded {len(full)} S&P 500 symbols from Wikipedia and cached to sp500.txt")
            return

        except Exception as e:
            raise RuntimeError(
                "Could not load full S&P 500 list. Create sp500.txt (one ticker per line) "
                "or ensure internet access to Wikipedia."
            ) from e

    # ------------------- Image-based "training" (no ML) -------------------

    def _find_images_in_folder(self, folder_path: str) -> List[str]:
        exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        files: List[str] = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
        return sorted(files)

    def train_phase1_formation_detection(self, folder_path="phase1_formation_only"):
        print("\n🔍 PHASE 1: W Formation Detection Training")
        print("=" * 60)
        print("📋 Training on clean W formations (no outcomes shown)")

        if not OPENCV_AVAILABLE:
            print("⚠️ OpenCV (cv2) not installed. Phase 1 requires it for image analysis.")
            return []

        image_files = self._find_images_in_folder(folder_path)
        if not image_files:
            print(f"❌ No images found in {folder_path}/")
            print("💡 Add your W formation screenshots to this folder.")
            return []

        phase1_results = []
        print(f"📸 Processing {len(image_files)} formation images...")
        for i, image_path in enumerate(image_files, 1):
            print(f"  • {i}/{len(image_files)}: {os.path.basename(image_path)}")
            result = self._analyze_w_formation_image(image_path)
            if result['success']:
                phase1_results.append(result)

        self._save_phase_data(phase1_results, "phase1_formation_training.json")
        print(f"\n✅ Phase 1 complete! Analyzed {len(phase1_results)} successful formations")
        return phase1_results

    def train_phase2_outcome_learning(self, folder_path="phase2_with_outcomes"):
        print("\n📈 PHASE 2: Outcome Learning Training")
        print("=" * 60)
        print("📋 Training on W formations + actual price developments")

        if not OPENCV_AVAILABLE:
            print("⚠️ OpenCV (cv2) not installed. Phase 2 requires it for image analysis.")
            return []

        image_files = self._find_images_in_folder(folder_path)
        if not image_files:
            print(f"❌ No images found in {folder_path}/")
            print("💡 Add your W formation + outcome screenshots.")
            return []

        results = []
        print(f"📸 Processing {len(image_files)} outcome images...")
        for i, image_path in enumerate(image_files, 1):
            print(f"  • {i}/{len(image_files)}: {os.path.basename(image_path)}")
            res = self._analyze_w_formation_image(image_path)
            if res['success']:
                # File name hint (e.g., *_success_*.png)
                fname = os.path.basename(image_path).lower()
                if any(k in fname for k in ['success', 'achieved', 'hit', 'reached']):
                    res['actual_outcome'] = True
                elif any(k in fname for k in ['fail', 'miss', 'broke', 'invalid']):
                    res['actual_outcome'] = False
                else:
                    res['actual_outcome'] = None
                results.append(res)

        self._save_phase_data(results, "phase2_outcome_training.json")
        # summary
        labeled = [r for r in results if r.get('actual_outcome') is not None]
        if labeled:
            succ = sum(1 for r in labeled if r['actual_outcome'])
            print(f"📊 Labeled outcomes: {succ}/{len(labeled)} success")
        print(f"✅ Phase 2 complete! {len(results)} items")
        return results

    def train_phase3_manual_analysis(self, folder_path="phase3_manual_analysis"):
        print("\n✏️ PHASE 3: Manual Analysis Learning")
        print("=" * 60)
        print("📋 Learning from your manual analysis screenshots")

        if not OPENCV_AVAILABLE:
            print("⚠️ OpenCV (cv2) not installed. Phase 3 requires it for image analysis.")
            return []

        image_files = self._find_images_in_folder(folder_path)
        if not image_files:
            print(f"❌ No images found in {folder_path}/")
            print("💡 Add screenshots with your drawn lines and targets.")
            return []

        results = []
        print(f"📸 Processing {len(image_files)} manual analysis images...")
        for i, image_path in enumerate(image_files, 1):
            print(f"  • {i}/{len(image_files)}: {os.path.basename(image_path)}")
            # Placeholder: we just record metadata that we could compare later
            results.append({
                'success': True,
                'filename': os.path.basename(image_path),
                'note': 'manual_analysis_screenshot'
            })

        self._save_phase_data(results, "phase3_manual_training.json")
        print(f"✅ Phase 3 complete! {len(results)} items")
        return results

    def _analyze_w_formation_image(self, image_path: str) -> Dict:
        """Analyze a chart image to detect W patterns (no ML).
        Returns {'success': bool, 'patterns': [...], 'filename': ...}
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image.")

            price_data = self._extract_price_data_from_image(img)
            if price_data is None or len(price_data) < 20:
                return {'success': False, 'error': 'No price data extracted', 'filename': os.path.basename(image_path)}

            patterns = self._detect_w_patterns_from_xy(price_data)
            if not patterns:
                return {'success': False, 'error': 'No W patterns found', 'filename': os.path.basename(image_path)}

            return {
                'success': True,
                'filename': os.path.basename(image_path),
                'patterns': patterns,
                'image_metadata': {'size': img.shape, 'points': len(price_data)}
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'filename': os.path.basename(image_path)}

    def _extract_price_data_from_image(self, image) -> Optional[List[tuple]]:
        """Very rough contour-based line extraction; returns [(x,y), ...]"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 30, 100)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        # area * width heuristic → likely main price stroke
        def score(c):
            xy = c.reshape(-1, 2)
            return cv2.contourArea(c) * (np.max(xy[:, 0]) - np.min(xy[:, 0]))
        best = max(cnts, key=score)
        pts = best.reshape(-1, 2)
        pts = pts[pts[:, 0].argsort()]
        # invert y (image origin top-left)
        h = image.shape[0]
        series = [(int(x), float(h - y)) for x, y in pts]
        return series

    def _detect_w_patterns_from_xy(self, xy: List[tuple]) -> List[Dict]:
        """Detect W patterns from a drawn price path (image). Enforces P1 >= P2."""
        if len(xy) < 20:
            return []
        x_vals = [p[0] for p in xy]
        y_vals = [p[1] for p in xy]

        # extrema
        prom = max(1e-9, 0.5 * np.std(y_vals))
        dist_val = max(1, len(y_vals) // 10)
        dist_peak = max(1, len(y_vals) // 12)

        valleys_idx, _ = find_peaks(-np.array(y_vals), prominence=prom, distance=dist_val)
        peaks_idx, _ = find_peaks(np.array(y_vals), prominence=prom, distance=dist_peak)

        valleys = [(x_vals[i], y_vals[i], i) for i in valleys_idx]
        peaks = [(x_vals[i], y_vals[i], i) for i in peaks_idx]

        out = []
        for i in range(len(valleys) - 1):
            for j in range(i + 1, len(valleys)):
                pat = self._validate_w_pattern_xy(valleys[i], valleys[j], peaks, len(y_vals))
                if pat:
                    out.append(pat)
        return out

    def _validate_w_pattern_xy(self, valley1, valley2, peaks, series_len: int) -> Optional[Dict]:
        x1, y1, idx1 = valley1
        x2, y2, idx2 = valley2

        height_tol = 0.20
        min_sep = series_len * 0.10

        height_diff = abs(y1 - y2) / max(y1, y2, 1e-12)
        if height_diff > height_tol:
            return None
        if abs(x2 - x1) < min_sep:
            return None

        peaks_between = [p for p in peaks if x1 < p[0] < x2]
        if not peaks_between:
            return None

        peaks_before = [p for p in peaks if p[0] < x1]
        if not peaks_before:
            return None

        left_peak = max(peaks_before, key=lambda p: p[0])     # by x position
        right_peak = max(peaks_between, key=lambda p: p[1])   # highest

        # Tilt filter
        tol = 1.0 - float(getattr(self, "p1_ge_tol", 0.0))
        if left_peak[1] < right_peak[1] * tol:
            return None

        neck_x, neck_y, neck_idx = right_peak
        if neck_y <= max(y1, y2):
            return None

        pattern_width = abs(x2 - x1)
        pattern_height = neck_y - min(y1, y2)
        depth_symmetry = 1.0 - height_diff

        conf = self._confidence(depth_symmetry, pattern_width, pattern_height, series_len)
        if conf < 0.4:
            return None

        # Target math in pixel-space (informational only)
        trough_mid = (y1 + y2) / 2.0
        target_mag = neck_y - trough_mid  # P2 - trough_mid in pixels (using P2 ~ neckline peak here)

        bars_between = max(1, (right_peak[2] - left_peak[2]))
        neckline_slope = (right_peak[1] - left_peak[1]) / bars_between
        breakout_est_price = right_peak[1] + neckline_slope * bars_between
        target_price = breakout_est_price + target_mag

        return {
            'left_valley': {'x': x1, 'y': y1, 'index': idx1},
            'right_valley': {'x': x2, 'y': y2, 'index': idx2},
            'neckline': {'x': neck_x, 'y': neck_y, 'index': neck_idx},
            'left_peak': {'x': left_peak[0], 'y': left_peak[1], 'index': left_peak[2]},
            'right_peak': {'x': right_peak[0], 'y': right_peak[1], 'index': right_peak[2]},
            'pattern_width': pattern_width,
            'pattern_height': pattern_height,
            'confidence': conf,
            'depth_symmetry': depth_symmetry,
            'price_target': target_price  # pixel-scale
        }

    # ------------------- Stock (Yahoo) scanning -------------------

    def _download_stock_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            data = yf.Ticker(symbol).history(start=start_date, end=end_date, interval='1d')
            if isinstance(data, pd.DataFrame) and len(data) >= 30:
                return data
            return None
        except Exception:
            return None

    def _detect_w_patterns_in_ohlc(self, oc_high: np.ndarray, oc_low: np.ndarray) -> List[Dict]:
        """Detect W patterns using OC extremes (peaks from oc_high, troughs from oc_low)."""
        n = len(oc_high)
        if n < 20:
            return []
        x = np.arange(n)

        prom_low = max(1e-9, 0.5 * np.std(oc_low))
        prom_high = max(1e-9, 0.5 * np.std(oc_high))
        dist_val = max(1, n // 10)
        dist_peak = max(1, n // 12)

        valleys_idx, _ = find_peaks(-oc_low, prominence=prom_low, distance=dist_val)
        peaks_idx, _ = find_peaks(oc_high, prominence=prom_high, distance=dist_peak)

        valleys = [(int(x[i]), float(oc_low[i]), int(i)) for i in valleys_idx]
        peaks = [(int(x[i]), float(oc_high[i]), int(i)) for i in peaks_idx]

        out = []
        for i in range(len(valleys) - 1):
            for j in range(i + 1, len(valleys)):
                pat = self._validate_w_pattern_oc(valleys[i], valleys[j], peaks, n)
                if pat:
                    out.append(pat)
        return out

    def _validate_w_pattern_oc(self, valley1, valley2, peaks, series_len: int) -> Optional[Dict]:
        x1, y1, idx1 = valley1
        x2, y2, idx2 = valley2

        height_tol = 0.20
        min_sep = series_len * 0.10

        height_diff = abs(y1 - y2) / max(y1, y2, 1e-12)
        if height_diff > height_tol:
            return None
        if abs(x2 - x1) < min_sep:
            return None

        peaks_between = [p for p in peaks if x1 < p[0] < x2]
        if not peaks_between:
            return None
        peaks_before = [p for p in peaks if p[0] < x1]
        if not peaks_before:
            return None

        left_peak = max(peaks_before, key=lambda p: p[0])    # closest before L1
        right_peak = max(peaks_between, key=lambda p: p[1])  # highest between L1-L2

        # Tilt (P1 >= P2 within tolerance)
        tol = 1.0 - float(getattr(self, "p1_ge_tol", 0.0))
        if left_peak[1] < right_peak[1] * tol:
            return None

        neck_x, neck_y, neck_idx = right_peak
        if neck_y <= max(y1, y2):
            return None

        pattern_width = abs(x2 - x1)
        pattern_height = neck_y - min(y1, y2)
        depth_symmetry = 1.0 - height_diff
        conf = self._confidence(depth_symmetry, pattern_width, pattern_height, series_len)
        if conf < 0.4:
            return None

        return {
            'left_valley': {'x': x1, 'y': y1, 'index': idx1},
            'right_valley': {'x': x2, 'y': y2, 'index': idx2},
            'neckline': {'x': neck_x, 'y': neck_y, 'index': neck_idx},
            'left_peak': {'x': left_peak[0], 'y': left_peak[1], 'index': left_peak[2]},
            'right_peak': {'x': right_peak[0], 'y': right_peak[1], 'index': right_peak[2]},
            'pattern_width': pattern_width,
            'pattern_height': pattern_height,
            'confidence': conf,
            'depth_symmetry': depth_symmetry
        }

    def _confidence(self, depth_symmetry: float, width: float, height: float, total: int) -> float:
        symmetry_factor = depth_symmetry
        size_factor = min(1.0, (width * height) / max(total * 10.0, 1e-12))
        proportion_factor = min(1.0, width / max(height * 2.0, 1e-12))
        conf = (symmetry_factor * 0.5 + size_factor * 0.3 + proportion_factor * 0.2)
        return max(0.0, min(1.0, conf))

    def _analyze_stock_for_w_patterns(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Analyze a single ticker's last N daily bars for W patterns using Open/Close extremes."""
        try:
            opens = df['Open'].values.astype(float)
            closes = df['Close'].values.astype(float)
            oc_high = np.maximum(opens, closes)  # for peaks
            oc_low = np.minimum(opens, closes)   # for troughs
            dates = df.index

            patterns = self._detect_w_patterns_in_ohlc(oc_high, oc_low)
            out: List[Dict] = []
            for pat in patterns:
                l1 = pat['left_valley']['index']
                l2 = pat['right_valley']['index']
                p1 = pat['left_peak']['index']
                p2 = pat['right_peak']['index']

                l1_price = oc_low[l1]
                l2_price = oc_low[l2]
                p1_price = oc_high[p1]
                p2_price = oc_high[p2]
                neck_price = oc_high[pat['neckline']['index']]

                # === Your target math ===
                trough_midpoint = (l1_price + l2_price) / 2.0
                target_magnitude = p2_price - trough_midpoint
                bars_between = max(1, p2 - p1)  # integer days
                neckline_slope = (p2_price - p1_price) / bars_between
                breakout_estimate_price = p2_price + neckline_slope * bars_between
                target_price = breakout_estimate_price + target_magnitude

                if p2 + bars_between < len(dates):
                    breakout_estimate_date = str(dates[p2 + bars_between].date())
                else:
                    breakout_estimate_date = str(dates[-1].date())

                current_price = max(opens[-1], closes[-1])
                upside_pct = ((target_price - current_price) / max(current_price, 1e-12)) * 100.0
                rr = upside_pct / 10.0  # placeholder

                enriched = dict(pat)
                enriched.update({
                    'symbol': symbol,
                    'current_price': float(current_price),
                    'left_valley_price': float(l1_price),
                    'right_valley_price': float(l2_price),
                    'neckline_price': float(neck_price),
                    'target_price': float(target_price),
                    'breakout_estimate_date': breakout_estimate_date,
                    'upside_potential': float(upside_pct),
                    'risk_reward_ratio': float(rr),
                })
                out.append(enriched)

            return out
        except Exception:
            return []

    def scan_sp500_for_w_patterns(self, lookback_days: Optional[int] = None,
                                  min_confidence: Optional[float] = None) -> List[Dict]:
        """Scan all loaded S&P 500 symbols."""
        if lookback_days is None:
            lookback_days = self.lookback_days_default
        if min_confidence is None:
            min_confidence = self.min_conf_default

        print("\n🔍 SCANNING S&P 500 FOR W PATTERNS (1D TIMEFRAME)")
        print("=" * 70)
        print(f"📊 Scanning {len(self.sp500_symbols)} stocks...")
        print(f"📅 Lookback period: {lookback_days} days")
        print(f"🎯 Minimum confidence: {min_confidence:.1%}")

        candidates: List[Dict] = []
        stats = {
            'scan_timestamp': datetime.now().isoformat(),
            'total_stocks_scanned': 0,
            'patterns_found': 0,
            'high_confidence_patterns': 0,
            'candidates': []
        }

        total = len(self.sp500_symbols)
        for i, sym in enumerate(self.sp500_symbols, 1):
            try:
                print(f"🔍 Scanning {i}/{total}: {sym}...", end=' ')
                df = self._download_stock_data(sym, lookback_days)
                if df is None:
                    print("❌ No data")
                    continue

                pats = self._analyze_stock_for_w_patterns(df, sym)
                stats['total_stocks_scanned'] += 1

                if pats:
                    stats['patterns_found'] += len(pats)
                    hi = [p for p in pats if p['confidence'] >= min_confidence]
                    stats['high_confidence_patterns'] += len(hi)
                    if hi:
                        print(f"✅ {len(hi)} pattern(s) found!")
                        candidates.extend(hi)
                        stats['candidates'].extend(hi)
                    else:
                        print("📊 Pattern found (low confidence)")
                else:
                    print("📊 No patterns")

            except Exception as e:
                print(f"❌ Error: {str(e)[:60]}")
                continue

        # Save results
        scan_file = os.path.join("sp500_scans", f"w_pattern_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(scan_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 Scan summary saved → {scan_file}")

        self._display_scan_results(candidates, stats)
        return candidates  # never None

    def _display_scan_results(self, candidates: List[Dict], stats: Dict):
        print("\n" + "=" * 70)
        print("📊 S&P 500 W PATTERN SCAN RESULTS")
        print("=" * 70)
        print(f"📈 Stocks scanned: {stats['total_stocks_scanned']}")
        print(f"🔍 Total patterns found: {stats['patterns_found']}")
        print(f"⭐ High-confidence patterns: {stats['high_confidence_patterns']}")

        if not candidates:
            print("\n📊 No high-confidence W patterns found in current scan")
            print("💡 Try lowering the confidence threshold or extending lookback days.")
            return

        # Sort by confidence desc, then upside desc
        candidates = sorted(candidates, key=lambda d: (d['confidence'], d['upside_potential']), reverse=True)

        print("\n🎯 TOP W PATTERN CANDIDATES:")
        print("-" * 70)
        head = min(20, len(candidates))
        for i, c in enumerate(candidates[:head], 1):
            print(f"{i:2d}. {c['symbol']:6} | conf={c['confidence']*100:5.1f}% "
                  f"| current=${c['current_price']:,.2f} | target=${c['target_price']:,.2f} "
                  f"| upside={c['upside_potential']:5.1f}% | est_breakout={c['breakout_estimate_date']}")

    # ------------------- Phase runner -------------------

    def run_three_phase_training(self):
        """Run all three phases (no ML fitting)."""
        print("\n" + "🚀 STARTING THREE-PHASE W PATTERN TRAINING".center(80, "="))

        print("\n" + "🔍 PHASE 1 — Formation Only".center(80, "="))
        try:
            p1 = self.train_phase1_formation_detection() or []
            print(f"✅ Phase 1 done (items: {len(p1)})")
        except Exception as e:
            print(f"⚠️ Phase 1 error: {e}")

        print("\n" + "📈 PHASE 2 — With Outcomes".center(80, "="))
        try:
            p2 = self.train_phase2_outcome_learning() or []
            print(f"✅ Phase 2 done (items: {len(p2)})")
        except Exception as e:
            print(f"⚠️ Phase 2 error: {e}")

        print("\n" + "✏️ PHASE 3 — Manual Analysis".center(80, "="))
        try:
            p3 = self.train_phase3_manual_analysis() or []
            print(f"✅ Phase 3 done (items: {len(p3)})")
        except Exception as e:
            print(f"⚠️ Phase 3 error: {e}")

        print("\n" + "✅ TRAINING PHASES COMPLETE".center(80, "="))
        return {"phase1": p1 if 'p1' in locals() else [],
                "phase2": p2 if 'p2' in locals() else [],
                "phase3": p3 if 'p3' in locals() else []}


# ------------------- CLI loop -------------------

def main():
    print("🎯 S&P 500 W Pattern Scanner & Trainer")
    print("=" * 60)
    scanner = SP500WPatternScanner()

    while True:
        print("\n📋 TRAINING & SCANNING OPTIONS:")
        print("1. Run complete 3-phase training")
        print("2. Phase 1 only (Formation detection)")
        print("3. Phase 2 only (Outcome learning)")
        print("4. Phase 3 only (Manual analysis)")
        print("5. Live S&P 500 scan only")
        print("6. Exit (or press q)")

        choice = input("\nSelect option (1-6, q to quit): ").strip().lower()
        if choice in {"6", "q", "quit", "exit"}:
            print("👋 Goodbye!")
            break

        try:
            if choice == "1":
                scanner.run_three_phase_training()

            elif choice == "2":
                scanner.train_phase1_formation_detection()

            elif choice == "3":
                scanner.train_phase2_outcome_learning()

            elif choice == "4":
                scanner.train_phase3_manual_analysis()

            elif choice == "5":
                # You can tweak these defaults if you want:
                lookback = scanner.lookback_days_default   # e.g., 120 for a wider window
                min_conf = scanner.min_conf_default        # e.g., 0.55 to be looser
                cands = scanner.scan_sp500_for_w_patterns(lookback_days=lookback,
                                                          min_confidence=min_conf)
                print(f"\n🎯 Found {len(cands)} high-confidence W candidates")

            else:
                print("❓ Invalid choice. Please select 1–6 or q.")

        except Exception as e:
            print(f"⚠️ Error: {e}")

        input("\nPress Enter to return to the menu...")

if __name__ == "__main__":
    main()
