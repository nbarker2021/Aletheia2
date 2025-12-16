"""
Improved Test 2: Market Anomaly Detection with Proper Phi Metric
"""
from pathlib import Path


import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proper_phi_metric import ProperPhiMetric
from layer4_governance.gravitational import GravitationalLayer

def test_market_anomaly_detection_improved():
    """
    Test market anomaly detection using proper phi metric with temporal coherence.
    """
    print("\n" + "="*80)
    print("IMPROVED TEST 2: Market Anomaly Detection with Proper Phi Metric")
    print("="*80)
    
    # Initialize components
    print("Initializing components...")
    phi = ProperPhiMetric()
    grav = GravitationalLayer()
    
    # Generate synthetic market data
    print("Generating synthetic market data...")
    np.random.seed(42)
    
    # Normal market behavior (100 time steps)
    normal_returns = np.random.randn(100) * 0.02  # 2% volatility
    
    # Inject anomaly at step 60-65 (flash crash)
    anomaly_returns = normal_returns.copy()
    anomaly_returns[60:65] = [-0.15, -0.12, -0.10, 0.08, 0.06]  # Crash and recovery
    
    # Convert to price series
    prices = 100 * np.exp(np.cumsum(anomaly_returns))
    
    print(f"Generated {len(prices)} price points with anomaly at t=60-65")
    
    # Map to 24D feature space
    print("Mapping price data to 24D geometric space...")
    
    feature_vectors = []
    phi_scores = []
    anomaly_flags = []
    
    window_size = 10
    for t in range(window_size, len(prices)):
        # Extract window
        window = prices[t-window_size:t]
        
        # Create 24D feature vector
        features = np.zeros(24)
        features[0] = np.mean(window)  # Mean price
        features[1] = np.std(window)   # Volatility
        features[2] = window[-1] - window[0]  # Total change
        features[3] = np.max(window) - np.min(window)  # Range
        
        # Add momentum indicators
        for i in range(5):
            if t-i-1 >= 0:
                features[4+i] = prices[t-i] - prices[t-i-1]
        
        # Add moving averages
        for i in range(5):
            if t-i*2 >= 0:
                features[9+i] = np.mean(prices[max(0,t-i*2-5):t-i*2])
        
        # Add FFT components (frequency analysis)
        if len(window) >= 8:
            fft = np.fft.fft(window)
            features[14:20] = np.abs(fft[:6])
        
        # Don't normalize - keep actual magnitudes for anomaly detection
        # Normalization makes everything too similar
        
        feature_vectors.append(features)
        
        # Calculate phi score with temporal context
        context = {
            'previous_vectors': feature_vectors[max(0, len(feature_vectors)-10):]
        }
        phi_score = phi.calculate(features, context)
        phi_scores.append(phi_score)
        
        # Detect anomaly using relative change in phi score
        if len(phi_scores) >= 5:
            # Look at recent phi scores
            recent_scores = phi_scores[-5:]
            mean_recent = np.mean(recent_scores[:-1])  # Exclude current
            
            # Anomaly if current score drops significantly from recent average
            drop_threshold = 0.15  # 15% drop
            relative_drop = (mean_recent - phi_score) / (mean_recent + 1e-10)
            
            is_anomaly = relative_drop > drop_threshold
            
            if is_anomaly:
                threshold = mean_recent - drop_threshold * mean_recent
            else:
                threshold = mean_recent - drop_threshold * mean_recent
        else:
            is_anomaly = False
            threshold = 0.0
        anomaly_flags.append(is_anomaly)
        
        if is_anomaly:
            print(f"  ⚠️  Anomaly detected at t={t}, phi={phi_score:.4f}, threshold={threshold:.4f}")
    
    # Evaluate detection performance
    print("\nEvaluating detection performance...")
    
    # Ground truth: anomaly at indices 60-65 (adjusted for window_size offset)
    true_anomaly_start = 60 - window_size
    true_anomaly_end = 65 - window_size
    
    detected_in_window = False
    detected_indices = []
    for i, flag in enumerate(anomaly_flags):
        if flag:
            detected_indices.append(i)
            if true_anomaly_start <= i <= true_anomaly_end:
                detected_in_window = True
    
    # Calculate metrics
    total_windows = len(anomaly_flags)
    anomalies_detected = sum(anomaly_flags)
    recall = 1.0 if detected_in_window else 0.0
    
    # Calculate precision (how many detected anomalies are true positives)
    true_positives = sum(1 for i in detected_indices if true_anomaly_start <= i <= true_anomaly_end)
    precision = true_positives / anomalies_detected if anomalies_detected > 0 else 0.0
    
    print(f"\nResults:")
    print(f"  Total windows analyzed: {total_windows}")
    print(f"  Anomalies detected: {anomalies_detected}")
    print(f"  Detected crash window: {detected_in_window}")
    print(f"  Recall: {recall*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Average phi score: {np.mean(phi_scores):.6f}")
    print(f"  Min phi score: {np.min(phi_scores):.6f}")
    print(f"  Phi score at crash (t=60): {phi_scores[true_anomaly_start]:.6f}")
    
    # Success if we detected the crash
    success = detected_in_window and recall >= 0.5
    
    if success:
        print(f"\n✅ TEST PASSED: Successfully detected market anomaly!")
    else:
        print(f"\n❌ TEST FAILED: Did not detect market anomaly")
    
    return success, {
        'total_windows': total_windows,
        'anomalies_detected': anomalies_detected,
        'detected_crash': detected_in_window,
        'recall': recall,
        'precision': precision,
        'avg_phi_score': np.mean(phi_scores),
        'min_phi_score': np.min(phi_scores),
        'crash_phi_score': phi_scores[true_anomaly_start],
        'detected_indices': detected_indices[:10]
    }

if __name__ == '__main__':
    success, metrics = test_market_anomaly_detection_improved()
    print(f"\nFinal Result: {'PASS' if success else 'FAIL'}")
    print(f"Metrics: {metrics}")
