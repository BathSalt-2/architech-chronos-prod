import torch
import torch.nn.functional as F
from collections import deque
import numpy as np

class SigmaMatrixAuditor:
    """Runtime ethical drift monitor using Σ-Matrix DMAIC principles."""
    
    def __init__(self, threshold=0.05, window_size=10):
        self.threshold = threshold
        self.window_size = window_size
        self.drift_history = deque(maxlen=window_size)
        self.baseline_ethic = None
        
    def set_baseline(self, ethic_vector):
        """Define baseline ethical state (from training)."""
        self.baseline_ethic = ethic_vector
        print(f"Baseline ethic set: shape {ethic_vector.shape}")
    
    def measure(self, current_ethic):
        """Measure drift from baseline (DMAIC: Measure)."""
        if self.baseline_ethic is None:
            return 0.0
        
        drift = F.mse_loss(current_ethic, self.baseline_ethic).item()
        self.drift_history.append(drift)
        return drift
    
    def analyze(self):
        """Analyze drift trends (DMAIC: Analyze)."""
        if len(self.drift_history) < 2:
            return {'status': 'insufficient_data', 'trend': 0.0}
        
        drifts = np.array(self.drift_history)
        mean_drift = drifts.mean()
        std_drift = drifts.std()
        trend = np.polyfit(range(len(drifts)), drifts, 1)[0]  # Linear trend
        
        return {
            'status': 'ok' if mean_drift < self.threshold else 'alert',
            'mean': mean_drift,
            'std': std_drift,
            'trend': trend,
            'sigma': (mean_drift / std_drift) if std_drift > 0 else 0.0
        }
    
    def improve(self, analysis):
        """Suggest corrective actions (DMAIC: Improve)."""
        if analysis['status'] == 'alert':
            if analysis['trend'] > 0:
                return {
                    'action': 'apply_correction',
                    'magnitude': min(analysis['mean'] * 0.5, 1.0),
                    'reason': 'Increasing drift detected'
                }
            else:
                return {
                    'action': 'monitor',
                    'reason': 'Drift high but decreasing'
                }
        return {'action': 'none', 'reason': 'Within acceptable range'}
    
    def control(self, model_state, correction):
        """Apply control actions (DMAIC: Control)."""
        if correction['action'] == 'apply_correction':
            # Stub: In practice, adjust model parameters or prompt
            print(f"Applying correction: magnitude={correction['magnitude']:.4f}")
            return True
        return False
    
    def dmaic_cycle(self, current_ethic, model_state=None):
        """Full DMAIC cycle for one inference step."""
        # Define (baseline already set)
        drift = self.measure(current_ethic)
        analysis = self.analyze()
        correction = self.improve(analysis)
        applied = self.control(model_state, correction)
        
        return {
            'drift': drift,
            'analysis': analysis,
            'correction': correction,
            'applied': applied
        }
    
    def get_report(self):
        """Generate audit report."""
        if not self.drift_history:
            return "No audit data available"
        
        analysis = self.analyze()
        return f"""
Σ-Matrix Audit Report
=====================
Status: {analysis['status'].upper()}
Mean Drift: {analysis['mean']:.6f} (threshold: {self.threshold})
Std Dev: {analysis['std']:.6f}
Trend: {'↑' if analysis['trend'] > 0 else '↓'} ({analysis['trend']:.6f})
Sigma Level: {analysis['sigma']:.2f}σ
Window: {len(self.drift_history)}/{self.window_size} samples
"""

# Example usage
if __name__ == '__main__':
    auditor = SigmaMatrixAuditor(threshold=0.05)
    
    # Set baseline
    baseline = torch.randn(1, 128)
    auditor.set_baseline(baseline)
    
    # Simulate inference with drift
    for i in range(15):
        current = baseline + torch.randn(1, 128) * (0.01 + i * 0.005)
        result = auditor.dmaic_cycle(current)
        print(f"Step {i}: Drift={result['drift']:.6f}, Action={result['correction']['action']}")
    
    print(auditor.get_report())
