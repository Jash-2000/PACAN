"""
PACAN Week 8 — CHB-MIT Deep Calibration + Pairwise MEM Energy Landscape

Implements:
  1. Multi-patient EEG feature extraction (CHB01, 02, 03, 05)
  2. WC parameter fitting to empirical interictal vs ictal statistics
  3. Pairwise Maximum Entropy Model (Ising) from binarized EEG
  4. Empirical energy landscape with local minima identification
  5. Comparison: empirical ΔE_barrier vs WC-predicted barrier
  
Theoretical foundations:
  - Ezaki et al. 2018: Pairwise MEM from neuroimaging data
  - Schindler et al. 2007: EEG signatures of seizure onset
  - Kramer & Cash 2012: Epilepsy as a disorder of cortical network organization

Dataset:
  CHB-MIT Scalp EEG Database (PhysioNet)
  https://physionet.org/content/chbmit/1.0.0/
  
  Patients: CHB01, CHB02, CHB03, CHB05
  Channels: 23 (select top 10 by seizure-relevant power)
  Sampling: 256 Hz

Usage:
    # Step 1: Download dataset (run once)
    bash download_chbmit.sh
    
    # Step 2: Extract features for all patients
    python pacan_week8.py --extract_features --data_dir ~/chbmit_data
    
    # Step 3: Fit WC parameters
    python pacan_week8.py --fit_wc --patient chb01
    
    # Step 4: Build pairwise MEM
    python pacan_week8.py --build_mem --patient chb01
    
    # Step 5: Compare barriers
    python pacan_week8.py --compare_barriers --patient chb01

Requirements:
    pip install mne pyedflib scipy numpy matplotlib scikit-learn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
    import pyedflib
    from scipy import signal, stats
    from scipy.optimize import minimize, differential_evolution
    from sklearn.decomposition import PCA
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install mne pyedflib scipy scikit-learn")
    exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. EEG FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

class EEGFeatureExtractor:
    """Extract seizure-relevant features from CHB-MIT EEG."""
    
    def __init__(self, data_dir, patient_id, n_channels=10):
        """
        Args:
            data_dir: Path to CHB-MIT dataset root
            patient_id: e.g., 'chb01'
            n_channels: Number of channels to select (by gamma/theta ratio)
        """
        self.data_dir = Path(data_dir)
        self.patient_id = patient_id
        self.patient_dir = self.data_dir / patient_id
        self.n_channels = n_channels
        
        # Load summary file to identify seizure files
        self.seizure_files = self._parse_summary()
        
    def _parse_summary(self):
        """Parse {patient}-summary.txt to identify seizure recordings."""
        summary_path = self.patient_dir / f"{self.patient_id}-summary.txt"
        seizures = []
        
        if not summary_path.exists():
            print(f"Warning: {summary_path} not found. Assuming no seizures.")
            return seizures
        
        with open(summary_path, 'r') as f:
            lines = f.readlines()
        
        current_file = None
        num_seizures = 0
        seizure_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith('File Name:'):
                current_file = line.split(':')[1].strip()
                num_seizures = 0
                seizure_idx = 0
            
            elif line.startswith('Number of Seizures in File:'):
                num_seizures = int(line.split(':')[1].strip())
            
            elif line.startswith('Seizure Start Time:') and current_file and num_seizures > 0:
                # Extract start time (in seconds)
                start_str = line.split(':')[1].strip()
                start = int(start_str.split()[0])  # "2996 seconds" -> 2996
                
                seizures.append({
                    'file': current_file, 
                    'start': start,
                    'seizure_idx': seizure_idx
                })
                seizure_idx += 1
        
        print(f"Found {len(seizures)} seizure(s) in {self.patient_id}")
        return seizures
    
    def load_eeg(self, filename, t_start=0, duration=60):
        """
        Load EEG segment using MNE.
        
        Args:
            filename: EDF file name
            t_start: Start time in seconds
            duration: Duration in seconds
        
        Returns:
            data: (n_channels, n_samples) array
            ch_names: Channel names
            sfreq: Sampling frequency
        """
        filepath = self.patient_dir / filename
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        
        # Preprocessing
        raw.resample(256)  # Standardize to 256 Hz
        raw.filter(0.5, 100, fir_design='firwin', verbose=False)  # Bandpass
        raw.notch_filter(60, verbose=False)  # Notch at 60 Hz
        
        # Extract segment
        data, times = raw[:, int(t_start*256):int((t_start+duration)*256)]
        
        return data, raw.ch_names, raw.info['sfreq']
    
    def extract_band_power(self, data, sfreq):
        """
        Compute band power in standard EEG bands.
        
        Returns:
            dict with keys: delta, theta, alpha, beta, gamma
        """
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 80),
        }
        
        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            # Welch's method for PSD
            freqs, psd = signal.welch(data, fs=sfreq, nperseg=256)
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_powers[band_name] = np.mean(psd[:, idx], axis=1)
        
        return band_powers
    
    def extract_features(self, segment_type='interictal', n_segments=10):
        """
        Extract features from multiple segments.
        
        Args:
            segment_type: 'interictal', 'preictal', 'ictal', 'postictal'
            n_segments: Number of segments to extract
        
        Returns:
            features: dict with statistics across segments
        """
        features = {
            'band_power': {'delta': [], 'theta': [], 'alpha': [], 'beta': [], 'gamma': []},
            'gamma_theta_ratio': [],
            'spectral_entropy': [],
            'line_length': [],
        }
        
        if segment_type == 'interictal':
            # Sample from non-seizure files, >30min from any seizure
            files = [f for f in os.listdir(self.patient_dir) if f.endswith('.edf')]
            seizure_files = set([s['file'] for s in self.seizure_files])
            interictal_files = [f for f in files if f not in seizure_files][:n_segments]
            
            for filename in tqdm(interictal_files, desc=f"Extracting {segment_type}"):
                data, ch_names, sfreq = self.load_eeg(filename, t_start=300, duration=60)
                
                bp = self.extract_band_power(data, sfreq)
                for band in bp:
                    features['band_power'][band].append(bp[band])
                
                # Gamma/theta ratio (E/I proxy)
                gt_ratio = bp['gamma'] / (bp['theta'] + 1e-12)
                features['gamma_theta_ratio'].append(gt_ratio)
                
                # Spectral entropy
                freqs, psd = signal.welch(data, fs=sfreq, nperseg=256)
                psd_norm = psd / (psd.sum(axis=1, keepdims=True) + 1e-12)
                se = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=1)
                features['spectral_entropy'].append(se)
                
                # Line length
                ll = np.sum(np.abs(np.diff(data, axis=1)), axis=1)
                features['line_length'].append(ll)
        
        elif segment_type == 'ictal':
            # Extract during seizures
            for seizure in self.seizure_files[:n_segments]:
                data, ch_names, sfreq = self.load_eeg(
                    seizure['file'], 
                    t_start=seizure['start'], 
                    duration=min(60, 120)  # Up to 2 min
                )
                
                bp = self.extract_band_power(data, sfreq)
                for band in bp:
                    features['band_power'][band].append(bp[band])
                
                gt_ratio = bp['gamma'] / (bp['theta'] + 1e-12)
                features['gamma_theta_ratio'].append(gt_ratio)
                
                freqs, psd = signal.welch(data, fs=sfreq, nperseg=256)
                psd_norm = psd / (psd.sum(axis=1, keepdims=True) + 1e-12)
                se = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=1)
                features['spectral_entropy'].append(se)
                
                ll = np.sum(np.abs(np.diff(data, axis=1)), axis=1)
                features['line_length'].append(ll)
        
        # Aggregate
        aggregated = {}
        for key in features:
            if key == 'band_power':
                aggregated[key] = {}
                for band in features[key]:
                    if len(features[key][band]) > 0:
                        aggregated[key][band] = np.concatenate(features[key][band])
                    else:
                        aggregated[key][band] = np.array([])
            else:
                if len(features[key]) > 0:
                    aggregated[key] = np.concatenate(features[key])
                else:
                    aggregated[key] = np.array([])
        
        return aggregated


# ══════════════════════════════════════════════════════════════════════════════
# 2. WC PARAMETER FITTING
# ══════════════════════════════════════════════════════════════════════════════

def wc_simulate_population(P, wEE, wEI, wIE, wII, sigma, T=1000, dt=0.1):
    """
    Simulate single WC population (for fitting).
    
    Returns:
        xE, xI: Activity timeseries
    """
    xE = 0.5
    xI = 0.2
    
    sqrt_dt = np.sqrt(dt)
    xE_trace = []
    xI_trace = []
    
    for t in range(T):
        IE = wEE * xE - wEI * xI + P
        II = wIE * xE - wII * xI
        
        sE = 1 / (1 + np.exp(-1.2 * IE))
        sI = 1 / (1 + np.exp(-1.2 * II))
        
        dxE = (-xE + sE) / 10.
        dxI = (-xI + sI) / 10.
        
        xE += dxE * dt + sigma * sqrt_dt * np.random.randn()
        xI += dxI * dt + sigma * sqrt_dt * np.random.randn()
        
        xE = np.clip(xE, 0, 1)
        xI = np.clip(xI, 0, 1)
        
        xE_trace.append(xE)
        xI_trace.append(xI)
    
    return np.array(xE_trace), np.array(xI_trace)


def fit_wc_to_eeg(features_interictal, features_ictal, n_trials=50):
    """
    Fit WC parameters to match ictal vs interictal statistics.
    
    Optimization target:
        Minimize KL divergence between:
        - WC(healthy) vs empirical interictal
        - WC(pathological) vs empirical ictal
    
    Returns:
        best_params: dict with P, wEE, wEI, sigma, etc.
    """
    print("\n[WC Parameter Fitting]")
    print("Matching WC model to empirical EEG statistics...")
    
    # Target statistics (from empirical data)
    target_inter_gamma_theta = np.mean(features_interictal['gamma_theta_ratio'])
    target_ictal_gamma_theta = np.mean(features_ictal['gamma_theta_ratio'])
    
    target_inter_entropy = np.mean(features_interictal['spectral_entropy'])
    target_ictal_entropy = np.mean(features_ictal['spectral_entropy'])
    
    def loss_fn(params):
        """Loss function for parameter optimization."""
        P, wEE, wEI, sigma = params
        wIE = 6.0  # Fixed
        wII = 1.0  # Fixed
        
        # Simulate healthy state (low P)
        xE_h, xI_h = wc_simulate_population(P*0.6, wEE, wEI, wIE, wII, sigma, T=2000)
        
        # Simulate pathological state (high P)
        xE_p, xI_p = wc_simulate_population(P*1.2, wEE, wEI, wIE, wII, sigma, T=2000)
        
        # Proxy: E/I ratio ≈ xE/xI
        model_inter_EI = np.mean(xE_h) / (np.mean(xI_h) + 1e-6)
        model_ictal_EI = np.mean(xE_p) / (np.mean(xI_p) + 1e-6)
        
        # Proxy: entropy ≈ std(xE)
        model_inter_entropy = np.std(xE_h)
        model_ictal_entropy = np.std(xE_p)
        
        # Normalize targets
        target_inter_EI = target_inter_gamma_theta
        target_ictal_EI = target_ictal_gamma_theta
        
        # MSE loss
        loss = (
            (model_inter_EI - target_inter_EI)**2 +
            (model_ictal_EI - target_ictal_EI)**2 +
            (model_inter_entropy - target_inter_entropy)**2 * 0.1 +
            (model_ictal_entropy - target_ictal_entropy)**2 * 0.1
        )
        
        return loss
    
    # Parameter bounds
    bounds = [
        (0.5, 2.5),   # P
        (8.0, 12.0),  # wEE
        (6.0, 10.0),  # wEI
        (0.01, 0.20), # sigma
    ]
    
    # Differential evolution (global optimization)
    result = differential_evolution(loss_fn, bounds, maxiter=n_trials, 
                                   seed=42, disp=True, workers=1)
    
    P_fit, wEE_fit, wEI_fit, sigma_fit = result.x
    
    fitted_params = {
        'P': P_fit,
        'wEE': wEE_fit,
        'wEI': wEI_fit,
        'wIE': 6.0,
        'wII': 1.0,
        'sigma': sigma_fit,
        'loss': result.fun,
    }
    
    print(f"\nFitted parameters:")
    for k, v in fitted_params.items():
        print(f"  {k:10s}: {v:.4f}")
    
    return fitted_params


# ══════════════════════════════════════════════════════════════════════════════
# 3. PAIRWISE MAXIMUM ENTROPY MODEL (Ezaki et al. 2018)
# ══════════════════════════════════════════════════════════════════════════════

def binarize_eeg(data, threshold='median'):
    """
    Binarize EEG amplitude per channel.
    
    Args:
        data: (n_channels, n_samples)
        threshold: 'median' or float
    
    Returns:
        binary_data: (n_channels, n_samples) in {-1, +1}
    """
    if threshold == 'median':
        thresh = np.median(data, axis=1, keepdims=True)
    else:
        thresh = threshold
    
    binary = np.where(data > thresh, 1, -1)
    return binary


def fit_pairwise_mem(binary_data):
    """
    Fit pairwise Ising model via pseudo-likelihood maximization.
    
    Model: P(s) ∝ exp(-E(s)) where E(s) = -Σ h_i s_i - Σ J_ij s_i s_j
    
    Args:
        binary_data: (n_channels, n_samples) in {-1, +1}
    
    Returns:
        h: (n_channels,) local fields
        J: (n_channels, n_channels) couplings
    """
    N, T = binary_data.shape
    
    # Pseudo-likelihood: fit each node independently
    h = np.zeros(N)
    J = np.zeros((N, N))
    
    for i in range(N):
        # Target: s_i
        # Predictors: s_{-i}
        si = binary_data[i, :]
        s_minus_i = np.delete(binary_data, i, axis=0)
        
        # Logistic regression
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(fit_intercept=True, max_iter=1000)
        model.fit(s_minus_i.T, (si + 1) / 2)  # Convert {-1,1} to {0,1}
        
        h[i] = model.intercept_[0]
        J_i = model.coef_[0]
        
        # Fill in coupling matrix
        idx = 0
        for j in range(N):
            if j != i:
                J[i, j] = J_i[idx]
                idx += 1
    
    # Symmetrize J
    J = (J + J.T) / 2
    
    return h, J


def find_mem_local_minima(h, J, n_samples=10000):
    """
    Find local minima of Ising energy landscape via gradient descent.
    
    Returns:
        minima: list of unique local minimum states
        energies: corresponding energies
    """
    N = len(h)
    
    def energy(s):
        return -np.dot(h, s) - 0.5 * np.dot(s, np.dot(J, s))
    
    def gradient_flip(s):
        """Flip coordinate that reduces energy most."""
        s_new = s.copy()
        best_delta = 0
        best_i = -1
        
        for i in range(N):
            s_flip = s.copy()
            s_flip[i] *= -1
            delta = energy(s_flip) - energy(s)
            if delta < best_delta:
                best_delta = delta
                best_i = i
        
        if best_i >= 0:
            s_new[best_i] *= -1
        
        return s_new, best_delta
    
    # Random initializations
    minima_states = []
    minima_energies = []
    
    for _ in tqdm(range(n_samples), desc="Finding local minima"):
        s = np.random.choice([-1, 1], size=N)
        
        # Gradient descent
        for _ in range(100):
            s_new, delta = gradient_flip(s)
            if delta >= 0:  # Converged
                break
            s = s_new
        
        # Check if new minimum
        e = energy(s)
        is_new = True
        for i, (sm, em) in enumerate(zip(minima_states, minima_energies)):
            if np.allclose(s, sm):
                is_new = False
                break
        
        if is_new:
            minima_states.append(s.copy())
            minima_energies.append(e)
    
    return minima_states, minima_energies


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    if args.extract_features:
        print("="*70)
        print("STEP 1: EXTRACT EEG FEATURES")
        print("="*70)
        
        extractor = EEGFeatureExtractor(args.data_dir, args.patient)
        
        # Extract interictal features
        print("\nExtracting interictal features...")
        feat_inter = extractor.extract_features('interictal', n_segments=5)
        
        # Extract ictal features
        print("\nExtracting ictal features...")
        feat_ictal = extractor.extract_features('ictal', n_segments=len(extractor.seizure_files))
        
        # Save
        def convert_to_serializable(obj):
            """Recursively convert numpy arrays to lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        results = {
            'interictal': convert_to_serializable(feat_inter),
            'ictal': convert_to_serializable(feat_ictal),
        }
        
        output_path = save_dir / f"eeg_features_{args.patient}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nFeatures saved: {output_path}")
    
    if args.fit_wc:
        print("\n" + "="*70)
        print("STEP 2: FIT WC PARAMETERS")
        print("="*70)
        
        # Load features
        feat_path = save_dir / f"eeg_features_{args.patient}.json"
        with open(feat_path, 'r') as f:
            features = json.load(f)
        
        # Fit WC model
        fitted_params = fit_wc_to_eeg(features['interictal'], features['ictal'])
        
        # Save
        output_path = save_dir / f"wc_params_{args.patient}.json"
        with open(output_path, 'w') as f:
            json.dump(fitted_params, f, indent=2)
        print(f"\nWC parameters saved: {output_path}")
    
    if args.build_mem:
        print("\n" + "="*70)
        print("STEP 3: BUILD PAIRWISE MEM")
        print("="*70)
        
        # Load raw EEG segment
        extractor = EEGFeatureExtractor(args.data_dir, args.patient)
        
        # Get ictal segment
        if len(extractor.seizure_files) == 0:
            print("No seizures found for this patient!")
            return
        
        seizure = extractor.seizure_files[0]
        data, ch_names, sfreq = extractor.load_eeg(seizure['file'], 
                                                    t_start=seizure['start'],
                                                    duration=60)
        
        # Select top 10 channels by gamma power
        bp = extractor.extract_band_power(data, sfreq)
        gamma_power = bp['gamma']
        top_channels = np.argsort(gamma_power)[-10:]
        data_selected = data[top_channels, :]
        
        # Binarize
        binary_data = binarize_eeg(data_selected)
        
        # Fit MEM
        print("Fitting pairwise MEM...")
        h, J = fit_pairwise_mem(binary_data)
        
        # Find local minima
        print("Finding local minima...")
        minima_states, minima_energies = find_mem_local_minima(h, J, n_samples=1000)
        
        results = {
            'h': h.tolist(),
            'J': J.tolist(),
            'minima_states': [m.tolist() for m in minima_states],
            'minima_energies': minima_energies,
            'n_minima': len(minima_states),
        }
        
        output_path = save_dir / f"mem_landscape_{args.patient}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nMEM landscape saved: {output_path}")
        print(f"Found {len(minima_states)} local minima")
    
    if args.compare_barriers:
        print("\n" + "="*70)
        print("STEP 4: COMPARE EMPIRICAL VS WC BARRIERS")
        print("="*70)
        
        # Load WC parameters
        wc_params_path = save_dir / f"wc_params_{args.patient}.json"
        with open(wc_params_path, 'r') as f:
            wc_params = json.load(f)
        
        # Load MEM landscape
        mem_path = save_dir / f"mem_landscape_{args.patient}.json"
        with open(mem_path, 'r') as f:
            mem_landscape = json.load(f)
        
        print(f"\nWC Parameters (fitted):")
        print(f"  P = {wc_params['P']:.4f}")
        print(f"  wEE = {wc_params['wEE']:.4f}")
        print(f"  wEI = {wc_params['wEI']:.4f}")
        print(f"  sigma = {wc_params['sigma']:.4f}")
        print(f"  Loss = {wc_params['loss']:.6f}")
        
        print(f"\nEmpirical MEM Landscape:")
        print(f"  Number of local minima: {mem_landscape['n_minima']}")
        
        minima_energies = np.array(mem_landscape['minima_energies'])
        min_idx = np.argmin(minima_energies)
        max_idx = np.argmax(minima_energies)
        
        empirical_barrier = minima_energies[max_idx] - minima_energies[min_idx]
        
        print(f"  Lowest energy (healthy): {minima_energies[min_idx]:.4f}")
        print(f"  Highest energy (pathological): {minima_energies[max_idx]:.4f}")
        print(f"  Empirical ΔE_barrier: {empirical_barrier:.4f}")
        
        # Estimate WC barrier from Week 7 data (approximate)
        # Use fitted parameters to estimate barrier
        # For now, use a simple proxy: barrier ~ wEI / P
        wc_barrier_estimate = wc_params['wEI'] / (wc_params['P'] + 1e-6)
        
        print(f"\nWC Model Barrier Estimate:")
        print(f"  (Proxy: wEI / P) = {wc_barrier_estimate:.4f}")
        
        # Comparison
        barrier_ratio = empirical_barrier / (wc_barrier_estimate + 1e-6)
        print(f"\nBarrier Comparison:")
        print(f"  Empirical / WC ratio: {barrier_ratio:.2f}×")
        print(f"  Status: {'✓ Match within 2-3×' if 0.33 < barrier_ratio < 3.0 else '✗ Mismatch'}")
        
        # Save comparison results
        comparison_results = {
            'patient': args.patient,
            'wc_params': wc_params,
            'empirical_barrier': float(empirical_barrier),
            'wc_barrier_estimate': float(wc_barrier_estimate),
            'barrier_ratio': float(barrier_ratio),
            'n_empirical_minima': mem_landscape['n_minima'],
        }
        
        output_path = save_dir / f"barrier_comparison_{args.patient}.json"
        with open(output_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\nComparison saved: {output_path}")
    
    print("\nWeek 8 pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PACAN Week 8: CHB-MIT Calibration")
    parser.add_argument("--data_dir", type=str, default="~/chbmit_data",
                       help="Path to CHB-MIT dataset")
    parser.add_argument("--patient", type=str, default="chb01",
                       help="Patient ID (chb01, chb02, chb03, chb05)")
    parser.add_argument("--save_dir", type=str, default="results_w8",
                       help="Output directory")
    
    parser.add_argument("--extract_features", action="store_true",
                       help="Extract EEG features")
    parser.add_argument("--fit_wc", action="store_true",
                       help="Fit WC parameters")
    parser.add_argument("--build_mem", action="store_true",
                       help="Build pairwise MEM")
    parser.add_argument("--compare_barriers", action="store_true",
                       help="Compare empirical vs WC barriers")
    
    args = parser.parse_args()
    
    main(args)
