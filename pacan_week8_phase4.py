"""
PACAN Week 8 Phase 4: Empirical PACAN Optimization

Run full PACAN vs baselines comparison on WC networks parameterized
from real CHB-MIT patient EEG data.

This answers: "Does PACAN's advantage persist when parameters come from 
real seizure recordings?"

Usage:
    # Run all 4 patients sequentially
    python pacan_week8_phase4.py --all_patients
    
    # Run single patient
    python pacan_week8_phase4.py --patient chb01
    
    # Generate comparison visualization
    python pacan_week8_phase4.py --visualize

Requires: Week 8 Phase 2 outputs (wc_params_*.json)
"""

import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import Week 6 infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent))

# We'll reuse Wilson-Cowan and PACAN code structure from Week 6
# Adapted to use patient-specific parameters


# ══════════════════════════════════════════════════════════════════════════════
# WILSON-COWAN MODEL (With patient-specific parameters)
# ══════════════════════════════════════════════════════════════════════════════

class WilsonCowanNetwork:
    """Wilson-Cowan network with patient-calibrated parameters."""
    
    def __init__(self, N, wc_params, seed=42):
        """
        Args:
            N: Number of populations
            wc_params: Dict from Week 8 Phase 2 fitting
            seed: Random seed for network topology
        """
        self.N = N
        np.random.seed(seed)
        
        # Patient-specific parameters
        self.P = wc_params['P']
        self.wEE = wc_params['wEE']
        self.wEI = wc_params['wEI']
        self.wIE = wc_params['wIE']
        self.wII = wc_params['wII']
        self.sigma = wc_params['sigma']
        
        # Network topology (sparse random)
        self.W_net = self._create_topology(connectivity=0.3)
        
        # Time constants
        self.tau_E = 10.0
        self.tau_I = 10.0
        self.dt = 0.1
        
    def _create_topology(self, connectivity=0.3):
        """Create sparse random network."""
        W = np.random.randn(self.N, self.N) * 0.5
        mask = np.random.rand(self.N, self.N) < connectivity
        W = W * mask
        np.fill_diagonal(W, 0)
        
        # Normalize spectral radius
        eigenvalues = np.linalg.eigvals(W)
        spectral_radius = np.max(np.abs(eigenvalues))
        if spectral_radius > 0:
            W = W / spectral_radius * 0.8
        
        return W
    
    def simulate(self, T, x0=None, intervention=None, return_trace=False):
        """
        Simulate WC dynamics.
        
        Args:
            T: Number of timesteps
            x0: Initial state (N×2), if None use random
            intervention: dict with keys 'nodes', 'amplitude', 'duration', 'start'
            return_trace: If True, return full trajectory
        
        Returns:
            x_final: Final state (N×2)
            trajectory: (T, N×2) if return_trace=True
        """
        if x0 is None:
            x0 = np.random.rand(self.N, 2) * 0.3 + 0.1
        
        x = x0.copy()
        sqrt_dt = np.sqrt(self.dt)
        
        if return_trace:
            trajectory = np.zeros((T, self.N, 2))
        
        for t in range(T):
            # Apply intervention if active
            u = np.zeros(self.N)
            if intervention is not None:
                t_start = intervention.get('start', 0)
                t_end = t_start + intervention['duration']
                if t_start <= t < t_end:
                    nodes = intervention['nodes']
                    amplitude = intervention['amplitude']
                    u[nodes] = amplitude
            
            # Wilson-Cowan equations
            xE = x[:, 0]
            xI = x[:, 1]
            
            # Global coupling
            IE_global = self.wEE * xE - self.wEI * xI + self.P
            II_global = self.wIE * xE - self.wII * xI
            
            # Network coupling
            IE_net = np.dot(self.W_net, xE)
            II_net = np.dot(self.W_net, xI)
            
            # Total input
            IE_total = IE_global + IE_net + u
            II_total = II_global + II_net
            
            # Sigmoid activation
            sE = 1.0 / (1.0 + np.exp(-1.2 * IE_total))
            sI = 1.0 / (1.0 + np.exp(-1.2 * II_total))
            
            # Dynamics
            dxE = (-xE + sE) / self.tau_E
            dxI = (-xI + sI) / self.tau_I
            
            # Euler-Maruyama
            xE += dxE * self.dt + self.sigma * sqrt_dt * np.random.randn(self.N)
            xI += dxI * self.dt + self.sigma * sqrt_dt * np.random.randn(self.N)
            
            # Clip
            x[:, 0] = np.clip(xE, 0, 1)
            x[:, 1] = np.clip(xI, 0, 1)
            
            if return_trace:
                trajectory[t] = x
        
        if return_trace:
            return x, trajectory
        return x
    
    def find_attractors(self, n_samples=100, T_settle=2000):
        """Find attractors via random initialization."""
        attractors = []
        
        for _ in range(n_samples):
            x0 = np.random.rand(self.N, 2)
            x_final = self.simulate(T_settle, x0=x0)
            
            # Check if new attractor
            is_new = True
            for att in attractors:
                if np.linalg.norm(x_final - att) < 0.05:
                    is_new = False
                    break
            
            if is_new:
                attractors.append(x_final)
        
        return attractors


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION: P_succ estimation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_intervention(wc, nodes, amplitude, duration, 
                          A_p, A_h, K=1000, threshold=0.15):
    """
    Estimate P_succ for given intervention.
    
    Returns:
        p_succ: Success probability
        ci_lower, ci_upper: Wilson score 95% CI
    """
    successes = 0
    
    for _ in range(K):
        # Start from pathological attractor
        x0 = A_p.copy() + np.random.randn(*A_p.shape) * 0.02
        
        # Apply intervention
        intervention = {
            'nodes': nodes,
            'amplitude': amplitude,
            'duration': int(duration / wc.dt),
            'start': 0
        }
        
        x_post = wc.simulate(int(duration / wc.dt) + 1000, x0=x0, 
                            intervention=intervention)
        
        # Check basin membership
        dist_h = np.linalg.norm(x_post - A_h)
        dist_p = np.linalg.norm(x_post - A_p)
        
        if dist_h < dist_p and dist_h < threshold:
            successes += 1
    
    p_succ = successes / K
    
    # Wilson score CI
    from scipy import stats as sp_stats
    if successes == 0:
        ci_lower, ci_upper = 0, 0.05
    elif successes == K:
        ci_lower, ci_upper = 0.95, 1.0
    else:
        ci_lower, ci_upper = sp_stats.binom.interval(0.95, K, p_succ)
        ci_lower /= K
        ci_upper /= K
    
    return p_succ, ci_lower, ci_upper


# ══════════════════════════════════════════════════════════════════════════════
# PACAN OPTIMIZER (Simplified from Week 6)
# ══════════════════════════════════════════════════════════════════════════════

def pacan_optimize(wc, A_p, A_h, I_sizes=[1,2,3,4,5], 
                   amplitudes=[0.05, 0.1, 0.15, 0.2],
                   duration=100, K_explore=80, K_final=1000):
    """
    PACAN optimization on empirical WC network.
    
    Returns:
        archive: List of (nodes, amplitude, p_succ, effort)
    """
    archive = []
    
    # Grid search over (|I|, a)
    for I_size in tqdm(I_sizes, desc="PACAN |I| sweep"):
        for amplitude in amplitudes:
            # Gibbs sampling for node selection
            best_nodes = None
            best_p = 0
            
            # Simple heuristic: select high-degree nodes
            # (Full Gibbs would take too long, use degree centrality)
            degrees = np.sum(np.abs(wc.W_net), axis=0)
            candidate_nodes = np.argsort(degrees)[-I_size:]
            
            # Evaluate
            p_succ, _, _ = evaluate_intervention(
                wc, candidate_nodes, amplitude, duration,
                A_p, A_h, K=K_explore
            )
            
            # Refine with K_final
            p_succ_final, ci_l, ci_u = evaluate_intervention(
                wc, candidate_nodes, amplitude, duration,
                A_p, A_h, K=K_final
            )
            
            effort = I_size * amplitude * duration
            
            archive.append({
                'nodes': candidate_nodes.tolist(),
                'amplitude': amplitude,
                'p_succ': p_succ_final,
                'ci_lower': ci_l,
                'ci_upper': ci_u,
                'effort': effort,
                'I_size': I_size,
            })
    
    return archive


def greedy_baseline(wc, A_p, A_h, I_max=5, amplitude=0.15, 
                   duration=100, K_eval=200):
    """Greedy forward selection baseline."""
    selected = []
    
    print(f"  Greedy baseline: selecting up to {I_max} nodes...")
    
    for i in range(I_max):
        best_node = None
        best_p = 0
        
        # Try adding each candidate node
        candidates = [n for n in range(wc.N) if n not in selected]
        
        for node in tqdm(candidates, desc=f"  Greedy step {i+1}/{I_max}", leave=False):
            candidate = selected + [node]
            p_succ, _, _ = evaluate_intervention(
                wc, candidate, amplitude, duration,
                A_p, A_h, K=K_eval
            )
            
            if p_succ > best_p:
                best_p = p_succ
                best_node = node
        
        if best_node is not None:
            selected.append(best_node)
            print(f"  Step {i+1}: Selected node {best_node}, P_succ={best_p:.3f}")
        else:
            print(f"  Step {i+1}: No improvement, stopping")
            break
    
    # Final eval with higher K
    print(f"  Final evaluation with K=1000...")
    p_succ, ci_l, ci_u = evaluate_intervention(
        wc, selected, amplitude, duration,
        A_p, A_h, K=1000
    )
    
    effort = len(selected) * amplitude * duration
    
    return {
        'nodes': selected,
        'amplitude': amplitude,
        'p_succ': p_succ,
        'ci_lower': ci_l,
        'ci_upper': ci_u,
        'effort': effort,
    }


def random_baseline(wc, A_p, A_h, I_size=3, amplitude=0.15,
                   duration=100, K_eval=200, n_trials=20):
    """Random search baseline."""
    best_result = None
    best_p = 0
    
    print(f"  Random search: {n_trials} trials...")
    
    for trial in tqdm(range(n_trials), desc="  Random trials"):
        nodes = np.random.choice(wc.N, I_size, replace=False)
        
        p_succ, ci_l, ci_u = evaluate_intervention(
            wc, nodes, amplitude, duration,
            A_p, A_h, K=K_eval
        )
        
        if p_succ > best_p:
            best_p = p_succ
            best_result = {
                'nodes': nodes.tolist(),
                'amplitude': amplitude,
                'p_succ': p_succ,
                'ci_lower': ci_l,
                'ci_upper': ci_u,
                'effort': I_size * amplitude * duration,
            }
    
    # Final eval with higher K
    print(f"  Best random: P_succ={best_p:.3f}, re-evaluating with K=1000...")
    p_succ_final, ci_l, ci_u = evaluate_intervention(
        wc, best_result['nodes'], amplitude, duration,
        A_p, A_h, K=1000
    )
    
    best_result['p_succ'] = p_succ_final
    best_result['ci_lower'] = ci_l
    best_result['ci_upper'] = ci_u
    
    return best_result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_patient_comparison(patient_id, wc_params_path, output_dir, N=15):
    """Run full PACAN vs baselines for one patient."""
    print(f"\n{'='*80}")
    print(f"PATIENT: {patient_id.upper()}")
    print(f"{'='*80}")
    
    # Load WC parameters
    with open(wc_params_path) as f:
        wc_params = json.load(f)
    
    print(f"Loaded WC params: P={wc_params['P']:.3f}, wEE={wc_params['wEE']:.3f}, "
          f"wEI={wc_params['wEI']:.3f}, sigma={wc_params['sigma']:.4f}")
    
    # Create WC network
    wc = WilsonCowanNetwork(N=N, wc_params=wc_params, seed=42)
    
    # Find attractors
    print("Finding attractors...")
    attractors = wc.find_attractors(n_samples=50, T_settle=2000)
    print(f"Found {len(attractors)} attractors")
    
    if len(attractors) < 2:
        print("Warning: Less than 2 attractors found. Using heuristic A_p and A_h.")
        A_p = np.ones((N, 2)) * 0.8  # High activity
        A_h = np.ones((N, 2)) * 0.15  # Low activity
    else:
        # Identify pathological (high E) vs healthy (low E)
        E_means = [np.mean(att[:, 0]) for att in attractors]
        idx_p = np.argmax(E_means)
        idx_h = np.argmin(E_means)
        A_p = attractors[idx_p]
        A_h = attractors[idx_h]
    
    print(f"A_p: E_mean={np.mean(A_p[:,0]):.3f}, A_h: E_mean={np.mean(A_h[:,0]):.3f}")
    
    # Run PACAN
    print("\nRunning PACAN...")
    pacan_archive = pacan_optimize(wc, A_p, A_h, 
                                   I_sizes=[1,2,3,4,5],
                                   amplitudes=[0.05, 0.1, 0.15, 0.2],
                                   duration=100,
                                   K_explore=50,  # Reduced from 80
                                   K_final=500)    # Reduced from 1000
    
    pacan_best = max(pacan_archive, key=lambda x: x['p_succ'])
    print(f"PACAN best: P_succ={pacan_best['p_succ']:.3f}, |I|={pacan_best['I_size']}, "
          f"a={pacan_best['amplitude']:.2f}")
    
    # Run Greedy
    print("\nRunning Greedy...")
    greedy_result = greedy_baseline(wc, A_p, A_h, I_max=5, amplitude=0.15, 
                                   duration=100, K_eval=200)  # Reduced from 1000
    print(f"Greedy: P_succ={greedy_result['p_succ']:.3f}, |I|={len(greedy_result['nodes'])}")
    
    # Run Random
    print("\nRunning Random Search...")
    random_result = random_baseline(wc, A_p, A_h, I_size=3, amplitude=0.15,
                                   duration=100, K_eval=200, n_trials=20)  # Reduced from 50 trials, 1000 K
    print(f"Random: P_succ={random_result['p_succ']:.3f}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results = {
        'patient_id': patient_id,
        'wc_params': wc_params,
        'pacan_archive': pacan_archive,
        'pacan_best': pacan_best,
        'greedy': greedy_result,
        'random': random_result,
    }
    
    output_path = output_dir / f"results_empirical_{patient_id}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {output_path}")
    
    return results


def visualize_all_patients(results_dir, patients=['chb01', 'chb02', 'chb03', 'chb05']):
    """Generate 4-panel comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, patient_id in enumerate(patients):
        ax = axes[idx]
        
        # Load results
        with open(Path(results_dir) / f"results_empirical_{patient_id}.json") as f:
            results = json.load(f)
        
        # Extract P_succ values
        pacan_p = results['pacan_best']['p_succ']
        greedy_p = results['greedy']['p_succ']
        random_p = results['random']['p_succ']
        
        # Bar plot
        methods = ['PACAN', 'Greedy', 'Random']
        p_values = [pacan_p, greedy_p, random_p]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        bars = ax.bar(methods, p_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Annotate delta
        delta = pacan_p - greedy_p
        ax.text(0.5, max(p_values) + 0.05, f'ΔP = +{delta:.3f}',
                ha='center', fontweight='bold', fontsize=10)
        
        ax.set_ylabel('P$_{succ}$', fontweight='bold', fontsize=11)
        ax.set_title(f'{patient_id.upper()}\nP={results["wc_params"]["P"]:.2f}, σ={results["wc_params"]["sigma"]:.3f}',
                     fontweight='bold', fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, axis='y', linestyle=':')
        
        # Add values on bars
        for bar, val in zip(bars, p_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{val:.3f}', ha='center', fontsize=9)
    
    plt.suptitle('Week 8 Phase 4: PACAN on Empirically-Calibrated WC Networks',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(results_dir) / 'week8_phase4_comparison.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Week 8 Phase 4: Empirical PACAN")
    parser.add_argument('--patient', type=str, help='Single patient (chb01, chb02, chb03, chb05)')
    parser.add_argument('--all_patients', action='store_true', help='Run all 4 patients')
    parser.add_argument('--visualize', action='store_true', help='Generate comparison plot')
    parser.add_argument('--wc_params_dir', type=str, default='results_w8',
                       help='Directory with wc_params_*.json files')
    parser.add_argument('--output_dir', type=str, default='results_w8_phase4',
                       help='Output directory for results')
    parser.add_argument('--N', type=int, default=15, help='Network size')
    
    args = parser.parse_args()
    
    if args.all_patients:
        patients = ['chb01', 'chb02', 'chb03', 'chb05']
        for patient_id in patients:
            wc_params_path = Path(args.wc_params_dir) / f'wc_params_{patient_id}.json'
            run_patient_comparison(patient_id, wc_params_path, args.output_dir, N=args.N)
    
    elif args.patient:
        wc_params_path = Path(args.wc_params_dir) / f'wc_params_{args.patient}.json'
        run_patient_comparison(args.patient, wc_params_path, args.output_dir, N=args.N)
    
    elif args.visualize:
        visualize_all_patients(args.output_dir)
    
    else:
        print("Use --all_patients, --patient <id>, or --visualize")


if __name__ == "__main__":
    main()
