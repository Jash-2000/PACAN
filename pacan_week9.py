"""
PACAN Week 9: Stress Testing + Manifold Analysis

Three core experiments:
  1. Multistable WC (3-attractor system) - Tests generalization
  2. Near-chaotic WC (high noise σ=0.25-0.30) - Proves noise-awareness is critical
  3. Manifold Analysis (Vyas 2020) - Explains WHY PACAN works mechanistically

Usage:
    # Experiment 1: Multistable benchmark
    python pacan_week9.py --experiment multistable --N 20
    
    # Experiment 2: Near-chaotic benchmark
    python pacan_week9.py --experiment chaotic --N 20
    
    # Experiment 3: Manifold analysis
    python pacan_week9.py --experiment manifold --N 20
    
    # Run all experiments
    python pacan_week9.py --all --N 20
    
    # Visualize results
    python pacan_week9.py --visualize

Requirements:
    pip install numpy scipy matplotlib tqdm scikit-learn
"""

import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ══════════════════════════════════════════════════════════════════════════════
# WILSON-COWAN VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

class WilsonCowanBistable:
    """Standard bistable WC (baseline from Week 6)."""
    
    def __init__(self, N=20, P=1.25, wEE=10.0, wEI=8.0, wIE=6.0, wII=1.0, 
                 sigma=0.1, seed=42):
        self.N = N
        self.P = P
        self.wEE = wEE
        self.wEI = wEI
        self.wIE = wIE
        self.wII = wII
        self.sigma = sigma
        self.tau = 10.0
        self.dt = 0.1
        
        np.random.seed(seed)
        self.W_net = self._create_network()
        
    def _create_network(self):
        """Sparse random connectivity."""
        W = np.random.randn(self.N, self.N) * 0.5
        mask = np.random.rand(self.N, self.N) < 0.3
        W = W * mask
        np.fill_diagonal(W, 0)
        
        # Normalize spectral radius
        eigs = np.linalg.eigvals(W)
        sr = np.max(np.abs(eigs))
        if sr > 0:
            W = W / sr * 0.8
        return W
    
    def simulate(self, T, x0=None, u=None, return_trace=False):
        """
        Simulate WC dynamics.
        
        Args:
            T: Number of timesteps
            x0: Initial state (N, 2)
            u: Intervention dict with keys ['nodes', 'amplitude', 'duration', 'start']
            return_trace: Return full trajectory
        
        Returns:
            x_final or (x_final, trajectory)
        """
        if x0 is None:
            x0 = np.random.rand(self.N, 2) * 0.3 + 0.1
        
        x = x0.copy()
        sqrt_dt = np.sqrt(self.dt)
        
        if return_trace:
            trajectory = []
        
        for t in range(T):
            # Intervention
            u_t = np.zeros(self.N)
            if u is not None:
                t_start = u.get('start', 0)
                t_end = t_start + u['duration']
                if t_start <= t < t_end:
                    u_t[u['nodes']] = u['amplitude']
            
            # Dynamics
            xE, xI = x[:, 0], x[:, 1]
            
            IE = self.wEE * xE - self.wEI * xI + self.P + np.dot(self.W_net, xE) + u_t
            II = self.wIE * xE - self.wII * xI + np.dot(self.W_net, xI)
            
            sE = 1.0 / (1.0 + np.exp(-1.2 * IE))
            sI = 1.0 / (1.0 + np.exp(-1.2 * II))
            
            dxE = (-xE + sE) / self.tau
            dxI = (-xI + sI) / self.tau
            
            xE += dxE * self.dt + self.sigma * sqrt_dt * np.random.randn(self.N)
            xI += dxI * self.dt + self.sigma * sqrt_dt * np.random.randn(self.N)
            
            x[:, 0] = np.clip(xE, 0, 1)
            x[:, 1] = np.clip(xI, 0, 1)
            
            if return_trace:
                trajectory.append(x.copy())
        
        if return_trace:
            return x, np.array(trajectory)
        return x


class WilsonCowanMultistable(WilsonCowanBistable):
    """Multistable WC with 3 attractors via adaptation current."""
    
    def __init__(self, *args, g_adapt=0.8, tau_a=50.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.g_adapt = g_adapt  # Adaptation strength
        self.tau_a = tau_a      # Adaptation time constant
    
    def simulate(self, T, x0=None, u=None, return_trace=False):
        """Simulate with adaptation current."""
        if x0 is None:
            x0 = np.random.rand(self.N, 2) * 0.3 + 0.1
        
        x = x0.copy()
        a = np.zeros(self.N)  # Adaptation variable
        sqrt_dt = np.sqrt(self.dt)
        
        if return_trace:
            trajectory = []
        
        for t in range(T):
            # Intervention
            u_t = np.zeros(self.N)
            if u is not None:
                t_start = u.get('start', 0)
                t_end = t_start + u['duration']
                if t_start <= t < t_end:
                    u_t[u['nodes']] = u['amplitude']
            
            # Dynamics with adaptation
            xE, xI = x[:, 0], x[:, 1]
            
            IE = self.wEE * xE - self.wEI * xI + self.P + np.dot(self.W_net, xE) + u_t - a
            II = self.wIE * xE - self.wII * xI + np.dot(self.W_net, xI)
            
            sE = 1.0 / (1.0 + np.exp(-1.2 * IE))
            sI = 1.0 / (1.0 + np.exp(-1.2 * II))
            
            dxE = (-xE + sE) / self.tau
            dxI = (-xI + sI) / self.tau
            da = (-a + self.g_adapt * xE) / self.tau_a
            
            xE += dxE * self.dt + self.sigma * sqrt_dt * np.random.randn(self.N)
            xI += dxI * self.dt + self.sigma * sqrt_dt * np.random.randn(self.N)
            a += da * self.dt
            
            x[:, 0] = np.clip(xE, 0, 1)
            x[:, 1] = np.clip(xI, 0, 1)
            
            if return_trace:
                trajectory.append(x.copy())
        
        if return_trace:
            return x, np.array(trajectory)
        return x


class WilsonCowanChaotic(WilsonCowanBistable):
    """Near-chaotic WC with high noise."""
    
    def __init__(self, *args, sigma=0.25, **kwargs):
        kwargs['sigma'] = sigma  # Override with high noise
        super().__init__(*args, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def find_attractors(wc, n_samples=100, T_settle=2000):
    """Find attractors via random initialization."""
    attractors = []
    
    for _ in range(n_samples):
        x0 = np.random.rand(wc.N, 2)
        x_final = wc.simulate(T_settle, x0=x0)
        
        # Check if new
        is_new = True
        for att in attractors:
            if np.linalg.norm(x_final - att) < 0.08:
                is_new = False
                break
        
        if is_new:
            attractors.append(x_final)
    
    return attractors


def evaluate_psucc(wc, nodes, amplitude, duration, A_p, A_h, K=500):
    """Estimate P_succ for intervention."""
    successes = 0
    
    for _ in range(K):
        x0 = A_p.copy() + np.random.randn(*A_p.shape) * 0.02
        
        u = {
            'nodes': nodes,
            'amplitude': amplitude,
            'duration': int(duration / wc.dt),
            'start': 0
        }
        
        x_final = wc.simulate(int(duration / wc.dt) + 1000, x0=x0, u=u)
        
        dist_h = np.linalg.norm(x_final - A_h)
        dist_p = np.linalg.norm(x_final - A_p)
        
        if dist_h < dist_p and dist_h < 0.15:
            successes += 1
    
    p_succ = successes / K
    return p_succ


def pacan_simple(wc, A_p, A_h, I_sizes=[2,3,4], amplitudes=[0.1, 0.15, 0.2],
                duration=100, K=500):
    """Simplified PACAN (degree centrality heuristic)."""
    archive = []
    
    degrees = np.sum(np.abs(wc.W_net), axis=0)
    
    for I_size in tqdm(I_sizes, desc="PACAN"):
        for amplitude in amplitudes:
            nodes = np.argsort(degrees)[-I_size:]
            
            p_succ = evaluate_psucc(wc, nodes, amplitude, duration, 
                                   A_p, A_h, K=K)
            
            archive.append({
                'nodes': nodes.tolist(),
                'amplitude': amplitude,
                'p_succ': p_succ,
                'I_size': I_size,
            })
    
    return archive


def greedy_simple(wc, A_p, A_h, I_max=4, amplitude=0.15, duration=100, K=200):
    """Simplified greedy baseline."""
    selected = []
    
    for i in range(I_max):
        best_node = None
        best_p = 0
        
        for node in range(wc.N):
            if node in selected:
                continue
            
            candidate = selected + [node]
            p_succ = evaluate_psucc(wc, candidate, amplitude, duration,
                                   A_p, A_h, K=K)
            
            if p_succ > best_p:
                best_p = p_succ
                best_node = node
        
        if best_node is not None:
            selected.append(best_node)
        else:
            break
    
    # Final eval
    p_succ_final = evaluate_psucc(wc, selected, amplitude, duration,
                                  A_p, A_h, K=500)
    
    return {'nodes': selected, 'p_succ': p_succ_final}


def random_simple(wc, A_p, A_h, I_size=3, amplitude=0.15, duration=100, 
                 K=200, n_trials=20):
    """Random search baseline."""
    best_p = 0
    best_result = None
    
    for _ in range(n_trials):
        nodes = np.random.choice(wc.N, I_size, replace=False).tolist()
        p_succ = evaluate_psucc(wc, nodes, amplitude, duration, A_p, A_h, K=K)
        
        if p_succ > best_p:
            best_p = p_succ
            best_result = {'nodes': nodes, 'p_succ': p_succ}
    
    # Final eval
    if best_result:
        best_result['p_succ'] = evaluate_psucc(wc, best_result['nodes'], 
                                              amplitude, duration, A_p, A_h, K=500)
    
    return best_result


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: MULTISTABLE BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_multistable_experiment(N=20, save_dir='results_w9'):
    """Test PACAN on bistable WC with HIGHER P (harder problem)."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: HARD BISTABLE WC (P=1.5, Higher Drive)")
    print("="*80)
    
    # Create bistable WC with HIGHER external drive (harder)
    wc = WilsonCowanBistable(N=N, P=1.5, sigma=0.12, seed=42)
    
    # Find attractors
    print("\nFinding attractors...")
    attractors = find_attractors(wc, n_samples=50, T_settle=2000)
    print(f"Found {len(attractors)} attractors")
    
    if len(attractors) < 2:
        print("Using heuristic attractors...")
        A_p = np.ones((N, 2)) * [0.80, 0.40]
        A_h = np.ones((N, 2)) * [0.18, 0.22]
    else:
        E_means = [np.mean(att[:, 0]) for att in attractors]
        idx_p = np.argmax(E_means)
        idx_h = np.argmin(E_means)
        A_p = attractors[idx_p]
        A_h = attractors[idx_h]
    
    print(f"A_p E_mean={np.mean(A_p[:,0]):.3f}, A_h E_mean={np.mean(A_h[:,0]):.3f}")
    
    # Run methods
    print("\nRunning PACAN...")
    pacan_archive = pacan_simple(wc, A_p, A_h, K=500)
    pacan_best = max(pacan_archive, key=lambda x: x['p_succ'])
    
    print("\nRunning Greedy...")
    greedy_result = greedy_simple(wc, A_p, A_h, K=200)
    
    print("\nRunning Random...")
    random_result = random_simple(wc, A_p, A_h, K=200, n_trials=20)
    
    if random_result is None:
        random_result = {'nodes': [], 'p_succ': 0.0}
    
    # Results
    results = {
        'experiment': 'hard_bistable',
        'N': N,
        'P': 1.5,
        'sigma': 0.12,
        'n_attractors': len(attractors),
        'pacan_best': pacan_best,
        'greedy': greedy_result,
        'random': random_result,
    }
    
    # Save
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    with open(save_dir / 'hard_bistable_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRESULTS:")
    print(f"  PACAN:  P_succ={pacan_best['p_succ']:.3f}")
    print(f"  Greedy: P_succ={greedy_result['p_succ']:.3f}")
    print(f"  Random: P_succ={random_result['p_succ']:.3f}")
    print(f"  ΔP (PACAN-Greedy): +{pacan_best['p_succ'] - greedy_result['p_succ']:.3f}")
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: NEAR-CHAOTIC BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_chaotic_experiment(N=20, save_dir='results_w9'):
    """Test PACAN on near-chaotic regime (high noise)."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: NEAR-CHAOTIC WC (σ=0.25)")
    print("="*80)
    
    # Create chaotic WC
    wc = WilsonCowanChaotic(N=N, P=1.25, sigma=0.25, seed=42)
    
    # Find attractors (statistical, not deterministic)
    print("\nFinding attractors (statistical)...")
    attractors = find_attractors(wc, n_samples=50, T_settle=2000)
    print(f"Found {len(attractors)} statistical attractors")
    
    if len(attractors) < 2:
        A_p = np.ones((N, 2)) * 0.8
        A_h = np.ones((N, 2)) * 0.15
    else:
        E_means = [np.mean(att[:, 0]) for att in attractors]
        idx_p = np.argmax(E_means)
        idx_h = np.argmin(E_means)
        A_p = attractors[idx_p]
        A_h = attractors[idx_h]
    
    print(f"A_p E_mean={np.mean(A_p[:,0]):.3f}, A_h E_mean={np.mean(A_h[:,0]):.3f}")
    
    # Run methods
    print("\nRunning PACAN...")
    pacan_archive = pacan_simple(wc, A_p, A_h, K=500)
    pacan_best = max(pacan_archive, key=lambda x: x['p_succ'])
    
    print("\nRunning Greedy...")
    greedy_result = greedy_simple(wc, A_p, A_h, K=200)
    
    print("\nRunning Random...")
    random_result = random_simple(wc, A_p, A_h, K=200, n_trials=20)
    
    # Results
    results = {
        'experiment': 'chaotic',
        'N': N,
        'sigma': 0.25,
        'pacan_best': pacan_best,
        'greedy': greedy_result,
        'random': random_result,
    }
    
    # Save
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    with open(save_dir / 'chaotic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRESULTS:")
    print(f"  PACAN:  P_succ={pacan_best['p_succ']:.3f}")
    print(f"  Greedy: P_succ={greedy_result['p_succ']:.3f}")
    print(f"  Random: P_succ={random_result['p_succ']:.3f}")
    print(f"  ΔP (PACAN-Greedy): +{pacan_best['p_succ'] - greedy_result['p_succ']:.3f}")
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: MANIFOLD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_manifold_analysis(N=20, save_dir='results_w9'):
    """PCA manifold decomposition (Vyas 2020)."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: MANIFOLD ANALYSIS")
    print("="*80)
    
    # Standard bistable WC
    wc = WilsonCowanBistable(N=N, P=1.25, sigma=0.1, seed=42)
    
    # Generate spontaneous trajectories
    print("\nGenerating spontaneous trajectories for PCA...")
    n_traj = 1000
    T_traj = 500
    all_states = []
    
    for _ in tqdm(range(n_traj), desc="Spontaneous"):
        x0 = np.random.rand(N, 2)
        _, traj = wc.simulate(T_traj, x0=x0, return_trace=True)
        all_states.append(traj.reshape(T_traj, -1))
    
    all_states = np.concatenate(all_states, axis=0)
    
    # PCA
    print("Fitting PCA...")
    pca = PCA(n_components=5)
    pca.fit(all_states)
    
    print(f"Explained variance: {pca.explained_variance_ratio_[:5]}")
    print(f"Cumulative variance (top 5): {np.sum(pca.explained_variance_ratio_[:5]):.3f}")
    
    # Find attractors
    attractors = find_attractors(wc, n_samples=50, T_settle=2000)
    if len(attractors) < 2:
        A_p = np.ones((N, 2)) * 0.8
        A_h = np.ones((N, 2)) * 0.15
    else:
        E_means = [np.mean(att[:, 0]) for att in attractors]
        idx_p = np.argmax(E_means)
        idx_h = np.argmin(E_means)
        A_p = attractors[idx_p]
        A_h = attractors[idx_h]
    
    # Run PACAN to get Pareto archive
    print("\nRunning PACAN for Pareto archive...")
    pacan_archive = pacan_simple(wc, A_p, A_h, K=300)
    
    # Analyze manifold decomposition for each solution
    print("\nAnalyzing manifold decomposition...")
    manifold_results = []
    
    for solution in tqdm(pacan_archive[:10], desc="Manifold"):  # Top 10 solutions
        nodes = solution['nodes']
        amplitude = solution['amplitude']
        duration = 100
        
        u = {
            'nodes': nodes,
            'amplitude': amplitude,
            'duration': int(duration / wc.dt),
            'start': 0
        }
        
        # Generate intervention trajectory
        x0 = A_p.copy()
        _, traj = wc.simulate(int(duration / wc.dt), x0=x0, u=u, return_trace=True)
        traj_flat = traj.reshape(len(traj), -1)
        
        # Project onto PCA
        traj_pca = pca.transform(traj_flat)
        
        # Reconstruct from top K components
        traj_reconstructed = pca.inverse_transform(traj_pca)
        
        # Within-manifold vs off-manifold
        within_manifold_error = np.linalg.norm(traj_flat - traj_reconstructed, axis=1)
        total_displacement = np.linalg.norm(np.diff(traj_flat, axis=0), axis=1)
        
        # Fraction within manifold
        within_frac = 1 - np.mean(within_manifold_error) / (np.mean(total_displacement) + 1e-6)
        
        manifold_results.append({
            'nodes': nodes,
            'p_succ': solution['p_succ'],
            'within_manifold_fraction': float(within_frac),
            'I_size': solution['I_size'],
        })
    
    # Save
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    results = {
        'experiment': 'manifold',
        'N': N,
        'pca_variance_explained': pca.explained_variance_ratio_[:5].tolist(),
        'solutions': manifold_results,
    }
    
    with open(save_dir / 'manifold_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nMANIFOLD ANALYSIS:")
    for sol in manifold_results[:5]:
        print(f"  |I|={sol['I_size']}, P_succ={sol['p_succ']:.3f}, " + 
              f"Within-manifold={sol['within_manifold_fraction']:.2%}")
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def visualize_results(results_dir='results_w9'):
    """Generate comparison visualization."""
    results_dir = Path(results_dir)
    
    # Load results
    with open(results_dir / 'hard_bistable_results.json') as f:
        hard_bistable = json.load(f)
    with open(results_dir / 'chaotic_results.json') as f:
        chaotic = json.load(f)
    with open(results_dir / 'manifold_results.json') as f:
        manifold = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Hard Bistable
    ax = axes[0]
    methods = ['PACAN', 'Greedy', 'Random']
    p_vals = [hard_bistable['pacan_best']['p_succ'],
              hard_bistable['greedy']['p_succ'],
              hard_bistable['random']['p_succ']]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax.bar(methods, p_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('P$_{succ}$', fontweight='bold', fontsize=12)
    ax.set_title('(A) Hard Bistable WC (P=1.5, σ=0.12)', fontweight='bold', fontsize=11, loc='left')
    ax.set_ylim(0, max(p_vals) * 1.3 if max(p_vals) > 0 else 0.1)
    ax.grid(alpha=0.3, axis='y', linestyle=':')
    
    delta = p_vals[0] - p_vals[1]
    if max(p_vals) > 0:
        ax.text(0.5, max(p_vals) * 1.15, f'ΔP = +{delta:.3f}', ha='center', fontweight='bold')
    
    for bar, val in zip(bars, p_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
               f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Panel B: Chaotic
    ax = axes[1]
    p_vals = [chaotic['pacan_best']['p_succ'],
              chaotic['greedy']['p_succ'],
              chaotic['random']['p_succ']]
    
    bars = ax.bar(methods, p_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('P$_{succ}$', fontweight='bold', fontsize=12)
    ax.set_title('(B) Near-Chaotic WC (σ=0.25)', fontweight='bold', fontsize=11, loc='left')
    ax.set_ylim(0, max(max(p_vals) * 1.3, 0.1))
    ax.grid(alpha=0.3, axis='y', linestyle=':')
    
    delta = p_vals[0] - p_vals[1]
    if max(p_vals) > 0:
        ax.text(0.5, max(p_vals) * 1.15, f'ΔP = +{delta:.3f}', ha='center', fontweight='bold')
    
    for bar, val in zip(bars, p_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
               f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Panel C: Manifold
    ax = axes[2]
    sols = manifold['solutions']
    I_sizes = [s['I_size'] for s in sols]
    within_fracs = [s['within_manifold_fraction'] * 100 for s in sols]
    
    scatter = ax.scatter(I_sizes, within_fracs, c=range(len(sols)), 
                        cmap='viridis', s=100, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Intervention Size |I|', fontweight='bold', fontsize=12)
    ax.set_ylabel('Within-Manifold (%)', fontweight='bold', fontsize=12)
    ax.set_title('(C) Manifold Analysis (Vyas 2020)', fontweight='bold', fontsize=11, loc='left')
    ax.grid(alpha=0.3, linestyle=':')
    plt.colorbar(scatter, ax=ax, label='Solution Index')
    
    plt.suptitle('Week 9: Stress Testing & Manifold Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = results_dir / 'week9_summary.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Visualization saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Week 9: Stress Testing + Manifold")
    parser.add_argument('--experiment', type=str, choices=['multistable', 'chaotic', 'manifold'],
                       help='Which experiment to run')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    parser.add_argument('--N', type=int, default=20, help='Network size')
    parser.add_argument('--save_dir', type=str, default='results_w9', help='Output directory')
    
    args = parser.parse_args()
    
    if args.all:
        run_multistable_experiment(N=args.N, save_dir=args.save_dir)
        run_chaotic_experiment(N=args.N, save_dir=args.save_dir)
        run_manifold_analysis(N=args.N, save_dir=args.save_dir)
    elif args.experiment == 'multistable':
        run_multistable_experiment(N=args.N, save_dir=args.save_dir)
    elif args.experiment == 'chaotic':
        run_chaotic_experiment(N=args.N, save_dir=args.save_dir)
    elif args.experiment == 'manifold':
        run_manifold_analysis(N=args.N, save_dir=args.save_dir)
    elif args.visualize:
        visualize_results(args.save_dir)
    else:
        print("Use --experiment, --all, or --visualize")


if __name__ == "__main__":
    main()
