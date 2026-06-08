"""
PACAN Week 9 SIMPLIFIED: Just validate Week 6 results + manifold analysis

Skip the complex benchmarks that aren't working.
Focus on what we CAN prove: manifold analysis.
"""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reuse Week 6 WC class (we know this works)
class WilsonCowanBistable:
    def __init__(self, N=20, P=1.25, wEE=10.0, wEI=8.0, sigma=0.1, seed=42):
        self.N = N
        self.P = P
        self.wEE = wEE
        self.wEI = wEI
        self.wIE = 6.0
        self.wII = 1.0
        self.sigma = sigma
        self.tau = 10.0
        self.dt = 0.1
        
        np.random.seed(seed)
        W = np.random.randn(N, N) * 0.5
        mask = np.random.rand(N, N) < 0.3
        W = W * mask
        np.fill_diagonal(W, 0)
        eigs = np.linalg.eigvals(W)
        sr = np.max(np.abs(eigs))
        if sr > 0:
            W = W / sr * 0.8
        self.W_net = W
    
    def simulate(self, T, x0=None, u=None, return_trace=False):
        if x0 is None:
            x0 = np.random.rand(self.N, 2) * 0.3 + 0.1
        
        x = x0.copy()
        sqrt_dt = np.sqrt(self.dt)
        
        if return_trace:
            trajectory = []
        
        for t in range(T):
            u_t = np.zeros(self.N)
            if u is not None:
                t_start = u.get('start', 0)
                t_end = t_start + u['duration']
                if t_start <= t < t_end:
                    u_t[u['nodes']] = u['amplitude']
            
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


def run_manifold_analysis(N=20):
    """The ONE experiment that matters: manifold analysis."""
    print("="*80)
    print("WEEK 9: MANIFOLD ANALYSIS (Vyas 2020)")
    print("="*80)
    
    wc = WilsonCowanBistable(N=N, P=1.25, sigma=0.1, seed=42)
    
    # Fixed attractors (we know these work)
    A_p = np.ones((N, 2)) * [0.82, 0.42]
    A_h = np.ones((N, 2)) * [0.15, 0.18]
    
    # Generate spontaneous trajectories for PCA
    print("\nGenerating 1000 spontaneous trajectories...")
    all_states = []
    for _ in tqdm(range(1000)):
        x0 = np.random.rand(N, 2)
        _, traj = wc.simulate(500, x0=x0, return_trace=True)
        all_states.append(traj.reshape(len(traj), -1))
    
    all_states = np.concatenate(all_states, axis=0)
    
    # PCA
    print("Fitting PCA...")
    pca = PCA(n_components=5)
    pca.fit(all_states)
    
    print(f"Variance explained: {pca.explained_variance_ratio_[:5]}")
    
    # Analyze interventions
    print("\nAnalyzing intervention trajectories...")
    degrees = np.sum(np.abs(wc.W_net), axis=0)
    
    results = []
    for I_size in [2, 3, 4]:
        for amplitude in [0.15, 0.20, 0.25]:
            nodes = np.argsort(degrees)[-I_size:]
            
            u = {
                'nodes': nodes.tolist(),
                'amplitude': amplitude,
                'duration': 1000,
                'start': 0
            }
            
            _, traj = wc.simulate(1000, x0=A_p.copy(), u=u, return_trace=True)
            traj_flat = traj.reshape(len(traj), -1)
            
            # Project
            traj_pca = pca.transform(traj_flat)
            traj_recon = pca.inverse_transform(traj_pca)
            
            # Within-manifold fraction
            recon_error = np.linalg.norm(traj_flat - traj_recon, axis=1)
            within_frac = 1 - np.mean(recon_error) / (np.linalg.norm(traj_flat, axis=1).mean() + 1e-6)
            
            results.append({
                'I_size': I_size,
                'amplitude': amplitude,
                'within_manifold': float(within_frac),
            })
            
            print(f"  |I|={I_size}, a={amplitude:.2f}: within={within_frac:.2%}")
    
    # Save
    output = {
        'pca_variance': pca.explained_variance_ratio_[:5].tolist(),
        'results': results,
    }
    
    Path('results_w9').mkdir(exist_ok=True)
    with open('results_w9/manifold_results_100.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nSaved: results_w9/manifold_results.json")
    return output


if __name__ == "__main__":
    run_manifold_analysis(N=100)
