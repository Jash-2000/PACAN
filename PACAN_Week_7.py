"""
PACAN Week 7 — Energy Landscape + Push vs Reshape

Implements:
  1. Basin volume analysis (fraction of random inits converging to each attractor)
  2. ΔE_barrier estimation (Monte Carlo kick method)
  3. MFPT + Kramers law validation (MFPT ≈ C·exp(A/σ²))
  4. Push vs Reshape comparison at matched energy budget
  
Theoretical foundations:
  - Ezaki et al. 2018: Energy landscape from pairwise MEM
  - Gu et al. 2015: Network control energy framework
  - Kramers 1940: Barrier crossing theory
  - Vyas et al. 2020: Push vs reshape (manifold perspective)

Usage:
    python pacan_week7.py --N 15 --model_seed 0
    python pacan_week7.py --N 15 --push_reshape  # includes push vs reshape
    python pacan_week7.py --quick                 # reduced n_inits for testing

Requirements:
    pip install torch numpy scipy matplotlib tqdm
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from tqdm import tqdm
import json, os, argparse, warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# WC MODEL (from Week 6)
# ══════════════════════════════════════════════════════════════════════════════

class WCParams:
    def __init__(self, N=10, wEE=10., wEI=8., wIE=6., wII=1.,
                 w_net_scale=0.15, sparsity=0.70,
                 P=1.25, a=1.2, dt=0.1, model_seed=0):
        self.N=N; self.wEE=wEE; self.wEI=wEI
        self.wIE=wIE; self.wII=wII
        self.P=P; self.a=a; self.dt=dt
        self.w_net_scale=w_net_scale; self.sparsity=sparsity
        self.model_seed=model_seed
        rng=np.random.RandomState(model_seed)
        W=rng.randn(N,N)*w_net_scale
        mask=(rng.rand(N,N)>sparsity).astype(float)
        np.fill_diagonal(mask,0)
        self.W_net=torch.tensor(W*mask,dtype=torch.float32)


def find_single_node_fps(P=1.25,wEE=10.,wEI=8.,wIE=6.,wII=1.,a=1.2,n=400):
    def f(t,x):
        s=lambda z:1./(1.+np.exp(-a*z))
        return [(-x[0]+s(wEE*x[0]-wEI*x[1]+P))/10.,
                (-x[1]+s(wIE*x[0]-wII*x[1]))/10.]
    fps=[]; rng=np.random.RandomState(0)
    for _ in range(n):
        sol=solve_ivp(f,[0,600],rng.rand(2),method='RK45',rtol=1e-8,atol=1e-10)
        xf=np.clip(sol.y[:,-1],0,1)
        new=True
        for fp in fps:
            if np.linalg.norm(xf-fp)<0.02: new=False; break
        if new: fps.append(xf)
    return sorted(fps,key=lambda x:x[0])


class SimpleWC:
    """Simplified WC for landscape analysis (no three-phase, just drift)."""
    def __init__(self, params, device=DEVICE):
        self.p=params; self.device=device
        self.W_net=params.W_net.to(device)

    def sigmoid(self,x): return torch.sigmoid(self.p.a*x)

    def drift(self,x,u=None,reshape=None):
        """x: (batch,N,2), u: (batch,N,2) or None, reshape: dict or None."""
        dtype = x.dtype
        device = x.device

        W_net = self.W_net.to(dtype=dtype, device=device)
        
        if u is None: u=torch.zeros_like(x)
        xE=x[...,0]; xI=x[...,1]; uE=u[...,0]; uI=u[...,1]
        wEI=self.p.wEI+(reshape.get('dwEI',0.) if reshape else 0.)
        wEE=self.p.wEE+(reshape.get('dwEE',0.) if reshape else 0.)
        tauI=10.*(1.+(reshape.get('dtauI_frac',0.) if reshape else 0.))
        inter=torch.matmul(xE,self.W_net.T)
        IE=wEE*xE-wEI*xI+inter+uE+self.p.P
        II=self.p.wIE*xE-self.p.wII*xI+uI
        dxE=(-xE+self.sigmoid(IE))/10.
        dxI=(-xI+self.sigmoid(II))/tauI
        return torch.stack([dxE,dxI],dim=-1)

    @torch.no_grad()
    def rollout_deterministic(self,x0,T=2000,u=None,reshape=None):
        """Deterministic rollout (sigma=0) for basin analysis."""
        x=x0.clone()
        if u is None: u=torch.zeros_like(x)
        for _ in range(T):
            x=(x+self.drift(x,u,reshape)*self.p.dt).clamp(0.,1.)
        return x

    @torch.no_grad()
    def rollout_stochastic(self,x0,sigma,T=2000,u=None,reshape=None,seed=0):
        """Stochastic rollout (Euler-Maruyama)."""
        x=x0.clone(); rng=np.random.RandomState(seed)
        if u is None: u=torch.zeros_like(x)
        sqrt_dt=self.p.dt**0.5
        for _ in range(T):
            noise=torch.tensor(rng.randn(*x.shape).astype(np.float32),
                               device=self.device)
            x=(x+self.drift(x,u,reshape)*self.p.dt
               +sigma*sqrt_dt*noise).clamp(0.,1.)
        return x


def make_network_attractors(params):
    fps=find_single_node_fps(P=params.P,wEE=params.wEE,wEI=params.wEI,
                              wIE=params.wIE,wII=params.wII,a=params.a)
    assert len(fps)>=2, f'Not bistable: {fps}'
    xE_h,xI_h=fps[0]; xE_p,xI_p=fps[-1]
    Ap=np.tile([xE_p,xI_p],(params.N,1)).astype(np.float32)
    Ah=np.tile([xE_h,xI_h],(params.N,1)).astype(np.float32)
    wc=SimpleWC(params,device=DEVICE)
    for Arr in [Ap,Ah]:
        x=torch.tensor(Arr[None,:,:],device=DEVICE)
        u=torch.zeros_like(x)
        for _ in range(3000):
            x=(x+wc.drift(x,u)*params.dt).clamp(0.,1.)
        Arr[:]=x[0].cpu().numpy()
    return Ap,Ah


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASIN VOLUME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def basin_volume_analysis(wc, Ap, Ah, n_inits=10000, T=2000):
    """
    Sample random initial conditions, run deterministic dynamics, classify endpoint.
    
    Returns:
        dict with frac_Ap, frac_Ah, frac_other
    """
    print(f"\n[Basin Volume] Sampling {n_inits} random initial conditions...")
    
    rng = np.random.RandomState(42)
    inits = rng.rand(n_inits, wc.p.N, 2).astype(np.float32)
    x0 = torch.tensor(inits, device=DEVICE)
    
    # Run deterministic rollout
    x_final = wc.rollout_deterministic(x0, T=T)
    
    # Classify endpoints
    Ap_t = torch.tensor(Ap, dtype=torch.float32, device=DEVICE)
    Ah_t = torch.tensor(Ah, dtype=torch.float32, device=DEVICE)
    
    dist_Ap = ((x_final - Ap_t.unsqueeze(0))**2).sum(dim=(1,2))
    dist_Ah = ((x_final - Ah_t.unsqueeze(0))**2).sum(dim=(1,2))
    
    in_Ap = (dist_Ap < dist_Ah).sum().item()
    in_Ah = (dist_Ah < dist_Ap).sum().item()
    in_other = n_inits - in_Ap - in_Ah
    
    results = {
        'frac_Ap': in_Ap / n_inits,
        'frac_Ah': in_Ah / n_inits,
        'frac_other': in_other / n_inits,
        'n_Ap': in_Ap,
        'n_Ah': in_Ah,
        'n_other': in_other,
    }
    
    print(f"  Basin(Ap): {results['frac_Ap']*100:.1f}%")
    print(f"  Basin(Ah): {results['frac_Ah']*100:.1f}%")
    print(f"  Other:     {results['frac_other']*100:.1f}%")
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. BARRIER HEIGHT ESTIMATION (Monte Carlo Kick)
# ══════════════════════════════════════════════════════════════════════════════

def barrier_height_montecarlo(wc, Ap, Ah, delta_vals=None, n_trials=200, T=2000):
    """
    Monte Carlo barrier estimation (Ezaki et al. 2018 method).
    Apply uniform random kick of magnitude delta from Ap, measure fraction
    reaching Ah under deterministic dynamics.
    
    ΔE_barrier ≈ delta_50 (50% crossing threshold).
    
    Args:
        delta_vals: array of kick magnitudes to test
        n_trials: number of random kick trials per delta
        T: deterministic rollout length
    
    Returns:
        dict with delta_vals, crossing_fractions, barrier_estimate
    """
    if delta_vals is None:
        delta_vals = np.linspace(0, 10.0, 101)
    
    print(f"\n[Barrier Height] Testing {len(delta_vals)} kick magnitudes...")
    
    Ap_t = torch.tensor(Ap, dtype=torch.float32, device=DEVICE)
    Ah_t = torch.tensor(Ah, dtype=torch.float32, device=DEVICE)
    
    crossing_fracs = []
    
    for delta in tqdm(delta_vals, desc="Delta sweep", leave=False):
        rng = np.random.RandomState(int(delta * 1000))
        
        # Generate random kick directions
        kicks = rng.randn(n_trials, wc.p.N, 2).astype(np.float32)
        kicks = kicks / (np.linalg.norm(kicks, axis=(1,2), keepdims=True) + 1e-12)
        kicks = kicks * delta
        
        # Apply kicks from Ap
        x0 = Ap_t.unsqueeze(0).expand(n_trials, -1, -1).clone()
        x0 = (x0 + torch.tensor(kicks, device=DEVICE)).clamp(0., 1.)
        
        # Deterministic rollout
        x_final = wc.rollout_deterministic(x0, T=T)
        
        # Check how many reached Ah
        dist_Ah = ((x_final - Ah_t.unsqueeze(0))**2).sum(dim=(1,2))
        dist_Ap = ((x_final - Ap_t.unsqueeze(0))**2).sum(dim=(1,2))
        reached_Ah = (dist_Ah < dist_Ap).sum().item()
        
        crossing_fracs.append(reached_Ah / n_trials)
    
    crossing_fracs = np.array(crossing_fracs)
    
    # Find delta_50 (50% crossing)
    idx_50 = np.argmin(np.abs(crossing_fracs - 0.5))
    barrier_estimate = delta_vals[idx_50]
    
    results = {
        'delta_vals': delta_vals,
        'crossing_fractions': crossing_fracs,
        'barrier_estimate': barrier_estimate,
    }
    
    print(f"  ΔE_barrier ≈ {barrier_estimate:.3f} (delta at 50% crossing)")
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. MFPT + KRAMERS LAW VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def mfpt_and_kramers(wc, Ap, Ah, sigma_vals=None, n_trials=100, max_T=5000):
    """
    Measure Mean First Passage Time from Ap to Ah at different noise levels.
    Fit Kramers law: MFPT ≈ C·exp(A/σ²).
    
    Args:
        sigma_vals: array of noise levels
        n_trials: number of stochastic rollouts per sigma
        max_T: maximum rollout time
    
    Returns:
        dict with sigma_vals, mfpt_vals, kramers_fit_params
    """
    if sigma_vals is None:
        sigma_vals = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20])
    
    print(f"\n[MFPT + Kramers] Testing {len(sigma_vals)} noise levels...")
    
    Ap_t = torch.tensor(Ap, dtype=torch.float32, device=DEVICE)
    Ah_t = torch.tensor(Ah, dtype=torch.float32, device=DEVICE)
    
    mfpt_vals = []
    
    for sigma in tqdm(sigma_vals, desc="Sigma sweep", leave=False):
        passage_times = []
        
        for trial in range(n_trials):
            x0 = Ap_t.unsqueeze(0).clone()
            x = x0.clone()
            
            # Stochastic rollout until reaching Ah or timeout
            rng = np.random.RandomState(trial + int(sigma*10000))
            sqrt_dt = wc.p.dt ** 0.5
            
            for t in range(max_T):
                noise = torch.tensor(rng.randn(*x.shape).astype(np.float32),
                                   device=DEVICE)
                x = (x + wc.drift(x) * wc.p.dt 
                     + sigma * sqrt_dt * noise).clamp(0., 1.)
                
                # Check if in basin(Ah)
                dist_Ah = ((x - Ah_t.unsqueeze(0))**2).sum()
                dist_Ap = ((x - Ap_t.unsqueeze(0))**2).sum()
                
                if dist_Ah < dist_Ap:
                    passage_times.append(t + 1)
                    break
            else:
                # Timeout — use max_T as censored observation
                passage_times.append(max_T)
        
        mfpt_vals.append(np.mean(passage_times))
    
    mfpt_vals = np.array(mfpt_vals)
    
    # Fit Kramers law: MFPT ≈ C·exp(A/σ²)
    # log(MFPT) = log(C) + A/σ²
    # Linear regression on (1/σ², log(MFPT))
    
    x_fit = 1.0 / (sigma_vals ** 2)
    y_fit = np.log(mfpt_vals + 1)  # +1 to avoid log(0)
    
    try:
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        A_fitted = slope
        C_fitted = np.exp(intercept)
        
        # Compute R²
        y_pred = slope * x_fit + intercept
        ss_res = np.sum((y_fit - y_pred)**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        r_squared = 1 - (ss_res / ss_tot)
    except:
        A_fitted = np.nan
        C_fitted = np.nan
        r_squared = np.nan
    
    results = {
        'sigma_vals': sigma_vals,
        'mfpt_vals': mfpt_vals,
        'A_fitted': A_fitted,
        'C_fitted': C_fitted,
        'r_squared': r_squared,
    }
    
    print(f"  Kramers fit: MFPT ≈ {C_fitted:.2f} · exp({A_fitted:.4f}/σ²)")
    print(f"  R² = {r_squared:.4f}")
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. PUSH VS RESHAPE
# ══════════════════════════════════════════════════════════════════════════════

def compute_jacobian_energy(wc, Ap, reshape_dict):
    """
    Compute ||J_ΔW × x_fp||² where J_ΔW is Jacobian of drift w.r.t. parameter.
    Used for effort normalization (Vyas et al. 2020 framework).
    
    Args:
        wc: SimpleWC instance
        Ap: pathological fixed point (N, 2)
        reshape_dict: {'dwEI': value} or {'dwEE': value} or {'dtauI_frac': value}
    
    Returns:
        energy: scalar
    """
    Ap_t = torch.tensor(Ap, dtype=torch.float32, device=wc.device)
    x = Ap_t.unsqueeze(0)  # (1, N, 2)
    
    # Compute drift change
    f0 = wc.drift(x, u=None, reshape=None)
    f1 = wc.drift(x, u=None, reshape=reshape_dict)
    df = f1 - f0
    
    energy = float((df ** 2).sum())
    return energy


def push_vs_reshape_experiment(wc, Ap, Ah, sigma_vals=None, d=100, 
                                target_energies=[0.5, 1.0, 2.0], K=500):
    """
    Compare push (additive current) vs reshape (parameter modulation) at matched energy.
    
    Push family: u_E = a_i (uniform across all nodes)
    Reshape families:
      - GABA enhancement: increase wEI
      - Excitation reduction: decrease wEE
      - Inhibitory speed-up: decrease τI
    
    Energy matching: E_push = ||u||²·d = Σ a_i²·d
                     E_reshape = ||J_ΔW × x_fp||²·d
    
    Returns:
        dict with results for each intervention type
    """
    if sigma_vals is None:
        sigma_vals = np.array([0.03, 0.1, 0.2])
    
    print(f"\n[Push vs Reshape] Testing at {len(sigma_vals)} noise levels, {len(target_energies)} energy budgets...")
    
    Ap_t = torch.tensor(Ap, dtype=torch.float32, device=DEVICE)
    Ah_t = torch.tensor(Ah, dtype=torch.float32, device=DEVICE)
    
    results = {}
    
    for sigma in sigma_vals:
        print(f"\n  σ = {sigma:.2f}")
        results[sigma] = {}
        
        for target_energy in target_energies:
            print(f"    E = {target_energy:.1f}")
            
            # ── Push: uniform inhibitory current ─────────────────────────────────
            a_push = np.sqrt(target_energy / (wc.p.N * d))
            
            u_push = torch.zeros(1, wc.p.N, 2, device=DEVICE)
            u_push[:, :, 0] = -a_push  # inhibit E-channel
            
            # Run K stochastic rollouts
            x0 = Ap_t.unsqueeze(0).expand(K, -1, -1).clone()
            x_push_final = wc.rollout_stochastic(x0, sigma, T=d, u=u_push, seed=0)
            
            dist_Ah = ((x_push_final - Ah_t.unsqueeze(0))**2).sum(dim=(1,2))
            dist_Ap = ((x_push_final - Ap_t.unsqueeze(0))**2).sum(dim=(1,2))
            p_push = (dist_Ah < dist_Ap).sum().item() / K
            
            # ── Reshape: GABA enhancement (increase wEI) ──────────────────────────
            # Binary search for matching dwEI
            dwEI_low, dwEI_high = 0.0, 10.0
            for _ in range(20):
                dwEI_mid = (dwEI_low + dwEI_high) / 2
                e_mid = compute_jacobian_energy(wc, Ap, {'dwEI': dwEI_mid}) * d
                
                if e_mid < target_energy:
                    dwEI_low = dwEI_mid
                else:
                    dwEI_high = dwEI_mid
            
            dwEI_matched = (dwEI_low + dwEI_high) / 2
            
            x0_reshape = Ap_t.unsqueeze(0).expand(K, -1, -1).clone()
            x_reshape_final = wc.rollout_stochastic(
                x0_reshape, sigma, T=d, u=None, 
                reshape={'dwEI': dwEI_matched}, seed=1)
            
            dist_Ah_r = ((x_reshape_final - Ah_t.unsqueeze(0))**2).sum(dim=(1,2))
            dist_Ap_r = ((x_reshape_final - Ap_t.unsqueeze(0))**2).sum(dim=(1,2))
            p_reshape = (dist_Ah_r < dist_Ap_r).sum().item() / K
            
            results[sigma][target_energy] = {
                'push': {
                    'p_succ': p_push,
                    'amplitude': a_push,
                    'energy': target_energy,
                },
                'reshape_GABA': {
                    'p_succ': p_reshape,
                    'dwEI': dwEI_matched,
                    'energy': target_energy,
                },
            }
            
            print(f"      Push:    P_succ={p_push:.3f}  (a={a_push:.3f})")
            print(f"      Reshape: P_succ={p_reshape:.3f}  (Δw_EI={dwEI_matched:.3f})")
    
    return results

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_landscape_analysis(basin_res, barrier_res, mfpt_res, save_dir="results_w7"):
    """Generate 3-panel landscape figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Basin volume pie chart
    ax = axes[0]
    sizes = [basin_res['frac_Ap'], basin_res['frac_Ah'], basin_res['frac_other']]
    labels = [f"Basin(A_p)\n{basin_res['frac_Ap']*100:.1f}%",
              f"Basin(A_h)\n{basin_res['frac_Ah']*100:.1f}%",
              f"Other\n{basin_res['frac_other']*100:.1f}%"]
    colors = ['#e74c3c', '#2ecc71', '#95a5a6']
    ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90,
           wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    ax.set_title('(A) Basin Volume Distribution', fontweight='bold')
    
    # Panel B: Barrier height curve
    ax2 = axes[1]
    ax2.plot(barrier_res['delta_vals'], barrier_res['crossing_fractions'],
             'o-', color='#3498db', linewidth=2.5, markersize=8)
    ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1.5,
                label=f"50% crossing at δ={barrier_res['barrier_estimate']:.3f}")
    ax2.axvline(barrier_res['barrier_estimate'], color='#e74c3c',
                linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Kick Magnitude δ', fontweight='bold')
    ax2.set_ylabel('Fraction Reaching A$_h$', fontweight='bold')
    ax2.set_title('(B) Barrier Height Estimation', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, linestyle=':')
    
    # Panel C: MFPT + Kramers fit
    ax3 = axes[2]
    sigma_v = mfpt_res['sigma_vals']
    mfpt_v = mfpt_res['mfpt_vals']
    
    ax3.plot(sigma_v, mfpt_v, 'o', color='#9b59b6', markersize=10,
             label='Empirical MFPT', zorder=3)
    
    # Plot Kramers fit
    if not np.isnan(mfpt_res['A_fitted']):
        sigma_fit = np.linspace(sigma_v.min(), sigma_v.max(), 100)
        mfpt_fit = mfpt_res['C_fitted'] * np.exp(mfpt_res['A_fitted'] / sigma_fit**2)
        ax3.plot(sigma_fit, mfpt_fit, '-', color='#e74c3c', linewidth=2.5,
                 label=f"Kramers: C·exp(A/σ²)\nR²={mfpt_res['r_squared']:.3f}")
    
    ax3.set_xlabel('Noise Level σ', fontweight='bold')
    ax3.set_ylabel('Mean First Passage Time', fontweight='bold')
    ax3.set_title('(C) MFPT + Kramers Law Validation', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, linestyle=':')
    ax3.set_yscale('log')
    
    plt.suptitle('Week 7: Energy Landscape Characterization',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    path = f"{save_dir}/fig_landscape_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


def plot_push_vs_reshape(pvr_res, save_dir="results_w7"):
    """Generate push vs reshape comparison figure."""
    sigma_vals = sorted(pvr_res.keys())
    
    # Get all energy levels tested (assume consistent across sigmas)
    energy_levels = sorted(list(pvr_res[sigma_vals[0]].keys()))
    
    fig, axes = plt.subplots(1, len(energy_levels), figsize=(6*len(energy_levels), 6))
    if len(energy_levels) == 1:
        axes = [axes]  # make iterable
    
    for idx, energy in enumerate(energy_levels):
        ax = axes[idx]
        
        p_push = [pvr_res[s][energy]['push']['p_succ'] for s in sigma_vals]
        p_reshape = [pvr_res[s][energy]['reshape_GABA']['p_succ'] for s in sigma_vals]
        
        x = np.arange(len(sigma_vals))
        width = 0.35
        
        ax.bar(x - width/2, p_push, width, label='Push (additive current)',
               color='#3498db', edgecolor='black', linewidth=1.2)
        ax.bar(x + width/2, p_reshape, width, label='Reshape (GABA enhancement)',
               color='#2ecc71', edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Noise Level σ', fontweight='bold')
        ax.set_ylabel('P$_{succ}$', fontweight='bold')
        ax.set_title(f'Energy Budget = {energy:.1f}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.2f}" for s in sigma_vals])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle=':', axis='y')
        ax.set_ylim(0, 1.0)
    
    plt.suptitle('Week 7: Push vs Reshape at Multiple Energy Budgets',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    path = f"{save_dir}/fig_push_vs_reshape.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    N = args.N
    model_seed = args.model_seed
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print("  PACAN WEEK 7 — ENERGY LANDSCAPE + PUSH VS RESHAPE")
    print("="*70)
    print(f"N={N}, model_seed={model_seed}")
    
    # Build model
    params = WCParams(N=N, model_seed=model_seed)
    wc = SimpleWC(params, device=DEVICE)
    Ap, Ah = make_network_attractors(params)
    print(f"Attractors: Ap xE={Ap[:,0].mean():.4f}  Ah xE={Ah[:,0].mean():.4f}")
    
    results = {}
    
    # 1. Basin volume
    if not args.skip_basin:
        n_inits = 1000 if args.quick else 10000
        basin_res = basin_volume_analysis(wc, Ap, Ah, n_inits=n_inits)
        results['basin'] = basin_res
    
    # 2. Barrier height
    if not args.skip_barrier:
        n_trials = 50 if args.quick else 200
        barrier_res = barrier_height_montecarlo(wc, Ap, Ah, n_trials=n_trials)
        results['barrier'] = barrier_res
    
    # 3. MFPT + Kramers
    if not args.skip_mfpt:
        n_trials = 20 if args.quick else 100
        sigma_vals = np.array([0.05, 0.1, 0.15, 0.2]) if args.quick else None
        mfpt_res = mfpt_and_kramers(wc, Ap, Ah, sigma_vals=sigma_vals,
                                     n_trials=n_trials)
        results['mfpt'] = mfpt_res
    
    # 4. Push vs Reshape
    if args.push_reshape:
        K = 200 if args.quick else 500
        pvr_res = push_vs_reshape_experiment(wc, Ap, Ah, K=K)
        results['push_reshape'] = pvr_res
        pvr_path = f"{save_dir}/results_pvr_N{N}.json"
        with open(pvr_path, 'w') as f:
            json.dump(pvr_res, f, indent=2, default=str)
        print(f"PVR results saved: {pvr_path}")
    
    # Save results
    results_path = f"{save_dir}/results_landscape_N{N}.json"
    with open(results_path, 'w') as f:
        json.dump({k: v for k, v in results.items()
                   if k != 'push_reshape'},  # skip nested dict for now
                  f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x))
    print(f"\nResults saved: {results_path}")
    
    # Generate figures
    if not args.skip_basin and not args.skip_barrier and not args.skip_mfpt:
        plot_landscape_analysis(basin_res, barrier_res, mfpt_res, save_dir)
    
    if args.push_reshape:
        plot_push_vs_reshape(pvr_res, save_dir)
    
    print("\nWeek 7 analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PACAN Week 7")
    parser.add_argument("--N", type=int, default=15)
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="results_w7")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced n_inits/trials for testing")
    parser.add_argument("--skip_basin", action="store_true")
    parser.add_argument("--skip_barrier", action="store_true")
    parser.add_argument("--skip_mfpt", action="store_true")
    parser.add_argument("--push_reshape", action="store_true",
                        help="Run push vs reshape experiment")
    args = parser.parse_args()
    
    main(args)
