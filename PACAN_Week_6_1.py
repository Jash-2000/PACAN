"""
PACAN Week 6 — Rigorous Experimental Foundation
Three-Phase Rollout · Seed Factorization · N-Scaling · Stronger Baselines

Usage:
    python pacan_week6.py --quick              # ~5 min test
    python pacan_week6.py --N_list 10 20       # partial run
    python pacan_week6.py                       # full study (N=10,20,30,50)

Requirements:
    pip install torch numpy scipy matplotlib tqdm
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed; saves PNGs to disk
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.integrate import solve_ivp
from tqdm import tqdm
import json, os, argparse, warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results_w6", exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_T_anneal(N, quick=False):
    """8 Gibbs passes per node, capped at 200."""
    return 25 if quick else min(8 * N, 200)

def get_K_eval(N, quick=False):
    """More nodes → need larger K for consistent CI width."""
    return 200 if quick else min(500 * (N // 10), 3000)


# ══════════════════════════════════════════════════════════════════════════════
# WC PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

class WCParams:
    """
    N-population WC parameters.
    model_seed controls ONLY W_net topology.

    Bistable single-node FPs at P=1.25, wEE=10, wEI=8:
        Interictal Ah: xE≈0.090, xI≈0.510
        Ictal     Ap: xE≈0.974, xI≈0.997
    """
    def __init__(self, N=10,
                 wEE=10.0, wEI=8.0, wIE=6.0, wII=1.0,
                 w_net_scale=0.15, sparsity=0.70,
                 P=1.25, a=1.2, dt=0.1,
                 model_seed=0):
        self.N = N
        self.wEE = wEE; self.wEI = wEI
        self.wIE = wIE; self.wII = wII
        self.P = P; self.a = a; self.dt = dt
        self.w_net_scale = w_net_scale
        self.sparsity = sparsity
        self.model_seed = model_seed

        rng  = np.random.RandomState(model_seed)
        W    = rng.randn(N, N) * w_net_scale
        mask = (rng.rand(N, N) > sparsity).astype(float)
        np.fill_diagonal(mask, 0)
        self.W_net = torch.tensor(W * mask, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ATTRACTOR FINDING
# ══════════════════════════════════════════════════════════════════════════════

def find_single_node_fps(P=1.25, wEE=10., wEI=8.,
                          wIE=6., wII=1., a=1.2, n=400):
    """Exact single-node WC fixed points via deterministic ODE."""
    def f(t, x):
        s = lambda z: 1.0 / (1.0 + np.exp(-a * z))
        return [(-x[0] + s(wEE*x[0] - wEI*x[1] + P)) / 10.0,
                (-x[1] + s(wIE*x[0] - wII*x[1]))     / 10.0]

    fps = []
    rng = np.random.RandomState(0)
    for _ in range(n):
        sol = solve_ivp(f, [0, 600], rng.rand(2),
                        method="RK45", rtol=1e-8, atol=1e-10)
        xf  = np.clip(sol.y[:, -1], 0, 1)
        new = True
        for fp in fps:
            if np.linalg.norm(xf - fp) < 0.02:
                new = False; break
        if new:
            fps.append(xf)
    return sorted(fps, key=lambda x: x[0])


def make_network_attractors(params: WCParams):
    """
    Build N-node attractor arrays from single-node fixed points,
    then refine with coupled deterministic dynamics.
    """
    fps = find_single_node_fps(
        P=params.P, wEE=params.wEE, wEI=params.wEI,
        wIE=params.wIE, wII=params.wII, a=params.a)
    assert len(fps) >= 2, \
        f"System not bistable at P={params.P}. FPs: {fps}"

    xE_h, xI_h = fps[0]    # interictal (low xE)
    xE_p, xI_p = fps[-1]   # ictal      (high xE)

    Ap = np.tile([xE_p, xI_p], (params.N, 1)).astype(np.float32)
    Ah = np.tile([xE_h, xI_h], (params.N, 1)).astype(np.float32)

    # Refine with coupled deterministic dynamics
    wc = ThreePhaseWC(params, device=DEVICE)
    for Arr in [Ap, Ah]:
        x = torch.tensor(Arr[None, :, :], device=DEVICE)
        u = torch.zeros_like(x)
        for _ in range(3000):
            x = (x + wc.drift(x, u) * params.dt).clamp(0., 1.)
        Arr[:] = x[0].cpu().numpy()

    return Ap, Ah


# ══════════════════════════════════════════════════════════════════════════════
# THREE-PHASE WC MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ThreePhaseWC:
    """
    Three-phase stochastic WC network.

    Phase 1 — Pre-settlement  : free dynamics until adaptive variance threshold.
    Phase 2 — Intervention    : u applied for d steps.
    Phase 3 — Post-settlement : free dynamics until adaptive variance threshold.

    noise_seed controls ALL Brownian motion across all three phases.
    Convergence criterion: Var(xE[-win:]) / Var(xE[0:t]) < tau_rel (1%).
    """
    def __init__(self, params: WCParams, device=DEVICE):
        self.p      = params
        self.device = device
        self.W_net  = params.W_net.to(device)

    def sigmoid(self, x):
        return torch.sigmoid(self.p.a * x)

    def drift(self, x, u, reshape=None):
        """x, u : (batch, N, 2).  reshape : dict with optional dwEI/dwEE/dtauI_frac."""
        xE = x[..., 0]; xI = x[..., 1]
        uE = u[..., 0]; uI = u[..., 1]
        wEI  = self.p.wEI + (reshape.get("dwEI",  0.) if reshape else 0.)
        wEE  = self.p.wEE + (reshape.get("dwEE",  0.) if reshape else 0.)
        tauI = 10. * (1. + (reshape.get("dtauI_frac", 0.) if reshape else 0.))
        inter = torch.matmul(xE, self.W_net.T)
        IE    = wEE*xE - wEI*xI + inter + uE + self.p.P
        II    = self.p.wIE*xE - self.p.wII*xI + uI
        dxE   = (-xE + self.sigmoid(IE)) / 10.
        dxI   = (-xI + self.sigmoid(II)) / tauI
        return torch.stack([dxE, dxI], dim=-1)

    def _settle(self, x, nrng, sigma,
                tau_rel=0.01, win=50, min_s=200, max_s=2000,
                u=None, reshape=None):
        """Adaptive-variance settlement. Returns (x_settled, n_steps)."""
        if u is None:
            u = torch.zeros_like(x)
        sqrt_dt = self.p.dt ** 0.5
        buf = []
        for step in range(max_s):
            noise = torch.tensor(
                nrng.randn(*x.shape).astype(np.float32), device=self.device)
            x = (x + self.drift(x, u, reshape) * self.p.dt
                 + sigma * sqrt_dt * noise).clamp(0., 1.)
            buf.append(float(x[..., 0].mean()))
            if step >= min_s and len(buf) >= win:
                vr = float(np.var(buf[-win:]))
                vt = float(np.var(buf)) + 1e-12
                if vr / vt < tau_rel:
                    return x, step + 1
        return x, max_s

    @torch.no_grad()
    def rollout(self, Ap, Ah, I_mask, amplitudes,
                sigma, noise_seed, d=100, perturb=0.02,
                reshape=None, K=500,
                tau_rel=0.01, win=50, min_settle=200, max_settle=2000):
        """
        Full 3-phase rollout.
        Returns dict: p_succ, ci_low, ci_high, effort, occupancy,
                      steps_pre, steps_post.
        """
        nrng = np.random.RandomState(noise_seed)
        Ap_t = torch.tensor(Ap, dtype=torch.float32, device=self.device)
        Ah_t = torch.tensor(Ah, dtype=torch.float32, device=self.device)

        # Init K trajectories near Ap
        init = nrng.randn(K, self.p.N, 2).astype(np.float32)
        x = (Ap_t.unsqueeze(0).expand(K, -1, -1).clone()
             + perturb * torch.tensor(init, device=self.device)).clamp(0., 1.)

        # Phase 1 — pre-settlement
        x, steps_pre = self._settle(
            x, nrng, sigma, tau_rel, win, min_settle, max_settle)

        # Phase 2 — intervention
        u_s = (I_mask.unsqueeze(-1) * amplitudes)
        u_b = u_s.unsqueeze(0).expand(K, -1, -1).contiguous()
        sqrt_dt = self.p.dt ** 0.5
        for _ in range(d):
            noise = torch.tensor(
                nrng.randn(*x.shape).astype(np.float32), device=self.device)
            x = (x + self.drift(x, u_b, reshape) * self.p.dt
                 + sigma * sqrt_dt * noise).clamp(0., 1.)

        # Phase 3 — post-settlement + occupancy tracking
        occ      = torch.zeros(K, device=self.device)
        u0       = torch.zeros_like(u_b)
        buf3     = []
        steps_post = max_settle
        for step in range(max_settle):
            noise = torch.tensor(
                nrng.randn(*x.shape).astype(np.float32), device=self.device)
            x = (x + self.drift(x, u0) * self.p.dt
                 + sigma * sqrt_dt * noise).clamp(0., 1.)
            dh = ((x - Ah_t.unsqueeze(0))**2).sum(dim=(1, 2))
            dp = ((x - Ap_t.unsqueeze(0))**2).sum(dim=(1, 2))
            occ += (dh < dp).float()
            buf3.append(float(x[..., 0].mean()))
            if step >= min_settle and len(buf3) >= win:
                vr = float(np.var(buf3[-win:]))
                vt = float(np.var(buf3)) + 1e-12
                if vr / vt < tau_rel:
                    steps_post = step + 1
                    break

        # Endpoint basin membership
        dh_f = ((x - Ah_t.unsqueeze(0))**2).sum(dim=(1, 2))
        dp_f = ((x - Ap_t.unsqueeze(0))**2).sum(dim=(1, 2))
        n_s  = (dh_f < dp_f).sum().item()
        p    = n_s / K

        # Wilson 95% CI
        z = 1.96; dn = 1. + z**2 / K
        c  = (p + z**2 / (2*K)) / dn
        hw = z * np.sqrt(p*(1-p)/K + z**2/(4*K**2)) / dn

        return dict(
            p_succ    = p,
            ci_low    = max(0., c - hw),
            ci_high   = min(1., c + hw),
            effort    = float((I_mask.unsqueeze(-1) * amplitudes.abs()).sum()),
            occupancy = float(occ.mean() / max(steps_post, 1)),
            steps_pre = steps_pre,
            steps_post= steps_post,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PACAN OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class PACANOptimizer:
    """
    PACAN: Gibbs p-bit sweep + FD gradient descent on amplitudes.
    use_gibbs=False → Simulated Annealing (Metropolis-Hastings) baseline.
    use_grad=False  → ablation: frozen amplitudes.
    T_anneal should = 8*N for full study (passed explicitly).
    """
    def __init__(self, wc: ThreePhaseWC, Ap, Ah,
                 sigma_opt=0.1,
                 lambda1=0.5, lambda2=0.1, lambda3=0.05, lambda4=5.0,
                 K_explore=80, K_eval=2000,
                 T_anneal=80, beta_start=0.5, beta_end=8.0,
                 amp_lr=0.2, amp_steps=5, amp_clip=4.0,
                 d_grid=(50, 100, 200, 400),
                 use_gibbs=True, use_grad=True,
                 opt_seed=0, noise_seed=0,
                 device=DEVICE):
        self.wc        = wc
        self.Ap        = np.array(Ap, dtype=np.float32)
        self.Ah        = np.array(Ah, dtype=np.float32)
        self.N         = Ap.shape[0]
        self.sigma_opt = sigma_opt
        self.l1=lambda1; self.l2=lambda2; self.l3=lambda3; self.l4=lambda4
        self.K_explore = K_explore; self.K_eval = K_eval; self.T = T_anneal
        self.betas     = np.linspace(beta_start, beta_end, T_anneal)
        self.amp_lr=amp_lr; self.amp_steps=amp_steps; self.amp_clip=amp_clip
        self.d_grid    = list(d_grid)
        self.use_gibbs = use_gibbs; self.use_grad = use_grad
        self.opt_rng   = np.random.RandomState(opt_seed)
        self.noise_seed= noise_seed
        self.device    = device

    def _eval(self, I, amps, d, K):
        return self.wc.rollout(self.Ap, self.Ah, I, amps,
                                self.sigma_opt, self.noise_seed,
                                d=d, K=K)

    def _loss(self, I, amps, d, K):
        r = self._eval(I, amps, d, K)
        L = (self.l1 * I.sum().item()
             + self.l2 * (I.unsqueeze(-1) * amps.abs()).sum().item()
             + self.l3 * (d / max(self.d_grid))
             + self.l4 * (1.0 - r["p_succ"]))
        return L, r["p_succ"]

    def _best_d(self, I, amps):
        best_d, best_p = self.d_grid[0], -1.0
        for d in self.d_grid:
            r = self._eval(I, amps, d, K=min(60, self.K_explore))
            if r["p_succ"] > best_p:
                best_p, best_d = r["p_succ"], d
        return best_d

    def _gibbs_prob(self, i, I, amps, d, beta):
        L0, _ = self._loss(I, amps, d, self.K_explore)
        I_f   = I.clone(); I_f[i] = 1.0 - I_f[i]
        L1, _ = self._loss(I_f, amps, d, self.K_explore)
        delta = (L1 - L0) if I[i].item() < 0.5 else -(L1 - L0)
        return float(torch.sigmoid(torch.tensor(-beta * delta)))

    def _metro_step(self, I, amps, d, beta):
        L0, _ = self._loss(I, amps, d, self.K_explore)
        i     = int(self.opt_rng.randint(0, self.N))
        I_f   = I.clone(); I_f[i] = 1.0 - I_f[i]
        L1, _ = self._loss(I_f, amps, d, self.K_explore)
        dL    = L1 - L0
        if dL < 0 or self.opt_rng.rand() < np.exp(-beta * dL):
            return I_f
        return I

    @torch.no_grad()
    def _grad_amps(self, I, amps, d):
        eps = 0.3
        for _ in range(self.amp_steps):
            L0, _ = self._loss(I, amps, d, self.K_explore)
            grad  = torch.zeros_like(amps)
            for i in range(self.N):
                if I[i].item() < 0.5: continue
                ap = amps.clone(); ap[i, 0] += eps
                Lp, _ = self._loss(I, ap, d, self.K_explore)
                grad[i, 0] = (Lp - L0) / eps
            amps = (amps - self.amp_lr * grad) * I.unsqueeze(-1)
            amps = amps.clamp(-self.amp_clip, self.amp_clip)
        return amps

    @staticmethod
    def _dominated(c, arch):
        for s in arch:
            if (s["n_nodes"] <= c["n_nodes"]
                    and s["effort"]  <= c["effort"]
                    and s["d"]       <= c["d"]
                    and s["p_succ"]  >= c["p_succ"]
                    and (s["n_nodes"] < c["n_nodes"]
                         or s["effort"]  < c["effort"]
                         or s["d"]       < c["d"]
                         or s["p_succ"]  > c["p_succ"])):
                return True
        return False

    def run(self, verbose=False):
        I    = torch.zeros(self.N, device=self.device)
        amps = torch.zeros(self.N, 2, device=self.device)
        amps[:, 0] = -1.5          # warm-start: inhibit E
        d    = self.d_grid[1]      # warm-start: 100 steps
        archive = []

        pbar = tqdm(range(self.T), desc="PACAN", leave=False)
        for t in pbar:
            beta = self.betas[t]

            # Node selection
            if self.use_gibbs:
                for i in self.opt_rng.permutation(self.N):
                    p1  = self._gibbs_prob(i, I, amps, d, beta)
                    I[i]= 1.0 if self.opt_rng.rand() < p1 else 0.0
            else:
                I = self._metro_step(I, amps, d, beta)

            # d-grid search every 10 iterations
            if t % 10 == 0 and I.sum().item() > 0:
                d = self._best_d(I, amps)

            # Amplitude gradient step
            if self.use_grad and I.sum().item() > 0:
                amps = self._grad_amps(I, amps, d)

            # Evaluate and update Pareto archive
            r   = self._eval(I, amps, d, self.K_eval)
            eff = float((I.unsqueeze(-1) * amps.abs()).sum())
            cand= dict(I=I.clone(), amps=amps.clone(), d=d,
                       n_nodes=int(I.sum()), effort=eff,
                       p_succ=r["p_succ"], ci_low=r["ci_low"],
                       ci_high=r["ci_high"], occupancy=r["occupancy"])
            if not self._dominated(cand, archive):
                archive = [s for s in archive
                           if not self._dominated(s, [cand])]
                archive.append(cand)

            pbar.set_postfix({"|I|": int(I.sum()),
                              "P":   f"{r['p_succ']:.3f}",
                              "arch":len(archive)})
            if verbose and (t % 20 == 0 or t == self.T - 1):
                print(f"  t={t:3d} β={beta:.2f} |I|={int(I.sum())} "
                      f"d={d} P={r['p_succ']:.3f} arch={len(archive)}")
        return archive


# ══════════════════════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════════════════════

def baseline_greedy(wc, Ap, Ah, N, sigma, noise_seed,
                    amp_val=-2.5, d_grid=(50,100,200,400), K=200):
    """Greedy forward node selection at given sigma."""
    selected, remaining, best_p = [], list(range(N)), 0.0
    best_d = d_grid[0]                      # always initialized
    for _ in range(N):
        gains = []
        for node in remaining:
            cand = selected + [node]
            Im = torch.zeros(N, device=DEVICE); Im[cand] = 1.0
            am = torch.zeros(N, 2, device=DEVICE); am[cand, 0] = amp_val
            nbp = 0.0; nbd = d_grid[0]      # always initialized
            for d in d_grid:
                r = wc.rollout(Ap, Ah, Im, am, sigma, noise_seed, d=d, K=K)
                if r["p_succ"] > nbp:
                    nbp = r["p_succ"]; nbd = d
            gains.append((nbp, node, nbd))
        gains.sort(reverse=True)
        gp, gn, gd = gains[0]
        if gp > best_p + 0.01:
            selected.append(gn); remaining.remove(gn)
            best_p, best_d = gp, gd
            if best_p >= 0.95: break
        else:
            break
    Im = torch.zeros(N, device=DEVICE); Im[selected] = 1.0
    am = torch.zeros(N, 2, device=DEVICE); am[selected, 0] = amp_val
    return Im, am, best_p, best_d, selected


def baseline_random_search(wc, Ap, Ah, N, sigma, noise_seed,
                            n_trials=30, d_grid=(50,100,200,400), K=200):
    rng    = np.random.RandomState(noise_seed + 500)
    best_p = 0.0
    bI     = torch.zeros(N, device=DEVICE)
    bA     = torch.zeros(N, 2, device=DEVICE)
    bd     = d_grid[0]
    for _ in range(n_trials):
        Im = torch.tensor(rng.choice([0., 1.], N).astype(np.float32),
                          device=DEVICE)
        if Im.sum() == 0: Im[rng.randint(N)] = 1.0
        am = torch.zeros(N, 2, device=DEVICE)
        am[:, 0] = torch.tensor(
            (rng.rand(N) * (-3.0)) * Im.cpu().numpy(), dtype=torch.float32)
        for d in d_grid:
            r = wc.rollout(Ap, Ah, Im, am, sigma, noise_seed, d=d, K=K)
            if r["p_succ"] > best_p:
                best_p = r["p_succ"]; bI = Im.clone(); bA = am.clone(); bd = d
    return bI, bA, best_p, bd


def baseline_degree(wc, Ap, Ah, N, sigma, noise_seed,
                    target_k, amp_val=-2.5, d_grid=(50,100,200,400), K=200):
    W_np   = wc.p.W_net.numpy()
    in_deg = np.maximum(W_np, 0).sum(axis=0)
    top    = np.argsort(in_deg)[-target_k:].tolist()
    Im = torch.zeros(N, device=DEVICE); Im[top] = 1.0
    am = torch.zeros(N, 2, device=DEVICE); am[top, 0] = amp_val
    best_p, bd = 0.0, d_grid[0]
    for d in d_grid:
        r = wc.rollout(Ap, Ah, Im, am, sigma, noise_seed, d=d, K=K)
        if r["p_succ"] > best_p: best_p, bd = r["p_succ"], d
    return Im, am, best_p, bd, top


def baseline_fvs(wc, Ap, Ah, N, sigma, noise_seed,
                 amp_val=-2.5, d_grid=(50,100,200,400), K=200):
    W_np = np.abs(wc.p.W_net.numpy())
    tot  = W_np.sum(0) + W_np.sum(1)
    fvs  = np.argsort(tot)[-max(1, int(np.ceil(N/3))):].tolist()
    Im = torch.zeros(N, device=DEVICE); Im[fvs] = 1.0
    am = torch.zeros(N, 2, device=DEVICE); am[fvs, 0] = amp_val
    best_p, bd = 0.0, d_grid[0]
    for d in d_grid:
        r = wc.rollout(Ap, Ah, Im, am, sigma, noise_seed, d=d, K=K)
        if r["p_succ"] > best_p: best_p, bd = r["p_succ"], d
    return Im, am, best_p, bd, fvs


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def hypervolume_2d(archive, ref_n, ref_p=0.0):
    if not archive: return 0.0
    pts    = sorted(archive, key=lambda s: s["n_nodes"])
    hv     = 0.0; prev_n = ref_n
    for s in reversed(pts):
        w = prev_n - s["n_nodes"]
        h = s["p_succ"] - ref_p
        if w > 0 and h > 0: hv += w * h
        prev_n = min(prev_n, s["n_nodes"])
    return hv


def variance_decomp(grid):
    """grid: {(ms,os,ns): p_succ} → % variance per source."""
    vals = list(grid.values())
    tv   = float(np.var(vals)) + 1e-12
    def btwn(idx):
        g = {}
        for keys, val in grid.items(): g.setdefault(keys[idx], []).append(val)
        gm    = [np.mean(v) for v in g.values()]
        grand = np.mean(vals)
        return float(np.mean([(m - grand)**2 for m in gm]))
    vm, vo, vn = btwn(0), btwn(1), btwn(2)
    vr = max(0., tv - vm - vo - vn)
    s  = vm + vo + vn + vr + 1e-12
    return dict(model=100*vm/s, opt=100*vo/s,
                noise=100*vn/s, residual=100*vr/s)


def to_json(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, np.ndarray):           return obj.tolist()
    if isinstance(obj, torch.Tensor):         return obj.cpu().numpy().tolist()
    if isinstance(obj, dict):   return {k: to_json(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json(i) for i in obj]
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION SUITE
# ══════════════════════════════════════════════════════════════════════════════

def run_validation():
    print("=" * 55)
    print("  PACAN WEEK 6 — VALIDATION SUITE")
    print("=" * 55)

    # 1. Bistability
    fps = find_single_node_fps()
    assert len(fps) >= 2, f"Not bistable: {fps}"
    print(f"[1] FPs: {[(round(f[0],4),round(f[1],4)) for f in fps]}  OK")

    # 2. Attractors
    p0 = WCParams(N=10, model_seed=0)
    wc0 = ThreePhaseWC(p0)
    Ap0, Ah0 = make_network_attractors(p0)
    assert Ap0[:,0].mean() > 0.7, f"Ap not ictal: {Ap0[:,0].mean():.4f}"
    assert Ah0[:,0].mean() < 0.3, f"Ah not interictal: {Ah0[:,0].mean():.4f}"
    print(f"[2] Attractors: Ap={Ap0[:,0].mean():.4f} Ah={Ah0[:,0].mean():.4f}  OK")

    # 3. Seed independence
    p1 = WCParams(N=10, model_seed=1)
    diff = float((p0.W_net - p1.W_net).abs().mean())
    assert diff > 0.01
    print(f"[3] Seed independence: W_net diff={diff:.4f}  OK")

    # 4. Null vs full intervention
    In = torch.zeros(10, device=DEVICE)
    an = torch.zeros(10, 2, device=DEVICE)
    rn = wc0.rollout(Ap0, Ah0, In, an, 0.05, 0, d=100, K=100)
    Ia = torch.ones(10, device=DEVICE)
    aa = torch.zeros(10, 2, device=DEVICE); aa[:, 0] = -2.5
    ra = wc0.rollout(Ap0, Ah0, Ia, aa, 0.05, 0, d=200, K=100)
    assert ra["p_succ"] > rn["p_succ"], "Full inhibit should beat null"
    print(f"[4] Null={rn['p_succ']:.3f}  Full={ra['p_succ']:.3f}  OK")

    # 5. Greedy (no UnboundLocalError)
    Im_g, am_g, p_g, d_g, ng = baseline_greedy(
        wc0, Ap0, Ah0, 10, 0.05, 0, d_grid=(50, 100), K=80)
    print(f"[5] Greedy: p={p_g:.3f}, d={d_g}, nodes={ng}  OK")

    # 6. PACAN smoke test
    opt_v = PACANOptimizer(wc0, Ap0, Ah0, sigma_opt=0.05,
                            K_explore=20, K_eval=60, T_anneal=5,
                            d_grid=(50, 100), opt_seed=0, noise_seed=0)
    arch_v = opt_v.run()
    assert len(arch_v) >= 1
    print(f"[6] PACAN smoke: {len(arch_v)} archive solutions  OK")

    print("=" * 55)
    print("  ALL VALIDATION CHECKS PASSED")
    print("=" * 55 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_scaling(all_results, save_dir="results_w6"):
    N_done = sorted(all_results.keys())
    if not N_done: return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    pm = [np.mean(all_results[N]["pacan_psucc"]) for N in N_done]
    ps = [np.std( all_results[N]["pacan_psucc"]) for N in N_done]
    dm = [np.mean(all_results[N]["det_psucc"])   for N in N_done]
    ds = [np.std( all_results[N]["det_psucc"])   for N in N_done]

    ax = axes[0]
    ax.fill_between(N_done, [m-s for m,s in zip(pm,ps)],
                             [m+s for m,s in zip(pm,ps)],
                    alpha=0.2, color="#2ecc71")
    ax.plot(N_done, pm, "o-", color="#2ecc71", lw=2.5, ms=9,
            label="PACAN@σ=0.1")
    ax.fill_between(N_done, [m-s for m,s in zip(dm,ds)],
                             [m+s for m,s in zip(dm,ds)],
                    alpha=0.2, color="#e74c3c")
    ax.plot(N_done, dm, "s--", color="#e74c3c", lw=2, ms=8,
            label="Det@σ=0 (eval@σ=0.1)")
    ax.set_xlabel("N"); ax.set_ylabel("P_succ")
    ax.set_title("Scaling: P_succ vs N\n(mean±std, 5 model seeds)")
    ax.legend(fontsize=9); ax.set_xticks(N_done)

    hv_r = [[p/max(d,1e-6) for p,d in
              zip(all_results[N]["hv_pacan"], all_results[N]["hv_det"])]
             for N in N_done]
    ax2 = axes[1]
    ax2.fill_between(N_done,
                     [np.mean(r)-np.std(r) for r in hv_r],
                     [np.mean(r)+np.std(r) for r in hv_r],
                     alpha=0.2, color="#3498db")
    ax2.plot(N_done, [np.mean(r) for r in hv_r],
             "D-", color="#3498db", lw=2.5, ms=9)
    ax2.axhline(1.0, color="gray", ls=":", lw=1.5, label="Equal HV")
    ax2.set_xlabel("N"); ax2.set_ylabel("HV ratio (PACAN/Det)")
    ax2.set_title("Hypervolume Ratio vs N\n(>1 = PACAN dominates)")
    ax2.legend(fontsize=9); ax2.set_xticks(N_done)

    ax3 = axes[2]
    N_vd = N_done[-1]
    vd_list = all_results[N_vd]["var_decomp"]
    if vd_list:
        comps  = ["model", "opt", "noise", "residual"]
        colors = ["#e74c3c", "#2ecc71", "#3498db", "#95a5a6"]
        means  = [np.mean([v[c] for v in vd_list]) for c in comps]
        bars   = ax3.bar(comps, means, color=colors, edgecolor="k")
        for bar, val in zip(bars, means):
            ax3.text(bar.get_x()+bar.get_width()/2, val+0.5,
                     f"{val:.1f}%", ha="center", fontsize=9)
        ax3.set_ylabel("% of P_succ variance")
        ax3.set_title(f"Variance Decomposition (N={N_vd})")

    plt.suptitle("Week 6: Scaling Experiment",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{save_dir}/fig_scaling.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_baselines(bl_res, N, sigma, save_dir="results_w6"):
    if not bl_res: return
    methods = list(bl_res.keys())
    means   = [np.mean(bl_res[m]) for m in methods]
    stds    = [np.std( bl_res[m]) for m in methods]
    colors  = ["#2ecc71","#27ae60","#e74c3c","#e67e22","#3498db","#9b59b6"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.bar(methods, means, yerr=stds,
           color=colors[:len(methods)],
           edgecolor="k", lw=0.8, capsize=5, width=0.55)
    for i, (m, v) in enumerate(zip(methods, means)):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center",
                fontsize=8, fontweight="bold")
    ax.set_ylabel("P_succ"); ax.set_ylim(0, 1.1)
    ax.set_title(f"Baseline Comparison N={N}, σ={sigma}")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")

    ax2 = axes[1]
    ref    = np.mean(bl_res.get("Det.greedy", [0.5]))
    deltas = [np.mean(bl_res[m]) - ref for m in methods]
    cols_d = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]
    ax2.bar(methods, deltas, color=cols_d, edgecolor="k", width=0.55)
    ax2.axhline(0, color="k", lw=1.2)
    for i, d in enumerate(deltas):
        ax2.text(i, d + (0.005 if d >= 0 else -0.015),
                 f"{d:+.3f}", ha="center", fontsize=8)
    ax2.set_ylabel("ΔP_succ vs Det.greedy")
    ax2.set_title("Advantage over Deterministic Greedy")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha="right")

    plt.suptitle(f"Week 6: All Baselines N={N}",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{save_dir}/fig_baselines_N{N}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

PRE_REGISTERED = {
    "C1": "Det P_succ at σ≥0.1 < 80% of σ=0, all N, ≥4/5 model seeds",
    "C2": "PACAN@σ=0.1 > Det@σ=0 (p<0.05), N≥20, ≥4/5 seeds",
    "C3": "PACAN HV > Det HV (p<0.01), ≥4/5 seeds at N=20",
}


def main(args):
    quick       = args.quick
    N_list      = args.N_list
    model_seeds = args.model_seeds
    opt_seeds   = args.opt_seeds
    noise_seeds = args.noise_seeds
    K_final     = args.K_final
    K_baseline  = args.K_baseline
    K_explore   = args.K_explore
    sigma_opt   = 0.1
    d_grid      = (50, 100, 200, 400)
    save_dir    = args.save_dir
    n_bl_seeds  = args.n_bl_seeds

    print("\n=== PRE-REGISTERED PASS/FAIL CRITERIA ===")
    for k, v in PRE_REGISTERED.items():
        print(f"  {k}: {v}")
    print()
    print(f"Config: N_list={N_list}, model_seeds={model_seeds}")
    print(f"  T_anneal per N: {[get_T_anneal(N, quick) for N in N_list]}")
    print(f"  K_eval   per N: {[get_K_eval(N, quick)   for N in N_list]}")
    print()

    # Verify bistability once
    fps = find_single_node_fps()
    print(f"Single-node FPs: {[(round(f[0],4),round(f[1],4)) for f in fps]}")
    assert len(fps) >= 2, "System not bistable — check parameters!"

    all_results = {}

    # ── Scaling experiment ────────────────────────────────────────────────────
    for N in N_list:
        T_anneal = get_T_anneal(N, quick)
        K_eval   = get_K_eval(N, quick)
        print(f"\n{'='*55}")
        print(f"N={N}  T_anneal={T_anneal}  K_eval={K_eval}")
        print(f"{'='*55}")

        res = dict(pacan_psucc=[], det_psucc=[],
                   hv_pacan=[], hv_det=[], var_decomp=[])
        var_grid = {}

        for ms in model_seeds:
            params = WCParams(N=N, model_seed=ms)
            wc     = ThreePhaseWC(params, device=DEVICE)
            print(f"\n  model_seed={ms}: building attractors...", end=" ", flush=True)
            Ap, Ah = make_network_attractors(params)
            print(f"Ap={Ap[:,0].mean():.3f}  Ah={Ah[:,0].mean():.3f}")

            ms_pacan = []
            p_det_fair = 0.0
            arch_last  = []
            I_det_last = torch.zeros(N, device=DEVICE)
            a_det_last = torch.zeros(N, 2, device=DEVICE)
            d_det_last = d_grid[0]

            for os_ in opt_seeds:
                for ns in noise_seeds:
                    # Det. greedy @ sigma=0
                    I_det, a_det, _, d_det, _ = baseline_greedy(
                        wc, Ap, Ah, N, sigma=0.0,
                        noise_seed=ns, d_grid=d_grid, K=K_baseline)
                    I_det_last = I_det
                    a_det_last = a_det
                    d_det_last = d_det

                    # Re-evaluate det at sigma_opt (fair comparison)
                    r_fair = wc.rollout(Ap, Ah, I_det, a_det,
                                         sigma_opt, ns, d=d_det, K=K_final)
                    p_det_fair = r_fair["p_succ"]

                    # PACAN @ sigma_opt=0.1
                    opt = PACANOptimizer(
                        wc, Ap, Ah,
                        sigma_opt=sigma_opt,
                        K_explore=K_explore, K_eval=K_eval,
                        T_anneal=T_anneal, d_grid=d_grid,
                        use_gibbs=True, use_grad=True,
                        opt_seed=os_, noise_seed=ns)
                    arch = opt.run()
                    arch_last = arch

                    if arch:
                        best    = max(arch, key=lambda s: s["p_succ"])
                        r_fin   = wc.rollout(Ap, Ah, best["I"], best["amps"],
                                              sigma_opt, ns,
                                              d=best["d"], K=K_final)
                        p_pacan = r_fin["p_succ"]
                    else:
                        p_pacan = 0.0

                    var_grid[(ms, os_, ns)] = p_pacan
                    ms_pacan.append(p_pacan)
                    print(f"    ms={ms} os={os_} ns={ns}: "
                          f"det_fair={p_det_fair:.3f}  pacan={p_pacan:.3f}")

            det_arch = [{"n_nodes": int(I_det_last.sum()),
                         "p_succ":  p_det_fair}]
            res["pacan_psucc"].append(np.mean(ms_pacan))
            res["det_psucc"].append(p_det_fair)
            res["hv_pacan"].append(hypervolume_2d(arch_last, ref_n=N+1))
            res["hv_det"].append(hypervolume_2d(det_arch,  ref_n=N+1))
            res["var_decomp"].append(variance_decomp(var_grid))

        all_results[N] = res
        with open(f"{save_dir}/results_N{N}.json", "w") as f:
            json.dump(to_json(res), f, indent=2)
        print(f"\n  N={N} summary:")
        print(f"    PACAN  {np.mean(res['pacan_psucc']):.3f}"
              f" ± {np.std(res['pacan_psucc']):.3f}")
        print(f"    Det    {np.mean(res['det_psucc']):.3f}"
              f" ± {np.std(res['det_psucc']):.3f}")

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_scaling(all_results, save_dir)

    # ── Claim validation ──────────────────────────────────────────────────────
    print(f"\n{'='*55}\n  CLAIM VALIDATION\n{'='*55}")
    for N, res in all_results.items():
        pp = res["pacan_psucc"]; dp = res["det_psucc"]
        if len(pp) >= 2:
            t, p = sp_stats.ttest_rel(pp, dp, alternative="greater")
            delta = np.mean(pp) - np.mean(dp)
            c2    = "PASS" if p < 0.05 and delta > 0 else "FAIL"
            print(f"  N={N}: ΔP={delta:+.3f}  p={p:.4f}  C2={c2}")
        if len(res["hv_pacan"]) >= 2:
            hp = res["hv_pacan"]; hd = res["hv_det"]
            t2, p2 = sp_stats.ttest_rel(hp, hd, alternative="greater")
            ratio  = np.mean(hp) / max(np.mean(hd), 1e-6)
            c3     = "PASS" if p2 < 0.01 else "FAIL"
            print(f"  N={N}: HV_ratio={ratio:.3f}  p={p2:.4f}  C3={c3}")

    # Variance decomp summary
    if all_results:
        N_last = sorted(all_results.keys())[-1]
        vd_list = all_results[N_last]["var_decomp"]
        if vd_list:
            print(f"\nVariance decomposition (N={N_last}):")
            for comp in ["model", "opt", "noise", "residual"]:
                m = np.mean([v[comp] for v in vd_list])
                print(f"  {comp:10s}: {m:.1f}%")

    # ── Baseline comparison (largest N run) ───────────────────────────────────
    N_bl    = N_list[-1]
    T_bl    = get_T_anneal(N_bl, quick)
    K_ev_bl = get_K_eval(N_bl, quick)
    params_bl = WCParams(N=N_bl, model_seed=0)
    wc_bl     = ThreePhaseWC(params_bl, device=DEVICE)
    Ap_bl, Ah_bl = make_network_attractors(params_bl)

    print(f"\n{'='*55}")
    print(f"Baseline comparison: N={N_bl}, σ={sigma_opt}, T={T_bl}")
    print(f"{'='*55}")

    bl_res = {}
    def run_pacan_bl(ns):
        arch = PACANOptimizer(wc_bl, Ap_bl, Ah_bl,
                               sigma_opt=sigma_opt,
                               K_explore=K_explore, K_eval=K_ev_bl,
                               T_anneal=T_bl, d_grid=d_grid,
                               use_gibbs=True, opt_seed=ns, noise_seed=ns
                               ).run()
        return max((s["p_succ"] for s in arch), default=0.)

    def run_sa_bl(ns):
        arch = PACANOptimizer(wc_bl, Ap_bl, Ah_bl,
                               sigma_opt=sigma_opt,
                               K_explore=K_explore, K_eval=K_ev_bl,
                               T_anneal=T_bl, d_grid=d_grid,
                               use_gibbs=False, opt_seed=ns, noise_seed=ns
                               ).run()
        return max((s["p_succ"] for s in arch), default=0.)

    for name, fn in [
        ("PACAN",      run_pacan_bl),
        ("SA",         run_sa_bl),
        ("Det.greedy", lambda ns: baseline_greedy(
                           wc_bl,Ap_bl,Ah_bl,N_bl,sigma_opt,ns,
                           d_grid=d_grid,K=K_baseline)[2]),
        ("Random",     lambda ns: baseline_random_search(
                           wc_bl,Ap_bl,Ah_bl,N_bl,sigma_opt,ns,
                           d_grid=d_grid,K=K_baseline)[2]),
        ("Degree",     lambda ns: baseline_degree(
                           wc_bl,Ap_bl,Ah_bl,N_bl,sigma_opt,ns,
                           max(1,N_bl//5),d_grid=d_grid,K=K_baseline)[2]),
        ("FVS",        lambda ns: baseline_fvs(
                           wc_bl,Ap_bl,Ah_bl,N_bl,sigma_opt,ns,
                           d_grid=d_grid,K=K_baseline)[2]),
    ]:
        vals = [fn(ns) for ns in range(n_bl_seeds)]
        bl_res[name] = vals
        print(f"  {name:12s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    plot_baselines(bl_res, N_bl, sigma_opt, save_dir)
    print(f"\nAll figures saved to {save_dir}/")
    return all_results, bl_res


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PACAN Week 6")
    parser.add_argument("--quick",       action="store_true",
                        help="Quick test: N=[10], 2 seeds, K=500, T=25")
    parser.add_argument("--N_list",      nargs="+", type=int,
                        default=[10, 20, 30, 50])
    parser.add_argument("--model_seeds", nargs="+", type=int,
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--opt_seeds",   nargs="+", type=int,
                        default=[0, 1, 2])
    parser.add_argument("--noise_seeds", nargs="+", type=int,
                        default=[0, 1, 2])
    parser.add_argument("--K_final",     type=int, default=5000)
    parser.add_argument("--K_baseline",  type=int, default=200)
    parser.add_argument("--K_explore",   type=int, default=80)
    parser.add_argument("--n_bl_seeds",  type=int, default=3)
    parser.add_argument("--save_dir",    type=str, default="results_w6")
    parser.add_argument("--validate",    action="store_true",
                        help="Run validation suite only, then exit")
    args = parser.parse_args()

    if args.quick:
        args.N_list      = [10]
        args.model_seeds = [0, 1]
        args.opt_seeds   = [0]
        args.noise_seeds = [0]
        args.K_final     = 500
        args.K_baseline  = 100
        args.K_explore   = 40
        args.n_bl_seeds  = 2
        print("QUICK MODE — reduced parameters for testing")

    if args.validate:
        run_validation()
    else:
        run_validation()   # always run validation first
        main(args)
