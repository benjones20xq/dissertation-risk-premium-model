# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist

DEBUG = True
def dprint(*args, **kwargs):
    if DEBUG: print(*args, **kwargs)

# ============================ Simulation ============================

def simulate_assets(N=10_000, S0=(100, 100), Lambda_true=(0.02, 0.06),
                    sigma=np.array([[0.2, 0.1], [0.1, 0.3]]),
                    r=0.05, dt=1/252, seed=1, plot=False):
    """
    Two-asset Euler scheme with constant 2x2 sigma.
    mu_i = r + lam1*sigma_{i1} + lam2*sigma_{i2}.
    We solve (least-squares) for (lam1,lam2) to approximately match requested
    asset scalar premia Lambda_true via row norms.
    """
    np.random.seed(seed)

    A = np.array([[sigma[0,0], sigma[0,1]],
                  [sigma[1,0], sigma[1,1]]], dtype=float)
    Sig1 = np.linalg.norm(sigma[0])
    Sig2 = np.linalg.norm(sigma[1])
    b = np.array([Lambda_true[0]*Sig1, Lambda_true[1]*Sig2], dtype=float)
    lam1, lam2 = np.linalg.lstsq(A, b, rcond=None)[0]
    dprint(f"True factor premia: lam1={lam1:.4f}, lam2={lam2:.4f}")

    S1 = np.zeros(N); S2 = np.zeros(N)
    S1[0], S2[0] = S0
    dB = np.random.randn(2, N) * np.sqrt(dt)
    mu1 = r + lam1*sigma[0,0] + lam2*sigma[0,1]
    mu2 = r + lam1*sigma[1,0] + lam2*sigma[1,1]
    for t in range(1, N):
        S1[t] = S1[t-1] * (1 + mu1*dt + sigma[0,0]*dB[0,t] + sigma[0,1]*dB[1,t])
        S2[t] = S2[t-1] * (1 + mu2*dt + sigma[1,0]*dB[0,t] + sigma[1,1]*dB[1,t])
    S = pd.DataFrame({'S1': S1, 'S2': S2})

    if plot and DEBUG:
        plt.figure(figsize=(10,3))
        plt.plot(S['S1'], label='S1'); plt.plot(S['S2'], label='S2'); plt.legend()
        plt.title('Simulated asset prices'); plt.tight_layout(); plt.show()
    return S, sigma

# ========================== Rolling σ(t) 2×2 ========================

def estimate_sigmas_eigen(S, dt=1/252, window=252):
    """
    Rolling cov of log-returns over 'window'; symmetric sqrt → instantaneous σ(t).
    """
    logrets = np.log(S).diff().dropna()
    sigmas, covs = [], []
    for i in range(window, len(logrets)):
        cov = np.cov(logrets.iloc[i-window:i].T, bias=True) / dt
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 0, None)
        sig_est = V @ np.diag(np.sqrt(w)) @ V.T
        covs.append(cov*dt)
        sigmas.append(sig_est)
    idx = logrets.index[window:]
    return idx, np.array(sigmas), np.array(covs)

def align_sigma_columns(rolling_sigmas, ref=None):
    """
    Stabilise column order/sign of 2×2 σ(t) to avoid factor swapping.
    """
    sigs = np.asarray(rolling_sigmas)
    T = sigs.shape[0]
    aligned = np.empty_like(sigs)
    prev = sigs[0] if ref is None else ref
    for t in range(T):
        S = sigs[t]
        best, best_err = None, np.inf
        for swap in (False, True):
            S2 = S[:, [1,0]] if swap else S.copy()
            for s1 in (+1,-1):
                for s2 in (+1,-1):
                    C = S2.copy()
                    C[:,0] *= s1; C[:,1] *= s2
                    err = np.linalg.norm(C - prev, 'fro')
                    if err < best_err: best, best_err = C, err
        aligned[t] = best; prev = best
    return aligned

# ========================= Synthetic portfolios ====================

def construct_V1(S, weights=(0.5,0.5)):
    w1, w2 = weights
    return pd.Series(w1*S['S1'].values + w2*S['S2'].values, index=S.index, name='V1')

def calculate_X_Y(S, sigmas, weights=(0.5,0.5)):
    """
    X = w1*S1*σ11 + w2*S2*σ21
    Y = w1*S1*σ12 + w2*S2*σ22
    """
    w1, w2 = weights
    sig = np.asarray(sigmas)
    S1 = S['S1'].values; S2 = S['S2'].values
    X = w1*S1*sig[:,0,0] + w2*S2*sig[:,1,0]
    Y = w1*S1*sig[:,0,1] + w2*S2*sig[:,1,1]
    return X, Y

def construct_V2_value_preserving(S, sigmas, X, Y, V1=None, eps=1e-10):
    """
    Orthogonal risky long–short, scaled to preserve value at each rebalance.
    P0=B, Q0=-A so dW2 = (Y dB1 - X dB2)/Phi.
    """
    S1, S2 = S['S1'].to_numpy(), S['S2'].to_numpy()
    sig = np.asarray(sigmas)
    s11, s12 = sig[:,0,0], sig[:,0,1]
    s21, s22 = sig[:,1,0], sig[:,1,1]
    A = S1*(X*s11 + Y*s12)
    B = S2*(X*s21 + Y*s22)
    P0, Q0 = B, -A

    n = len(S1)
    P = np.zeros(n); Q = np.zeros(n); V2 = np.zeros(n)
    V2[0] = float(V1.iloc[0]) if V1 is not None else 0.5*(S1[0]+S2[0])
    denom0 = P0[0]*S1[0] + Q0[0]*S2[0]
    k0 = 0.0 if abs(denom0) < eps*(abs(S1[0])+abs(S2[0])+1.0) else V2[0]/denom0
    P[0], Q[0] = k0*P0[0], k0*Q0[0]
    for t in range(1, n):
        V_pre = P[t-1]*S1[t] + Q[t-1]*S2[t]
        denom = P0[t]*S1[t] + Q0[t]*S2[t]
        ill = (abs(A[t]-B[t]) < eps*(abs(A[t])+abs(B[t])+1.0)) or (abs(denom) < eps*(abs(S1[t])+abs(S2[t])+1.0))
        if ill:
            P[t], Q[t], V2[t] = P[t-1], Q[t-1], V_pre
            continue
        k = V_pre/denom
        P[t], Q[t] = k*P0[t], k*Q0[t]
        V2[t] = P[t]*S1[t] + Q[t]*S2[t]
    idx = S.index
    return (pd.Series(P, idx, name='P'),
            pd.Series(Q, idx, name='Q'),
            pd.Series(V2, idx, name='V2'),
            pd.Series(A, idx, name='A'),
            pd.Series(B, idx, name='B'))

# ========================= Instantaneous σ(t) 1D ===================

def ewma_sigma_dt(prices, dt=1/252, alpha=0.94, use_logreturns=True):
    """
    Instantaneous diffusion σ(t) (per √year). EWMA on return^2, then divide by dt.
    """
    s = pd.Series(prices)
    rets = (np.log(s).diff() if use_logreturns else s.pct_change()).dropna()
    var = np.zeros(len(rets))
    var[0] = np.var(rets.iloc[:20]) if len(rets) >= 20 else np.var(rets)
    for t in range(1, len(rets)):
        var[t] = alpha*var[t-1] + (1-alpha)*(rets.iloc[t-1]**2)
    sigma_inst = np.sqrt(np.maximum(var, 0.0) / max(dt, 1e-12))
    return pd.Series(sigma_inst, index=rets.index)

# ====================== Likelihood (Gaussian) ======================

def eq47_gaussian(s_t, s_prev, sigma_inst, r_annual, lam_grid, prior_pdf, dt, return_type='log'):
    """
    r_log: log(S_t/S_{t-1})  ~ N( (r + λσ - 0.5σ^2)dt, σ^2 dt )
    r_ari: (S_t/S_{t-1}-1)   ~ N( (r + λσ)dt,         σ^2 dt )
    """
    if not (np.isfinite(s_prev) and np.isfinite(s_t)):
        return np.nan, prior_pdf

    sig = float(sigma_inst) if (np.isfinite(sigma_inst) and sigma_inst > 0) else 1e-8
    if return_type == 'log':
        if s_prev <= 0 or s_t <= 0: return np.nan, prior_pdf
        r_obs = np.log(s_t/s_prev)
        mu = (r_annual + lam_grid*sig - 0.5*sig*sig) * dt
    else:
        if s_prev == 0: return np.nan, prior_pdf
        r_obs = (s_t/s_prev) - 1.0
        mu = (r_annual + lam_grid*sig) * dt

    var = (sig*sig) * dt
    ll = -0.5 * ((r_obs - mu)**2 / (var + 1e-16))
    w = np.exp(ll - ll.max()) * prior_pdf
    Z = w.sum() + 1e-16
    post = w / Z
    lam_hat = (lam_grid * post).sum()
    return lam_hat, post

# ========================= Prior construction ======================

def _implied_lambda_window(series_al, sigma_al, prior_end, dt, r, return_type):
    """Model-implied λ_t in the prior window using instantaneous σ(t)."""
    s = series_al.iloc[:prior_end+1].values
    sig = sigma_al.iloc[:prior_end+1].values
    if return_type == 'log':
        r_obs = np.log(s[1:] / s[:-1])
        mu_part = (r - 0.5 * sig[1:]**2) * dt
    else:
        r_obs = (s[1:] / s[:-1]) - 1.0
        mu_part = (r * dt)
    num = r_obs - mu_part
    den = np.clip(sig[1:] * dt, 1e-10, None)
    lam_inst = num / den
    return lam_inst[np.isfinite(lam_inst)]

def _mom_gamma_params(x):
    """Stable method-of-moments Gamma(a, theta) fit on positive data."""
    x = np.asarray(x); x = x[np.isfinite(x) & (x > 0)]
    if x.size < 4:
        return 1.0, 0.3
    m = float(np.mean(x))
    v = float(np.var(x, ddof=1))
    v = max(v, 1e-6); m = max(m, 1e-6)
    a = max(m*m / v, 0.5)          # shape
    th = max(v / m, 1e-4)          # scale
    return a, th

def _two_piece_gamma_prior(grid, lam_sample):
    """
    Two asymmetric Gammas: one on λ>=0, one on |λ| for λ<0, mixed by tail weight.
    Uses method-of-moments (stable).
    """
    lam_sample = np.asarray(lam_sample)
    pos = lam_sample[lam_sample >= 0.0]
    neg = -lam_sample[lam_sample <  0.0]  # magnitudes

    a_p, th_p = _mom_gamma_params(pos)
    a_n, th_n = _mom_gamma_params(neg)

    w_pos = len(pos) / max(len(lam_sample), 1)
    w_neg = 1.0 - w_pos

    g = np.zeros_like(grid, dtype=float)
    mask_p = grid >= 0.0
    if mask_p.any():
        g[mask_p] = w_pos * gamma_dist.pdf(grid[mask_p], a_p, loc=0.0, scale=th_p)
    mask_n = grid < 0.0
    if mask_n.any():
        x = -grid[mask_n]
        g[mask_n] = w_neg * gamma_dist.pdf(x, a_n, loc=0.0, scale=th_n)

    Z = float(np.trapz(g, grid)) + 1e-16
    return g / Z

def _anchor_gamma_pdf(grid, mean=0.04, loc=0.0, shape=2.0):
    """
    One-sided anchor Gamma with given mean, loc, shape.
    mean = loc + shape*scale  =>  scale = (mean - loc)/shape.
    """
    scale = max((mean - loc) / max(shape, 1e-6), 1e-6)
    g = gamma_dist.pdf(np.clip(grid - loc, 0.0, None), a=shape, loc=0.0, scale=scale)
    Z = float(np.trapz(g, grid)) + 1e-16
    return g / Z

def construct_initial_prior(
    series, grid, prior_end=252, method="fixed",
    prior_family="gamma", symmetric=True, shape=1.0, scale=0.25,
    mu0=0.0, sigma0=0.30, sigma_series=None, dt=1/252, r=0.05, return_type='log',
    # anchor knobs (applied only for Gamma families)
    anchor_on=False, anchor_mean=0.04, anchor_loc=0.0, anchor_shape=2.0, anchor_weight=0.25
):
    """
    π(λ) on 'grid'.

    Gamma options:
      - method="estimate_gamma":
          Shifted one-sided Gamma with fixed loc = low-quantile of implied λ,
          fit (shape, scale) only; clamps over-flat scales.
      - method="estimate_gamma_two_piece":
          Asymmetric two-piece: Gamma on λ≥0 and Gamma on |λ| for λ<0 (MoM).
      - method="fixed":
          Uses provided (shape, scale, mu0). If symmetric=True, mirror around mu0.

    Normal/Laplace remain unchanged (kept for completeness).
    """
    fam = prior_family.lower()

    # Data-driven Gamma paths
    if fam == "gamma" and method in ("estimate_gamma", "estimate_gamma_two_piece"):
        if sigma_series is None:
            raise ValueError("sigma_series required for Gamma estimation.")
        s_al, sig_al = series.align(sigma_series, join='inner')
        s_al = s_al.iloc[1:]; sig_al = sig_al.loc[s_al.index]
        if len(s_al) <= prior_end:
            raise ValueError(f"prior_end={prior_end} too large for series length {len(s_al)}")

        lam_sample = _implied_lambda_window(s_al, sig_al, prior_end, dt, r, return_type)
        lam_sample = lam_sample[np.isfinite(lam_sample)]
        if lam_sample.size < 4:
            base = gamma_dist.pdf(np.clip(grid - 0.0, 0, None), a=1.0, loc=0.0, scale=0.5)
            g = base / (np.trapz(base, grid) + 1e-16)
        else:
            if method == "estimate_gamma_two_piece":
                g = _two_piece_gamma_prior(grid, lam_sample)
            else:
                # estimate_gamma: shifted one-sided Gamma with fixed loc
                loc0 = float(np.quantile(lam_sample, 0.02))
                x = lam_sample - loc0
                x = x[x > 0]
                if x.size < 4:
                    x = lam_sample - loc0
                    x = np.clip(x, 1e-6, None)
                # fit shape/scale with loc fixed at 0
                a_hat, _, th_hat = gamma_dist.fit(x, floc=0.0)
                a_hat = float(max(a_hat, 0.5))
                grid_span = float(grid.max() - grid.min())
                th_hat = float(np.clip(th_hat, 1e-4, 0.75*grid_span))
                g = gamma_dist.pdf(np.clip(grid - loc0, 0.0, None), a_hat, loc=0.0, scale=th_hat)
                g /= (np.trapz(g, grid) + 1e-16)

        # optional anchor blend
        if anchor_on:
            g_anchor = _anchor_gamma_pdf(grid, mean=anchor_mean, loc=anchor_loc, shape=anchor_shape)
            w = float(np.clip(anchor_weight, 0.0, 1.0))
            g = (1.0 - w) * g + w * g_anchor
            g /= float(np.trapz(g, grid)) + 1e-16
        return g

    # ------------- fixed gamma -------------
    if fam == "gamma":
        if symmetric:
            base = gamma_dist.pdf(np.abs(grid - mu0), a=shape, scale=scale)
            g = base
        else:
            pos = (grid - mu0).clip(min=0.0)
            base = gamma_dist.pdf(pos, a=shape, scale=scale)
            base[grid < mu0] = 0.0
            g = base
        # optional anchor blend even for fixed
        if anchor_on:
            g_anchor = _anchor_gamma_pdf(grid, mean=anchor_mean, loc=anchor_loc, shape=anchor_shape)
            w = float(np.clip(anchor_weight, 0.0, 1.0))
            g = (1.0 - w) * g + w * g_anchor
        return g / (np.trapz(g, grid) + 1e-16)

    # ---------- keep normal/laplace branches for completeness ----------
    if fam == "normal":
        z = (grid - mu0) / max(sigma0, 1e-12)
        g = np.exp(-0.5 * z * z)
        return g / (np.trapz(g, grid) + 1e-16)

    if fam == "laplace":
        b = max(scale, 1e-6)
        g = 0.5 / b * np.exp(-np.abs(grid - mu0) / b)
        return g / (np.trapz(g, grid) + 1e-16)

    raise ValueError("Unknown prior_family. Use 'gamma', 'normal', or 'laplace'.")

# ======================== Grid-Bayes filter ========================

def iterative_lambda_series(series, sigma_series, lam_grid, prior_end=252, return_type='log',
                            dt=1/252, r=0.05, prior_method="fixed", return_debug=False,
                            prior_family="gamma", symmetric=True, shape=1.0, scale=0.25,
                            mu0=0.0, sigma0=0.30,
                            # anchor passthrough
                            anchor_on=False, anchor_mean=0.04, anchor_loc=0.0,
                            anchor_shape=2.0, anchor_weight=0.25):
    """
    Single-asset grid Bayes filter for λ.
    """
    s_al, sig_al = series.align(sigma_series, join='inner')
    s_al = s_al.iloc[1:]; sig_al = sig_al.loc[s_al.index]
    if len(s_al) <= prior_end:
        raise ValueError(f"prior_end={prior_end} too large for series length {len(s_al)}")

    prior_pdf = construct_initial_prior(
        s_al, lam_grid, prior_end=prior_end, method=prior_method,
        prior_family=prior_family, symmetric=symmetric, shape=shape, scale=scale,
        mu0=mu0, sigma0=sigma0, sigma_series=sig_al, dt=dt, r=r, return_type=return_type,
        anchor_on=anchor_on, anchor_mean=anchor_mean, anchor_loc=anchor_loc,
        anchor_shape=anchor_shape, anchor_weight=anchor_weight
    )

    s_prev = s_al.iloc[prior_end-1]
    out = []; post = prior_pdf
    for s_t, sigma_t in zip(s_al.iloc[prior_end:], sig_al.iloc[prior_end:]):
        lam_hat, post = eq47_gaussian(s_t, s_prev, sigma_t, r, lam_grid, post, dt, return_type=return_type)
        out.append(lam_hat); s_prev = s_t

    idx = s_al.index[prior_end:]
    lam_df = pd.DataFrame({'lambda_est': out}, index=idx)

    if return_debug:
        return lam_df, {
            'prior_grid': lam_grid,
            'initial_prior_pdf': prior_pdf,
            'series_aligned': s_al,
            'sigma_aligned': sig_al,
            'aligned_index': idx
        }
    return lam_df

# ========== Reconstruct Λ1, Λ2 from (λ0, λ~0, X, Y, σ) ==============

def reconstruct_Lambdas(lambda0_df, lambda_tilde_df, sigmas, X, Y, eps=1e-12):
    lam0 = np.asarray(lambda0_df['lambda_est'])
    lamt = np.asarray(lambda_tilde_df['lambda_est'])
    X_arr = np.asarray(X); Y_arr = np.asarray(Y)
    s11 = sigmas[:,0,0]; s12 = sigmas[:,0,1]
    s21 = sigmas[:,1,0]; s22 = sigmas[:,1,1]
    m = min(len(lam0), len(lamt), len(X_arr), len(Y_arr), len(s11), len(s12), len(s21), len(s22))
    lam0, lamt = lam0[:m], lamt[:m]; X_arr, Y_arr = X_arr[:m], Y_arr[:m]
    s11, s12, s21, s22 = s11[:m], s12[:m], s21[:m], s22[:m]
    Phi = np.sqrt(np.maximum(X_arr**2 + Y_arr**2, eps))
    Sigma1 = np.sqrt(np.maximum(s11**2 + s12**2, eps))
    Sigma2 = np.sqrt(np.maximum(s21**2 + s22**2, eps))
    num1 = lamt*(s11*Y_arr - s12*X_arr) + lam0*(s12*Y_arr + s11*X_arr)
    num2 = lamt*(s21*Y_arr - s22*X_arr) + lam0*(s22*Y_arr + s21*X_arr)
    Lam1 = num1 / (Sigma1*Phi)
    Lam2 = num2 / (Sigma2*Phi)
    idx = lambda0_df.index[:m]
    return (pd.DataFrame({'lambda_est': Lam1}, index=idx),
            pd.DataFrame({'lambda_est': Lam2}, index=idx))

# ============================= Pipelines ===========================

def vector_lambda_pipeline(S, sigma_true=None, weights=(0.5,0.5), dt=1/252, window=252,
                           ewma_alpha=0.94, prior_end=252, grid1=None, grid2=None,
                           r=0.05, v1_return_type='log', v2_return_type='arithmetic',
                           prior_method="estimate_gamma",
                           prior_family="gamma", symmetric=True, shape=1.0, scale=0.25,
                           mu0=0.0, sigma0=0.30,
                           # anchor passthrough
                           anchor_on=False, anchor_mean=0.04, anchor_loc=0.0,
                           anchor_shape=2.0, anchor_weight=0.25):
    idx, sigmas_raw, _ = estimate_sigmas_eigen(S, dt=dt, window=window)
    S_roll = S.loc[idx]
    sigmas_roll = align_sigma_columns(sigmas_raw, ref=sigma_true if sigma_true is not None else sigmas_raw[0])

    V1 = construct_V1(S_roll, weights=weights)
    X, Y = calculate_X_Y(S_roll, sigmas_roll, weights=weights)
    P, Q, V2, A, B = construct_V2_value_preserving(S_roll, sigmas_roll, X, Y, V1=V1)

    sig_V1 = ewma_sigma_dt(V1, dt=dt, alpha=ewma_alpha, use_logreturns=True)
    sig_V2 = ewma_sigma_dt(V2, dt=dt, alpha=ewma_alpha, use_logreturns=False)

    V1a, sV1a = V1.align(sig_V1, join='inner'); V1a = V1a.iloc[1:]; sV1a = sV1a.loc[V1a.index]
    V2a, sV2a = V2.align(sig_V2, join='inner'); V2a = V2a.iloc[1:]; sV2a = sV2a.loc[V2a.index]

    if grid1 is None: grid1 = np.linspace(-0.5, 1.5, 400)
    if grid2 is None: grid2 = np.linspace(-1.0, 2.0, 500)

    lam0_df, dbg1 = iterative_lambda_series(
        V1a, sV1a, grid1, prior_end=prior_end, return_type=v1_return_type,
        dt=dt, r=r, prior_method=prior_method, return_debug=True,
        prior_family=prior_family, symmetric=symmetric, shape=shape, scale=scale,
        mu0=mu0, sigma0=sigma0,
        anchor_on=anchor_on, anchor_mean=anchor_mean, anchor_loc=anchor_loc,
        anchor_shape=anchor_shape, anchor_weight=anchor_weight
    )
    lamt_df, dbg2 = iterative_lambda_series(
        V2a, sV2a, grid2, prior_end=prior_end, return_type=v2_return_type,
        dt=dt, r=r, prior_method=prior_method, return_debug=True,
        prior_family=prior_family, symmetric=symmetric, shape=shape, scale=scale,
        mu0=mu0, sigma0=sigma0,
        anchor_on=anchor_on, anchor_mean=anchor_mean, anchor_loc=anchor_loc,
        anchor_shape=anchor_shape, anchor_weight=anchor_weight
    )

    base_off = V1.index.get_loc(V1a.index[0])
    aidx = base_off + np.arange(len(lam0_df))
    aidx = aidx[aidx < len(S_roll)]
    X_al, Y_al = X[aidx], Y[aidx]
    sig_al     = sigmas_roll[aidx]
    Lam1_df, Lam2_df = reconstruct_Lambdas(lam0_df, lamt_df, sig_al, X_al, Y_al)

    return {
        'idx_prices': idx,
        'sigmas_raw': sigmas_raw,
        'sigmas_roll': sigmas_roll,
        'V1': V1, 'V2': V2,
        'sig_V1': sig_V1, 'sig_V2': sig_V2,
        'lam0_df': lam0_df, 'lamt_df': lamt_df,
        'Lambda1_df': Lam1_df, 'Lambda2_df': Lam2_df,
        'dbg_V1': dbg1, 'dbg_V2': dbg2
    }

def single_asset_pipeline(S, prior_end=252, dt=1/252, ewma_alpha=0.94, grid=None,
                          r=0.05, prior_method="estimate_gamma", return_debug=False,
                          prior_family="gamma", symmetric=True, shape=1.0, scale=0.25,
                          mu0=0.0, sigma0=0.30,
                          anchor_on=False, anchor_mean=0.04, anchor_loc=0.0,
                          anchor_shape=2.0, anchor_weight=0.25):
    sig1 = ewma_sigma_dt(S['S1'], dt=dt, alpha=ewma_alpha, use_logreturns=True)
    sig2 = ewma_sigma_dt(S['S2'], dt=dt, alpha=ewma_alpha, use_logreturns=True)
    if grid is None: grid = np.linspace(-0.5, 1.5, 400)

    Lam1_df, dbg1 = iterative_lambda_series(
        S['S1'], sig1, grid, prior_end=prior_end, return_type='log', dt=dt, r=r,
        prior_method=prior_method, return_debug=True,
        prior_family=prior_family, symmetric=symmetric, shape=shape, scale=scale,
        mu0=mu0, sigma0=sigma0,
        anchor_on=anchor_on, anchor_mean=anchor_mean, anchor_loc=anchor_loc,
        anchor_shape=anchor_shape, anchor_weight=anchor_weight
    )
    Lam2_df, dbg2 = iterative_lambda_series(
        S['S2'], sig2, grid, prior_end=prior_end, return_type='log', dt=dt, r=r,
        prior_method=prior_method, return_debug=True,
        prior_family=prior_family, symmetric=symmetric, shape=shape, scale=scale,
        mu0=mu0, sigma0=sigma0,
        anchor_on=anchor_on, anchor_mean=anchor_mean, anchor_loc=anchor_loc,
        anchor_shape=anchor_shape, anchor_weight=anchor_weight
    )

    out = {'Lambda1_df': Lam1_df, 'Lambda2_df': Lam2_df}
    if return_debug:
        out.update({'dbg_S1': dbg1, 'dbg_S2': dbg2, 'sig1': sig1, 'sig2': sig2})
    return out

# ============================== Plots ==============================

def plot_final_lambdas(Lambda1_df, Lambda2_df, true=None, title=r'Estimated $\hat{\Lambda}_1$ and $\hat{\Lambda}_2$ (Vector)'):
    plt.figure(figsize=(12,3.5))
    plt.plot(Lambda1_df.index, Lambda1_df['lambda_est'], color='C0', lw=2, label=r'$\hat{\Lambda}_1$ (Vector)')
    plt.plot(Lambda2_df.index, Lambda2_df['lambda_est'], color='C1', lw=2, label=r'$\hat{\Lambda}_2$ (Vector)')
    if true is not None:
        plt.plot(Lambda1_df.index, np.full(len(Lambda1_df), true[0]), '--', color='C0', lw=1.6, label=r'True $\Lambda_1$')
        plt.plot(Lambda2_df.index, np.full(len(Lambda2_df), true[1]), '--', color='C1', lw=1.6, label=r'True $\Lambda_2$')
    plt.xlabel('Step'); plt.ylabel(r'$\Lambda$'); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()

def plot_compare_methods(L1_vec, L2_vec, L1_single, L2_single, true=None):
    fig, ax = plt.subplots(1, 2, figsize=(14,3.8), sharex=False)

    A1 = L1_vec['lambda_est'].rename('Vector').to_frame().join(
         L1_single['lambda_est'].rename('Single'), how='inner')
    A2 = L2_vec['lambda_est'].rename('Vector').to_frame().join(
         L2_single['lambda_est'].rename('Single'), how='inner')

    ax[0].plot(A1.index, A1['Vector'], color='C0', lw=2, label=r'$\hat{\Lambda}_1$ Vector', alpha=0.7)
    ax[0].plot(A1.index, A1['Single'], color='skyblue', lw=1.6, ls='--', label=r'$\hat{\Lambda}_1$ Single', alpha=0.7)
    if true is not None:
        ax[0].plot(A1.index, np.full(len(A1), true[0]), ':', color='C0', lw=1.4, label=r'True $\Lambda_1$', alpha=0.7)

    ax[1].plot(A2.index, A2['Vector'], color='C1', lw=2, label=r'$\hat{\Lambda}_2$ Vector', alpha=0.7)
    ax[1].plot(A2.index, A2['Single'], color='lightsalmon', lw=1.6, ls='--', label=r'$\hat{\Lambda}_2$ Single', alpha=0.7)
    if true is not None:
        ax[1].plot(A2.index, np.full(len(A2), true[1]), ':', color='C1', lw=1.4, label=r'True $\Lambda_2$', alpha=0.7)

    ax[0].set_title('Asset 1'); ax[1].set_title('Asset 2')
    for a in ax:
        a.set_xlabel('Step'); a.set_ylabel(r'$\Lambda$'); a.legend(); a.grid(alpha=.15)
    fig.suptitle(r'Comparison: Vector vs Single-Asset $\hat{\Lambda}$', y=1.02)
    plt.tight_layout(); plt.show()

def plot_prior_pdf(grid, prior_pdf, title=r'Initial prior on $\lambda$'):
    plt.figure(figsize=(6,3))
    plt.plot(grid, prior_pdf, lw=2)
    plt.title(title); plt.xlabel(r'$\lambda$'); plt.ylabel('density')
    plt.grid(alpha=.2); plt.tight_layout(); plt.show()

def plot_asset_sigmas(sig1, sig2, title=r'Instantaneous $\sigma(t)$ (assets)'):
    fig, ax = plt.subplots(1,2, figsize=(12,3), sharex=False)
    ax[0].plot(sig1.index, sig1.values, lw=1.2, label=r'$\hat{\sigma}_1$')
    ax[1].plot(sig2.index, sig2.values, lw=1.2, label=r'$\hat{\sigma}_2$', color='C1')
    ax[0].legend(); ax[1].legend()
    for a in ax: a.set_xlabel('Step'); a.set_ylabel(r'$\sigma$'); a.grid(alpha=.15)
    fig.suptitle(title, y=1.02)
    plt.tight_layout(); plt.show()

def plot_portfolio_sigmas(sig_V1, sig_V2, title=r'Instantaneous $\sigma(t)$ (V1, V2)'):
    fig, ax = plt.subplots(1,2, figsize=(12,3), sharex=False)
    ax[0].plot(sig_V1.index, sig_V1.values, lw=1.2, label=r'$\hat{\sigma}_{V1}$')
    ax[1].plot(sig_V2.index, sig_V2.values, lw=1.2, label=r'$\hat{\sigma}_{V2}$', color='C1')
    ax[0].legend(); ax[1].legend()
    for a in ax: a.set_xlabel('Step'); a.set_ylabel(r'$\sigma$'); a.grid(alpha=.15)
    fig.suptitle(title, y=1.02)
    plt.tight_layout(); plt.show()

def plot_sigma_norms(sigmas_roll, title=r'Asset norms $\Sigma_i(t)$ from 2x2 $\sigma(t)$'):
    sig = np.asarray(sigmas_roll)
    s11, s12 = sig[:,0,0], sig[:,0,1]
    s21, s22 = sig[:,1,0], sig[:,1,1]
    Sig1 = np.sqrt(s11**2+s12**2); Sig2 = np.sqrt(s21**2+s22**2)
    plt.figure(figsize=(12,3))
    plt.plot(Sig1, lw=1.2, label=r'$\hat{\Sigma}_1$')
    plt.plot(Sig2, lw=1.2, label=r'$\hat{\Sigma}_2$')
    plt.title(title); plt.xlabel('Window step'); plt.ylabel(r'$\Sigma$'); plt.legend()
    plt.grid(alpha=.2); plt.tight_layout(); plt.show()

# =============================== main ==============================

def main():
    # --- Mode ---
    MODE = "vector"               # "vector", "single", or "both"

    # --- Simulation ---
    N = 1400
    S0 = (150,100)
    TRUE_LAMBDA = (0.05, 0.07)            # reference lines for plots only
    SIGMA_CONST = np.array([[0.2, 0.1],
                            [0.1, 0.15]])
    r = 0.05
    dt = 1/252
    seed = 3

    # --- Estimation knobs ---
    weights = (0.6, 0.4)        # V1 weights
    window = 252                # rolling window for 2x2 σ(t)
    ewma_alpha = 0.94           # smoothing for instantaneous σ(t)
    prior_end = 252             # prior window length
    v1_return_type = 'log'          # 'log' for V1 (positive)
    v2_return_type = 'log'   # 'arithmetic' for V2 (long-short)

    # --- Prior controls (Gamma-focused + Anchor) ---
    PRIOR_METHOD    = "estimate_gamma_two_piece"      # "estimate_gamma" | "estimate_gamma_two_piece" | "fixed"
    PRIOR_FAMILY    = "gamma"
    PRIOR_SYMMETRIC = False                 # only used by "fixed" gamma
    PRIOR_SHAPE     = 1.2                   # used by "fixed"
    PRIOR_SCALE     = 0.25                  # used by "fixed"
    PRIOR_MU0       = 0.04                   # used by "fixed"
    PRIOR_SIGMA0    = 0.30                  # for "normal" (unused here)

    # Anchor toward mean ≈ 0.04 (applies to gamma priors)
    PRIOR_ANCHOR_ON     = True
    PRIOR_ANCHOR_MEAN   = 0.04
    PRIOR_ANCHOR_LOC    = 0.0
    PRIOR_ANCHOR_SHAPE  = 2.0
    PRIOR_ANCHOR_WEIGHT = 0.25

    # --- Grids ---
    grid_single = np.linspace(-0.5, 1.5, 400)
    grid_v1     = np.linspace(-0.5, 1.5, 400)
    grid_v2     = np.linspace(-1.0, 2.0, 500)

    # --- Plot toggles ---
    SHOW_PRICES              = True
    SHOW_SIG_ASSETS          = True
    SHOW_SIG_PORTFOLIOS      = True
    SHOW_SIGMA_MATRIX_NORMS  = True
    SHOW_PRIOR_SINGLE        = True
    SHOW_PRIOR_VECTOR        = True
    SHOW_VECTOR_LAMBDAS      = True
    SHOW_COMPARISON          = True

    # --- Simulate ---
    S, sigma_true = simulate_assets(N=N, S0=S0, Lambda_true=TRUE_LAMBDA,
                                    sigma=SIGMA_CONST, r=r, dt=dt, seed=seed,
                                    plot=SHOW_PRICES)

    # --- Pipelines ---
    out_vec = out_single = None

    if MODE in ("vector", "both"):
        out_vec = vector_lambda_pipeline(
            S, sigma_true=SIGMA_CONST, weights=weights, dt=dt, window=window,
            ewma_alpha=ewma_alpha, prior_end=prior_end,
            grid1=grid_v1, grid2=grid_v2, r=r,
            v1_return_type=v1_return_type, v2_return_type=v2_return_type,
            prior_method=PRIOR_METHOD, prior_family=PRIOR_FAMILY,
            symmetric=PRIOR_SYMMETRIC, shape=PRIOR_SHAPE, scale=PRIOR_SCALE,
            mu0=PRIOR_MU0, sigma0=PRIOR_SIGMA0,
            anchor_on=PRIOR_ANCHOR_ON, anchor_mean=PRIOR_ANCHOR_MEAN,
            anchor_loc=PRIOR_ANCHOR_LOC, anchor_shape=PRIOR_ANCHOR_SHAPE,
            anchor_weight=PRIOR_ANCHOR_WEIGHT
        )
        if SHOW_SIG_PORTFOLIOS:
            plot_portfolio_sigmas(out_vec['sig_V1'], out_vec['sig_V2'])
        if SHOW_SIGMA_MATRIX_NORMS:
            plot_sigma_norms(out_vec['sigmas_roll'])
        if SHOW_PRIOR_VECTOR:
            plot_prior_pdf(out_vec['dbg_V1']['prior_grid'], out_vec['dbg_V1']['initial_prior_pdf'],
                           title=r'Initial prior for $\lambda_0$ (V1)')
            plot_prior_pdf(out_vec['dbg_V2']['prior_grid'], out_vec['dbg_V2']['initial_prior_pdf'],
                           title=r'Initial prior for $\tilde{\lambda}_0$ (V2)')
        if SHOW_VECTOR_LAMBDAS and MODE == "vector":
            plot_final_lambdas(out_vec['Lambda1_df'], out_vec['Lambda2_df'], true=TRUE_LAMBDA)

    if MODE in ("single", "both"):
        out_single = single_asset_pipeline(
            S, prior_end=prior_end, dt=dt, ewma_alpha=ewma_alpha,
            grid=grid_single, r=r, prior_method=PRIOR_METHOD, return_debug=True,
            prior_family=PRIOR_FAMILY, symmetric=PRIOR_SYMMETRIC, shape=PRIOR_SHAPE,
            scale=PRIOR_SCALE, mu0=PRIOR_MU0, sigma0=PRIOR_SIGMA0,
            anchor_on=PRIOR_ANCHOR_ON, anchor_mean=PRIOR_ANCHOR_MEAN,
            anchor_loc=PRIOR_ANCHOR_LOC, anchor_shape=PRIOR_ANCHOR_SHAPE,
            anchor_weight=PRIOR_ANCHOR_WEIGHT
        )
        if SHOW_SIG_ASSETS:
            plot_asset_sigmas(out_single['sig1'], out_single['sig2'])
        if SHOW_PRIOR_SINGLE:
            plot_prior_pdf(grid_single, out_single['dbg_S1']['initial_prior_pdf'],
                           title=r'Initial prior for single-asset $\lambda$ (S1)')
            plot_prior_pdf(grid_single, out_single['dbg_S2']['initial_prior_pdf'],
                           title=r'Initial prior for single-asset $\lambda$ (S2)')

    if MODE == "both" and SHOW_COMPARISON:
        plot_compare_methods(out_vec['Lambda1_df'], out_vec['Lambda2_df'],
                             out_single['Lambda1_df'], out_single['Lambda2_df'],
                             true=TRUE_LAMBDA)
    out_vec = vector_lambda_pipeline(S, sigma_true=SIGMA_CONST)

# Plot V1 and V2
    plt.figure(figsize=(10,4))
    plt.plot(out_vec['V1'], label='V1 (synthetic long-only)')
    plt.plot(out_vec['V2'], label='V2 (orthogonal long-short)')
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.title("Synthetic portfolios V1 and V2")
    plt.tight_layout()
    plt.show()
    
    L1 = out_vec['Lambda1_df']   # expected to have a column like 'lambda' or 'lambda_mean'
    L2 = out_vec['Lambda2_df']
    
    # be robust to column naming:
    def get_lambda_series(df):
        for c in ['lambda', 'lambda_mean', 'Lambda', 'Lambda_mean']:
            if c in df.columns:
                return df[c].values
        # fallback: assume the first numeric column is the lambda series
        return df.select_dtypes('number').iloc[:,0].values
    
    lam1 = get_lambda_series(L1)
    lam2 = get_lambda_series(L2)
    
    plt.figure(figsize=(10,4))
    plt.plot(lam1, label=r'$\hat{\Lambda}_1$ (for $V_1$)')
    plt.plot(lam2, label=r'$\hat{\Lambda}_2$ (for $V_2$)')
    plt.xlabel('Time step')
    plt.ylabel('Implied market price of risk')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    post1 = out_vec['posteriors_factor1']     # e.g. T x G
    post2 = out_vec['posteriors_factor2']     # e.g. T x G
    lambda_grid = out_vec['lambda_grid']      # e.g. length G
    
    # compute posterior means over time
    lam1 = (post1 * lambda_grid[None, :]).sum(axis=1)
    lam2 = (post2 * lambda_grid[None, :]).sum(axis=1)
    
    plt.figure(figsize=(10,4))
    plt.plot(lam1, label=r'$\hat{\Lambda}_1$ (for $V_1$)')
    plt.plot(lam2, label=r'$\hat{\Lambda}_2$ (for $V_2$)')
    plt.xlabel('Time step')
    plt.ylabel('Implied market price of risk')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    def _pick_lambda_series(df):
        """Return a 1D np.array for the posterior-mean lambda from a DataFrame-like."""
        # common column names you might be using
        for c in ['lambda', 'lambda_mean', 'Lambda', 'Lambda_mean', 'posterior_mean']:
            if hasattr(df, 'columns') and c in df.columns:
                return np.asarray(df[c].values, dtype=float)
        # fallback: take first numeric column
        if hasattr(df, 'select_dtypes'):
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] > 0:
                return np.asarray(num.iloc[:, 0].values, dtype=float)
        # if it's already an array/series:
        return np.asarray(df, dtype=float)
    
    def plot_synthetic_before_reconstruction(out_vec, true=None, burn_in=None, save=None):
        """
        Plots synthetic lambdas for V1,V2 *before reconstruction*.
        Optionally also plots synthetic portfolios V1,V2 if present in out_vec.
    
        Parameters
        ----------
        out_vec : dict
            Expected keys:
              - 'Lambda1_df', 'Lambda2_df' (posterior means over time for V1,V2), or
              - 'posteriors_factor1', 'posteriors_factor2' (T x G arrays) and 'lambda_grid'
              - optionally 'V1', 'V2' (portfolio value series)
        true : tuple or None
            (true_L1, true_L2) to draw as dashed reference lines.
        burn_in : int or None
            If set, shades [0, burn_in) as a burn-in region.
        save : str or None
            If set, path to save PNG (e.g., 'synthetic_lambdas.png').
        """
    
        # --- get lambda series for V1,V2 before reconstruction ---
        if 'Lambda1_df' in out_vec and 'Lambda2_df' in out_vec:
            lam1 = _pick_lambda_series(out_vec['Lambda1_df'])
            lam2 = _pick_lambda_series(out_vec['Lambda2_df'])
        elif all(k in out_vec for k in ['posteriors_factor1', 'posteriors_factor2', 'lambda_grid']):
            post1 = np.asarray(out_vec['posteriors_factor1'], dtype=float)  # T x G
            post2 = np.asarray(out_vec['posteriors_factor2'], dtype=float)  # T x G
            grid  = np.asarray(out_vec['lambda_grid'], dtype=float)         # G
            lam1 = (post1 * grid[None, :]).sum(axis=1)
            lam2 = (post2 * grid[None, :]).sum(axis=1)
        else:
            raise KeyError("Provide either Lambda1_df/Lambda2_df, or posteriors_factor{1,2}+lambda_grid.")
    
        # --- plot lambdas (synthetic space) ---
        plt.figure(figsize=(10, 4))
        plt.plot(lam1, label=r'$\hat{\Lambda}_1$  (synthetic $V_1$)')
        plt.plot(lam2, label=r'$\hat{\Lambda}_2$  (synthetic $V_2$)')
        if burn_in:
            plt.axvspan(0, int(burn_in), alpha=0.08)
        if true:
            t1, t2 = true
            plt.axhline(t1, linestyle='--', linewidth=1)
            plt.axhline(t2, linestyle='--', linewidth=1)
        plt.xlabel('Time step')
        plt.ylabel('Implied market price of risk')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.show()
    
        # --- optional: portfolios V1,V2 if present (still synthetic space) ---
        if 'V1' in out_vec and 'V2' in out_vec:
            plt.figure(figsize=(10, 4))
            plt.plot(out_vec['V1'], label='V1 (synthetic)')
            plt.plot(out_vec['V2'], label='V2 (synthetic)')
            if burn_in:
                plt.axvspan(0, int(burn_in), alpha=0.08)
            plt.xlabel('Time step')
            plt.ylabel('Portfolio value')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            if save:
                base = save.rsplit('.', 1)[0] + '_V1V2.png'
                plt.savefig(base, dpi=300, bbox_inches='tight')
            plt.show()
            
    out_vec = vector_lambda_pipeline(S, sigma_true=SIGMA_CONST)
    plot_synthetic_before_reconstruction(out_vec, true=(0.05, 0.07), burn_in=250,
                                         save='synthetic_lambdas_before_recon.png')

if __name__ == "__main__":
    main()
