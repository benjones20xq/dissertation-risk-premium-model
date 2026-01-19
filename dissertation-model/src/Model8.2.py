# ============================================
# Model8.2 — compact, self-contained (real + artificial)
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
import yfinance as yf
from scipy.stats import gamma as _gamma

# --- Palette fallback (keeps your custom palette if present) ---
if 'palette' not in globals() or not isinstance(globals().get('palette'), dict):
    palette = {"S1": "C0", "S2": "C1", "grid": "0.85"}

# =========================
# OPTIONS (streamlined)
# =========================
REAL_OR_ARTIFICIAL = "artifical"     # "real" to import real asset histories, "artifical" to simulate

# Real data options
REAL_TICKERS = ("AMZN", "SPY")
REAL_START_DATE, REAL_END_DATE = "1980-01-01", None   # None -> latest
BUFFER_BDAYS = 504

# Simulation options (used if REAL_OR_ARTIFICIAL starts with "art")
years = 40
steps_per_year = 252
S0 = (100.0, 50.0)
r = 0.05
sigma = np.array([[0.25, 0.10],
                  [0.1, 0.15]], dtype=float)
lam = np.array([0.08, 0.02], dtype=float)
seed = 1

# Estimation windows / params
eigen_window = 252
ewma_alpha   = 0.94

# Plots
DO_PLOT_PRICES = True
DO_PLOT_EIGEN  = True
DO_PLOT_EWMA   = True

# Single-asset pipelines on S1/S2
DO_RUN_SINGLE_ASSET  = True
DO_RUN_MF    = True          # return-likelihood + Gamma prior
DO_RUN_INFO  = True          # info-process + discrete prior
DO_PLOT_SINGLE_ASSET_COMBINED   = True   # one figure, 2 panels
DO_PLOT_SINGLE_ASSET_INDIVIDUAL = False  # separate figures (one per asset)

SINGLE_ASSET_PRIOR_YEARS  = 50
SINGLE_ASSET_PRIOR_METHOD = "estimate"   # 'estimate' | 'fixed'
LAMBDA_GRID_LO, LAMBDA_GRID_HI, LAMBDA_GRID_N = 0.0, 1.0, 401
PRIOR_DISC_LO, PRIOR_DISC_HI, PRIOR_DISC_N    = -0.6, 0.6, 41   # info-process prior grid

# Virtual portfolios (V1,V2) + reconstruction
DO_RUN_VP                 = True
DO_PLOT_VP_PRICES         = True
DO_PLOT_VP_MF             = True
DO_PLOT_VP_INFO           = True
DO_RECONSTRUCT            = True   # reconstruct Λ₁,Λ₂ (and optionally λ₁,λ₂)
DO_PLOT_VECTOR_LAMBDAS    = True   # plot λ₁, λ₂ (factor)
DO_PLOT_CAPITAL_LAMBDAS   = True   # plot Λ₁, Λ₂ (asset-direction)
DO_PLOT_CAPITAL_VS_SINGLE = True   # compare Λ vs single-asset λ for S1,S2

PLOT_TITLES = {
    "prices":        "Simulated asset prices",
    "eigen":         "Sigma matrix (Eigen)",
    "ewma":          "EWMA scalar volatilities",
    "sa_mf":         "Return-likelihood λ on S1 & S2",
    "sa_info":       "Info-process λ on S1 & S2",
    "vp_mf":         "Return-likelihood λ on V1 & V2",
    "vp_info":       "Info-process λ on V1 & V2",
    "vector":        "Factor λ estimates",
    "capital":       "Asset-direction Λ estimates",
    "cap_vs_single": "Λ (reconstructed) vs single-asset λ",
    "vp_prices":     "Virtual portfolios V1 & V2",
}

# ======================================
# CORE MODEL & UTILITIES
# ======================================
def simulate_correlated_artifical_assets(years, steps_per_year, S0, r, sigma, lam, seed=None):
    if seed is not None:
        np.random.seed(seed)
    N  = int(years * steps_per_year) + 1
    dt = 1.0 / steps_per_year
    lam = np.asarray(lam, float)
    sigma = np.asarray(sigma, float)
    mu1 = r + sigma[0] @ lam
    mu2 = r + sigma[1] @ lam
    n1  = np.linalg.norm(sigma[0])
    n2  = np.linalg.norm(sigma[1])
    dW = np.random.randn(N-1, 2) * np.sqrt(dt)
    S1 = np.empty(N); S2 = np.empty(N)
    S1[0], S2[0] = S0
    for t in range(N-1):
        z1 = sigma[0,0]*dW[t,0] + sigma[0,1]*dW[t,1]
        z2 = sigma[1,0]*dW[t,0] + sigma[1,1]*dW[t,1]
        S1[t+1] = S1[t] * np.exp((mu1 - 0.5*n1**2)*dt + z1)
        S2[t+1] = S2[t] * np.exp((mu2 - 0.5*n2**2)*dt + z2)
    return pd.DataFrame({"S1": S1, "S2": S2})

def prep_log_returns(S, steps_per_year, epsilon=0.0):
    if epsilon > 0.0: S = S.clip(lower=epsilon)
    R = np.log(S).diff().dropna()
    dt = 1.0 / steps_per_year
    return R, dt

def estimate_sigmas_eigen(S, steps_per_year, window=252):
    R, dt = prep_log_returns(S, steps_per_year)
    T = len(R)
    if T < window:
        return R.index[0:0], np.empty((0,2,2)), np.empty((0,2,2))
    rolling_covs, rolling_sigmas = [], []
    for i in range(window, T):
        cov = np.cov(R.iloc[i-window:i].T, bias=True)   # 2x2
        eigvals, eigvecs = np.linalg.eigh(cov / dt)
        eigvals = np.clip(eigvals, 0.0, None)
        sigma_est = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        rolling_covs.append(cov); rolling_sigmas.append(sigma_est)
    idx = R.index[window:]
    return idx, np.asarray(rolling_sigmas), np.asarray(rolling_covs)

def sigma_ewma(R, dt, alpha=0.94):
    X = R.to_numpy()
    T, K = X.shape
    v = np.empty((T, K))
    a, b = float(alpha), 1.0 - float(alpha)
    v0 = X[0] ** 2
    for t in range(T):
        if t == 0: v[t] = a * v0 + b * (X[t] ** 2)
        else:      v[t] = a * v[t - 1] + b * (X[t] ** 2)
    sig = np.sqrt(v / dt)
    return pd.DataFrame(sig, index=R.index, columns=[f"sigma_{c}" for c in R.columns])

# ===== Single-asset prior + likelihood (inlined from your earlier model) =====
def gamma_param(df, start_idx, end_idx, method="estimate"):
    if method == "estimate":
        window   = df.loc[start_idx:end_idx, "Close"]
        logrets  = np.log(window).diff().dropna()
        mu_ann   = logrets.mean() * 252
        sigma_ann= logrets.std(ddof=0) * np.sqrt(252)
        rr       = 0.05
        lam_hat  = (mu_ann - rr) / sigma_ann if sigma_ann > 0 else 0.2
        scale    = 0.2
        shape    = max(lam_hat / scale, 0.5)
        return {"shape": shape, "scale": scale}
    elif method == "fixed":
        return {"shape": 1.2, "scale": 0.2}
    else:
        raise ValueError("gamma_param: method must be 'estimate' or 'fixed'.")

def gamma_pdf_3p2(x, params):
    return _gamma.pdf(x, a=params["shape"], scale=params["scale"])

def initial_pdf(lambda_grid, prior_params, floor=1e-12, mix=1e-3):
    g = gamma_pdf_3p2(lambda_grid, prior_params).astype(float)
    g = np.maximum(g, float(floor))
    if mix and mix > 0.0:
        u = np.ones_like(g, float) / g.size
        g = (1.0 - mix) * g + mix * u
    g_sum = g.sum()
    return g / g_sum if g_sum > 0 else np.ones_like(g)/g.size

def eq47(s_t, s0, sigma_step, r_step, lambda_grid, prior_pdf, dt, t, date):
    A = np.log(s_t / s0) * (lambda_grid / sigma_step)
    B = -0.5 * (lambda_grid**2) * t
    C =  0.5 * sigma_step * t * lambda_grid
    D = -r_step * t * (lambda_grid / sigma_step)
    exponent = A + B + C + D
    w = np.exp(exponent) * prior_pdf
    denom = w.sum()
    posterior_pdf = w / denom if denom > 0 else np.ones_like(w)/w.size
    numer = (lambda_grid * np.exp(exponent) * prior_pdf).sum()
    lambda_estimate = numer / denom if denom > 0 else np.nan
    return lambda_estimate, posterior_pdf, date
# (These mirror your previous definitions so downstream results match.)  # :contentReference[oaicite:1]{index=1}

def iterative_lambda(df, prior_method, prior_start, prior_end,
                     lambda_start, lambda_end, lambda_grid=None,
                     dt=1/252, sigma_series=None):
    if lambda_grid is None:
        lambda_grid = np.linspace(0, 1, 200)
    prior_params = gamma_param(df, prior_start, prior_end, method=prior_method)
    prior_pdf = initial_pdf(lambda_grid, prior_params)
    dates = df.loc[lambda_start:lambda_end].index
    results = []

    first_idx = dates[0]
    if isinstance(first_idx, (int, np.integer)):
        s_prev = df.loc[first_idx - 1, "Close"] if (first_idx - 1) in df.index else df.iloc[0]["Close"]
    else:
        try:
            prev_date = first_idx - BDay(1)
            s_prev = df.loc[prev_date, "Close"] if prev_date in df.index else df.iloc[0]["Close"]
        except Exception:
            s_prev = df.iloc[0]["Close"]

    t_accum = dt
    for date in dates:
        df_idx = df.index.get_loc(date)
        sigma_idx = df_idx - 1 if df_idx > 0 else 0
        if sigma_series is None or len(sigma_series) == 0:
            sigma_step = 0.20
        else:
            sigma_step = sigma_series[sigma_idx] if sigma_idx < len(sigma_series) else sigma_series[-1]
        s_t = df.loc[date, "Close"]
        lam_est, post_pdf, _ = eq47(s_t, s_prev, sigma_step, 0.05, lambda_grid, prior_pdf, dt, dt, date)
        results.append({"date": date, "lambda_est": lam_est, "posterior_pdf": post_pdf.copy()})
        prior_pdf = post_pdf
        s_prev = s_t
        t_accum += dt

    return pd.DataFrame(results).set_index("date")

def make_prior(lo=-0.6, hi=0.6, n=41, probs=None):
    x = np.linspace(lo, hi, int(n), dtype=float)
    p = (np.ones_like(x)/x.size) if probs is None else np.asarray(probs, float)
    p = p/p.sum()
    if p.size != x.size: raise ValueError("probs length must equal n")
    return x, p

def info_incr(V, sigma, r, dt, eps=1e-10):
    V = pd.Series(V, dtype=float)
    ret = V.pct_change().dropna()
    if isinstance(sigma, pd.Series):
        s = sigma.astype(float)
    else:
        arr = np.asarray(sigma, float).reshape(-1)
        s = pd.Series(arr, index=V.index[-len(arr):])
    s = s.reindex(ret.index, method="ffill")
    if s.isna().any():
        s = s.fillna(method="bfill")
        if s.isna().any():
            s = s.fillna(float(s.dropna().iloc[0]) if s.dropna().size else 1.0)
    sig = s.to_numpy()
    sig_finite = np.isfinite(sig)
    smin = max(eps, np.nanpercentile(sig[sig_finite], 0.5)*1e-6)
    smax = np.nanpercentile(sig[sig_finite], 99.5)*10.0
    sig = np.clip(sig, smin, smax)
    dxi = (ret.to_numpy() / sig) - (r * dt) / sig
    return pd.Series(dxi, index=ret.index, name="d_xi")

def filter_disc(x, p0, dxi, dt):
    x = np.asarray(x, float)
    wlog = np.log(np.asarray(p0, float)/np.sum(p0))
    lam, posts = [], []
    for z in dxi.to_numpy(float):
        wlog = wlog + x*z - 0.5*(x**2)*dt
        m = np.max(wlog); w = np.exp(wlog - m); w = w/w.sum()
        lam.append((w*x).sum()); posts.append(w.copy())
        wlog = np.log(w)
    return pd.DataFrame({"lambda_est": np.asarray(lam, float),
                         "posterior_pdf": posts}, index=dxi.index)

# ------ Virtual portfolios & reconstruction ------
def construct_V1(S, weights=(0.5,0.5)):
    w1, w2 = weights
    return pd.Series(w1*S['S1'].values + w2*S['S2'].values, index=S.index, name='V1')

def calculate_X_Y(S, sigmas, weights=(0.5,0.5)):
    w1, w2 = weights
    sig = np.asarray(sigmas, float)
    n = len(S)
    if sig.ndim == 2:
        sig = np.broadcast_to(sig, (n, 2, 2))
    elif sig.shape[0] != n:
        m = min(n, sig.shape[0]); S = S.iloc[-m:]; sig = sig[-m:]
    S1 = S['S1'].to_numpy(float); S2 = S['S2'].to_numpy(float)
    X = w1*S1*sig[:,0,0] + w2*S2*sig[:,1,0]
    Y = w1*S1*sig[:,0,1] + w2*S2*sig[:,1,1]
    return (pd.Series(X, index=S.index, name='X'),
            pd.Series(Y, index=S.index, name='Y'))

def construct_V2_value_preserving(S, sigmas, X, Y, V1=None, eps=1e-10):
    idx = S.index
    S1 = S["S1"].to_numpy(float); S2 = S["S2"].to_numpy(float)
    sig = np.asarray(sigmas, float)
    n = len(S1)
    if sig.ndim == 2:
        sig = np.broadcast_to(sig, (n, 2, 2))
    elif sig.shape[0] != n:
        m = min(n, sig.shape[0])
        S1, S2, X, Y = S1[-m:], S2[-m:], np.asarray(X)[-m:], np.asarray(Y)[-m:]
        idx = idx[-m:]; sig = sig[-m:]
    s11, s12 = sig[:, 0, 0], sig[:, 0, 1]
    s21, s22 = sig[:, 1, 0], sig[:, 1, 1]
    A = S1 * (X * s11 + Y * s12)
    B = S2 * (X * s21 + Y * s22)
    P0, Q0 = B, -A
    P = np.zeros_like(S1); Q = np.zeros_like(S2); V2 = np.zeros_like(S1)
    V2[0] = float(V1.iloc[0]) if V1 is not None else 0.5 * (S1[0] + S2[0])
    denom0 = P0[0] * S1[0] + Q0[0] * S2[0]
    k0 = 0.0 if abs(denom0) < eps * (abs(S1[0]) + abs(S2[0]) + 1.0) else V2[0] / denom0
    P[0], Q[0] = k0 * P0[0], k0 * Q0[0]
    for t in range(1, len(S1)):
        V_pre = P[t-1] * S1[t] + Q[t-1] * S2[t]
        denom = P0[t] * S1[t] + Q0[t] * S2[t]
        if abs(denom) < eps * (abs(S1[t]) + abs(S2[t]) + 1.0):
            P[t], Q[t], V2[t] = P[t-1], Q[t-1], V_pre
            continue
        k = V_pre / denom
        P[t], Q[t] = k * P0[t], k * Q0[t]
        V2[t] = P[t] * S1[t] + Q[t] * S2[t]
    return (pd.Series(P, idx, name="P"),
            pd.Series(Q, idx, name="Q"),
            pd.Series(V2, idx, name="V2"),
            pd.Series(A, idx, name="A"),
            pd.Series(B, idx, name="B"))

def compute_sigma_V1_V2(S, sigmas, V1, P, Q, V2, X, Y, eps=1e-12):
    sig = np.asarray(sigmas, float)
    nS = len(S)
    if sig.ndim == 2:
        sig = np.broadcast_to(sig, (nS, 2, 2))
    m = min(nS, sig.shape[0], len(V1), len(V2), len(P), len(Q), len(X), len(Y))
    idx = S.index[-m:]
    S1 = S["S1"].to_numpy(float)[-m:]; S2 = S["S2"].to_numpy(float)[-m:]
    V1a = V1.to_numpy(float)[-m:];     V2a = V2.to_numpy(float)[-m:]
    Pa  = P.to_numpy(float)[-m:];      Qa  = Q.to_numpy(float)[-m:]
    Xa  = np.asarray(X, float)[-m:];   Ya  = np.asarray(Y, float)[-m:]
    s11 = sig[-m:,0,0]; s12 = sig[-m:,0,1]; s21 = sig[-m:,1,0]; s22 = sig[-m:,1,1]
    phi1 = np.sqrt(np.maximum(Xa*Xa + Ya*Ya, eps))
    sigma_V1 = pd.Series(phi1 / np.maximum(V1a, eps), index=idx, name="sigma_V1")
    Z1 = Pa*S1*s11 + Qa*S2*s21
    Z2 = Pa*S1*s12 + Qa*S2*s22
    phi2 = np.sqrt(np.maximum(Z1*Z1 + Z2*Z2, eps))
    sigma_V2 = pd.Series(phi2 / np.maximum(V2a, eps), index=idx, name="sigma_V2")
    return sigma_V1, sigma_V2

def reconstruct_lambdas(lambda0_df, lambdat_df, sigmas, X, Y, eps=1e-12, return_factor=False):
    lam0 = np.asarray(lambda0_df["lambda_est"], float)
    lamt = np.asarray(lambdat_df["lambda_est"], float)
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    sig = np.asarray(sigmas, float)
    if sig.ndim == 2:
        T = max(len(lam0), len(lamt), len(X), len(Y))
        sig = np.broadcast_to(sig, (T, 2, 2))
    s11, s12, s21, s22 = sig[:,0,0], sig[:,0,1], sig[:,1,0], sig[:,1,1]
    m = min(len(lam0), len(lamt), len(X), len(Y), len(s11))
    lam0, lamt = lam0[-m:], lamt[-m:]
    X, Y = X[-m:], Y[-m:]
    s11, s12, s21, s22 = s11[-m:], s12[-m:], s21[-m:], s22[-m:]
    Phi    = np.sqrt(np.maximum(X*X + Y*Y, eps))
    Sigma1 = np.sqrt(np.maximum(s11*s11 + s12*s12, eps))
    Sigma2 = np.sqrt(np.maximum(s21*s21 + s22*s22, eps))
    num1 = lam0*(s11*X + s12*Y) + lamt*(s11*Y - s12*X)
    num2 = lam0*(s21*X + s22*Y) + lamt*(s21*Y - s22*X)
    Lam1 = num1 / (Phi*Sigma1)
    Lam2 = num2 / (Phi*Sigma2)
    idx = lambda0_df.index[-m:]
    out1 = pd.DataFrame({"lambda_est": Lam1}, index=idx)
    out2 = pd.DataFrame({"lambda_est": Lam2}, index=idx)
    if not return_factor:
        return out1, out2
    lam1 = (X*lam0 + Y*lamt)/Phi
    lam2 = (Y*lam0 - X*lamt)/Phi
    out_l1 = pd.DataFrame({"lambda_est": lam1}, index=idx)
    out_l2 = pd.DataFrame({"lambda_est": lam2}, index=idx)
    return out1, out2, out_l1, out_l2

# ======================================================
# Importer (no sigma work inside; clean + strict)
# ======================================================
def import_prep_asset(ticker, start_date, end_date, prior_years=5, steps_per_year=252, buffer_bdays=0):
    dt = 1.0 / steps_per_year
    start_eff = pd.Timestamp(start_date) - buffer_bdays * BDay(1) if buffer_bdays else pd.Timestamp(start_date)
    df_yf = yf.download(ticker, start=start_eff, end=end_date, auto_adjust=True, progress=False)
    if df_yf.empty or "Close" not in df_yf:
        raise ValueError(f"No price data for {ticker} in this range.")
    close_obj = df_yf["Close"]
    if isinstance(close_obj, pd.DataFrame):
        close_obj = close_obj.iloc[:, 0]
    closes_1d = np.asarray(close_obj, dtype=float).reshape(-1)
    dates = df_yf.index.to_list()
    df = pd.DataFrame({"Close": closes_1d, "lambda": np.nan, "date": dates},
                      index=np.arange(len(closes_1d), dtype=int))
    N = len(df)
    prior_len    = max(2, min(int(prior_years * steps_per_year), max(N - 2, 2)))
    lambda_start = prior_len
    lambda_end   = N - 1
    return {"df": df, "prior_start": 0, "prior_end": prior_len,
            "lambda_start": lambda_start, "lambda_end": lambda_end, "dt": dt}

# =============================
# Plotting (raw, no rebase/log)
# =============================
def _t_years(idx, steps_per_year):
    try:
        return np.asarray(idx, float) / steps_per_year
    except Exception:
        if hasattr(idx, "to_numpy") and np.issubdtype(idx.to_numpy().dtype, np.datetime64):
            t0 = idx[0]; return (idx - t0).days / 365.25
        return np.arange(len(idx), dtype=float) / steps_per_year

def plot_S(S, steps_per_year, mode="artificial", labels=None, title=None):
    L = labels or {}; n1, n2 = L.get("S1", "S1"), L.get("S2", "S2")
    if title is None:
        title = "Simulated asset prices" if mode.startswith("art") else f"Asset prices: {n1} vs {n2}"
    t = np.arange(len(S)) / steps_per_year
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.2))
    ax.plot(t, S["S1"].to_numpy(float), label=n1, color=palette["S1"], lw=1.8)
    ax.plot(t, S["S2"].to_numpy(float), label=n2, color=palette["S2"], lw=1.8)
    ax.set_title(title); ax.set_xlabel("Time (years)"); ax.set_ylabel("Price")
    ax.grid(alpha=0.25, color=palette["grid"]); ax.legend(frameon=False)
    fig.tight_layout(); return fig, ax

def plot_sigma_matrix_eigen(idx, sigma_est, steps_per_year, sigma_true=None, alpha_est=0.6, mode="artificial"):
    show_truth = (mode.startswith("art") and (sigma_true is not None) and (np.ndim(sigma_true) == 2))
    t = np.arange(len(idx)) / steps_per_year
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, sigma_est[:, 0, 0], label='σ11 (eigen est.)', color='C0', alpha=alpha_est, lw=1.8)
    if show_truth: axes[0].plot(t, np.full_like(t, sigma_true[0, 0]), label='σ11 (true)', color='C0', lw=1.4, alpha=0.5)
    axes[0].set_ylabel('σ11'); axes[0].grid(alpha=0.25)
    axes[1].plot(t, sigma_est[:, 1, 1], label='σ22 (eigen est.)', color='C3', alpha=alpha_est, lw=1.8)
    if show_truth: axes[1].plot(t, np.full_like(t, sigma_true[1, 1]), label='σ22 (true)', color='C3', lw=1.4, alpha=0.5)
    axes[1].set_ylabel('σ22'); axes[1].grid(alpha=0.25)
    off_est = 0.5 * (sigma_est[:, 0, 1] + sigma_est[:, 1, 0])
    axes[2].plot(t, off_est, label='σ12=σ21 (eigen est.)', color='C2', alpha=alpha_est, lw=1.8)
    if show_truth:
        axes[2].plot(t, np.full_like(t, sigma_true[0, 1]), label='σ12 (true)', color='C2', lw=1.4, alpha=0.5)
        axes[2].plot(t, np.full_like(t, sigma_true[1, 0]), label='σ21 (true)', color='C1', lw=1.4, alpha=0.5)
    axes[2].set_ylabel('σ12 / σ21'); axes[2].set_xlabel('Years'); axes[2].grid(alpha=0.25)
    for a in axes: a.legend(frameon=False, loc='upper right')
    fig.suptitle('Sigma matrix (Eigen est.)' if not show_truth else 'Sigma matrix (Eigen)', x=0.03, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes

def plot_sigma_ewma(sig_df, steps_per_year, row_norms_true=None, alpha_est=0.6, mode="artificial", labels=None):
    L = labels or {}; n1, n2 = L.get("S1", "S1"), L.get("S2", "S2")
    show_truth = (mode.startswith("art") and (row_norms_true is not None))
    t = np.arange(len(sig_df)) / steps_per_year
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    ax.plot(t, sig_df["sigma_S1"].to_numpy(float), label=f"{n1} σ (EWMA)", color="C0", alpha=alpha_est, lw=1.8)
    ax.plot(t, sig_df["sigma_S2"].to_numpy(float), label=f"{n2} σ (EWMA)", color="C1", alpha=alpha_est, lw=1.8)
    if show_truth:
        ax.plot(t, np.full_like(t, row_norms_true[0]), label=f"{n1} σ (true)", color="C0", lw=1.4, alpha=0.5)
        ax.plot(t, np.full_like(t, row_norms_true[1]), label=f"{n2} σ (true)", color="C1", lw=1.4, alpha=0.5)
    ax.set_xlabel("Years"); ax.set_ylabel("σ"); ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); return fig, ax

def plot_simple_model_lambdas(S, lambda_results, steps_per_year, title=None, mode="artificial", labels=None):
    L = labels or {}
    if title is None:
        title = "Return-likelihood λ (simulated)" if mode.startswith("art") else "Estimated λ (return-likelihood)"
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    g = palette.get("grid", "0.85")
    for i, col in enumerate(["S1", "S2"]):
        if col not in lambda_results: ax[i].set_visible(False); continue
        res = lambda_results[col]; tL = _t_years(res.index, steps_per_year)
        label_name = L.get(col, col)
        ax[i].plot(tL, res["lambda_est"].to_numpy(float), lw=1.8, color=f"C{i}", label=f"{label_name} λ̂")
        ax[i].set_ylabel("λ"); ax[i].grid(alpha=0.25, color=g); ax[i].legend(frameon=False, loc="upper right", fontsize=9)
    ax[-1].set_xlabel("Time (years)")
    fig.suptitle(title, ha="center"); fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, ax

def plot_single_asset_both_methods(lambda_results_mf, lambda_results_info, steps_per_year,
                                   labels=None, title=None, separate=False):
    L = labels or {}; names = {"S1": L.get("S1", "S1"), "S2": L.get("S2", "S2")}
    base_title = "Single-asset λ — info-process vs likelihood"
    def _align_two(dfA, dfB):
        if (dfA is None) and (dfB is None): return None, None, None
        if dfA is None:
            t = _t_years(dfB.index, steps_per_year); return t, None, dfB["lambda_est"].to_numpy(float)
        if dfB is None:
            t = _t_years(dfA.index, steps_per_year); return t, dfA["lambda_est"].to_numpy(float), None
        idx = dfA.index.intersection(dfB.index)
        t = _t_years(idx, steps_per_year)
        a = dfA.loc[idx, "lambda_est"].to_numpy(float)
        b = dfB.loc[idx, "lambda_est"].to_numpy(float)
        return t, a, b
    g = palette.get("grid","0.85")
    pairs = [("S1", (lambda_results_info or {}).get("S1"), (lambda_results_mf or {}).get("S1")),
             ("S2", (lambda_results_info or {}).get("S2"), (lambda_results_mf or {}).get("S2"))]
    if separate:
        outs = []
        for key, dI, dM in pairs:
            t, yI, yM = _align_two(dI, dM)
            if t is None: continue
            fig, ax = plt.subplots(1, 1, figsize=(11, 4.2))
            if yI is not None: ax.plot(t, yI, lw=1.9, color="C0", label="λ (info-process)")
            if yM is not None: ax.plot(t, yM, lw=1.4, color="C3", label="λ (likelihood)")
            ax.set_ylabel(names[key]); ax.set_xlabel("Time (years)")
            ax.grid(alpha=0.25, color=g); ax.legend(frameon=False, loc="upper right", fontsize=9)
            fig.suptitle(f"{base_title}: {names[key]}", ha="center")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            outs.append((fig, ax))
        return outs
    if title is None: title = base_title
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t1, yI1, yM1 = _align_two(pairs[0][1], pairs[0][2])
    if t1 is not None:
        if yI1 is not None: ax[0].plot(t1, yI1, lw=1.9, color="C0", label="λ (info-process)")
        if yM1 is not None: ax[0].plot(t1, yM1, lw=1.4, color="C3", label="λ (likelihood)")
        ax[0].set_ylabel(names["S1"]); ax[0].grid(alpha=0.25, color=g); ax[0].legend(frameon=False, loc="upper right", fontsize=9)
    else:
        ax[0].set_visible(False)
    t2, yI2, yM2 = _align_two(pairs[1][1], pairs[1][2])
    if t2 is not None:
        if yI2 is not None: ax[1].plot(t2, yI2, lw=1.9, color="C0", label="λ (info-process)")
        if yM2 is not None: ax[1].plot(t2, yM2, lw=1.4, color="C3", label="λ (likelihood)")
        ax[1].set_ylabel(names["S2"]); ax[1].set_xlabel("Time (years)")
        ax[1].grid(alpha=0.25, color=g); ax[1].legend(frameon=False, loc="upper right", fontsize=9)
    else:
        ax[1].set_visible(False)
    fig.suptitle(title, ha="center"); fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, ax

def plot_v1_v2(V1, V2, steps_per_year, title=None, mode="artificial", labels=None, rebase=False):
    L = labels or {}; n1, n2 = L.get("S1", "S1"), L.get("S2", "S2")
    if title is None: title = PLOT_TITLES.get("vp_prices", "Virtual portfolios V1 & V2")
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t1 = _t_years(V1.index, steps_per_year); t2 = _t_years(V2.index, steps_per_year)
    y1 = (V1.to_numpy(float) / float(V1.iloc[0])) if rebase else V1.to_numpy(float)
    y2 = (V2.to_numpy(float) / float(V2.iloc[0])) if rebase else V2.to_numpy(float)
    l1 = f"V1 ({n1})" + (" rebased" if rebase else ""); l2 = f"V2 ({n2})" + (" rebased" if rebase else "")
    ax[0].plot(t1, y1, lw=1.8, color="C0", label=l1); ax[1].plot(t2, y2, lw=1.8, color="C1", label=l2)
    for i in (0, 1): ax[i].grid(alpha=0.25, color=palette.get("grid", "0.85")); ax[i].legend(frameon=False)
    ax[1].set_xlabel("Time (years)"); ax[0].set_ylabel("V1"); ax[1].set_ylabel("V2")
    fig.suptitle(title, ha="center"); fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, ax

def plot_vector_lambda_estimates(l1_df, l2_df, lam_true=None, steps_per_year=252, title=None, mode="artificial", labels=None):
    L = labels or {}; n1, n2 = L.get("S1", "S1"), L.get("S2", "S2")
    show_truth = (mode.startswith("art") and (lam_true is not None))
    if title is None: title = "Factor λ estimates"
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t1 = _t_years(l1_df.index, steps_per_year); t2 = _t_years(l2_df.index, steps_per_year)
    ax[0].plot(t1, l1_df["lambda_est"].to_numpy(float), lw=1.8, label=f"λ₁ {n1} (est.)", color="C0")
    if show_truth: ax[0].axhline(float(lam_true[0]), lw=1.4, color="C0", alpha=0.5)
    ax[0].set_ylabel("λ₁"); ax[0].grid(alpha=0.25, color=palette.get("grid","0.85")); ax[0].legend(frameon=False)
    ax[1].plot(t2, l2_df["lambda_est"].to_numpy(float), lw=1.8, label=f"λ₂ {n2} (est.)", color="C1")
    if show_truth: ax[1].axhline(float(lam_true[1]), lw=1.4, color="C1", alpha=0.5)
    ax[1].set_ylabel("λ₂"); ax[1].set_xlabel("Time (years)")
    ax[1].grid(alpha=0.25, color=palette.get("grid","0.85")); ax[1].legend(frameon=False)
    fig.suptitle(title, ha="center"); fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, ax

def plot_capital_lambda_estimates(Lam1_df, Lam2_df, base_sigma=None, lam_vec=None, steps_per_year=252, title=None, mode="artificial", labels=None):
    L = labels or {}; n1, n2 = L.get("S1", "S1"), L.get("S2", "S2")
    show_truth = (mode.startswith("art") and (base_sigma is not None) and (lam_vec is not None))
    if title is None: title = "Asset-direction Λ estimates"
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t1 = _t_years(Lam1_df.index, steps_per_year); t2 = _t_years(Lam2_df.index, steps_per_year)
    if show_truth:
        s = np.asarray(base_sigma, float); n1n, n2n = np.linalg.norm(s[0]), np.linalg.norm(s[1])
        L1_true = float(s[0] @ np.asarray(lam_vec, float) / n1n)
        L2_true = float(s[1] @ np.asarray(lam_vec, float) / n2n)
    else:
        L1_true = L2_true = None
    ax[0].plot(t1, Lam1_df["lambda_est"].to_numpy(float), lw=1.8, color=palette.get("S1","C0"), label=f"Λ {n1} (recon)")
    if show_truth: ax[0].axhline(L1_true, lw=1.4, color=palette.get("S1","C0"), alpha=0.5)
    ax[0].set_ylabel("Λ"); ax[0].grid(alpha=0.25, color=palette.get("grid","0.85")); ax[0].legend(frameon=False)
    ax[1].plot(t2, Lam2_df["lambda_est"].to_numpy(float), lw=1.8, color=palette.get("S2","C1"), label=f"Λ {n2} (recon)")
    if show_truth: ax[1].axhline(L2_true, lw=1.4, color=palette.get("S2","C1"), alpha=0.5)
    ax[1].set_ylabel("Λ"); ax[1].set_xlabel("Time (years)")
    ax[1].grid(alpha=0.25, color=palette.get("grid","0.85")); ax[1].legend(frameon=False)
    fig.suptitle(title, ha="center"); fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, ax

def plot_capital_vs_single(Lam1_df, Lam2_df, sa_results, base_sigma=None, lam_vec=None, steps_per_year=252,
                           title=None, cut_years=0, mode="artificial", labels=None):
    L = labels or {}; n1, n2 = L.get("S1","S1"), L.get("S2","S2")
    show_truth = (mode.startswith("art") and (base_sigma is not None) and (lam_vec is not None))
    if title is None: title = "Λ vs λ — covariance reconstruction vs single-asset"
    g = palette.get("grid", "0.85")
    def _align_pair(Lam_df, sa_df):
        idx = Lam_df.index.intersection(sa_df.index)
        t   = _t_years(idx, steps_per_year)
        yR  = Lam_df.loc[idx, "lambda_est"].to_numpy(float)
        yS  = sa_df.loc[idx, "lambda_est"].to_numpy(float)
        if cut_years and cut_years > 0:
            m = t >= float(cut_years); t, yR, yS = t[m], yR[m], yS[m]
        return t, yR, yS
    if show_truth:
        s = np.asarray(base_sigma, float); N1, N2 = np.linalg.norm(s[0]), np.linalg.norm(s[1])
        L1_true = float(s[0] @ np.asarray(lam_vec, float) / N1)
        L2_true = float(s[1] @ np.asarray(lam_vec, float) / N2)
    else:
        L1_true = L2_true = None
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True)
    if "S1" in sa_results:
        t1, y1R, y1S = _align_pair(Lam1_df, sa_results["S1"])
        ax[0].plot(t1, y1R, lw=1.9, color=palette.get("S1","C0"), label="Λ (covariance)")
        ax[0].plot(t1, y1S, lw=1.4, color="0.25",                 label="λ (single-asset)")
        if show_truth: ax[0].axhline(L1_true, lw=1.2, color=palette.get("S1","C0"), alpha=0.5)
        ax[0].grid(alpha=0.25, color=g); ax[0].set_ylabel(n1)
    if "S2" in sa_results:
        t2, y2R, y2S = _align_pair(Lam2_df, sa_results["S2"])
        ax[1].plot(t2, y2R, lw=1.9, color=palette.get("S2","C1"))
        ax[1].plot(t2, y2S, lw=1.4, color="0.25")
        if show_truth: ax[1].axhline(L2_true, lw=1.2, color=palette.get("S2","C1"), alpha=0.5)
        ax[1].grid(alpha=0.25, color=g); ax[1].set_ylabel(n2); ax[1].set_xlabel("Time (years)")
    fig.legend(["Λ (covariance)", "λ (single-asset)"], loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, ha="center"); fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig, ax

# =========================
# RUN (raw / no rebase)
# =========================
if __name__ == "__main__":
    figs = []

    # --- Data source switch ---
    _is_art = str(REAL_OR_ARTIFICIAL).lower().startswith("art")
    if _is_art:
        S = simulate_correlated_artifical_assets(years, steps_per_year, S0, r, sigma, lam, seed)
        sigma_true_for_plots = sigma; lam_true_for_plots = lam
        mode   = "artificial"; labels = {"S1": "S1", "S2": "S2"}
    else:
        tickers      = REAL_TICKERS
        start_date   = REAL_START_DATE
        end_date     = REAL_END_DATE
        buffer_bdays = BUFFER_BDAYS
        d1 = import_prep_asset(tickers[0], start_date, end_date, steps_per_year=steps_per_year, buffer_bdays=buffer_bdays)["df"]
        d2 = import_prep_asset(tickers[1], start_date, end_date, steps_per_year=steps_per_year, buffer_bdays=buffer_bdays)["df"]
        merged = (d1.loc[:, ["date", "Close"]].rename(columns={"Close": "S1"})
                    .merge(d2.loc[:, ["date", "Close"]].rename(columns={"Close": "S2"}), on="date", how="inner")
                    .sort_values("date", kind="mergesort"))
        S = pd.DataFrame({"S1": merged["S1"].to_numpy(dtype=float).reshape(-1),
                          "S2": merged["S2"].to_numpy(dtype=float).reshape(-1)})
        sigma_true_for_plots = None; lam_true_for_plots = None
        mode   = "real"; labels = {"S1": tickers[0], "S2": tickers[1]}

    # --- Prices ---
    if DO_PLOT_PRICES:
        f,_ = plot_S(S, steps_per_year, mode=mode, labels=labels); figs.append(f)

    # --- Eigen σ ---
    idx_est, sigma_est, covs_est = estimate_sigmas_eigen(S=S, steps_per_year=steps_per_year, window=eigen_window)
    if DO_PLOT_EIGEN and len(idx_est) > 0:
        f,_ = plot_sigma_matrix_eigen(idx=idx_est, sigma_est=sigma_est, steps_per_year=steps_per_year,
                                      sigma_true=sigma_true_for_plots, mode=mode); figs.append(f)

    # --- EWMA σ ---
    R_all, dt = prep_log_returns(S, steps_per_year)
    sig_df = sigma_ewma(R_all, dt, alpha=ewma_alpha)
    if DO_PLOT_EWMA:
        rn = (np.linalg.norm(sigma[0]), np.linalg.norm(sigma[1])) if (sigma_true_for_plots is not None) else None
        f,_ = plot_sigma_ewma(sig_df=sig_df, steps_per_year=steps_per_year, row_norms_true=rn,
                              mode=mode, labels=labels); figs.append(f)

    # --- Single-asset pipelines ---
    lambda_results_mf, lambda_results_info = {}, {}
    if DO_RUN_SINGLE_ASSET and DO_RUN_MF:
        lam_grid = np.linspace(LAMBDA_GRID_LO, LAMBDA_GRID_HI, int(LAMBDA_GRID_N))
        for col in ("S1","S2"):
            df_asset = pd.DataFrame({"Close": S[col].to_numpy(dtype=float).reshape(-1)}, index=np.arange(len(S)))
            prior_len = max(2, int(min(SINGLE_ASSET_PRIOR_YEARS*steps_per_year, len(df_asset)-2)))
            sigma_series = sig_df[f"sigma_{col}"].to_numpy()
            res = iterative_lambda(df=df_asset, prior_method=SINGLE_ASSET_PRIOR_METHOD,
                                   prior_start=0, prior_end=prior_len, lambda_start=prior_len, lambda_end=len(df_asset)-1,
                                   lambda_grid=lam_grid, dt=1.0/steps_per_year, sigma_series=sigma_series)
            lambda_results_mf[col] = res
    if DO_RUN_SINGLE_ASSET and DO_RUN_INFO:
        xg, p0 = make_prior(PRIOR_DISC_LO, PRIOR_DISC_HI, PRIOR_DISC_N)
        dt_loc = 1.0/steps_per_year
        for col in ("S1","S2"):
            dxi = info_incr(S[col], sig_df[f"sigma_{col}"], r=r, dt=dt_loc)
            res = filter_disc(xg, p0, dxi, dt_loc)
            lambda_results_info[col] = res

    # --- Single-asset plots (combined and/or individual) ---
    if (DO_PLOT_SINGLE_ASSET_COMBINED or DO_PLOT_SINGLE_ASSET_INDIVIDUAL) and (lambda_results_mf or lambda_results_info):
        if DO_PLOT_SINGLE_ASSET_COMBINED:
            f,_ = plot_single_asset_both_methods(lambda_results_mf, lambda_results_info, steps_per_year,
                                                 labels=labels, separate=False); figs.append(f)
        if DO_PLOT_SINGLE_ASSET_INDIVIDUAL:
            outs = plot_single_asset_both_methods(lambda_results_mf, lambda_results_info, steps_per_year,
                                                  labels=labels, separate=True)
            for f,_ in outs: figs.append(f)

    # --- Virtual portfolios (V1,V2) + reconstruction ---
    if DO_RUN_VP:
        sig_base = sigma_est if len(sigma_est) > 0 else (sigma if sigma_true_for_plots is not None else sigma_est)
        V1 = construct_V1(S, (0.5,0.5))
        X, Y = calculate_X_Y(S, sig_base, (0.5,0.5))
        P, Q, V2, A, B = construct_V2_value_preserving(S, sig_base, X, Y, V1=V1)

        if DO_PLOT_VP_PRICES:
            f,_ = plot_v1_v2(V1, V2, steps_per_year, title=PLOT_TITLES.get("vp_prices","Virtual portfolios V1 & V2"),
                             mode=mode, labels=labels, rebase=False); figs.append(f)

        sigma_V1, sigma_V2 = compute_sigma_V1_V2(S, sig_base, V1, P, Q, V2, X, Y)

        lambda_results_mf_V, lambda_results_info_V = {}, {}
        if DO_RUN_MF:
            grid = np.linspace(LAMBDA_GRID_LO, LAMBDA_GRID_HI, int(LAMBDA_GRID_N))
            for name, V, sigV in (("S1", V1, sigma_V1), ("S2", V2, sigma_V2)):
                df_asset = pd.DataFrame({"Close": np.asarray(V, float).reshape(-1)}, index=np.arange(len(V)))
                prior_len = max(2, int(min(SINGLE_ASSET_PRIOR_YEARS*steps_per_year, len(df_asset)-2)))
                res = iterative_lambda(df=df_asset, prior_method=SINGLE_ASSET_PRIOR_METHOD,
                                       prior_start=0, prior_end=prior_len, lambda_start=prior_len, lambda_end=len(df_asset)-1,
                                       lambda_grid=grid, dt=1.0/steps_per_year,
                                       sigma_series=sigV.iloc[1:].to_numpy())
                lambda_results_mf_V[name] = res
        if DO_RUN_INFO:
            xg_v, p0_v = make_prior(PRIOR_DISC_LO, PRIOR_DISC_HI, PRIOR_DISC_N)
            dt_loc = 1.0/steps_per_year
            for name, V, sigV in (("S1", V1, sigma_V1), ("S2", V2, sigma_V2)):
                dxi = info_incr(V, sigV, r=r, dt=dt_loc)
                res = filter_disc(xg_v, p0_v, dxi, dt_loc)
                lambda_results_info_V[name] = res

        S_V = pd.DataFrame({"S1": V1, "S2": V2})
        if DO_PLOT_VP_MF and lambda_results_mf_V:
            f,_ = plot_simple_model_lambdas(S_V, lambda_results_mf_V, steps_per_year,
                                            title=PLOT_TITLES.get("vp_mf","Return-likelihood λ on V1 & V2"),
                                            mode=mode, labels=labels); figs.append(f)
        if DO_PLOT_VP_INFO and lambda_results_info_V:
            f,_ = plot_simple_model_lambdas(S_V, lambda_results_info_V, steps_per_year,
                                            title=PLOT_TITLES.get("vp_info","Info-process λ on V1 & V2"),
                                            mode=mode, labels=labels); figs.append(f)

        if DO_RECONSTRUCT:
            use_res = lambda_results_info_V if lambda_results_info_V else lambda_results_mf_V
            if use_res and ("S1" in use_res and "S2" in use_res):
                Lam1_df, Lam2_df, l1_df, l2_df = reconstruct_lambdas(
                    use_res["S1"], use_res["S2"], np.asarray(sig_base, float), X, Y, return_factor=True
                )
                if DO_PLOT_VECTOR_LAMBDAS and (lam_true_for_plots is not None):
                    f,_ = plot_vector_lambda_estimates(l1_df, l2_df, lam_true_for_plots, steps_per_year,
                                                       title=PLOT_TITLES.get("vector","Factor λ estimates"),
                                                       mode=mode, labels=labels); figs.append(f)
                if DO_PLOT_CAPITAL_LAMBDAS and (lam_true_for_plots is not None):
                    f,_ = plot_capital_lambda_estimates(Lam1_df, Lam2_df, sigma_true_for_plots, lam_true_for_plots,
                                                        steps_per_year, title=PLOT_TITLES.get("capital","Asset-direction Λ estimates"),
                                                        mode=mode, labels=labels); figs.append(f)
                if DO_PLOT_CAPITAL_VS_SINGLE:
                    sa_res = lambda_results_mf if lambda_results_mf else lambda_results_info
                    f,_ = plot_capital_vs_single(Lam1_df, Lam2_df, sa_res,
                                                 base_sigma=(sigma_true_for_plots if lam_true_for_plots is not None else None),
                                                 lam_vec=(lam_true_for_plots if lam_true_for_plots is not None else None),
                                                 steps_per_year=steps_per_year,
                                                 title=PLOT_TITLES.get("cap_vs_single","Λ vs λ — covariance vs single"),
                                                 mode=mode, labels=labels); figs.append(f)

    plt.ioff()
    for f in figs:
        try: f.canvas.draw()
        except: pass
    plt.show()
