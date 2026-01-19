import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf
from arch import arch_model

TITLE_X = 0.46
n_sims = 50

def _small_legend(ax=None, ncol=3, x=1.0, y=1.22):
    """
    Compact legend placed outside the axes (top-right).
    Tweak `y` if you need it a touch higher/lower.
    """
    if ax is None:
        ax = plt.gca()
    return ax.legend(
        ncol=ncol,
        fontsize=7,
        frameon=True,
        loc='upper right',
        bbox_to_anchor=(x, y),
        borderaxespad=0.2,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.25,
        columnspacing=0.6,
    )

#%% Simulate Asset Data
def simulate_assets_timevarying_sigma(
    N=10000,
    S0=(10, 10),
    Lambda_true=(0.02, 0.03),  
    sigma_init=np.array([[0.3, 0.1], [0.075, 0.15]]),
    dt=1/252,
    n=2,
    sigma_bounds=(0.05, 0.4),
    walk_std=0.001,
    mean_revert=0.007,
    seed=None,
    plot=False,
):
    if seed is not None:
        np.random.seed(seed)
    S0 = np.array(S0)
    sigma_t = np.zeros((N, n, n))
    sigma_t[0] = sigma_init.copy()
    for t in range(1, N):
        step = walk_std * np.random.randn(n, n)
        new_sigma = sigma_t[t-1] + step + mean_revert * (sigma_init - sigma_t[t-1])
        # Optionally clip to bounds (elementwise)
        new_sigma = np.clip(new_sigma, sigma_bounds[0], sigma_bounds[1])
        sigma_t[t] = new_sigma
    dW = np.random.randn(N, n) * np.sqrt(dt)
    dlogS = np.zeros((N, n))
    for i in range(n):
        for t in range(N):
            dlogS[t, i] = (Lambda_true[i] - 0.5 * np.sum(sigma_t[t,i]**2)) * dt + dW[t] @ sigma_t[t,i]
    logS = np.log(S0) + np.cumsum(dlogS, axis=0)
    S = np.exp(logS)
    S = pd.DataFrame(S, columns=['S1', 'S2'])
    if plot:
        # Sigma elements over time (x in years)
        t_years = np.arange(N) * dt
        plt.figure(figsize=(10,6))
        plt.plot(t_years, sigma_t[:,0,0], label='sigma_11')
        plt.plot(t_years, sigma_t[:,0,1], label='sigma_12')
        plt.plot(t_years, sigma_t[:,1,0], label='sigma_21')
        plt.plot(t_years, sigma_t[:,1,1], label='sigma_22')
        plt.title("Time-varying Sigma Matrix Elements (Non-Symmetric)", x=TITLE_X)
        plt.xlabel("Years")
        plt.ylabel("Sigma value")
        _small_legend()
        plt.tight_layout(rect=[0,0,1,0.88])
        plt.show()
        # Asset prices (x in years)
        plt.figure(figsize=(10,5))
        plt.plot(t_years, S['S1'], label='Asset 1')
        plt.plot(t_years, S['S2'], label='Asset 2')
        plt.xlabel('Years')
        plt.ylabel('Price')
        plt.title('Simulated Asset Prices', x=TITLE_X)
        _small_legend(ncol=2)
        plt.tight_layout(rect=[0,0,1,0.88])
        plt.show()
    return S, sigma_t

def estimate_sigmas_symmetric(
    S,
    dt=1/252,
    window=252,
    sigma_true=None,
    plot=True
):
    from scipy.linalg import sqrtm
    logrets = np.log(S).diff().dropna()
    rolling_covs = []
    rolling_sigmas = []
    for i in range(window, len(logrets)):
        cov = np.cov(logrets.iloc[i-window:i].T, bias=True)
        rolling_covs.append(cov)
        sigma_est = sqrtm(cov / dt)
        if np.iscomplexobj(sigma_est):
            sigma_est = sigma_est.real
        rolling_sigmas.append(sigma_est)
    rolling_covs = np.array(rolling_covs)
    rolling_sigmas = np.array(rolling_sigmas)
    idx = logrets.index[window:]

    if plot:
        plt.figure(figsize=(10,6))
        n = len(idx)
        t_years = (np.arange(window, window + n)) * dt  # convert to years for plotting only

        # Estimated (solid)
        plt.plot(t_years, rolling_sigmas[:, 0, 0], label=r'Estimated $\sigma_{11}$', color='C0')
        plt.plot(t_years, rolling_sigmas[:, 1, 1], label=r'Estimated $\sigma_{22}$', color='C3')
        sigma_off_diag = 0.5 * (rolling_sigmas[:, 0, 1] + rolling_sigmas[:, 1, 0])
        plt.plot(t_years, sigma_off_diag, label=r'Estimated (sym) $\sigma_{12}=\sigma_{21}$', color='C2')

        if sigma_true is not None and sigma_true.ndim == 3:  # true (solid, lighter alpha)
            plt.plot(t_years, sigma_true[window:window+n, 0, 0], color='C0', alpha=0.5, label=r'True $\sigma_{11}$')
            plt.plot(t_years, sigma_true[window:window+n, 1, 1], color='C3', alpha=0.5, label=r'True $\sigma_{22}$')
            plt.plot(t_years, sigma_true[window:window+n, 0, 1], color='C2', alpha=0.5, label=r'True $\sigma_{12}$')
            plt.plot(t_years, sigma_true[window:window+n, 1, 0], color='C1', alpha=0.5, label=r'True $\sigma_{21}$')

        plt.title('Rolling Sigma (Symmetric Square Root)', x=TITLE_X)
        plt.xlabel('Years')
        plt.ylabel('Sigma value')
        plt.ylim(bottom=0)
        _small_legend(ncol=3)
        plt.tight_layout(rect=[0,0,1,0.88])
        plt.show()

    return idx, rolling_sigmas, rolling_covs

"""
TRY CHOLSKEY
"""

def estimate_sigmas_eigen(
    S,
    dt=1/252,
    window=252,
    sigma_true=None,
    plot=True
):
    logrets = np.log(S).diff().dropna()
    rolling_covs = []
    rolling_sigmas = []
    for i in range(window, len(logrets)):
        cov = np.cov(logrets.iloc[i-window:i].T, bias=True)
        rolling_covs.append(cov)
        eigvals, eigvecs = np.linalg.eigh(cov / dt)
        # To avoid negative/complex values from rounding, clip to zero
        eigvals = np.clip(eigvals, 0, None)
        sigma_est = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        rolling_sigmas.append(sigma_est)
    rolling_covs = np.array(rolling_covs)
    rolling_sigmas = np.array(rolling_sigmas)
    idx = logrets.index[window:]

    if plot:
        plt.figure(figsize=(10,6))
        n = len(idx)
        t_years = (np.arange(window, window + n)) * dt

        plt.plot(t_years, rolling_sigmas[:, 0, 0], label=r'Eigen $\sigma_{11}$', color='C0')
        plt.plot(t_years, rolling_sigmas[:, 1, 1], label=r'Eigen $\sigma_{22}$', color='C3')
        sigma_off_diag = 0.5 * (rolling_sigmas[:, 0, 1] + rolling_sigmas[:, 1, 0])
        plt.plot(t_years, sigma_off_diag, label=r'Eigen (sym) $\sigma_{12}=\sigma_{21}$', color='C2')

        if sigma_true is not None and sigma_true.ndim == 3:
            plt.plot(t_years, sigma_true[window:window+n, 0, 0], color='C0', alpha=0.5, label=r'True $\sigma_{11}$')
            plt.plot(t_years, sigma_true[window:window+n, 1, 1], color='C3', alpha=0.5, label=r'True $\sigma_{22}$')
            plt.plot(t_years, sigma_true[window:window+n, 0, 1], color='C2', alpha=0.5, label=r'True $\sigma_{12}$')
            plt.plot(t_years, sigma_true[window:window+n, 1, 0], color='C1', alpha=0.5, label=r'True $\sigma_{21}$')

        plt.title('Rolling Sigma (Eigenvalue Decomposition)', x=TITLE_X)
        plt.xlabel('Years')
        plt.ylabel('Sigma value')
        plt.ylim(bottom=0)
        _small_legend(ncol=3)
        plt.tight_layout(rect=[0,0,1,0.88])
        plt.show()

    return idx, rolling_sigmas, rolling_covs

def estimate_sigmas_ledoit_wolf(
    S,
    dt=1/252,
    window=252,
    sigma_true=None,
    plot=True
):

    logrets = np.log(S).diff().dropna()
    rolling_covs = []
    rolling_sigmas = []
    for i in range(window, len(logrets)):
        lw = LedoitWolf().fit(logrets.iloc[i-window:i].values)
        cov = lw.covariance_
        rolling_covs.append(cov)
        sigma_est = sqrtm(cov / dt)
        if np.iscomplexobj(sigma_est):
            sigma_est = sigma_est.real
        rolling_sigmas.append(sigma_est)
    rolling_covs = np.array(rolling_covs)
    rolling_sigmas = np.array(rolling_sigmas)
    idx = logrets.index[window:]

    if plot:
        plt.figure(figsize=(10,6))
        n = len(idx)
        t_years = (np.arange(window, window + n)) * dt

        plt.plot(t_years, rolling_sigmas[:, 0, 0], label=r'LW $\sigma_{11}$', color='C0')
        plt.plot(t_years, rolling_sigmas[:, 1, 1], label=r'LW $\sigma_{22}$', color='C3')
        sigma_off_diag = 0.5 * (rolling_sigmas[:, 0, 1] + rolling_sigmas[:, 1, 0])
        plt.plot(t_years, sigma_off_diag, label=r'LW (sym) $\sigma_{12}=\sigma_{21}$', color='C2')

        if sigma_true is not None and sigma_true.ndim == 3:
            plt.plot(t_years, sigma_true[window:window+n, 0, 0], color='C0', alpha=0.5, label=r'True $\sigma_{11}$')
            plt.plot(t_years, sigma_true[window:window+n, 1, 1], color='C3', alpha=0.5, label=r'True $\sigma_{22}$')
            plt.plot(t_years, sigma_true[window:window+n, 0, 1], color='C2', alpha=0.5, label=r'True $\sigma_{12}$')
            plt.plot(t_years, sigma_true[window:window+n, 1, 0], color='C1', alpha=0.5, label=r'True $\sigma_{21}$')

        plt.title('Rolling Sigma (Ledoit-Wolf Robust Covariance)', x=TITLE_X)
        plt.xlabel('Years')
        plt.ylabel('Sigma value')
        plt.ylim(bottom=0)
        _small_legend(ncol=3)
        plt.tight_layout(rect=[0,0,1,0.88])
        plt.show()

    return idx, rolling_sigmas, rolling_covs

"""
TRY GARCH
"""
def random_sigma(low=0.05, high=0.4, size=(2,2)):
    # Ensure positive-definite
    A = np.random.uniform(low, high, size)
    return (A + A.T)/2 + 0.2*np.eye(2)

def random_lambda(low=0.0, high=0.12):
    return np.random.uniform(low, high, 2)

dt = 1/252
window = 252
n_sims = 2

methods = [
    ("Symmetric", estimate_sigmas_symmetric),
    ("Eigen", estimate_sigmas_eigen),
    ("Ledoit-Wolf", estimate_sigmas_ledoit_wolf)
]
n_methods = len(methods)
total_time = np.zeros(n_methods)
total_err = np.zeros(n_methods)
errs_time_series = [ [] for _ in range(n_methods) ]

for sim in range(n_sims):
    Lambda_true = random_lambda()
    sigma = random_sigma()
    S, sigma_t = simulate_assets_timevarying_sigma(plot=False)
    sim_times = []
    min_len = None
    sigmas_all = []
    for m, (name, func) in enumerate(methods):
        t0 = time.time()
        idx, sigmas, _ = func(S, dt=dt, window=window, sigma_true=sigma_t, plot=False)
        sim_times.append(time.time() - t0)
        sigmas_all.append(sigmas)
        if min_len is None or sigmas.shape[0] < min_len:
            min_len = sigmas.shape[0]
    for m, (name, func) in enumerate(methods):
        sigma_est = sigmas_all[m][:min_len]
        # Standard deviation of error (Frobenius norm per time)
        err = np.sqrt(np.mean((sigma_est - sigma)**2, axis=(1,2)))  # shape: (n_t,)
        errs_time_series[m].append(err)
        total_err[m] += err.mean()  # average std deviation for this sim
        total_time[m] += sim_times[m]
    print("done sim: ", sim)

# Average across sims
total_err /= n_sims
total_time /= n_sims

mean_errs = []
for m in range(n_methods):
    all_errs = np.vstack(errs_time_series[m])
    mean_errs.append(all_errs.mean(axis=0))

min_len_final = min(len(e) for e in mean_errs)
x_years = np.arange(window, window + min_len_final) * dt

plt.figure(figsize=(12,6))
"""
for m, (name, _) in enumerate(methods):
    plt.plot(x_years, mean_errs[m][:min_len_final], label=name)
plt.title(f"Mean Std Dev (Frobenius) of Sigma Error\n(averaged over {n_sims} random simulated assets)", x=TITLE_X)
plt.xlabel("Years")
plt.ylabel(r"Mean $\sqrt{ \frac{1}{4} \sum_{i,j} (\hat{\sigma}_{ij} - \sigma_{ij}^{\mathrm{true}} )^2 }$")
_small_legend(ncol=3)
plt.tight_layout(rect=[0,0,1,0.88])
plt.show()
"""
# --- Grouped bar chart: average std deviation and average time ---
labels = [name for name, _ in methods]
x = np.arange(n_methods)
width = 0.35
fig, ax1 = plt.subplots(figsize=(8,4))

bar1 = ax1.bar(x - width/2, total_err, width, color='C0', label='Mean Std Dev',alpha=0.8)
ax2 = ax1.twinx()
bar2 = ax2.bar(x + width/2, total_time, width, color='C1', label='Mean Time (s)',alpha=0.8)

# axis titles: plain black
ax1.set_ylabel('Mean Std Deviation of Sigma Error', color='black')
ax2.set_ylabel('Mean Time per Run (seconds)', color='black')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel('Method')
ax1.set_title('Average Std Dev and Time by Sigma Estimation Method', x=TITLE_X)

# compact combined legend above, stacked (one column), top-right outside
handles = [bar1, bar2]
labels_combined = ['Mean Std Dev', 'Mean Time (s)']
ax1.legend(
    handles, labels_combined,
    fontsize=7, frameon=True, loc='upper right', bbox_to_anchor=(1.0, 1.22),
    borderaxespad=0.2, handlelength=1.0, handletextpad=0.35, labelspacing=0.25, ncol=1
)

plt.tight_layout(rect=[0,0,1,0.86])  # extra top margin for the legend
plt.show()
