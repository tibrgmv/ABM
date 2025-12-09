import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_rel, wilcoxon, ks_2samp

from model import ABMModel


def bs_price_call(S0, K, r, sigma, T):
    S0 = float(S0); K = float(K); r = float(r); sigma = float(sigma); T = float(T)
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        return max(S0 - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_price_put(S0, K, r, sigma, T):
    S0 = float(S0); K = float(K); r = float(r); sigma = float(sigma); T = float(T)
    if T <= 0:
        return max(K - S0, 0.0)
    if sigma <= 0:
        return max(K * np.exp(-r * T) - S0, 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))


def bs_delta_call(S, K, r, sigma, T):
    S = float(S)
    K = float(K)
    r = float(r)
    sigma = float(sigma)
    T = float(T)
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1))

def bs_delta_put(S, K, r, sigma, T):
    return float(bs_delta_call(S, K, r, sigma, T) - 1.0)

def sample_positive_stable(alpha, size, rng):
    alpha = float(alpha)
    U = rng.uniform(0.0, np.pi, size=size)
    W = rng.exponential(1.0, size=size)
    part1 = np.sin(alpha * U) / (np.sin(U) ** alpha)
    part2 = (np.sin((1.0 - alpha) * U) / W) ** (1.0 - alpha)
    return part1 * part2


def fractional_price_call_mc(S0, K, r, sigma, T, alpha, n_mc, seed):
    rng = np.random.default_rng(seed)
    S = sample_positive_stable(alpha, n_mc, rng)
    E = (T / np.maximum(S, 1e-16)) ** alpha
    Z = rng.normal(0.0, 1.0, size=n_mc)
    ST = S0 * np.exp((r - 0.5 * sigma * sigma) * E + sigma * np.sqrt(E) * Z)
    payoff = np.maximum(ST - K, 0.0)
    return float(np.exp(-r * T) * np.mean(payoff))


def fractional_price_put_mc(S0, K, r, sigma, T, alpha, n_mc, seed):
    rng = np.random.default_rng(seed)
    S = sample_positive_stable(alpha, n_mc, rng)
    E = (T / np.maximum(S, 1e-16)) ** alpha
    Z = rng.normal(0.0, 1.0, size=n_mc)
    ST = S0 * np.exp((r - 0.5 * sigma * sigma) * E + sigma * np.sqrt(E) * Z)
    payoff = np.maximum(K - ST, 0.0)
    return float(np.exp(-r * T) * np.mean(payoff))


def run_abm_paths(cfg, n_paths, seed0):
    paths = []
    for i in range(int(n_paths)):
        m = ABMModel(seed=seed0 + 10_000 * i, **cfg)
        paths.append(m.run())
    return np.array(paths, dtype=float)


def reference_price_call(paths, K, r, T):
    ST = paths[:, -1]
    payoff = np.maximum(ST - float(K), 0.0)
    return float(np.exp(-float(r) * float(T)) * np.mean(payoff))


def reference_price_put(paths, K, r, T):
    ST = paths[:, -1]
    payoff = np.maximum(float(K) - ST, 0.0)
    return float(np.exp(-float(r) * float(T)) * np.mean(payoff))


def estimate_sigma(paths, dt):
    if not np.all(np.isfinite(paths)) or np.min(paths) <= 0:
        raise ValueError("ABM paths contain non-positive or non-finite prices. Fix price dynamics first.")
    lr = np.diff(np.log(paths), axis=1)
    s = float(np.std(lr) / np.sqrt(float(dt)))
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Estimated sigma is non-positive/non-finite. Increase variability or fix returns.")
    return s


def run_h1_pricing_experiment(
    cfg,
    dt,
    steps,
    r,
    K_grid,
    alpha_frac=0.85,
    n_paths=300,
    n_mc=40_000,
    n_rep=10,
    seed0=1,
    option_type="put"
):
    T = float(steps) * float(dt)
    rows = []

    for rep in range(int(n_rep)):
        paths = run_abm_paths(cfg, n_paths=n_paths, seed0=seed0 + rep * 999)
        sigma = estimate_sigma(paths, dt=dt)

        for K in K_grid:
            K = float(K)

            if option_type == "call":
                ref = reference_price_call(paths, K=K, r=r, T=T)
                p_bs = bs_price_call(cfg["S0"], K, r, sigma, T)
                p_fr = fractional_price_call_mc(cfg["S0"], K, r, sigma, T, alpha=alpha_frac, n_mc=n_mc, seed=seed0 + rep * 123 + int(K * 10))
            else:
                ref = reference_price_put(paths, K=K, r=r, T=T)
                p_bs = bs_price_put(cfg["S0"], K, r, sigma, T)
                p_fr = fractional_price_put_mc(cfg["S0"], K, r, sigma, T, alpha=alpha_frac, n_mc=n_mc, seed=seed0 + rep * 123 + int(K * 10))

            rows.append({
                "rep": rep,
                "K": K,
                "sigma": sigma,
                "ref": ref,
                "bs": p_bs,
                "frac": p_fr,
                "abs_err_bs": abs(p_bs - ref),
                "abs_err_frac": abs(p_fr - ref)
            })

    df = pd.DataFrame(rows)
    by_rep = df.groupby("rep")[["abs_err_bs", "abs_err_frac"]].mean().reset_index()
    diff = by_rep["abs_err_bs"] - by_rep["abs_err_frac"]

    tstat, pval = ttest_rel(by_rep["abs_err_bs"], by_rep["abs_err_frac"])
    wstat, wp = wilcoxon(diff)
    ksstat, ksp = ks_2samp(df["abs_err_bs"].values, df["abs_err_frac"].values)

    summary = {
        "option_type": option_type,
        "alpha_frac": float(alpha_frac),
        "mean_abs_err_bs": float(by_rep["abs_err_bs"].mean()),
        "mean_abs_err_frac": float(by_rep["abs_err_frac"].mean()),
        "paired_ttest_t": float(tstat),
        "paired_ttest_p": float(pval),
        "wilcoxon_stat": float(wstat),
        "wilcoxon_p": float(wp),
        "ks_stat": float(ksstat),
        "ks_p": float(ksp),
        "n_rep": int(n_rep),
        "n_paths": int(n_paths),
        "n_mc": int(n_mc)
    }
    return df, by_rep, summary
