from scipy.stats import norm
from math import sqrt

def _ols_beta(df):
    X = np.column_stack([
        np.ones(len(df)),
        df["abs_log_m"].values,
        df["T"].values,
        (df["abs_log_m"].values * df["T"].values),
    ])
    y = df["improvement"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta

def _cluster_bootstrap_reps(df, B=5000, seed=1):
    rng = np.random.default_rng(seed)
    reps = np.array(sorted(df["rep"].unique()))
    betas = np.zeros((B, 4), dtype=float)
    for b in range(B):
        reps_b = rng.choice(reps, size=len(reps), replace=True)
        d_b = pd.concat([df[df["rep"] == r] for r in reps_b], ignore_index=True)
        betas[b] = _ols_beta(d_b)
    return betas

def _summ_ci_p(v):
    v = np.asarray(v, dtype=float)
    ci_lo, ci_hi = np.quantile(v, [0.025, 0.975])
    p_one_sided_gt0 = float(np.mean(v <= 0.0))
    return float(v.mean()), (float(ci_lo), float(ci_hi)), p_one_sided_gt0

def _two_prop_ztest(posA, nA, posB, nB):
    pA = posA / nA
    pB = posB / nB
    p_pool = (posA + posB) / (nA + nB)
    se = sqrt(max(1e-12, p_pool * (1 - p_pool) * (1 / nA + 1 / nB)))
    z = (pA - pB) / se
    p_one_sided = float(1 - norm.cdf(z))
    return float(pA), float(pB), float(z), p_one_sided

def _trend_test_Tbin_from_surf(surf):
    d = surf.copy()
    d["pos"] = (d["share_improvement_pos"] * d["count"]).round().astype(int)

    agg = d.groupby("T_bin", observed=False)[["pos", "count"]].sum().reset_index()

    def mid_from_interval(s):
        s = str(s).strip()
        a, b = s[1:-1].split(",")
        return 0.5 * (float(a) + float(b))

    agg["T_mid"] = agg["T_bin"].apply(mid_from_interval)
    agg = agg.sort_values("T_mid").reset_index(drop=True)

    x = agg["pos"].to_numpy()
    n = agg["count"].to_numpy()
    w = np.arange(1, len(agg) + 1, dtype=float)

    N = n.sum()
    X = x.sum()
    p_hat = X / N
    wbar = (n * w).sum() / N
    num = ((w - wbar) * (x - n * p_hat)).sum()
    den = np.sqrt(max(1e-12, p_hat * (1 - p_hat) * (n * (w - wbar) ** 2).sum()))
    Z = num / den
    p_one_sided = float(1 - norm.cdf(Z))
    return agg, float(Z), p_one_sided

def _format_overall(d):
    n = int(len(d))
    n_reps = int(d["rep"].nunique())
    win = d["win"].astype(int)
    pos = int(win.sum())
    neg = int(n - pos)
    share = float(pos / n)
    mean_imp = float(d["improvement"].mean())
    med_imp = float(d["improvement"].median())
    p0 = 0.5
    se = sqrt(max(1e-12, p0 * (1 - p0) / n))
    z = (share - p0) / se
    p_one = float(1 - norm.cdf(z))

    lines = [
        f"TFBS wins (improvement>0): {pos}/{n} = {share:.1%}  (loses: {neg}/{n} = {(1-share):.1%})",
        f"Mean improvement: {mean_imp:+.6f}   |   Median improvement: {med_imp:+.6f}",
        f"Replicates: {n_reps}   |   Total points: {n}",
        f"Sign test vs 50% wins (one-sided): z={z:.3f}, p={p_one:.6g}",
    ]
    return "\n".join(lines)

def _wilson_ci(k, n, alpha=0.05):
    if n <= 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha/2)
    phat = k / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = z * np.sqrt((phat*(1-phat)/n) + (z*z/(4*n*n))) / denom
    return float(center - half), float(center + half)
