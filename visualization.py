import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot
from utils import *

def plot_one_path(paths):
    plt.figure()
    plt.plot(paths)
    plt.title("ABM price path")
    plt.xlabel("step")
    plt.ylabel("price")
    plt.show()

def plot_error_box(by_rep):
    data = [by_rep["abs_err_bs"].values, by_rep["abs_err_frac"].values]
    plt.figure()
    plt.boxplot(data, labels=["BS", "Fractional"])
    plt.title("Mean absolute pricing error per replication")
    plt.ylabel("MAE")
    plt.show()


def plot_market_dashboard(model, paths=None):
    prices = np.array(model.market.prices, dtype=float)
    rets = np.diff(np.log(prices + 1e-12))
    t = np.arange(len(prices))

    fig = plt.figure(figsize=(14, 16))

    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(t, prices)
    ax1.set_title("Price")

    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(model.regime_log)
    ax2.set_title("Regime (0 calm, 1 stress)")

    ax3 = fig.add_subplot(4, 2, 3)
    ax3.plot(model.spread_log)
    ax3.set_title("Spread")

    ax4 = fig.add_subplot(4, 2, 4)
    ax4.plot(model.imbalance_log)
    ax4.set_title("Order book imbalance")

    ax5 = fig.add_subplot(4, 2, 5)
    ax5.hist(rets, bins=80)
    ax5.set_title("Log-returns histogram")

    ax6 = fig.add_subplot(4, 2, 6)
    probplot(rets, dist="norm", plot=ax6)
    ax6.set_title("QQ-plot vs Normal")

    ax7 = fig.add_subplot(4, 2, 7)
    ax7.plot(model.trade_count_log)
    ax7.set_title("Trades per step")

    ax8 = fig.add_subplot(4, 2, 8)
    ax8.plot(model.volume_log)
    ax8.set_title("Volume per step")

    plt.tight_layout()
    plt.show()

def plot_h3_graphs(df_h3, surf, res, B=5000, seed=1):
    d = df_h3.copy()
    d["win"] = (d["improvement"].values > 0.0).astype(int)

    betas = _cluster_bootstrap_reps(d, B=B, seed=seed)
    beta_T = betas[:, 2]
    beta_int = betas[:, 3]

    plt.figure(figsize=(9, 4))
    plt.hist(beta_T, bins=40, alpha=0.7, label="beta_T")
    plt.axvline(0.0)
    m, (lo, hi), p1 = _summ_ci_p(beta_T)
    plt.title(f"Bootstrap beta_T (mean={m:+.4f}, CI95=({lo:+.4f},{hi:+.4f}), p_one_sided_gt0={p1:.3g})")
    plt.xlabel("beta_T")
    plt.ylabel("count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def gr_for_h2(df_bias):
    bias_by_K = (
        df_bias
        .groupby("K")[["signed_err_bs", "signed_err_frac"]]
        .mean()
        .rename(columns={"signed_err_bs": "bias_bs", "signed_err_frac": "bias_frac"})
        .reset_index()
        .sort_values("K")
    )

    plt.figure(figsize=(9, 4))
    plt.plot(bias_by_K["K"].values, bias_by_K["bias_bs"].values, marker="o", label="BS bias (mean bs-ref)")
    plt.plot(bias_by_K["K"].values, bias_by_K["bias_frac"].values, marker="o", label="TFBS bias (mean frac-ref)")
    plt.axhline(0.0)
    plt.title("Mean signed bias by strike K")
    plt.xlabel("Strike K")
    plt.ylabel("Mean signed error")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


    abs_bs = np.abs(df_bias["signed_err_bs"].values)
    abs_fr = np.abs(df_bias["signed_err_frac"].values)
    delta_abs = abs_bs - abs_fr

    plt.figure(figsize=(9, 4))
    plt.hist(abs_bs, bins=40, alpha=0.6, label="|BS error|")
    plt.hist(abs_fr, bins=40, alpha=0.6, label="|TFBS error|")
    plt.title("Absolute pricing error vs reference (distribution)")
    plt.xlabel("|error|")
    plt.ylabel("Count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()