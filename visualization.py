import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot


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
