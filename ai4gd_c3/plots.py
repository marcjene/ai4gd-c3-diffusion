"""Plot helpers for the notebook. Power in kW, irradiance in kW/m², time axis 00:00-23:30."""
import matplotlib.pyplot as plt
import torch

from .data import TIME

TICKS = range(0, 48, 8)


def _time_axis(ax):
    ax.set_xticks(TICKS, [TIME[i] for i in TICKS])
    ax.set_xlim(0, 47)


def _ghi(ax, ghi):
    ax2 = ax.twinx()
    ax2.plot(ghi, "--", color="orange", lw=1.2, label="GHI_d")
    ax2.set_ylabel("GHI (kW/m²)", color="orange")
    ax2.set_ylim(0, 1.2)
    return ax2


def day(d):
    """PV, NC_d, NC_d-1 (kW) and GHI_d (kW/m²) of one day."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(d["pv_kw"], "k", lw=2.2, label="PV (target x_0)")
    ax.plot(d["nc_kw"], color="tab:blue", label="NC_d")
    ax.plot(d["nc_lag1_kw"], color="tab:blue", alpha=0.4, label="NC_d−1")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("kW")
    _time_axis(ax)
    ax2 = _ghi(ax, d["ghi"])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.set_title(f"Customer {d['customer']}, {d['date']} ({d['capacity_kwp']:.2f} kWp)")
    plt.show()


def forward(x0, abar, ks, x_k_of):
    """x_k for several k (left) and the signal/noise shares (right). x_k_of(k) returns x_k [T] in pu."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for k in ks:
        a1.plot(x_k_of(k), lw=1.6 if k == 0 else 1, label=f"k = {k}")
    a1.set_ylabel("x_k (pu)")
    a1.legend(fontsize=8, ncol=2)
    _time_axis(a1)
    kk = torch.arange(len(abar))
    a2.plot(kk, abar.sqrt(), label="signal share √ᾱ_k")
    a2.plot(kk, (1 - abar).sqrt(), label="noise share √(1 − ᾱ_k)")
    a2.set_xlabel("k")
    a2.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def losses(values):
    plt.figure(figsize=(7, 3))
    plt.plot(values, lw=0.5, alpha=0.4)
    w = 50
    smooth = torch.tensor(values).unfold(0, w, 1).mean(1) if len(values) > w else torch.tensor(values)
    plt.plot(range(w - 1, w - 1 + len(smooth)), smooth, lw=1.6, label=f"moving mean ({w} steps)")
    plt.yscale("log")
    plt.xlabel("gradient step")
    plt.ylabel("loss (MSE on ε)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def samples(ax, s, d, title=""):
    """Samples (pu, [N, T]) and truth of day d on ax, in kW."""
    cap = d["capacity_kwp"]
    ax.plot((s * cap).T, color="tab:blue", alpha=0.12, lw=1)
    ax.plot(d["pv_kw"], "k", lw=2, label="true PV")
    ax.set_title(title, fontsize=10)
    _time_axis(ax)
    ghi_ax = _ghi(ax, d["ghi"])  # GHI only for its shape: no second scale in grids
    ghi_ax.set_yticks([])
    ghi_ax.set_ylabel("")


def compare(rows, cols, panels, figsize=None):
    """Grid of samples(): panels is a list of (samples, day, title). One kW scale for all panels."""
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (4.6 * cols, 3.2 * rows), sharey=True, squeeze=False)
    for ax, (s, d, title) in zip(axes.flat, panels):
        samples(ax, s, d, title)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("PV (kW)")
    axes.flat[0].plot([], [], "--", color="orange", label="GHI_d (shape)")
    axes.flat[0].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.show()


def trajectory(ks, rows, truth, labels=("x_k", "x̂_0")):
    """2 x len(ks) grid: rows = (x_k list, x0_hat list), each [T] in pu."""
    fig, axes = plt.subplots(2, len(ks), figsize=(2.3 * len(ks), 4.4), sharex=True)
    for j, k in enumerate(ks):
        for i in range(2):
            ax = axes[i, j]
            ax.plot(truth, color="grey", lw=1)
            ax.plot(rows[i][j], color="tab:blue" if i == 0 else "tab:green", lw=1.2)
            ax.set_ylim(-3, 3) if i == 0 else ax.set_ylim(-0.5, 1.5)
            ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(labels[i])
        axes[0, j].set_title(f"k = {k}")
    plt.tight_layout()
    plt.show()


def band(d, median, lower, upper, level):
    """Median and PI band, all in kW."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.fill_between(range(48), lower, upper, color="tab:blue", alpha=0.25, label=f"{level} % PI")
    ax.plot(median, color="tab:blue", lw=2, label="median")
    ax.plot(d["pv_kw"], "k", lw=1.6, label="true PV")
    ax.set_ylabel("PV (kW)")
    _time_axis(ax)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Customer {d['customer']}, {d['date']}")
    plt.show()


def repeats(rows, at="12:00"):
    """Ex8c. rows: {n: [(q10, q50, q90, picp80) per repeat]} with quantiles in kW."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    ns = list(rows)
    for j, (label, color) in enumerate([("10 %", "tab:orange"), ("median", "tab:blue"), ("90 %", "tab:green")]):
        for i, n in enumerate(ns):
            a1.scatter([i + (j - 1) * 0.15] * len(rows[n]), [r[j] for r in rows[n]], color=color, s=18,
                       label=label if i == 0 else None)
    a1.set_xticks(range(len(ns)), [f"N = {n}" for n in ns])
    a1.set_ylabel(f"quantile at {at} (kW)")
    a1.legend(fontsize=8)
    for i, n in enumerate(ns):
        a2.scatter([i] * len(rows[n]), [r[3] for r in rows[n]], color="tab:blue", s=18)
    a2.axhline(80, color="grey", lw=1)
    a2.set_xticks(range(len(ns)), [f"N = {n}" for n in ns])
    a2.set_ylabel("PICP of the 80 % PI (%)")
    plt.tight_layout()
    plt.show()
