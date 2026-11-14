"""Chapter IV figures for RQ2 (static, print). Palette validated with the dataviz validator (light mode): blue
#2a78d6 (ACC), orange #eb6834 (per-scenario best fixed W), aqua #1baf7a (global fixed W; contrast < 3:1 on white ->
every line is also direct-labelled and has its own line style, so identity never depends on colour alone).
IEEE sizes: single column 3.5 in, double column 7.16 in; 8-pt text.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import scenarios as S
from .sim import simulate
from .status import LATENCY_SAMPLE

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f1f0ed"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]   # blue ramp, light -> dark


def _style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6, length=2.5)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _rc():
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.titlesize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def _series(gw, horizon: float, step: float) -> dict[str, np.ndarray]:
    """Binned time series of one run: window (at dispatch), goodput (successful calls completed per s), p95 latency of
    latency samples (HTTP 200 + timeouts, by dispatch time; None when < 5 samples)."""
    edges = np.arange(0.0, horizon + 1e-9, step)
    mids = (edges[:-1] + edges[1:]) / 2
    att = gw.attempts
    t_att = np.array([a.t_dispatch for a in att])
    w = np.array([a.window for a in att])
    done = np.array([c.t_done for c in gw.calls if c.ok])
    gp = np.histogram(done, edges)[0] / step
    p95, win = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [a.latency for a in att if lo <= a.t_dispatch < hi and a.status in LATENCY_SAMPLE]
        p95.append(float(np.percentile(sel, 95)) if len(sel) >= 5 else np.nan)
        m = (t_att >= lo) & (t_att < hi)
        win.append(float(w[m].mean()) if m.any() else np.nan)
    return {"t": mids, "window": np.array(win), "goodput": gp, "p95": np.array(p95)}


def timeseries_figure(frozen: dict, m_agents: int = 32, seed: int = 100, step: float = 2.0,
                      stored: pd.DataFrame | None = None):
    """Two columns (CAP-drop, CAP-burst) x three rows (window, goodput, p95 latency) for frozen ACC, the scenario's
    best fixed W and the global fixed W, on ONE test seed. If `stored` (the test results) is given, each re-simulated
    run's goodput must equal the stored value (determinism check); a mismatch raises."""
    import matplotlib.pyplot as plt
    _rc()
    scns = [s for s, sc in S.SCENARIOS.items() if sc.get("analysis") is not None and sc["analysis"].event is not None
            and s in frozen["best_fixed_w_per_scenario"]]
    fig, axes = plt.subplots(3, len(scns), figsize=(7.16, 5.2), sharex=True, squeeze=False)
    rows = ["Window W (concurrent calls)", "Goodput (successful calls/s)", "p95 server latency (s)"]
    for j, scn in enumerate(scns):
        methods = [("ACC", S.acc_method(**frozen["acc"]), BLUE, "-"),
                   (f"Fixed W={frozen['best_fixed_w_per_scenario'][scn]} (best for this scenario)",
                    S.fixed_method(frozen["best_fixed_w_per_scenario"][scn]), ORANGE, "--"),
                   (f"Fixed W={frozen['best_fixed_w']} (global)", S.fixed_method(frozen["best_fixed_w"]), AQUA, ":")]
        sc0 = S.SCENARIOS[scn]
        ev = sc0["analysis"].event
        for lab, meth, col, ls in methods:
            sc = S.build(scn, m_agents, meth, seed)
            gw = simulate(sc)
            if stored is not None:
                row = stored[(stored["scenario"] == scn) & (stored["M"] == m_agents) & (stored["seed"] == seed)
                             & (stored["tag"] == S.method_tag(meth))]
                ok = sum(c.ok for c in gw.calls) / S.HORIZON
                if len(row) != 1 or abs(float(row["goodput"].iloc[0]) - ok) > 1e-9:
                    raise AssertionError(f"re-simulation differs from the stored test result: {scn} {lab}")
            ser = _series(gw, S.HORIZON, step)
            for i, key in enumerate(("window", "goodput", "p95")):
                axes[i, j].plot(ser["t"], ser[key], color=col, linestyle=ls, linewidth=1.4, label=lab)
        for i in range(3):
            ax = axes[i, j]
            _style(ax)
            ax.axvspan(ev[0], ev[1], color=BAND, zorder=0, linewidth=0)
            if j == 0:
                ax.set_ylabel(rows[i], fontsize=7, color=INK2)
        axes[0, j].axhline(m_agents, color=INK2, linewidth=0.8, linestyle=(0, (1, 2)))
        axes[0, j].text(S.HORIZON, m_agents, f" M={m_agents}", va="center", ha="left", fontsize=6.5, color=INK2)
        axes[2, j].axhline(S.L_TARGET, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
        axes[2, j].text(S.HORIZON, S.L_TARGET, " SLO", va="center", ha="left", fontsize=6.5, color=INK2)
        axes[0, j].set_title(f"{scn}, M = {m_agents}" + (" (capacity / 3)" if scn == "CAP-drop" else " (+80 calls/s)"),
                             loc="left", color=INK, fontsize=8)
        axes[2, j].set_xlabel("simulated time (s)", fontsize=7, color=INK2)
        axes[2, j].set_xlim(0, S.HORIZON)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    best = ", ".join(f"W={frozen['best_fixed_w_per_scenario'][s]} in {s}" for s in scns)
    labels = ["ACC (frozen configuration)", f"Fixed W, best for the scenario ({best})",
              f"Fixed W={frozen['best_fixed_w']} (global baseline)"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def frontier_figure(desc: pd.DataFrame, frozen: dict):
    """Static frontier on the TEST seeds: score (mean normalised goodput over loaded cells) vs worst-cell SLO
    violation (log scale). Shape = family; filled = feasible in every loaded cell. Selected points are labelled."""
    import matplotlib.pyplot as plt
    _rc()
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    fam = {"fixed": ("s", ORANGE, "Fixed window"), "tokenbucket": ("^", AQUA, "Token bucket"), "acc": ("o", BLUE, "ACC")}
    acc_tag = S.method_tag(S.acc_method(**frozen["acc"]))
    for kind, (mk, col, lab) in fam.items():
        d = desc[desc["kind"] == kind]
        x = np.maximum(d["worst_cell_slo_violation"].to_numpy(), 1e-3)
        full = d["feasible_all_loaded_cells"].to_numpy()
        ax.scatter(x[~full], d["score"].to_numpy()[~full], s=22, marker=mk, facecolors="white", edgecolors=col,
                   linewidths=1.2, label=f"{lab} (open: infeasible somewhere)", zorder=3)
        if full.any():
            ax.scatter(x[full], d["score"].to_numpy()[full], s=38, marker=mk, color=col, edgecolors="white",
                       linewidths=1.0, zorder=4, label=f"{lab}, feasible in all loaded cells")
    ax.axvline(S.MAX_SLO_VIOLATION, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(S.MAX_SLO_VIOLATION, 1.06, " 5% SLO limit", fontsize=6.5, color=INK2, va="top")
    lab_tags = {acc_tag: "ACC", "fixed_w4": "W=4", "fixed_w12": "W=12", "fixed_w24": "W=24", "tb_r15": "15/s"}
    for _, r in desc[desc["tag"].isin(lab_tags)].iterrows():
        ax.annotate(lab_tags[r["tag"]], (max(r["worst_cell_slo_violation"], 1e-3), r["score"]), xytext=(4, -2),
                    textcoords="offset points", fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1.0)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("worst-cell SLO violation (log scale; < 0.1% shown at 0.1%)", fontsize=7, color=INK2)
    ax.set_ylabel("score (normalised goodput)", fontsize=7, color=INK2)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=6, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def robustness_heatmap(rm: pd.DataFrame):
    """Methods x scenarios, normalised goodput (sequential blue); a cell where the method violates a constraint in
    some loaded cell of that scenario is hatched and marked with an x (identity never by colour alone)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _rc()
    piv = rm.pivot(index="method", columns="scenario", values="norm_goodput")
    feas = rm.pivot(index="method", columns="scenario", values="feasible_all_loaded_cells")
    order = list(dict.fromkeys(rm["method"]))
    piv, feas = piv.loc[order], feas.loc[order]
    cols = [c for c in S.SCENARIOS if c in piv.columns]
    piv, feas = piv[cols], feas[cols]
    cmap = LinearSegmentedColormap.from_list("blue", SEQ)
    fig, ax = plt.subplots(figsize=(7.16, 0.30 * len(piv) + 0.9))
    v = piv.to_numpy(dtype=float)
    ax.imshow(np.clip(v, 0, 1.05), cmap=cmap, vmin=0, vmax=1.05, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ok = bool(feas.iloc[i, j])
            if not ok:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor="white", alpha=0.55,
                                           linewidth=0))
            txt = f"{v[i, j]:.2f}" + ("" if ok else " ×")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=INK,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9))
    ax.set_xticks(range(len(cols)), cols, fontsize=7, color=INK)
    ax.set_yticks(range(len(piv)), [_short(m) for m in piv.index], fontsize=6.5, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Normalised goodput on the test seeds by scenario (× and hatching: violates a constraint in a "
                 "loaded cell of that scenario)", loc="left", fontsize=7, color=INK2)
    fig.tight_layout()
    return fig


def _short(label: str) -> str:
    return (label.replace(" (global, closest to feasible; dev: feasible in ", " (global; dev feasible ")
            .replace(" loaded cells)", ")").replace("secondary global, D35; dev: feasible in ", "secondary; dev feasible ")
            .replace(" scenarios)", " scen.)"))


def save(fig, path_stem: str) -> list[str]:
    out = []
    for ext in ("pdf", "png"):
        p = f"{path_stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        out.append(p)
    return out
