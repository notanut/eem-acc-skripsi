"""Figures for the notebooks: a live illustration of the ACC and a progress monitor.

Design (dataviz skill): one measure per axis (small multiples, never dual axis), 2 px lines,
recessive grid, text in ink tokens (never series colours), legend for >= 2 series plus direct
end labels, reference-palette slots 1-2 (validated light-mode: all checks PASS).
The illustration is NOT a result: it uses the ACC start values, not tuned values.
"""

from __future__ import annotations

import numpy as np

from . import scenarios as S
from .monitor import fmt_duration, summary_line
from .sim import simulate

SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#efeeea"
BLUE, ORANGE = "#2a78d6", "#eb6834"                      # categorical slots 1, 2
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]   # sequential blue (light -> dark)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _binned(times, values, width, t_end, fn):
    edges = np.arange(0.0, t_end + width, width)
    idx = np.digitize(times, edges) - 1
    xs, ys = [], []
    for i in range(len(edges) - 1):
        sel = [v for v, j in zip(values, idx) if j == i]
        r = fn(sel)
        if r is not None:
            xs.append(edges[i] + width / 2)
            ys.append(r)
    return np.array(xs), np.array(ys)


def demo_figure(scenario: str = "CAP-drop", m_agents: int = 16, seed: int = 0, fixed_w: int = 16):
    """ACC (start values) vs a fixed window under one approved scenario. Illustration only."""
    import matplotlib.pyplot as plt

    runs = {
        "ACC": simulate(S.build(scenario, m_agents, S.acc_method(**S.ACC_START), seed)),
        f"Fixed W={fixed_w}": simulate(S.build(scenario, m_agents, S.fixed_method(fixed_w), seed)),
    }
    colors = {"ACC": BLUE, f"Fixed W={fixed_w}": ORANGE}
    t_end = max(max(a.t_dispatch for a in gw.attempts) for gw in runs.values())

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True, facecolor=SURFACE)
    titles = ["In-flight window W (max concurrent API calls)",
              "Throttled attempts per second (HTTP 429 + 503)",
              f"Server latency p95 per 2.5 s (s); dashed = SLO target {S.L_TARGET} s"]
    for ax, title in zip(axes, titles):
        _style(ax)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=6)
        for _, (regime, phases, events) in [(scenario, S.SCENARIOS[scenario])]:
            if len(events) == 2:
                ax.axvspan(events[0][0], events[1][0], color=BAND, zorder=0)

    for name, gw in runs.items():
        c = colors[name]
        att = gw.attempts
        t = np.array([a.t_dispatch for a in att])
        axes[0].step(t, [a.window for a in att], where="post", color=c, linewidth=2)
        xs, ys = _binned(t, [a.status in (429, 503) for a in att], 1.0, t_end, lambda v: float(sum(v)) if v else 0.0)
        axes[1].plot(xs, ys, color=c, linewidth=2)
        ok = [(a.t_dispatch, a.latency) for a in att if a.status == 200]
        xs, ys = _binned(np.array([o[0] for o in ok]), [o[1] for o in ok], 2.5, t_end,
                         lambda v: float(np.percentile(v, 95)) if len(v) >= 5 else None)
        axes[2].plot(xs, ys, color=c, linewidth=2)

    axes[2].axhline(S.L_TARGET, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    axes[0].set_ylim(0, max(a.window for gw in runs.values() for a in gw.attempts) * 1.12)
    axes[0].yaxis.get_major_locator().set_params(integer=True)
    axes[2].set_xlabel("simulated time (s)", color=INK2, fontsize=9.5)
    axes[2].set_xlim(0, t_end)
    regime, phases, events = S.SCENARIOS[scenario]
    if len(events) == 2:
        axes[0].text((events[0][0] + events[1][0]) / 2, axes[0].get_ylim()[1] * 0.97,
                     f"capacity x{events[0][1]:g}", ha="center", va="top", fontsize=9, color=INK2)

    handles = [plt.Line2D([0], [0], color=colors[n], linewidth=2.5, label=n) for n in runs]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.985), ncol=2, frameon=False,
               fontsize=10, labelcolor=INK)
    fig.text(0.02, 0.985, f"Illustration, not a result: ACC vs a fixed window ({scenario}, {m_agents} agents)",
             fontsize=12, color=INK, fontweight="bold", va="top")
    fig.text(0.02, 0.955, "ACC uses its start values (alpha=beta=0.5, epsilon=0.1, n_c=16), not tuned values. One seed.",
             fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def monitor_figure(snap: dict):
    """Overall progress bar + one tile per (scenario, M) cell showing runs done / total."""
    import matplotlib.pyplot as plt

    scns = list(S.SCENARIOS)
    ms = list(S.M_LEVELS)
    fig, (axb, axh) = plt.subplots(2, 1, figsize=(8.6, 4.6), facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [0.55, 3]})
    frac = snap["done"] / max(snap["total"], 1)
    axb.set_facecolor(SURFACE)
    axb.barh([0], [1], color=GRID, height=0.5)
    axb.barh([0], [frac], color=SEQ[3], height=0.5)
    axb.set_xlim(0, 1); axb.set_ylim(-0.6, 0.6)
    axb.axis("off")
    axb.text(0, 0.78, summary_line(snap), fontsize=10, color=INK, va="bottom", transform=axb.transData)

    _style(axh); axh.grid(False)
    for i, scn in enumerate(scns):
        for j, m in enumerate(ms):
            tot = snap["cell_total"].get((scn, m), 0)
            dn = snap["cell_done"].get((scn, m), 0)
            f = dn / tot if tot else 0.0
            col = SEQ[min(len(SEQ) - 1, int(f * (len(SEQ) - 1) + 0.5))] if f > 0 else "#f3f2ef"
            axh.add_patch(plt.Rectangle((j + 0.03, len(scns) - 1 - i + 0.03), 0.94, 0.94, color=col, zorder=1))
            dark = f > 0.55
            axh.text(j + 0.5, len(scns) - 1 - i + 0.5, f"{dn}/{tot}", ha="center", va="center", fontsize=10,
                     color="#ffffff" if dark else INK)
    axh.set_xlim(0, len(ms)); axh.set_ylim(0, len(scns))
    axh.set_xticks([j + 0.5 for j in range(len(ms))]); axh.set_xticklabels([f"M = {m}" for m in ms])
    axh.set_yticks([len(scns) - 1 - i + 0.5 for i in range(len(scns))]); axh.set_yticklabels(scns)
    axh.tick_params(length=0)
    for sp in axh.spines.values():
        sp.set_visible(False)
    axh.set_title("Runs finished per scenario x agent count", loc="left", fontsize=10, color=INK2, pad=6)
    fig.tight_layout()
    return fig
