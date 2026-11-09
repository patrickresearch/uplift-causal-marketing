from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .evaluate_uplift import SEGMENTS

FIGURES = Path(__file__).resolve().parents[2] / "docs" / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color=INK),
    margin=dict(l=40, r=30, t=60, b=40),
)


def save(fig, name, width=900, height=500):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.write_image(FIGURES / f"{name}.png", width=width, height=height, scale=2)
    return fig


def estimate_plot(summary, truth, title="Naive vs. adjusted vs. true effect"):
    est, lo, hi = truth
    colors = [SERIES[1] if i == "Naive" else SERIES[0] for i in summary.index]
    fig = go.Figure(
        go.Scatter(
            x=summary.est,
            y=summary.index,
            mode="markers",
            marker=dict(size=11, color=colors, line=dict(width=2, color="white")),
            error_x=dict(
                type="data",
                array=summary.hi - summary.est,
                arrayminus=summary.est - summary.lo,
                thickness=2,
                width=0,
                color=MUTED,
            ),
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    fig.add_vrect(x0=lo, x1=hi, fillcolor=MUTED, opacity=0.12, line_width=0)
    fig.add_vline(
        x=est,
        line=dict(dash="dash", color=MUTED, width=2),
        annotation_text="true effect (RCT)",
        annotation_font_color=MUTED,
    )
    fig.update_layout(
        **LAYOUT, title=title, showlegend=False, xaxis_tickformat=".1%", yaxis_autorange="reversed"
    )
    return fig


def dag_plot(confounders, covariates, treatment="E-Mail", outcome="Visit"):
    pos = {treatment: (0, 0), outcome: (2, 0)}
    pos |= {c: (x, 1) for c, x in zip(confounders, np.linspace(0.4, 1.6, len(confounders)), strict=True)}
    pos |= {c: (x, -1) for c, x in zip(covariates, np.linspace(1.4, 2.6, len(covariates)), strict=True)}
    edges = [(treatment, outcome)] + [(c, n) for c in confounders for n in (treatment, outcome)]
    edges += [(c, outcome) for c in covariates]
    fig = go.Figure()
    for a, b in edges:
        fig.add_annotation(
            x=pos[b][0], y=pos[b][1], ax=pos[a][0], ay=pos[a][1],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=MUTED, standoff=22, startstandoff=22,
        )
    color = {treatment: SERIES[0], outcome: SERIES[1]}
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in pos.values()],
            y=[p[1] for p in pos.values()],
            text=list(pos),
            mode="markers+text",
            textposition="middle center",
            marker=dict(size=44, color=[color.get(n, GRID) for n in pos], line=dict(width=2, color="white")),
            textfont=dict(size=11, color=[("white" if n in color else INK) for n in pos]),
            hoverinfo="skip",
        )
    )
    axis = dict(visible=False, range=[-0.4, 3.0])
    fig.update_layout(**LAYOUT, showlegend=False, xaxis=axis, yaxis=axis | dict(range=[-1.5, 1.5]))
    return fig


def qini_plot(curves, title="Qini curve"):
    fig = go.Figure()
    for (name, (x, q)), c in zip(curves.items(), SERIES, strict=False):
        step = max(1, len(x) // 1000)
        fig.add_trace(go.Scatter(x=x[::step], y=q[::step], name=name, line=dict(width=2, color=c)))
    x, q = next(iter(curves.values()))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, q[-1]], mode="lines", name="Random", line=dict(width=2, dash="dash", color=MUTED))
    )
    fig.update_layout(
        **LAYOUT,
        title=title,
        xaxis_title="Share of customers targeted",
        yaxis_title="Incremental outcomes",
        xaxis_tickformat=".0%",
        hovermode="x unified",
    )
    return fig


def quadrant_plot(mu0, tau, seg, threshold, delta, n=5000, seed=0):
    idx = np.random.default_rng(seed).choice(len(mu0), min(n, len(mu0)), replace=False)
    mu0, tau, seg = np.asarray(mu0)[idx], np.asarray(tau)[idx], np.asarray(seg)[idx]
    fig = go.Figure()
    for s, c, sym in zip(SEGMENTS, SERIES, ["circle", "diamond", "x", "square"], strict=True):
        m = seg == s
        fig.add_trace(
            go.Scattergl(
                x=mu0[m],
                y=tau[m],
                mode="markers",
                name=s,
                marker=dict(size=8, color=c, symbol=sym, opacity=0.6, line=dict(width=0)),
                hovertemplate="baseline %{x:.1%}<br>effect %{y:+.1%}<extra>" + s + "</extra>",
            )
        )
    for y in (delta, -delta):
        fig.add_hline(y=y, line=dict(dash="dash", color=MUTED, width=1))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.add_vline(x=threshold, line=dict(dash="dot", color=MUTED, width=1))
    fig.update_layout(
        **LAYOUT,
        title="Customer segments by baseline response and estimated effect",
        xaxis_title="P(visit | no e-mail)",
        yaxis_title="Estimated effect of the e-mail",
        xaxis_tickformat=".0%",
        yaxis_tickformat="+.0%",
    )
    return fig
