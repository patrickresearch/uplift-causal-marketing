import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent / "src"))

from uplift_causal.evaluate_uplift import uplift_band  # noqa: E402
from uplift_causal.plots import LAYOUT, MUTED, SERIES  # noqa: E402

st.set_page_config(page_title="Uplift budget calculator", layout="wide")


@st.cache_data
def load(name):
    return pd.read_parquet(APP / f"scores_{name}.parquet")


@st.cache_data
def band(name, model):
    df = load(name)
    return uplift_band(df.y.to_numpy(), df.t.to_numpy(), df[model].to_numpy())


datasets = sorted(p.stem.removeprefix("scores_") for p in APP.glob("scores_*.parquet"))

with st.sidebar:
    name = st.selectbox("Dataset", datasets)
    model = st.selectbox("Model", [c for c in load(name) if c not in ("y", "t")])
    customers = st.number_input("Customers", 1_000, 100_000_000, 100_000, step=10_000)
    cost = st.number_input("Cost per contact (€)", 0.0, 100.0, 0.10, step=0.05, format="%.2f")
    value = st.number_input("Value per incremental visit (€)", 0.0, 1_000.0, 5.0, step=0.5)
    budget = st.number_input("Budget (€)", 0.0, 1e9, 5_000.0, step=500.0)

grid, est, lo, hi = band(name, model)
contacts = grid * customers


def profit(u):
    return u * customers * value - contacts * cost


p, p_lo, p_hi = profit(est), profit(lo), profit(hi)
feasible = contacts * cost <= budget
i = int(np.argmax(np.where(feasible, p, -np.inf)))

st.title("Whom should we contact?")
st.caption(
    f"Customers ranked by predicted uplift ({model}). Bands are 95 % bootstrap intervals "
    "on held-out data from a randomized experiment."
)

c = st.columns(5)
c[0].metric("Target share", f"{grid[i]:.0%}")
c[1].metric("Contacts", f"{contacts[i]:,.0f}")
c[2].metric("Incremental visits", f"{est[i] * customers:,.0f}", f"95 % CI {lo[i] * customers:,.0f}–{hi[i] * customers:,.0f}", delta_color="off")
c[3].metric("Spend", f"€{contacts[i] * cost:,.0f}")
c[4].metric("Profit", f"€{p[i]:,.0f}", f"€{p[i] - p[-1]:+,.0f} vs. contacting all")

if est[-1] > 0:
    st.markdown(
        f"Contacting the top **{grid[i]:.0%}** reaches **{est[i] / est[-1]:.0%}** of the effect "
        f"of contacting everyone at **{grid[i]:.0%}** of the cost."
    )

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.r_[grid, grid[::-1]],
        y=np.r_[p_hi, p_lo[::-1]],
        fill="toself",
        fillcolor="rgba(42,120,214,0.15)",
        line=dict(width=0),
        hoverinfo="skip",
        name="95 % CI",
    )
)
fig.add_trace(
    go.Scatter(
        x=grid,
        y=p,
        mode="lines",
        line=dict(width=2, color=SERIES[0]),
        name="Expected profit",
        hovertemplate="%{x:.0%} targeted<br>€%{y:,.0f}<extra></extra>",
    )
)
if not feasible.all():
    fig.add_vrect(
        x0=grid[feasible].max(), x1=1, fillcolor=MUTED, opacity=0.1, line_width=0,
        annotation_text="over budget", annotation_position="top left",
    )
fig.add_vline(x=grid[i], line=dict(dash="dash", color=SERIES[1], width=2))
fig.add_hline(y=0, line=dict(color=MUTED, width=1))
fig.update_layout(
    **LAYOUT,
    xaxis_title="Share of customers contacted (by predicted uplift)",
    yaxis_title="Profit (€)",
    xaxis_tickformat=".0%",
    hovermode="x unified",
    height=480,
)
st.plotly_chart(fig)
