from __future__ import annotations

from dataclasses import dataclass, replace
from math import erf, exp, log, sqrt
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
# import streamlit.components.v1 as components

@dataclass(frozen=True)
class InputSpec:
    key: str
    label: str
    default: float
    minimum: float
    maximum: float
    is_percentage: bool = False
    dynamic_max_key: Optional[str] = None


@dataclass(frozen=True)
class ERMParameters:
    rw_hpi: float = 0.02
    deferment_rate: float = 0.035
    loan_rate: float = 0.065
    house_price_volatility: float = 0.13
    house_price_start: float = 100_000.0
    time_years: float = 10.0
    funding_cost: float = 0.02
    ltv: float = 0.30
    coc_rate: float = 0.045
    risk_free_rate: float = 0.045
    scr_level: float = 0.995
    scr_decay_factor: float = 0.12

    def updated(self, changes: Dict[str, float]) -> "ERMParameters":
        return replace(self, **changes)


@dataclass(frozen=True)
class ERMResults:
    scr: float
    scr_annuity_factor: float
    cost_of_capital: float
    day1_gain: float
    profit: float
    profit_at_sale: float
    pv_expected_payoff: float
    expected_payoff_at_sale: float
    accumulated_loan: float
    loan_amount: float


class NormalDistribution:
    @staticmethod
    def cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    @staticmethod
    def ppf(p: float) -> float:
        if not 0.0 < p < 1.0:
            raise ValueError("p must be strictly between 0 and 1.")

        a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
             138.3577518672690, -30.66479806614716, 2.506628277459239]
        b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
             66.80131188771972, -13.28068155288572]
        c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
             -2.549732539343734, 4.374664141464968, 2.938163982698783]
        d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
             3.754408661907416]

        plow = 0.02425
        phigh = 1.0 - plow

        if p < plow:
            q = sqrt(-2.0 * log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /                    (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0)

        if p > phigh:
            q = sqrt(-2.0 * log(1.0 - p))
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /                     (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0)

        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


class ERMModel:
    def __init__(self, normal: Optional[NormalDistribution] = None) -> None:
        self._normal = normal or NormalDistribution()

    def calculate(self, params: ERMParameters) -> ERMResults:
        sigma = params.house_price_volatility * sqrt(params.time_years)
        ln_s0 = log(params.house_price_start)

        scr_mu = ln_s0 + (params.rw_hpi - 0.5 * params.house_price_volatility ** 2) * params.time_years
        scr_percentile = self._normal.ppf(1.0 - params.scr_level)
        ln_price_scr = scr_mu + sigma * scr_percentile
        house_price_scr = exp(ln_price_scr)

        loan_amount = params.house_price_start * params.ltv
        accumulated_loan = loan_amount * (1.0 + params.loan_rate) ** params.time_years
        scr = max(accumulated_loan - house_price_scr, 0.0)

        base = exp(-params.scr_decay_factor) / (1.0 + params.risk_free_rate)
        if abs(1.0 - base) < 1e-12:
            scr_annuity_factor = params.time_years
        else:
            scr_annuity_factor = base * (1.0 - base ** params.time_years) / (1.0 - base)

        cost_of_capital = params.coc_rate * scr * scr_annuity_factor

        pricing_drift = params.risk_free_rate - params.deferment_rate
        pricing_mu = ln_s0 + (pricing_drift - 0.5 * params.house_price_volatility ** 2) * params.time_years
        d = (log(accumulated_loan) - pricing_mu) / sigma

        term_full_repayment = accumulated_loan * (1.0 - self._normal.cdf(d))
        term_shortfall_repayment = exp(pricing_mu + 0.5 * sigma ** 2) * self._normal.cdf(
            (log(accumulated_loan) - pricing_mu - sigma ** 2) / sigma
        )
        expected_payoff_at_sale = term_full_repayment + term_shortfall_repayment
        pv_expected_payoff = expected_payoff_at_sale * exp(-params.risk_free_rate * params.time_years)

        funding_cost_value = params.funding_cost * loan_amount
        day1_gain = pv_expected_payoff - loan_amount - funding_cost_value
        profit = day1_gain - cost_of_capital
        profit_at_sale = profit * exp(params.risk_free_rate * params.time_years)

        return ERMResults(
            scr=scr,
            scr_annuity_factor=scr_annuity_factor,
            cost_of_capital=cost_of_capital,
            day1_gain=day1_gain,
            profit=profit,
            profit_at_sale=profit_at_sale,
            pv_expected_payoff=pv_expected_payoff,
            expected_payoff_at_sale=expected_payoff_at_sale,
            accumulated_loan=accumulated_loan,
            loan_amount=loan_amount,
        )


class ParameterCatalog:
    def __init__(self) -> None:
        self._specs: Dict[str, InputSpec] = {
            "deferment_rate": InputSpec("deferment_rate", "Deferment rate", 0.035, -0.10, 0.10, is_percentage=True),
            "loan_rate": InputSpec("loan_rate", "Loan accumulation rate", 0.065, 0.01, 0.20, is_percentage=True),
            "house_price_volatility": InputSpec("house_price_volatility", "House price volatility", 0.13, 0.01, 0.30, is_percentage=True),
            "house_price_start": InputSpec("house_price_start", "House price at inception", 100_000.0, 50_000.0, 50_000_000.0),
            "time_years": InputSpec("time_years", "Projection term (years)", 10.0, 1.0, 50.0),
            "funding_cost": InputSpec("funding_cost", "Funding cost", 0.02, 0.0, 0.20, is_percentage=True),
            "ltv": InputSpec("ltv", "LTV", 0.30, 0.01, 0.80, is_percentage=True),
            "coc_rate": InputSpec("coc_rate", "SCR CoC %", 0.045, 0.01, 0.10, is_percentage=True),
            "risk_free_rate": InputSpec("risk_free_rate", "Risk-free rate", 0.045, 0.0005, 0.20, is_percentage=True),
            "rw_hpi": InputSpec("rw_hpi", "HPI (for SCR only)", 0.02, -0.10, 0.30, is_percentage=True),
            "scr_level": InputSpec("scr_level", "SCR percentile", 0.995, 0.50, 0.9999, is_percentage=True),
            "scr_decay_factor": InputSpec("scr_decay_factor", "SCR decay factor", 0.12, 0.03, 0.25, is_percentage=True),
        }
        self._default_parameters = ERMParameters(
            rw_hpi=self._specs["rw_hpi"].default,
            deferment_rate=self._specs["deferment_rate"].default,
            loan_rate=self._specs["loan_rate"].default,
            house_price_volatility=self._specs["house_price_volatility"].default,
            house_price_start=self._specs["house_price_start"].default,
            time_years=self._specs["time_years"].default,
            funding_cost=self._specs["funding_cost"].default,
            ltv=self._specs["ltv"].default,
            coc_rate=self._specs["coc_rate"].default,
            risk_free_rate=self._specs["risk_free_rate"].default,
            scr_level=self._specs["scr_level"].default,
            scr_decay_factor=self._specs["scr_decay_factor"].default,
        )

    def spec(self, key: str) -> InputSpec:
        return self._specs[key]

    def all_specs(self) -> Sequence[InputSpec]:
        return list(self._specs.values())

    def label_for(self, key: str) -> str:
        return self._specs[key].label

    def default_parameters(self) -> ERMParameters:
        return self._default_parameters


class ValueFormatter:
    @staticmethod
    def rounded_to_3_sig(value: float) -> float:
        if value == 0:
            return 0.0
        return float(f"{value:.3g}")

    @staticmethod
    def format_point_value(value: float, key_or_label: str) -> str:
        percentage_names = {
            "rw_hpi", "deferment_rate", "loan_rate", "house_price_volatility",
            "funding_cost", "coc_rate", "risk_free_rate", "scr_level", "scr_decay_factor",
            "ltv", "HPI", "Deferment rate", "Loan accumulation rate", "House price volatility",
            "Funding cost", "SCR CoC %", "Risk-free rate",
            "SCR percentile", "SCR decay factor", "LTV",
        }
        money_names = {"house_price_start", "loan_amount", "Day1Gain", "Profit", "House price at inception", "Loan amount"}
        if key_or_label in percentage_names:
            return f"{value * 100:.2f}%"
        if key_or_label in money_names:
            return f"{value / 1000:.3f}K" if abs(value) >= 1000 else f"{value:.2f}"
        return f"{value:.6g}"

    @staticmethod
    def format_profit_with_loan_ratio(profit: float, loan_amount: float) -> str:
        main = f"{profit / 1000:.3f}K" if abs(profit) >= 1000 else f"{profit:.2f}"
        ratio = 0.0 if loan_amount == 0 else profit / loan_amount
        return f"{main} ({ratio * 100:.1f}%)"


def build_range_grid(minimum: float, maximum: float) -> np.ndarray:
    raw = np.linspace(minimum, maximum, 30)
    rounded = np.array([ValueFormatter.rounded_to_3_sig(float(v)) for v in raw], dtype=float)
    rounded[0] = minimum
    rounded[-1] = maximum
    for i in range(1, len(rounded)):
        if rounded[i] <= rounded[i - 1]:
            rounded[i] = rounded[i - 1]
    if np.allclose(rounded, rounded[0]):
        rounded = raw
    return rounded


def get_base_values(catalog: ParameterCatalog) -> Dict[str, float]:
    p = catalog.default_parameters()
    return {
        "rw_hpi": p.rw_hpi,
        "deferment_rate": p.deferment_rate,
        "loan_rate": p.loan_rate,
        "house_price_volatility": p.house_price_volatility,
        "house_price_start": p.house_price_start,
        "time_years": p.time_years,
        "funding_cost": p.funding_cost,
        "ltv": p.ltv,
        "coc_rate": p.coc_rate,
        "risk_free_rate": p.risk_free_rate,
        "scr_level": p.scr_level,
        "scr_decay_factor": p.scr_decay_factor,
    }

def apply_number_input(spec: InputSpec, value: float, disabled: bool = False) -> float:
    c1, c2, c3 = st.columns([1.55, 1.0, 0.10])

    with c1:
        st.markdown(
            f"<div class='tight-row-label'>{spec.label}</div>",
            unsafe_allow_html=True,
        )

    if spec.is_percentage:
        with c2:
            out = st.number_input(
                label=f"{spec.label}_value",
                label_visibility="collapsed",
                min_value=float(spec.minimum * 100),
                max_value=float(spec.maximum * 100),
                value=float(value * 100),
                step=0.1,
                format="%.2f",
                disabled=disabled,
                key=f"input_{spec.key}",
            )
        with c3:
            st.markdown(
                "<div class='percent-cell'>%</div>",
                unsafe_allow_html=True,
            )
        return out / 100.0

    if spec.key == "loan_amount_display":
        with c2:
            st.text_input(
                label=f"{spec.label}_value",
                label_visibility="collapsed",
                value=f"{round(value):.0f}",
                disabled=True,
            )
            return float(round(value))

    if spec.key == "house_price_start":
        with c2:
            return st.number_input(
                label=f"{spec.label}_value",
                label_visibility="collapsed",
                min_value=float(spec.minimum),
                max_value=float(spec.maximum),
                value=float(round(value)),
                step=100.0,
                format="%.0f",
                disabled=disabled,
                key=f"input_{spec.key}",
            )

    step = 1.0
    fmt = "%.0f" if spec.key == "time_years" else "%.2f"

    with c2:
        return st.number_input(
            label=f"{spec.label}_value",
            label_visibility="collapsed",
            min_value=float(spec.minimum),
            max_value=float(spec.maximum),
            value=float(value),
            step=step,
            format=fmt,
            disabled=disabled,
            key=f"input_{spec.key}",
        )

def apply_range_input(spec: InputSpec, value: float) -> float:
    c1, c2, c3 = st.columns([0.95, 1.00, 0.12])

    with c1:
        st.markdown(
            f"<div class='tight-row-label'>{spec.label}</div>",
            unsafe_allow_html=True,
        )

    if spec.is_percentage:
        with c2:
            out = st.number_input(
                label=f"range_{spec.key}",
                label_visibility="collapsed",
                min_value=float(spec.minimum * 100),
                max_value=float(spec.maximum * 100),
                value=float(value * 100),
                step=0.1,
                format="%.2f",
                key=f"range_input_{spec.key}",
            )
        with c3:
            st.markdown("<div class='percent-cell'>%</div>", unsafe_allow_html=True)
        return out / 100.0

    step = 100.0 if spec.key == "house_price_start" else 1.0
    if spec.key == "time_years":
        fmt = "%.0f"
    elif spec.key == "house_price_start":
        fmt = "%.0f"
    else:
        fmt = "%.2f"

    display_value = float(value)

    with c2:
        out = st.number_input(
            label=f"range_{spec.key}",
            label_visibility="collapsed",
            min_value=float(spec.minimum),
            max_value=float(spec.maximum),
            value=display_value,
            step=step,
            format=fmt,
            key=f"range_input_{spec.key}",
        )
    with c3:
        st.markdown(
            "<div class='tight-row-label'></div>",
            unsafe_allow_html=True,
        )
    return out

def tickvals_for_range(values: np.ndarray, n: int = 5) -> list[float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-12:
        return [vmin]
    return list(np.linspace(vmin, vmax, n))

def ticktext_for_key(vals: list[float], key: str) -> list[str]:
    out = []
    for v in vals:
        if key in {"rw_hpi", "deferment_rate", "loan_rate", "house_price_volatility",
                   "funding_cost", "coc_rate", "risk_free_rate", "scr_level",
                   "scr_decay_factor", "ltv"}:
            out.append(f"{v * 100:.1f}%")
        elif key in {"house_price_start", "loan_amount", "Day1Gain", "Profit"} or abs(v) >= 1000:
            out.append(f"{v/1000:.1f}K")
        else:
            out.append(f"{v:.1f}")
    return out

def build_line_chart(x: np.ndarray, y: np.ndarray, x_key: str, output_key: str, loan_amount: float) -> go.Figure:
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if ymin < 0 < ymax:
        cmin, cmax = ymin, ymax
    else:
        limit = max(abs(ymin), abs(ymax), 1e-12)
        cmin, cmax = -limit, limit
    customdata = []
    for xv, yv in zip(x, y):
        x_txt = ValueFormatter.format_point_value(float(xv), x_key)
        y_txt = ValueFormatter.format_profit_with_loan_ratio(float(yv), loan_amount) if output_key == "Profit" else ValueFormatter.format_point_value(float(yv), output_key)
        customdata.append([x_txt, y_txt])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers",
        marker=dict(size=8, color=y, colorscale="RdYlGn", cmin=cmin, cmax=cmax, colorbar=dict(title=output_key)),
        line=dict(width=2, color="rgba(120,120,120,0.65)"),
        customdata=customdata,
        hovertemplate=f"{x_key}: %{{customdata[0]}}<br>{output_key}: %{{customdata[1]}}<extra></extra>",
    ))
    fig.update_layout(
        height=460,
        margin=dict(l=5, r=5, t=25, b=5),
        title=f"{output_key} vs {x_key}",
        xaxis_title=x_key,
        yaxis_title=output_key,
    )
    x_ticks = tickvals_for_range(x, 5)
    y_ticks = tickvals_for_range(y, 5)

    fig.update_xaxes(
        tickmode="array",
        tickvals=x_ticks,
        ticktext=ticktext_for_key(x_ticks, x_key),
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_ticks,
        ticktext=ticktext_for_key(y_ticks, output_key),
    )
    return fig

def build_surface_chart(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, x_key: str, y_key: str, output_key: str) -> go.Figure:
    zmin, zmax = float(np.min(zz)), float(np.max(zz))
    if zmin < 0 < zmax:
        cmin, cmax = zmin, zmax
    else:
        limit = max(abs(zmin), abs(zmax), 1e-12)
        cmin, cmax = -limit, limit

    customdata = []
    for xv, yv, zv in zip(xx.ravel(), yy.ravel(), zz.ravel()):
        x_txt = ValueFormatter.format_point_value(float(xv), x_key)
        y_txt = ValueFormatter.format_point_value(float(yv), y_key)
        z_txt = ValueFormatter.format_point_value(float(zv), output_key)
        customdata.append([x_txt, y_txt, z_txt])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=xx.ravel(),
            y=yy.ravel(),
            z=zz.ravel(),
            mode="markers",
            marker=dict(
                size=5,
                color=zz.ravel(),
                colorscale="RdYlGn",
                cmin=cmin,
                cmax=cmax,
                opacity=1.0,
                colorbar=dict(title=output_key),
            ),
            customdata=customdata,
            hovertemplate=(
                f"{x_key}: %{{customdata[0]}}<br>"
                f"{y_key}: %{{customdata[1]}}<br>"
                f"{output_key}: %{{customdata[2]}}<extra></extra>"
            ),
        )
    )

    x_ticks = tickvals_for_range(xx.ravel(), 5)
    y_ticks = tickvals_for_range(yy.ravel(), 5)
    z_ticks = tickvals_for_range(zz.ravel(), 5)

    fig.update_layout(
        height=560,
        margin=dict(l=5, r=5, t=30, b=5),
        title=f"{output_key} surface",
        hovermode="closest",
        scene=dict(
            xaxis=dict(
                title=x_key,
                tickmode="array",
                tickvals=x_ticks,
                ticktext=ticktext_for_key(x_ticks, x_key),
                nticks=5,
            ),
            yaxis=dict(
                title=y_key,
                tickmode="array",
                tickvals=y_ticks,
                ticktext=ticktext_for_key(y_ticks, y_key),
                nticks=5,
            ),
            zaxis=dict(
                title=output_key,
                tickmode="array",
                tickvals=z_ticks,
                ticktext=ticktext_for_key(z_ticks, output_key),
                nticks=5,
            ),
        ),
    )
    return fig

def compute_one_way(model: ERMModel, base_params: ERMParameters, variable_key: str, values: Sequence[float], output_key: str) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array(values, dtype=float)
    y = np.zeros_like(x)
    for i, value in enumerate(x):
        result = model.calculate(base_params.updated({variable_key: float(value)}))
        y[i] = result.day1_gain if output_key == "Day1Gain" else result.profit
    return x, y

def compute_two_way(model: ERMModel, base_params: ERMParameters, x_key: str, x_values: Sequence[float], y_key: str, y_values: Sequence[float], output_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xv = np.array(x_values, dtype=float)
    yv = np.array(y_values, dtype=float)
    xx, yy = np.meshgrid(xv, yv)
    zz = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            result = model.calculate(base_params.updated({x_key: float(xx[i, j]), y_key: float(yy[i, j])}))
            zz[i, j] = result.day1_gain if output_key == "Day1Gain" else result.profit
    return xx, yy, zz


st.set_page_config(page_title="ERM Sensitivity Explorer", layout="wide")

st.markdown("""
<style>
.block-container {
    max-width: 1600px;
    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
    padding-left: 0.8rem;
    padding-right: 0.8rem;
}
.percent-cell {
    min-height: 1.35rem !important;
    height: 1.35rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.0 !important;
    position: relative !important;
    top: -7px !important;
    left: -14px !important;
}           
div[data-testid="stNumberInput"] button {
    display: none !important;
}

div[data-baseweb="select"] > div {
    min-height: 1.35rem !important;
}

div[data-testid="stNumberInput"] {
    min-height: 1.35rem !important;
}

div[data-testid="stNumberInput"] > div,
div[data-testid="stTextInput"] > div {
    min-height: 1.35rem !important;
    height: 1.35rem !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

div[data-testid="stNumberInput"] [data-baseweb="input"],
div[data-testid="stTextInput"] [data-baseweb="input"] {
    min-height: 1.35rem !important;
    height: 1.35rem !important;
}

div[data-testid="stNumberInput"] [data-baseweb="input"] > div,
div[data-testid="stTextInput"] [data-baseweb="input"] > div {
    min-height: 1.35rem !important;
    height: 1.35rem !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    text-align: right !important;
    min-height: 1.35rem !important;
    height: 1.35rem !important;
    line-height: 1.35rem !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
div[data-testid="stHorizontalBlock"] {
    align-items: center !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
div[data-testid="column"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
.tight-row-label {
    min-height: 1.35rem !important;
    height: 1.35rem !important;
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.35rem !important;
    position: static !important;
}

div[data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
st.subheader("ERM Sensitivity Explorer", divider=False)


catalog = ParameterCatalog()
model = ERMModel()
base_values = get_base_values(catalog)

left_col, right_col = st.columns([1.05, 2.15])

with left_col:
    st.markdown("<div style='text-align:center;'><b>Scenario controls</b></div>", unsafe_allow_html=True)
    labels = [spec.label for spec in catalog.all_specs()]
    label_to_key = {spec.label: spec.key for spec in catalog.all_specs()}

    v1label_col, v1box_col = st.columns([0.50, 1.50])
    with v1label_col:
        st.markdown(
            f"<div class='tight-row-label'>Variable 1</div>",
            unsafe_allow_html=True,
        )
    with v1box_col:
        var1_label = st.selectbox(
            "Variable 1",
            labels,
            index=labels.index(catalog.label_for("risk_free_rate")),
            label_visibility="collapsed",
        )
    var1_key = label_to_key[var1_label]

    var1_spec = catalog.spec(var1_key)
    v1c1, v1c2 = st.columns(2)

    with v1c1:
        v1_min = apply_range_input(
            InputSpec(
                f"{var1_key}_min",
                "Minimum",
                var1_spec.minimum,
                var1_spec.minimum,
                var1_spec.maximum,
                var1_spec.is_percentage,
            ),
            var1_spec.minimum,
        )

    with v1c2:
        v1_max = apply_range_input(
            InputSpec(
                f"{var1_key}_max",
                "Maximum",
                var1_spec.maximum,
                var1_spec.minimum,
                var1_spec.maximum,
                var1_spec.is_percentage,
            ),
            var1_spec.maximum,
        )

    if v1_min >= v1_max:
        st.error("Variable 1 minimum must be strictly smaller than maximum.")
        st.stop()

    v1_grid = build_range_grid(v1_min, v1_max)

    var2_options = ["None"] + [label for label in labels if label != var1_label]

    v2label_col, v2box_col = st.columns([0.50, 1.50])
    with v2label_col:
        st.markdown(
            f"<div class='tight-row-label'>Variable 2</div>",
            unsafe_allow_html=True,
        )
    with v2box_col:
        var2_label = st.selectbox(
            "Variable 2",
            var2_options,
            index=0,
            label_visibility="collapsed",
        )

    var2_key = None if var2_label == "None" else label_to_key[var2_label]

    var2_range_container = st.container()

    with var2_range_container:
        if var2_key is not None:
            var2_spec = catalog.spec(var2_key)
            v2c1, v2c2 = st.columns(2)

            with v2c1:
                v2_min = apply_range_input(
                    InputSpec(
                        f"{var2_key}_min",
                        "Minimum",
                        var2_spec.minimum,
                        var2_spec.minimum,
                        var2_spec.maximum,
                        var2_spec.is_percentage,
                    ),
                    var2_spec.minimum,
                )

            with v2c2:
                v2_max = apply_range_input(
                    InputSpec(
                        f"{var2_key}_max",
                        "Maximum",
                        var2_spec.maximum,
                        var2_spec.minimum,
                        var2_spec.maximum,
                        var2_spec.is_percentage,
                    ),
                    var2_spec.maximum,
                )

            if v2_min >= v2_max:
                st.error("Variable 2 minimum must be strictly smaller than maximum.")
                st.stop()

            v2_grid = build_range_grid(v2_min, v2_max)
        else:
            st.markdown("<div style='height: 2.0rem;'></div>", unsafe_allow_html=True)
            v2_grid = None

    outlabel_col, outbox_col = st.columns([0.50, 1.50])
    with outlabel_col:
        st.markdown(
            f"<div class='tight-row-label'>Output metric</div>",
            unsafe_allow_html=True,
        )
    with outbox_col:
        output_key = st.selectbox(
            "Output metric",
            ("Day1Gain", "Profit"),
            index=1,
            label_visibility="collapsed",
        )

    st.markdown("<div style='text-align:center;'><b>Constant inputs</b></div>", unsafe_allow_html=True)
    edited_values = dict(base_values)
    selected = {var1_key, var2_key}

    for spec in catalog.all_specs():
        edited_values[spec.key] = apply_number_input(
            spec,
            edited_values[spec.key],
            disabled=(spec.key in selected),
        )

        if spec.key == "ltv":
            loan_amount = edited_values["house_price_start"] * edited_values["ltv"]
            loan_spec = InputSpec(
                key="loan_amount_display",
                label="Loan amount",
                default=loan_amount,
                minimum=0.0,
                maximum=1_000_000_000.0,
                is_percentage=False,
            )
            apply_number_input(loan_spec, loan_amount, disabled=True)

    loan_amount = edited_values["house_price_start"] * edited_values["ltv"]

    render_chart = st.button("Update", type="primary", width="stretch")

base_params = catalog.default_parameters().updated(edited_values)

with right_col:
    if render_chart or "initial_render_rndr" not in st.session_state:
        st.session_state["initial_render_rndr"] = True
        if var2_key is None:
            x, y = compute_one_way(model, base_params, var1_key, v1_grid, output_key)
            fig = build_line_chart(x, y, var1_key, output_key, loan_amount)
            st.plotly_chart(fig, width="stretch")
            st.info("Hover over the chart to inspect values.")
        else:
            xx, yy, zz = compute_two_way(model, base_params, var1_key, v1_grid, var2_key, v2_grid, output_key)
            fig = build_surface_chart(xx, yy, zz, var1_key, var2_key, output_key)
            st.plotly_chart(fig, width="stretch")
            st.info("Hover over the surface to inspect values.")
