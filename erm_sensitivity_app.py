from __future__ import annotations

from dataclasses import dataclass, replace
from math import erf, exp, log, sqrt
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d


# ============================================================
# Domain layer
# ============================================================

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
    loan_amount: float = 30_000.0
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


class NormalDistribution:
    @staticmethod
    def cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    @staticmethod
    def ppf(p: float) -> float:
        if not 0.0 < p < 1.0:
            raise ValueError("p must be strictly between 0 and 1.")

        # Acklam inverse-normal approximation
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
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                   (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0)

        if p > phigh:
            q = sqrt(-2.0 * log(1.0 - p))
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                    (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0)

        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


class ERMModel:
    def __init__(self, normal: Optional[NormalDistribution] = None) -> None:
        self._normal = normal or NormalDistribution()

    def calculate(self, params: ERMParameters) -> ERMResults:
        sigma = params.house_price_volatility * sqrt(params.time_years)
        ln_s0 = log(params.house_price_start)

        # SCR block
        scr_mu = ln_s0 + (params.rw_hpi - 0.5 * params.house_price_volatility ** 2) * params.time_years
        scr_percentile = self._normal.ppf(1.0 - params.scr_level)
        ln_price_scr = scr_mu + sigma * scr_percentile
        house_price_scr = exp(ln_price_scr)

        accumulated_loan = params.loan_amount * (1.0 + params.loan_rate) ** params.time_years
        scr = accumulated_loan - house_price_scr

        base = exp(-params.scr_decay_factor) / (1.0 + params.risk_free_rate)
        if abs(1.0 - base) < 1e-12:
            scr_annuity_factor = params.time_years
        else:
            scr_annuity_factor = base * (1.0 - base ** params.time_years) / (1.0 - base)

        cost_of_capital = params.coc_rate * scr * scr_annuity_factor

        # Day 1 gain block
        pricing_drift = params.risk_free_rate - params.deferment_rate
        pricing_mu = ln_s0 + (pricing_drift - 0.5 * params.house_price_volatility ** 2) * params.time_years
        d = (log(accumulated_loan) - pricing_mu) / sigma

        term_full_repayment = accumulated_loan * (1.0 - self._normal.cdf(d))
        term_shortfall_repayment = exp(pricing_mu + 0.5 * sigma ** 2) * self._normal.cdf(
            (log(accumulated_loan) - pricing_mu - sigma ** 2) / sigma
        )
        expected_payoff_at_sale = term_full_repayment + term_shortfall_repayment
        pv_expected_payoff = expected_payoff_at_sale * exp(-params.risk_free_rate * params.time_years)

        funding_cost_value = params.funding_cost * params.loan_amount
        day1_gain = pv_expected_payoff - params.loan_amount - funding_cost_value
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
        )


class ParameterCatalog:
    def __init__(self) -> None:
        self._specs: Dict[str, InputSpec] = {
            "rw_hpi": InputSpec("rw_hpi", "HPI", 0.02, -0.10, 0.30, is_percentage=True),
            "deferment_rate": InputSpec("deferment_rate", "Deferment rate", 0.035, -0.10, 0.10, is_percentage=True),
            "loan_rate": InputSpec("loan_rate", "Loan accumulation rate", 0.065, 0.01, 0.20, is_percentage=True),
            "house_price_volatility": InputSpec("house_price_volatility", "House price volatility", 0.13, 0.01, 0.30, is_percentage=True),
            "house_price_start": InputSpec("house_price_start", "House price at inception", 100_000.0, 50_000.0, 50_000_000.0),
            "time_years": InputSpec("time_years", "Projection term (years)", 10.0, 1.0, 50.0),
            "funding_cost": InputSpec("funding_cost", "Funding cost", 0.02, 0.0, 0.20, is_percentage=True),
            "loan_amount": InputSpec("loan_amount", "Loan amount", 30_000.0, 10_000.0, 100_000.0, dynamic_max_key="house_price_start"),
            "coc_rate": InputSpec("coc_rate", "SCR CoC percentage used for pricing", 0.045, 0.01, 0.10, is_percentage=True),
            "risk_free_rate": InputSpec("risk_free_rate", "Risk-free rate", 0.045, 0.0005, 0.20, is_percentage=True),
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
            loan_amount=self._specs["loan_amount"].default,
            coc_rate=self._specs["coc_rate"].default,
            risk_free_rate=self._specs["risk_free_rate"].default,
            scr_level=self._specs["scr_level"].default,
            scr_decay_factor=self._specs["scr_decay_factor"].default,
        )

    def spec(self, key: str) -> InputSpec:
        return self._specs[key]

    def all_specs(self) -> Sequence[InputSpec]:
        return list(self._specs.values())

    def keys(self) -> Sequence[str]:
        return list(self._specs.keys())

    def label_for(self, key: str) -> str:
        return self._specs[key].label

    def default_parameters(self) -> ERMParameters:
        return self._default_parameters

    def bounds_for(self, key: str, current_values: Dict[str, float]) -> Tuple[float, float]:
        spec = self.spec(key)
        maximum = spec.maximum
        if spec.dynamic_max_key:
            maximum = min(maximum, current_values.get(spec.dynamic_max_key, maximum))
        return spec.minimum, maximum


class ValueFormatter:
    @staticmethod
    def rounded_to_3_sig(value: float) -> float:
        if value == 0:
            return 0.0
        return float(f"{value:.3g}")

    @staticmethod
    def display_value(value: float, is_percentage: bool) -> str:
        if is_percentage:
            return f"{value * 100:.6g}%"
        if abs(value) >= 1000:
            return f"{value:,.6g}"
        return f"{value:.6g}"

    @staticmethod
    def parse_user_value(text: str, is_percentage: bool) -> float:
        cleaned = text.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1].strip()
            return float(cleaned) / 100.0
        value = float(cleaned)
        return value / 100.0 if is_percentage else value


class RangeGrid:
    def __init__(self, values: Sequence[float]) -> None:
        if len(values) != 30:
            raise ValueError("RangeGrid must contain exactly 30 discrete values.")
        self.values = np.array(values, dtype=float)

    @classmethod
    def build(cls, minimum: float, maximum: float) -> "RangeGrid":
        raw = np.linspace(minimum, maximum, 30)
        rounded = np.array([ValueFormatter.rounded_to_3_sig(float(v)) for v in raw], dtype=float)

        rounded[0] = minimum
        rounded[-1] = maximum

        for i in range(1, len(rounded)):
            if rounded[i] <= rounded[i - 1]:
                rounded[i] = rounded[i - 1]

        if np.allclose(rounded, rounded[0]):
            rounded = raw

        return cls(rounded)

    def value_at(self, index: int) -> float:
        bounded = max(0, min(29, int(round(index))))
        return float(self.values[bounded])

    def index_of_closest(self, value: float) -> int:
        return int(np.abs(self.values - value).argmin())


@dataclass
class VariableRangeState:
    key: Optional[str]
    min_value: float
    max_value: float
    grid: RangeGrid


class RangeManager:
    def __init__(self, catalog: ParameterCatalog) -> None:
        self._catalog = catalog

    def create_default_state(self, key: Optional[str], current_values: Dict[str, float]) -> VariableRangeState:
        if key is None:
            grid = RangeGrid.build(0.0, 1.0)
            return VariableRangeState(key=None, min_value=0.0, max_value=1.0, grid=grid)

        minimum, maximum = self._catalog.bounds_for(key, current_values)
        spec = self._catalog.spec(key)
        min_value = max(minimum, min(spec.default, maximum))
        max_value = maximum
        if min_value >= max_value:
            min_value = minimum
            max_value = maximum
        return VariableRangeState(key=key, min_value=min_value, max_value=max_value, grid=RangeGrid.build(min_value, max_value))

    def validate(self, key: str, min_value: float, max_value: float, current_values: Dict[str, float]) -> Tuple[bool, str]:
        lower, upper = self._catalog.bounds_for(key, current_values)
        if min_value < lower or max_value > upper:
            spec = self._catalog.spec(key)
            return False, (
                f"{spec.label}: values must stay within "
                f"[{ValueFormatter.display_value(lower, spec.is_percentage)}, "
                f"{ValueFormatter.display_value(upper, spec.is_percentage)}]."
            )
        if min_value >= max_value:
            return False, "Minimum must be strictly smaller than maximum."
        return True, ""

    def update_state(self, state: VariableRangeState, min_value: float, max_value: float) -> VariableRangeState:
        if state.key is None:
            return state
        return VariableRangeState(
            key=state.key,
            min_value=min_value,
            max_value=max_value,
            grid=RangeGrid.build(min_value, max_value),
        )


class ERMSensitivityService:
    def __init__(self, model: ERMModel) -> None:
        self._model = model
        self._extractors: Dict[str, Callable[[ERMResults], float]] = {
            "Day1Gain": lambda r: r.day1_gain,
            "Profit": lambda r: r.profit,
        }

    def one_way(
        self,
        base_params: ERMParameters,
        variable_key: str,
        values: Sequence[float],
        output_key: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.array(values, dtype=float)
        y = np.zeros_like(x)
        extractor = self._extractors[output_key]
        for i, value in enumerate(x):
            scenario = base_params.updated({variable_key: float(value)})
            y[i] = extractor(self._model.calculate(scenario))
        return x, y

    def two_way(
        self,
        base_params: ERMParameters,
        x_key: str,
        x_values: Sequence[float],
        y_key: str,
        y_values: Sequence[float],
        output_key: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xv = np.array(x_values, dtype=float)
        yv = np.array(y_values, dtype=float)
        xx, yy = np.meshgrid(xv, yv)
        zz = np.zeros_like(xx)
        extractor = self._extractors[output_key]
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                scenario = base_params.updated({x_key: float(xx[i, j]), y_key: float(yy[i, j])})
                zz[i, j] = extractor(self._model.calculate(scenario))
        return xx, yy, zz


# ============================================================
# Presentation layer
# ============================================================

@dataclass
class SelectionSnapshot:
    variable_1: str
    variable_2: str
    output_key: str


class PlotController:
    def __init__(self, figure: Figure, info_callback: Callable[[str], None]) -> None:
        self.figure = figure
        self._info_callback = info_callback
        self._canvas: Optional[FigureCanvasTkAgg] = None
        self._last_mode: Optional[str] = None
        self._one_way_data: Optional[Tuple[np.ndarray, np.ndarray, str, str]] = None
        self._two_way_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]] = None
        self._cid: Optional[int] = None

    def bind_canvas(self, canvas: FigureCanvasTkAgg) -> None:
        self._canvas = canvas
        self._cid = canvas.mpl_connect("button_press_event", self._on_click)

    def show_line(self, x: np.ndarray, y: np.ndarray, x_label: str, output_label: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y, marker="o", linewidth=1.5, markersize=4)
        ax.set_xlabel(x_label)
        ax.set_ylabel(output_label)
        ax.set_title(f"{output_label} vs {x_label}")
        ax.grid(True, alpha=0.35)
        self._one_way_data = (x, y, x_label, output_label)
        self._two_way_data = None
        self._last_mode = "line"
        self._info_callback("Click a plotted point to inspect its coordinates.")
        if self._canvas:
            self._canvas.draw_idle()

    def show_surface(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        zz: np.ndarray,
        x_label: str,
        y_label: str,
        output_label: str,
    ) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha=0.75)
        ax.scatter(xx, yy, zz, s=8, depthshade=False)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(output_label)
        ax.set_title(f"{output_label} surface")
        self._one_way_data = None
        self._two_way_data = (xx, yy, zz, x_label, y_label, output_label)
        self._last_mode = "surface"
        self._info_callback("Rotate the chart as needed. Click near a grid point to inspect its coordinates.")
        if self._canvas:
            self._canvas.draw_idle()

    def _on_click(self, event) -> None:
        if self._last_mode == "line" and self._one_way_data is not None:
            x, y, x_label, output_label = self._one_way_data
            if event.xdata is None or event.ydata is None:
                return
            distances = (x - event.xdata) ** 2 + (y - event.ydata) ** 2
            idx = int(np.argmin(distances))
            self._info_callback(
                f"Nearest point: {x_label} = {x[idx]:.6g}, {output_label} = {y[idx]:.6g}"
            )
            return

        if self._last_mode == "surface" and self._two_way_data is not None:
            xx, yy, zz, x_label, y_label, output_label = self._two_way_data
            if event.x is None or event.y is None:
                return

            ax = self.figure.axes[0]
            points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

            projected = []
            for px, py, pz in points:
                x3, y3, _ = proj3d.proj_transform(px, py, pz, ax.get_proj())
                xdisp, ydisp = ax.transData.transform((x3, y3))
                projected.append((xdisp, ydisp))

            projected_arr = np.array(projected)
            distances = (projected_arr[:, 0] - event.x) ** 2 + (projected_arr[:, 1] - event.y) ** 2
            idx = int(np.argmin(distances))
            nearest = points[idx]

            self._info_callback(
                f"Nearest point: {x_label} = {nearest[0]:.6g}, "
                f"{y_label} = {nearest[1]:.6g}, {output_label} = {nearest[2]:.6g}"
            )

class VariableControl:
    def __init__(
        self,
        parent: ttk.Frame,
        title: str,
        catalog: ParameterCatalog,
        range_manager: RangeManager,
        on_variable_changed: Callable[[], None],
        get_current_values: Callable[[], Dict[str, float]],
    ) -> None:
        self._catalog = catalog
        self._range_manager = range_manager
        self._on_variable_changed = on_variable_changed

        self.frame = ttk.LabelFrame(parent, text=title, padding=10)

        self.variable_var = tk.StringVar(value="None")
        self.min_var = tk.StringVar()
        self.max_var = tk.StringVar()
        self.slider_var = tk.IntVar(value=0)
        self.current_value_var = tk.StringVar(value="-")
        self._get_current_values = get_current_values

        self._entry_enabled = False
        self._state = VariableRangeState(None, 0.0, 1.0, RangeGrid.build(0.0, 1.0))
        self._previous_min_text = ""
        self._previous_max_text = ""

        ttk.Label(self.frame, text="Variable").grid(row=0, column=0, sticky="w")
        self.variable_combo = ttk.Combobox(self.frame, textvariable=self.variable_var, state="readonly", width=34)
        self.variable_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 0))

        ttk.Label(self.frame, text="Minimum").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.min_entry = ttk.Entry(self.frame, textvariable=self.min_var, width=18)
        self.min_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))

        ttk.Label(self.frame, text="Maximum").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.max_entry = ttk.Entry(self.frame, textvariable=self.max_var, width=18)
        self.max_entry.grid(row=1, column=3, sticky="ew", pady=(8, 0))

        ttk.Label(self.frame, text="Scenario point").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.scale = ttk.Scale(self.frame, from_=0, to=29, variable=self.slider_var, orient=tk.HORIZONTAL, command=self._on_slider_move)
        self.scale.grid(row=2, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(10, 0))

        ttk.Label(self.frame, text="Selected value").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Label(self.frame, textvariable=self.current_value_var).grid(row=3, column=1, columnspan=3, sticky="w", padx=(8, 0), pady=(8, 0))

        for col in range(4):
            self.frame.columnconfigure(col, weight=1 if col > 0 else 0)

        self.variable_combo.bind("<<ComboboxSelected>>", self._handle_variable_change)
        self.min_entry.bind("<FocusOut>", self._handle_range_edit)
        self.max_entry.bind("<FocusOut>", self._handle_range_edit)
        self.min_entry.bind("<Return>", self._handle_range_edit)
        self.max_entry.bind("<Return>", self._handle_range_edit)

    @property
    def selected_key(self) -> Optional[str]:
        label = self.variable_var.get()
        if label == "None":
            return None
        for spec in self._catalog.all_specs():
            if spec.label == label:
                return spec.key
        return None

    @property
    def state(self) -> VariableRangeState:
        return self._state

    def set_variable_options(self, labels: List[str], preserve_current: bool = True) -> None:
        current = self.variable_var.get()
        self.variable_combo["values"] = labels
        if preserve_current and current in labels:
            self.variable_var.set(current)
        elif labels:
            self.variable_var.set(labels[0])

    def refresh_for_current_variable(self, current_values: Dict[str, float]) -> None:
        key = self.selected_key
        if key is None:
            self._state = self._range_manager.create_default_state(None, current_values)
            self._set_entries_enabled(False)
            self.min_var.set("")
            self.max_var.set("")
            self.current_value_var.set("-")
            self.slider_var.set(0)
            self.scale.state(["disabled"])
            return

        if self._state.key != key:
            self._state = self._range_manager.create_default_state(key, current_values)
            self.slider_var.set(0)

        self._set_entries_enabled(True)
        self.scale.state(["!disabled"])
        self._apply_state_to_widgets()

    def scenario_values(self) -> Optional[np.ndarray]:
        if self._state.key is None:
            return None
        return self._state.grid.values

    def current_slider_value(self) -> Optional[float]:
        if self._state.key is None:
            return None
        return self._state.grid.value_at(self.slider_var.get())

    def _set_entries_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.min_entry.configure(state=state)
        self.max_entry.configure(state=state)
        self._entry_enabled = enabled

    def _apply_state_to_widgets(self) -> None:
        if self._state.key is None:
            return
        spec = self._catalog.spec(self._state.key)
        self.min_var.set(ValueFormatter.display_value(self._state.min_value, spec.is_percentage))
        self.max_var.set(ValueFormatter.display_value(self._state.max_value, spec.is_percentage))
        self._previous_min_text = self.min_var.get()
        self._previous_max_text = self.max_var.get()
        index = self._state.grid.index_of_closest(self._state.grid.value_at(self.slider_var.get()))
        self.slider_var.set(index)
        self.current_value_var.set(ValueFormatter.display_value(self._state.grid.value_at(index), spec.is_percentage))

    def _handle_variable_change(self, _event=None) -> None:
        self._on_variable_changed()

    def _handle_range_edit(self, _event=None) -> None:
        if not self._entry_enabled or self._state.key is None:
            return

        spec = self._catalog.spec(self._state.key)
        try:
            new_min = ValueFormatter.parse_user_value(self.min_var.get(), spec.is_percentage)
            new_max = ValueFormatter.parse_user_value(self.max_var.get(), spec.is_percentage)
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numeric values.")
            self.min_var.set(self._previous_min_text)
            self.max_var.set(self._previous_max_text)
            return

        current_values = self._current_value_context()
        is_valid, error_message = self._range_manager.validate(self._state.key, new_min, new_max, current_values)
        if not is_valid:
            messagebox.showerror("Out of range", error_message)
            self.min_var.set(self._previous_min_text)
            self.max_var.set(self._previous_max_text)
            return

        previous_value = self._state.grid.value_at(self.slider_var.get())
        self._state = self._range_manager.update_state(self._state, new_min, new_max)
        self._previous_min_text = ValueFormatter.display_value(new_min, spec.is_percentage)
        self._previous_max_text = ValueFormatter.display_value(new_max, spec.is_percentage)
        self.min_var.set(self._previous_min_text)
        self.max_var.set(self._previous_max_text)

        new_index = self._state.grid.index_of_closest(previous_value)
        self.slider_var.set(new_index)
        self.current_value_var.set(ValueFormatter.display_value(self._state.grid.value_at(new_index), spec.is_percentage))

    def _current_value_context(self) -> Dict[str, float]:
        values = self._get_current_values()
        if self._state.key is not None:
            values[self._state.key] = self._state.max_value
        return values

    def _on_slider_move(self, _value) -> None:
        if self._state.key is None:
            self.current_value_var.set("-")
            return
        spec = self._catalog.spec(self._state.key)
        selected = self._state.grid.value_at(self.slider_var.get())
        self.current_value_var.set(ValueFormatter.display_value(selected, spec.is_percentage))

class ConstantInputControl:
    def __init__(self, parent: ttk.Frame, spec: InputSpec) -> None:
        self.spec = spec
        self.var = tk.StringVar()
        self._last_valid_value = spec.default

        self.label = ttk.Label(parent, text=spec.label)
        self.entry = ttk.Entry(parent, textvariable=self.var, width=18)

        if spec.is_percentage:
            self.suffix = ttk.Label(parent, text="%")
        else:
            self.suffix = ttk.Label(parent, text="")

        self.set_value(spec.default)

        self.entry.bind("<FocusOut>", self._format_on_focus_out)
        self.entry.bind("<Return>", self._format_on_focus_out)

    def grid(self, row: int) -> None:
        self.label.grid(row=row, column=0, sticky="w", pady=2)
        self.entry.grid(row=row, column=1, sticky="ew", padx=(8, 6), pady=2)
        self.suffix.grid(row=row, column=2, sticky="w", pady=2)

    def set_enabled(self, enabled: bool) -> None:
        self.entry.configure(state="normal" if enabled else "disabled")

    def set_value(self, value: float) -> None:
        self._last_valid_value = value
        self.var.set(self._format_value(value))

    def get_value(self) -> float:
        text = self.var.get().strip().replace(",", "")
        value = float(text)
        if self.spec.is_percentage:
            value = value / 100.0
        lower = self.spec.minimum
        upper = self.spec.maximum
        if value < lower or value > upper:
            raise ValueError(
                f"{self.spec.label} must be within "
                f"[{ValueFormatter.display_value(lower, self.spec.is_percentage)}, "
                f"{ValueFormatter.display_value(upper, self.spec.is_percentage)}]"
            )
        return value

    def restore_previous(self) -> None:
        self.var.set(self._format_value(self._last_valid_value))

    def _format_on_focus_out(self, _event=None) -> None:
        try:
            value = self.get_value()
            self._last_valid_value = value
            self.var.set(self._format_value(value))
        except ValueError:
            self.restore_previous()

    def _format_value(self, value: float) -> str:
        if self.spec.is_percentage:
            return f"{value * 100:.6g}"
        if abs(value) >= 1000:
            if float(value).is_integer():
                return f"{int(round(value)):,}"
            return f"{value:,.6f}".rstrip("0").rstrip(".")
        return f"{value:.6g}"


class SensitivityApp:
    OUTPUT_OPTIONS = ("Day1Gain", "Profit")

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("ERM Sensitivity Explorer")
        self.root.geometry("1400x860")

        self.catalog = ParameterCatalog()
        self.model = ERMModel()
        self.range_manager = RangeManager(self.catalog)
        self.service = ERMSensitivityService(self.model)

        self.output_var = tk.StringVar(value="Profit")
        self.info_var = tk.StringVar(value="Ready.")
        self.figure = Figure(figsize=(8.5, 6.2))
        self.plot_controller = PlotController(self.figure, self.info_var.set)
        self.constant_controls: Dict[str, ConstantInputControl] = {}

        self._build_layout()
        self._initialize_controls()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="nsw")
        controls.columnconfigure(0, weight=1)

        self.var1_control = VariableControl(
            controls,
            "Variable 1",
            self.catalog,
            self.range_manager,
            on_variable_changed=self._handle_variable_selection_change,
            get_current_values=lambda: self._read_constant_input_values(silent=True),
        )
        self.var1_control.frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        self.var2_control = VariableControl(
            controls,
            "Variable 2",
            self.catalog,
            self.range_manager,
            on_variable_changed=self._handle_variable_selection_change,
            get_current_values=lambda: self._read_constant_input_values(silent=True),
        )
        self.var2_control.frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        output_frame = ttk.LabelFrame(controls, text="Output", padding=10)
        output_frame.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Output metric").grid(row=0, column=0, sticky="w")
        self.output_combo = ttk.Combobox(output_frame, textvariable=self.output_var, state="readonly", values=self.OUTPUT_OPTIONS)
        self.output_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.constants_frame = ttk.LabelFrame(controls, text="Constant inputs", padding=10)
        self.constants_frame.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        self.constants_frame.columnconfigure(1, weight=1)

        self.update_button = ttk.Button(controls, text="Update", command=self.update_chart)
        self.update_button.grid(row=4, column=0, sticky="ew")


        chart_frame = ttk.Frame(self.root, padding=12)
        chart_frame.grid(row=0, column=1, sticky="nsew")
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        chart_frame.rowconfigure(1, weight=0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.plot_controller.bind_canvas(self.canvas)

        info_frame = ttk.LabelFrame(chart_frame, text="Selected point", padding=10)
        info_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(info_frame, textvariable=self.info_var).grid(row=0, column=0, sticky="w")

    def _build_constant_inputs(self) -> None:
        for child in self.constants_frame.winfo_children():
            child.destroy()

        self.constant_controls = {}
        for row, spec in enumerate(self.catalog.all_specs()):
            control = ConstantInputControl(self.constants_frame, spec)
            control.grid(row=row)
            self.constant_controls[spec.key] = control


    def _initialize_controls(self) -> None:
        self._build_constant_inputs()
        all_labels = ["None"] + [spec.label for spec in self.catalog.all_specs()]
        self.var1_control.set_variable_options(all_labels, preserve_current=False)
        self.var2_control.set_variable_options(all_labels, preserve_current=False)
        self.var1_control.variable_var.set(self.catalog.label_for("risk_free_rate"))
        self.var2_control.variable_var.set("None")
        self._handle_variable_selection_change()

    def _refresh_constant_input_states(self) -> None:
        selected = {self.var1_control.selected_key, self.var2_control.selected_key}
        for key, control in self.constant_controls.items():
            control.set_enabled(key not in selected)

    def _handle_variable_selection_change(self) -> None:
        key1 = self.var1_control.selected_key
        key2 = self.var2_control.selected_key

        labels1 = ["None"] + [spec.label for spec in self.catalog.all_specs() if spec.key != key2]
        labels2 = ["None"] + [spec.label for spec in self.catalog.all_specs() if spec.key != key1]

        self.var1_control.set_variable_options(labels1)
        self.var2_control.set_variable_options(labels2)

        current_values = self._read_constant_input_values(silent=True)
        self.var1_control.refresh_for_current_variable(current_values)
        self.var2_control.refresh_for_current_variable(current_values)
        self._refresh_constant_input_states()

    def _read_constant_input_values(self, silent: bool = False) -> Dict[str, float]:
        values = {spec.key: spec.default for spec in self.catalog.all_specs()}

        for key, control in self.constant_controls.items():
            try:
                values[key] = control.get_value()
            except ValueError as exc:
                control.restore_previous()
                if not silent:
                    messagebox.showerror("Invalid constant input", str(exc))
                values[key] = control.spec.default

        return values

    def _base_parameters(self) -> ERMParameters:
        values = self._read_constant_input_values(silent=False)
        return self.catalog.default_parameters().updated(values)

    def update_chart(self) -> None:
        key1 = self.var1_control.selected_key
        key2 = self.var2_control.selected_key

        if key1 is None:
            messagebox.showerror("Selection required", "Please choose at least one input variable.")
            return

        base_params = self._base_parameters()
        current_values = self._read_constant_input_values(silent=True)
        self.var1_control.refresh_for_current_variable(current_values)
        self.var2_control.refresh_for_current_variable(current_values)
        output_key = self.output_var.get()

        if key2 is None:
            x_values = self.var1_control.scenario_values()
            if x_values is None:
                return
            x, y = self.service.one_way(base_params, key1, x_values, output_key)
            self.plot_controller.show_line(x, y, self.catalog.label_for(key1), output_key)
            return

        x_values = self.var1_control.scenario_values()
        y_values = self.var2_control.scenario_values()
        if x_values is None or y_values is None:
            return

        xx, yy, zz = self.service.two_way(base_params, key1, x_values, key2, y_values, output_key)
        self.plot_controller.show_surface(
            xx,
            yy,
            zz,
            self.catalog.label_for(key1),
            self.catalog.label_for(key2),
            output_key,
        )

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = SensitivityApp()
    app.run()


if __name__ == "__main__":
    main()
