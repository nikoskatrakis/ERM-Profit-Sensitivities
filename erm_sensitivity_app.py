from __future__ import annotations

from dataclasses import dataclass, replace
from math import erf, exp, log, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =============================
# Domain model
# =============================

@dataclass(frozen=True)
class InputSpec:
    key: str
    label: str
    default: float
    min_value: float
    max_value: float
    step_count: int = 30
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

    def with_updates(self, updates: Dict[str, float]) -> "ERMParameters":
        return replace(self, **updates)


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
            raise ValueError("p must be in (0, 1)")

        a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239]
        b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
        c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
        d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]

        plow = 0.02425
        phigh = 1.0 - plow

        if p < plow:
            q = sqrt(-2.0 * log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0)
        if p > phigh:
            q = sqrt(-2.0 * log(1.0 - p))
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0)

        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


class ERMModel:
    def __init__(self, normal: Optional[NormalDistribution] = None) -> None:
        self._normal = normal or NormalDistribution()

    def calculate(self, params: ERMParameters) -> ERMResults:
        sigma = params.house_price_volatility * sqrt(params.time_years)
        ln_s0 = log(params.house_price_start)

        scr_mu = ln_s0 + (params.rw_hpi - 0.5 * params.house_price_volatility**2) * params.time_years
        scr_percentile = self._normal.ppf(1.0 - params.scr_level)
        ln_price_scr = scr_mu + sigma * scr_percentile
        price_scr = exp(ln_price_scr)

        accumulated_loan = params.loan_amount * (1.0 + params.loan_rate) ** params.time_years
        scr = accumulated_loan - price_scr

        base = exp(-params.scr_decay_factor) / (1.0 + params.risk_free_rate)
        if abs(1.0 - base) < 1e-12:
            scr_annuity_factor = params.time_years
        else:
            scr_annuity_factor = base * (1.0 - base**params.time_years) / (1.0 - base)
        cost_of_capital = params.coc_rate * scr * scr_annuity_factor

        pricing_drift = params.risk_free_rate - params.deferment_rate
        pricing_mu = ln_s0 + (pricing_drift - 0.5 * params.house_price_volatility**2) * params.time_years
        d = (log(accumulated_loan) - pricing_mu) / sigma
        term_1 = accumulated_loan * (1.0 - self._normal.cdf(d))
        term_2 = exp(pricing_mu + 0.5 * sigma**2) * self._normal.cdf((log(accumulated_loan) - pricing_mu - sigma**2) / sigma)
        expected_payoff_at_sale = term_1 + term_2
        pv_expected_payoff = expected_payoff_at_sale * exp(-params.risk_free_rate * params.time_years)

        funding_cost_value = params.funding_cost * params.loan_amount
        day1_gain = pv_expected_payoff - params.loan_amount - funding_cost_value
        profit = day1_gain - cost_of_capital
        profit_at_sale = profit * exp(params.time_years * params.risk_free_rate)

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


# =============================
# Configuration and validation
# =============================

class InputCatalog:
    def __init__(self) -> None:
        self._specs: Dict[str, InputSpec] = {
            "rw_hpi": InputSpec("rw_hpi", "RW house price inflation", 0.02, -0.10, 0.30, is_percentage=True),
            "deferment_rate": InputSpec("deferment_rate", "Deferment rate", 0.035, -0.10, 0.10, is_percentage=True),
            "loan_rate": InputSpec("loan_rate", "Loan rate", 0.065, 0.01, 0.20, is_percentage=True),
            "house_price_volatility": InputSpec("house_price_volatility", "House price volatility", 0.13, 0.01, 0.30, is_percentage=True),
            "house_price_start": InputSpec("house_price_start", "House price at start", 100_000.0, 50_000.0, 50_000_000.0),
            "time_years": InputSpec("time_years", "Time (years)", 10.0, 1.0, 50.0),
            "funding_cost": InputSpec("funding_cost", "Funding cost", 0.02, 0.0, 0.20, is_percentage=True),
            "loan_amount": InputSpec("loan_amount", "Loan amount", 30_000.0, 10_000.0, 100_000.0, dynamic_max_key="house_price_start"),
            "coc_rate": InputSpec("coc_rate", "SCR CoC percentage used for pricing", 0.045, 0.01, 0.10, is_percentage=True),
            "risk_free_rate": InputSpec("risk_free_rate", "Risk-free rate", 0.045, 0.0005, 0.20, is_percentage=True),
            "scr_level": InputSpec("scr_level", "SCR level", 0.995, 0.50, 0.9999, is_percentage=True),
            "scr_decay_factor": InputSpec("scr_decay_factor", "SCR decay factor", 0.12, 0.03, 0.25, is_percentage=True),
        }

    def options(self, include_none: bool = False) -> List[str]:
        labels = [spec.label for spec in self._specs.values()]
        return (["None"] + labels) if include_none else labels

    def label_to_key(self, label: str) -> Optional[str]:
        if label == "None":
            return None
        for key, spec in self._specs.items():
            if spec.label == label:
                return key
        return None

    def get(self, key: str) -> InputSpec:
        return self._specs[key]

    def specs(self) -> Dict[str, InputSpec]:
        return self._specs

    def resolved_bounds(self, key: str, current_values: Dict[str, float]) -> Tuple[float, float]:
        spec = self.get(key)
        upper = spec.max_value
        if spec.dynamic_max_key is not None:
            upper = min(upper, float(current_values.get(spec.dynamic_max_key, upper)))
        return spec.min_value, upper


class RangeValidator:
    def __init__(self, catalog: InputCatalog) -> None:
        self._catalog = catalog

    def validate_range(self, key: str, minimum: float, maximum: float, current_values: Dict[str, float]) -> Tuple[bool, str]:
        lower, upper = self._catalog.resolved_bounds(key, current_values)
        if minimum < lower or maximum > upper:
            return False, f"{self._catalog.get(key).label}: values must stay within [{self._format(lower)}, {self._format(upper)}]."
        if minimum >= maximum:
            return False, f"{self._catalog.get(key).label}: minimum must be strictly smaller than maximum."
        return True, ""

    @staticmethod
    def _format(value: float) -> str:
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.4f}".rstrip("0").rstrip(".")


class ScenarioGridBuilder:
    def __init__(self, model: ERMModel) -> None:
        self._model = model

    def line_data(self, params: ERMParameters, variable_key: str, minimum: float, maximum: float, output_key: str, steps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(minimum, maximum, steps)
        y = []
        for value in x:
            updated = params.with_updates({variable_key: float(value)})
            y.append(self._output_value(self._model.calculate(updated), output_key))
        return x, np.array(y)

    def surface_data(
        self,
        params: ERMParameters,
        x_key: str,
        x_min: float,
        x_max: float,
        y_key: str,
        y_min: float,
        y_max: float,
        output_key: str,
        steps: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(x_min, x_max, steps)
        y = np.linspace(y_min, y_max, steps)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                updated = params.with_updates({x_key: float(xx[i, j]), y_key: float(yy[i, j])})
                zz[i, j] = self._output_value(self._model.calculate(updated), output_key)
        return xx, yy, zz

    @staticmethod
    def _output_value(results: ERMResults, output_key: str) -> float:
        return results.day1_gain if output_key == "Day1Gain" else results.profit


# =============================
# UI helpers
# =============================

class VariableControlFrame(ttk.LabelFrame):
    def __init__(self, master: tk.Widget, title: str, catalog: InputCatalog, allow_none: bool = False) -> None:
        super().__init__(master, text=title, padding=10)
        self._catalog = catalog
        self._allow_none = allow_none
        self._last_valid_range: Optional[Tuple[float, float]] = None

        self.variable_var = tk.StringVar(value="None" if allow_none else catalog.options()[0])
        self.min_var = tk.StringVar()
        self.max_var = tk.StringVar()
        self.info_var = tk.StringVar(value="")
        self.min_scale_var = tk.IntVar(value=0)
        self.max_scale_var = tk.IntVar(value=29)

        ttk.Label(self, text="Variable").grid(row=0, column=0, sticky="w")
        self.variable_combo = ttk.Combobox(self, textvariable=self.variable_var, values=catalog.options(include_none=allow_none), state="readonly", width=34)
        self.variable_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 6))
        self.variable_combo.bind("<<ComboboxSelected>>", self._on_variable_changed)

        ttk.Label(self, textvariable=self.info_var, foreground="#555555").grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(self, text="Min").grid(row=3, column=0, sticky="w")
        ttk.Label(self, text="Max").grid(row=3, column=1, sticky="w")
        self.min_entry = ttk.Entry(self, textvariable=self.min_var, width=18)
        self.max_entry = ttk.Entry(self, textvariable=self.max_var, width=18)
        self.min_entry.grid(row=4, column=0, sticky="ew", padx=(0, 6))
        self.max_entry.grid(row=4, column=1, sticky="ew")

        ttk.Label(self, text="Min slider (30 steps)").grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.min_scale = tk.Scale(self, from_=0, to=29, orient="horizontal", variable=self.min_scale_var, showvalue=False, command=self._on_slider_changed)
        self.min_scale.grid(row=6, column=0, columnspan=2, sticky="ew")

        ttk.Label(self, text="Max slider (30 steps)").grid(row=7, column=0, columnspan=2, sticky="w")
        self.max_scale = tk.Scale(self, from_=0, to=29, orient="horizontal", variable=self.max_scale_var, showvalue=False, command=self._on_slider_changed)
        self.max_scale.grid(row=8, column=0, columnspan=2, sticky="ew")

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self._apply_variable_state(self.selected_key(), {})

    def selected_key(self) -> Optional[str]:
        return self._catalog.label_to_key(self.variable_var.get())

    def refresh_dynamic_bounds(self, current_values: Dict[str, float]) -> None:
        self._apply_variable_state(self.selected_key(), current_values, keep_existing=True)

    def validate_and_commit(self, current_values: Dict[str, float], validator: RangeValidator) -> Tuple[bool, str]:
        key = self.selected_key()
        if key is None:
            return True, ""
        try:
            minimum = float(self.min_var.get())
            maximum = float(self.max_var.get())
        except ValueError:
            self.restore_last_valid(current_values)
            return False, f"{self._catalog.get(key).label}: enter numeric minimum and maximum values."

        ok, message = validator.validate_range(key, minimum, maximum, current_values)
        if not ok:
            self.restore_last_valid(current_values)
            return False, message + " Previous valid values were restored."

        self._last_valid_range = (minimum, maximum)
        self._set_slider_positions(minimum, maximum, current_values)
        return True, ""

    def get_range(self) -> Optional[Tuple[float, float]]:
        key = self.selected_key()
        if key is None:
            return None
        return float(self.min_var.get()), float(self.max_var.get())

    def midpoint(self) -> Optional[float]:
        selected_range = self.get_range()
        if selected_range is None:
            return None
        minimum, maximum = selected_range
        return 0.5 * (minimum + maximum)

    def restore_last_valid(self, current_values: Dict[str, float]) -> None:
        key = self.selected_key()
        if key is None:
            return
        self._apply_variable_state(key, current_values, keep_existing=False)

    def _on_variable_changed(self, _event: object) -> None:
        self._apply_variable_state(self.selected_key(), {}, keep_existing=False)

    def _on_slider_changed(self, _value: str) -> None:
        key = self.selected_key()
        if key is None:
            self.min_var.set("")
            self.max_var.set("")
            return

        lower, upper = self._catalog.resolved_bounds(key, {})
        min_value = self._position_to_value(self.min_scale_var.get(), lower, upper)
        max_value = self._position_to_value(self.max_scale_var.get(), lower, upper)
        if min_value >= max_value:
            if self.min_scale_var.get() >= self.max_scale_var.get():
                if self.max_scale_var.get() < 29:
                    self.max_scale_var.set(self.max_scale_var.get() + 1)
                else:
                    self.min_scale_var.set(max(0, self.min_scale_var.get() - 1))
            min_value = self._position_to_value(self.min_scale_var.get(), lower, upper)
            max_value = self._position_to_value(self.max_scale_var.get(), lower, upper)
        self.min_var.set(self._format_number(min_value))
        self.max_var.set(self._format_number(max_value))

    def _apply_variable_state(self, key: Optional[str], current_values: Dict[str, float], keep_existing: bool = False) -> None:
        state = "normal" if key is not None else "disabled"
        self.min_entry.configure(state=state)
        self.max_entry.configure(state=state)
        self.min_scale.configure(state=state)
        self.max_scale.configure(state=state)

        if key is None:
            self.info_var.set("Select a variable")
            self.min_var.set("")
            self.max_var.set("")
            self._last_valid_range = None
            return

        spec = self._catalog.get(key)
        lower, upper = self._catalog.resolved_bounds(key, current_values)
        self.info_var.set(f"Allowed range: {self._format_number(lower)} to {self._format_number(upper)}")

        if keep_existing and self._last_valid_range is not None:
            minimum, maximum = self._last_valid_range
            minimum = max(lower, min(minimum, upper))
            maximum = max(lower, min(maximum, upper))
            if minimum >= maximum:
                minimum, maximum = lower, upper
        else:
            minimum = max(lower, spec.default)
            maximum = upper
            if minimum >= maximum:
                minimum, maximum = lower, upper

        self._last_valid_range = (minimum, maximum)
        self.min_var.set(self._format_number(minimum))
        self.max_var.set(self._format_number(maximum))
        self._set_slider_positions(minimum, maximum, current_values)

    def _set_slider_positions(self, minimum: float, maximum: float, current_values: Dict[str, float]) -> None:
        key = self.selected_key()
        if key is None:
            return
        lower, upper = self._catalog.resolved_bounds(key, current_values)
        self.min_scale_var.set(self._value_to_position(minimum, lower, upper))
        self.max_scale_var.set(self._value_to_position(maximum, lower, upper))

    @staticmethod
    def _value_to_position(value: float, lower: float, upper: float, steps: int = 30) -> int:
        if upper <= lower:
            return 0
        ratio = (value - lower) / (upper - lower)
        return int(round(ratio * (steps - 1)))

    @staticmethod
    def _position_to_value(position: int, lower: float, upper: float, steps: int = 30) -> float:
        if steps <= 1 or upper <= lower:
            return lower
        return lower + (upper - lower) * (position / (steps - 1))

    @staticmethod
    def _format_number(value: float) -> str:
        if abs(value) >= 1000:
            return f"{value:,.2f}" if abs(value) < 1_000_000 else f"{value:,.0f}"
        return f"{value:.6f}".rstrip("0").rstrip(".")


class ChartPresenter:
    def __init__(self, parent: tk.Widget) -> None:
        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.widget = self.canvas.get_tk_widget()
        self.widget.pack(fill="both", expand=True)
        self._artist_points: Optional[np.ndarray] = None
        self._labels: Optional[Tuple[str, Optional[str], str]] = None
        self._selection_callback = None
        self.canvas.mpl_connect("pick_event", self._on_pick)

    def set_selection_callback(self, callback) -> None:
        self._selection_callback = callback

    def draw_line(self, x: np.ndarray, y: np.ndarray, x_label: str, output_label: str) -> None:
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        line = ax.plot(x, y, marker="o", markersize=4, picker=5)[0]
        ax.set_title(f"{output_label} sensitivity")
        ax.set_xlabel(x_label)
        ax.set_ylabel(output_label)
        ax.grid(True, alpha=0.25)
        self._artist_points = np.column_stack([x, y])
        self._labels = (x_label, None, output_label)
        line._erm_points = self._artist_points  # type: ignore[attr-defined]
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def draw_surface(self, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, x_label: str, y_label: str, output_label: str) -> None:
        self.figure.clf()
        ax = self.figure.add_subplot(111, projection="3d")
        ax.plot_surface(xx, yy, zz, cmap="viridis", alpha=0.8, linewidth=0, antialiased=True)
        scatter = ax.scatter(xx.ravel(), yy.ravel(), zz.ravel(), c=zz.ravel(), cmap="viridis", s=14, alpha=0.45, picker=True)
        ax.set_title(f"{output_label} surface")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(output_label)
        self._artist_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        self._labels = (x_label, y_label, output_label)
        scatter._erm_points = self._artist_points  # type: ignore[attr-defined]
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_pick(self, event) -> None:
        if self._selection_callback is None:
            return
        points = getattr(event.artist, "_erm_points", None)
        if points is None or len(event.ind) == 0:
            return
        selected = points[event.ind[0]]
        self._selection_callback(selected, self._labels)


# =============================
# Application controller
# =============================

class ERMSensitivityApp:
    def __init__(self) -> None:
        self.catalog = InputCatalog()
        self.validator = RangeValidator(self.catalog)
        self.model = ERMModel()
        self.grid_builder = ScenarioGridBuilder(self.model)

        self.root = tk.Tk()
        self.root.title("ERM Sensitivity Explorer")
        self.root.geometry("1500x900")

        self.output_var = tk.StringVar(value="Profit")
        self.warning_var = tk.StringVar(value="")
        self.selection_var = tk.StringVar(value="Click a chart point to inspect the nearest grid value.")

        self._build_ui()
        self._refresh_dynamic_bounds()
        self.update_chart()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer)
        controls.pack(side="left", fill="y", padx=(0, 12))

        ttk.Label(controls, text="Output variable").pack(anchor="w")
        output_combo = ttk.Combobox(controls, textvariable=self.output_var, values=["Day1Gain", "Profit"], state="readonly", width=24)
        output_combo.pack(anchor="w", fill="x", pady=(2, 10))

        self.variable_1_frame = VariableControlFrame(controls, "Variable 1", self.catalog, allow_none=False)
        self.variable_1_frame.pack(fill="x", pady=(0, 10))
        self.variable_2_frame = VariableControlFrame(controls, "Variable 2", self.catalog, allow_none=True)
        self.variable_2_frame.pack(fill="x", pady=(0, 10))

        self.variable_1_frame.variable_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_dynamic_bounds())
        self.variable_2_frame.variable_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_dynamic_bounds())

        ttk.Button(controls, text="Update", command=self.update_chart).pack(anchor="w", pady=(4, 8))
        ttk.Label(controls, textvariable=self.warning_var, foreground="#b45309", wraplength=360, justify="left").pack(anchor="w", fill="x")

        chart_frame = ttk.Frame(outer)
        chart_frame.pack(side="left", fill="both", expand=True)
        self.chart_presenter = ChartPresenter(chart_frame)
        self.chart_presenter.set_selection_callback(self._show_selected_point)
        ttk.Label(chart_frame, textvariable=self.selection_var, wraplength=900, justify="left", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))

    def _current_parameter_values(self) -> Dict[str, float]:
        values = ERMParameters().__dict__.copy()
        for frame in (self.variable_1_frame, self.variable_2_frame):
            key = frame.selected_key()
            midpoint = frame.midpoint()
            if key is not None and midpoint is not None:
                values[key] = midpoint
        return values

    def _refresh_dynamic_bounds(self) -> None:
        current_values = self._current_parameter_values()
        self.variable_1_frame.refresh_dynamic_bounds(current_values)
        current_values = self._current_parameter_values()
        self.variable_2_frame.refresh_dynamic_bounds(current_values)

    def _build_base_parameters(self) -> ERMParameters:
        params = ERMParameters()
        updates: Dict[str, float] = {}
        for frame in (self.variable_1_frame, self.variable_2_frame):
            key = frame.selected_key()
            midpoint = frame.midpoint()
            if key is not None and midpoint is not None:
                updates[key] = midpoint
        return params.with_updates(updates)

    def update_chart(self) -> None:
        self.warning_var.set("")
        self._refresh_dynamic_bounds()
        current_values = self._current_parameter_values()

        messages: List[str] = []
        for frame in (self.variable_1_frame, self.variable_2_frame):
            ok, message = frame.validate_and_commit(current_values, self.validator)
            if not ok:
                messages.append(message)

        if messages:
            self.warning_var.set("\n".join(messages))
            current_values = self._current_parameter_values()

        variable_1 = self.variable_1_frame.selected_key()
        variable_2 = self.variable_2_frame.selected_key()
        if variable_1 is None:
            messagebox.showerror("Missing input", "Please choose at least one input variable.")
            return
        if variable_1 == variable_2 and variable_2 is not None:
            self.warning_var.set("Variable 2 matched Variable 1 and was ignored.")
            variable_2 = None
            self.variable_2_frame.variable_var.set("None")
            self.variable_2_frame.refresh_dynamic_bounds(current_values)

        base_params = self._build_base_parameters()
        output_key = self.output_var.get()

        if variable_2 is None:
            minimum, maximum = self.variable_1_frame.get_range()  # type: ignore[misc]
            x, y = self.grid_builder.line_data(base_params, variable_1, minimum, maximum, output_key)
            self.chart_presenter.draw_line(x, y, self.catalog.get(variable_1).label, output_key)
        else:
            x_min, x_max = self.variable_1_frame.get_range()  # type: ignore[misc]
            y_min, y_max = self.variable_2_frame.get_range()  # type: ignore[misc]
            if (variable_1 == "loan_amount" and x_max > y_min) or (variable_2 == "loan_amount" and y_max > x_min):
                self.warning_var.set((self.warning_var.get() + "\n" if self.warning_var.get() else "") + "Some plotted loan / house value combinations may be economically invalid because loan exceeds house value.")
            xx, yy, zz = self.grid_builder.surface_data(base_params, variable_1, x_min, x_max, variable_2, y_min, y_max, output_key)
            self.chart_presenter.draw_surface(xx, yy, zz, self.catalog.get(variable_1).label, self.catalog.get(variable_2).label, output_key)

    def _show_selected_point(self, point: np.ndarray, labels: Tuple[str, Optional[str], str]) -> None:
        x_label, y_label, z_label = labels
        if y_label is None:
            self.selection_var.set(f"Closest point: {x_label} = {point[0]:,.6g}, {z_label} = {point[1]:,.6g}")
        else:
            self.selection_var.set(
                f"Closest point: {x_label} = {point[0]:,.6g}, {y_label} = {point[1]:,.6g}, {z_label} = {point[2]:,.6g}"
            )

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = ERMSensitivityApp()
    app.run()


if __name__ == "__main__":
    main()
