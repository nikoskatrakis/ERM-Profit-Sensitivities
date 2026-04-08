from __future__ import annotations

from dataclasses import dataclass, replace
from math import erf, exp, log, sqrt
from typing import Dict, Optional, Sequence, Tuple
import csv
import random
import numpy as np

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
        scr = np.maximum(accumulated_loan - house_price_scr,0)

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

def resolve_bounds(catalog: ParameterCatalog, key: str, current_values: Dict[str, float]) -> Tuple[float, float]:
    spec = catalog.spec(key)
    upper = spec.maximum
    if spec.dynamic_max_key:
        upper = min(upper, current_values[spec.dynamic_max_key])
    return spec.minimum, upper


def base_case_dict(catalog: ParameterCatalog) -> Dict[str, float]:
    params = catalog.default_parameters()
    return {
        "rw_hpi": params.rw_hpi,
        "deferment_rate": params.deferment_rate,
        "loan_rate": params.loan_rate,
        "house_price_volatility": params.house_price_volatility,
        "house_price_start": params.house_price_start,
        "time_years": params.time_years,
        "funding_cost": params.funding_cost,
        "loan_amount": params.loan_amount,
        "coc_rate": params.coc_rate,
        "risk_free_rate": params.risk_free_rate,
        "scr_level": params.scr_level,
        "scr_decay_factor": params.scr_decay_factor,
    }


def scenario_to_params(base: ERMParameters, scenario: Dict[str, float]) -> ERMParameters:
    return base.updated(scenario)


def generate_extreme_scenarios(catalog: ParameterCatalog) -> list[Dict[str, float]]:
    base = base_case_dict(catalog)
    scenarios = []

    for key in base.keys():
        lo_base = dict(base)
        hi_base = dict(base)

        lo, hi = resolve_bounds(catalog, key, lo_base)
        lo_base[key] = lo
        hi_base[key] = hi

        if key == "house_price_start":
            loan_lo, loan_hi = resolve_bounds(catalog, "loan_amount", lo_base)
            lo_base["loan_amount"] = min(lo_base["loan_amount"], loan_hi)

            loan_lo, loan_hi = resolve_bounds(catalog, "loan_amount", hi_base)
            hi_base["loan_amount"] = min(hi_base["loan_amount"], loan_hi)

        scenarios.append(lo_base)
        scenarios.append(hi_base)

    return scenarios


def generate_corner_scenarios(catalog: ParameterCatalog) -> list[Dict[str, float]]:
    base = base_case_dict(catalog)
    scenarios = []

    keys = list(base.keys())
    pairs = [
        ("risk_free_rate", "deferment_rate"),
        ("loan_amount", "house_price_start"),
        ("loan_rate", "time_years"),
        ("house_price_volatility", "scr_level"),
        ("rw_hpi", "risk_free_rate"),
        ("coc_rate", "scr_decay_factor"),
    ]

    for k1, k2 in pairs:
        s = dict(base)
        lo1, hi1 = resolve_bounds(catalog, k1, s)
        lo2, hi2 = resolve_bounds(catalog, k2, s)

        s[k1] = hi1
        s[k2] = hi2

        if "house_price_start" in (k1, k2):
            _, loan_hi = resolve_bounds(catalog, "loan_amount", s)
            s["loan_amount"] = min(s["loan_amount"], loan_hi)

        scenarios.append(s)

        s = dict(base)
        lo1, hi1 = resolve_bounds(catalog, k1, s)
        lo2, hi2 = resolve_bounds(catalog, k2, s)

        s[k1] = lo1
        s[k2] = lo2

        if "house_price_start" in (k1, k2):
            _, loan_hi = resolve_bounds(catalog, "loan_amount", s)
            s["loan_amount"] = min(s["loan_amount"], loan_hi)

        scenarios.append(s)

    return scenarios


def generate_random_scenarios(catalog: ParameterCatalog, n: int, seed: int = 42) -> list[Dict[str, float]]:
    rng = random.Random(seed)
    base = base_case_dict(catalog)
    scenarios = []

    keys = list(base.keys())

    for _ in range(n):
        s = dict(base)

        for key in keys:
            lo, hi = resolve_bounds(catalog, key, s)
            s[key] = rng.uniform(lo, hi)

            if key == "house_price_start":
                _, loan_hi = resolve_bounds(catalog, "loan_amount", s)
                s["loan_amount"] = min(s["loan_amount"], loan_hi)

        _, loan_hi = resolve_bounds(catalog, "loan_amount", s)
        s["loan_amount"] = rng.uniform(catalog.spec("loan_amount").minimum, loan_hi)

        scenarios.append(s)

    return scenarios


def deduplicate_scenarios(scenarios: list[Dict[str, float]]) -> list[Dict[str, float]]:
    seen = set()
    unique = []

    for s in scenarios:
        key = tuple(round(s[k], 12) for k in sorted(s.keys()))
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def write_scenarios_to_csv(path: str, scenarios: list[Dict[str, float]], model: ERMModel, base_params: ERMParameters) -> None:
    fieldnames = [
        "rw_hpi",
        "deferment_rate",
        "loan_rate",
        "house_price_volatility",
        "house_price_start",
        "time_years",
        "funding_cost",
        "loan_amount",
        "coc_rate",
        "risk_free_rate",
        "scr_level",
        "scr_decay_factor",
        "Day1Gain",
        "Profit",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for scenario in scenarios:
            params = scenario_to_params(base_params, scenario)
            results = model.calculate(params)

            row = dict(scenario)
            row["Day1Gain"] = results.day1_gain
            row["Profit"] = results.profit
            writer.writerow(row)


def main() -> None:
    catalog = ParameterCatalog()
    model = ERMModel()
    base_params = catalog.default_parameters()

    scenarios = []
    scenarios.append(base_case_dict(catalog))
    scenarios.extend(generate_extreme_scenarios(catalog))
    scenarios.extend(generate_corner_scenarios(catalog))
    scenarios.extend(generate_random_scenarios(catalog, n=150, seed=42))

    scenarios = deduplicate_scenarios(scenarios)

    write_scenarios_to_csv(
        path="erm_test_cases.csv",
        scenarios=scenarios,
        model=model,
        base_params=base_params,
    )

    print(f"Wrote {len(scenarios)} scenarios to erm_test_cases.csv")


if __name__ == "__main__":
    main()