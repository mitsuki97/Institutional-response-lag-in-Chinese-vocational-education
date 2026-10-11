"""Describe and test pathway-by-domain variation with upstream-event clustering."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "event_analysis" / "chains_all.csv"
OUTPUT = ROOT / "outputs" / "event_analysis" / "tables" / "industry_pathway_interaction.csv"
PATHWAYS = [
    "policy_to_firm",
    "policy_to_school",
    "firm_to_school",
    "standard_to_curriculum",
]
DOMAINS = ["AI", "New energy", "Smart elderly care", "Smart manufacturing"]
DOMAIN_MAP = {
    "人工智能领域": "AI",
    "新能源领域": "New energy",
    "智慧养老": "Smart elderly care",
    "智能制造": "Smart manufacturing",
}


def beta_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction used for the regularized incomplete beta function."""
    maximum_iterations, tolerance, minimum = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = math.copysign(1.0 / max(abs(d), minimum), d)
    fraction = d
    for m in range(1, maximum_iterations + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = math.copysign(1.0 / max(abs(1.0 + numerator * d), minimum), 1.0 + numerator * d)
        c = math.copysign(max(abs(1.0 + numerator / c), minimum), 1.0 + numerator / c)
        fraction *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = math.copysign(1.0 / max(abs(1.0 + numerator * d), minimum), 1.0 + numerator * d)
        c = math.copysign(max(abs(1.0 + numerator / c), minimum), 1.0 + numerator / c)
        change = d * c
        fraction *= change
        if abs(change - 1.0) < tolerance:
            return fraction
    raise RuntimeError("Incomplete-beta fraction did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    log_front += a * math.log(x) + b * math.log1p(-x)
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_fraction(a, b, x) / a
    return 1.0 - front * beta_fraction(b, a, 1.0 - x) / b


def f_survival_probability(statistic: float, numerator_df: int, denominator_df: int) -> float:
    x = numerator_df * statistic / (numerator_df * statistic + denominator_df)
    return 1.0 - regularized_beta(x, numerator_df / 2.0, denominator_df / 2.0)


def main() -> None:
    chains = pd.read_csv(INPUT)
    sample = chains[chains["use_in_main_auto"].astype(str).str.lower().eq("true")].copy()
    sample["lag_months_num"] = pd.to_numeric(sample["lag_months_num"], errors="coerce")
    sample["domain"] = sample["domain_from_file"].map(DOMAIN_MAP)
    sample = sample.dropna(subset=["lag_months_num", "domain"])

    cell_rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        for pathway in PATHWAYS:
            cell = sample[(sample["domain"] == domain) & (sample["chain_type_norm"] == pathway)]
            lags = cell["lag_months_num"]
            cell_rows.append(
                {
                    "record_type": "cell_summary",
                    "domain": domain,
                    "pathway": pathway,
                    "n_chains": len(cell),
                    "n_upstream_event_clusters": cell["upstream_event_id"].nunique(),
                    "mean_lag_months": round(float(lags.mean()), 2),
                    "median_lag_months": round(float(lags.median()), 2),
                    "within_24_months_pct": round(float((lags < 24).mean() * 100), 2),
                    "test_statistic": np.nan,
                    "df_numerator": np.nan,
                    "df_denominator": np.nan,
                    "p_value": np.nan,
                    "model": "Observed cell summary",
                }
            )

    pathway_dummies = np.column_stack(
        [(sample["chain_type_norm"] == pathway).to_numpy(float) for pathway in PATHWAYS[1:]]
    )
    domain_dummies = np.column_stack(
        [(sample["domain"] == domain).to_numpy(float) for domain in DOMAINS[1:]]
    )
    interactions = np.column_stack(
        [pathway_dummies[:, i] * domain_dummies[:, j] for i in range(3) for j in range(3)]
    )
    design = np.column_stack([np.ones(len(sample)), pathway_dummies, domain_dummies, interactions])
    outcome = np.log1p(sample["lag_months_num"].to_numpy(float))
    inverse = np.linalg.inv(design.T @ design)
    coefficients = inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients

    cluster_ids, cluster_index = np.unique(sample["upstream_event_id"].to_numpy(), return_inverse=True)
    meat = np.zeros((design.shape[1], design.shape[1]))
    for cluster in range(len(cluster_ids)):
        score = design[cluster_index == cluster].T @ residuals[cluster_index == cluster]
        meat += np.outer(score, score)
    n_observations, n_parameters, n_clusters = len(sample), design.shape[1], len(cluster_ids)
    correction = (n_clusters / (n_clusters - 1)) * ((n_observations - 1) / (n_observations - n_parameters))
    covariance = correction * inverse @ meat @ inverse

    restrictions = np.zeros((9, n_parameters))
    restrictions[:, -9:] = np.eye(9)
    wald_chi_square = float(
        (restrictions @ coefficients).T
        @ np.linalg.inv(restrictions @ covariance @ restrictions.T)
        @ (restrictions @ coefficients)
    )
    numerator_df = restrictions.shape[0]
    denominator_df = n_clusters - 1
    f_statistic = wald_chi_square / numerator_df
    p_value = f_survival_probability(f_statistic, numerator_df, denominator_df)
    cell_rows.append(
        {
            "record_type": "joint_interaction_test",
            "domain": "All four domains",
            "pathway": "All four pathways",
            "n_chains": n_observations,
            "n_upstream_event_clusters": n_clusters,
            "mean_lag_months": np.nan,
            "median_lag_months": np.nan,
            "within_24_months_pct": np.nan,
            "test_statistic": round(f_statistic, 4),
            "df_numerator": numerator_df,
            "df_denominator": denominator_df,
            "p_value": p_value,
            "model": "OLS log(1 + lag months), pathway x domain interaction; upstream-event clustered covariance",
        }
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cell_rows).to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
