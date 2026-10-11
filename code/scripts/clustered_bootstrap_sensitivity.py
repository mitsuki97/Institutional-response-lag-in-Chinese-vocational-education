"""Estimate pathway uncertainty by resampling upstream-event clusters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "event_analysis" / "chains_all.csv"
OUTPUT = ROOT / "outputs" / "event_analysis" / "tables" / "clustered_bootstrap_sensitivity.csv"
PATHWAYS = [
    "policy_to_firm",
    "policy_to_school",
    "firm_to_school",
    "standard_to_curriculum",
]
N_BOOT = 2000
SEED = 20261020


def metric_values(values: np.ndarray) -> dict[str, float]:
    return {
        "Mean lag (months)": float(np.mean(values)),
        "Median lag (months)": float(np.median(values)),
        "Within 12 months (%)": float(np.mean(values < 12) * 100),
        "Within 24 months (%)": float(np.mean(values < 24) * 100),
    }


def clustered_intervals(groups: list[np.ndarray], seed: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in metric_values(np.concatenate(groups))}
    for _ in range(N_BOOT):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[index] for index in sampled_groups])
        for name, value in metric_values(sample).items():
            draws[name].append(value)
    return {
        name: tuple(np.quantile(values, [0.025, 0.975]))
        for name, values in draws.items()
    }


def main() -> None:
    chains = pd.read_csv(INPUT)
    main_sample = chains[chains["use_in_main_auto"].astype(str).str.lower().eq("true")].copy()
    main_sample["lag_months_num"] = pd.to_numeric(main_sample["lag_months_num"], errors="coerce")
    main_sample = main_sample.dropna(subset=["lag_months_num"])

    rows: list[dict[str, float | int | str]] = []
    for index, pathway in enumerate(PATHWAYS):
        subset = main_sample[main_sample["chain_type_norm"].eq(pathway)]
        values = subset["lag_months_num"].to_numpy(dtype=float)
        groups = [
            group["lag_months_num"].to_numpy(dtype=float)
            for _, group in subset.groupby("upstream_event_id", sort=False)
        ]
        intervals = clustered_intervals(groups, SEED + index)
        for metric, estimate in metric_values(values).items():
            low, high = intervals[metric]
            rows.append(
                {
                    "pathway": pathway,
                    "n_chains": len(values),
                    "n_upstream_event_clusters": len(groups),
                    "metric": metric,
                    "estimate": round(estimate, 2),
                    "clustered_ci_95_low": round(float(low), 2),
                    "clustered_ci_95_high": round(float(high), 2),
                    "bootstrap_repetitions": N_BOOT,
                    "cluster_variable": "upstream_event_id",
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
