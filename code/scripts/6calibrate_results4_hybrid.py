from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_DIR = PROJECT_ROOT / "outputs" / "results4_llm_historical_replay"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "results4_llm_historical_replay_v3_hybrid"

FOCAL_CHAINS = [
    "policy_to_firm",
    "firm_to_school",
    "policy_to_school",
    "standard_to_curriculum",
]


def constraint_penalty(text: str) -> float:
    t = str(text).lower()
    score = 0.0
    if "approval" in t or "curriculum" in t or "programme" in t:
        score += 0.22
    if "faculty" in t or "teacher" in t or "capacity" in t:
        score += 0.18
    if "standard" in t or "consensus" in t or "certification" in t:
        score += 0.22
    if "coordination" in t or "cooperation" in t or "cross" in t:
        score += 0.16
    if "investment" in t or "cycle" in t:
        score += 0.12
    if "uncertain" in t or "interpretation" in t:
        score += 0.10
    if "economic" in t or "market" in t:
        score -= 0.05
    return score


def decision_penalty(decision: str) -> float:
    d = str(decision).lower()
    if d == "respond_now":
        return -0.25
    if d == "request_coordination":
        return 0.10
    if d == "delay":
        return 0.22
    if d == "ignore":
        return 0.45
    return 0.0


def resolve_base_delay(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index)


def hybrid_propensity(df: pd.DataFrame) -> pd.Series:
    urgency = pd.to_numeric(df.get("perceived_urgency"), errors="coerce").fillna(0.5)
    feasibility = pd.to_numeric(df.get("perceived_feasibility"), errors="coerce").fillna(0.5)
    llm_delay = pd.to_numeric(df.get("sim_delay_months"), errors="coerce").fillna(df["lag_months_num"].median())
    llm_delay_norm = llm_delay.rank(pct=True)
    penalties = df.get("dominant_constraint", "").map(constraint_penalty) + df.get("decision", "").map(decision_penalty)
    score = (1 - urgency) * 0.28 + (1 - feasibility) * 0.34 + llm_delay_norm * 0.25 + penalties * 0.55
    return score


def assign_empirical_quantiles(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = group.copy()
    observed = np.sort(out["lag_months_num"].to_numpy(dtype=float))
    n = len(out)
    if n == 0:
        return out
    score = hybrid_propensity(out).to_numpy(dtype=float)
    score = score + rng.normal(0, 1e-6, size=n)
    rank_order = np.argsort(score)
    if len(observed) == n:
        target = observed.copy()
    else:
        q = (np.arange(n) + 0.5) / n
        target = np.quantile(observed, q, method="linear")
    calibrated = np.empty(n, dtype=float)
    calibrated[rank_order] = target
    out["hybrid_delay_propensity"] = score
    out["hybrid_sim_delay_months"] = calibrated
    return out


def assign_empirical_only(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = group.copy()
    observed = np.sort(out["lag_months_num"].to_numpy(dtype=float))
    n = len(out)
    if n == 0:
        return out
    order = np.argsort(rng.normal(0, 1e-6, size=n))
    if len(observed) == n:
        target = observed.copy()
    else:
        q = (np.arange(n) + 0.5) / n
        target = np.quantile(observed, q, method="linear")
    calibrated = np.empty(n, dtype=float)
    calibrated[order] = target
    out["empirical_only_sim_delay_months"] = calibrated
    return out


def summarize(df: pd.DataFrame, sim_col: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    def spearman_no_scipy(a: np.ndarray, b: np.ndarray) -> float:
        ar = pd.Series(a).rank(method="average").to_numpy(dtype=float)
        br = pd.Series(b).rank(method="average").to_numpy(dtype=float)
        if np.std(ar) == 0 or np.std(br) == 0:
            return float("nan")
        return float(np.corrcoef(ar, br)[0, 1])

    rows = []
    sim = pd.to_numeric(df[sim_col], errors="coerce").to_numpy(dtype=float)
    obs = df["lag_months_num"].to_numpy(dtype=float)
    for chain_type in FOCAL_CHAINS:
        sub = df[df["chain_type_norm"] == chain_type]
        if sub.empty:
            continue
        sub_obs = sub["lag_months_num"].to_numpy(dtype=float)
        sub_sim = pd.to_numeric(sub[sim_col], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "chain_type": chain_type,
                "n": len(sub),
                f"observed_mean": round(float(np.mean(sub_obs)), 2),
                f"{label}_mean": round(float(np.mean(sub_sim)), 2),
                f"observed_median": round(float(np.median(sub_obs)), 2),
                f"{label}_median": round(float(np.median(sub_sim)), 2),
                f"observed_lt12_pct": round(float(np.mean(sub_obs < 12) * 100), 1),
                f"{label}_lt12_pct": round(float(np.mean(sub_sim < 12) * 100), 1),
                f"observed_lt24_pct": round(float(np.mean(sub_obs < 24) * 100), 1),
                f"{label}_lt24_pct": round(float(np.mean(sub_sim < 24) * 100), 1),
                "mae_months": round(float(np.mean(np.abs(sub_obs - sub_sim))), 2),
                "spearman_rank": round(spearman_no_scipy(sub_obs, sub_sim), 3),
            }
        )
    by_pathway = pd.DataFrame(rows)
    aggregate = pd.DataFrame(
        [
            {
                "n": len(df),
                f"observed_mean": round(float(np.mean(obs)), 2),
                f"{label}_mean": round(float(np.mean(sim)), 2),
                f"observed_median": round(float(np.median(obs)), 2),
                f"{label}_median": round(float(np.median(sim)), 2),
                f"observed_lt12_pct": round(float(np.mean(obs < 12) * 100), 1),
                f"{label}_lt12_pct": round(float(np.mean(sim < 12) * 100), 1),
                f"observed_lt24_pct": round(float(np.mean(obs < 24) * 100), 1),
                f"{label}_lt24_pct": round(float(np.mean(sim < 24) * 100), 1),
                "spearman_rank": round(spearman_no_scipy(obs, sim), 3),
            }
        ]
    )
    return by_pathway, aggregate


def save_outputs(df: pd.DataFrame, out_dir: Path, prefix: str, sim_col: str, label: str, seed: int) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    by_pathway, aggregate = summarize(df, sim_col=sim_col, label=label)
    df.to_csv(out_dir / f"{prefix}.csv", index=False, encoding="utf-8-sig")
    by_pathway.to_csv(table_dir / f"{prefix}_by_pathway.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(table_dir / f"{prefix}_aggregate.csv", index=False, encoding="utf-8-sig")
    by_pathway.to_markdown(table_dir / f"{prefix}_by_pathway.md", index=False)
    aggregate.to_markdown(table_dir / f"{prefix}_aggregate.md", index=False)
    status = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "prefix": prefix,
        "seed": seed,
        "n": int(len(df)),
        "sim_col": sim_col,
        "label": label,
    }
    (out_dir / f"{prefix}_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default=str(DEFAULT_IN_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=20260526)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(in_dir / "llm_replay_results.csv")
    source = source[source["chain_type_norm"].isin(FOCAL_CHAINS)].copy()
    source["lag_months_num"] = pd.to_numeric(source["lag_months_num"], errors="coerce")
    source = source[source["lag_months_num"].notna()].copy()

    rng = np.random.default_rng(args.seed)
    calibrated_list = []
    empirical_only_list = []
    for name, group in source.groupby("chain_type_norm"):
        g_calib = assign_empirical_quantiles(group, rng)
        g_calib["chain_type_norm"] = name
        calibrated_list.append(g_calib)
        g_emp = assign_empirical_only(group, rng)
        g_emp["chain_type_norm"] = name
        empirical_only_list.append(g_emp)

    hybrid = pd.concat(calibrated_list, ignore_index=True)
    empirical_only = pd.concat(empirical_only_list, ignore_index=True)
    raw = source.copy()
    raw["raw_llm_delay_months"] = pd.to_numeric(raw.get("sim_delay_months"), errors="coerce")
    raw["raw_llm_delay_months"] = raw["raw_llm_delay_months"].fillna(raw["lag_months_num"])

    hybrid["replay_mode"] = "hybrid_calibration"
    empirical_only["replay_mode"] = "empirical_only_resampling"
    raw["replay_mode"] = "raw_llm_delay"

    hybrid["calibrated_delay_months"] = hybrid["hybrid_sim_delay_months"]
    empirical_only["calibrated_delay_months"] = empirical_only["empirical_only_sim_delay_months"]
    raw["calibrated_delay_months"] = raw["raw_llm_delay_months"]

    save_outputs(raw, out_dir, "llm_replay_results_raw", "raw_llm_delay_months", "raw", args.seed)
    save_outputs(empirical_only, out_dir, "llm_replay_results_empirical_only", "empirical_only_sim_delay_months", "empirical_only", args.seed)
    save_outputs(hybrid, out_dir, "llm_replay_results_hybrid_calibrated", "hybrid_sim_delay_months", "hybrid", args.seed)

    compare = pd.DataFrame(
        [
            {
                "mode": "raw_llm_delay",
                "n": int(len(raw)),
                "mean_delay": round(float(raw["raw_llm_delay_months"].mean()), 2),
                "median_delay": round(float(raw["raw_llm_delay_months"].median()), 2),
                "lt12_pct": round(float((raw["raw_llm_delay_months"] < 12).mean() * 100), 1),
                "lt24_pct": round(float((raw["raw_llm_delay_months"] < 24).mean() * 100), 1),
            },
            {
                "mode": "empirical_only_resampling",
                "n": int(len(empirical_only)),
                "mean_delay": round(float(empirical_only["empirical_only_sim_delay_months"].mean()), 2),
                "median_delay": round(float(empirical_only["empirical_only_sim_delay_months"].median()), 2),
                "lt12_pct": round(float((empirical_only["empirical_only_sim_delay_months"] < 12).mean() * 100), 1),
                "lt24_pct": round(float((empirical_only["empirical_only_sim_delay_months"] < 24).mean() * 100), 1),
            },
            {
                "mode": "hybrid_calibration",
                "n": int(len(hybrid)),
                "mean_delay": round(float(hybrid["hybrid_sim_delay_months"].mean()), 2),
                "median_delay": round(float(hybrid["hybrid_sim_delay_months"].median()), 2),
                "lt12_pct": round(float((hybrid["hybrid_sim_delay_months"] < 12).mean() * 100), 1),
                "lt24_pct": round(float((hybrid["hybrid_sim_delay_months"] < 24).mean() * 100), 1),
            },
        ]
    )
    compare.to_csv(table_dir / "calibration_mode_comparison.csv", index=False, encoding="utf-8-sig")
    compare.to_markdown(table_dir / "calibration_mode_comparison.md", index=False)

    status = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source_dir": str(in_dir),
        "output_dir": str(out_dir),
        "n_raw": int(len(raw)),
        "n_empirical_only": int(len(empirical_only)),
        "n_hybrid": int(len(hybrid)),
        "method": "raw LLM delay vs empirical-only resampling vs hybrid calibration",
    }
    (out_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
