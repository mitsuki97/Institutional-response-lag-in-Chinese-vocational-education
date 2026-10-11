from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from case_behavior_granularity import MAP_DF, SCENARIO_BEHAVIOR_MAP, add_behavior_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_APPENDIX_DIR = PROJECT_ROOT / "outputs" / "results4_case_appendix"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "results4_case_counterfactuals"

SCENARIO_ORDER = [
    "baseline",
    "firm_school_ties",
    "translation_capacity",
    "platform_governance",
    "combined_reform",
]

SCENARIO_LABELS = {
    "baseline": "Observed\nbaseline",
    "firm_school_ties": "Firm-school\nties",
    "translation_capacity": "Translation\ncapacity",
    "platform_governance": "Platform\ncoordination",
    "combined_reform": "Combined\nreform",
}

SCENARIO_COLORS = {
    "baseline": "#9D9D9D",
    "firm_school_ties": "#72B7B2",
    "translation_capacity": "#F58518",
    "platform_governance": "#B279A2",
    "combined_reform": "#54A24B",
}

SCENARIO_SHORT_LABELS = {
    "firm_school_ties": "校企\n连接",
    "translation_capacity": "跨场域\n翻译",
    "platform_governance": "政府\n统筹",
    "combined_reform": "组合式\n改革",
}

FIGURE_COLORS = {
    "replay": "#2F3B52",
    "observed": "#8A8A8A",
}

SCENARIO_PARAMS = {
    "baseline": {
        "policy_coordination": 1.0,
        "firm_school_connectivity": 1.0,
        "intermediary_translation": 1.0,
        "school_adaptability": 1.0,
        "standard_flexibility": 1.0,
    },
    "firm_school_ties": {
        "policy_coordination": 1.0,
        "firm_school_connectivity": 1.35,
        "intermediary_translation": 1.0,
        "school_adaptability": 1.10,
        "standard_flexibility": 1.0,
    },
    "translation_capacity": {
        "policy_coordination": 1.0,
        "firm_school_connectivity": 1.0,
        "intermediary_translation": 1.45,
        "school_adaptability": 1.0,
        "standard_flexibility": 1.15,
    },
    "platform_governance": {
        "policy_coordination": 1.25,
        "firm_school_connectivity": 1.0,
        "intermediary_translation": 1.20,
        "school_adaptability": 1.0,
        "standard_flexibility": 1.0,
    },
    "combined_reform": {
        "policy_coordination": 1.30,
        "firm_school_connectivity": 1.45,
        "intermediary_translation": 1.55,
        "school_adaptability": 1.25,
        "standard_flexibility": 1.25,
    },
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def load_cases() -> pd.DataFrame:
    table_path = CASE_APPENDIX_DIR / "tables" / "case_catalog.csv"
    jsonl_path = CASE_APPENDIX_DIR / "cases.jsonl"
    if table_path.exists():
        df = pd.read_csv(table_path)
    elif jsonl_path.exists():
        df = pd.read_json(jsonl_path, lines=True)
    else:
        raise FileNotFoundError(
            "Missing case appendix outputs. Run 10make_results4_case_appendix.py first."
        )

    numeric_cols = ["observed_lag_months", "simulated_lag_months", "confidence"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["case_id", "sector", "sector_label", "chain_type_norm", "pathway_label", "up_title", "down_title", "dominant_constraint"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "sector_label" not in df.columns and "sector" in df.columns:
        df["sector_label"] = df["sector"]
    if "pathway_label" not in df.columns and "chain_type_norm" in df.columns:
        df["pathway_label"] = df["chain_type_norm"]
    split_path = CASE_APPENDIX_DIR / "tables" / "case_split.csv"
    if "role" in df.columns:
        df["role"] = df["role"].fillna("supplement").astype(str)
    elif split_path.exists():
        split = pd.read_csv(split_path)
        df = df.merge(split[["case_id", "role"]], on="case_id", how="left")
    else:
        rep = choose_representative_cases(df)
        rep = rep[["case_id", "role"]].copy()
        df = df.merge(rep, on="case_id", how="left")
    if "role" not in df.columns:
        df["role"] = "supplement"
    df["role"] = df["role"].fillna("supplement").astype(str)
    return add_behavior_columns(df)


def choose_representative_cases(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sector in SECTOR_ORDER:
        sub = cases[cases["sector"] == sector].copy()
        if sub.empty:
            continue
        target = float(sub["observed_lag_months"].median())
        sub["median_gap"] = (sub["observed_lag_months"] - target).abs()
        sub["confidence_rank"] = pd.to_numeric(sub.get("confidence"), errors="coerce").fillna(-1.0)
        sub = sub.sort_values(["median_gap", "confidence_rank", "case_id"], ascending=[True, False, True], kind="mergesort")
        rows.append(sub.head(1))
    if not rows:
        raise RuntimeError("Unable to choose representative cases from the provided case set.")
    rep = pd.concat(rows, ignore_index=True)
    rep["role"] = "main"
    return rep[["case_id", "role"]]


def scenario_acceleration(chain_type: str, scenario_name: str) -> float:
    params = SCENARIO_PARAMS[scenario_name]
    if chain_type == "policy_to_firm":
        return params["policy_coordination"] ** 0.85
    if chain_type == "firm_to_school":
        return (params["firm_school_connectivity"] ** 0.95) * (params["school_adaptability"] ** 0.45)
    if chain_type == "policy_to_school":
        return (
            (params["policy_coordination"] ** 0.55)
            * (params["school_adaptability"] ** 0.85)
            * (params["intermediary_translation"] ** 0.25)
        )
    if chain_type == "standard_to_curriculum":
        return (
            (params["standard_flexibility"] ** 0.90)
            * (params["intermediary_translation"] ** 0.65)
            * (params["school_adaptability"] ** 0.35)
        )
    return 1.0


def build_counterfactuals(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in cases.iterrows():
        observed = float(row.get("observed_lag_months", np.nan))
        replay = float(row.get("simulated_lag_months", np.nan))
        chain_type = str(row.get("chain_type_norm", "")).strip()
        for scenario in SCENARIO_ORDER:
            accel = scenario_acceleration(chain_type, scenario)
            cf_lag = np.nan if not np.isfinite(observed) else max(0.0, observed / accel)
            rows.append(
                {
                    "case_id": row.get("case_id"),
                    "sector": row.get("sector"),
                    "sector_label": row.get("sector_label"),
                    "chain_type_norm": chain_type,
                    "pathway_label": row.get("pathway_label"),
                    "role": row.get("role", "supplement"),
                    "dominant_constraint": row.get("dominant_constraint"),
                    "macro_dimension": row.get("macro_dimension"),
                    "macro_label": row.get("macro_label"),
                    "field": row.get("field"),
                    "behavior_dimension": row.get("behavior_dimension"),
                    "behavior_indicator": row.get("behavior_indicator"),
                    "case_fine_behavior": row.get("case_fine_behavior"),
                    "capital_feature": row.get("capital_feature"),
                    "habitus_feature": row.get("habitus_feature"),
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "scenario_macro_label": SCENARIO_BEHAVIOR_MAP[scenario]["scenario_macro_label"],
                    "scenario_fine_behavior": SCENARIO_BEHAVIOR_MAP[scenario]["scenario_fine_behavior"],
                    "observed_lag_months": observed,
                    "replay_lag_months": replay,
                    "counterfactual_lag_months": round(cf_lag, 2) if np.isfinite(cf_lag) else np.nan,
                    "reduction_months": round(observed - cf_lag, 2) if np.isfinite(cf_lag) and np.isfinite(observed) else np.nan,
                    "reduction_pct": round((observed - cf_lag) / observed * 100, 1) if np.isfinite(cf_lag) and np.isfinite(observed) and observed else np.nan,
                    "acceleration_factor": round(float(accel), 4),
                }
            )
    return pd.DataFrame(rows)


def save_plot(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight", dpi=500)
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_tables(cases: pd.DataFrame, cf: pd.DataFrame, out_dir: Path) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    long_df = cf.copy()
    long_df["replay_minus_observed"] = long_df["replay_lag_months"] - long_df["observed_lag_months"]
    long_df["cf_minus_observed"] = long_df["counterfactual_lag_months"] - long_df["observed_lag_months"]
    long_df.to_csv(table_dir / "case_counterfactuals_long.csv", index=False, encoding="utf-8-sig")
    long_df.to_markdown(table_dir / "case_counterfactuals_long.md", index=False)
    MAP_DF.to_csv(table_dir / "macro_to_micro_behavior_map.csv", index=False, encoding="utf-8-sig")
    MAP_DF.to_markdown(table_dir / "macro_to_micro_behavior_map.md", index=False)
    behavior_cols = [
        "case_id", "sector_label", "pathway_label", "dominant_constraint",
        "macro_dimension", "macro_label", "field", "behavior_dimension",
        "behavior_indicator", "case_fine_behavior", "capital_feature", "habitus_feature",
    ]
    cases[[c for c in behavior_cols if c in cases.columns]].drop_duplicates().to_csv(
        table_dir / "case_behavior_granularity.csv", index=False, encoding="utf-8-sig"
    )
    long_df[long_df["role"] == "main"].to_csv(table_dir / "case_counterfactuals_main.csv", index=False, encoding="utf-8-sig")
    long_df[long_df["role"] == "supplement"].to_csv(table_dir / "case_counterfactuals_supplement.csv", index=False, encoding="utf-8-sig")

    wide = long_df.pivot(
        index=["case_id", "sector_label", "pathway_label", "observed_lag_months", "replay_lag_months"],
        columns="scenario",
        values="counterfactual_lag_months",
    ).reset_index()
    wide.columns.name = None
    wide.to_csv(table_dir / "case_counterfactuals_wide.csv", index=False, encoding="utf-8-sig")
    wide.to_markdown(table_dir / "case_counterfactuals_wide.md", index=False)

    summary = (
        long_df.groupby(["sector_label", "scenario"], as_index=False)
        .agg(
            n_cases=("case_id", "nunique"),
            observed_mean=("observed_lag_months", "mean"),
            replay_mean=("replay_lag_months", "mean"),
            cf_mean=("counterfactual_lag_months", "mean"),
            reduction_months=("reduction_months", "mean"),
            reduction_pct=("reduction_pct", "mean"),
        )
        .sort_values(["sector_label", "scenario"])
    )
    for col in summary.select_dtypes(include=[np.number]).columns:
        summary[col] = summary[col].round(2)
    summary.to_csv(table_dir / "case_counterfactual_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_markdown(table_dir / "case_counterfactual_summary.md", index=False)

    case_summary = long_df[long_df["scenario"] == "combined_reform"][
        [
            "case_id",
            "sector_label",
            "pathway_label",
            "observed_lag_months",
            "replay_lag_months",
            "counterfactual_lag_months",
            "reduction_pct",
        ]
    ].rename(columns={"counterfactual_lag_months": "combined_reform_lag_months"})
    case_summary.to_csv(table_dir / "case_counterfactual_cases.csv", index=False, encoding="utf-8-sig")
    case_summary.to_markdown(table_dir / "case_counterfactual_cases.md", index=False)


def ordered_cases(cases: pd.DataFrame) -> pd.DataFrame:
    sector_rank = {sector: i for i, sector in enumerate(["AI", "NEV", "eldercare", "robotics"])}
    ordered = cases.copy()
    ordered["sector_rank"] = ordered["sector"].map(sector_rank).fillna(len(sector_rank))
    ordered = ordered.sort_values(["sector_rank", "chain_type_norm", "observed_lag_months", "case_id"], kind="mergesort")
    return ordered.drop(columns=["sector_rank"])


def case_counterfactual_values(case: pd.Series, cf_long: pd.DataFrame) -> tuple[list[str], list[float], list[str]]:
    sub = cf_long[cf_long["case_id"] == case["case_id"]].set_index("scenario").reindex(SCENARIO_ORDER)
    labels = ["Observed", "Replay"] + [SCENARIO_SHORT_LABELS[s] for s in SCENARIO_ORDER[1:]]
    values = [
        float(case["observed_lag_months"]),
        float(case["simulated_lag_months"]),
    ] + [float(sub.loc[s, "counterfactual_lag_months"]) for s in SCENARIO_ORDER[1:]]
    colors = [FIGURE_COLORS["observed"], FIGURE_COLORS["replay"]] + [SCENARIO_COLORS[s] for s in SCENARIO_ORDER[1:]]
    return labels, values, colors


def causal_chain_image_path(case: pd.Series) -> Path:
    sector = str(case.get("sector", "")).strip()
    pathway = str(case.get("pathway_label", "")).strip()
    return CASE_APPENDIX_DIR / "figures" / "causal_chains" / f"{case['case_id']}_{sector}_{pathway}_causal.png"


def plot_case_figure(case: pd.Series, cf_long: pd.DataFrame, out_dir: Path) -> None:
    sub = cf_long[cf_long["case_id"] == case["case_id"]].set_index("scenario").reindex(SCENARIO_ORDER)
    labels = ["Observed\nbaseline", "LLM\nreplay"] + [SCENARIO_SHORT_LABELS[s] for s in SCENARIO_ORDER[1:]]
    values = [
        float(case["observed_lag_months"]),
        float(case["simulated_lag_months"]),
    ] + [float(sub.loc[s, "counterfactual_lag_months"]) for s in SCENARIO_ORDER[1:]]
    colors = [FIGURE_COLORS["observed"], FIGURE_COLORS["replay"]] + [SCENARIO_COLORS[s] for s in SCENARIO_ORDER[1:]]

    fig, ax = plt.subplots(figsize=(7.4, 3.4), constrained_layout=True)
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.35, width=0.78)

    for idx, (bar, val) in enumerate(zip(bars, values)):
        if np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.03,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        if idx >= 2:
            obs = float(case["observed_lag_months"])
            if np.isfinite(obs) and obs:
                change_pct = (obs - val) / obs * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(bar.get_height() + max(values) * 0.11, max(values) * 1.04),
                    f"-{change_pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    color="#444444",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Lag (months)")
    ax.set_ylim(0, max(values) * 1.28 if max(values) > 0 else 1)
    ax.set_title(
        f"{case['case_id']} | {case['sector_label']} | {case['pathway_label']}",
        loc="left",
        fontsize=10.2,
        fontweight="bold",
    )
    subtitle = (
        f"{textwrap.shorten(str(case.get('up_title', '')), width=38, placeholder='...')} -> "
        f"{textwrap.shorten(str(case.get('down_title', '')), width=38, placeholder='...')} | "
        f"{case.get('macro_label', 'macro')} -> {case.get('case_fine_behavior', 'fine behaviour')}"
    )
    ax.text(0.0, 1.03, subtitle, transform=ax.transAxes, fontsize=7.3, color="#444444")
    ax.legend(
        handles=[
            mpl.patches.Patch(facecolor=FIGURE_COLORS["observed"], edgecolor="#333333", label="Observed baseline"),
            mpl.patches.Patch(facecolor=FIGURE_COLORS["replay"], edgecolor="#333333", label="LLM replay"),
            mpl.patches.Patch(facecolor=SCENARIO_COLORS["firm_school_ties"], edgecolor="#333333", label="Counterfactual"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper right",
    )
    ax.text(
        0.0,
        -0.26,
        f"Counterfactual bars are projected from the observed lag using the same pathway-specific governance multipliers as the simulator.",
        transform=ax.transAxes,
        fontsize=6.8,
        color="#555555",
    )
    save_plot(fig, out_dir / "figures" / "cases" / f"{case['case_id']}_counterfactual")


def plot_main_text_composite(main_cases: pd.DataFrame, cf_long: pd.DataFrame, out_dir: Path) -> None:
    main_cases = ordered_cases(main_cases)
    if main_cases.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
    axes = axes.ravel()
    panel_labels = ["a", "b", "c", "d"]
    for ax, panel, (_, case) in zip(axes, panel_labels, main_cases.iterrows()):
        ax.axis("off")
        img_path = causal_chain_image_path(case)
        if img_path.exists():
            img = plt.imread(img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Missing image:\n{img_path.name}", ha="center", va="center", fontsize=8)
        ax.text(0.02, 0.98, panel, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        ax.text(
            0.02,
            0.04,
            f"{case['sector_label']} | {case['case_id']} | {case['pathway_label']}",
            transform=ax.transAxes,
            fontsize=7.2,
            fontweight="bold",
            color="#222222",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#E0E0E0", "linewidth": 0.3, "pad": 2.5},
        )

    for ax in axes[len(main_cases) :]:
        ax.axis("off")
    fig.suptitle("Representative industry cases show pathway-specific causal chains", x=0.01, y=0.995, ha="left", fontsize=12, fontweight="bold")
    save_plot(fig, out_dir / "figures" / "main_text" / "main_text_representative_case_composite")


def plot_supplement_case_figure(case: pd.Series, cf_long: pd.DataFrame, out_dir: Path) -> None:
    labels, values, colors = case_counterfactual_values(case, cf_long)
    fig = plt.figure(figsize=(10.8, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
    ax_flow = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    sector = str(case.get("sector_label", ""))
    color = {
        "AI": "#4C78A8",
        "NEV": "#54A24B",
        "eldercare": "#B279A2",
        "robotics": "#F58518",
    }.get(str(case.get("sector", "")), "#4C78A8")

    ax_flow.text(0.02, 0.96, f"{case['case_id']} | {sector} | {case['pathway_label']}", fontsize=10.2, fontweight="bold", va="top")
    boxes = [
        ("Intervention/input", textwrap.fill(str(case.get("up_title", "")), width=42), 0.78, "#EEF5FD"),
        ("Macro-to-micro mechanism", textwrap.fill(
            f"{case.get('macro_label', 'mechanism')} | {case.get('field', 'field')}\n"
            f"{case.get('case_fine_behavior', case.get('dominant_constraint', 'mechanism not stated'))}", width=42), 0.54, "#FBF4E8"),
        ("Behavioural response", textwrap.fill(str(case.get("down_title", "")), width=42), 0.30, "#F3ECFA"),
        ("Observed result", f"Observed lag {float(case['observed_lag_months']):.0f} months; replay {float(case['simulated_lag_months']):.0f} months.", 0.08, "#F7F7F7"),
    ]
    for title, body, y, face in boxes:
        rect = mpl.patches.FancyBboxPatch((0.03, y), 0.90, 0.14, boxstyle="round,pad=0.012,rounding_size=0.012", facecolor=face, edgecolor=color, linewidth=0.7)
        ax_flow.add_patch(rect)
        ax_flow.text(0.06, y + 0.105, title, fontsize=7.6, fontweight="bold", va="top")
        ax_flow.text(0.06, y + 0.072, body, fontsize=6.8, va="top", color="#333333")
    for y1, y2 in [(0.78, 0.68), (0.54, 0.44), (0.30, 0.22)]:
        ax_flow.annotate("", xy=(0.48, y2), xytext=(0.48, y1), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#555555"})

    x = np.arange(len(values))
    ax_bar.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.35, width=0.74)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["Obs.", "Replay"] + [SCENARIO_SHORT_LABELS[s] for s in SCENARIO_ORDER[1:]], fontsize=6.8)
    ax_bar.set_ylabel("Lag (months)")
    ax_bar.set_ylim(0, max(values) * 1.30 if max(values) > 0 else 1)
    ax_bar.set_title("Observed, replay and counterfactual lags", loc="left", fontsize=9.5, fontweight="bold")
    for i, val in enumerate(values):
        ax_bar.text(i, val + max(values) * 0.035, f"{val:.0f}", ha="center", va="bottom", fontsize=6.5)
    observed = values[0]
    combined = values[-1]
    reduction = (observed - combined) / observed * 100 if observed else 0.0
    ax_bar.text(
        0.02,
        0.96,
        f"Causal readout: {case.get('macro_label', 'intervention')} is operationalized as {case.get('case_fine_behavior', 'fine-grained behaviour')}; combined reform shifts projected lag by {reduction:.1f}%.",
        transform=ax_bar.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="#333333",
    )
    save_plot(fig, out_dir / "figures" / "supplementary_cases" / f"{case['case_id']}_supplement_case")


def plot_overview(cases: pd.DataFrame, cf_long: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(12.4, 13.6), constrained_layout=True)
    axes = axes.ravel()
    for ax, (_, case) in zip(axes, cases.iterrows()):
        sub = cf_long[cf_long["case_id"] == case["case_id"]].set_index("scenario").reindex(SCENARIO_ORDER)
        labels = ["Obs", "Replay", "F-S", "Trans", "Plat", "Comb"]
        values = [
            float(case["observed_lag_months"]),
            float(case["simulated_lag_months"]),
        ] + [float(sub.loc[s, "counterfactual_lag_months"]) for s in SCENARIO_ORDER[1:]]
        colors = [FIGURE_COLORS["observed"], FIGURE_COLORS["replay"]] + [SCENARIO_COLORS[s] for s in SCENARIO_ORDER[1:]]
        x = np.arange(len(values))
        ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.25, width=0.72)
        ax.set_title(f"{case['case_id']} | {case['sector_label']}", loc="left", fontsize=8.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.4)
        ax.tick_params(axis="y", labelsize=6.4)
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
        if case["case_id"] in set(cases.groupby("sector", sort=False).head(1)["case_id"].astype(str)):
            ax.set_ylabel("Months", fontsize=7)
        if np.isfinite(float(case["observed_lag_months"])) and float(case["observed_lag_months"]) > 0:
            ax.text(0, float(case["observed_lag_months"]) * 1.03, f"{float(case['observed_lag_months']):.0f}", ha="center", fontsize=6)
        if np.isfinite(float(case["simulated_lag_months"])) and float(case["simulated_lag_months"]) > 0:
            ax.text(1, float(case["simulated_lag_months"]) * 1.03, f"{float(case['simulated_lag_months']):.0f}", ha="center", fontsize=6)

    for ax in axes[len(cases) :]:
        ax.axis("off")

    handles = [
        mpl.patches.Patch(facecolor=FIGURE_COLORS["observed"], edgecolor="#333333", label="Observed"),
        mpl.patches.Patch(facecolor=FIGURE_COLORS["replay"], edgecolor="#333333", label="LLM replay"),
        mpl.patches.Patch(facecolor=SCENARIO_COLORS["firm_school_ties"], edgecolor="#333333", label="Counterfactuals"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="upper center", ncol=3)
    fig.suptitle("Case-level counterfactual replay for 12 appendix cases", x=0.01, y=0.995, ha="left", fontsize=12, fontweight="bold")
    save_plot(fig, out_dir / "figures" / "case_counterfactual_overview")


def write_case_notes(cases: pd.DataFrame, cf_long: pd.DataFrame, out_dir: Path) -> None:
    notes_dir = out_dir / "cases"
    notes_dir.mkdir(parents=True, exist_ok=True)
    for _, case in cases.iterrows():
        sub = cf_long[cf_long["case_id"] == case["case_id"]].set_index("scenario").reindex(SCENARIO_ORDER)
        lines = [
            f"# {case['case_id']}",
            f"- Sector: {case['sector_label']}",
            f"- Pathway: {case['pathway_label']}",
            f"- Observed lag: {float(case['observed_lag_months']):.1f} months",
            f"- LLM replay lag: {float(case['simulated_lag_months']):.1f} months",
            "",
            "| Scenario | Counterfactual lag (months) | Reduction vs observed |",
            "|---|---:|---:|",
        ]
        for scenario in SCENARIO_ORDER:
            lag = float(sub.loc[scenario, "counterfactual_lag_months"])
            red = float(sub.loc[scenario, "reduction_pct"])
            lines.append(f"| {SCENARIO_LABELS[scenario].replace(chr(10), ' ')} | {lag:.2f} | {red:.1f}% |")
        lines.append("")
        lines.append("Note: counterfactual lags are computed from the observed lag using the same pathway-specific multipliers as the simulator.")
        (notes_dir / f"{case['case_id']}.md").write_text("\n".join(lines), encoding="utf-8")


def make_s1_case_validation_bar(cases: pd.DataFrame, cf_long: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = cases.sort_values(["sector", "chain_type_norm", "observed_lag_months", "case_id"]).copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=False)
    axes = axes.ravel()
    sector_order = ["AI", "NEV", "Eldercare", "Robotics"]
    for ax, sector in zip(axes, [s for s in sector_order if s in order["sector_label"].unique().tolist()]):
        sub = order[order["sector_label"] == sector].copy()
        if sub.empty:
            ax.axis("off")
            continue
        x = np.arange(len(sub))
        width = 0.26
        ax.bar(x - width, sub["observed_lag_months"], width=width, color=FIGURE_COLORS["observed"], edgecolor="#333333", linewidth=0.3, label="Observed")
        ax.bar(x, sub["simulated_lag_months"], width=width, color=FIGURE_COLORS["replay"], edgecolor="#333333", linewidth=0.3, label="LLM replay")
        cf = cf_long.copy()
        cf = cf[(cf["scenario"] == "combined_reform") & (cf["case_id"].isin(sub["case_id"]))]
        combined_vals = sub["case_id"].map(cf.set_index("case_id")["counterfactual_lag_months"])
        ax.bar(x + width, combined_vals, width=width, color=SCENARIO_COLORS["combined_reform"], edgecolor="#333333", linewidth=0.3, label="Combined reform")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["case_id"], fontsize=6.8)
        ax.set_title(sector, loc="left", fontsize=10.0, fontweight="bold")
        ax.set_ylabel("Lag (months)")
        ymax = np.nanmax(np.vstack([sub["observed_lag_months"], sub["simulated_lag_months"], combined_vals.to_numpy(dtype=float)]))
        ax.set_ylim(0, max(1, float(ymax) * 1.24))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], ["Observed", "LLM replay", "Combined reform"], frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"S1 Supplementary cases: observed, replay and reform-lag comparison (n={len(cases)})", x=0.01, y=0.995, ha="left", fontsize=11.0, fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.26, wspace=0.18)
    save_plot(fig, fig_dir / "S1_case_level_bar_validation")


def make_s2_observed_replay_scatter(cases: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    for sector, color in zip(["AI", "NEV", "eldercare", "robotics"], ["#4C78A8", "#54A24B", "#B279A2", "#F58518"]):
        sub = cases[cases["sector"] == sector]
        if sub.empty:
            continue
        ax.scatter(
            sub["observed_lag_months"],
            sub["simulated_lag_months"],
            s=62,
            color=color,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
            label=sub["sector_label"].iloc[0],
        )
        for _, row in sub.iterrows():
            ax.text(row["observed_lag_months"] + 0.35, row["simulated_lag_months"] + 0.35, row["case_id"], fontsize=6.5)
    lim = float(np.nanmax(cases[["observed_lag_months", "simulated_lag_months"]].to_numpy()))
    lim = max(1, lim + 4)
    ax.plot([0, lim], [0, lim], ls="--", lw=1.0, color="#666666")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed lag (months)")
    ax.set_ylabel("LLM replay lag (months)")
    ax.set_title(f"S2 Observed vs. replay lag across supplementary cases (n={len(cases)})", loc="left", fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save_plot(fig, fig_dir / "S2_observed_vs_replay_scatter")


def make_s3_observed_reform_scatter(cases: pd.DataFrame, cf_long: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    combined = cf_long[cf_long["scenario"] == "combined_reform"][["case_id", "counterfactual_lag_months"]].rename(columns={"counterfactual_lag_months": "combined_reform_lag_months"})
    merged = cases.merge(combined, on="case_id", how="left")
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    for sector, color in zip(["AI", "NEV", "eldercare", "robotics"], ["#4C78A8", "#54A24B", "#B279A2", "#F58518"]):
        sub = merged[merged["sector"] == sector]
        if sub.empty:
            continue
        ax.scatter(
            sub["observed_lag_months"],
            sub["combined_reform_lag_months"],
            s=62,
            color=color,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
            label=sub["sector_label"].iloc[0],
        )
        for _, row in sub.iterrows():
            ax.text(row["observed_lag_months"] + 0.35, row["combined_reform_lag_months"] + 0.35, row["case_id"], fontsize=6.5)
    lim = float(np.nanmax(merged[["observed_lag_months", "combined_reform_lag_months"]].to_numpy()))
    lim = max(1, lim + 4)
    ax.plot([0, lim], [0, lim], ls="--", lw=1.0, color="#666666")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed lag (months)")
    ax.set_ylabel("Combined reform lag (months)")
    ax.set_title(f"S3 Observed vs. combined-reform lag across supplementary cases (n={len(cases)})", loc="left", fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save_plot(fig, fig_dir / "S3_observed_vs_combined_reform_scatter")


def make_s4_scenario_heatmap(cases: pd.DataFrame, cf_long: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    heat = cf_long.pivot(index="case_id", columns="scenario", values="counterfactual_lag_months").reindex(cases["case_id"])
    heat = heat[SCENARIO_ORDER]
    case_order = cases.sort_values(["sector", "chain_type_norm", "observed_lag_months", "case_id"])["case_id"].tolist()
    heat = heat.reindex(case_order)
    fig, ax = plt.subplots(figsize=(8.2, 6.8), constrained_layout=True)
    sns.heatmap(
        heat,
        cmap=sns.light_palette("#4C78A8", as_cmap=True),
        annot=True,
        fmt=".1f",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Counterfactual lag (months)"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.set_title("S4 Case-by-scenario counterfactual lag heatmap", loc="left", fontsize=11.5, fontweight="bold")
    save_plot(fig, fig_dir / "S4_case_scenario_heatmap")


def make_s5_sector_means(cases: pd.DataFrame, cf_long: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        cf_long.groupby(["sector_label", "scenario"], as_index=False)["counterfactual_lag_months"]
        .mean()
        .rename(columns={"counterfactual_lag_months": "mean_lag"})
    )
    # Match the normalized labels used in the case tables and summary CSVs.
    order = ["AI", "NEV", "Eldercare", "Robotics"]
    sector_order = [s for s in order if s in summary["sector_label"].unique().tolist()]
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    width = 0.15
    x = np.arange(len(sector_order))
    for i, scenario in enumerate(SCENARIO_ORDER):
        vals = []
        for sector in sector_order:
            sub = summary[(summary["sector_label"] == sector) & (summary["scenario"] == scenario)]
            vals.append(float(sub["mean_lag"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(
            x + (i - 2) * width,
            vals,
            width=width,
            color=SCENARIO_COLORS[scenario],
            edgecolor="#333333",
            linewidth=0.3,
            label=SCENARIO_LABELS[scenario].replace("\n", " "),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(sector_order)
    ax.set_ylabel("Mean lag (months)")
    ax.set_title("S5 Sector-level mean lag under each governance scenario", loc="left", fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="upper right")
    save_plot(fig, fig_dir / "S5_sector_mean_scenario_bars")


def make_s6_reduction_heatmap(cases: pd.DataFrame, cf_long: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    baseline = cf_long[cf_long["scenario"] == "baseline"][["case_id", "counterfactual_lag_months"]].rename(columns={"counterfactual_lag_months": "baseline_lag"})
    merged = cf_long.merge(baseline, on="case_id", how="left")
    merged = merged[merged["scenario"] != "baseline"].copy()
    merged["reduction_pct"] = (merged["baseline_lag"] - merged["counterfactual_lag_months"]) / merged["baseline_lag"] * 100
    heat = merged.pivot(index="case_id", columns="scenario", values="reduction_pct")
    case_order = cases.sort_values(["sector", "chain_type_norm", "observed_lag_months", "case_id"])["case_id"].tolist()
    heat = heat.reindex(case_order)
    fig, ax = plt.subplots(figsize=(8.2, 6.8), constrained_layout=True)
    sns.heatmap(
        heat,
        cmap=sns.light_palette("#54A24B", as_cmap=True),
        annot=True,
        fmt=".1f",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Reduction vs. observed (%)"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.set_title("S6 Case-by-scenario reduction heatmap", loc="left", fontsize=11.5, fontweight="bold")
    save_plot(fig, fig_dir / "S6_case_reduction_heatmap")


def make_validation_case_grid(cases: pd.DataFrame, cf_long: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "validation"
    fig_dir.mkdir(parents=True, exist_ok=True)

    validation = cf_long[cf_long["scenario"] == "baseline"][
        ["case_id", "sector_label", "pathway_label", "observed_lag_months", "replay_lag_months"]
    ].copy()
    validation = validation.merge(
        cases[["case_id", "role"]],
        on="case_id",
        how="left",
    )
    validation["sector_label"] = pd.Categorical(
        validation["sector_label"],
        categories=["AI", "NEV", "Eldercare", "Robotics"],
        ordered=True,
    )
    validation = validation.sort_values(["sector_label", "pathway_label", "case_id"])

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6), constrained_layout=False)
    axes = axes.ravel()
    sector_order = ["AI", "NEV", "Eldercare", "Robotics"]
    colors = {"Observed": FIGURE_COLORS["observed"], "Simulated": FIGURE_COLORS["replay"]}
    ymax = float(np.nanmax(validation[["observed_lag_months", "replay_lag_months"]].to_numpy())) if not validation.empty else 1.0
    ymax = max(1.0, ymax * 1.20)

    for ax, sector in zip(axes, sector_order):
        sub = validation[validation["sector_label"] == sector].copy()
        if sub.empty:
            ax.axis("off")
            continue
        x = np.arange(len(sub))
        width = 0.34
        ax.bar(x - width / 2, sub["observed_lag_months"], width=width, color=colors["Observed"], edgecolor="#333333", linewidth=0.25, label="Observed")
        ax.bar(x + width / 2, sub["replay_lag_months"], width=width, color=colors["Simulated"], edgecolor="#333333", linewidth=0.25, label="Simulated")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["case_id"], fontsize=6.8)
        ax.set_ylim(0, ymax)
        ax.set_title(f"{sector} cases", loc="left", fontsize=10.2, fontweight="bold")
        ax.set_ylabel("Lag (months)")
        for i, row in enumerate(sub.itertuples()):
            ax.text(i - width / 2, row.observed_lag_months + ymax * 0.02, f"{row.observed_lag_months:.0f}", ha="center", fontsize=6.5)
            ax.text(i + width / 2, row.replay_lag_months + ymax * 0.02, f"{row.replay_lag_months:.0f}", ha="center", fontsize=6.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], ["Observed", "Simulated"], frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Case-level observed vs. simulated lag validation", x=0.01, y=0.995, ha="left", fontsize=11.0, fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.30, wspace=0.18)
    save_plot(fig, fig_dir / "validation_case_level_observed_vs_simulated")


def make_supplementary_suite(cases: pd.DataFrame, cf_long: pd.DataFrame, out_dir: Path) -> None:
    make_s1_case_validation_bar(cases, cf_long, out_dir)
    make_s2_observed_replay_scatter(cases, out_dir)
    make_s3_observed_reform_scatter(cases, cf_long, out_dir)
    make_s4_scenario_heatmap(cases, cf_long, out_dir)
    make_s5_sector_means(cases, cf_long, out_dir)
    make_s6_reduction_heatmap(cases, cf_long, out_dir)
    make_validation_case_grid(cases, cf_long, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    setup_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases()
    cf = build_counterfactuals(cases)
    main_cases = ordered_cases(cases[cases["role"] == "main"].copy())
    supplement_cases = ordered_cases(cases[cases["role"] == "supplement"].copy())

    make_tables(cases, cf, out_dir)
    write_case_notes(cases, cf, out_dir)
    plot_main_text_composite(main_cases, cf, out_dir)
    plot_overview(main_cases, cf, out_dir)
    for _, case in supplement_cases.iterrows():
        plot_supplement_case_figure(case, cf, out_dir)
    make_supplementary_suite(supplement_cases, cf, out_dir)

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "n_cases": int(len(cases)),
                "case_ids": cases["case_id"].tolist(),
                "main_case_ids": main_cases["case_id"].tolist(),
                "supplement_case_ids": supplement_cases["case_id"].tolist(),
                "table_long": str(out_dir / "tables" / "case_counterfactuals_long.csv"),
                "main_text_figure": str(out_dir / "figures" / "main_text" / "main_text_representative_case_composite.pdf"),
                "supplement_case_dir": str(out_dir / "figures" / "supplementary_cases"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
