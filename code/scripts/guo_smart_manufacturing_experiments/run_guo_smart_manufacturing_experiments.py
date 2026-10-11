"""Run the existing simulator and counterfactual logic for eight cases."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = CODE_DIR.parent
ADVICE_DIR = PROJECT_ROOT / "04_advice" / "郭博士" / "7.17建议"
SIMULATOR_FILE = CODE_DIR / "scripts" / "9run_social_simulator.py"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
sys.path.insert(0, str(CODE_DIR / "scripts"))

from publication_style import NHB_COLORS, apply_nhb_style

CASE_LABELS = {
    "太仓中德智能制造联合体": "Taicang smart manufacturing",
    "轨道交通装备共同体_常州": "Changzhou rail transit",
    "吴中机器人智能制造联合体": "Wuzhong robotics",
    "全国人工智能行业产教融合共同体": "National AI industry-education",
    "常州新能源产教联合体": "Changzhou NEV",
    "全国新能源行业产教融合共同体": "National NEV industry-education",
    "全国智慧健康养老产教融合共同体": "National smart health & eldercare",
    "全国智慧社区养老行业产教融合共同体": "National smart community eldercare",
}
SCENARIO_ORDER = ["baseline", "firm_school_ties", "translation_capacity", "platform_governance", "combined_reform"]
SCENARIO_LABELS = {"baseline": "Baseline", "firm_school_ties": "Firm-demand contracting", "translation_capacity": "Policy/standard translation", "platform_governance": "Carrier institutionalization", "combined_reform": "Closed-loop reform"}
SCENARIO_COLORS = {"baseline": NHB_COLORS["grey"], "firm_school_ties": NHB_COLORS["orange"], "translation_capacity": NHB_COLORS["blue"], "platform_governance": NHB_COLORS["violet"], "combined_reform": NHB_COLORS["teal"]}
SCENARIO_MARKERS = {"baseline": "o", "firm_school_ties": "s", "translation_capacity": "D", "platform_governance": "P", "combined_reform": "^"}
SCENARIO_HATCHES = {"baseline": "", "firm_school_ties": "///", "translation_capacity": "\\\\", "platform_governance": "xx", "combined_reform": "--"}
DISCRETE_BLUE = ListedColormap(["#F2F4F5", "#D6E2EE", "#A9C5DF", "#6A9BC2", "#0072B2"], name="nhb_discrete_blue")
PATHWAY_TICKS = {"Policy-to-firm": "Policy-\nto-firm", "Firm-to-school": "Firm-\nto-school", "Policy-to-school": "Policy-\nto-school", "Standard-to-curriculum": "Standard-\nto-curriculum"}
SCENARIO_TICKS = {"Baseline": "Baseline", "Firm-demand contracting": "Firm-demand\ncontracting", "Policy/standard translation": "Policy/standard\ntranslation", "Carrier institutionalization": "Carrier\ninstitutionalization", "Closed-loop reform": "Closed-loop\nreform"}
SHORT_BARRIERS = [
    "Policy signal hierarchy translation lag",
    "Industry-education carrier institutionalization lag",
    "Firm demand visibility and project contracting lag",
    "Technical standard curriculum translation lag",
    "School teaching-supply preparation lag",
    "Learning-outcome feedback lag",
]
BARRIER_KEYWORDS = {
    SHORT_BARRIERS[0]: "政策信号层级转译与治理合法化滞后",
    SHORT_BARRIERS[1]: "资源配置与组织载体实体化滞后",
    SHORT_BARRIERS[2]: "产业需求显性化与校企协同契约化滞后",
    SHORT_BARRIERS[3]: "技术标准共识形成与课程知识化滞后",
    SHORT_BARRIERS[4]: "教育场域教学供给合法化与能力准备滞后",
    SHORT_BARRIERS[5]: "育人成效兑现与反馈再嵌入滞后",
}
MECHANISMS = [
    "Firm-demand-list-driven project contracting",
    "Policy and standard to curriculum translation",
    "Consortium institutionalization and resource pooling",
    "Policy-platform-curriculum-learning closed loop",
]

apply_nhb_style()


def advice_file(prefix: str) -> Path:
    matches = sorted(ADVICE_DIR.glob(f"{prefix}*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"No advice workbook matching {prefix!r} in {ADVICE_DIR}")
    return matches[0]


def load_simulator():
    spec = importlib.util.spec_from_file_location("existing_social_simulator", SIMULATOR_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load simulator: {SIMULATOR_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_evidence() -> pd.DataFrame:
    rows = []
    for sheet_name, frame in pd.read_excel(advice_file("1"), sheet_name=None, engine="openpyxl").items():
        frame = frame.copy()
        frame.columns = [str(c).strip() for c in frame.columns]
        frame["case"] = sheet_name
        frame["case_label"] = CASE_LABELS.get(sheet_name, sheet_name)
        frame["date"] = pd.to_datetime(frame.get("时间"), errors="coerce")
        frame["dimension"] = frame.get("链路/维度", "").fillna("").astype(str)
        frame["text"] = frame.get("内容（已融入载体/办学/育人具体类型）", "").fillna("").astype(str)
        frame["documented"] = frame["date"].notna() & ~frame["text"].str.contains("待补", na=False)
        rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    data["case_order"] = pd.Categorical(data["case"], categories=[x for x in CASE_LABELS if x in data["case"].unique()], ordered=True)
    return data


def case_lag_metrics(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case in evidence["case"].drop_duplicates().tolist():
        documented = evidence[(evidence["case"] == case) & evidence["documented"]].sort_values("date")
        if documented.empty:
            rows.append({"case": case, "case_label": CASE_LABELS.get(case, case), "policy_start": pd.NaT, "latest_documented_date": pd.NaT, "documented_lag_months": np.nan, "documented_records": 0})
            continue
        central = documented.loc[documented["dimension"].str.contains("中央政策", na=False), "date"]
        start = central.min() if not central.empty else documented["date"].min()
        non_central = documented[~documented["dimension"].str.contains("中央政策", na=False)]
        end = non_central["date"].max() if not non_central.empty else documented["date"].max()
        lag = (end.year - start.year) * 12 + end.month - start.month if pd.notna(start) and pd.notna(end) else np.nan
        rows.append({"case": case, "case_label": CASE_LABELS.get(case, case), "policy_start": start, "latest_documented_date": end, "documented_lag_months": lag, "documented_records": len(documented)})
    return pd.DataFrame(rows)


def scenario_acceleration(simulator, chain_type: str, scenario_name: str) -> float:
    scenario = next(x for x in simulator.SCENARIOS if x.name == scenario_name)
    return float(simulator.scenario_acceleration(chain_type, scenario))


def build_experiment_tables(simulator, evidence: pd.DataFrame, metrics: pd.DataFrame, n_rep: int):
    empirical = simulator.empirical_data()
    params = simulator.pathway_params(empirical)
    simulations = []
    for scenario in simulator.SCENARIOS:
        sim = simulator.simulate(params, empirical, scenario, n_rep=n_rep, seed=20260714)
        sim["scenario_label"] = SCENARIO_LABELS.get(scenario.name, scenario.name)
        simulations.append(sim)
    sim_long = pd.concat(simulations, ignore_index=True)
    sim_summary = simulator.summarize(sim_long, empirical)
    sim_summary.to_csv(OUTPUT_DIR / "simulator_pathway_summary.csv", index=False, encoding="utf-8-sig")
    intervals = sim_long.groupby(["scenario", "scenario_label", "chain_type", "pathway"], as_index=False)["lag_months"].agg(
        q025=lambda x: x.quantile(0.025),
        sim_median="median",
        q975=lambda x: x.quantile(0.975),
    )
    baseline = intervals[intervals["scenario"] == "baseline"][["chain_type", "sim_median"]].rename(columns={"sim_median": "baseline_median"})
    effects = intervals[intervals["scenario"] != "baseline"].merge(baseline, on="chain_type", how="left")
    effects["reduction_months"] = effects["baseline_median"] - effects["sim_median"]
    effects["reduction_pct"] = 100 * effects["reduction_months"] / effects["baseline_median"]
    intervals.to_csv(OUTPUT_DIR / "simulator_pathway_intervals.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(OUTPUT_DIR / "simulator_pathway_effects.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "case_metrics.csv", index=False, encoding="utf-8-sig")
    return sim_long, sim_summary, intervals, effects


def barrier_counts(evidence: pd.DataFrame) -> pd.DataFrame:
    detail = pd.read_excel(advice_file("2"), sheet_name="逐条证据提取表", engine="openpyxl")
    rows = []
    for case in evidence["case"].drop_duplicates().tolist():
        sub = detail[detail["案例名称"].astype(str).eq(case)]
        for label, keyword in BARRIER_KEYWORDS.items():
            rows.append({"case": case, "case_label": CASE_LABELS.get(case, case), "barrier": label, "evidence_records": int(sub["六类滞后指标"].astype(str).str.contains(keyword, na=False).sum())})
    return pd.DataFrame(rows)


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [("pdf", {"bbox_inches": "tight"}), ("png", {"dpi": 600, "bbox_inches": "tight"}), ("svg", {"bbox_inches": "tight"})]:
        fig.savefig(OUTPUT_DIR / f"{name}.{suffix}", **kwargs)
    plt.close(fig)


def plot_evidence_timeline(evidence: pd.DataFrame) -> None:
    cases = evidence["case"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(cases), 1, figsize=(8.0, max(10.5, 1.45 * len(cases))), sharex=True, squeeze=False)
    colors = {"中央政策": "#0072B2", "省级政策": "#56B4E9", "地方政策": "#56B4E9", "联合体": "#999999", "载体建设": "#E69F00", "办学建设": "#56B4E9", "育人建设": "#0072B2", "合作成效": "#009E73"}
    for ax, case in zip(axes[:, 0], cases):
        group = evidence[(evidence["case"] == case) & evidence["documented"]].sort_values("date")
        if not group.empty:
            ax.hlines(0, group["date"].min(), group["date"].max(), color="#BCC6D1", lw=1.2)
            for _, row in group.iterrows():
                color = next((v for k, v in colors.items() if k in row["dimension"]), "#8C8C8C")
                marker = "o" if "政策" in row["dimension"] else "D" if "联合体" in row["dimension"] else "P" if "载体" in row["dimension"] else "^" if "办学" in row["dimension"] else "v" if "育人" in row["dimension"] else "X"
                ax.scatter(row["date"], 0, s=28, marker=marker, color=color, edgecolor="#4A4A4A", linewidth=.4, zorder=3)
        ax.set_title(CASE_LABELS.get(case, case), loc="left", fontweight="bold", fontsize=8, pad=8); ax.set_yticks([]); ax.grid(axis="x", color="#E5E5E5", lw=0.5)
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1, 0].tick_params(axis="x", labelbottom=True)
    axes[-1, 0].set_xlabel("Documented date")
    legend_items = [("中央政策", "Central policy"), ("地方政策", "Local policy"), ("联合体", "Consortium"), ("载体建设", "Carrier"), ("办学建设", "Education"), ("育人建设", "Training"), ("合作成效", "Outcome")]
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4.5, color=colors[key], label=label) for key, label in legend_items]
    fig.legend(handles=handles, labels=[label for _, label in legend_items], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.973), fontsize=6, frameon=False)
    fig.suptitle("Documented evidence timelines across eight cases", y=0.995, fontweight="bold")
    fig.subplots_adjust(hspace=0.55, top=0.93, bottom=0.08)
    save(fig, "fig1_case_evidence_timeline")


def plot_case_counterfactuals(metrics: pd.DataFrame, sim_summary: pd.DataFrame, cf: pd.DataFrame) -> None:
    sim_rows = sim_summary[(sim_summary["scenario"] == "baseline") & (sim_summary["chain_type"] == "policy_to_school")]
    sim_median = float(sim_rows["sim_median"].iloc[0]) if not sim_rows.empty else np.nan
    labels = {"baseline": "Baseline", "firm_school_ties": "Firm demand", "translation_capacity": "Translation", "platform_governance": "Carrier", "combined_reform": "Closed loop"}
    rows = []
    for _, case in metrics.iterrows():
        rows += [{"case_label": case["case_label"], "scenario_label": "Documented case", "lag": case["documented_lag_months"]}, {"case_label": case["case_label"], "scenario_label": "Simulator median", "lag": sim_median}]
    rows += [{"case_label": r["case_label"], "scenario_label": labels[r["scenario"]], "lag": r["counterfactual_lag_months"]} for r in cf[cf["scenario"] != "baseline"].to_dict("records")]
    frame = pd.DataFrame(rows)
    order = ["Documented case", "Simulator median", "Firm demand", "Translation", "Carrier", "Closed loop"]
    ncols = 2; nrows = int(np.ceil(metrics["case_label"].nunique() / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.2, max(5.0, 2.2 * nrows)), sharey=True, squeeze=False)
    for ax, case in zip(axes.flat, metrics["case_label"]):
        sub = frame[frame["case_label"] == case].copy(); sub["scenario_label"] = pd.Categorical(sub["scenario_label"], order, ordered=True); sub = sub.sort_values("scenario_label")
        sns.barplot(data=sub, x="scenario_label", y="lag", hue="scenario_label", palette={"Documented case": NHB_COLORS["grey"], "Simulator median": NHB_COLORS["blue"], "Firm demand": NHB_COLORS["orange"], "Translation": NHB_COLORS["sky_blue"], "Carrier": NHB_COLORS["violet"], "Closed loop": NHB_COLORS["teal"]}, legend=False, ax=ax, errorbar=None)
        ax.set_title(case, fontsize=8, fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("Lag (months)"); ax.set_xticklabels([SCENARIO_TICKS.get(str(label.get_text()), str(label.get_text())) for label in ax.get_xticklabels()], rotation=0, ha="center", fontsize=7)
    for ax in axes.flat[len(metrics):]: ax.axis("off")
    fig.suptitle("Case evidence, calibrated simulator reference and counterfactual projection", y=1.01, fontweight="bold")
    save(fig, "fig2_case_observed_simulated_counterfactual")


def plot_reduction_heatmap(effects: pd.DataFrame) -> None:
    table = effects.pivot(index="pathway", columns="scenario_label", values="reduction_pct")
    table = table[[SCENARIO_LABELS[x] for x in SCENARIO_ORDER[1:]]]
    table.columns = ["Firm demand", "Policy/standard", "Carrier", "Closed loop"]
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    sns.heatmap(table, annot=True, fmt=".1f", cmap=DISCRETE_BLUE, vmin=0, vmax=max(table.max().max(), 1), linewidths=0.6, linecolor="white", cbar_kws={"label": "Reduction in simulated median lag (%)"}, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Pathway-specific modelled lag reduction under four governance scenarios", loc="left", fontweight="bold"); ax.set_xticklabels(["Firm demand\ncontracting", "Policy/standard\ntranslation", "Carrier\ninstitutionalization", "Closed-loop\nreform"], rotation=0, ha="center", fontsize=7)
    fig.text(0.01, 0.01, "Values compare simulated medians with the baseline for each pathway. They are model-based scenario effects, not observed policy impacts.", fontsize=6.3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig3_counterfactual_reduction_heatmap")


def plot_pathway_scenario_effects(intervals: pd.DataFrame) -> None:
    pathway_order = intervals["pathway"].drop_duplicates().tolist()
    scenario_order = [SCENARIO_LABELS[x] for x in SCENARIO_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.8), sharex=False, squeeze=False)
    for ax, pathway in zip(axes.flat, pathway_order):
        sub = intervals[intervals["pathway"] == pathway].copy()
        sub["scenario_label"] = pd.Categorical(sub["scenario_label"], scenario_order, ordered=True)
        sub = sub.sort_values("scenario_label").reset_index(drop=True)
        y = np.arange(len(sub))
        colors = [SCENARIO_COLORS[SCENARIO_ORDER[i]] for i in range(len(sub))]
        ax.errorbar(sub["sim_median"], y, xerr=[sub["sim_median"] - sub["q025"], sub["q975"] - sub["sim_median"]], fmt="none", ecolor="#8C8C8C", elinewidth=1.0, capsize=2, zorder=1)
        for xx, yy, scenario, color in zip(sub["sim_median"], y, sub["scenario"], colors):
            ax.scatter(xx, yy, c=color, marker=SCENARIO_MARKERS[scenario], s=34, edgecolor="#4A4A4A", linewidth=.4, zorder=2)
        ax.set_yticks(y); ax.set_yticklabels([SCENARIO_TICKS.get(str(label), str(label)) for label in sub["scenario_label"]], fontsize=7)
        ax.invert_yaxis(); ax.set_title(pathway, loc="left", fontweight="bold", fontsize=8); ax.set_xlabel("Simulated lag (months)")
    fig.suptitle("Pathway-level scenario effects with 95% simulation intervals", fontweight="bold")
    fig.text(0.01, 0.01, "Intervals reflect simulated outcome variation conditional on the existing calibration; they are not parameter-confidence intervals.", fontsize=6.3)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, "fig2_observed_simulated_counterfactual")


def plot_simulator_distributions(sim_long: pd.DataFrame) -> None:
    sub = sim_long[sim_long["chain_type"] == "policy_to_school"].copy(); sub["scenario_label"] = pd.Categorical(sub["scenario"].map(SCENARIO_LABELS), [SCENARIO_LABELS[x] for x in SCENARIO_ORDER], ordered=True)
    short_labels = {SCENARIO_LABELS["baseline"]: "Baseline", SCENARIO_LABELS["firm_school_ties"]: "Firm demand", SCENARIO_LABELS["translation_capacity"]: "Translation", SCENARIO_LABELS["platform_governance"]: "Carrier", SCENARIO_LABELS["combined_reform"]: "Closed loop"}
    sub["scenario_short"] = sub["scenario_label"].map(short_labels)
    order = [short_labels[SCENARIO_LABELS[x]] for x in SCENARIO_ORDER]
    fig, ax = plt.subplots(figsize=(8.0, 3.4)); sns.violinplot(data=sub, x="scenario_short", y="lag_months", order=order, color=NHB_COLORS["sky_blue"], inner="quartile", cut=0, ax=ax); sns.stripplot(data=sub.sample(min(len(sub), 1000), random_state=7), x="scenario_short", y="lag_months", order=order, color=NHB_COLORS["blue"], size=1.3, alpha=0.18, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("Simulated lag (months)"); ax.set_title("Policy-to-school simulated lag by governance scenario", loc="left", fontweight="bold"); ax.set_xticks(np.arange(len(order))); ax.set_xticklabels([SCENARIO_TICKS.get(label, label) for label in order], rotation=0, ha="center", fontsize=7); save(fig, "fig4_simulator_pathway_distributions")


def plot_barriers(evidence: pd.DataFrame) -> None:
    table = barrier_counts(evidence).pivot(index="barrier", columns="case_label", values="evidence_records")
    fig, ax = plt.subplots(figsize=(8.0, 4.2)); sns.heatmap(table, annot=True, fmt=".0f", cmap=DISCRETE_BLUE, vmin=0, vmax=max(table.max().max(), 1), linewidths=0.6, linecolor="white", cbar_kws={"label": "Coded evidence records"}, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Six documented cross-field lag factors", loc="left", fontweight="bold"); ax.tick_params(axis="x", rotation=35, labelsize=7); save(fig, "fig5_documented_barriers")


def plot_intervention_ladder() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.8)); y = np.arange(len(MECHANISMS))[::-1]
    colors = [NHB_COLORS["orange"], NHB_COLORS["violet"], NHB_COLORS["sky_blue"], NHB_COLORS["teal"]]
    for i, (yy, label) in enumerate(zip(y, MECHANISMS)):
        ax.barh(yy, 0.22, left=0.04, color=colors[i], height=0.55); ax.text(0.15, yy, f"M{i + 1}", color="white", ha="center", va="center", fontweight="bold"); ax.text(0.32, yy, label, va="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(MECHANISMS) - 0.4); ax.axis("off"); ax.set_title("Four mechanisms translated from the simulator scenarios", loc="left", fontweight="bold"); save(fig, "fig6_intervention_ladder")


def plot_composite(sim_summary: pd.DataFrame, effects: pd.DataFrame, sim_long: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.5))
    baseline = sim_summary[sim_summary["scenario"] == "baseline"].copy()
    calibration = baseline.melt(id_vars="pathway", value_vars=["emp_median", "sim_median"], var_name="source", value_name="lag")
    sns.barplot(data=calibration, x="pathway", y="lag", hue="source", palette={"emp_median": NHB_COLORS["grey"], "sim_median": NHB_COLORS["blue"]}, ax=axes[0, 0], errorbar=None)
    for i, bar in enumerate(axes[0, 0].patches):
        bar.set_edgecolor("#4A4A4A"); bar.set_linewidth(.45)
    axes[0, 0].set_title("A  Baseline distributional calibration", loc="left", fontweight="bold"); axes[0, 0].set_xlabel(""); axes[0, 0].set_ylabel("Median lag (months)"); axes[0, 0].set_xticks(np.arange(len(baseline))); axes[0, 0].set_xticklabels([PATHWAY_TICKS.get(str(label), str(label)) for label in baseline["pathway"]], rotation=0, ha="center", fontsize=7); axes[0, 0].legend(["Empirical", "Simulated"], fontsize=7, title=None)
    effect_table = effects.pivot(index="pathway", columns="scenario_label", values="reduction_pct")[[SCENARIO_LABELS[x] for x in SCENARIO_ORDER[1:]]]
    sns.heatmap(effect_table, cmap=DISCRETE_BLUE, vmin=0, vmax=max(effect_table.max().max(), 1), annot=True, fmt=".1f", cbar=False, linewidths=0.4, linecolor="white", ax=axes[0, 1])
    axes[0, 1].set_title("B  Reduction in simulated median (%)", loc="left", fontweight="bold"); axes[0, 1].set_xlabel(""); axes[0, 1].set_ylabel(""); axes[0, 1].set_xticklabels(["Firm-demand\ncontracting", "Policy/standard\ntranslation", "Carrier\ninstitutionalization", "Closed-loop\nreform"], rotation=0, ha="center", fontsize=7)
    sub = sim_long[sim_long["chain_type"] == "policy_to_school"].copy()
    sub["scenario_label"] = pd.Categorical(sub["scenario"].map(SCENARIO_LABELS), [SCENARIO_LABELS[x] for x in SCENARIO_ORDER], ordered=True)
    sns.violinplot(data=sub, x="scenario_label", y="lag_months", color=NHB_COLORS["sky_blue"], inner="quartile", cut=0, ax=axes[1, 0])
    axes[1, 0].set_title("C  Policy-to-school scenario distribution", loc="left", fontweight="bold"); axes[1, 0].set_xlabel(""); axes[1, 0].set_ylabel("Simulated lag (months)"); axes[1, 0].set_xticks(np.arange(len(SCENARIO_ORDER))); axes[1, 0].set_xticklabels([SCENARIO_TICKS[SCENARIO_LABELS[item]] for item in SCENARIO_ORDER], rotation=0, ha="center", fontsize=7)
    combined = sim_summary[sim_summary["scenario"] == "combined_reform"].set_index("pathway").reindex(baseline["pathway"])
    ypos = np.arange(len(baseline))
    for y, start, end in zip(ypos, baseline["sim_median"], combined["sim_median"]):
        axes[1, 1].plot([start, end], [y, y], color="#B8C2CC", lw=1.2, zorder=1)
    axes[1, 1].scatter(baseline["sim_median"], ypos, color=NHB_COLORS["grey"], edgecolor="#4A4A4A", linewidth=.45, label="Baseline", zorder=2)
    axes[1, 1].scatter(combined["sim_median"], ypos, color=NHB_COLORS["teal"], marker="^", edgecolor="#006B4F", linewidth=.45, label="Closed-loop reform", zorder=2)
    axes[1, 1].set_yticks(np.arange(len(baseline))); axes[1, 1].set_yticklabels([PATHWAY_TICKS.get(str(label), str(label)) for label in baseline["pathway"]], fontsize=7); axes[1, 1].set_xlabel("Simulated median lag (months)"); axes[1, 1].set_title("D  Closed-loop comparison", loc="left", fontweight="bold"); axes[1, 1].legend(fontsize=7)
    fig.suptitle("Calibrated pathway-level scenario analysis", y=0.985, fontweight="bold")
    fig.text(0.01, 0.018, "The baseline simulation is distributionally calibrated to the empirical pathways; this panel is not an independent validation.", fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.93, wspace=0.28, hspace=0.48)
    save(fig, "fig_guo_smart_manufacturing_experiment_package")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--n-rep", type=int, default=5000); args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stale_case_projection = OUTPUT_DIR / "case_counterfactuals.csv"
    if stale_case_projection.exists():
        stale_case_projection.unlink()
    simulator = load_simulator(); evidence = load_evidence(); metrics = case_lag_metrics(evidence)
    sim_long, sim_summary, intervals, effects = build_experiment_tables(simulator, evidence, metrics, args.n_rep)
    plot_evidence_timeline(evidence); plot_pathway_scenario_effects(intervals); plot_reduction_heatmap(effects)
    plot_simulator_distributions(sim_long); plot_barriers(evidence); plot_intervention_ladder(); plot_composite(sim_summary, effects, sim_long)
    print(f"Generated pathway-level experiment outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
