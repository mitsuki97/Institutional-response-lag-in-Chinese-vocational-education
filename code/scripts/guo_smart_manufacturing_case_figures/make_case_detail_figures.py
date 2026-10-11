"""Generate granular supplementary figures for all eight supplied cases."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import re
import sys
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parents[1]
OUTPUT_DIR = SCRIPT_DIR / "outputs"
EXPERIMENT_OUTPUT = CODE_DIR / "scripts" / "guo_smart_manufacturing_experiments" / "outputs"
FIGURE_SCRIPT = CODE_DIR / "scripts" / "guo_smart_manufacturing_figures" / "make_guo_smart_manufacturing_figures.py"
sys.path.insert(0, str(CODE_DIR / "scripts"))

from publication_style import NHB_COLORS, apply_nhb_style

spec = importlib.util.spec_from_file_location("eight_case_figures", FIGURE_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load shared evidence rules from {FIGURE_SCRIPT}")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
CASE_LABELS = shared.CASE_LABELS
STAGES = shared.STAGES
STAGE_CODES = shared.STAGE_CODES
STAGE_COLORS = {"Central policy": NHB_COLORS["blue"], "Local policy": NHB_COLORS["sky_blue"], "Consortium": NHB_COLORS["grey"], "Carrier": NHB_COLORS["orange"], "Education": NHB_COLORS["violet"], "Training": NHB_COLORS["vermillion"], "Outcome": NHB_COLORS["teal"]}
SHORT_BARRIERS = shared.SHORT_BARRIERS
BARRIER_KEYWORDS = shared.BARRIER_KEYWORDS
MECHANISMS = shared.MECHANISMS

apply_nhb_style()


def save(fig: plt.Figure, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [("pdf", {"bbox_inches": "tight"}), ("png", {"dpi": 600, "bbox_inches": "tight"}), ("svg", {"bbox_inches": "tight"})]:
        fig.savefig(folder / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def load_evidence() -> pd.DataFrame:
    data = shared.load_cases()
    data["stage"] = data.apply(shared.stage_of, axis=1)
    return data


def stage_table(case_data: pd.DataFrame) -> pd.DataFrame:
    dated = case_data[case_data["documented"]].sort_values("date")
    central = dated.loc[dated["stage"] == "Central policy", "date"]
    anchor = central.min() if not central.empty else (dated["date"].min() if not dated.empty else pd.NaT)
    rows = []
    for stage in STAGES:
        values = dated.loc[dated["stage"] == stage, "date"]
        date = values.min() if not values.empty else pd.NaT
        lag = (date.year - anchor.year) * 12 + date.month - anchor.month if pd.notna(date) and pd.notna(anchor) else np.nan
        rows.append({"stage": stage, "stage_label": stage, "first_documented_date": date, "cumulative_lag_months": lag, "documented": pd.notna(date)})
    return pd.DataFrame(rows)


def evidence_counts(case_data: pd.DataFrame) -> pd.DataFrame:
    detail = pd.read_excel(shared.advice_file("2"), sheet_name="逐条证据提取表", engine="openpyxl")
    case = case_data["case"].iloc[0]
    sub = detail[detail["案例名称"].astype(str).eq(case)]
    rows = []
    for label, keyword in BARRIER_KEYWORDS.items():
        rows.append({"barrier": label, "evidence_records": int(sub["六类滞后指标"].astype(str).str.contains(keyword, na=False).sum())})
    return pd.DataFrame(rows)


def plot_chain(case_data: pd.DataFrame, stages: pd.DataFrame, folder: Path) -> None:
    dated = case_data[case_data["documented"]].sort_values("date")
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    if dated.empty:
        ax.text(0.5, 0.5, "No dated evidence", ha="center", va="center")
    else:
        ax.hlines(0, dated["date"].min(), dated["date"].max(), color="#B8C2CC", lw=1.5)
        for i, (_, row) in enumerate(stages[stages["documented"]].iterrows()):
            x = row["first_documented_date"]
            ax.scatter(x, 0, s=80, color=STAGE_COLORS[row["stage"]], zorder=3)
            yoff = 36 if i % 2 == 0 else -42
            ax.annotate(f"{STAGE_CODES[row['stage']]}\n{x:%Y-%m}\n{int(row['cumulative_lag_months'])} mo", xy=(x, 0), xytext=(0, yoff), textcoords="offset points", ha="center", va="bottom" if yoff > 0 else "top", fontsize=8, arrowprops={"arrowstyle": "-", "color": "#8C8C8C", "lw": 0.75})
        ax.set_xlim(dated["date"].min() - pd.Timedelta(days=100), dated["date"].max() + pd.Timedelta(days=100))
    observed_stages = [stage for stage in STAGES if stages.loc[stages["stage"].eq(stage), "documented"].any()]
    legend_handles = [
        Line2D([0], [0], marker="o", markersize=6.5, linestyle="none",
               markerfacecolor=STAGE_COLORS[stage], markeredgecolor="none",
               label=f"{STAGE_CODES[stage]}  {stage}")
        for stage in observed_stages
    ]
    fig.legend(handles=legend_handles, title="Evidence stage", loc="lower center",
               bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False,
               fontsize=7.2, title_fontsize=8, handletextpad=0.35, columnspacing=1.15)
    fig.subplots_adjust(bottom=0.31)
    ax.set_yticks([]); ax.set_xlabel("Documented date", fontsize=9); ax.tick_params(axis="x", labelsize=8)
    # This panel is raster-composited into a half-width overview column.
    # Compensate for that down-scaling so its title matches panels c and d.
    ax.set_title(f"a  {case_data['case_label'].iloc[0]}: documented transmission chain", loc="left", fontsize=13.5, fontweight="bold")
    save(fig, folder, "figA_documented_chain")


def plot_stage_lag(stages: pd.DataFrame, case_label: str, folder: Path) -> None:
    sub = stages.copy(); sub["plot_lag"] = sub["cumulative_lag_months"].fillna(0); colors = [STAGE_COLORS[stage] if ok else "#D9D9D9" for stage, ok in zip(sub["stage"], sub["documented"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.9)); bars = ax.bar(sub["stage_label"], sub["plot_lag"], color=colors, edgecolor="#FFFFFF", linewidth=.5)
    for bar, (_, row) in zip(bars, sub.iterrows()):
        height = bar.get_height()
        y = height + 0.5 if height >= 0 else height - 0.75
        va = "bottom" if height >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{int(row['cumulative_lag_months'])} mo" if row["documented"] else "NA", ha="center", va=va, fontsize=8)
    ax.axhline(0, color="#7F8C8D", linewidth=0.7)
    ax.set_xticks(range(len(sub))); ax.set_xticklabels([fill(x, 13) for x in sub["stage_label"]], rotation=0, ha="center", fontsize=8); ax.set_xlabel(""); ax.set_ylabel("Cumulative lag from central policy (months)", fontsize=9); ax.tick_params(axis="y", labelsize=8)
    ax.set_title(f"b  {case_label}: lag accumulated across transmission stages", loc="left", fontsize=13.5, fontweight="bold")
    fig.subplots_adjust(bottom=0.28); save(fig, folder, "figB_stage_lag")


def plot_barriers_and_interventions(case_data: pd.DataFrame, case_label: str, folder: Path) -> None:
    documented = case_data[case_data["documented"]]
    factor = evidence_counts(case_data).set_index("barrier")
    record_text = documented["dimension"].fillna("") + " " + documented["text"].fillna("")
    interventions = {"Consortium / joint body": r"共同体|联合体|联盟", "Industry college / platform": r"产业学院|企业学院|产业园|平台|中心", "Curriculum / standards": r"课程|教材|专业|标准", "Apprenticeship / dual system": r"双元制|学徒", "Field engineer": r"现场工程师|职业资格|鉴定", "Employment / R&D outcome": r"就业|岗位|研发|创新|成果|技术"}
    right = pd.DataFrame({"item": list(interventions), "evidence_records": [int(record_text.str.contains(p, regex=True, na=False).sum()) for p in interventions.values()]}).set_index("item")
    combined = pd.concat([factor.rename_axis("item"), right])
    combined.to_csv(folder / "coded_evidence_counts.csv", encoding="utf-8-sig")
    panels = [
        (factor.reset_index(), "barrier", "C  Six lag-factor records", "#C96B32", "figC_lag_factors"),
        (right.reset_index(), "item", "D  Documented intervention records", "#4E9B63", "figD_interventions"),
    ]
    for frame, y_column, title, color, stem in panels:
        fig, ax = plt.subplots(figsize=(7.2, 3.9))
        sns.barplot(data=frame, y=y_column, x="evidence_records", color=color, ax=ax, errorbar=None)
        upper = max(1, int(frame["evidence_records"].max()))
        ax.set_xlim(0, upper * 1.18)
        for bar, value in zip(ax.patches, frame["evidence_records"]):
            ax.text(value + upper * 0.03, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=8)
        ax.set_xlabel("Coded records", fontsize=9)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(f"{title}: {case_label}", loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        save(fig, folder, stem)


def plot_counterfactual(case_label: str, folder: Path) -> None:
    path = EXPERIMENT_OUTPUT / "case_counterfactuals.csv"
    if not path.exists():
        return
    cf = pd.read_csv(path); sub = cf[cf["case_label"] == case_label].copy()
    if sub.empty:
        return
    sim_path = EXPERIMENT_OUTPUT / "simulator_pathway_summary.csv"; sim = pd.read_csv(sim_path) if sim_path.exists() else pd.DataFrame(); sim_rows = sim[(sim["scenario"] == "baseline") & (sim["chain_type"] == "policy_to_school")] if not sim.empty else pd.DataFrame(); sim_median = float(sim_rows["sim_median"].iloc[0]) if not sim_rows.empty else np.nan
    labels = {"baseline": "Documented case", "firm_school_ties": "Firm demand", "translation_capacity": "Translation", "platform_governance": "Carrier", "combined_reform": "Closed loop"}
    rows = [{"scenario": "Documented case", "lag": float(sub.loc[sub["scenario"] == "baseline", "counterfactual_lag_months"].iloc[0])}, {"scenario": "Simulator median", "lag": sim_median}]
    rows += [{"scenario": labels[r["scenario"]], "lag": r["counterfactual_lag_months"]} for r in sub[sub["scenario"] != "baseline"].to_dict("records")]
    frame = pd.DataFrame(rows); order = ["Documented case", "Simulator median", "Firm demand", "Translation", "Carrier", "Closed loop"]; frame["scenario"] = pd.Categorical(frame["scenario"], order, ordered=True); frame = frame.sort_values("scenario")
    palette = {"Documented case": NHB_COLORS["grey"], "Simulator median": NHB_COLORS["blue"], "Firm demand": NHB_COLORS["orange"], "Translation": NHB_COLORS["sky_blue"], "Carrier": NHB_COLORS["violet"], "Closed loop": NHB_COLORS["teal"]}
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); sns.barplot(data=frame, x="scenario", y="lag", hue="scenario", palette=palette, legend=False, ax=ax, errorbar=None); ax.set_title(f"D  {case_label}: model-based counterfactual projection", loc="left", fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("Lag (months)"); ax.tick_params(axis="x", labelrotation=45, labelsize=7); save(fig, folder, "figD_counterfactual_projection")


def main() -> None:
    data = load_evidence(); cases = data["case"].drop_duplicates().tolist(); summary_rows = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_case_dir in OUTPUT_DIR.glob("case_*"):
        if old_case_dir.is_dir():
            shutil.rmtree(old_case_dir)
    for case in cases:
        case_data = data[data["case"] == case].copy(); folder = OUTPUT_DIR / f"case_{cases.index(case) + 1:02d}_{case.replace('/', '_')}"; folder.mkdir(parents=True, exist_ok=True); stages = stage_table(case_data); stage_rows = stages.assign(case=case, case_label=case_data["case_label"].iloc[0]); stage_rows.to_csv(folder / "stage_lag_metrics.csv", index=False, encoding="utf-8-sig"); case_data.to_csv(folder / "evidence_records.csv", index=False, encoding="utf-8-sig"); plot_chain(case_data, stages, folder); plot_stage_lag(stages, case_data["case_label"].iloc[0], folder); plot_barriers_and_interventions(case_data, case_data["case_label"].iloc[0], folder); summary_rows.append({"case": case, "case_label": case_data["case_label"].iloc[0], "documented_records": int(case_data["documented"].sum()), "documented_span_months": float(stages["cumulative_lag_months"].dropna().max()) if stages["cumulative_lag_months"].notna().any() else np.nan})
    summary = pd.DataFrame(summary_rows); summary.to_csv(OUTPUT_DIR / "case_summary.csv", index=False, encoding="utf-8-sig"); summary.to_markdown(OUTPUT_DIR / "case_summary.md", index=False); print(f"Generated granular outputs for {len(summary)} cases in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
