"""Generate the eight-case supplementary evidence figure package."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = CODE_DIR.parent
ADVICE_DIR = PROJECT_ROOT / "04_advice" / "郭博士" / "7.17建议"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

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
COMPACT_CASE_LABELS = {
    "Taicang smart manufacturing": "Taicang",
    "Changzhou rail transit": "CZ rail",
    "Wuzhong robotics": "Wuzhong",
    "National AI industry-education": "National AI",
    "Changzhou NEV": "CZ NEV",
    "National NEV industry-education": "National NEV",
    "National smart health & eldercare": "Health eldercare",
    "National smart community eldercare": "Community eldercare",
}
CASE_TICK_LABELS = {
    "Taicang smart manufacturing": "Taicang\nsmart manufacturing",
    "Changzhou rail transit": "Changzhou\nrail transit",
    "Wuzhong robotics": "Wuzhong\nrobotics",
    "National AI industry-education": "National AI\nindustry-education",
    "Changzhou NEV": "Changzhou\nNEV",
    "National NEV industry-education": "National NEV\nindustry-education",
    "National smart health & eldercare": "Smart health\nand eldercare",
    "National smart community eldercare": "Smart community\neldercare",
}
CASE_CODES = {
    "Taicang smart manufacturing": "C1", "Changzhou rail transit": "C2",
    "Wuzhong robotics": "C3", "National AI industry-education": "C4",
    "Changzhou NEV": "C5", "National NEV industry-education": "C6",
    "National smart health & eldercare": "C7", "National smart community eldercare": "C8",
}

SHORT_BARRIERS = [
    "Policy-signal translation and governance-legitimation lag",
    "Industry-education carrier operationalization lag",
    "Firm-demand articulation and school-enterprise project-contracting lag",
    "Technical-standard curriculumization lag",
    "School instructional-supply preparation lag",
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
    "Firm-demand-list-driven school-enterprise project-contracting mechanism",
    "Rapid policy- and standard-to-curriculum translation mechanism",
    "Consortium operationalization and resource-coordination mechanism",
    "Policy-platform-curriculum-learning closed-loop coordination mechanism",
]
MECHANISM_BARRIER_MAP = {
    MECHANISMS[0]: [SHORT_BARRIERS[2], SHORT_BARRIERS[5]],
    MECHANISMS[1]: [SHORT_BARRIERS[0], SHORT_BARRIERS[3], SHORT_BARRIERS[4]],
    MECHANISMS[2]: [SHORT_BARRIERS[1], SHORT_BARRIERS[2], SHORT_BARRIERS[4]],
    MECHANISMS[3]: SHORT_BARRIERS,
}
COMPACT_MECHANISM_LABELS = {
    MECHANISMS[0]: "Firm-demand contracts",
    MECHANISMS[1]: "Policy/standard translation",
    MECHANISMS[2]: "Consortium operations",
    MECHANISMS[3]: "Closed-loop coordination",
}
STAGES = ["Central policy", "Local policy", "Consortium", "Carrier", "Education", "Training", "Outcome"]
STAGE_CODES = {"Central policy": "CP", "Local policy": "LP", "Consortium": "CO", "Carrier": "CA", "Education": "ED", "Training": "TR", "Outcome": "OUT"}
PALETTE = {"policy": "#0072B2", "carrier": "#999999", "enterprise": "#E69F00", "education": "#56B4E9", "outcome": "#009E73", "text": "#222222"}
STAGE_MARKERS = {"Central policy": "o", "Local policy": "s", "Consortium": "D", "Carrier": "P", "Education": "^", "Training": "v", "Outcome": "X"}
DISCRETE_BLUE = ListedColormap(["#F2F4F5", "#D6E2EE", "#A9C5DF", "#6A9BC2", "#0072B2"], name="nhb_discrete_blue")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})
sns.set_theme(style="white", context="paper")


def advice_file(prefix: str) -> Path:
    matches = sorted(ADVICE_DIR.glob(f"{prefix}*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"No advice workbook matching {prefix!r} in {ADVICE_DIR}")
    return matches[0]


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [("png", {"dpi": 600}), ("svg", {})]:
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def load_cases() -> pd.DataFrame:
    source = advice_file("1")
    frames = []
    for sheet_name, frame in pd.read_excel(source, sheet_name=None, engine="openpyxl").items():
        frame = frame.copy()
        frame.columns = [str(c).strip() for c in frame.columns]
        frame["case"] = sheet_name
        frame["case_label"] = CASE_LABELS.get(sheet_name, sheet_name)
        frame["date"] = pd.to_datetime(frame.get("时间"), errors="coerce")
        frame["dimension"] = frame.get("链路/维度", "").fillna("").astype(str)
        frame["text"] = frame.get("内容（已融入载体/办学/育人具体类型）", "").fillna("").astype(str)
        frame["actor"] = frame.get("政府/企业/学校名称（具体）", "").fillna("").astype(str)
        frame["documented"] = frame["date"].notna() & ~frame["text"].str.contains("待补", na=False)
        frames.append(frame)
    if not frames:
        raise ValueError("The evidence workbook contains no case sheets")
    data = pd.concat(frames, ignore_index=True)
    data["case_order"] = pd.Categorical(data["case"], categories=[x for x in CASE_LABELS if x in data["case"].unique()], ordered=True)
    return data


def stage_of(row: pd.Series) -> str:
    dim, text = row["dimension"], row["text"]
    if "中央政策" in dim:
        return "Central policy"
    if any(x in dim for x in ["省级政策", "地方政策", "区域布局", "联合体培育"]):
        return "Local policy"
    if any(x in dim for x in ["行业共同体建设", "联合体建设", "成员单位"]):
        return "Consortium"
    if "载体建设" in dim or dim.startswith("载体-"):
        return "Carrier"
    if "办学建设" in dim or dim.startswith("办学-"):
        return "Education"
    if "育人建设" in dim or dim.startswith("育人-"):
        return "Training"
    if "合作成效" in dim or "成效" in dim:
        return "Outcome"
    if "企业" in str(row.get("主体类型", "")) or "企业" in text:
        return "Carrier"
    return "Carrier"


def stage_color(stage: str) -> str:
    return {"Central policy": PALETTE["policy"], "Local policy": PALETTE["education"], "Consortium": PALETTE["carrier"], "Carrier": PALETTE["enterprise"], "Education": PALETTE["education"], "Training": PALETTE["policy"], "Outcome": PALETTE["outcome"]}.get(stage, "#8C8C8C")


def case_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case in data["case"].drop_duplicates().tolist():
        group = data[data["case"] == case]
        dated = group[group["documented"]].sort_values("date")
        if dated.empty:
            continue
        central = dated.loc[dated["stage"] == "Central policy", "date"]
        anchor = central.min() if not central.empty else dated["date"].min()
        for stage in STAGES:
            values = dated.loc[dated["stage"] == stage, "date"]
            date = values.min() if not values.empty else pd.NaT
            # Preserve records that predate the common central-policy anchor.
            lag = (date.year - anchor.year) * 12 + date.month - anchor.month if pd.notna(date) else np.nan
            rows.append({"case": case, "case_label": CASE_LABELS.get(case, case), "stage": stage, "date": date, "lag_from_policy_months": lag, "documented": pd.notna(date)})
    return pd.DataFrame(rows)


def load_barrier_counts(data: pd.DataFrame) -> pd.DataFrame:
    source = advice_file("2")
    detail = pd.read_excel(source, sheet_name="逐条证据提取表", engine="openpyxl")
    case_col, factor_col = "案例名称", "六类滞后指标"
    rows = []
    for case in data["case"].drop_duplicates().tolist():
        sub = detail[detail[case_col].astype(str).eq(case)] if case_col in detail else pd.DataFrame()
        for label in SHORT_BARRIERS:
            keyword = BARRIER_KEYWORDS[label]
            count = int(sub[factor_col].astype(str).str.contains(keyword, na=False).sum()) if factor_col in sub else 0
            rows.append({"case": case, "case_label": CASE_LABELS.get(case, case), "barrier": label, "evidence_records": count})
    return pd.DataFrame(rows)


def mechanism_priority_scores(barriers: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the evidence for lag factors each intervention is designed to address.

    These are diagnosis-based priorities, not observed causal effects of interventions.
    """
    rows = []
    for case, group in barriers.groupby("case", sort=False):
        for mechanism, linked_barriers in MECHANISM_BARRIER_MAP.items():
            score = group.loc[group["barrier"].isin(linked_barriers), "evidence_records"].sum()
            rows.append({
                "case": case,
                "case_label": CASE_LABELS.get(case, case),
                "mechanism": mechanism,
                "linked_evidence_records": int(score),
                "linked_lag_factors": "; ".join(linked_barriers),
            })
    return pd.DataFrame(rows)


def plot_lag_timeline(data: pd.DataFrame, metrics: pd.DataFrame) -> None:
    cases = data["case"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(cases), 1, figsize=(9.2, max(8.0, 1.1 * len(cases))), sharex=True, squeeze=False)
    for ax, case in zip(axes[:, 0], cases):
        dated = data[(data["case"] == case) & data["documented"]].sort_values("date")
        if dated.empty:
            ax.text(0.5, 0.5, "No dated evidence", ha="center", va="center")
            continue
        ax.hlines(0, dated["date"].min(), dated["date"].max(), color="#B8C2CC", lw=1.2)
        for _, row in dated.iterrows():
            ax.scatter(row["date"], 0, s=22, color=stage_color(row["stage"]), zorder=3)
        last = metrics[(metrics["case"] == case) & metrics["documented"]].sort_values("date").tail(1)
        lag = int(last.iloc[0]["lag_from_policy_months"]) if not last.empty and pd.notna(last.iloc[0]["lag_from_policy_months"]) else None
        title = CASE_LABELS.get(case, case) + (f" | observed span: {lag} months" if lag is not None else "")
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        ax.set_yticks([])
        ax.grid(axis="x", color="#E6E6E6", lw=0.6)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1, 0].set_xlabel("Documented date")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=stage_color(stage), label=stage, markersize=5) for stage in STAGES]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.975), frameon=False, fontsize=7)
    fig.suptitle("Observed transmission timelines across eight industry-education cases", y=1.01, fontweight="bold")
    fig.text(0.01, 0.01, "Observed span runs from the case policy anchor to the last dated record; it is a documented coverage interval, not a causal estimate. Blank evidence is excluded.", fontsize=6.3)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.95))
    save_figure(fig, "fig1_case_lag_timeline")


def plot_lag_heatmap(metrics: pd.DataFrame) -> None:
    order = [CASE_LABELS[case] for case in CASE_LABELS if case in metrics["case"].unique()]
    table = metrics.pivot(index="case_label", columns="stage", values="lag_from_policy_months").reindex(index=order, columns=STAGES)
    fig, ax = plt.subplots(figsize=(8.0, max(3.4, 0.48 * len(table) + 1.4)))
    sns.heatmap(table, mask=table.isna(), annot=True, fmt=".0f", cmap=DISCRETE_BLUE, vmin=0, vmax=max(table.max().max(), 1), linewidths=0.6, linecolor="white", cbar_kws={"label": "Cumulative lag from central policy (months)"}, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Earliest dated record at each transmission stage", loc="left", fontweight="bold")
    ax.set_xticklabels(["Central", "Local", "Consortium", "Carrier", "Education", "Training", "Outcome"], rotation=0)
    fig.text(0.01, 0.01, "Blank = no dated record in the current case file. A value of 0 denotes the policy anchor or a record dated before it.", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "fig2_lag_prevalence_heatmap")


def plot_barriers(barriers: pd.DataFrame) -> None:
    case_order = [CASE_LABELS[case] for case in CASE_LABELS if case in barriers["case"].unique()]
    table = barriers.pivot(index="barrier", columns="case_label", values="evidence_records").reindex(index=SHORT_BARRIERS, columns=case_order)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.heatmap(table, annot=True, fmt=".0f", cmap=DISCRETE_BLUE, vmin=0, vmax=max(table.max().max(), 1), linewidths=0.6, linecolor="white", cbar_kws={"label": "Coded evidence records"}, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Six field-theoretic lag factors in the eight-case evidence base", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.text(0.01, 0.01, "Counts indicate coded records, not effect sizes. The zero feedback row is retained to show the current evidence gap.", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "fig3_barrier_evidence_heatmap")


def plot_interventions(priorities: pd.DataFrame) -> None:
    case_order = [CASE_LABELS[case] for case in CASE_LABELS if case in priorities["case"].unique()]
    table = priorities.pivot(index="mechanism", columns="case_label", values="linked_evidence_records").reindex(index=MECHANISMS, columns=case_order)
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    sns.heatmap(table, annot=True, fmt=".0f", cmap=DISCRETE_BLUE, vmin=0, vmax=max(table.max().max(), 1), linewidths=0.8, linecolor="white", cbar_kws={"label": "Linked coded evidence records"}, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Diagnosis-based priorities for four implementable mechanisms", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.text(0.01, 0.01, "Each score aggregates evidence for the lag factors that the mechanism is designed to address; it is not an estimated intervention effect.", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "fig4_intervention_matrix")


def plot_composite(data: pd.DataFrame, metrics: pd.DataFrame, barriers: pd.DataFrame, priorities: pd.DataFrame) -> None:
    """Combine the four cross-case views without changing their evidence meaning."""
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 10.0))
    case_order = [CASE_CODES[CASE_LABELS[case]] for case in CASE_LABELS if case in data["case"].unique()]
    compact_metrics = metrics.assign(case_short=metrics["case_label"].map(CASE_CODES))
    compact_barriers = barriers.assign(case_short=barriers["case_label"].map(CASE_CODES))
    compact_priorities = priorities.assign(
        case_short=priorities["case_label"].map(CASE_CODES),
        mechanism_short=priorities["mechanism"].map(COMPACT_MECHANISM_LABELS),
    )

    # a. Observed event timelines: one row per case preserves the original temporal evidence.
    timeline_ax = axes[0, 0]
    cases = [case for case in CASE_LABELS if case in data["case"].unique()]
    for y, case in enumerate(cases):
        dated = data[(data["case"] == case) & data["documented"]].sort_values("date")
        if dated.empty:
            continue
        timeline_ax.hlines(y, dated["date"].min(), dated["date"].max(), color="#B8C2CC", lw=1.0, zorder=1)
        for _, row in dated.iterrows():
            timeline_ax.scatter(row["date"], y, s=18, marker=STAGE_MARKERS[row["stage"]], color=stage_color(row["stage"]), edgecolor="#4A4A4A", linewidth=.3, zorder=2)
    timeline_ax.set_yticks(range(len(cases)), [CASE_TICK_LABELS[CASE_LABELS[case]] for case in cases])
    timeline_ax.invert_yaxis()
    timeline_ax.xaxis.set_major_locator(mdates.YearLocator())
    timeline_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    timeline_ax.grid(axis="x", color="#E6E6E6", lw=0.6)
    timeline_ax.set_xlabel("Documented date")
    timeline_ax.set_title("a  Observed transmission timelines", loc="left", fontweight="bold")
    timeline_ax.tick_params(axis="y", labelsize=6.5)

    lag = compact_metrics.pivot(index="case_short", columns="stage", values="lag_from_policy_months").reindex(case_order)
    sns.heatmap(lag, mask=lag.isna(), annot=True, fmt=".0f", cmap=DISCRETE_BLUE, vmin=0, vmax=max(lag.max().max(), 1), cbar=False, linewidths=0.5, linecolor="white", ax=axes[0, 1])
    axes[0, 1].set_title("b  Earliest record by transmission stage", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel(""); axes[0, 1].set_ylabel("")
    axes[0, 1].set_xticklabels([STAGE_CODES[stage] for stage in STAGES], rotation=0, fontsize=6.5)
    axes[0, 1].tick_params(axis="y", labelsize=6.5)

    barrier_order = [
        "Policy translation", "Carrier operations", "Firm demand/contracts",
        "Standards to curriculum", "School supply", "Outcome feedback",
    ]
    barrier_table = compact_barriers.pivot(index="barrier", columns="case_short", values="evidence_records").reindex(index=SHORT_BARRIERS, columns=case_order)
    barrier_table.index = barrier_order
    sns.heatmap(barrier_table, annot=True, fmt=".0f", cmap=DISCRETE_BLUE, vmin=0, vmax=max(barrier_table.max().max(), 1), cbar=False, linewidths=0.5, linecolor="white", ax=axes[1, 0])
    axes[1, 0].set_title("c  Coded evidence for six lag factors", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel(""); axes[1, 0].set_ylabel("")
    axes[1, 0].tick_params(axis="x", rotation=0, labelsize=6.2)
    axes[1, 0].tick_params(axis="y", labelsize=6.3)

    mechanism_order = [COMPACT_MECHANISM_LABELS[mechanism] for mechanism in MECHANISMS]
    table = compact_priorities.pivot(index="mechanism_short", columns="case_short", values="linked_evidence_records").reindex(index=mechanism_order, columns=case_order)
    sns.heatmap(table, annot=True, fmt=".0f", cmap=DISCRETE_BLUE, vmin=0, vmax=max(table.max().max(), 1), cbar=False, linewidths=0.5, linecolor="white", ax=axes[1, 1])
    axes[1, 1].set_title("d  Diagnosis-based mechanism priorities", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel(""); axes[1, 1].set_ylabel("")
    axes[1, 1].tick_params(axis="x", rotation=0, labelsize=6.2)
    axes[1, 1].tick_params(axis="y", labelsize=6.3)
    fig.subplots_adjust(left=.11, right=.98, bottom=.18, top=.95, wspace=.30, hspace=.30)
    fig.text(0.01, .095, "C1, Taicang smart manufacturing; C2, Changzhou rail transit; C3, Wuzhong robotics; C4, National AI industry-education.", fontsize=6)
    fig.text(0.01, .070, "C5, Changzhou NEV; C6, National NEV industry-education; C7, National smart health and eldercare; C8, National smart community eldercare.", fontsize=6)
    fig.text(0.01, .035, "CP, central policy; LP, local policy; CO, consortium; CA, carrier; ED, education; TR, training; OUT, outcome. Blank = no dated record; 0 = policy anchor or earlier record; positive values = cumulative months from the central-policy anchor. Counts are coded records, not effect sizes.", fontsize=5.8)
    save_figure(fig, "fig_guo_smart_manufacturing_4results")


def main() -> None:
    data = load_cases()
    data["stage"] = data.apply(stage_of, axis=1)
    metrics = case_metrics(data)
    barriers = load_barrier_counts(data)
    priorities = mechanism_priority_scores(barriers)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DIR / "coded_evidence_records.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "case_lag_metrics.csv", index=False, encoding="utf-8-sig")
    barriers.to_csv(OUTPUT_DIR / "barrier_evidence_counts.csv", index=False, encoding="utf-8-sig")
    priorities.to_csv(OUTPUT_DIR / "mechanism_priority_scores.csv", index=False, encoding="utf-8-sig")
    priorities.to_csv(OUTPUT_DIR / "intervention_matrix.csv", index=False, encoding="utf-8-sig")
    plot_lag_timeline(data, metrics)
    plot_lag_heatmap(metrics)
    plot_barriers(barriers)
    plot_interventions(priorities)
    plot_composite(data, metrics, barriers, priorities)
    print(f"Generated eight-case figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
