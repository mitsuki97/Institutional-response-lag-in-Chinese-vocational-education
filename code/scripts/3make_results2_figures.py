from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from publication_style import apply_nhb_style
from matplotlib.patches import FancyArrowPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "outputs" / "event_analysis"
OUT_DIR = PROJECT_ROOT / "outputs" / "results2_governance_pathways"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"


PATHWAY_MAP = {
    "policy_to_firm": {
        "pathway": "Policy-to-firm",
        "governance_model": "state-led market activation",
        "source_actor": "Government",
        "target_actor": "Firm",
        "coordination_logic": "policy signal interpreted as industrial demand",
        "coordination_distance": 2,
    },
    "policy_to_school": {
        "pathway": "Policy-to-school",
        "governance_model": "administrative education response",
        "source_actor": "Government",
        "target_actor": "School",
        "coordination_logic": "policy signal translated into educational adjustment",
        "coordination_distance": 2,
    },
    "firm_to_school": {
        "pathway": "Firm-to-school",
        "governance_model": "demand-driven education response",
        "source_actor": "Firm",
        "target_actor": "School",
        "coordination_logic": "industrial demand translated into training provision",
        "coordination_distance": 2,
    },
    "standard_to_curriculum": {
        "pathway": "Standard-to-curriculum",
        "governance_model": "standard-mediated curricular conversion",
        "source_actor": "Standard body",
        "target_actor": "Curriculum",
        "coordination_logic": "codified standards incorporated into curricula",
        "coordination_distance": 3,
    },
}

ORDER = ["Policy-to-firm", "Firm-to-school", "Policy-to-school", "Standard-to-curriculum"]
DOMAIN_MAP = {
    "人工智能领域": "AI",
    "新能源领域": "New energy",
    "智能制造": "Smart mfg",
    "智慧养老": "Elder care",
}
COLORS = {
    "Policy-to-firm": "#4C78A8",
    "Firm-to-school": "#72B7B2",
    "Policy-to-school": "#F58518",
    "Standard-to-curriculum": "#B279A2",
}


def setup_style() -> None:
    apply_nhb_style()


def add_pathway_fields(chains: pd.DataFrame) -> pd.DataFrame:
    df = chains[chains["chain_type_norm"].isin(PATHWAY_MAP)].copy()
    for key in [
        "pathway",
        "governance_model",
        "source_actor",
        "target_actor",
        "coordination_logic",
        "coordination_distance",
    ]:
        df[key] = df["chain_type_norm"].map({k: v[key] for k, v in PATHWAY_MAP.items()})
    df["domain_en"] = df["domain_from_file"].map(DOMAIN_MAP).fillna(df["domain_from_file"])
    df["lag_months_num"] = pd.to_numeric(df["lag_months_num"], errors="coerce")
    return df[df["lag_months_num"].notna()].copy()


def bootstrap_ci(values: np.ndarray, stat=np.median, n_boot: int = 5000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return math.nan, math.nan
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = stat(values[idx], axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]))


def summarize_pathways(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pathway in ORDER:
        sub = df[df["pathway"] == pathway]
        values = sub["lag_months_num"].to_numpy()
        ci_lo, ci_hi = bootstrap_ci(values)
        rows.append(
            {
                "pathway": pathway,
                "governance_model": sub["governance_model"].iloc[0],
                "n": len(sub),
                "mean_lag_months": round(float(np.mean(values)), 1),
                "median_lag_months": round(float(np.median(values)), 1),
                "median_ci_95_low": round(float(ci_lo), 1),
                "median_ci_95_high": round(float(ci_hi), 1),
                "response_lt_12_pct": round(float(np.mean(values < 12) * 100), 1),
                "response_lt_24_pct": round(float(np.mean(values < 24) * 100), 1),
                "lag_over_60_pct": round(float(np.mean(values > 60) * 100), 1),
            }
        )
    return pd.DataFrame(rows)


def make_classification_table() -> pd.DataFrame:
    rows = []
    for chain_type, info in PATHWAY_MAP.items():
        rows.append(
            {
                "chain_type": chain_type,
                "pathway": info["pathway"],
                "governance_model": info["governance_model"],
                "source_actor": info["source_actor"],
                "target_actor": info["target_actor"],
                "coordination_logic": info["coordination_logic"],
                "coordination_distance": info["coordination_distance"],
            }
        )
    return pd.DataFrame(rows)


def draw_schematic(ax) -> None:
    ax.set_axis_off()
    positions = {
        "Government": (0.14, 0.72),
        "Firm": (0.50, 0.83),
        "School": (0.86, 0.61),
        "Standard body": (0.31, 0.25),
        "Curriculum": (0.73, 0.25),
    }
    for name, (x, y) in positions.items():
        ax.add_patch(
            Rectangle(
                (x - 0.14, y - 0.09),
                0.28,
                0.18,
                facecolor="#EEF1F4",
                edgecolor="#444444",
                lw=0.75,
            )
        )
        ax.text(x, y, name, ha="center", va="center", fontsize=5.9)
    arrows = [
        ("Policy to firm", COLORS["Policy-to-firm"], 0.0, (0.28, 0.755), (0.36, 0.83), (0.25, 0.98)),
        ("Firm to school", COLORS["Firm-to-school"], 0.0, (0.64, 0.79), (0.72, 0.66), (0.77, 0.82)),
        ("Policy to school", COLORS["Policy-to-school"], -0.06, (0.28, 0.65), (0.72, 0.57), (0.48, 0.43)),
        ("Standard to curriculum", COLORS["Standard-to-curriculum"], 0.0, (0.45, 0.25), (0.59, 0.25), (0.52, 0.11)),
    ]
    for label, color, rad, start, end, (label_x, label_y) in arrows:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=9,
            lw=1.2,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(arrow)
        ax.text(
            label_x,
            label_y,
            label,
            color=color,
            ha="center",
            va="center",
            fontsize=5.3,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 0.15},
        )
    ax.text(0.0, 1.02, "a", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def make_figure(df: pd.DataFrame, summary: pd.DataFrame, domain_counts: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.05], width_ratios=[0.92, 1.08])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_schematic(ax_a)

    sns.boxplot(
    data=df,
    x="lag_months_num",
    y="pathway",
    hue="pathway",      # 新增：把 y 变量赋给 hue
    order=ORDER,
    hue_order=ORDER,    # 新增：和 order 保持一致
    ax=ax_b,
    palette=COLORS,
    width=0.55,
    fliersize=1.4,
    linewidth=0.7,
    legend=False        # 新增：关掉多余图例
    )
    ax_b.set_xlabel("Response lag (months)")
    ax_b.set_ylabel("")
    ax_b.set_xlim(0, max(96, np.percentile(df["lag_months_num"], 98)))
    ax_b.axvline(12, color="#666666", lw=0.7, ls="--")
    ax_b.axvline(24, color="#999999", lw=0.7, ls=":")
    ax_b.text(12, 1.02, "12 mo", ha="center", va="bottom", fontsize=5.8, color="#555555", transform=ax_b.get_xaxis_transform())
    ax_b.text(24, 1.02, "24 mo", ha="center", va="bottom", fontsize=5.8, color="#777777", transform=ax_b.get_xaxis_transform())
    ax_b.text(-0.12, 1.04, "b", transform=ax_b.transAxes, fontsize=8, fontweight="bold", va="bottom")

    y = np.arange(len(ORDER))
    lt12 = summary.set_index("pathway").reindex(ORDER)["response_lt_12_pct"]
    lt24 = summary.set_index("pathway").reindex(ORDER)["response_lt_24_pct"]
    windows = [
        ("<12 months", lt12, "#AAB7C4"),
        ("12-24 months", (lt24 - lt12).clip(lower=0), "#6F879B"),
        (">=24 months", (100 - lt24).clip(lower=0), "#344D61"),
    ]
    left = np.zeros(len(ORDER))
    for label, values, color in windows:
        vals = values.to_numpy(dtype=float)
        ax_c.barh(y, vals, left=left, height=0.62, color=color, edgecolor="white", linewidth=0.45, label=label)
        left += vals
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(ORDER)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Share of matched chains (%)")
    ax_c.set_ylabel("")
    ax_c.set_xlim(0, 100)
    ax_c.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.55, 1.13),
        ncol=3,
        fontsize=5.4,
        columnspacing=0.8,
    )
    ax_c.text(-0.12, 1.04, "c", transform=ax_c.transAxes, fontsize=8, fontweight="bold", va="bottom")

    matrix = domain_counts.pivot(index="pathway", columns="domain_en", values="n").reindex(ORDER).fillna(0)
    matrix_pct = matrix.div(matrix.sum(axis=1), axis=0) * 100
    sns.heatmap(
        matrix_pct,
        ax=ax_d,
        cmap=sns.light_palette("#4C78A8", as_cmap=True),
        cbar_kws={"label": "Share within pathway (%)"},
        linewidths=0.35,
        linecolor="white",
        annot=matrix.astype(int),
        fmt="d",
        annot_kws={"fontsize": 5.8},
    )
    ax_d.set_xlabel("")
    ax_d.set_ylabel("")
    ax_d.tick_params(axis="x", rotation=0)
    ax_d.tick_params(axis="y", rotation=0)
    ax_d.text(-0.12, 1.04, "d", transform=ax_d.transAxes, fontsize=8, fontweight="bold", va="bottom")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        try:
            fig.savefig(FIG_DIR / f"fig2_governance_pathways.{ext}", **kwargs)
        except OSError as exc:
            print(f"Skipped {ext} export: {exc}")
    plt.close(fig)


def write_latex_tables(classification: pd.DataFrame, summary: pd.DataFrame) -> None:
    latex = []
    latex.append("% Results 2 table: governance pathway classification")
    try:
        latex.append(classification.to_latex(index=False, escape=True))
    except ImportError:
        latex.append("% LaTeX table export skipped: optional jinja2 dependency is unavailable.")
    latex.append("\n% Results 2 table: response lag by governance pathway")
    cols = [
        "pathway",
        "n",
        "median_lag_months",
        "median_ci_95_low",
        "median_ci_95_high",
        "mean_lag_months",
        "response_lt_12_pct",
        "response_lt_24_pct",
        "lag_over_60_pct",
    ]
    try:
        latex.append(summary[cols].to_latex(index=False, escape=True))
    except ImportError:
        latex.append("% LaTeX table export skipped: optional jinja2 dependency is unavailable.")
    (OUT_DIR / "results2_tables_latex.tex").write_text("\n".join(latex), encoding="utf-8")


def write_caption(summary: pd.DataFrame) -> None:
    fastest = summary.sort_values(["median_lag_months", "mean_lag_months"], ascending=True).iloc[0]
    longest_median = summary.sort_values("median_lag_months", ascending=False).iloc[0]
    weakest_short = summary.sort_values("response_lt_12_pct", ascending=True).iloc[0]
    caption = f"""Figure 2 | Governance pathways shape the speed of institutional response.
a, Conceptual mapping of the four focal event-chain pathways used in the analysis.
b, Distribution of response lags across governance pathways. Dashed and dotted vertical lines mark 12- and 24-month response windows.
c, Mutually exclusive response-window composition by pathway: below 12 months, 12--24 months and 24 months or longer.
d, Domain composition of each pathway; cell labels report chain counts and shading reports within-pathway percentages.
The fastest pathway by median lag is {fastest['pathway']} ({fastest['median_lag_months']:.0f} months), whereas {longest_median['pathway']} has the longest median lag ({longest_median['median_lag_months']:.0f} months). {weakest_short['pathway']} shows the weakest short-window response, consistent with the argument that codification and curricular conversion create longer institutional delays.
"""
    (OUT_DIR / "fig2_caption.txt").write_text(caption, encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)

    chains = pd.read_csv(IN_DIR / "chains_all.csv")
    df = add_pathway_fields(chains)
    df.to_csv(OUT_DIR / "results2_pathway_chains.csv", index=False, encoding="utf-8-sig")

    classification = make_classification_table()
    summary = summarize_pathways(df)
    domain_counts = (
        df.groupby(["pathway", "domain_en"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["pathway", "domain_en"])
    )

    classification.to_csv(TABLE_DIR / "table2a_pathway_classification.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "table2b_pathway_lag_summary.csv", index=False, encoding="utf-8-sig")
    domain_counts.to_csv(TABLE_DIR / "table2c_pathway_domain_counts.csv", index=False, encoding="utf-8-sig")
    classification.to_markdown(TABLE_DIR / "table2a_pathway_classification.md", index=False)
    summary.to_markdown(TABLE_DIR / "table2b_pathway_lag_summary.md", index=False)
    domain_counts.to_markdown(TABLE_DIR / "table2c_pathway_domain_counts.md", index=False)
    write_latex_tables(classification, summary)
    make_figure(df, summary, domain_counts)
    write_caption(summary)
    print(f"Results 2 outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
