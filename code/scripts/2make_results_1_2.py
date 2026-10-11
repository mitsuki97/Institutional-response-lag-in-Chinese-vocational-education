from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from publication_style import apply_nhb_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "event_analysis"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
MANUSCRIPT_DIR = PROJECT_ROOT / "outputs" / "manuscript"

MAIN_CHAIN_TYPES = [
    "policy_to_firm",
    "policy_to_school",
    "firm_to_school",
    "standard_to_curriculum",
]

CHAIN_LABELS = {
    "policy_to_firm": "Policy to firm",
    "policy_to_school": "Policy to school",
    "firm_to_school": "Firm to school",
    "standard_to_curriculum": "Standard to curriculum",
}

EVENT_LABELS = {
    "policy": "Policy",
    "demand": "Firm demand",
    "school_response": "School response",
    "standard": "Standard",
    "intermediary": "Intermediary",
    "firm_action": "Firm action",
}

DOMAIN_LABELS = {
    "人工智能领域": "Artificial intelligence",
    "新能源领域": "New energy",
    "智慧养老": "Smart elderly care",
    "智能制造": "Smart manufacturing",
}

COLORS = {
    "policy": "#7F8FA6",
    "demand": "#D99E73",
    "school_response": "#6EA6A1",
    "standard": "#B9A1D4",
    "intermediary": "#9CB97A",
    "firm_action": "#C77C7C",
    "policy_to_firm": "#8CA0C7",
    "policy_to_school": "#D08A76",
    "firm_to_school": "#78A89E",
    "standard_to_curriculum": "#AE9AC8",
}


def setup_style() -> None:
    apply_nhb_style()


def save_pub(fig: plt.Figure, stem: Path, width: float = 7.2, height: float = 4.8) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(width, height)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    try:
        fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    except OSError:
        pass


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.25, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")


def read_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(ANALYSIS_DIR / "events_all.csv")
    chains = pd.read_csv(ANALYSIS_DIR / "chains_all.csv")
    events["year_final"] = pd.to_numeric(events["year_final"], errors="coerce")
    chains["lag_months_num"] = pd.to_numeric(chains["lag_months_num"], errors="coerce")
    return events, chains


def compress_event_year(year: float) -> str:
    year_int = int(year)
    if year_int <= 2004:
        return "2000-2004"
    if year_int <= 2009:
        return "2005-2009"
    if year_int <= 2012:
        return "2010-2012"
    if year_int <= 2017:
        return "2013-2017"
    if year_int <= 2022:
        return "2018-2022"
    if year_int == 2023:
        return "2023"
    if year_int == 2024:
        return "2024"
    return "2025"


def event_year_sort_key(label: str) -> tuple[int, int]:
    if "-" in label:
        start = int(label.split("-", 1)[0])
        end = int(label.split("-", 1)[1])
        return start, end
    year = int(label)
    return year, year


def make_fig1(events: pd.DataFrame, chains: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.25, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    order_events = ["policy", "demand", "firm_action", "school_response", "intermediary", "standard"]
    events_a = events[events["event_type_norm"].isin(order_events)].copy()
    events_a["year_bin"] = events_a["year_final"].map(compress_event_year)
    year_order = sorted(events_a["year_bin"].dropna().unique().tolist(), key=event_year_sort_key)
    annual = (
        events_a
        .groupby(["year_bin", "event_type_norm"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order_events, fill_value=0)
        .reindex(year_order)
    )
    annual.plot(
        kind="bar",
        stacked=True,
        ax=ax_a,
        width=0.86,
        color=[COLORS[x] for x in order_events],
        linewidth=0,
    )
    ax_a.set_xlabel("Event period (years aggregated)")
    ax_a.set_ylabel("Number of events")
    ax_a.set_title("Temporal distribution of events", fontsize=7, pad=5)
    ax_a.legend(
        [EVENT_LABELS.get(x, x) for x in order_events],
        ncol=2,
        fontsize=5.0,
        loc="upper left",
        bbox_to_anchor=(0, 1.02),
    )
    ax_a.tick_params(axis="x", labelrotation=20, labelsize=5.2)
    panel_label(ax_a, "a")

    ax_b.set_axis_off()
    nodes = {
        "Policy": (0.12, 0.72),
        "Firm demand": (0.56, 0.72),
        "School": (0.56, 0.30),
        "Standard": (0.12, 0.30),
    }
    for name, (x, y) in nodes.items():
        ax_b.add_patch(Rectangle((x - 0.105, y - 0.055), 0.21, 0.11, fc="#F7F7F5", ec="#555555", lw=0.65))
        ax_b.text(x, y, name, ha="center", va="center", fontsize=6.5)
    arrow_specs = [
        ("Policy", "Firm demand", "policy_to_firm"),
        ("Policy", "School", "policy_to_school"),
        ("Firm demand", "School", "firm_to_school"),
        ("Standard", "School", "standard_to_curriculum"),
    ]
    chain_counts = chains[chains["chain_type_norm"].isin(MAIN_CHAIN_TYPES)]["chain_type_norm"].value_counts().to_dict()
    for start, end, ctype in arrow_specs:
        sx, sy = nodes[start]
        ex, ey = nodes[end]
        rad = 0.12 if ctype == "policy_to_school" else 0.0
        arrow = FancyArrowPatch(
            (sx + 0.11, sy),
            (ex - 0.11, ey),
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.1,
            color=COLORS[ctype],
            connectionstyle=f"arc3,rad={rad}",
        )
        if ctype == "standard_to_curriculum":
            arrow.set_positions((sx + 0.11, sy), (ex - 0.11, ey))
        ax_b.add_patch(arrow)
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        if ctype == "policy_to_school":
            mx -= 0.08
        ax_b.text(mx, my + 0.055, f"{CHAIN_LABELS[ctype]}\n(n={chain_counts.get(ctype, 0)})", ha="center", fontsize=5.8, color="#333333")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.set_title("Matched event-chain structure", fontsize=7.5, pad=5)
    panel_label(ax_b, "b")

    main = chains[chains["chain_type_norm"].isin(MAIN_CHAIN_TYPES)].copy()
    main["domain_label"] = main["domain_from_file"].map(DOMAIN_LABELS).fillna(main["domain_from_file"])
    main["chain_label"] = main["chain_type_norm"].map(CHAIN_LABELS)
    sns.boxplot(
        data=main,
        x="lag_months_num",
        y="chain_label",
        order=[CHAIN_LABELS[x] for x in MAIN_CHAIN_TYPES],
        ax=ax_c,
        showfliers=False,
        color="#D8D8D8",
        linewidth=0.65,
    )
    sns.stripplot(
        data=main.sample(min(len(main), 900), random_state=7),
        x="lag_months_num",
        y="chain_label",
        order=[CHAIN_LABELS[x] for x in MAIN_CHAIN_TYPES],
        ax=ax_c,
        hue="chain_type_norm",
        palette={k: COLORS[k] for k in MAIN_CHAIN_TYPES},
        size=1.5,
        alpha=0.35,
        jitter=0.18,
        legend=False,
    )
    ax_c.axvline(12, color="#444444", lw=0.7, ls="--")
    ax_c.axvline(24, color="#444444", lw=0.7, ls=":")
    ax_c.set_xlabel("Response lag (months)")
    ax_c.set_ylabel("")
    ax_c.set_xlim(-3, 120)
    ax_c.set_title("Lag distributions by event-chain type", fontsize=7, pad=5)
    panel_label(ax_c, "c")

    summary = []
    for ctype in MAIN_CHAIN_TYPES:
        s = main.loc[main["chain_type_norm"] == ctype, "lag_months_num"].dropna()
        summary.append(
            {
                "chain_type": ctype,
                "label": CHAIN_LABELS[ctype],
                "<12 months": 100 * (s < 12).mean(),
                "<24 months": 100 * (s < 24).mean(),
            }
        )
    resp = pd.DataFrame(summary).melt(id_vars=["chain_type", "label"], var_name="window", value_name="pct")
    sns.barplot(
        data=resp,
        x="label",
        y="pct",
        hue="window",
        ax=ax_d,
        palette=["#A7BBCB", "#5E7F91"],
        edgecolor="#333333",
        linewidth=0.25,
    )
    ax_d.set_ylabel("Matched chains (%)")
    ax_d.set_xlabel("")
    ax_d.set_ylim(0, 75)
    ax_d.tick_params(axis="x", rotation=28, labelsize=5.9)
    ax_d.set_title("Short-window response varies sharply across pathways", fontsize=7, pad=5)
    ax_d.legend(title="", loc="upper right", fontsize=6)
    panel_label(ax_d, "d")

    save_pub(fig, FIG_DIR / "fig1_event_chains_institutional_lag", width=7.25, height=5.4)
    plt.close(fig)


def make_fig2(events: pd.DataFrame, chains: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.25, 5.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.08], height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    main = chains[chains["chain_type_norm"].isin(MAIN_CHAIN_TYPES)].copy()
    main["domain_label"] = main["domain_from_file"].map(DOMAIN_LABELS).fillna(main["domain_from_file"])
    pathway_names = {
        "policy_to_firm": "Policy to firm",
        "policy_to_school": "Policy to school",
        "firm_to_school": "Firm to school",
        "standard_to_curriculum": "Standard to curriculum",
    }
    main["pathway"] = main["chain_type_norm"].map(pathway_names)

    ax_a.set_axis_off()
    y_positions = [0.83, 0.61, 0.39, 0.17]
    for y, ctype in zip(y_positions, MAIN_CHAIN_TYPES):
        ax_a.text(0.01, y, pathway_names[ctype], ha="left", va="center", fontsize=5.8, linespacing=0.95)
        ax_a.add_patch(Rectangle((0.45, y - 0.045), 0.15, 0.09, fc="#F7F7F5", ec="#555555", lw=0.55))
        ax_a.add_patch(Rectangle((0.78, y - 0.045), 0.16, 0.09, fc="#F7F7F5", ec="#555555", lw=0.55))
        left, right = {
            "policy_to_firm": ("Policy signal", "Firm"),
            "policy_to_school": ("Policy signal", "School"),
            "firm_to_school": ("Firm demand", "School"),
            "standard_to_curriculum": ("Standard", "Curriculum"),
        }[ctype]
        ax_a.text(0.525, y, left, ha="center", va="center", fontsize=5.6)
        ax_a.text(0.86, y, right, ha="center", va="center", fontsize=5.6)
        ax_a.add_patch(
            FancyArrowPatch(
                (0.61, y),
                (0.77, y),
                arrowstyle="-|>",
                mutation_scale=8,
                lw=1.15,
                color=COLORS[ctype],
            )
        )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_title("Governance pathways encoded by chains", fontsize=7.5, pad=5)
    panel_label(ax_a, "a")

    stats = (
        main.groupby("chain_type_norm")["lag_months_num"]
        .agg(median="median", mean="mean", n="count")
        .loc[MAIN_CHAIN_TYPES]
        .reset_index()
    )
    stats["label"] = stats["chain_type_norm"].map(CHAIN_LABELS)
    y = np.arange(len(stats))
    ax_b.hlines(y, 0, stats["median"], color="#BBBBBB", lw=2)
    ax_b.scatter(stats["median"], y, s=36, color=[COLORS[x] for x in stats["chain_type_norm"]], zorder=3)
    for _, row in stats.iterrows():
        ax_b.text(row["median"] + 1.6, y[stats.index[stats["chain_type_norm"] == row["chain_type_norm"]][0]], f"{row['median']:.0f} mo", va="center", fontsize=6)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(stats["label"])
    ax_b.set_xlabel("Median response lag (months)")
    ax_b.set_xlim(0, max(45, stats["median"].max() + 12))
    ax_b.invert_yaxis()
    ax_b.set_title("Education-facing pathways are slower", fontsize=7, pad=5)
    panel_label(ax_b, "b")

    comp = main.groupby(["domain_label", "chain_type_norm"]).size().reset_index(name="n")
    comp["pct"] = 100 * comp["n"] / comp.groupby("domain_label")["n"].transform("sum")
    comp["chain_label"] = comp["chain_type_norm"].map(CHAIN_LABELS)
    pivot = comp.pivot(index="domain_label", columns="chain_label", values="pct").fillna(0)
    pivot = pivot[[CHAIN_LABELS[x] for x in MAIN_CHAIN_TYPES]]
    sns.heatmap(pivot, ax=ax_c, cmap="Greys", cbar_kws={"label": "% of focal chains"}, linewidths=0.4, linecolor="white")
    ax_c.set_xlabel("")
    ax_c.set_ylabel("")
    ax_c.tick_params(axis="x", labelrotation=22, labelsize=6.2)
    ax_c.tick_params(axis="y", labelsize=5.8)
    ax_c.set_title("Pathway composition varies across domains", fontsize=7.5, pad=5)
    panel_label(ax_c, "c")

    domain_lag = (
        main.groupby(["domain_label", "chain_type_norm"])["lag_months_num"]
        .median()
        .reset_index(name="median_lag")
    )
    domain_lag["chain_label"] = domain_lag["chain_type_norm"].map(CHAIN_LABELS)
    pivot_lag = domain_lag.pivot(index="domain_label", columns="chain_label", values="median_lag")
    pivot_lag = pivot_lag[[CHAIN_LABELS[x] for x in MAIN_CHAIN_TYPES]]
    sns.heatmap(pivot_lag, ax=ax_d, cmap="mako_r", cbar_kws={"label": "Median lag (months)"}, linewidths=0.4, linecolor="white", annot=True, fmt=".0f", annot_kws={"fontsize": 5.8})
    ax_d.set_xlabel("")
    ax_d.set_ylabel("")
    ax_d.tick_params(axis="x", labelrotation=22, labelsize=6.2)
    ax_d.tick_params(axis="y", labelsize=6.2)
    ax_d.set_title("Slow pathways recur across technological domains", fontsize=7.5, pad=5)
    panel_label(ax_d, "d")

    save_pub(fig, FIG_DIR / "fig2_governance_pathways_response_speed", width=7.25, height=5.35)
    plt.close(fig)


def make_tables(events: pd.DataFrame, chains: pd.DataFrame) -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    main = chains[chains["chain_type_norm"].isin(MAIN_CHAIN_TYPES)].copy()
    table1 = pd.DataFrame(
        [
            ["Total events", f"{len(events):,}"],
            ["Policy events", f"{(events['event_type_norm'] == 'policy').sum():,}"],
            ["Firm demand events", f"{(events['event_type_norm'] == 'demand').sum():,}"],
            ["School response events", f"{(events['event_type_norm'] == 'school_response').sum():,}"],
            ["Industry organization or standard events", f"{((events['event_type_norm'] == 'intermediary') | (events['event_type_norm'] == 'standard')).sum():,}"],
            ["Matched event chains", f"{len(chains):,}"],
            ["Years covered", f"{int(events['year_final'].min())}--{int(events['year_final'].max())}"],
            ["Technological domains", f"{events['domain_from_file'].nunique()}"],
        ],
        columns=["Item", "Value"],
    )
    table2_rows = []
    for ctype in MAIN_CHAIN_TYPES:
        s = main.loc[main["chain_type_norm"] == ctype, "lag_months_num"].dropna()
        table2_rows.append(
            [
                CHAIN_LABELS[ctype],
                len(s),
                f"{s.mean():.1f}",
                f"{s.median():.0f}",
                f"{100 * (s < 12).mean():.1f}",
                f"{100 * (s < 24).mean():.1f}",
            ]
        )
    table2 = pd.DataFrame(
        table2_rows,
        columns=["Event-chain type", "N", "Mean lag", "Median lag", "<12 months (%)", "<24 months (%)"],
    )
    table1.to_csv(MANUSCRIPT_DIR / "table1_event_database_summary.csv", index=False)
    table2.to_csv(MANUSCRIPT_DIR / "table2_response_lag_by_chain.csv", index=False)


def latex_table1(events: pd.DataFrame, chains: pd.DataFrame) -> str:
    rows = [
        ("Total events", f"{len(events):,}"),
        ("Policy events", f"{(events['event_type_norm'] == 'policy').sum():,}"),
        ("Firm demand events", f"{(events['event_type_norm'] == 'demand').sum():,}"),
        ("School response events", f"{(events['event_type_norm'] == 'school_response').sum():,}"),
        ("Industry organization or standard events", f"{((events['event_type_norm'] == 'intermediary') | (events['event_type_norm'] == 'standard')).sum():,}"),
        ("Matched event chains", f"{len(chains):,}"),
        ("Years covered", f"{int(events['year_final'].min())}--{int(events['year_final'].max())}"),
        ("Technological domains", f"{events['domain_from_file'].nunique()}"),
    ]
    body = "\n".join([f"{item} & {value} \\\\" for item, value in rows])
    return textwrap.dedent(
        rf"""
        \begin{{table}}[t]
        \caption{{Event database used to measure institutional response lag.}}
        \label{{tab:event-database-summary}}
        \centering
        \begin{{tabular}}{{lr}}
        \toprule
        Item & Value \\
        \midrule
        {body}
        \bottomrule
        \end{{tabular}}
        \end{{table}}
        """
    ).strip()


def latex_table2(chains: pd.DataFrame) -> str:
    main = chains[chains["chain_type_norm"].isin(MAIN_CHAIN_TYPES)].copy()
    rows = []
    for ctype in MAIN_CHAIN_TYPES:
        s = main.loc[main["chain_type_norm"] == ctype, "lag_months_num"].dropna()
        rows.append(
            f"{CHAIN_LABELS[ctype]} & {len(s):,} & {s.mean():.1f} & {s.median():.0f} & {100 * (s < 12).mean():.1f} & {100 * (s < 24).mean():.1f} \\\\"
        )
    body = "\n".join(rows)
    return textwrap.dedent(
        rf"""
        \begin{{table}}[t]
        \caption{{Response lag across four focal event-chain types. Lag is measured in months between the upstream signal and the downstream institutional or organizational response.}}
        \label{{tab:response-lag-main-chains}}
        \centering
        \begin{{tabular}}{{lrrrrr}}
        \toprule
        Event-chain type & $N$ & Mean & Median & $<12$ months & $<24$ months \\
        \midrule
        {body}
        \bottomrule
        \end{{tabular}}
        \end{{table}}
        """
    ).strip()


def write_results(events: pd.DataFrame, chains: pd.DataFrame) -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    text = rf"""
\section*{{Results}}

\subsection*{{Event chains reveal measurable institutional lag}}

We first asked whether institutional lag in industry--education adaptation could be observed as a measurable temporal pattern rather than inferred only from retrospective description. To do so, we assembled a longitudinal event database covering four technology-facing domains in China from 2000 to 2025. The database contains {len(events):,} events and {len(chains):,} matched event chains linking upstream signals to downstream institutional or organizational responses (Table~\ref{{tab:event-database-summary}}). Events include policy announcements, firm demand signals, school responses, intermediary activities and standard-setting actions. To improve readability in the main figure, the earliest and latest years are grouped into broader temporal periods. Because one upstream event can trigger multiple downstream responses, the number of matched chains exceeds the number of individual events.

The resulting event chains show that response lag is not uniformly distributed across the system. Policy signals were followed by firm-facing responses more quickly than by education-facing responses. In the four focal chains used for the main analysis, policy-to-firm chains had a mean lag of 24.1 months and a median lag of 12 months, whereas policy-to-school chains had a mean lag of 36.9 months and a median lag of 27 months (Table~\ref{{tab:response-lag-main-chains}}). Firm-to-school chains also displayed substantial delay despite their more direct connection to skill demand, with a mean lag of 30.5 months and a median lag of 12 months. The slowest pathway was standard-to-curriculum translation, with a mean lag of 41.6 months and a median lag of 24 months.

Short-window response rates make this lag especially visible. Within one year, 34.3\% of policy-to-firm chains and 32.4\% of firm-to-school chains had generated a downstream response, compared with 18.9\% of policy-to-school chains and only 7.5\% of standard-to-curriculum chains. Within two years, policy-to-firm and firm-to-school response rates both reached roughly 64\%, whereas policy-to-school responses reached 38.8\% and standard-to-curriculum responses reached 31.3\% (Fig.~\ref{{fig:event-chains-lag}}). These differences indicate that institutional lag is concentrated not simply at the point where technology first enters the economy, but at the point where technological signals must be translated into educational routines, curriculum structures and formal standards.

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/fig1_event_chains_institutional_lag.pdf}}
    \caption{{\textbf{{Event chains reveal measurable institutional response lag.}} \textbf{{a}}, Temporal distribution of events in the longitudinal database by event type, with early and later years aggregated into broader periods for readability. \textbf{{b}}, Schematic of the four focal event-chain types used to measure response lag. Numbers indicate matched chains. \textbf{{c}}, Distribution of response lag in months for each focal chain type. Dashed and dotted vertical lines mark 12- and 24-month response windows. Points show sampled chains and boxes show the interquartile range; outliers beyond the plotted range are omitted from the visual display but retained in the statistics. \textbf{{d}}, Share of chains producing a downstream response within 12 and 24 months.}}
\label{{fig:event-chains-lag}}
\end{{figure}}

{latex_table1(events, chains)}

{latex_table2(chains)}

\subsection*{{Governance pathways shape the speed of institutional response}}

We next examined whether the observed delays were associated with the governance pathway through which a technological signal moved. The four focal chain types correspond to distinct institutional routes: a policy-centred industry route, a policy-centred education route, a horizontal firm--school route and a standard-mediated curricular route. These routes differ in the actors that carry the signal and in the degree of translation required before a downstream response can occur.

The fastest route was the policy-centred industry pathway. Policy-to-firm chains reached a median response lag of 12 months, suggesting that firms and industry-facing organizations can often react to public signals through investment, recruitment, product adjustment or demand articulation. The horizontal firm--school pathway also had a median lag of 12 months, but its mean lag was higher, indicating that direct demand signals can produce rapid responses in some cases while remaining delayed in others. By contrast, policy-to-school chains showed a median lag of 27 months, consistent with the slower organizational routines of educational institutions. The standard-mediated curricular pathway was also slow, with a median lag of 24 months and the lowest one-year response rate.

This pathway pattern supports a governance interpretation of institutional lag. When technological change is routed through actors able to adjust organizational behaviour directly, response can occur within one to two years. When the same signal must be translated into educational programmes, curriculum standards or formal training arrangements, delays become longer and more variable. Across the four technological domains, the relative ordering of pathways remains visible even though the composition of chains varies by domain (Fig.~\ref{{fig:governance-pathways}}). Thus, lag appears less as a generic property of the education system than as an outcome of the pathway through which demand, policy and standards are connected.

These findings provide the empirical basis for the simulation analysis that follows. The simulator treats government agencies, firms, schools, intermediary organizations and standard-setting bodies as actors occupying different positions in an institutional field. The empirical lag distributions reported here serve as calibration targets for the baseline model, while the differences across pathways motivate counterfactual experiments that alter the structure of inter-actor ties and the capacity for demand translation.

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/fig2_governance_pathways_response_speed.pdf}}
\caption{{\textbf{{Governance pathways shape the speed of institutional response.}} \textbf{{a}}, Four governance pathways encoded by matched event-chain types. \textbf{{b}}, Median response lag by pathway, showing slower education-facing and curriculum-facing routes. \textbf{{c}}, Composition of focal governance pathways across the four technological domains. \textbf{{d}}, Median lag by domain and pathway. The heatmaps are descriptive and are used to assess whether pathway-level delays are concentrated in a single domain or recur across domains.}}
\label{{fig:governance-pathways}}
\end{{figure}}
"""
    (MANUSCRIPT_DIR / "results_1_2.tex").write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    events, chains = read_data()
    make_fig1(events, chains)
    make_fig2(events, chains)
    make_tables(events, chains)
    write_results(events, chains)
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote manuscript section to {MANUSCRIPT_DIR / 'results_1_2.tex'}")


if __name__ == "__main__":
    main()
