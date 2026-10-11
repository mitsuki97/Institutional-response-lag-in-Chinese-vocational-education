from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from publication_style import apply_nhb_style
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "outputs" / "event_analysis"
OUT_DIR = PROJECT_ROOT / "outputs" / "results3_agent_conditioning"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

ROLE_ORDER = ["government", "firm", "school", "intermediary", "standard_body"]
ROLE_LABELS = {
    "government": "Government",
    "firm": "Firm",
    "school": "School",
    "intermediary": "Intermediary",
    "standard_body": "Standard body",
}
SHORT_ROLE_LABELS = {
    "Government": "Government",
    "Firm": "Firm",
    "School": "School",
    "Intermediary": "Intermediary",
    "Standard body": "Standard\nbody",
}
FULL_ROLE_SHORT = {
    "Government": "Government",
    "Firm": "Firm",
    "School": "School",
    "Intermediary": "Intermediary",
    "Standard body": "Standard body",
}
DOMAIN_LABELS = {
    "AI": "AI",
    "NEV": "New energy",
    "robotics": "Smart mfg",
    "eldercare": "Elder care",
}

CAPITAL_PRIORS = pd.DataFrame(
    [
        ["government", 0.95, 0.42, 0.18, 0.46, 0.86, 0.78],
        ["firm", 0.24, 0.92, 0.30, 0.52, 0.56, 0.70],
        ["school", 0.46, 0.32, 0.92, 0.48, 0.72, 0.52],
        ["intermediary", 0.38, 0.54, 0.46, 0.86, 0.70, 0.82],
        ["standard_body", 0.52, 0.36, 0.54, 0.82, 0.88, 0.64],
    ],
    columns=[
        "agent_role",
        "regulatory",
        "economic",
        "pedagogical",
        "translation",
        "symbolic",
        "network",
    ],
)


ROLE_DESIGN = {
    "government": {
        "field_position": "regulatory centre",
        "dominant_capital": "regulatory; symbolic",
        "historical_evidence": "policy events and policy-triggered chains",
        "llm_decision_outputs": "policy urgency, coordination mode, resource allocation, expected diffusion delay",
    },
    "firm": {
        "field_position": "demand-side producer",
        "dominant_capital": "economic; network",
        "historical_evidence": "demand reports, firm actions and firm-to-school chains",
        "llm_decision_outputs": "skill demand intensity, willingness to cooperate, hiring pressure, signal clarity",
    },
    "school": {
        "field_position": "training institution",
        "dominant_capital": "pedagogical; symbolic",
        "historical_evidence": "school response events and downstream education adjustments",
        "llm_decision_outputs": "perceived feasibility, resource constraint, curriculum action, expected response delay",
    },
    "intermediary": {
        "field_position": "translation broker",
        "dominant_capital": "translation; network",
        "historical_evidence": "industry reports, talent reports, forums and intermediary chains",
        "llm_decision_outputs": "translation confidence, coordination request, standardization need, brokerage action",
    },
    "standard_body": {
        "field_position": "codification authority",
        "dominant_capital": "symbolic; translation",
        "historical_evidence": "standards, certifications and standard-to-curriculum chains",
        "llm_decision_outputs": "codification readiness, consensus threshold, curriculum compatibility, expected conversion delay",
    },
}


def setup_style() -> None:
    apply_nhb_style()


def map_agent_role(row: pd.Series) -> str:
    event_type = str(row.get("event_type", "")).strip()
    actor_type = str(row.get("actor_type", "")).strip()
    if event_type == "standard":
        return "standard_body"
    if actor_type == "government":
        return "government"
    if actor_type == "firm":
        return "firm"
    if actor_type == "school":
        return "school"
    if actor_type in {"intermediary", "third_party"}:
        return "intermediary"
    return "intermediary"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(IN_DIR / "events_all.csv")
    chains = pd.read_csv(IN_DIR / "chains_all.csv")
    events["agent_role"] = events.apply(map_agent_role, axis=1)
    events["domain_label"] = events["sector"].map(DOMAIN_LABELS).fillna(events["sector"])
    chains["lag_months_num"] = pd.to_numeric(chains["lag_months_num"], errors="coerce")
    id_to_role = events.set_index("event_id")["agent_role"].to_dict()
    id_to_domain = events.set_index("event_id")["domain_label"].to_dict()
    chains["upstream_role"] = chains["upstream_event_id"].map(id_to_role)
    chains["downstream_role"] = chains["downstream_event_id"].map(id_to_role)
    chains["domain_label"] = chains["upstream_event_id"].map(id_to_domain)
    chains = chains[chains["upstream_role"].notna() & chains["downstream_role"].notna()].copy()
    return events, chains


def minmax(series: pd.Series, low: float = 0.15, high: float = 0.95) -> pd.Series:
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series(np.full(len(s), (low + high) / 2), index=s.index)
    return low + (s - s.min()) / (s.max() - s.min()) * (high - low)


def build_tables(events: pd.DataFrame, chains: pd.DataFrame) -> dict[str, pd.DataFrame]:
    evidence_counts = (
        events.groupby(["agent_role", "domain_label"]).size().reset_index(name="n_events")
    )
    evidence_matrix = (
        evidence_counts.pivot(index="agent_role", columns="domain_label", values="n_events")
        .reindex(ROLE_ORDER)
        .fillna(0)
    )

    transition_counts = (
        chains.groupby(["upstream_role", "downstream_role"]).size().reset_index(name="n_chains")
    )
    transition_matrix = (
        transition_counts.pivot(index="upstream_role", columns="downstream_role", values="n_chains")
        .reindex(index=ROLE_ORDER, columns=ROLE_ORDER)
        .fillna(0)
    )

    rows = []
    for role in ROLE_ORDER:
        ev = events[events["agent_role"] == role]
        incoming = chains[chains["downstream_role"] == role]
        outgoing = chains[chains["upstream_role"] == role]
        incident = len(incoming) + len(outgoing)
        median_in = float(incoming["lag_months_num"].median()) if len(incoming) else np.nan
        rows.append(
            {
                "agent_role": role,
                "agent_label": ROLE_LABELS[role],
                "n_memory_events": len(ev),
                "n_outgoing_chains": len(outgoing),
                "n_incoming_chains": len(incoming),
                "median_incoming_lag_months": round(median_in, 1) if not np.isnan(median_in) else np.nan,
                "memory_weight": len(ev),
                "network_exposure": incident,
            }
        )
    priors = pd.DataFrame(rows)
    priors["memory_weight_norm"] = minmax(priors["memory_weight"])
    priors["network_exposure_norm"] = minmax(priors["network_exposure"])
    max_lag = priors["median_incoming_lag_months"].max(skipna=True)
    priors["delay_constraint_prior"] = (
        priors["median_incoming_lag_months"].fillna(max_lag) / max_lag
    ).clip(0, 1)
    priors = priors.merge(CAPITAL_PRIORS, on="agent_role", how="left")

    role_design_rows = []
    role_counts = events.groupby("agent_role").size().to_dict()
    for role in ROLE_ORDER:
        item = ROLE_DESIGN[role]
        role_design_rows.append(
            {
                "agent_role": ROLE_LABELS[role],
                "field_position": item["field_position"],
                "dominant_capital": item["dominant_capital"],
                "historical_evidence": item["historical_evidence"],
                "n_memory_events": role_counts.get(role, 0),
                "llm_decision_outputs": item["llm_decision_outputs"],
            }
        )
    role_design = pd.DataFrame(role_design_rows)

    memory_examples = []
    for role in ROLE_ORDER:
        sample = (
            events[events["agent_role"] == role]
            .sort_values(["year_final", "event_id"])
            .head(5)[["event_id", "year_final", "domain_label", "event_type", "actor_name", "title"]]
        )
        sample.insert(0, "agent_role", ROLE_LABELS[role])
        memory_examples.append(sample)
    memory_examples = pd.concat(memory_examples, ignore_index=True)
    return {
        "evidence_counts": evidence_counts,
        "evidence_matrix": evidence_matrix,
        "transition_counts": transition_counts,
        "transition_matrix": transition_matrix,
        "agent_behavior_priors": priors,
        "agent_role_design": role_design,
        "agent_memory_examples": memory_examples,
    }


def draw_pipeline(ax) -> None:
    ax.set_axis_off()
    boxes = [
        (0.10, 0.79, 0.76, 0.11, "Historical event chains"),
        (0.10, 0.58, 0.76, 0.11, "Field-theoretic role coding"),
        (0.10, 0.37, 0.76, 0.11, "Agent memory and priors"),
        (0.10, 0.16, 0.76, 0.11, "Structured LLM decision"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.018,rounding_size=0.017",
                facecolor="#EEF2F6",
                edgecolor="#333333",
                linewidth=0.7,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.3)
    for i in range(len(boxes) - 1):
        x, y, w, h, _ = boxes[i]
        nx, ny, nw, nh, _ = boxes[i + 1]
        ax.add_patch(
            FancyArrowPatch(
                (x + w / 2, y - 0.012),
                (nx + nw / 2, ny + nh + 0.012),
                arrowstyle="-|>",
                mutation_scale=8,
                lw=0.9,
                color="#333333",
            )
        )
    ax.text(
        0.06,
        0.015,
        "Field position, capital endowment and observed chains\nconstrain each role before historical replay.",
        fontsize=6.6,
        va="bottom",
    )
    ax.text(-0.04, 1.04, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def make_figure(tables: dict[str, pd.DataFrame]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.5, 7.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.98, 1.02], width_ratios=[0.96, 1.04])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_pipeline(ax_a)

    evidence = tables["evidence_matrix"].rename(index=ROLE_LABELS)
    sns.heatmap(
        evidence,
        ax=ax_b,
        cmap=sns.light_palette("#4C78A8", as_cmap=True),
        annot=True,
        fmt=".0f",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Historical events"},
        annot_kws={"fontsize": 6.3},
    )
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")
    ax_b.tick_params(axis="x", rotation=10, labelsize=8.0)
    ax_b.tick_params(axis="y", rotation=0)
    ax_b.text(-0.12, 1.04, "b", transform=ax_b.transAxes, fontsize=9, fontweight="bold", va="bottom")

    trans = tables["transition_matrix"].rename(index=ROLE_LABELS, columns=ROLE_LABELS)
    trans = trans.rename(index=SHORT_ROLE_LABELS, columns=SHORT_ROLE_LABELS)
    sns.heatmap(
        trans,
        ax=ax_c,
        cmap=sns.light_palette("#F58518", as_cmap=True),
        annot=True,
        fmt=".0f",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Observed event chains"},
        annot_kws={"fontsize": 6.0},
    )
    ax_c.set_xlabel("Downstream agent")
    ax_c.set_ylabel("Upstream agent")
    ax_c.tick_params(axis="x", rotation=30, labelsize=7.2)
    for label in ax_c.get_xticklabels():
        label.set_horizontalalignment("right")
    ax_c.tick_params(axis="y", rotation=0)
    ax_c.text(-0.12, 1.04, "c", transform=ax_c.transAxes, fontsize=9, fontweight="bold", va="bottom")

    priors = tables["agent_behavior_priors"].set_index("agent_label")
    cols = [
        "regulatory",
        "economic",
        "pedagogical",
        "translation",
        "symbolic",
        "network",
        "delay_constraint_prior",
    ]
    prior_matrix = priors[cols].rename(index=SHORT_ROLE_LABELS)
    sns.heatmap(
        prior_matrix,
        ax=ax_d,
        cmap=sns.light_palette("#54A24B", as_cmap=True),
        vmin=0,
        vmax=1,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Prior strength"},
        annot=False,
        fmt=".2f",
    )
    ax_d.set_xlabel("")
    ax_d.set_ylabel("")
    ax_d.set_xticklabels(
        ["regulatory", "economic", "pedagogical", "translation", "symbolic", "network", "delay"],
        rotation=30,
        ha="right",
    )
    ax_d.tick_params(axis="x", labelsize=7.0)
    ax_d.tick_params(axis="y", rotation=0)
    ax_d.text(-0.12, 1.04, "d", transform=ax_d.transAxes, fontsize=9, fontweight="bold", va="bottom")

    for ext in ["png", "svg", "pdf", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG_DIR / f"fig3_agent_conditioning.{ext}", **kwargs)
    plt.close(fig)


def write_outputs(tables: dict[str, pd.DataFrame]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name.endswith("_matrix"):
            out = table.rename(index=ROLE_LABELS)
        else:
            out = table.copy()
        out.to_csv(TABLE_DIR / f"{name}.csv", encoding="utf-8-sig")
        out.to_markdown(TABLE_DIR / f"{name}.md", index=True)

    latex_parts = ["% Table 3a: Field-theoretic role design"]
    try:
        latex_parts.append(tables["agent_role_design"].to_latex(index=False, escape=True))
        latex_parts.append("\n% Table 3b: Agent behaviour priors")
        latex_parts.append(tables["agent_behavior_priors"][["agent_label", "n_memory_events", "n_outgoing_chains", "n_incoming_chains", "median_incoming_lag_months", "memory_weight_norm", "network_exposure_norm", "delay_constraint_prior"]].to_latex(index=False, escape=True))
    except ImportError:
        latex_parts.append("% LaTeX table export skipped: optional jinja2 dependency is unavailable.")
    (OUT_DIR / "results3_tables_latex.tex").write_text("\n".join(latex_parts), encoding="utf-8")

    caption = """Figure 3 | Historical event chains condition role-specific agent behaviour.
a, Workflow by which observed event chains are transformed into field-theoretic agent memories and structured LLM decision inputs.
b, Historical event evidence available for each agent role across technological domains.
c, Observed inter-agent event-chain transitions used to condition role-to-role interaction priors.
d, Field-theoretic capital endowments and delay priors assigned to each role before historical replay. These priors constrain the LLM agents' structured decisions rather than allowing unconstrained narrative generation.
"""
    (OUT_DIR / "fig3_caption.txt").write_text(caption, encoding="utf-8")

    json_schema = """{
  "agent_role": "school",
  "field_position": "training institution",
  "incoming_signal": "firm demand or policy signal",
  "perceived_urgency": 0.0,
  "perceived_feasibility": 0.0,
  "dominant_constraint": "curriculum approval, faculty capacity, demand uncertainty, funding, or coordination failure",
  "decision": "respond_now | delay | request_coordination | ignore",
  "expected_delay_months": 0,
  "coordination_request": ["firm", "intermediary"],
  "latent_reason": "brief mechanism-level explanation",
  "confidence": 0.0
}"""
    (OUT_DIR / "agent_decision_json_schema.txt").write_text(json_schema, encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events, chains = load_data()
    tables = build_tables(events, chains)
    write_outputs(tables)
    make_figure(tables)
    print(f"Results 3 outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
