from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from case_behavior_granularity import MAP_DF, add_behavior_columns, infer_case_behavior
from case_node_config import CASE_NODE_CONFIG, TARGET_EVENT_IDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_DIR = PROJECT_ROOT / "outputs" / "event_analysis"
RAW_REPLAY_DIR = PROJECT_ROOT / "outputs" / "results4_llm_historical_replay"
HYBRID_REPLAY_DIR = PROJECT_ROOT / "outputs" / "results4_llm_historical_replay_v3_hybrid"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "results4_case_appendix"

SECTOR_ORDER = ["AI", "NEV", "robotics", "eldercare"]
CHAIN_ORDER = [
    "policy_to_firm",
    "firm_to_school",
    "policy_to_school",
    "standard_to_curriculum",
]

CHAIN_LABELS = {
    "policy_to_firm": "Policy-to-firm",
    "firm_to_school": "Firm-to-school",
    "policy_to_school": "Policy-to-school",
    "standard_to_curriculum": "Standard-to-curriculum",
}

CHAIN_STAGE_TEMPLATES = {
    "policy_to_firm": [
        ("signal", "上游信号", "政策发文", "trigger"),
        ("interpretation", "政策解读", "政策翻译", "policy_translation"),
        ("coordination", "政企协调", "校企对接", "coordination"),
        ("approval", "资源审批", "经费/项目审批", "approval"),
        ("response", "院校/企业响应", "合作落地", "response"),
    ],
    "firm_to_school": [
        ("signal", "上游信号", "企业用工需求", "trigger"),
        ("interpretation", "需求翻译", "岗位信号识别", "policy_translation"),
        ("coordination", "校企协调", "合作磋商", "coordination"),
        ("approval", "课程调整", "专业/课标审批", "curriculum_approval"),
        ("response", "院校响应", "课程或专业落地", "response"),
    ],
    "policy_to_school": [
        ("signal", "上游信号", "产业扶持政策", "trigger"),
        ("interpretation", "政策解读", "政策翻译", "policy_translation"),
        ("coordination", "政校协调", "跨部门协同", "coordination"),
        ("approval", "专业申报", "课标转化 / 审批", "curriculum_approval"),
        ("response", "学校落地", "新增专业 / 更新课程", "response"),
    ],
    "standard_to_curriculum": [
        ("signal", "上游信号", "行业新标准", "trigger"),
        ("interpretation", "标准解读", "标准翻译", "standard_interpretation"),
        ("coordination", "标准协同", "校企/行业协同", "coordination"),
        ("approval", "课标修订", "标准教学化", "curriculum_approval"),
        ("response", "课程落地", "课程更新 / 设备配套", "response"),
    ],
}

STAGE_WEIGHT_BASE = np.array([0.12, 0.22, 0.26, 0.22, 0.18], dtype=float)

SECTOR_LABELS = {
    "AI": "AI",
    "NEV": "NEV",
    "robotics": "Robotics",
    "eldercare": "Eldercare",
}

SECTOR_COLORS = {
    "AI": "#4C78A8",
    "NEV": "#54A24B",
    "robotics": "#F58518",
    "eldercare": "#B279A2",
}


@dataclass(frozen=True)
class SelectedCase:
    case_id: str
    sector: str
    chain_type_norm: str
    upstream_event_id: str
    downstream_event_id: str
    upstream_title: str
    downstream_title: str
    upstream_actor: str
    downstream_role: str
    observed_lag_months: float
    simulated_lag_months: float
    dominant_constraint: str
    action_type: str
    decision: str
    perceived_urgency: float
    perceived_feasibility: float
    confidence: float
    coordination_request: list[str]
    explanation: str
    latent_reason: str
    raw_output: str
    model_name: str
    model_label: str
    replay_mode: str


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
        }
    )


def load_frame() -> pd.DataFrame:
    hybrid_path = HYBRID_REPLAY_DIR / "llm_replay_results_hybrid_calibrated.csv"
    raw_path = RAW_REPLAY_DIR / "llm_replay_results.csv"

    raw_is_targeted = False
    if raw_path.exists():
        try:
            raw_ids = set(pd.read_csv(raw_path, usecols=["upstream_event_id", "downstream_event_id"]).astype(str).stack().tolist())
            raw_is_targeted = set(TARGET_EVENT_IDS).issubset(raw_ids)
        except Exception:
            raw_is_targeted = False

    if raw_is_targeted:
        df = pd.read_csv(raw_path)
        source_label = "raw_targeted"
    elif hybrid_path.exists():
        df = pd.read_csv(hybrid_path)
        source_label = "hybrid"
    elif raw_path.exists():
        df = pd.read_csv(raw_path)
        source_label = "raw"
    else:
        raise FileNotFoundError(
            "Missing replay results. Expected either "
            f"{hybrid_path} or {raw_path}"
        )

    if raw_path.exists() and source_label == "hybrid":
        raw = pd.read_csv(raw_path)
        join_cols = [col for col in ["chain_id", "episode_index", "upstream_event_id", "downstream_event_id"] if col in df.columns and col in raw.columns]
        if join_cols:
            raw_cols = [c for c in ["explanation", "latent_reason", "raw_output", "prompt_text", "memory_examples_json", "calibrated_prompt"] if c in raw.columns]
            raw_small = raw[join_cols + raw_cols].drop_duplicates(subset=join_cols)
            df = df.merge(raw_small, on=join_cols, how="left", suffixes=("", "_raw"))

            for col in raw_cols:
                raw_col = f"{col}_raw"
                if raw_col in df.columns:
                    if col in df.columns:
                        df[col] = df[col].where(df[col].notna(), df[raw_col])
                    else:
                        df[col] = df[raw_col]
                    df.drop(columns=[raw_col], inplace=True)

    if "simulated_lag_months" in df.columns:
        sim_col = "simulated_lag_months"
    elif "hybrid_sim_delay_months" in df.columns:
        sim_col = "hybrid_sim_delay_months"
    else:
        sim_col = "sim_delay_months"

    df["observed_lag_months"] = pd.to_numeric(df.get("lag_months_num"), errors="coerce")
    df["simulated_lag_months"] = pd.to_numeric(df.get(sim_col), errors="coerce")
    df["sector"] = df.get("sector", "").astype(str).str.strip()
    df["chain_type_norm"] = df.get("chain_type_norm", "").astype(str).str.strip()
    df["downstream_role"] = df.get("downstream_role", "").astype(str).str.strip()
    df["dominant_constraint"] = df.get("dominant_constraint", "").astype(str).str.strip()
    df["action_type"] = df.get("action_type", "").astype(str).str.strip()
    df["decision"] = df.get("decision", "").astype(str).str.strip()
    df["model_name"] = df.get("model_name", "").astype(str).str.strip()
    df["model_label"] = df.get("model_label", "").astype(str).str.strip()
    if "replay_mode" in df.columns:
        df["replay_mode"] = df["replay_mode"].astype(str).str.strip()
    else:
        df["replay_mode"] = source_label
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce")
    df["perceived_urgency"] = pd.to_numeric(df.get("perceived_urgency"), errors="coerce")
    df["perceived_feasibility"] = pd.to_numeric(df.get("perceived_feasibility"), errors="coerce")
    df["parse_error"] = df.get("parse_error", False).fillna(False).astype(bool)
    return df


def load_events_and_chains() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(EVENT_DIR / "events_all.csv")
    chains = pd.read_csv(EVENT_DIR / "chains_all.csv")
    return events, chains


def normalize_coordination_request(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = text
    if isinstance(parsed, (list, tuple, set)):
        return [str(v).strip() for v in parsed if str(v).strip()]
    return [str(parsed).strip()]


def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return default
    return text


def wrap(text: str, width: int) -> str:
    text = safe_text(text)
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def file_slug(text: str) -> str:
    text = safe_text(text)
    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    slug = "".join(safe_chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "item"


def quantile_pick(group: pd.DataFrame, n: int) -> pd.Index:
    if n <= 0 or group.empty:
        return group.index[:0]

    sort_col = "simulated_lag_months" if group["simulated_lag_months"].notna().any() else "observed_lag_months"
    ordered = group.sort_values([sort_col, "chain_type_norm", "chain_id"], kind="mergesort").copy()
    q_values = np.linspace(0.2, 0.8, n) if n > 1 else np.array([0.5])
    positions = []
    for q in q_values:
        pos = int(round((len(ordered) - 1) * float(q)))
        positions.append(max(0, min(len(ordered) - 1, pos)))

    chosen = list(dict.fromkeys(ordered.iloc[positions].index.tolist()))
    if len(chosen) >= n:
        return pd.Index(chosen[:n])

    remaining = ordered.drop(index=chosen, errors="ignore")
    if remaining.empty:
        return pd.Index(chosen)

    center = float(np.nanmedian(ordered[sort_col].to_numpy(dtype=float)))

    def score_row(row: pd.Series) -> tuple[float, float, float]:
        sim = row.get(sort_col, np.nan)
        sim = center if pd.isna(sim) else float(sim)
        gap = abs(sim - center)
        chain_penalty = 0.0 if row.get("chain_type_norm", "") not in ordered.loc[chosen, "chain_type_norm"].tolist() else 0.25
        return (gap, -float(row.get("confidence", 0.0) or 0.0), chain_penalty)

    extra = sorted(remaining.index, key=lambda idx: score_row(remaining.loc[idx]))
    for idx in extra:
        chosen.append(idx)
        if len(chosen) >= n:
            break
    return pd.Index(chosen[:n])


def allocate_counts(total: int, sectors: list[str]) -> dict[str, int]:
    sectors = [s for s in sectors if s]
    if not sectors:
        return {}
    if total <= 0:
        return {s: 0 for s in sectors}
    if total < len(sectors):
        return {s: 1 if i < total else 0 for i, s in enumerate(sectors)}
    base = total // len(sectors)
    remainder = total % len(sectors)
    counts = {s: base for s in sectors}
    for sector in sectors[:remainder]:
        counts[sector] += 1
    for sector in sectors:
        if counts[sector] == 0:
            counts[sector] = 1
    return counts


def select_cases(df: pd.DataFrame, total_cases: int, sectors: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = allocate_counts(total_cases, sectors)
    chosen_rows: list[pd.DataFrame] = []
    used_chain_ids: set[str] = set()

    for sector in sectors:
        target_n = counts.get(sector, 0)
        if target_n <= 0:
            continue
        sub = df[(df["sector"] == sector) & (df["parse_error"] == False)].copy()  # noqa: E712
        if sub.empty:
            continue
        # Prefer diverse pathways within each sector.
        picked_idx: list[int] = []
        sector_shift = sectors.index(sector) % len(CHAIN_ORDER)
        preferred_chain_order = CHAIN_ORDER[sector_shift:] + CHAIN_ORDER[:sector_shift]
        for chain_type in preferred_chain_order:
            if len(picked_idx) >= target_n:
                break
            part = sub[sub["chain_type_norm"] == chain_type]
            if part.empty:
                continue
            pick = quantile_pick(part, 1)
            if len(pick):
                idx = int(pick[0])
                if sub.loc[idx, "chain_id"] not in used_chain_ids:
                    picked_idx.append(idx)
                    used_chain_ids.add(str(sub.loc[idx, "chain_id"]))

        if len(picked_idx) < target_n:
            remaining = sub.drop(index=picked_idx, errors="ignore")
            if not remaining.empty:
                # Use quantile picks to provide short/long contrast.
                pick = quantile_pick(remaining, target_n - len(picked_idx))
                for idx in pick.tolist():
                    cid = str(sub.loc[idx, "chain_id"])
                    if cid not in used_chain_ids:
                        picked_idx.append(int(idx))
                        used_chain_ids.add(cid)
                        if len(picked_idx) >= target_n:
                            break

        if len(picked_idx) < target_n:
            remaining = sub[~sub["chain_id"].astype(str).isin(used_chain_ids)].copy()
            if not remaining.empty:
                remaining = remaining.sample(n=min(target_n - len(picked_idx), len(remaining)), random_state=int(rng.integers(0, 1_000_000)))
                for idx, row in remaining.iterrows():
                    picked_idx.append(int(idx))
                    used_chain_ids.add(str(row["chain_id"]))

        if picked_idx:
            chosen_rows.append(sub.loc[picked_idx].copy())

    combined = pd.concat(chosen_rows, ignore_index=False) if chosen_rows else pd.DataFrame()
    if combined.empty:
        raise RuntimeError("No cases selected. Check the available sectors and chain types.")

    # Fill up to total_cases if necessary with the best remaining records.
    combined = combined.drop_duplicates(subset=["chain_id"]).copy()
    if len(combined) < total_cases:
        remaining = df[(df["parse_error"] == False) & (~df["chain_id"].astype(str).isin(set(combined["chain_id"].astype(str))))].copy()  # noqa: E712
        if not remaining.empty:
            # Keep sector balance while using the most informative records next.
            remaining["sector_rank"] = remaining["sector"].map({s: i for i, s in enumerate(sectors)}).fillna(len(sectors))
            remaining = remaining.sort_values(["sector_rank", "simulated_lag_months", "chain_id"], kind="mergesort")
            extra = remaining.head(total_cases - len(combined))
            combined = pd.concat([combined, extra], ignore_index=False)

    combined = combined.sort_values(["sector", "chain_type_norm", "simulated_lag_months", "chain_id"], kind="mergesort")
    combined = combined.head(total_cases).copy()
    combined["case_rank"] = np.arange(1, len(combined) + 1)
    combined["case_id"] = [f"case_{i:02d}" for i in combined["case_rank"]]
    return combined.reset_index(drop=True)


def select_configured_cases(df: pd.DataFrame, total_cases: int) -> pd.DataFrame:
    """Select the fixed 12 case nodes without changing the replay model."""
    if total_cases != len(CASE_NODE_CONFIG):
        return pd.DataFrame()

    rows = []
    used_chain_ids = set()
    for item in CASE_NODE_CONFIG:
        target = str(item["source_event_id"])
        preferred = str(item["preferred_chain_type"])
        matches = df[
            (df["parse_error"] == False)  # noqa: E712
            & (
                (df["upstream_event_id"].astype(str) == target)
                | (df["downstream_event_id"].astype(str) == target)
            )
        ].copy()
        matches["target_is_upstream"] = matches["upstream_event_id"].astype(str) == target
        matches["preferred"] = matches["chain_type_norm"].astype(str) == preferred
        matches = matches[~matches["chain_id"].astype(str).isin(used_chain_ids)]
        if matches.empty:
            raise RuntimeError(f"Configured case event is absent from replay results: {target}")
        matches = matches.sort_values(
            ["preferred", "target_is_upstream", "confidence", "chain_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        row = matches.iloc[0].copy()
        row["case_id"] = item["case_id"]
        row["case_node_position"] = item["position"]
        row["case_node_label"] = item["node_label"]
        row["source_event_id"] = target
        rows.append(row)
        used_chain_ids.add(str(row["chain_id"]))

    return pd.DataFrame(rows).reset_index(drop=True)


def choose_representative_cases(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sector_order = {sector: i for i, sector in enumerate(SECTOR_ORDER)}
    for sector in SECTOR_ORDER:
        sub = cases[cases["sector"] == sector].copy()
        if sub.empty:
            continue
        target = float(sub["observed_lag_months"].median())
        sub["median_gap"] = (sub["observed_lag_months"] - target).abs()
        sub["confidence_rank"] = pd.to_numeric(sub.get("confidence"), errors="coerce").fillna(-1.0)
        sub = sub.sort_values(
            ["median_gap", "confidence_rank", "case_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        rows.append(sub.head(1))
    if not rows:
        raise RuntimeError("Unable to choose representative cases from an empty case set.")
    main_cases = pd.concat(rows, ignore_index=True)
    main_cases["sector_rank"] = main_cases["sector"].map(sector_order).fillna(len(sector_order))
    main_cases = main_cases.sort_values(["sector_rank", "case_id"], kind="mergesort").drop(columns=["sector_rank"])
    supplement_cases = cases[~cases["case_id"].astype(str).isin(main_cases["case_id"].astype(str))].copy()
    supplement_cases = supplement_cases.sort_values(["sector", "chain_type_norm", "observed_lag_months", "case_id"], kind="mergesort")
    main_cases = main_cases.reset_index(drop=True)
    supplement_cases = supplement_cases.reset_index(drop=True)
    main_cases["role"] = "main"
    supplement_cases["role"] = "supplement"
    return pd.concat([main_cases, supplement_cases], ignore_index=True)


def merge_event_details(cases: pd.DataFrame, events: pd.DataFrame, chains: pd.DataFrame) -> pd.DataFrame:
    event_fields = [
        "event_id",
        "event_date",
        "year_final",
        "sector",
        "event_type",
        "actor_type",
        "actor_name",
        "action_type",
        "title",
        "description",
        "evidence_text",
    ]
    ev = events[event_fields].copy()
    ev = ev.rename(
        columns={
            "event_id": "event_id",
            "event_date": "event_date",
            "year_final": "event_year",
            "sector": "event_sector",
            "event_type": "event_type",
            "actor_type": "actor_type",
            "actor_name": "actor_name",
            "action_type": "event_action_type",
            "title": "title",
            "description": "description",
            "evidence_text": "evidence_text",
        }
    )

    up = ev.add_prefix("up_")
    down = ev.add_prefix("down_")
    merged = cases.merge(up, left_on="upstream_event_id", right_on="up_event_id", how="left")
    merged = merged.merge(down, left_on="downstream_event_id", right_on="down_event_id", how="left")
    merged = merged.merge(
        chains[["chain_id", "chain_type_norm", "lag_months_num"]].drop_duplicates("chain_id"),
        on=["chain_id", "chain_type_norm"],
        how="left",
        suffixes=("", "_chain"),
    )
    return merged


def build_selected_cases(df: pd.DataFrame, events: pd.DataFrame, chains: pd.DataFrame, total_cases: int, sectors: list[str], seed: int) -> pd.DataFrame:
    selected = select_configured_cases(df, total_cases=total_cases)
    if selected.empty:
        selected = select_cases(df, total_cases=total_cases, sectors=sectors, seed=seed)
    selected = merge_event_details(selected, events, chains)
    selected["coordination_request"] = selected["coordination_request"].apply(normalize_coordination_request)
    selected["sector_label"] = selected["sector"].map(SECTOR_LABELS).fillna(selected["sector"])
    selected["pathway_label"] = selected["chain_type_norm"].map(CHAIN_LABELS).fillna(selected["chain_type_norm"])
    selected["observed_lag_months"] = pd.to_numeric(selected["observed_lag_months"], errors="coerce")
    selected["simulated_lag_months"] = pd.to_numeric(selected["simulated_lag_months"], errors="coerce")
    selected["confidence"] = pd.to_numeric(selected["confidence"], errors="coerce")
    selected["perceived_urgency"] = pd.to_numeric(selected["perceived_urgency"], errors="coerce")
    selected["perceived_feasibility"] = pd.to_numeric(selected["perceived_feasibility"], errors="coerce")
    return selected


def case_to_record(row: pd.Series) -> SelectedCase:
    return SelectedCase(
        case_id=safe_text(row.get("case_id")),
        sector=safe_text(row.get("sector")),
        chain_type_norm=safe_text(row.get("chain_type_norm")),
        upstream_event_id=safe_text(row.get("upstream_event_id")),
        downstream_event_id=safe_text(row.get("downstream_event_id")),
        upstream_title=safe_text(row.get("up_title")),
        downstream_title=safe_text(row.get("down_title")),
        upstream_actor=safe_text(row.get("up_actor_name")),
        downstream_role=safe_text(row.get("downstream_role")),
        observed_lag_months=float(row.get("observed_lag_months", np.nan)) if pd.notna(row.get("observed_lag_months", np.nan)) else float("nan"),
        simulated_lag_months=float(row.get("simulated_lag_months", np.nan)) if pd.notna(row.get("simulated_lag_months", np.nan)) else float("nan"),
        dominant_constraint=safe_text(row.get("dominant_constraint")),
        action_type=safe_text(row.get("action_type")),
        decision=safe_text(row.get("decision")),
        perceived_urgency=float(row.get("perceived_urgency", np.nan)) if pd.notna(row.get("perceived_urgency", np.nan)) else float("nan"),
        perceived_feasibility=float(row.get("perceived_feasibility", np.nan)) if pd.notna(row.get("perceived_feasibility", np.nan)) else float("nan"),
        confidence=float(row.get("confidence", np.nan)) if pd.notna(row.get("confidence", np.nan)) else float("nan"),
        coordination_request=normalize_coordination_request(row.get("coordination_request")),
        explanation=safe_text(row.get("explanation")),
        latent_reason=safe_text(row.get("latent_reason")),
        raw_output=safe_text(row.get("raw_output")),
        model_name=safe_text(row.get("model_name")),
        model_label=safe_text(row.get("model_label")),
        replay_mode=safe_text(row.get("replay_mode")),
    )


def case_story(case: SelectedCase) -> str:
    coord = ", ".join(case.coordination_request) if case.coordination_request else "none"
    lines = [
        f"# {case.case_id}",
        "",
        f"- Sector: {case.sector}",
        f"- Pathway: {case.chain_type_norm}",
        f"- Upstream event: {case.upstream_title} ({case.upstream_event_id})",
        f"- Downstream event: {case.downstream_title} ({case.downstream_event_id})",
        f"- Observed lag: {case.observed_lag_months:.1f} months" if np.isfinite(case.observed_lag_months) else "- Observed lag: NA",
        f"- Simulated lag: {case.simulated_lag_months:.1f} months" if np.isfinite(case.simulated_lag_months) else "- Simulated lag: NA",
        f"- Dominant constraint: {case.dominant_constraint or 'NA'}",
        f"- Action type: {case.action_type or 'NA'}",
        f"- Decision: {case.decision or 'NA'}",
        f"- Perceived urgency: {case.perceived_urgency:.2f}" if np.isfinite(case.perceived_urgency) else "- Perceived urgency: NA",
        f"- Perceived feasibility: {case.perceived_feasibility:.2f}" if np.isfinite(case.perceived_feasibility) else "- Perceived feasibility: NA",
        f"- Confidence: {case.confidence:.2f}" if np.isfinite(case.confidence) else "- Confidence: NA",
        f"- Coordination request: {coord}",
        "",
        "## Causal chain",
        "",
        f"Input conditions -> {case.upstream_title or case.upstream_event_id}",
        f"Field interpretation -> {case.dominant_constraint or 'mechanism not stated'}",
        f"Behavioral response -> {case.decision or case.action_type or 'response'}",
        f"Outcome -> {case.simulated_lag_months:.1f} months simulated lag" if np.isfinite(case.simulated_lag_months) else "Outcome -> simulated lag NA",
        "",
        "## Explanation",
        "",
        case.explanation or case.latent_reason or "No explanation text available in the source file.",
    ]
    return "\n".join(lines).strip() + "\n"


def save_figure_triplet(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight", dpi=500)
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")


def split_text_to_lines(text: str, width: int) -> str:
    return wrap(text, width) if text else ""


def _pick_bottleneck_stage(row: pd.Series) -> str:
    text = " ".join(
        [
            safe_text(row.get("dominant_constraint")),
            safe_text(row.get("action_type")),
            safe_text(row.get("decision")),
            " ".join(normalize_coordination_request(row.get("coordination_request"))),
        ]
    ).lower()
    if any(key in text for key in ["curriculum", "approval", "standard"]):
        return "approval"
    if any(key in text for key in ["coordination", "collaboration", "tie", "connect"]):
        return "coordination"
    if any(key in text for key in ["policy", "translation", "interpret"]):
        return "interpretation"
    return "coordination"


def _bottleneck_label(stage_key: str, dominant_constraint: str) -> str:
    dom = safe_text(dominant_constraint).lower()
    if stage_key == "signal":
        return "触发信号"
    if stage_key == "interpretation":
        if "standard" in dom:
            return "标准教学化困难"
        return "政策解读难"
    if stage_key == "coordination":
        return "校企协调成本高"
    if stage_key == "approval":
        if "standard" in dom:
            return "标准转课标困难"
        if "curriculum" in dom or "approval" in dom:
            return "课程审批周期长"
        return "流程审批滞后"
    if stage_key == "response":
        return "学校落地响应"
    return "机制节点"


def _allocate_stage_durations(total_months: float, bottleneck_stage: str) -> list[float]:
    if not np.isfinite(total_months) or total_months <= 0:
        return [np.nan] * len(STAGE_WEIGHT_BASE)
    weights = STAGE_WEIGHT_BASE.copy()
    stage_index = {stage_key: idx for idx, (stage_key, *_rest) in enumerate(CHAIN_STAGE_TEMPLATES["policy_to_school"])}.get(bottleneck_stage, 2)
    weights[stage_index] *= 1.65
    weights = weights / weights.sum()
    raw = weights * float(total_months)
    rounded = np.round(raw, 1)
    if len(rounded):
        rounded[-1] = np.round(float(total_months) - float(np.sum(rounded[:-1])), 1)
    return rounded.tolist()


def _format_case_date(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return "NA"
    try:
        dt = pd.to_datetime(text, errors="coerce")
    except Exception:
        dt = pd.NaT
    if pd.isna(dt):
        return text
    return dt.strftime("%Y-%m")


def build_case_timeline(row: pd.Series) -> list[dict[str, Any]]:
    chain_type = safe_text(row.get("chain_type_norm"))
    template = CHAIN_STAGE_TEMPLATES.get(chain_type, CHAIN_STAGE_TEMPLATES["policy_to_school"])
    bottleneck_stage = _pick_bottleneck_stage(row)
    observed_total = float(row.get("observed_lag_months", np.nan))
    simulated_total = float(row.get("simulated_lag_months", np.nan))
    observed_parts = _allocate_stage_durations(observed_total, bottleneck_stage)
    simulated_parts = _allocate_stage_durations(simulated_total, bottleneck_stage)

    signal_date = _format_case_date(row.get("up_event_date"))
    response_date = _format_case_date(row.get("down_event_date"))
    dominant_constraint = safe_text(row.get("dominant_constraint"), "mechanism")
    behavior = infer_case_behavior(row)
    coord_text = " / ".join(normalize_coordination_request(row.get("coordination_request"))[:3]) or "none"

    timeline: list[dict[str, Any]] = []
    cum_obs = 0.0
    cum_sim = 0.0
    for idx, (stage_key, stage_label, node_text, bottleneck_key) in enumerate(template, start=1):
        obs_part = observed_parts[idx - 1] if idx - 1 < len(observed_parts) else np.nan
        sim_part = simulated_parts[idx - 1] if idx - 1 < len(simulated_parts) else np.nan
        cum_obs = float(cum_obs + obs_part) if np.isfinite(obs_part) else np.nan
        cum_sim = float(cum_sim + sim_part) if np.isfinite(sim_part) else np.nan
        if stage_key == "signal":
            detail = textwrap.shorten(safe_text(row.get("up_title"), safe_text(row.get("upstream_event_id"))), width=20, placeholder="…")
        elif stage_key == "interpretation":
            detail = textwrap.shorten(dominant_constraint, width=20, placeholder="…")
        elif stage_key == "coordination":
            detail = "校企/政校协同"
        elif stage_key == "approval":
            detail = textwrap.shorten(safe_text(row.get("decision"), safe_text(row.get("action_type"), "response")), width=20, placeholder="…")
        else:
            detail = textwrap.shorten(safe_text(row.get("down_title"), safe_text(row.get("downstream_event_id"))), width=20, placeholder="…")
        timeline.append(
            {
                "case_id": safe_text(row.get("case_id")),
                "sector": safe_text(row.get("sector")),
                "chain_type_norm": chain_type,
                "pathway_label": safe_text(row.get("pathway_label")),
                "stage_order": idx,
                "stage_key": stage_key,
                "stage_label": stage_label,
                "stage_text": node_text,
                "stage_detail": detail,
                "bottleneck_key": bottleneck_key,
                "bottleneck_label": _bottleneck_label(stage_key, dominant_constraint),
                "is_primary_bottleneck": stage_key == bottleneck_stage,
                "observed_stage_lag_months": obs_part,
                "simulated_stage_lag_months": sim_part,
                "observed_cumulative_months": cum_obs,
                "simulated_cumulative_months": cum_sim,
                "signal_date": signal_date,
                "response_date": response_date,
                "dominant_constraint": dominant_constraint,
                "macro_dimension": behavior["macro_dimension"],
                "macro_label": behavior["macro_label"],
                "field": behavior["field"],
                "behavior_dimension": behavior["behavior_dimension"],
                "behavior_indicator": behavior["behavior_indicator"],
                "case_fine_behavior": behavior["case_fine_behavior"],
                "capital_feature": behavior["capital_feature"],
                "habitus_feature": behavior["habitus_feature"],
                "coordination_request": coord_text,
            }
        )
    return timeline


def case_step_text(row: pd.Series) -> dict[str, str]:
    upstream_title = safe_text(row.get("up_title"), safe_text(row.get("upstream_event_id"), "Upstream event"))
    downstream_title = safe_text(row.get("down_title"), safe_text(row.get("downstream_event_id"), "Downstream event"))
    coord = normalize_coordination_request(row.get("coordination_request"))
    coord_txt = " / ".join(coord[:3]) if coord else "none"
    urgency = row.get("perceived_urgency", np.nan)
    feasibility = row.get("perceived_feasibility", np.nan)
    obs = row.get("observed_lag_months", np.nan)
    sim = row.get("simulated_lag_months", np.nan)
    behavior = infer_case_behavior(row)

    return {
        "intervention": split_text_to_lines(
            f"干预\n{safe_text(row.get('up_event_type'), 'upstream signal')}\n{safe_text(row.get('up_actor_name'), 'actor')}\n{upstream_title}",
            24,
        ),
        "input": split_text_to_lines(
            f"输入\n行业: {safe_text(row.get('sector'))}\n链路: {safe_text(row.get('pathway_label'))}\n下游角色: {safe_text(row.get('downstream_role'), 'downstream')}\n紧迫性: {urgency:.2f}" if pd.notna(urgency) else f"输入\n行业: {safe_text(row.get('sector'))}\n链路: {safe_text(row.get('pathway_label'))}\n下游角色: {safe_text(row.get('downstream_role'), 'downstream')}\n紧迫性: NA",
            24,
        ),
        "process": split_text_to_lines(
            f"中间过程\n宏观: {behavior['macro_label']}\n主体: {behavior['field']}\n细行为: {behavior['case_fine_behavior']}\n协同: {coord_txt}\n可行性: {feasibility:.2f}" if pd.notna(feasibility) else f"中间过程\n宏观: {behavior['macro_label']}\n主体: {behavior['field']}\n细行为: {behavior['case_fine_behavior']}\n协同: {coord_txt}\n可行性: NA",
            24,
        ),
        "behavior": split_text_to_lines(
            f"行为\n{safe_text(row.get('decision'), safe_text(row.get('action_type'), 'response'))}\n{safe_text(row.get('action_type'), 'response')}\n{downstream_title}",
            24,
        ),
        "result": split_text_to_lines(
            f"仿真结果\n仿真时滞: {sim:.1f} 月" if pd.notna(sim) else "仿真结果\n仿真时滞: NA",
            24,
        ),
        "compare": split_text_to_lines(
            f"真实 {obs:.1f} 月 / 仿真 {sim:.1f} 月" if pd.notna(obs) and pd.notna(sim) else "真实/仿真对比 NA",
            30,
        ),
        "flow": split_text_to_lines(
            f"{upstream_title} -> {safe_text(row.get('dominant_constraint'), 'mechanism')} -> {safe_text(row.get('decision'), safe_text(row.get('action_type'), 'response'))} -> {sim:.1f} 月"
            if pd.notna(sim)
            else f"{upstream_title} -> {safe_text(row.get('dominant_constraint'), 'mechanism')} -> {safe_text(row.get('decision'), safe_text(row.get('action_type'), 'response'))} -> NA",
            72,
        ),
    }


def _plot_box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, facecolor: str, edgecolor: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, color="#1b1b1b", linespacing=1.25)


def _plot_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12, linewidth=1.1, color="#555555")
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, label, ha="center", va="bottom", fontsize=6.5, color="#444444")


def _plot_timeline_node(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    detail: str,
    stage_info: str,
    facecolor: str,
    edgecolor: str,
    is_primary: bool = False,
) -> None:
    lw = 1.6 if is_primary else 1.0
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=lw,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center", fontsize=7.8, fontweight="bold", color="#1b1b1b")
    ax.text(x + w / 2, y + h * 0.45, detail, ha="center", va="center", fontsize=6.8, color="#1b1b1b", linespacing=1.25)
    ax.text(x + w / 2, y + h * 0.15, stage_info, ha="center", va="center", fontsize=6.4, color="#555555")


def render_case_schematic(row: pd.Series, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "case_schematics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    step = case_step_text(row)
    sector = safe_text(row.get("sector"))
    case_id = safe_text(row.get("case_id"))
    pathway = safe_text(row.get("pathway_label"))
    color = SECTOR_COLORS.get(sector, "#4C78A8")

    fig, ax = plt.subplots(figsize=(10.8, 4.15), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.945, f"{case_id} | {SECTOR_LABELS.get(sector, sector)} | {pathway}", fontsize=11.0, fontweight="bold", color="#1f1f1f")
    ax.text(0.02, 0.885, split_text_to_lines(safe_text(row.get("up_title"), safe_text(row.get("upstream_event_id"))), 110), fontsize=7.0, color="#444444")

    cards = [
        (0.03, 0.30, 0.22, 0.46, "输入", step["input"], "#EAF2FB"),
        (0.275, 0.30, 0.22, 0.46, "过程", step["process"], "#FBF4E8"),
        (0.52, 0.30, 0.22, 0.46, "行为", step["behavior"], "#F3ECFA"),
        (0.765, 0.30, 0.20, 0.46, "结果", step["result"], "#F8F1FB"),
    ]
    for x, y, w, h, title, text, face in cards:
        _plot_box(ax, x, y, w, h, f"{title}\n\n{text}", face, color if title == "输入" else "#C78A24" if title == "过程" else "#8F5AA8")

    # A light top ribbon links the four cards without making the figure look like a causal diagram.
    behavior = infer_case_behavior(row)
    ax.text(0.03, 0.80, f"案例总览: 输入 - 过程 - 行为 - 结果 | 细颗粒度: {behavior['macro_label']} → {behavior['case_fine_behavior']}", fontsize=7.0, color="#666666")
    ax.text(0.03, 0.16, step["compare"], fontsize=8.6, fontweight="bold", color="#7a2f94")
    ax.text(0.25, 0.16, f"观测时滞 {float(row['observed_lag_months']):.1f} 月", fontsize=8.0, color="#333333")
    ax.text(0.03, 0.07, split_text_to_lines(f"模型: {safe_text(row.get('model_name'))} | 版本: {safe_text(row.get('model_label'))} | 模式: {safe_text(row.get('replay_mode'))}", 120), fontsize=6.8, color="#555555")

    save_figure_triplet(fig, fig_dir / f"{file_slug(case_id)}_{file_slug(sector)}_{file_slug(pathway)}_schematic")
    plt.close(fig)


def render_causal_chain(row: pd.Series, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "causal_chains"
    fig_dir.mkdir(parents=True, exist_ok=True)
    timeline = build_case_timeline(row)
    sector = safe_text(row.get("sector"))
    case_id = safe_text(row.get("case_id"))
    pathway = safe_text(row.get("pathway_label"))
    color = SECTOR_COLORS.get(sector, "#4C78A8")

    fig, ax = plt.subplots(figsize=(12.0, 4.1), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.94, f"因果链路图 | {case_id} | {SECTOR_LABELS.get(sector, sector)} | {pathway}", fontsize=11.0, fontweight="bold")
    ax.text(0.02, 0.875, "时间传导: 上游信号 → 跨场域转化 → 院校落地", fontsize=7.6, color="#555555")

    x_positions = np.linspace(0.03, 0.82, len(timeline))
    widths = [0.16, 0.16, 0.18, 0.18, 0.16]
    facecolors = ["#EEF5FD", "#FBF4E8", "#F9F0E7", "#F3ECFA", "#F8F1FB"]
    edgecolors = [color, "#C78A24", "#D38B16", "#8F5AA8", "#8F5AA8"]

    for i, (x, node, w, face, edge) in enumerate(zip(x_positions, timeline, widths, facecolors, edgecolors)):
        is_primary = bool(node["is_primary_bottleneck"])
        title = f"{node['stage_order']}  {node['stage_label']}"
        detail = split_text_to_lines(f"{node['stage_text']}\n{node['stage_detail']}\n{node['case_fine_behavior']}", 18)
        if pd.notna(node["observed_stage_lag_months"]):
            stage_info = f"真实 {node['observed_stage_lag_months']:.1f} 月 | 模拟 {node['simulated_stage_lag_months']:.1f} 月"
        else:
            stage_info = "时长 NA"
        _plot_timeline_node(
            ax,
            float(x),
            0.38,
            float(w),
            0.27,
            title,
            detail,
            stage_info,
            face,
            edge,
            is_primary=is_primary,
        )
        if i < len(timeline) - 1:
            next_x = float(x_positions[i + 1])
            _plot_arrow(ax, float(x + w), 0.515, next_x - 0.01, 0.515, f"{node['observed_stage_lag_months']:.1f} 月")

    bottleneck = next((node for node in timeline if node["is_primary_bottleneck"]), timeline[1] if len(timeline) > 1 else timeline[0])
    ax.text(
        0.02,
        0.23,
        f"主瓶颈: {bottleneck['stage_label']} / {bottleneck['bottleneck_label']}",
        fontsize=8.2,
        fontweight="bold",
        color="#7a2f94",
    )
    ax.text(
        0.02,
        0.145,
        split_text_to_lines(
            f"信号时间: {timeline[0]['signal_date']}  |  落地时间: {timeline[-1]['response_date']}  |  总滞后: {float(row['observed_lag_months']):.1f} 月  |  模拟滞后: {float(row['simulated_lag_months']):.1f} 月",
            140,
        ),
        fontsize=7.2,
        color="#333333",
    )
    ax.text(
        0.02,
        0.07,
        split_text_to_lines(
            f"可追溯性证据: 宏观={bottleneck['macro_label']}；细行为={bottleneck['case_fine_behavior']}；约束={safe_text(row.get('dominant_constraint'), 'NA')}；协同请求={' / '.join(normalize_coordination_request(row.get('coordination_request'))[:3]) or 'none'}；置信度={row.get('confidence', np.nan):.2f}"
            if pd.notna(row.get("confidence", np.nan))
            else f"可追溯性证据: 宏观={bottleneck['macro_label']}；细行为={bottleneck['case_fine_behavior']}；约束={safe_text(row.get('dominant_constraint'), 'NA')}；协同请求={' / '.join(normalize_coordination_request(row.get('coordination_request'))[:3]) or 'none'}；置信度=NA",
            140,
        ),
        fontsize=6.8,
        color="#555555",
    )

    save_figure_triplet(fig, fig_dir / f"{file_slug(case_id)}_{file_slug(sector)}_{file_slug(pathway)}_causal")
    plt.close(fig)


def build_validation_summary(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["observed_lag_months"].notna() & df["simulated_lag_months"].notna()].copy()
    rows = []
    for chain_type in CHAIN_ORDER:
        sub = valid[valid["chain_type_norm"] == chain_type]
        if sub.empty:
            continue
        obs = sub["observed_lag_months"].to_numpy(dtype=float)
        sim = sub["simulated_lag_months"].to_numpy(dtype=float)
        rows.append(
            {
                "chain_type": chain_type,
                "pathway": CHAIN_LABELS[chain_type],
                "n": len(sub),
                "observed_mean": float(np.mean(obs)),
                "simulated_mean": float(np.mean(sim)),
                "observed_median": float(np.median(obs)),
                "simulated_median": float(np.median(sim)),
                "observed_lt24_pct": float(np.mean(obs < 24) * 100),
                "simulated_lt24_pct": float(np.mean(sim < 24) * 100),
            }
        )
    return pd.DataFrame(rows)


def render_validation_chart(df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "validation"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = build_validation_summary(df)
    if summary.empty:
        raise RuntimeError("Validation summary is empty. Check the replay result file.")

    order = [chain for chain in CHAIN_ORDER if chain in summary["chain_type"].tolist()]
    x = np.arange(len(order))
    width = 0.34
    palette = {"observed": "#4C78A8", "simulated": "#F58518"}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.18, wspace=0.18)
    ax1, ax2 = axes

    med = summary.set_index("chain_type").loc[order]
    ax1.bar(x - width / 2, med["observed_median"], width=width, color=palette["observed"], label="真实数据")
    ax1.bar(x + width / 2, med["simulated_median"], width=width, color=palette["simulated"], label="模拟数据")
    ax1.set_xticks(x)
    ax1.set_xticklabels([CHAIN_LABELS[c] for c in order], rotation=22, ha="right")
    ax1.set_ylabel("中位响应时滞（月）")
    ax1.set_title("A  中位时滞对比", loc="left", fontsize=10, fontweight="bold")
    ax1.legend(frameon=False, fontsize=8)
    for i, row in enumerate(med.itertuples()):
        ax1.text(i - width / 2, row.observed_median + 0.8, f"{row.observed_median:.0f}", ha="center", fontsize=7)
        ax1.text(i + width / 2, row.simulated_median + 0.8, f"{row.simulated_median:.0f}", ha="center", fontsize=7)

    lt24 = summary.set_index("chain_type").loc[order]
    ax2.bar(x - width / 2, lt24["observed_lt24_pct"], width=width, color=palette["observed"])
    ax2.bar(x + width / 2, lt24["simulated_lt24_pct"], width=width, color=palette["simulated"])
    ax2.set_xticks(x)
    ax2.set_xticklabels([CHAIN_LABELS[c] for c in order], rotation=22, ha="right")
    ax2.set_ylabel("24个月内响应比例（%）")
    ax2.set_title("B  24个月内响应比例对比", loc="left", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(100, float(np.nanmax(lt24[["observed_lt24_pct", "simulated_lt24_pct"]].to_numpy())) + 10))
    for i, row in enumerate(lt24.itertuples()):
        ax2.text(i - width / 2, row.observed_lt24_pct + 1.4, f"{row.observed_lt24_pct:.1f}", ha="center", fontsize=7)
        ax2.text(i + width / 2, row.simulated_lt24_pct + 1.4, f"{row.simulated_lt24_pct:.1f}", ha="center", fontsize=7)

    fig.text(0.01, 0.97, "真实数据与模拟数据一致性验证", ha="left", va="top", fontsize=11.5, fontweight="bold")
    save_figure_triplet(fig, fig_dir / "validation_observed_vs_simulated")
    plt.close(fig)


def write_case_timeline_exports(cases: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    table_dir = out_dir / "tables"
    trace_dir = out_dir / "cases" / "decision_traces"
    table_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for _, row in cases.iterrows():
        timeline = build_case_timeline(row)
        trace_payload = {
            "case_id": safe_text(row.get("case_id")),
            "sector": safe_text(row.get("sector")),
            "chain_type_norm": safe_text(row.get("chain_type_norm")),
            "pathway_label": safe_text(row.get("pathway_label")),
            "upstream_event_id": safe_text(row.get("upstream_event_id")),
            "downstream_event_id": safe_text(row.get("downstream_event_id")),
            "observed_lag_months": float(row.get("observed_lag_months", np.nan)) if pd.notna(row.get("observed_lag_months", np.nan)) else None,
            "simulated_lag_months": float(row.get("simulated_lag_months", np.nan)) if pd.notna(row.get("simulated_lag_months", np.nan)) else None,
            "dominant_constraint": safe_text(row.get("dominant_constraint")),
            "action_type": safe_text(row.get("action_type")),
            "decision": safe_text(row.get("decision")),
            "coordination_request": normalize_coordination_request(row.get("coordination_request")),
            "confidence": float(row.get("confidence", np.nan)) if pd.notna(row.get("confidence", np.nan)) else None,
            "explanation": safe_text(row.get("explanation")),
            "latent_reason": safe_text(row.get("latent_reason")),
            "raw_output": safe_text(row.get("raw_output")),
            "timeline": timeline,
        }
        (trace_dir / f"{file_slug(trace_payload['case_id'])}.json").write_text(
            json.dumps(trace_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.extend(timeline)

    long_df = pd.DataFrame(rows)
    if not long_df.empty:
        long_df.to_csv(table_dir / "case_timeline_long.csv", index=False, encoding="utf-8-sig")
        long_df.to_markdown(table_dir / "case_timeline_long.md", index=False)

        summary = (
            long_df.groupby(["case_id", "sector", "chain_type_norm", "pathway_label"], as_index=False)
            .agg(
                signal_date=("signal_date", "first"),
                response_date=("response_date", "first"),
                total_observed_lag=("observed_stage_lag_months", "sum"),
                total_simulated_lag=("simulated_stage_lag_months", "sum"),
                primary_bottleneck_stage=("stage_label", lambda s: s[long_df.loc[s.index, "is_primary_bottleneck"]].iloc[0] if long_df.loc[s.index, "is_primary_bottleneck"].any() else s.iloc[0]),
                primary_bottleneck_label=("bottleneck_label", lambda s: s[long_df.loc[s.index, "is_primary_bottleneck"]].iloc[0] if long_df.loc[s.index, "is_primary_bottleneck"].any() else s.iloc[0]),
            )
        )
        summary.to_csv(table_dir / "case_timeline_summary.csv", index=False, encoding="utf-8-sig")
        summary.to_markdown(table_dir / "case_timeline_summary.md", index=False)
    return long_df


def normalize_mechanism_label(value: Any) -> str:
    text = safe_text(value).lower()
    if not text or text in {"nan", "none", "unknown"}:
        return "Unknown"
    if "policy" in text:
        return "Policy interpretation"
    if "standard" in text or "certification" in text:
        return "Standard consensus"
    if "coordination" in text:
        return "Coordination"
    if "market" in text or "investment" in text:
        return "Market incentive"
    if "curriculum" in text or "approval" in text:
        return "Curriculum approval"
    return "Other"


def build_case_summary(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in cases.iterrows():
        rows.append(
            {
                "case_id": row["case_id"],
                "sector": row["sector"],
                "sector_label": row["sector_label"],
                "chain_type_norm": row["chain_type_norm"],
                "pathway_label": row["pathway_label"],
                "observed_lag_months": float(row["observed_lag_months"]),
                "simulated_lag_months": float(row["simulated_lag_months"]),
                "dominant_constraint": safe_text(row.get("dominant_constraint"), "Unknown"),
                "mechanism_label": normalize_mechanism_label(row.get("dominant_constraint")),
                "confidence": row.get("confidence", np.nan),
            }
        )
    return pd.DataFrame(rows)


def save_plot(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight", dpi=500)
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_s1_grouped_bar(cases: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = build_case_summary(cases)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), constrained_layout=True)
    axes = axes.ravel()
    for ax, sector in zip(axes, SECTOR_ORDER):
        sub = summary[summary["sector"] == sector].copy().reset_index(drop=True)
        if sub.empty:
            ax.axis("off")
            continue
        x = np.arange(len(sub))
        width = 0.34
        ax.bar(x - width / 2, sub["observed_lag_months"], width=width, color="#4C78A8", label="真实")
        ax.bar(x + width / 2, sub["simulated_lag_months"], width=width, color="#F58518", label="模拟")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["case_id"])
        ax.set_title(SECTOR_LABELS.get(sector, sector), loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel("时滞（月）")
        ax.set_ylim(0, max(1, float(np.nanmax(sub[["observed_lag_months", "simulated_lag_months"]].to_numpy())) + 10))
        for i, row in enumerate(sub.itertuples()):
            ax.text(i - width / 2, row.observed_lag_months + 0.7, f"{row.observed_lag_months:.0f}", ha="center", fontsize=7)
            ax.text(i + width / 2, row.simulated_lag_months + 0.7, f"{row.simulated_lag_months:.0f}", ha="center", fontsize=7)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.01, 0.98, "S1 真实数据与模拟数据的分组柱状对比", ha="left", va="top", fontsize=11.2, fontweight="bold")
    save_plot(fig, fig_dir / "S1_grouped_bar_case_level")


def render_s2_scatter(cases: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = build_case_summary(cases)
    fig, ax = plt.subplots(figsize=(7.8, 6.0), constrained_layout=True)
    for sector in SECTOR_ORDER:
        sub = summary[summary["sector"] == sector]
        if sub.empty:
            continue
        ax.scatter(
            sub["observed_lag_months"],
            sub["simulated_lag_months"],
            s=62,
            color=SECTOR_COLORS.get(sector, "#4C78A8"),
            label=SECTOR_LABELS.get(sector, sector),
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
        for _, row in sub.iterrows():
            ax.text(row["observed_lag_months"] + 0.35, row["simulated_lag_months"] + 0.35, row["case_id"], fontsize=6.5)
    lim = float(np.nanmax(summary[["observed_lag_months", "simulated_lag_months"]].to_numpy()))
    lim = max(1, lim + 4)
    ax.plot([0, lim], [0, lim], ls="--", lw=1.0, color="#666666")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("真实时滞（月）")
    ax.set_ylabel("模拟时滞（月）")
    ax.set_title("S2 真实数据与模拟数据的一致性散点图", loc="left", fontsize=11.2, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save_plot(fig, fig_dir / "S2_scatter_consistency")


def render_s3_distribution(cases: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = build_case_summary(cases)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    for ax, col, label in zip(
        axes,
        ["observed_lag_months", "simulated_lag_months"],
        ["真实时滞分布", "模拟时滞分布"],
    ):
        for sector in SECTOR_ORDER:
            sub = summary[summary["sector"] == sector]
            if sub.empty:
                continue
            x = np.full(len(sub), SECTOR_ORDER.index(sector))
            ax.scatter(
                x + np.linspace(-0.12, 0.12, len(sub)),
                sub[col],
                color=SECTOR_COLORS.get(sector, "#4C78A8"),
                s=40,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
        for i, sector in enumerate(SECTOR_ORDER):
            sub = summary[summary["sector"] == sector]
            if sub.empty:
                continue
            vals = sub[col].to_numpy(dtype=float)
            ax.vlines(i, np.min(vals), np.max(vals), color="#555555", lw=1.1, alpha=0.8)
            ax.scatter(i, np.mean(vals), marker="D", s=38, color="#222222", zorder=4)
        ax.set_xticks(np.arange(len(SECTOR_ORDER)))
        ax.set_xticklabels([SECTOR_LABELS[s] for s in SECTOR_ORDER], rotation=18)
        ax.set_ylabel("时滞（月）")
        ax.set_title(label, loc="left", fontsize=10.8, fontweight="bold")
    fig.suptitle("S3 各产业响应时滞分布图", x=0.01, y=1.02, ha="left", fontsize=11.2, fontweight="bold")
    save_plot(fig, fig_dir / "S3_lag_distribution")


def render_s4_mechanism_stacked(cases: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = build_case_summary(cases)
    mech_order = ["Policy interpretation", "Standard consensus", "Coordination", "Market incentive", "Curriculum approval", "Other", "Unknown"]
    stacked = (
        summary.groupby(["sector", "mechanism_label"], observed=False)
        .size()
        .reset_index(name="count")
    )
    pivot = stacked.pivot(index="sector", columns="mechanism_label", values="count").fillna(0)
    pivot = pivot.reindex(SECTOR_ORDER, fill_value=0)
    pivot = pivot.reindex(columns=mech_order, fill_value=0)
    pct = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    colors = ["#4C78A8", "#54A24B", "#F58518", "#B279A2", "#E45756", "#9D9D9D", "#C7C7C7"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    bottom = np.zeros(len(pct))
    x = np.arange(len(pct.index))
    for color, col in zip(colors, pct.columns):
        vals = pct[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.45, label=col)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([SECTOR_LABELS[s] for s in pct.index])
    ax.set_ylim(0, 100)
    ax.set_ylabel("构成占比（%）")
    ax.set_title("S4 各产业机制构成堆叠柱状图", loc="left", fontsize=11.2, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.2, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    save_plot(fig, fig_dir / "S4_mechanism_stacked_bar")


def render_s5_trend(cases: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = build_case_summary(cases)
    summary = summary.sort_values(["sector", "observed_lag_months", "case_id"]).copy()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    axes = axes.ravel()
    for ax, sector in zip(axes, SECTOR_ORDER):
        sub = summary[summary["sector"] == sector].copy()
        if sub.empty:
            ax.axis("off")
            continue
        x = np.arange(1, len(sub) + 1)
        ax.plot(x, sub["observed_lag_months"], marker="o", color="#4C78A8", label="真实")
        ax.plot(x, sub["simulated_lag_months"], marker="o", color="#F58518", label="模拟")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["case_id"], rotation=0)
        ax.set_title(SECTOR_LABELS.get(sector, sector), loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel("时滞（月）")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.01, 0.98, "S5 时滞趋势折线图", ha="left", va="top", fontsize=11.2, fontweight="bold")
    save_plot(fig, fig_dir / "S5_lag_trend_line")


def render_s6_heatmap(cases: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = build_case_summary(cases)
    mech_order = ["Policy interpretation", "Standard consensus", "Coordination", "Market incentive", "Curriculum approval", "Other", "Unknown"]
    heat = (
        summary.groupby(["sector", "mechanism_label"], observed=False)["simulated_lag_months"]
        .mean()
        .reset_index()
    )
    pivot = heat.pivot(index="sector", columns="mechanism_label", values="simulated_lag_months")
    pivot = pivot.reindex(SECTOR_ORDER, fill_value=np.nan)
    pivot = pivot.reindex(columns=mech_order)
    fig, ax = plt.subplots(figsize=(9.4, 4.6), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([SECTOR_LABELS[s] for s in pivot.index])
    ax.set_title("S6 产业-机制热力图", loc="left", fontsize=11.2, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("平均模拟时滞（月）")
    save_plot(fig, fig_dir / "S6_industry_mechanism_heatmap")


def render_supplementary_suite(cases: pd.DataFrame, out_dir: Path) -> None:
    render_s1_grouped_bar(cases, out_dir)
    render_s2_scatter(cases, out_dir)
    render_s3_distribution(cases, out_dir)
    render_s4_mechanism_stacked(cases, out_dir)
    render_s5_trend(cases, out_dir)
    render_s6_heatmap(cases, out_dir)


def write_outputs(
    cases: pd.DataFrame,
    out_dir: Path,
    validation_source: pd.DataFrame,
    split_cases: pd.DataFrame | None = None,
) -> None:
    cases = add_behavior_columns(cases)
    table_dir = out_dir / "tables"
    cases_dir = out_dir / "cases"
    table_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    table_cols = [
        "case_id",
        "sector",
        "sector_label",
        "chain_type_norm",
        "pathway_label",
        "upstream_event_id",
        "downstream_event_id",
        "up_title",
        "down_title",
        "observed_lag_months",
        "simulated_lag_months",
        "dominant_constraint",
        "macro_dimension",
        "macro_label",
        "field",
        "behavior_dimension",
        "behavior_indicator",
        "case_fine_behavior",
        "capital_feature",
        "habitus_feature",
        "action_type",
        "decision",
        "confidence",
    ]
    table = cases[table_cols].copy()
    if "role" in cases.columns:
        table["role"] = cases["role"].astype(str)
    table.to_csv(table_dir / "case_catalog.csv", index=False, encoding="utf-8-sig")
    table.to_markdown(table_dir / "case_catalog.md", index=False)
    MAP_DF.to_csv(table_dir / "macro_to_micro_behavior_map.csv", index=False, encoding="utf-8-sig")
    MAP_DF.to_markdown(table_dir / "macro_to_micro_behavior_map.md", index=False)

    if split_cases is not None and not split_cases.empty:
        split_cols = [c for c in ["case_id", "sector", "sector_label", "chain_type_norm", "pathway_label", "role"] if c in split_cases.columns]
        split = split_cases[split_cols].copy()
        split.to_csv(table_dir / "case_split.csv", index=False, encoding="utf-8-sig")
        split.to_markdown(table_dir / "case_split.md", index=False)
        main_only = split_cases[split_cases["role"] == "main"].copy()
        supp_only = split_cases[split_cases["role"] == "supplement"].copy()
        if not main_only.empty:
            main_only.to_csv(table_dir / "case_catalog_main.csv", index=False, encoding="utf-8-sig")
            main_only.to_markdown(table_dir / "case_catalog_main.md", index=False)
        if not supp_only.empty:
            supp_only.to_csv(table_dir / "case_catalog_supplement.csv", index=False, encoding="utf-8-sig")
            supp_only.to_markdown(table_dir / "case_catalog_supplement.md", index=False)

    jsonl_path = out_dir / "cases.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for _, row in cases.iterrows():
            record = asdict(case_to_record(row))
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    case_md_blocks = []
    for _, row in cases.iterrows():
        case = case_to_record(row)
        case_md = case_story(case)
        case_md_blocks.append(case_md)
        (cases_dir / f"{case.case_id}.md").write_text(case_md, encoding="utf-8")
        render_case_schematic(row, out_dir)
        render_causal_chain(row, out_dir)

    (out_dir / "cases.md").write_text("\n\n".join(case_md_blocks), encoding="utf-8")

    timeline_df = write_case_timeline_exports(cases, out_dir)
    render_validation_chart(validation_source, out_dir)
    render_supplementary_suite(cases, out_dir)

    summary = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_cases": int(len(cases)),
        "sectors": sorted(cases["sector"].astype(str).unique().tolist()),
        "chain_types": sorted(cases["chain_type_norm"].astype(str).unique().tolist()),
        "case_ids": cases["case_id"].astype(str).tolist(),
        "output_dir": str(out_dir),
    }
    if not timeline_df.empty:
        summary["timeline_rows"] = int(len(timeline_df))
    if split_cases is not None and not split_cases.empty and "role" in split_cases.columns:
        summary["main_case_ids"] = split_cases[split_cases["role"] == "main"]["case_id"].astype(str).tolist()
        summary["supplement_case_ids"] = split_cases[split_cases["role"] == "supplement"]["case_id"].astype(str).tolist()
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def make_overview(cases: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig, axes = plt.subplots(
        nrows=int(np.ceil(len(cases) / 2)),
        ncols=2,
        figsize=(10.0, 2.4 * int(np.ceil(len(cases) / 2))),
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(-1)
    for ax, (_, row) in zip(axes, cases.iterrows()):
        ax.axis("off")
        sector = safe_text(row.get("sector"))
        color = SECTOR_COLORS.get(sector, "#4C78A8")
        rect = FancyBboxPatch((0.04, 0.08), 0.92, 0.82, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#ffffff", edgecolor=color, linewidth=1.0)
        ax.add_patch(rect)
        ax.text(0.08, 0.80, f"{safe_text(row.get('case_id'))} | {safe_text(row.get('sector_label'))}", fontsize=9.2, fontweight="bold", transform=ax.transAxes)
        ax.text(0.08, 0.66, safe_text(row.get("pathway_label")), fontsize=8.0, transform=ax.transAxes)
        ax.text(0.08, 0.52, wrap(safe_text(row.get("up_title")), 34), fontsize=7.0, transform=ax.transAxes)
        ax.text(0.08, 0.30, f"Obs {row.get('observed_lag_months', np.nan):.1f} mo | Sim {row.get('simulated_lag_months', np.nan):.1f} mo", fontsize=7.0, transform=ax.transAxes)
        ax.text(0.08, 0.16, wrap(f"Constraint: {safe_text(row.get('dominant_constraint'))}", 34), fontsize=7.0, transform=ax.transAxes)
    for ax in axes[len(cases):]:
        ax.axis("off")
    save_figure_triplet(fig, fig_dir / "case_appendix_overview")
    plt.close(fig)


def run_case_counterfactuals_script() -> None:
    script_path = Path(__file__).with_name("11make_results4_case_counterfactuals.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Missing counterfactual script: {script_path}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--total-cases", type=int, default=12)
    parser.add_argument("--sectors", default="AI,NEV,robotics,eldercare")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--no-overview", action="store_false", dest="write_overview", default=True)
    args = parser.parse_args()

    setup_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_frame()
    events, chains = load_events_and_chains()
    sectors = [s.strip() for s in args.sectors.split(",") if s.strip()]
    cases = build_selected_cases(df, events, chains, total_cases=args.total_cases, sectors=sectors, seed=args.seed)
    split_cases = choose_representative_cases(cases)
    write_outputs(split_cases, out_dir, validation_source=df, split_cases=split_cases)
    if args.write_overview:
        make_overview(split_cases[split_cases["role"] == "main"].copy(), out_dir)
    run_case_counterfactuals_script()

    print(json.dumps(
        {
            "output_dir": str(out_dir),
            "n_cases": int(len(cases)),
            "case_ids": cases["case_id"].tolist(),
            "sectors": cases["sector"].tolist(),
            "main_case_ids": split_cases[split_cases["role"] == "main"]["case_id"].tolist(),
            "supplement_case_ids": split_cases[split_cases["role"] == "supplement"]["case_id"].tolist(),
            "validation_fig": str(out_dir / "figures" / "validation" / "validation_observed_vs_simulated.pdf"),
            "counterfactual_output_dir": str(PROJECT_ROOT / "outputs" / "results4_case_counterfactuals"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
