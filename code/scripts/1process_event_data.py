from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "event_analysis"

MAIN_CHAIN_TYPES = [
    "policy_to_firm",
    "policy_to_school",
    "firm_to_school",
    "standard_to_curriculum",
]

EVENT_TYPE_MAP = {
    "政策事件": "policy",
    "企业需求事件": "demand",
    "学校响应事件": "school_response",
    "行业组织/标准事件": "standard_or_intermediary",
}


def clean_col(name: object) -> str:
    text = str(name).strip()
    text = re.sub(r"\s+", "_", text)
    return text


def read_domain_workbook(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    book = pd.ExcelFile(path)
    events = pd.read_excel(book, sheet_name=book.sheet_names[0])
    chains = pd.read_excel(book, sheet_name=book.sheet_names[1])
    events.columns = [clean_col(c) for c in events.columns]
    chains.columns = [clean_col(c) for c in chains.columns]
    domain = path.stem.replace("(1)", "")
    events["source_workbook"] = path.name
    chains["source_workbook"] = path.name
    events["domain_from_file"] = domain
    chains["domain_from_file"] = domain
    return events, chains


def normalize_date(value: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.to_datetime(value)
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="D", origin="1899-12-30", errors="coerce")
    text = str(value).strip()
    if not text:
        return pd.NaT
    return pd.to_datetime(text, errors="coerce")


def _dedupe_signature(df: pd.DataFrame, subset_cols: list[str]) -> pd.Series:
    frame = df[subset_cols].copy().fillna("<NA>").astype(str)
    return frame.agg("\x1f".join, axis=1)


def dedupe_events(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], int]:
    df = events.copy()
    if "event_id" not in df.columns:
        return df, {}, 0
    subset_cols = [c for c in df.columns if c != "event_id"]
    signature = _dedupe_signature(df, subset_cols)
    keep_mask = ~signature.duplicated(keep="first")
    if keep_mask.all():
        return df, {}, 0
    first_ids: dict[str, str] = {}
    id_map: dict[str, str] = {}
    for event_id, sig in zip(df["event_id"].astype(str), signature, strict=False):
        if sig in first_ids:
            id_map[event_id] = first_ids[sig]
        else:
            first_ids[sig] = event_id
    return df.loc[keep_mask].copy(), id_map, int((~keep_mask).sum())


def remap_chain_event_ids(chains: pd.DataFrame, id_map: dict[str, str]) -> pd.DataFrame:
    if not id_map:
        return chains.copy()
    df = chains.copy()
    for col in ["upstream_event_id", "downstream_event_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: id_map.get(x, x))
    return df


def dedupe_chains(chains: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = chains.copy()
    if "chain_id" not in df.columns:
        return df, 0
    subset_cols = [c for c in df.columns if c != "chain_id"]
    signature = _dedupe_signature(df, subset_cols)
    keep_mask = ~signature.duplicated(keep="first")
    if keep_mask.all():
        return df, 0
    return df.loc[keep_mask].copy(), int((~keep_mask).sum())


def normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    for col in ["event_id", "event_type", "sector", "source_url", "title", "event_date"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["event_date_parsed"] = df["event_date"].map(normalize_date)
    df["year_parsed"] = df["event_date_parsed"].dt.year
    df["month_parsed"] = df["event_date_parsed"].dt.month
    if "year" in df.columns:
        df["year_final"] = pd.to_numeric(df["year"], errors="coerce").fillna(df["year_parsed"])
    else:
        df["year_final"] = df["year_parsed"]
    if "month" in df.columns:
        df["month_final"] = pd.to_numeric(df["month"], errors="coerce").fillna(df["month_parsed"])
    else:
        df["month_final"] = df["month_parsed"]
    df["event_type_norm"] = df["event_type"].astype(str).str.strip()
    df["source_url_norm"] = df["source_url"].astype(str).str.strip()
    generic_patterns = [
        r"^https?://www\.most\.gov\.cn/?$",
        r"^https?://www\.ndrc\.gov\.cn/?$",
        r"^https?://www\.moe\.gov\.cn/?$",
        r"^https?://www\.miit\.gov\.cn/?$",
        r"^https?://www\.gov\.cn/?$",
    ]
    generic_re = re.compile("|".join(generic_patterns), re.I)
    df["has_specific_url"] = df["source_url_norm"].ne("") & ~df["source_url_norm"].str.match(generic_re, na=False)
    return df


def normalize_chains(chains: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    df = chains.copy()
    for col in ["upstream_event_id", "downstream_event_id", "chain_type", "lag_months", "confidence"]:
        if col not in df.columns:
            df[col] = pd.NA
    ids = set(events["event_id"].dropna().astype(str))
    df["upstream_event_id_norm"] = df["upstream_event_id"].astype(str)
    df["downstream_event_id_norm"] = df["downstream_event_id"].astype(str)
    df["chain_type_norm"] = df["chain_type"].astype(str).str.strip()
    df["lag_months_num"] = pd.to_numeric(df["lag_months"], errors="coerce")
    df["confidence_norm"] = df["confidence"].astype(str).str.lower().str.strip()
    df["bad_upstream_ref"] = ~df["upstream_event_id_norm"].isin(ids)
    df["bad_downstream_ref"] = ~df["downstream_event_id_norm"].isin(ids)
    df["is_main_chain_type"] = df["chain_type_norm"].isin(MAIN_CHAIN_TYPES)
    df["is_high_or_medium"] = df["confidence_norm"].isin(["high", "medium"])
    df["use_in_main_auto"] = (
        df["is_main_chain_type"]
        & df["is_high_or_medium"]
        & df["lag_months_num"].notna()
        & (df["lag_months_num"] >= 0)
        & ~df["bad_upstream_ref"]
        & ~df["bad_downstream_ref"]
    )
    return df


def pct(series: pd.Series, predicate) -> float:
    if series.empty:
        return float("nan")
    return round(100 * predicate(series).mean(), 1)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    frame = df.copy()
    frame = frame.fillna("")
    headers = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in frame.iterrows():
        values = [str(row[col]) for col in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_tables(events: pd.DataFrame, chains: pd.DataFrame) -> dict[str, pd.DataFrame]:
    event_type_summary = (
        events.groupby("event_type_norm", dropna=False)
        .size()
        .reset_index(name="n_events")
        .sort_values("n_events", ascending=False)
    )
    sector_summary = (
        events.groupby("sector", dropna=False)
        .size()
        .reset_index(name="n_events")
        .sort_values("n_events", ascending=False)
    )
    chain_type_summary = (
        chains.groupby("chain_type_norm", dropna=False)
        .size()
        .reset_index(name="n_chains")
        .sort_values("n_chains", ascending=False)
    )
    main = chains[chains["chain_type_norm"].isin(MAIN_CHAIN_TYPES)].copy()
    lag_summary = (
        main.groupby("chain_type_norm")["lag_months_num"]
        .agg(n="count", mean_months="mean", median_months="median", min_months="min", max_months="max")
        .reset_index()
    )
    if not lag_summary.empty:
        lag_summary["mean_months"] = lag_summary["mean_months"].round(1)
        lag_summary["median_months"] = lag_summary["median_months"].round(1)
        short = main.groupby("chain_type_norm")["lag_months_num"].apply(
            lambda s: pd.Series(
                {
                    "response_lt_12_pct": pct(s, lambda x: x < 12),
                    "response_lt_24_pct": pct(s, lambda x: x < 24),
                }
            )
        )
        lag_summary = lag_summary.merge(short.unstack().reset_index(), on="chain_type_norm", how="left")
    source_quality = pd.DataFrame(
        [
            {
                "n_events": len(events),
                "missing_event_date": int(events["event_date_parsed"].isna().sum()),
                "missing_source_url": int(events["source_url_norm"].isin(["", "nan", "None"]).sum()),
                "specific_source_url": int(events["has_specific_url"].sum()),
                "generic_or_missing_source_url": int((~events["has_specific_url"]).sum()),
            }
        ]
    )
    qc_chains = pd.DataFrame(
        [
            {
                "n_chains": len(chains),
                "bad_upstream_ref": int(chains["bad_upstream_ref"].sum()),
                "bad_downstream_ref": int(chains["bad_downstream_ref"].sum()),
                "negative_lag": int((chains["lag_months_num"] < 0).sum()),
                "lag_over_120_months": int((chains["lag_months_num"] > 120).sum()),
                "main_chain_auto_sample": int(chains["use_in_main_auto"].sum()),
            }
        ]
    )
    annual_events = (
        events.groupby(["year_final", "event_type_norm"], dropna=False)
        .size()
        .reset_index(name="n_events")
        .sort_values(["year_final", "event_type_norm"])
    )
    return {
        "event_type_summary": event_type_summary,
        "sector_summary": sector_summary,
        "chain_type_summary": chain_type_summary,
        "lag_summary_main_chains": lag_summary,
        "source_quality": source_quality,
        "chain_quality": qc_chains,
        "annual_events": annual_events,
    }


def write_markdown_report(
    events: pd.DataFrame,
    chains: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    dedupe_stats: dict[str, int] | None = None,
) -> None:
    main_auto = chains[chains["use_in_main_auto"]].copy()
    lines = [
        "# Event database processing report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Core counts",
        "",
        f"- Total events: {len(events)}",
        f"- Total event chains: {len(chains)}",
        f"- Auto-selected main-text chains: {len(main_auto)}",
        f"- Years covered: {int(events['year_final'].min())}–{int(events['year_final'].max())}",
        f"- Domains covered: {events['domain_from_file'].nunique()}",
    ]
    if dedupe_stats:
        lines.extend(
            [
                f"- Deduplicated event rows removed: {dedupe_stats.get('events_removed', 0)}",
                f"- Deduplicated chain rows removed: {dedupe_stats.get('chains_removed', 0)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Main Results table candidate",
            "",
            dataframe_to_markdown(tables["lag_summary_main_chains"]),
            "",
            "## Quality checks",
            "",
            dataframe_to_markdown(tables["source_quality"]),
            "",
            dataframe_to_markdown(tables["chain_quality"]),
            "",
            "## Chain-type distribution",
            "",
            dataframe_to_markdown(tables["chain_type_summary"]),
            "",
            "## Notes for manuscript use",
            "",
            "- Use the four focal chains for the first Results subsection: policy_to_firm, policy_to_school, firm_to_school and standard_to_curriculum.",
            "- Treat generic home-page URLs as records needing source verification before main-text submission.",
            "- Treat low-confidence chains and non-focal chain types as supplementary or robustness material until manually verified.",
            "- Event chains may exceed event counts because one upstream event can be linked to multiple downstream responses.",
        ]
    )
    (OUT_DIR / "qc_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_files = sorted(RAW_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise SystemExit(f"No xlsx files found in {RAW_DIR}")
    event_frames: list[pd.DataFrame] = []
    chain_frames: list[pd.DataFrame] = []
    for path in xlsx_files:
        events, chains = read_domain_workbook(path)
        event_frames.append(events)
        chain_frames.append(chains)
    raw_events = pd.concat(event_frames, ignore_index=True)
    raw_chains = pd.concat(chain_frames, ignore_index=True)

    deduped_events_raw, id_map, events_removed = dedupe_events(raw_events)
    deduped_chains_raw = remap_chain_event_ids(raw_chains, id_map)
    deduped_chains_raw, chains_removed = dedupe_chains(deduped_chains_raw)

    events = normalize_events(deduped_events_raw)
    chains = normalize_chains(deduped_chains_raw, events)
    tables = make_tables(events, chains)

    events.to_csv(OUT_DIR / "events_all.csv", index=False, encoding="utf-8-sig")
    chains.to_csv(OUT_DIR / "chains_all.csv", index=False, encoding="utf-8-sig")
    chains[chains["use_in_main_auto"]].to_csv(OUT_DIR / "chains_main_auto.csv", index=False, encoding="utf-8-sig")
    events.to_excel(OUT_DIR / "events_all.xlsx", index=False)
    chains.to_excel(OUT_DIR / "chains_all.xlsx", index=False)
    chains[chains["use_in_main_auto"]].to_excel(OUT_DIR / "chains_main_auto.xlsx", index=False)

    table_dir = OUT_DIR / "tables"
    table_dir.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        try:
            table.to_markdown(table_dir / f"{name}.md", index=False)
        except ImportError:
            (table_dir / f"{name}.md").write_text(table.to_string(index=False), encoding="utf-8")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_files": [p.name for p in xlsx_files],
        "n_events": int(len(events)),
        "n_chains": int(len(chains)),
        "n_main_auto_chains": int(chains["use_in_main_auto"].sum()),
        "deduplicated_event_rows_removed": int(events_removed),
        "deduplicated_chain_rows_removed": int(chains_removed),
        "output_dir": str(OUT_DIR),
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(events, chains, tables, {"events_removed": events_removed, "chains_removed": chains_removed})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
