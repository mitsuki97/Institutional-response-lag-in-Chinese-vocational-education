from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")  # 关掉你说的刷屏警告
from tqdm import tqdm  # 进度条
torch.set_num_threads(8)
torch.set_num_interop_threads(2)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
from case_node_config import CASE_NODE_CONFIG
EVENT_DIR = PROJECT_ROOT / "outputs" / "event_analysis"
RESULT3_DIR = PROJECT_ROOT / "outputs" / "results3_agent_conditioning"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "results4_llm_historical_replay"
DEFAULT_SUITE_OUT_DIR = PROJECT_ROOT / "outputs" / "results4_llm_historical_replay_suite"
VALIDATION_OUT_DIR = PROJECT_ROOT / "outputs" / "results4_model_comparison"
SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_API_KEY_FILE = SCRIPT_DIR / ".deepseek_api_key"

DEFAULT_OUT_NAME = "results4_llm_historical_replay"
DEFAULT_COMPARE_OUT_NAME = "results4_model_comparison"

API_BASE_URL = os.environ.get("LLM_REPLAY_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com"
API_KEY = (os.environ.get("LLM_REPLAY_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
API_MODEL = os.environ.get("LLM_REPLAY_MODEL", "deepseek-chat")

VALIDATION_API_BASE_URL = os.environ.get(
    "LLM_REPLAY_VALIDATION_API_BASE_URL",
    "https://ark.cn-beijing.volces.com/api/v3",
)
VALIDATION_API_KEY = (os.environ.get("LLM_REPLAY_VALIDATION_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
VALIDATION_API_MODEL = os.environ.get("LLM_REPLAY_VALIDATION_MODEL", "doubao-seed-2-0-lite-260428")

LOCAL_MODEL = os.environ.get("LLM_REPLAY_LOCAL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
SECOND_LOCAL_MODEL = os.environ.get("LLM_REPLAY_SECOND_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

EPISODES = int(os.environ.get("LLM_REPLAY_EPISODES", "100"))
MAX_NEW_TOKENS = int(os.environ.get("LLM_REPLAY_MAX_NEW_TOKENS", "120"))
VALIDATION_MAX_NEW_TOKENS = int(os.environ.get("LLM_REPLAY_VALIDATION_MAX_NEW_TOKENS", "220"))
FLUSH_EVERY = int(os.environ.get("LLM_REPLAY_FLUSH_EVERY", "4"))
CALIBRATED = os.environ.get("LLM_REPLAY_CALIBRATED", "1").strip() not in {"0", "false", "False"}
MODEL_LABEL = os.environ.get("LLM_REPLAY_MODEL_LABEL", "")
SECOND_MODEL_LABEL = os.environ.get("LLM_REPLAY_SECOND_MODEL_LABEL", "qwen")
API_WORKERS = int(os.environ.get("LLM_REPLAY_API_WORKERS", "4"))
API_TIMEOUT_SECONDS = int(os.environ.get("LLM_REPLAY_API_TIMEOUT_SECONDS", "120"))
API_MAX_RETRIES = int(os.environ.get("LLM_REPLAY_API_MAX_RETRIES", "2"))
VALIDATION_EPISODES = int(os.environ.get("LLM_REPLAY_VALIDATION_EPISODES", "30"))
VALIDATION_FLUSH_EVERY = int(os.environ.get("LLM_REPLAY_VALIDATION_FLUSH_EVERY", "2"))
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_REPLAY_TEMPERATURE", "0"))
DEFAULT_VARIANT = os.environ.get("LLM_REPLAY_VARIANT", "main")
DEFAULT_INCLUDE_MEMORY = os.environ.get("LLM_REPLAY_INCLUDE_MEMORY", "1").strip() not in {"0", "false", "False"}
DEFAULT_ROLE_VARIANT = os.environ.get("LLM_REPLAY_ROLE_VARIANT", "field_theoretic")
DEFAULT_PROMPT_STYLE = os.environ.get("LLM_REPLAY_PROMPT_STYLE", "full")
DEFAULT_RUN_SUITE = os.environ.get("LLM_REPLAY_RUN_SUITE", "0").strip() in {"1", "true", "True", "all"}
OUT_DIR = str(VALIDATION_OUT_DIR if os.environ.get("LLM_REPLAY_VERIFY", "1").strip() not in {"0", "false", "False"} else DEFAULT_OUT_DIR)
COMPARE_OUT_DIR = str(VALIDATION_OUT_DIR)
RUN_MODE = int(os.environ.get("LLM_REPLAY_RUN_MODE", "0"))

FOCAL_CHAINS = [
    "policy_to_firm",
    "firm_to_school",
    "policy_to_school",
    "standard_to_curriculum",
]

PATHWAY_CALIBRATION = {
    "policy_to_firm": {
        "median": 12,
        "mean": 24.1,
        "lt12": 34.3,
        "lt24": 64.2,
        "typical_range": "6-36 months",
        "friction": "policy interpretation, investment cycle, hiring plan revision and local implementation uncertainty",
    },
    "firm_to_school": {
        "median": 12,
        "mean": 30.5,
        "lt12": 32.4,
        "lt24": 64.0,
        "typical_range": "6-48 months",
        "friction": "school-enterprise negotiation, curriculum design, faculty capacity and training-base construction",
    },
    "policy_to_school": {
        "median": 27,
        "mean": 36.9,
        "lt12": 18.9,
        "lt24": 38.8,
        "typical_range": "12-60 months",
        "friction": "administrative translation, programme approval, faculty recruitment and curriculum governance",
    },
    "standard_to_curriculum": {
        "median": 24,
        "mean": 41.6,
        "lt12": 7.5,
        "lt24": 31.3,
        "typical_range": "18-72 months",
        "friction": "standard-setting consensus, certification alignment, textbook revision and curriculum approval",
    },
}

ROLE_TEXT = {
    "government": {
        "field_position": "regulatory centre",
        "capital": "regulatory and symbolic capital",
        "instruction": "Judge whether the signal should trigger policy coordination, resource allocation or delayed administrative diffusion.",
    },
    "firm": {
        "field_position": "demand-side producer",
        "capital": "economic and network capital",
        "instruction": "Judge whether the signal should trigger skill-demand expression, hiring expansion or school cooperation.",
    },
    "school": {
        "field_position": "training institution",
        "capital": "pedagogical and symbolic capital",
        "instruction": "Judge whether the signal can be translated into a programme, curriculum, laboratory or partnership response.",
    },
    "intermediary": {
        "field_position": "translation broker",
        "capital": "translation and network capital",
        "instruction": "Judge whether the signal should be translated into reports, forums, standards or cross-actor coordination.",
    },
    "standard_body": {
        "field_position": "codification authority",
        "capital": "symbolic and translation capital",
        "instruction": "Judge whether the signal can be codified into standards, certifications or curricular reference frameworks.",
    },
}

COORDINATION_TAGS = [
    "policy translation",
    "administrative reporting",
    "standard negotiation",
    "school-enterprise coordination",
    "curriculum alignment",
    "resource mobilization",
    "certification alignment",
    "forum / meeting",
    "brokerage / intermediary",
    "other coordination",
]

VALIDATION_MODELS = [
    VALIDATION_API_MODEL,
]
BASELINE_RESULT_PATH = DEFAULT_OUT_DIR / "llm_replay_results.csv"
BASELINE_MODEL_NAME = "deepseek-chat"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def load_local_api_key() -> str:
    for candidate in [
        os.environ.get("LLM_REPLAY_API_KEY_FILE", "").strip(),
        str(LOCAL_API_KEY_FILE),
    ]:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if text:
                return text
    return ""


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (dict, list, tuple, str, int, float, bool)) or value is None:
        return value
    return str(value)


def json_safe_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {str(k): json_safe(v) for k, v in data.items()}


API_KEY = API_KEY or load_local_api_key()
VALIDATION_API_KEY = VALIDATION_API_KEY or load_local_api_key()


def make_experiment_suite(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = {
        "model_name": args.model,
        "model_label": args.model_label.strip() or slugify_model_name(args.model),
        "out_dir": str(Path(args.out_dir.strip()) if args.out_dir.strip() else Path(OUT_DIR)),
        "max_new_tokens": args.max_new_tokens,
        "flush_every": args.flush_every,
        "calibrated": args.calibrated,
        "temperature": args.temperature,
        "include_memory": args.include_memory,
        "role_variant": args.role_variant,
        "prompt_style": args.prompt_style,
        "variant": args.variant,
        "run_mode": args.run_mode,
        "api_base_url": args.api_base_url or API_BASE_URL,
        "api_key": args.api_key or API_KEY,
        "validation_api_key": args.validation_api_key or VALIDATION_API_KEY,
    }
    suite = [
        {**base, "variant": "main", "experiment_name": "main_replay"},
        {**base, "variant": "memory_ablation", "experiment_name": "memory_ablation", "include_memory": False},
        {**base, "variant": "role_ablation", "experiment_name": "role_ablation", "role_variant": "generic"},
        {**base, "variant": "prompt_short", "experiment_name": "prompt_short", "prompt_style": "short"},
        {**base, "variant": "prompt_long", "experiment_name": "prompt_long", "prompt_style": "long"},
        {**base, "variant": "temperature_sensitivity", "experiment_name": "temperature_sensitivity", "temperature": max(base["temperature"], 0.3)},
        {**base, "variant": "calibration_ablation", "experiment_name": "calibration_ablation", "calibrated": False},
    ]
    return suite


def role_from_event(row: pd.Series) -> str:
    event_type = str(row.get("event_type", "")).strip()
    actor_type = str(row.get("actor_type", "")).strip()
    if event_type == "standard":
        return "standard_body"
    if actor_type in ROLE_TEXT:
        return actor_type
    if actor_type in {"third_party", "intermediary"}:
        return "intermediary"
    return "intermediary"


def slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(model_name).strip()).strip("-")
    return slug.lower()[:48] or "model"


def normalize_mechanism(text: object) -> str:
    t = str(text).strip().lower()
    if not t or t in {"nan", "none"}:
        return "unknown"
    if any(k in t for k in [
        "policy interpretation",
        "administrative translation",
        "policy translation",
        "implementation",
        "policy coordination",
        "regulatory",
        "监管",
        "政策",
        "制度翻译",
        "行政转译",
        "落实",
    ]):
        return "policy interpretation"
    if any(k in t for k in [
        "standard consensus",
        "standard-setting consensus",
        "consensus building",
        "consensus",
        "standardization",
        "certification alignment",
        "certification",
        "qualification",
        "标准",
        "认证",
        "对标",
    ]):
        return "standard consensus"
    if any(k in t for k in [
        "school-enterprise",
        "coordination",
        "negotiation",
        "cooperation",
        "partnership",
        "collaboration",
        "校企",
        "协同",
        "对接",
        "合作",
    ]):
        return "coordination"
    if any(k in t for k in [
        "market incentive",
        "investment cycle",
        "market",
        "economic",
        "hiring plan",
        "investment",
        "激励",
        "市场",
        "经济",
        "招聘",
        "投资",
    ]):
        return "market incentive"
    if any(k in t for k in [
        "curriculum approval",
        "curriculum mapping",
        "curriculum design",
        "curriculum",
        "programme approval",
        "program approval",
        "major setting",
        "课程",
        "专业",
        "培养方案",
        "教学",
    ]):
        return "curriculum approval"
    if any(k in t for k in [
        "teacher",
        "faculty",
        "resource",
        "capacity",
        "equipment",
        "funding",
        "budget",
        "procurement",
        "教师",
        "师资",
        "资源",
        "设备",
        "经费",
        "预算",
    ]):
        return "other"
    return "unknown"


def infer_mechanism(*parts: object) -> str:
    combined = " ".join(str(part) for part in parts if str(part).strip() and str(part).strip().lower() not in {"nan", "none"})
    return normalize_mechanism(combined)


def normalize_action_type(value: object) -> str:
    t = str(value).strip().lower()
    if not t or t in {"nan", "none"}:
        return ""
    if any(k in t for k in [
        "respond_now",
        "respond",
        "response",
        "immediate",
        "now",
        "act now",
        "take action",
        "立刻",
        "立即",
        "马上",
        "尽快",
        "及时",
    ]):
        return "respond_now"
    if any(k in t for k in [
        "request_coordination",
        "coordination",
        "coordinate",
        "cooperation",
        "school cooperation",
        "translate",
        "translation",
        "report",
        "reports",
        "standards",
        "standard",
        "forum",
        "meeting",
        "partnership",
        "协同",
        "协调",
        "合作",
        "对接",
        "转译",
        "翻译",
        "上报",
        "报告",
        "标准",
        "论坛",
        "会议",
    ]):
        return "request_coordination"
    if any(k in t for k in ["delay", "wait", "postpone", "defer", "slow", "延迟", "等待", "推迟", "拖延"]):
        return "delay"
    if any(k in t for k in ["ignore", "ignore signal", "不理", "忽略"]):
        return "ignore"
    return ""


def normalize_coordination_tags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "[]"}:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = [text]
    else:
        parsed = value

    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]

    out: list[str] = []
    for item in parsed:
        t = str(item).strip().lower()
        if not t or t in {"nan", "none", "[]"}:
            continue
        if "policy" in t and "translation" in t:
            out.append("policy translation")
        elif any(k in t for k in ["report", "reporting", "上报", "报告"]):
            out.append("administrative reporting")
        elif any(k in t for k in ["standard", "certification", "认证", "标准", "对标"]):
            out.append("standard negotiation" if "negoti" in t or "consensus" in t or "协商" in t else "certification alignment")
        elif any(k in t for k in ["school-enterprise", "school", "校企", "合作", "协同", "对接"]):
            out.append("school-enterprise coordination")
        elif any(k in t for k in ["curriculum", "programme", "program", "课程", "专业", "培养方案"]):
            out.append("curriculum alignment")
        elif any(k in t for k in ["resource", "fund", "budget", "equipment", "teacher", "faculty", "师资", "资源", "经费", "设备"]):
            out.append("resource mobilization")
        elif any(k in t for k in ["forum", "meeting", "conference", "forum", "会议", "论坛"]):
            out.append("forum / meeting")
        elif any(k in t for k in ["broker", "intermediary", "translation broker", "中介", "brokerage"]):
            out.append("brokerage / intermediary")
        else:
            out.append("other coordination")
    # preserve order, deduplicate
    seen: set[str] = set()
    deduped: list[str] = []
    for tag in out:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped


def get_openai_client(base_url: str, api_key: str, timeout_seconds: int = API_TIMEOUT_SECONDS):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing openai package. Install it with: pip install openai"
        ) from exc
    resolved_api_key = str(api_key or "").strip()
    if not resolved_api_key:
        raise RuntimeError(
            "Missing API credentials. Set LLM_REPLAY_API_KEY or OPENAI_API_KEY before running replay."
        )
    return OpenAI(base_url=base_url, api_key=resolved_api_key, timeout=timeout_seconds, max_retries=API_MAX_RETRIES)


def get_ark_client(api_key: str, timeout_seconds: int = API_TIMEOUT_SECONDS):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing openai package. Install it with: pip install openai"
        ) from exc
    resolved_api_key = str(api_key or "").strip()
    if not resolved_api_key:
        raise RuntimeError(
            "Missing ARK credentials. Set VALIDATION_API_KEY before running validation."
        )
    return OpenAI(base_url=VALIDATION_API_BASE_URL, api_key=resolved_api_key, timeout=timeout_seconds, max_retries=API_MAX_RETRIES)


def infer_action_type(*parts: object) -> str:
    combined = " ".join(str(part) for part in parts if str(part).strip() and str(part).strip().lower() not in {"nan", "none"})
    action = normalize_action_type(combined)
    if action:
        return action
    return "respond_now" if combined else ""


def generate_json_via_api(
    client,
    model: str,
    prompt: str,
    max_new_tokens: int,
    use_responses_api: bool = False,
    temperature: float = 0.0,
) -> str:
    if use_responses_api:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
            temperature=temperature,
            max_output_tokens=max_new_tokens,
        )
        text = ""
        try:
            text = getattr(response, "output_text", "") or ""
        except Exception:
            text = ""
        if not text:
            try:
                chunks: list[str] = []
                for item in getattr(response, "output", []) or []:
                    for content in getattr(item, "content", []) or []:
                        if getattr(content, "type", "") in {"output_text", "text"}:
                            chunks.append(getattr(content, "text", "") or "")
                text = "".join(chunks).strip()
            except Exception:
                text = ""
        if not text:
            text = str(response)
        return text.strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output strict JSON only. No markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_new_tokens,
        stream=False,
    )
    return response.choices[0].message.content.strip()


def api_replay_worker(payload: tuple[int, pd.Series, pd.DataFrame, dict[str, Any], str, int, str, str, int, bool, float]) -> dict[str, Any]:
    idx, row, events, settings, api_base_url, max_new_tokens, api_key, model_name, timeout_seconds, use_responses_api, temperature = payload
    client = get_openai_client(api_base_url, api_key, timeout_seconds=timeout_seconds)
    memories = memory_examples(events, str(row["down_agent_role"]), str(row.get("up_sector", "")), row.get("up_year_final", np.nan))
    if not settings.get("include_memory", True):
        memories = []
    prompt = prompt_for_episode(
        row,
        memories,
        calibrated=bool(settings.get("calibrated", True)),
        validation_mode=False,
        prompt_style=str(settings.get("prompt_style", "full")),
        role_variant=str(settings.get("role_variant", "field_theoretic")),
    )
    raw = generate_json_via_api(client, model_name, prompt, max_new_tokens, use_responses_api=use_responses_api, temperature=temperature)
    parsed = parse_json(raw)
    sim_delay = clamp_num(parsed.get("expected_delay_months"), 0, 240)
    dominant_constraint = normalize_mechanism(parsed.get("dominant_constraint", ""))
    explanation = parsed.get("explanation", parsed.get("latent_reason", ""))
    action_type = infer_action_type(parsed.get("action_type", ""), parsed.get("decision", ""), explanation)
    return {
        "episode_index": int(idx),
        "chain_id": row.get("chain_id", ""),
        "chain_type_norm": row.get("chain_type_norm", ""),
        "experiment_name": settings.get("experiment_name", ""),
        "variant": settings.get("variant", ""),
        "upstream_event_id": row.get("upstream_event_id", ""),
        "downstream_event_id": row.get("downstream_event_id", ""),
        "downstream_role": row.get("down_agent_role", ""),
        "sector": row.get("up_sector", ""),
        "lag_months_num": float(row.get("lag_months_num", np.nan)),
        "sim_delay_months": sim_delay,
        "prompt_style": settings.get("prompt_style", ""),
        "role_variant": settings.get("role_variant", ""),
        "temperature": float(temperature),
        "include_memory": bool(settings.get("include_memory", True)),
        "perceived_urgency": clamp_num(parsed.get("perceived_urgency"), 0, 1),
        "perceived_feasibility": clamp_num(parsed.get("perceived_feasibility"), 0, 1),
        "dominant_constraint": dominant_constraint,
        "action_type": action_type,
        "decision": parsed.get("decision", action_type),
        "coordination_request": json.dumps(normalize_coordination_tags(parsed.get("coordination_request", [])), ensure_ascii=False),
        "explanation": explanation,
        "latent_reason": parsed.get("latent_reason", explanation),
        "confidence": clamp_num(parsed.get("confidence"), 0, 1),
        "parse_error": bool(parsed.get("parse_error", False)),
        "raw_output": raw[:2000],
                "calibrated_prompt": bool(settings.get("calibrated", True)),
        "model_name": model_name,
        "model_label": slugify_model_name(model_name),
        "memory_examples_json": json.dumps(memories, ensure_ascii=False),
        "prompt_text": prompt,
        "api_base_url": api_base_url,
    }


def ark_replay_worker(payload: tuple[int, pd.Series, pd.DataFrame, dict[str, Any], int, str, str, int, float]) -> dict[str, Any]:
    idx, row, events, settings, max_new_tokens, api_key, model_name, timeout_seconds, temperature = payload
    client = get_ark_client(api_key, timeout_seconds=timeout_seconds)
    memories = memory_examples(events, str(row["down_agent_role"]), str(row.get("up_sector", "")), row.get("up_year_final", np.nan))
    if not settings.get("include_memory", True):
        memories = []
    prompt = prompt_for_episode(
        row,
        memories,
        calibrated=bool(settings.get("calibrated", True)),
        validation_mode=True,
        prompt_style=str(settings.get("prompt_style", "full")),
        role_variant=str(settings.get("role_variant", "field_theoretic")),
    )
    raw = generate_json_via_api(client, model_name, prompt, max_new_tokens, use_responses_api=False, temperature=temperature)
    parsed = parse_json(raw)
    sim_delay = clamp_num(parsed.get("expected_delay_months"), 0, 240)
    dominant_constraint = normalize_mechanism(parsed.get("dominant_constraint", ""))
    explanation = parsed.get("explanation", parsed.get("latent_reason", ""))
    action_type = infer_action_type(parsed.get("action_type", ""), parsed.get("decision", ""), explanation)
    return {
        "episode_index": int(idx),
        "chain_id": row.get("chain_id", ""),
        "chain_type_norm": row.get("chain_type_norm", ""),
        "experiment_name": settings.get("experiment_name", ""),
        "variant": settings.get("variant", ""),
        "upstream_event_id": row.get("upstream_event_id", ""),
        "downstream_event_id": row.get("downstream_event_id", ""),
        "downstream_role": row.get("down_agent_role", ""),
        "sector": row.get("up_sector", ""),
        "lag_months_num": float(row.get("lag_months_num", np.nan)),
        "sim_delay_months": sim_delay,
        "prompt_style": settings.get("prompt_style", ""),
        "role_variant": settings.get("role_variant", ""),
        "temperature": float(temperature),
        "include_memory": bool(settings.get("include_memory", True)),
        "perceived_urgency": clamp_num(parsed.get("perceived_urgency"), 0, 1),
        "perceived_feasibility": clamp_num(parsed.get("perceived_feasibility"), 0, 1),
        "dominant_constraint": dominant_constraint,
        "action_type": action_type,
        "decision": parsed.get("decision", action_type),
        "coordination_request": json.dumps(normalize_coordination_tags(parsed.get("coordination_request", [])), ensure_ascii=False),
        "explanation": explanation,
        "latent_reason": parsed.get("latent_reason", explanation),
        "confidence": clamp_num(parsed.get("confidence"), 0, 1),
        "parse_error": bool(parsed.get("parse_error", False)),
        "raw_output": raw[:2000],
                "calibrated_prompt": bool(settings.get("calibrated", True)),
        "model_name": model_name,
        "model_label": slugify_model_name(model_name),
        "memory_examples_json": json.dumps(memories, ensure_ascii=False),
        "prompt_text": prompt,
    }


def _extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]


def extract_structured_fields(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    json_blob = _extract_balanced_json(cleaned)
    parsed: dict[str, Any] = {}
    if json_blob:
        try:
            return json.loads(json_blob)
        except Exception:
            pass

    patterns = {
        "agent_role": r'"agent_role"\s*:\s*"([^"]+)"',
        "dominant_constraint": r'"dominant_constraint"\s*:\s*"([^"]+)"',
        "action_type": r'"action_type"\s*:\s*"([^"]+)"',
        "explanation": r'"explanation"\s*:\s*"((?:\\.|[^"\\])*)"',
        "confidence": r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)',
        "expected_delay_months": r'"expected_delay_months"\s*:\s*([0-9]+)',
        "perceived_urgency": r'"perceived_urgency"\s*:\s*([0-9]*\.?[0-9]+)',
        "perceived_feasibility": r'"perceived_feasibility"\s*:\s*([0-9]*\.?[0-9]+)',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, cleaned, flags=re.S)
        if not m:
            continue
        value = m.group(1)
        if key in {"confidence", "perceived_urgency", "perceived_feasibility"}:
            try:
                parsed[key] = float(value)
            except Exception:
                pass
        elif key == "expected_delay_months":
            try:
                parsed[key] = int(value)
            except Exception:
                pass
        else:
            parsed[key] = bytes(value, "utf-8").decode("unicode_escape") if "\\u" in value or "\\n" in value or "\\t" in value else value

    m = re.search(r'"coordination_request"\s*:\s*\[(.*?)\]', cleaned, flags=re.S)
    if m:
        items = re.findall(r'"([^"]+)"', m.group(1))
        parsed["coordination_request"] = items
    return parsed


def read_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(EVENT_DIR / "events_all.csv")
    chains = pd.read_csv(EVENT_DIR / "chains_all.csv")
    events["agent_role"] = events.apply(role_from_event, axis=1)
    events["year_final"] = pd.to_numeric(events["year_final"], errors="coerce")
    chains["lag_months_num"] = pd.to_numeric(chains["lag_months_num"], errors="coerce")
    chains = chains[
        chains["chain_type_norm"].isin(FOCAL_CHAINS)
        & chains["lag_months_num"].notna()
        & (chains["lag_months_num"] >= 0)
        & chains["upstream_event_id"].isin(set(events["event_id"]))
        & chains["downstream_event_id"].isin(set(events["event_id"]))
    ].copy()
    return events, chains


def build_episode_table(events: pd.DataFrame, chains: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n == len(CASE_NODE_CONFIG):
        rows = []
        used_chain_ids = set()
        for item in CASE_NODE_CONFIG:
            target = str(item["source_event_id"])
            preferred = str(item["preferred_chain_type"])
            matches = chains[
                (chains["upstream_event_id"].astype(str) == target)
                | (chains["downstream_event_id"].astype(str) == target)
            ].copy()
            matches["preferred"] = matches["chain_type_norm"].astype(str) == preferred
            matches["target_is_upstream"] = matches["upstream_event_id"].astype(str) == target
            matches = matches[~matches["chain_id"].astype(str).isin(used_chain_ids)]
            if matches.empty:
                raise RuntimeError(f"Configured case event is absent from chain table: {target}")
            matches = matches.sort_values(
                ["preferred", "target_is_upstream", "confidence_norm", "chain_id"],
                ascending=[False, False, False, True],
                kind="mergesort",
            )
            row = matches.iloc[[0]].copy()
            row["case_id"] = item["case_id"]
            row["case_node_position"] = item["position"]
            row["case_node_label"] = item["node_label"]
            row["source_event_id"] = target
            rows.append(row)
            used_chain_ids.add(str(row.iloc[0]["chain_id"]))
        selected = pd.concat(rows, ignore_index=True)
    else:
        rng = np.random.default_rng(seed)
        frames = []
        per_type = max(1, n // len(FOCAL_CHAINS))
        for chain_type in FOCAL_CHAINS:
            sub = chains[chains["chain_type_norm"] == chain_type]
            sample_n = min(per_type, len(sub))
            frames.append(sub.sample(n=sample_n, random_state=int(rng.integers(0, 1_000_000))))
        selected = pd.concat(frames, ignore_index=True)
        if len(selected) < n:
            rest = chains.drop(index=selected.index, errors="ignore")
            if len(rest):
                selected = pd.concat(
                    [selected, rest.sample(n=min(n - len(selected), len(rest)), random_state=seed + 1)],
                    ignore_index=True,
                )
    event_cols = [
        "event_id",
        "event_date",
        "year_final",
        "sector",
        "event_type",
        "actor_type",
        "agent_role",
        "actor_name",
        "action_type",
        "title",
        "description",
        "evidence_text",
    ]
    ev = events[event_cols].copy()
    merged = selected.merge(ev.add_prefix("up_"), left_on="upstream_event_id", right_on="up_event_id", how="left")
    merged = merged.merge(ev.add_prefix("down_"), left_on="downstream_event_id", right_on="down_event_id", how="left")
    return merged


def memory_examples(events: pd.DataFrame, role: str, sector: str, before_year: float, k: int = 4) -> list[dict[str, Any]]:
    pool = events[(events["agent_role"] == role) & (events["sector"] == sector)].copy()
    if not math.isnan(before_year):
        pool = pool[pool["year_final"].fillna(9999) <= before_year]
    if pool.empty:
        pool = events[events["agent_role"] == role].copy()
    pool = pool.sort_values(["year_final", "event_id"]).tail(k)
    fields = ["year_final", "event_type", "actor_name", "action_type", "title"]
    return pool[fields].fillna("").to_dict("records")


def role_profile(role: str, role_variant: str = "field_theoretic") -> dict[str, str]:
    if role_variant == "generic":
        return {
            "field_position": "institutional actor",
            "capital": "limited and context-dependent institutional capital",
            "instruction": "Judge whether the signal should trigger action, coordination or delay.",
        }
    return ROLE_TEXT.get(role, ROLE_TEXT["intermediary"])


def prompt_for_episode(
    row: pd.Series,
    memories: list[dict[str, Any]],
    calibrated: bool = False,
    validation_mode: bool = False,
    prompt_style: str = "full",
    role_variant: str = "field_theoretic",
) -> str:
    role = str(row["down_agent_role"])
    info = role_profile(role, role_variant=role_variant)
    chain_type = str(row["chain_type_norm"])
    cal = PATHWAY_CALIBRATION.get(chain_type, {})
    memory_block = ""
    if memories:
        memory_block = json.dumps(memories, ensure_ascii=False, indent=2)
    if prompt_style == "short":
        task_intro = "Replay the downstream institutional response to the upstream event and return strict JSON only."
    elif prompt_style == "long":
        task_intro = (
            "Replay the downstream institutional response to the upstream event using the field role, "
            "historical memory examples and pathway-specific calibration as structured cues."
        )
    else:
        task_intro = "Replay the downstream institutional response to the upstream event."
    if validation_mode:
        payload = {
            "chain_type_to_replay": row["chain_type_norm"],
            "technology_domain": row.get("up_sector", ""),
            "upstream_event": {
                "year": row.get("up_year_final", ""),
                "actor_type": row.get("up_actor_type", ""),
                "actor_name": row.get("up_actor_name", ""),
                "event_type": row.get("up_event_type", ""),
                "action_type": row.get("up_action_type", ""),
                "title": row.get("up_title", ""),
            },
            "downstream_role_to_simulate": role,
            "historical_memory_examples": memories if prompt_style != "short" else memories[:2],
        }
    else:
        payload = {
            "chain_type_to_replay": row["chain_type_norm"],
            "observed_lag_months_hidden_from_decision": "withheld during reasoning; used only after simulation for validation",
            "technology_domain": row.get("up_sector", ""),
            "upstream_event": {
                "year": row.get("up_year_final", ""),
                "actor_type": row.get("up_actor_type", ""),
                "actor_name": row.get("up_actor_name", ""),
                "event_type": row.get("up_event_type", ""),
                "action_type": row.get("up_action_type", ""),
                "title": row.get("up_title", ""),
                "description": row.get("up_description", ""),
                "evidence": row.get("up_evidence_text", ""),
            },
            "downstream_role_to_simulate": role,
            "historical_memory_examples": memories if prompt_style != "short" else memories[:2],
        }
    calibration_block = ""
    if calibrated:
        calibration_block = f"""
Empirical calibration constraints for this pathway:
- Observed pathway: {chain_type}
- Historical median response lag: about {cal.get('median')} months
- Historical mean response lag: about {cal.get('mean')} months
- Historical share responding within 12 months: {cal.get('lt12')}%
- Historical share responding within 24 months: {cal.get('lt24')}%
- Typical plausible range for observable institutional response: {cal.get('typical_range')}
- Main field friction: {cal.get('friction')}

Important calibration rule:
Estimate the delay for an OBSERVABLE institutional event, not the first internal awareness of the signal. A response is observable only when it becomes a policy action, firm demand event, school programme/curriculum response, intermediary translation event or standard/certification event. Do not output unrealistically immediate delays unless the episode clearly describes an already operational downstream response. Anchor your estimate near the historical pathway distribution while still adjusting for the specific event. Treat this as hybrid calibration / distributional replay rather than direct event reconstruction.
"""
    if prompt_style == "short":
        calibration_block = calibration_block.replace("Important calibration rule:\n", "")
    elif prompt_style == "long" and calibrated:
        calibration_block += "\nEmphasize that the model output is a structured mechanism cue, not a factual reconstruction of private organizational reasoning.\n"
    return f"""You are a constrained role-based agent in a field-theoretic institutional simulator.

Role: {role}
Field position: {info['field_position']}
Dominant capital: {info['capital']}
Decision instruction: {info['instruction']}
{calibration_block}

Task:
{task_intro}
Use the historical memory examples as behavioural conditioning.
Infer a plausible response delay from field position, capital, constraints, historical pathway calibration and signal strength.

Return ONLY one compact JSON object. No markdown, no prose, no code fences.
Use exactly these fields and keep explanation short.

Allowed action_type values:
- respond_now
- delay
- request_coordination
- ignore

Important:
- Use request_coordination for anything about translation, reporting, standards, meetings, forums, or cross-actor coordination.
- Use respond_now only for immediate action.
- Do not invent new action_type labels.

Allowed dominant_constraint labels:
- policy interpretation
- standard consensus
- coordination
- market incentive
- curriculum approval
- other

JSON keys:
{{
  "agent_role": "{role}",
  "perceived_urgency": number between 0 and 1,
  "perceived_feasibility": number between 0 and 1,
  "dominant_constraint": string,
  "action_type": one of ["respond_now", "delay", "request_coordination", "ignore"],
  "expected_delay_months": integer between 0 and 240,
  "coordination_request": array of coordination mechanism tags chosen from ["policy translation", "administrative reporting", "standard negotiation", "school-enterprise coordination", "curriculum alignment", "resource mobilization", "certification alignment", "forum / meeting", "brokerage / intermediary", "other coordination"],
  "explanation": string short enough to fit in one sentence,
  "confidence": number between 0 and 1
}}

Episode data:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def load_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    print("\n模型设备检查：")
    print(f" CUDA 是否可用：{torch.cuda.is_available()}")
    print(f" 模型所在设备  ：{next(model.parameters()).device}\n")
    return tokenizer, model


def generate_json(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": "You output strict JSON only. No markdown."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return generated.strip()


def parse_json(text: str) -> dict[str, Any]:
    try:
        parsed = extract_structured_fields(text)
        if parsed:
            parsed["dominant_constraint"] = normalize_mechanism(parsed.get("dominant_constraint", ""))
            parsed["action_type"] = infer_action_type(parsed.get("action_type", ""), parsed.get("decision", ""), parsed.get("explanation", ""))
            parsed["parse_error"] = False
            return parsed
    except Exception:
        pass
    return {
        "parse_error": True,
        "raw_text": text[:1000],
        "expected_delay_months": np.nan,
        "perceived_urgency": np.nan,
        "perceived_feasibility": np.nan,
        "dominant_constraint": "",
        "action_type": "",
        "decision": "",
        "explanation": "",
        "latent_reason": "",
        "coordination_request": [],
        "confidence": 0,
    }


def clamp_num(value: Any, low: float, high: float) -> float:
    try:
        v = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(v):
        return np.nan
    return max(low, min(high, v))


def resolve_delay_series(df: pd.DataFrame) -> pd.Series:
    if "expected_delay_months" in df.columns and df["expected_delay_months"].notna().any():
        return df["expected_delay_months"]
    if "sim_delay_months" in df.columns:
        return df["sim_delay_months"]
    return pd.Series([np.nan] * len(df), index=df.index)


def load_replay_results(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "episode_index" not in df.columns:
        raise ValueError(f"Missing episode_index in {path}")
    for col in [
        "dominant_constraint",
        "action_type",
        "decision",
        "explanation",
        "latent_reason",
        "expected_delay_months",
        "sim_delay_months",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    df["model_name"] = model_name
    df["mechanism_norm"] = [
        infer_mechanism(d, e, a, r)
        for d, e, a, r in zip(
            df.get("dominant_constraint", ""),
            df.get("explanation", ""),
            df.get("action_type", ""),
            df.get("decision", ""),
        )
    ]
    df["action_type_norm"] = [
        infer_action_type(a, d, e)
        for a, d, e in zip(
            df.get("action_type", ""),
            df.get("decision", ""),
            df.get("explanation", ""),
        )
    ]
    df["delay_months"] = pd.to_numeric(resolve_delay_series(df), errors="coerce")
    return df


def prepare_comparison_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "dominant_constraint" not in frame.columns:
        frame["dominant_constraint"] = ""
    if "action_type" not in frame.columns:
        if "decision" in frame.columns:
            frame["action_type"] = frame["decision"]
        else:
            frame["action_type"] = ""
    if "expected_delay_months" not in frame.columns and "sim_delay_months" in frame.columns:
        frame["expected_delay_months"] = frame["sim_delay_months"]
    frame["mechanism_norm"] = [
        infer_mechanism(d, e, a, r)
        for d, e, a, r in zip(
            frame.get("dominant_constraint", ""),
            frame.get("explanation", ""),
            frame.get("action_type", ""),
            frame.get("decision", ""),
        )
    ]
    frame["action_type_norm"] = [
        infer_action_type(a, d, e)
        for a, d, e in zip(
            frame.get("action_type", ""),
            frame.get("decision", ""),
            frame.get("explanation", ""),
        )
    ]
    frame["delay_months"] = pd.to_numeric(resolve_delay_series(frame), errors="coerce")
    return frame


def compare_models(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    left = prepare_comparison_frame(left)
    right = prepare_comparison_frame(right)
    join_cols = ["episode_index"]
    if "chain_id" in left.columns and "chain_id" in right.columns:
        join_cols = ["episode_index", "chain_id"]
    merged = left.merge(
        right,
        on=join_cols,
        how="inner",
        suffixes=(f"__{left_name}", f"__{right_name}"),
        validate="one_to_one",
    )
    merged["mechanism_match"] = merged[f"mechanism_norm__{left_name}"] == merged[f"mechanism_norm__{right_name}"]
    merged["action_match"] = merged[f"action_type_norm__{left_name}"] == merged[f"action_type_norm__{right_name}"]
    merged["delay_abs_diff"] = (
        merged[f"delay_months__{left_name}"] - merged[f"delay_months__{right_name}"]
    ).abs()

    summary = pd.DataFrame(
        [
            {
                "left_model": left_name,
                "right_model": right_name,
                "n_compared": int(len(merged)),
                "mechanism_match_pct": round(float(merged["mechanism_match"].mean() * 100), 1) if len(merged) else np.nan,
                "action_match_pct": round(float(merged["action_match"].mean() * 100), 1) if len(merged) else np.nan,
                "left_mean_delay": round(float(merged[f"delay_months__{left_name}"].mean()), 2),
                "right_mean_delay": round(float(merged[f"delay_months__{right_name}"].mean()), 2),
                "left_median_delay": round(float(merged[f"delay_months__{left_name}"].median()), 2),
                "right_median_delay": round(float(merged[f"delay_months__{right_name}"].median()), 2),
                "left_lt12_pct": round(float((merged[f"delay_months__{left_name}"] < 12).mean() * 100), 1),
                "right_lt12_pct": round(float((merged[f"delay_months__{right_name}"] < 12).mean() * 100), 1),
                "left_lt24_pct": round(float((merged[f"delay_months__{left_name}"] < 24).mean() * 100), 1),
                "right_lt24_pct": round(float((merged[f"delay_months__{right_name}"] < 24).mean() * 100), 1),
                "mean_abs_delay_diff": round(float(merged["delay_abs_diff"].mean()), 2),
            }
        ]
    )

    mech_counts = pd.crosstab(
        merged[f"mechanism_norm__{left_name}"],
        merged[f"mechanism_norm__{right_name}"],
    )
    episode_detail = merged[
        join_cols
        + [
            f"mechanism_norm__{left_name}",
            f"mechanism_norm__{right_name}",
            f"action_type_norm__{left_name}",
            f"action_type_norm__{right_name}",
            f"delay_months__{left_name}",
            f"delay_months__{right_name}",
            "mechanism_match",
            "action_match",
            "delay_abs_diff",
        ]
    ].copy()
    episode_detail = episode_detail.rename(
        columns={
            f"mechanism_norm__{left_name}": f"mechanism_{left_name}",
            f"mechanism_norm__{right_name}": f"mechanism_{right_name}",
            f"action_type_norm__{left_name}": f"action_{left_name}",
            f"action_type_norm__{right_name}": f"action_{right_name}",
            f"delay_months__{left_name}": f"delay_{left_name}",
            f"delay_months__{right_name}": f"delay_{right_name}",
        }
    )
    return summary, mech_counts, episode_detail


def write_comparison_report(
    out_dir: Path,
    summary: pd.DataFrame,
    mech_counts: pd.DataFrame,
    episode_detail: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(table_dir / "model_comparison_summary.csv", index=False, encoding="utf-8-sig")
    mech_counts.to_csv(table_dir / "mechanism_contingency.csv", encoding="utf-8-sig")
    episode_detail.to_csv(table_dir / "episode_comparison.csv", index=False, encoding="utf-8-sig")
    summary.to_markdown(table_dir / "model_comparison_summary.md", index=False)
    mech_counts.to_markdown(table_dir / "mechanism_contingency.md")
    episode_detail.head(50).to_markdown(table_dir / "episode_comparison_head.md", index=False)

    report = [
        "# Results 4 model comparison",
        "",
        f"Compared models: {left_name} vs {right_name}",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Mechanism contingency",
        "",
        mech_counts.to_markdown(),
        "",
        "## Interpretation",
        "",
        "- Compare mechanism match and action match rather than raw month-by-month delay equality.",
        "- If one model consistently predicts shorter delays, treat that as evidence that raw LLM delay is not robust.",
        "- Use the mechanism labels as the main diagnostic output in the manuscript.",
    ]
    (out_dir / "comparison_report.md").write_text("\n".join(report), encoding="utf-8")

    payload = {
        "left_model": left_name,
        "right_model": right_name,
        "n_compared": int(len(episode_detail)),
        "output_dir": str(out_dir),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_model_comparison_summary(results_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    names = list(results_map.keys())
    for name, df in results_map.items():
        valid = df[df["sim_delay_months"].notna()].copy()
        rows.append(
            {
                "model": name,
                "n": int(len(valid)),
                "mean_delay": round(float(valid["sim_delay_months"].mean()), 2) if len(valid) else np.nan,
                "median_delay": round(float(valid["sim_delay_months"].median()), 2) if len(valid) else np.nan,
                "lt12_pct": round(float((valid["sim_delay_months"] < 12).mean() * 100), 1) if len(valid) else np.nan,
                "lt24_pct": round(float((valid["sim_delay_months"] < 24).mean() * 100), 1) if len(valid) else np.nan,
            }
        )

    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            left = prepare_comparison_frame(results_map[left_name])
            right = prepare_comparison_frame(results_map[right_name])
            join_cols = ["episode_index"]
            if "chain_id" in left.columns and "chain_id" in right.columns:
                join_cols = ["episode_index", "chain_id"]
            merged = left.merge(
                right,
                on=join_cols,
                how="inner",
                suffixes=(f"__{left_name}", f"__{right_name}"),
                validate="one_to_one",
            )
            rows.append(
                {
                    "model": f"{left_name} vs {right_name}",
                    "n": int(len(merged)),
                    "mean_delay": np.nan,
                    "median_delay": np.nan,
                    "lt12_pct": np.nan,
                    "lt24_pct": np.nan,
                    "mechanism_match_pct": round(float((merged[f"mechanism_norm__{left_name}"] == merged[f"mechanism_norm__{right_name}"]).mean() * 100), 1) if len(merged) else np.nan,
                    "action_match_pct": round(float((merged[f"action_type_norm__{left_name}"] == merged[f"action_type_norm__{right_name}"]).mean() * 100), 1) if len(merged) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def build_pairwise_validation_tables(results_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    mech_rows: list[dict[str, Any]] = []
    names = list(results_map.keys())
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            summary, mech_counts, episode_detail = compare_models(
                results_map[left_name],
                results_map[right_name],
                left_name,
                right_name,
            )
            summary_row = summary.iloc[0].to_dict()
            pair_rows.append(
                {
                    "left_model": left_name,
                    "right_model": right_name,
                    "n_compared": int(summary_row.get("n_compared", len(episode_detail))),
                    "mechanism_match_pct": summary_row.get("mechanism_match_pct", np.nan),
                    "action_match_pct": summary_row.get("action_match_pct", np.nan),
                    "mean_abs_delay_diff": summary_row.get("mean_abs_delay_diff", np.nan),
                    "left_mean_delay": summary_row.get("left_mean_delay", np.nan),
                    "right_mean_delay": summary_row.get("right_mean_delay", np.nan),
                    "left_lt12_pct": summary_row.get("left_lt12_pct", np.nan),
                    "right_lt12_pct": summary_row.get("right_lt12_pct", np.nan),
                    "left_lt24_pct": summary_row.get("left_lt24_pct", np.nan),
                    "right_lt24_pct": summary_row.get("right_lt24_pct", np.nan),
                }
            )
            for left_mech in mech_counts.index:
                for right_mech in mech_counts.columns:
                    mech_rows.append(
                        {
                            "pair": f"{left_name} vs {right_name}",
                            "left_model": left_name,
                            "right_model": right_name,
                            "left_mechanism": left_mech,
                            "right_mechanism": right_mech,
                            "count": int(mech_counts.loc[left_mech, right_mech]),
                        }
                    )
    pairwise_summary = pd.DataFrame(pair_rows)
    if mech_rows:
        pairwise_mechanisms = pd.DataFrame(mech_rows)
    else:
        pairwise_mechanisms = pd.DataFrame()
    return pairwise_summary, pairwise_mechanisms


def run_api_model_replay(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    model_name: str,
    out_dir: Path,
    max_new_tokens: int,
    flush_every: int,
    api_base_url: str,
    api_key: str,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, Path, str]:
    out_dir = Path(out_dir)
    table_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(exist_ok=True)
    model_label = slugify_model_name(model_name)

    episodes.to_csv(out_dir / "historical_replay_episodes.csv", index=False, encoding="utf-8-sig")
    started = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "model_label": model_label,
        "episodes": int(len(episodes)),
        "status": "loading_model",
        "settings": settings,
    }
    (out_dir / "run_status.json").write_text(json.dumps(json_safe_dict(started), ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    result_path = out_dir / "llm_replay_results_partial.csv"
    payloads = [
        (int(i), row, events, settings, api_base_url.strip(), max_new_tokens, api_key.strip(), model_name, API_TIMEOUT_SECONDS, False, float(settings.get("temperature", 0.0)))
        for i, row in episodes.iterrows()
    ]
    workers = 1 if "reasoner" in model_name.lower() else API_WORKERS
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(api_replay_worker, payload) for payload in payloads]
        for idx, future in enumerate(
            tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"运行bar[{model_label}]"),
            start=1,
        ):
            rows.append(future.result())
            if idx % flush_every == 0:
                pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
                status = {
                    "started_at": started["started_at"],
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "model": model_name,
                    "model_label": model_label,
                    "episodes": int(len(episodes)),
                    "completed": int(idx),
                    "status": "running",
                    "settings": settings,
                }
                (out_dir / "run_status.json").write_text(json.dumps(json_safe_dict(status), ensure_ascii=False, indent=2), encoding="utf-8")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "llm_replay_results.csv", index=False, encoding="utf-8-sig")
    by_pathway, aggregate = summarize(results)
    by_pathway.to_csv(table_dir / "validation_by_pathway.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(table_dir / "validation_aggregate.csv", index=False, encoding="utf-8-sig")
    by_pathway.to_markdown(table_dir / "validation_by_pathway.md", index=False)
    aggregate.to_markdown(table_dir / "validation_aggregate.md", index=False)
    status = {
        "started_at": started["started_at"],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "model_label": model_label,
        "episodes": int(len(episodes)),
        "completed": int(len(results)),
        "status": "completed",
        "output_dir": str(out_dir),
        "settings": settings,
    }
    (out_dir / "run_status.json").write_text(json.dumps(json_safe_dict(status), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return results, out_dir, model_label


def run_ark_model_replay(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    model_name: str,
    out_dir: Path,
    max_new_tokens: int,
    flush_every: int,
    api_key: str,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, Path, str]:
    out_dir = Path(out_dir)
    table_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(exist_ok=True)
    model_label = slugify_model_name(model_name)

    episodes.to_csv(out_dir / "historical_replay_episodes.csv", index=False, encoding="utf-8-sig")
    started = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "model_label": model_label,
        "episodes": int(len(episodes)),
        "status": "loading_model",
        "settings": settings,
    }
    (out_dir / "run_status.json").write_text(json.dumps(json_safe_dict(started), ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    result_path = out_dir / "llm_replay_results_partial.csv"
    payloads = [
        (int(i), row, events, settings, max_new_tokens, api_key.strip(), model_name, API_TIMEOUT_SECONDS, float(settings.get("temperature", 0.0)))
        for i, row in episodes.iterrows()
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(ark_replay_worker, payload) for payload in payloads]
        for idx, future in enumerate(
            tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"运行bar[{model_label}]") ,
            start=1,
        ):
            rows.append(future.result())
            if idx % flush_every == 0:
                pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
                status = {
                    "started_at": started["started_at"],
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "model": model_name,
                    "model_label": model_label,
                    "episodes": int(len(episodes)),
                    "completed": int(idx),
                    "status": "running",
                    "settings": settings,
                }
                (out_dir / "run_status.json").write_text(json.dumps(json_safe_dict(status), ensure_ascii=False, indent=2), encoding="utf-8")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "llm_replay_results.csv", index=False, encoding="utf-8-sig")
    by_pathway, aggregate = summarize(results)
    by_pathway.to_csv(table_dir / "validation_by_pathway.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(table_dir / "validation_aggregate.csv", index=False, encoding="utf-8-sig")
    by_pathway.to_markdown(table_dir / "validation_by_pathway.md", index=False)
    aggregate.to_markdown(table_dir / "validation_aggregate.md", index=False)
    status = {
        "started_at": started["started_at"],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "model_label": model_label,
        "episodes": int(len(episodes)),
        "completed": int(len(results)),
        "status": "completed",
        "output_dir": str(out_dir),
        "settings": settings,
    }
    (out_dir / "run_status.json").write_text(json.dumps(json_safe_dict(status), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return results, out_dir, model_label


def run_single_replay(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    model_name: str,
    out_dir: Path,
    max_new_tokens: int,
    flush_every: int,
    model_label: str = "",
    mode: int = 1,
    api_base_url: str = "",
    api_key: str = "",
    settings: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, Path, str]:
    out_dir = Path(out_dir)
    table_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(exist_ok=True)
    model_label = model_label.strip() or slugify_model_name(model_name)
    settings = settings or {}

    episodes.to_csv(out_dir / "historical_replay_episodes.csv", index=False, encoding="utf-8-sig")
    started = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "model_label": model_label,
        "episodes": int(len(episodes)),
        "status": "loading_model",
        "settings": settings,
    }
    (out_dir / "run_status.json").write_text(json.dumps(json_safe_dict(started), ensure_ascii=False, indent=2), encoding="utf-8")

    client = None
    tokenizer = None
    model = None
    if mode == 0:
        if not api_base_url.strip() or not api_key.strip():
            raise RuntimeError("API mode selected but --api-base-url or --api-key is missing.")
        client = get_openai_client(api_base_url.strip(), api_key.strip())
    else:
        tokenizer, model = load_model(model_name)
    rows = []
    result_path = out_dir / "llm_replay_results_partial.csv"
    if mode == 0:
        payloads = [
            (int(i), row, events, settings, api_base_url.strip(), max_new_tokens, api_key.strip(), model_name, API_TIMEOUT_SECONDS, False, float(settings.get("temperature", 0.0)))
            for i, row in episodes.iterrows()
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=API_WORKERS) as executor:
            futures = [executor.submit(api_replay_worker, payload) for payload in payloads]
            for idx, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"运行bar[{model_label}]"), start=1):
                rows.append(future.result())
                if idx % flush_every == 0:
                    pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
                    status = {
                        "started_at": started["started_at"],
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "model": model_name,
                        "model_label": model_label,
                    "episodes": int(len(episodes)),
                    "completed": int(idx),
                    "status": "running",
                    "settings": settings,
                }
                    (out_dir / "run_status.json").write_text(json.dumps(json_safe_dict(status), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        for i, row in tqdm(episodes.iterrows(), total=len(episodes), desc=f"运行bar[{model_label}]"):
            memories = memory_examples(events, str(row["down_agent_role"]), str(row.get("up_sector", "")), row.get("up_year_final", np.nan))
            if not settings.get("include_memory", True):
                memories = []
            prompt = prompt_for_episode(
                row,
                memories,
                calibrated=bool(settings.get("calibrated", True)),
                validation_mode=False,
                prompt_style=str(settings.get("prompt_style", "full")),
                role_variant=str(settings.get("role_variant", "field_theoretic")),
            )
            raw = generate_json(tokenizer, model, prompt, max_new_tokens)
            parsed = parse_json(raw)
            sim_delay = clamp_num(parsed.get("expected_delay_months"), 0, 240)
            dominant_constraint = normalize_mechanism(parsed.get("dominant_constraint", ""))
            explanation = parsed.get("explanation", parsed.get("latent_reason", ""))
            action_type = infer_action_type(parsed.get("action_type", ""), parsed.get("decision", ""), explanation)
            out = {
                "episode_index": int(i),
                "chain_id": row.get("chain_id", ""),
                "chain_type_norm": row.get("chain_type_norm", ""),
                "upstream_event_id": row.get("upstream_event_id", ""),
                "downstream_event_id": row.get("downstream_event_id", ""),
                "downstream_role": row.get("down_agent_role", ""),
                "sector": row.get("up_sector", ""),
                "lag_months_num": float(row.get("lag_months_num", np.nan)),
                "sim_delay_months": sim_delay,
                "experiment_name": settings.get("experiment_name", ""),
                "variant": settings.get("variant", ""),
                "prompt_style": settings.get("prompt_style", ""),
                "role_variant": settings.get("role_variant", ""),
                "temperature": float(settings.get("temperature", 0.0)),
                "include_memory": bool(settings.get("include_memory", True)),
                "perceived_urgency": clamp_num(parsed.get("perceived_urgency"), 0, 1),
                "perceived_feasibility": clamp_num(parsed.get("perceived_feasibility"), 0, 1),
                "dominant_constraint": dominant_constraint,
                "action_type": action_type,
                "decision": parsed.get("decision", action_type),
                "coordination_request": json.dumps(normalize_coordination_tags(parsed.get("coordination_request", [])), ensure_ascii=False),
                "explanation": explanation,
                "latent_reason": parsed.get("latent_reason", explanation),
                "confidence": clamp_num(parsed.get("confidence"), 0, 1),
                "parse_error": bool(parsed.get("parse_error", False)),
                "raw_output": raw[:2000],
                "calibrated_prompt": bool(settings.get("calibrated", True)),
                "model_name": model_name,
                "model_label": model_label,
                "memory_examples_json": json.dumps(memories, ensure_ascii=False),
                "prompt_text": prompt,
            }
            rows.append(out)
            if (i + 1) % flush_every == 0:
                pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
                status = {
                    "started_at": started["started_at"],
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "model": model_name,
                    "model_label": model_label,
                    "episodes": int(len(episodes)),
                    "completed": int(i + 1),
                    "status": "running",
                }
                (out_dir / "run_status.json").write_text(json.dumps(json_safe_dict(status), ensure_ascii=False, indent=2), encoding="utf-8")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "llm_replay_results.csv", index=False, encoding="utf-8-sig")
    by_pathway, aggregate = summarize(results)
    by_pathway.to_csv(table_dir / "validation_by_pathway.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(table_dir / "validation_aggregate.csv", index=False, encoding="utf-8-sig")
    by_pathway.to_markdown(table_dir / "validation_by_pathway.md", index=False)
    aggregate.to_markdown(table_dir / "validation_aggregate.md", index=False)
    status = {
        "started_at": started["started_at"],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "model_label": model_label,
        "episodes": int(len(episodes)),
        "completed": int(len(results)),
        "status": "completed",
        "output_dir": str(out_dir),
        "settings": settings,
    }
    (out_dir / "run_status.json").write_text(json.dumps(json_safe_dict(status), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return results, out_dir, model_label


def resolve_run_settings(args: argparse.Namespace) -> dict[str, int]:
    """
    Compute runtime settings for normal or fast runs.

    Fast mode keeps the same logic and comparison structure, but trims
    episode count and generation length to reduce wall-clock time.
    """
    if not args.fast:
        return {
            "episodes": int(args.episodes),
            "max_new_tokens": int(args.max_new_tokens),
            "flush_every": int(args.flush_every),
        }
    return {
        "episodes": max(1, min(int(args.episodes), int(args.fast_episodes))),
        "max_new_tokens": max(1, min(int(args.max_new_tokens), int(args.fast_max_new_tokens))),
        "flush_every": max(1, min(int(args.flush_every), int(args.fast_flush_every))),
    }


def get_second_out_dir(args: argparse.Namespace, out_dir: Path) -> Path:
    second_model_label = args.second_model_label.strip() or slugify_model_name(args.second_model)
    if args.second_out_dir.strip():
        return Path(args.second_out_dir.strip())
    return out_dir.parent / f"{out_dir.name}_{second_model_label}"


def run_analysis_only(
    args: argparse.Namespace,
    out_dir: Path,
    default_left_path: Path,
    default_right_path: Path,
) -> dict[str, object]:
    left_path = Path(args.compare_left.strip()) if args.compare_left.strip() else default_left_path
    right_path = Path(args.compare_right.strip()) if args.compare_right.strip() else default_right_path
    left_name = args.compare_left_name.strip() or args.model_label.strip() or slugify_model_name(args.model)
    right_name = args.compare_right_name.strip() or args.second_model_label.strip() or slugify_model_name(args.second_model)
    left = load_replay_results(left_path, left_name)
    right = load_replay_results(right_path, right_name)
    summary, mech_counts, episode_detail = compare_models(left, right, left_name, right_name)
    write_comparison_report(out_dir, summary, mech_counts, episode_detail, left_name, right_name)
    return {
        "comparison_output_dir": str(out_dir),
        "left_path": str(left_path),
        "right_path": str(right_path),
        "n_compared": int(len(episode_detail)),
    }


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = results[results["sim_delay_months"].notna()].copy()
    rows = []
    for chain_type, sub in valid.groupby("chain_type_norm"):
        obs = sub["lag_months_num"].to_numpy(dtype=float)
        sim = sub["sim_delay_months"].to_numpy(dtype=float)
        rows.append(
            {
                "chain_type": chain_type,
                "n": len(sub),
                "observed_mean": round(float(np.mean(obs)), 2),
                "simulated_mean": round(float(np.mean(sim)), 2),
                "observed_median": round(float(np.median(obs)), 2),
                "simulated_median": round(float(np.median(sim)), 2),
                "observed_lt12_pct": round(float(np.mean(obs < 12) * 100), 1),
                "simulated_lt12_pct": round(float(np.mean(sim < 12) * 100), 1),
                "observed_lt24_pct": round(float(np.mean(obs < 24) * 100), 1),
                "simulated_lt24_pct": round(float(np.mean(sim < 24) * 100), 1),
                "mae_months": round(float(np.mean(np.abs(obs - sim))), 2),
            }
        )
    by_pathway = pd.DataFrame(rows)
    aggregate = pd.DataFrame(
        [
            {
                "n": len(valid),
                "parse_success_pct": round(float((~results.get("parse_error", False).fillna(False)).mean() * 100), 1),
                "observed_mean": round(float(valid["lag_months_num"].mean()), 2),
                "simulated_mean": round(float(valid["sim_delay_months"].mean()), 2),
                "observed_median": round(float(valid["lag_months_num"].median()), 2),
                "simulated_median": round(float(valid["sim_delay_months"].median()), 2),
                "observed_lt12_pct": round(float((valid["lag_months_num"] < 12).mean() * 100), 1),
                "simulated_lt12_pct": round(float((valid["sim_delay_months"] < 12).mean() * 100), 1),
                "observed_lt24_pct": round(float((valid["lag_months_num"] < 24).mean() * 100), 1),
                "simulated_lt24_pct": round(float((valid["sim_delay_months"] < 24).mean() * 100), 1),
            }
        ]
    )
    return by_pathway, aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--out-dir", default=str(DEFAULT_SUITE_OUT_DIR if DEFAULT_RUN_SUITE else Path(OUT_DIR)))
    parser.add_argument("--model", default=API_MODEL)
    parser.add_argument("--model-label", default=MODEL_LABEL)
    parser.add_argument("--second-model", default=VALIDATION_API_MODEL)
    parser.add_argument("--second-model-label", default=SECOND_MODEL_LABEL)
    parser.add_argument("--api-base-url", default=API_BASE_URL)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--validation-api-key", default=VALIDATION_API_KEY)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--flush-every", type=int, default=FLUSH_EVERY)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--calibrated", action="store_true", default=CALIBRATED)
    parser.add_argument("--no-calibrated", action="store_false", dest="calibrated")
    parser.add_argument("--include-memory", action="store_true", default=DEFAULT_INCLUDE_MEMORY)
    parser.add_argument("--no-include-memory", action="store_false", dest="include_memory")
    parser.add_argument("--role-variant", default=DEFAULT_ROLE_VARIANT)
    parser.add_argument("--prompt-style", default=DEFAULT_PROMPT_STYLE)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--run-mode", type=int, default=RUN_MODE)
    parser.add_argument("--run-suite", action="store_true", default=DEFAULT_RUN_SUITE)
    parser.add_argument("--compare-left", default="")
    parser.add_argument("--compare-right", default="")
    parser.add_argument("--compare-left-name", default="left")
    parser.add_argument("--compare-right-name", default="right")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    events, chains = read_data()

    if args.compare_left and args.compare_right:
        left = load_replay_results(Path(args.compare_left), args.compare_left_name)
        right = load_replay_results(Path(args.compare_right), args.compare_right_name)
        summary, mech_counts, episode_detail = compare_models(left, right, args.compare_left_name, args.compare_right_name)
        write_comparison_report(Path(COMPARE_OUT_DIR), summary, mech_counts, episode_detail, args.compare_left_name, args.compare_right_name)
        print(json.dumps({"output_dir": COMPARE_OUT_DIR, "n_compared": int(len(episode_detail))}, ensure_ascii=False, indent=2))
        return

    if args.run_suite and not args.out_dir.strip():
        args.out_dir = str(DEFAULT_SUITE_OUT_DIR)

    episodes = build_episode_table(events, chains, args.episodes, args.seed)
    settings_base = {
        "calibrated": bool(args.calibrated),
        "include_memory": bool(args.include_memory),
        "role_variant": args.role_variant,
        "prompt_style": args.prompt_style,
        "variant": args.variant,
        "temperature": float(args.temperature),
    }
    if args.run_suite:
        suite = make_experiment_suite(args)
        suite_results = []
        for exp in suite:
            exp_settings = {**settings_base, **exp}
            exp_out_dir = Path(args.out_dir) / exp["experiment_name"]
            if args.run_mode == 0:
                results, out_dir, model_label = run_api_model_replay(
                    events=events,
                    episodes=episodes,
                    model_name=args.model,
                    out_dir=exp_out_dir,
                    max_new_tokens=args.max_new_tokens,
                    flush_every=args.flush_every,
                    api_base_url=args.api_base_url,
                    api_key=args.api_key,
                    settings=exp_settings,
                )
            else:
                results, out_dir, model_label = run_single_replay(
                    events=events,
                    episodes=episodes,
                    model_name=args.model,
                    out_dir=exp_out_dir,
                    max_new_tokens=args.max_new_tokens,
                    flush_every=args.flush_every,
                    model_label=args.model_label,
                    mode=args.run_mode,
                    api_base_url=args.api_base_url,
                    api_key=args.api_key,
                    settings=exp_settings,
                )
            suite_results.append({"experiment": exp["experiment_name"], "output_dir": str(out_dir), "model_label": model_label, "n": int(len(results))})
        print(json.dumps({"suite": suite_results, "output_root": str(args.out_dir)}, ensure_ascii=False, indent=2))
        return

    if args.run_mode == 0:
        results, out_dir, model_label = run_api_model_replay(
            events=events,
            episodes=episodes,
            model_name=args.model,
            out_dir=Path(args.out_dir),
            max_new_tokens=args.max_new_tokens,
            flush_every=args.flush_every,
            api_base_url=args.api_base_url,
            api_key=args.api_key,
            settings=settings_base,
        )
    else:
        results, out_dir, model_label = run_single_replay(
            events=events,
            episodes=episodes,
            model_name=args.model,
            out_dir=Path(args.out_dir),
            max_new_tokens=args.max_new_tokens,
            flush_every=args.flush_every,
            model_label=args.model_label,
            mode=args.run_mode,
            api_base_url=args.api_base_url,
            api_key=args.api_key,
            settings=settings_base,
        )
    print(json.dumps({"output_dir": str(out_dir), "model_label": model_label, "n": int(len(results)), "mode": int(args.run_mode)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
