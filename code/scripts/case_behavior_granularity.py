from __future__ import annotations

from typing import Any

import pandas as pd


# The mapping follows 0327-agent-behaviour-table.docx. It is an interpretation
# layer for case traces and counterfactual figures; it does not change simulator parameters.
BEHAVIOR_MAP = [
    {
        "macro_dimension": "policy_coordination",
        "macro_label": "政府统筹协调",
        "field": "政府",
        "behavior_dimension": "统筹协调",
        "behavior_indicator": "规划布局设计；治理架构确立；专业/产业匹配预警",
        "capital_feature": "社会资本：政策制定权、政策权威性、资源配置权力",
        "habitus_feature": "层级传递、风险规避、程序理性",
    },
    {
        "macro_dimension": "firm_school_connectivity",
        "macro_label": "校企协同连接",
        "field": "产业",
        "behavior_dimension": "载体建设 / 办学建设",
        "behavior_indicator": "共建产教融合共同体；参与课程建设；提供生产性实训基地；提供生产性设备设施",
        "capital_feature": "经济资本：技术、市场与设备资源",
        "habitus_feature": "需求驱动、选择性合作、有限参与",
    },
    {
        "macro_dimension": "intermediary_translation",
        "macro_label": "跨场域翻译能力",
        "field": "跨场域相互作用",
        "behavior_dimension": "契约与组织融合",
        "behavior_indicator": "联合签署合作协议；成立理事会/董事会/秘书处；制定产教项目管理章程；建设供需对接平台",
        "capital_feature": "关系与组织资本：契约、数据共享与协调网络",
        "habitus_feature": "协调导向、程序协商、组织惯性",
    },
    {
        "macro_dimension": "school_adaptability",
        "macro_label": "学校办学适应性",
        "field": "教育",
        "behavior_dimension": "办学建设 / 育人建设",
        "behavior_indicator": "开发产业课程资源；专业动态调整；教材更新；根据用工需求招生；实施订单/学徒/现场工程师培养",
        "capital_feature": "文化资本：技能教育、师资与实践教学资源",
        "habitus_feature": "就业导向、路径依赖、调整周期约束",
    },
    {
        "macro_dimension": "standard_flexibility",
        "macro_label": "标准教学化弹性",
        "field": "产业—教育",
        "behavior_dimension": "标准转化 / 课程建设",
        "behavior_indicator": "运行标准制定；参与专业群动态调整；参与教材建设；调整技术技能评价内容并实施评价",
        "capital_feature": "技术资本与文化资本：标准、课程、评价资源",
        "habitus_feature": "标准化合作与程序理性之间的张力",
    },
]

MAP_DF = pd.DataFrame(BEHAVIOR_MAP)

SCENARIO_BEHAVIOR_MAP = {
    "baseline": {
        "scenario_macro_label": "基准状态",
        "scenario_fine_behavior": "不施加额外干预，保留原始主体行为",
    },
    "firm_school_ties": {
        "scenario_macro_label": "校企协同连接",
        "scenario_fine_behavior": "共建产教融合载体 / 参与课程与实训建设",
    },
    "translation_capacity": {
        "scenario_macro_label": "跨场域翻译能力",
        "scenario_fine_behavior": "联合协议、理事会章程与供需信息共享",
    },
    "platform_governance": {
        "scenario_macro_label": "政府统筹协调",
        "scenario_fine_behavior": "规划布局、治理架构与专业产业匹配预警",
    },
    "combined_reform": {
        "scenario_macro_label": "组合式主体行为改革",
        "scenario_fine_behavior": "统筹协调 + 校企载体 + 跨场域翻译 + 学校办学调整",
    },
}


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def infer_case_behavior(row: Any) -> dict[str, str]:
    """Map an existing case mechanism to one macro dimension and a traceable action."""
    constraint = _text(row.get("dominant_constraint", ""))
    chain = _text(row.get("chain_type_norm", ""))
    if constraint in {"policy interpretation", "coordination"} or chain == "policy_to_firm":
        macro = "policy_coordination" if constraint == "policy interpretation" else "firm_school_connectivity"
    elif constraint == "standard consensus" or chain == "standard_to_curriculum":
        macro = "standard_flexibility"
    elif constraint == "curriculum approval":
        macro = "school_adaptability"
    else:
        macro = "intermediary_translation"

    record = next(item for item in BEHAVIOR_MAP if item["macro_dimension"] == macro)
    if macro == "policy_coordination":
        fine = "政策转译细化 / 专业产业匹配预警"
    elif macro == "firm_school_connectivity":
        fine = "校企对接 / 共建产教融合载体"
    elif macro == "intermediary_translation":
        fine = "联合协议与理事会章程 / 供需信息共享"
    elif macro == "school_adaptability":
        fine = "专业动态调整 / 课程资源开发 / 教材更新"
    else:
        fine = "行业标准解读 / 标准教学化 / 技能评价调整"

    return {
        **record,
        "case_fine_behavior": fine,
        "case_bottleneck": constraint or "未标注",
    }


def add_behavior_columns(df: Any):
    mapped = df.apply(infer_case_behavior, axis=1, result_type="expand")
    existing = [column for column in mapped.columns if column in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df.join(mapped)
