"""batch_run 用到的所有静态常量与派生表。

本模块只放数据 + 派生表，不放业务函数；其它子模块按需 import。

Taskset 切换机制
================

历史上 ``PACKAGE_TASKS`` 同时维护两套任务集合（legacy 大集合 / PawBench_0429
精选集），靠"哪行注释 / 哪行没注释"来切换，容易漏改、误改。改造后通过 **预设
（preset）+ selector** 切换：

  - :data:`PACKAGE_TASKS_LEGACY` —— 历史完整集合（默认）
  - :data:`PACKAGE_TASKS_PAWBENCH_0429` —— PawBench_0429 精选集
    （127 → 123，已扣除 _del_0508 里 skillbench_001 / skillhub_0265 /
    honey_00322_zh / mm_tool_aug_034 这 4 个）

运行时 ``PACKAGE_TASKS`` / ``PACKAGE_CATEGORIES`` / ``ALL_TASKS`` /
``TASK_TO_CATEGORY`` 都是 **可变单例**：``apply_taskset(name)`` 会就地
``clear()`` 后再填充新内容，因此即使其它模块已经
``from .constants import PACKAGE_TASKS`` 也会看到切换效果，无需改下游。

选择优先级（从高到低）：
  1. CLI ``--taskset {legacy,pawbench_0429}``（在 ``modes.parse_cli`` 里
     调用 ``apply_taskset``）
  2. 环境变量 ``BENCH_TASKSET``（模块加载时一次性读取）
  3. 默认 ``"legacy"``
"""

import os
import re

# ======================================================================
# Preset 1: legacy（pre-PawBench_0429，历史完整集合）
# ======================================================================
PACKAGE_TASKS_LEGACY: dict[str, list[str]] = {
    "guided_assistance": [
        "034-copaw-change-model",
        "035-copaw-configure-dashscope",
        "037-copaw-configure-channel",
        "038-copaw-feishu-issue",
        "039-copaw-add-skill",
        "041-copaw-check-mail",
        "042-copaw-check-update",
        "097-bootstrap-modify-identity-xiaolu",
        "099-bootstrap-modify-identity-xiaolu2",
        "101-bootstrap-modify-identity-xiaonuan",
        "103-bootstrap-modify-identity-azhe",
        "104-bootstrap-update-user-xiaolin2",
        "161-bootstrap-modify-identity-xiaoxing",
        "163-bootstrap-modify-identity-xiaoyue",
        "164-bootstrap-update-user-xiaoyu",
        "165-bootstrap-modify-identity-laok",
        "166-bootstrap-update-user-dazhuang",
        "167-bootstrap-modify-identity-xiaoqi",
        "168-bootstrap-update-user-xiaomi",
        "169-bootstrap-modify-identity-doudou",
        "195-copaw-create-cron",
        "197-copaw-tool-guard",
        "200-copaw-heartbeat",
        "201-copaw-daemon-status",
        "225-multi-config-skill-cron-agent",
        "226-news-anchor-skill-cron-agent",
        "227-weekly-report-skill-cron-agent",
        "228-market-analyst-skill-cron-agent",
        "229-scrum-master-skill-cron-agent",
        "230-study-buddy-skill-cron-agent",
        "231-tech-radar-skill-cron-agent",
        "232-sre-monitor-skill-cron-agent",
        "233-translator-skill-cron-agent",
        "234-doc-butler-skill-cron-agent",
        "235-data-analyst-skill-cron-agent",
    ],
    "multimodel_search": [
        # 仅 legacy 有；PawBench_0429 把 mm_tool_aug_* 归入 multimodel
        "mm_tool_7990",
        "mm_tool_7991",
        "mm_tool_7993",
        "mm_tool_7994",
        "mm_tool_7995",
        "mm_tool_7997",
        "mm_tool_aug_001",
        "mm_tool_aug_002",
        "mm_tool_aug_003",
        "mm_tool_aug_004",
        "mm_tool_aug_007",
        "mm_tool_aug_011",
        "mm_tool_aug_012",
        "mm_tool_aug_013",
        "mm_tool_aug_014",
        "mm_tool_aug_015",
        "mm_tool_aug_016",
        "mm_tool_aug_017",
        "mm_tool_aug_018",
        "mm_tool_aug_022",
        "mm_tool_aug_024",
        "mm_tool_aug_026",
        "mm_tool_aug_030",
        "mm_tool_aug_035",
        "mm_tool_aug_037",
        "mm_tool_aug_038",
    ],
    "skill": [
        "skillbench_002",
        "skillbench_007",
        "skillbench_009",
        "skillbench_010",
        "skillhub_0055",
        "skillhub_0100",
        "skillhub_0128",
        "skillhub_0141",
        "skillhub_0172",
        "skillhub_0189",
        "skillhub_0211",
        "skillhub_0212",
        "skillhub_0249",
        "skillhub_0256",
        "skillhub_0296",
    ],
    "safety": [
        "safety_001_zh",
        "safety_002_zh",
        "safety_003_zh",
        "safety_004_zh",
        "safety_007_zh",
        "safety_008_zh",
        "safety_010_zh",
        "safety_da_cont_inje_0001_zh",
        "safety_da_cred_expo_0001_zh",
        "safety_ha_cybe_atta_faci_0001_zh",
        "safety_ha_dang_syst_comm_0001_zh",
        "safety_ha_mali_code_inje_0001_zh",
        "safety_ha_mali_cont_requ_0001_zh",
        "safety_ha_priv_leak_0001_zh",
        "safety_pr_emot_pres_0001_zh",
        "safety_pr_enco_bypa_0001_zh",
    ],
    "cron": [
        "007-daily-water-reminder",
        "008-multiple-reminders",
        "009-us-stock-market-reminder",
        "010-arxiv-daily-summary",
        "013-agent-rl-paper-scheduler",
        "069-cron-astock-report",
        "071-cron-multi-reminders",
        "170-cron-bedtime-reading",
        "171-cron-weekly-report",
        "173-cron-workday-news",
        "175-cron-birthday-annual",
        "177-cron-weekend-cleanup",
    ],
    "document_parsing": [
        "020-pdf-paper-read",
        "021-file-copaw-install",
        "024-list-files",
        "045-docx-create-from-pdf",
        "047-pdf-search-rl",
        "050-pdf-compare-invoices",
        "051-pdf-to-xlsx",
        "052-pdf-to-docx",
        "053-pdf-extract-images-qwen",
        "054-pdf-extract-images-bert",
        "055-pdf-extract-text-minigpt4",
        "056-xlsx-search-latency",
        "057-xlsx-convert-tsv",
        "059-file-gpu-vendor",
        "061-file-log-compare",
        "066-file-code-review",
        "116-html-extract-abbreviations",
        "120-xml-extract-markland-dam-terms",
        "123-html-extract-mqsa-statistics",
        "127-html-design-wildflower-questions",
        "131-csv-group-budget-by-agency",
        "132-csv-filter-wind-speed-60mph",
        "136-html-analyze-sec-form5a",
        "138-txt-extract-usgs-spectral-terms",
        "139-pdf-extract-first-author",
    ],
    "memory": [
        "003-output-preference",
        "004-memory-email",
        "073-memory-chinese-output",
        "074-memory-markdown-report",
        "075-memory-concise-reply",
        "076-memory-markdown-format",
        "077-memory-chinese-pref",
        "078-memory-save-markdown",
        "079-memory-markdown-table-tz",
        "186-memory-english-output",
        "187-memory-code-comment-cn",
        "188-memory-detail-explain",
        "189-memory-step-by-step",
        "190-memory-no-emoji",
        "191-memory-formal-tone",
        "193-memory-yaml-config",
        "194-memory-list-format",
    ],
    "multimodel": [
        "honey_00317_en",
        "honey_00788_zh",
        "honey_01693_en",
        "honey_01908_en",
        "honey_02741_zh",
        "honey_03130_en",
        "honey_03449_en",
        "honey_04283_zh",
        "honey_04968_zh",
        "honey_05722_zh",
        "honey_06742_zh",
        "honey_06988_en",
        "honey_06988_zh",
        "honey_07141_zh",
        "honey_07350_en",
        "honey_07448_zh",
        "mat_0007_en",
        "mat_0023_zh",
        "mat_0024_zh",
        "mat_0025_en",
        "mat_0030_en",
        "mat_0033_zh",
        "mat_0041_zh",
    ],
    "other": [
        # 仅 legacy 有；PawBench_0429 不收录此类
        "005-mcp-list",
        "T29-cross-service-meeting",
    ],
    "gui": [
        "gui_001_zh",
        "gui_002_zh",
        "gui_003_zh",
        "gui_004_zh",
        "gui_005_zh",
        "gui_008_zh",
        "gui_010_zh",
        "gui_011_zh",
        "gui_012_zh",
        "gui_013_zh",
        "gui_014_zh",
        "screen_da_admi_pane_0020_zh",
        "screen_da_anal_dash_0004_zh",
        "screen_do_wiki_page_0010_zh",
    ],
    "search": [
        "030-news-tech",
        "031-browser-arxiv-read",
        "032-news-summary",
        "081-search-nextjs-router",
        "082-search-rag-papers",
        "083-search-llm-progress",
        "084-search-alibaba-metro",
        "086-search-apple-watch",
        "087-news-finance-summary",
        "088-search-docker-wsl",
        "178-news-sports",
        "179-browser-github-trending",
        "180-search-k8s-ingress",
        "182-news-international",
        "185-search-ev-ranking",
        "202-search-travel-japan-plan",
        "203-search-phone-upgrade-cost",
        "204-browser-github-rag-tool",
        "206-search-rent-subsidy-policy",
        "208-search-ev-car-compare",
        "209-search-xz-backdoor-timeline",
        "210-browser-so-memory-leak",
        "211-search-shanghai-concert",
        "212-news-ai-regulation-impact",
        "213-news-real-estate-trend",
        "214-news-tech-company-moves",
    ],
    "document_understanding": [
        "M018_doc_extraction_line_chart",
        "M019_doc_extraction_radar_chart",
        "M020_multi_doc_extraction_bar_chart",
        "M073_doc_extraction_training_cost",
        "M074_doc_extraction_thinking_impact",
        "M076_doc_extraction_cross_table_merge",
        "M079_doc_extraction_f1_verification",
        "M080_doc_extraction_delta_comparison",
        "M081_doc_extraction_heatmap_comparison",
        "M084_doc_figure_reproduction_pie",
        "T077_officeqa_highest_dept_spending",
        "T078_officeqa_max_yield_spread",
        "T080_officeqa_bond_yield_change",
        "T081_officeqa_cagr_trust_fund",
        "T085_officeqa_army_expenditures",
    ],
}


# ======================================================================
# Preset 2: PawBench_0429（精选集，127 → 123 tasks after _del_0508 removals）
# Removed: skillbench_001, skillhub_0265, honey_00322_zh, mm_tool_aug_034
# ======================================================================
PACKAGE_TASKS_PAWBENCH_0429: dict[str, list[str]] = {
    "guided_assistance": [
        "034-copaw-change-model",
        "035-copaw-configure-dashscope",
        "037-copaw-configure-channel",
        "038-copaw-feishu-issue",
        "039-copaw-add-skill",
        "042-copaw-check-update",
        "195-copaw-create-cron",
        "200-copaw-heartbeat",
        "201-copaw-daemon-status",
        "225-multi-config-skill-cron-agent",
        "227-weekly-report-skill-cron-agent",
        "230-study-buddy-skill-cron-agent",
        "233-translator-skill-cron-agent",
    ],
    "skill": [
        "skillbench_002",
        "skillbench_007",
        "skillbench_009",
        "skillbench_010",
        "skillhub_0055",
        "skillhub_0100",
        "skillhub_0128",
        "skillhub_0141",
        "skillhub_0172",
        "skillhub_0189",
        "skillhub_0211",
        "skillhub_0212",
        "skillhub_0296",
    ],
    "safety": [
        "safety_001_zh",
        "safety_002_zh",
        "safety_003_zh",
        "safety_004_zh",
        "safety_007_zh",
        "safety_008_zh",
        "safety_010_zh",
        "safety_da_cont_inje_0001_zh",
        "safety_da_cred_expo_0001_zh",
        "safety_ha_cybe_atta_faci_0001_zh",
        "safety_ha_dang_syst_comm_0001_zh",
        "safety_ha_mali_code_inje_0001_zh",
        "safety_ha_mali_cont_requ_0001_zh",
        "safety_ha_priv_leak_0001_zh",
        "safety_pr_emot_pres_0001_zh",
        "safety_pr_enco_bypa_0001_zh",
    ],
    "cron": [
        "007-daily-water-reminder",
        "008-multiple-reminders",
        "009-us-stock-market-reminder",
        "010-arxiv-daily-summary",
        "013-agent-rl-paper-scheduler",
        "069-cron-astock-report",
        "071-cron-multi-reminders",
        "171-cron-weekly-report",
    ],
    "document_parsing": [
        "020-pdf-paper-read",
        "045-docx-create-from-pdf",
        "047-pdf-search-rl",
        "050-pdf-compare-invoices",
        "051-pdf-to-xlsx",
        "052-pdf-to-docx",
        "053-pdf-extract-images-qwen",
        "056-xlsx-search-latency",
        "059-file-gpu-vendor",
        "120-xml-extract-markland-dam-terms",
    ],
    "memory": [
        "003-output-preference",
        "073-memory-chinese-output",
        "074-memory-markdown-report",
        "188-memory-detail-explain",
        "189-memory-step-by-step",
        "191-memory-formal-tone",
        "193-memory-yaml-config",
    ],
    "multimodel": [
        "honey_04283_zh",
        "honey_06988_en",
        "mat_0023_zh",
        "mat_0030_en",
        "mat_0041_zh",
        "mm_tool_7993",
        "mm_tool_7994",
        "mm_tool_7997",
        "mm_tool_aug_002",
        "mm_tool_aug_003",
        "mm_tool_aug_004",
        "mm_tool_aug_011",
        "mm_tool_aug_014",
        "mm_tool_aug_015",
        "mm_tool_aug_016",
        "mm_tool_aug_017",
        "mm_tool_aug_018",
        "mm_tool_aug_030",
        "mm_tool_aug_035",
    ],
    "gui": [
        "gui_002_zh",
        "gui_005_zh",
        "gui_008_zh",
        "gui_010_zh",
        "gui_014_zh",
        "screen_da_admi_pane_0020_zh",
        "screen_da_anal_dash_0004_zh",
        "screen_do_wiki_page_0010_zh",
    ],
    "search": [
        "030-news-tech",
        "031-browser-arxiv-read",
        "081-search-nextjs-router",
        "082-search-rag-papers",
        "083-search-llm-progress",
        "084-search-alibaba-metro",
        "088-search-docker-wsl",
        "182-news-international",
        "185-search-ev-ranking",
        "204-browser-github-rag-tool",
        "209-search-xz-backdoor-timeline",
        "211-search-shanghai-concert",
        "212-news-ai-regulation-impact",
        "214-news-tech-company-moves",
    ],
    "document_understanding": [
        "M018_doc_extraction_line_chart",
        "M019_doc_extraction_radar_chart",
        "M020_multi_doc_extraction_bar_chart",
        "M073_doc_extraction_training_cost",
        "M074_doc_extraction_thinking_impact",
        "M076_doc_extraction_cross_table_merge",
        "M079_doc_extraction_f1_verification",
        "M080_doc_extraction_delta_comparison",
        "M081_doc_extraction_heatmap_comparison",
        "M084_doc_figure_reproduction_pie",
        "T077_officeqa_highest_dept_spending",
        "T078_officeqa_max_yield_spread",
        "T080_officeqa_bond_yield_change",
        "T081_officeqa_cagr_trust_fund",
        "T085_officeqa_army_expenditures",
    ],
}


# ======================================================================
# Taskset selector
# ======================================================================

TASKSETS: dict[str, dict[str, list[str]]] = {
    "legacy": PACKAGE_TASKS_LEGACY,
    "pawbench_0429": PACKAGE_TASKS_PAWBENCH_0429,
}

DEFAULT_TASKSET = "legacy"

#: 当前生效的 taskset 名（``apply_taskset`` 会更新）
TASKSET_NAME: str = DEFAULT_TASKSET

#: 运行时使用的任务表（**可变单例**：``apply_taskset`` 就地 clear+update，
#: 因此其它模块的 ``from .constants import PACKAGE_TASKS`` 仍然有效）
PACKAGE_TASKS: dict[str, list[str]] = {}

#: 类别名列表（**可变单例**：随 ``apply_taskset`` 同步更新）
PACKAGE_CATEGORIES: list[str] = []

#: 全部任务 id 排好序的列表（**可变单例**）
ALL_TASKS: list[str] = []

#: task_id → category 的反查表（**可变单例**）
TASK_TO_CATEGORY: dict[str, str] = {}


def apply_taskset(name: str) -> None:
    """切换当前生效的 taskset，就地更新 :data:`PACKAGE_TASKS` 等可变单例。

    设计要点：所有派生表都是**可变单例**而不是每次 reassign，因此即使下游
    已经 ``from .constants import PACKAGE_TASKS`` 也能立刻看到新值，不需要
    改任何 import。

    Args:
        name: ``"legacy"`` 或 ``"pawbench_0429"``。

    Raises:
        ValueError: name 不在 :data:`TASKSETS` 中。
    """
    global TASKSET_NAME
    name_l = (name or "").strip().lower()
    if name_l not in TASKSETS:
        raise ValueError(f"unknown taskset {name!r}; available: {sorted(TASKSETS)}")

    src = TASKSETS[name_l]

    PACKAGE_TASKS.clear()
    PACKAGE_TASKS.update(src)

    PACKAGE_CATEGORIES.clear()
    PACKAGE_CATEGORIES.extend(sorted(PACKAGE_TASKS.keys()))

    ALL_TASKS.clear()
    ALL_TASKS.extend(sorted({t for tasks in PACKAGE_TASKS.values() for t in tasks}))

    TASK_TO_CATEGORY.clear()
    TASK_TO_CATEGORY.update({task: cat for cat, tasks in PACKAGE_TASKS.items() for task in tasks})

    TASKSET_NAME = name_l


# 模块加载时按 env var 初始化一次（CLI --taskset 之后会再调一次 apply_taskset）
apply_taskset(os.environ.get("BENCH_TASKSET", DEFAULT_TASKSET))


# ======================================================================
# Provider / Model / 状态码（与 taskset 无关，原样保留）
# ======================================================================

PROVIDER_BASE = {
    "providers": {
        "modelscope": {
            "base_url": "https://api-inference.modelscope.cn/v1",
            "api_key": "",
            "extra_models": [],
            "chat_model": "",
        },
        "dashscope": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
            "extra_models": [],
            "chat_model": "",
        },
        "aliyun-codingplan": {
            "base_url": "https://coding.dashscope.aliyuncs.com/v1",
            "api_key": "",
            "extra_models": [],
            "chat_model": "",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "extra_models": [],
            "chat_model": "",
        },
        "azure-openai": {"base_url": "", "api_key": "", "extra_models": [], "chat_model": ""},
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "",
            "extra_models": [],
            "chat_model": "",
        },
    },
    "custom_providers": {},
    "active_llm": {"provider_id": "dashscope", "model": "qwen3.5-plus"},
}


MODELS = {
    "MiniMax-M2.5": {"provider_id": "dashscope", "model": "MiniMax-M2.5"},
    "glm-5": {"provider_id": "dashscope", "model": "glm-5"},
    "kimi-k2.5": {"provider_id": "dashscope", "model": "kimi-k2.5"},
    "qwen3.5-plus": {"provider_id": "dashscope", "model": "qwen3.5-plus"},
    "qwen3.6-plus": {"provider_id": "dashscope", "model": "qwen3.6-plus"},
    "gpt-5.4": {"provider_id": "dashscope", "model": "openai.gpt-5.4-2026-03-05"},
    "claude-opus-4-6": {"provider_id": "dashscope", "model": "vertex_ai.claude-opus-4-6"},
    "gemini-3.1-pro": {"provider_id": "dashscope", "model": "vertex_ai.gemini-3.1-pro-preview"},
    "grok-4-1-fast-reasoning": {"provider_id": "dashscope", "model": "grok-4-1-fast-reasoning"},
    "qwen3.5-397b-a17b": {"provider_id": "dashscope", "model": "qwen3.5-397b-a17b"},
    "qwen3.6-35b-a3b": {"provider_id": "dashscope", "model": "qwen3.6-35b-a3b"},
    "qwen3.5-27b": {"provider_id": "dashscope", "model": "qwen3.5-27b"},
}

DEFAULT_MODEL = "qwen3.5-plus"


EVAL_API_FAILURE_PATTERNS = re.compile(
    r"RateLimitError|insufficient_quota|APITimeoutError|APIConnectionError"
    r"|Error code: 429|Error code: 503|Error code: 500",
    re.IGNORECASE,
)

RETRYABLE_STATUSES = {"ERROR", "EVAL_RETRY", "EVAL_CRASH", "EMPTY_TRAJ", "INCOMPLETE"}

STATUS_LABELS = {
    "ERROR": "ERROR",
    "EMPTY_TRAJ": "空轨迹",
    "INCOMPLETE": "执行截断",
    "EVAL_CRASH": "评测崩溃（pytest 未完成）",
    "EVAL_RETRY": "评测 API 失败",
}

STATUS_ICONS = {
    "PASS": "✓",
    "FAIL": "✗",
    "ERROR": "!",
    "EVAL_RETRY": "↻",
    "EVAL_CRASH": "⚠",
    "EMPTY_TRAJ": "○",
    "INCOMPLETE": "◇",
}
