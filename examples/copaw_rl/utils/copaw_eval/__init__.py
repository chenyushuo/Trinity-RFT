"""CoPaw 评估工具集 — 包入口。

此包将原先 2000+ 行的 ``copaw_eval.py`` 拆为多个内部子模块，并通过本 ``__init__``
re-export **稳定公开 API**。所有 ``benchmark_v3/**/test_outputs.py`` 仅依赖这里
列出的 16 个符号；子模块（前缀 ``_``）属于实现细节，外部不应直接 import。

Layered submodules (top-down):

    _prompts      — 纯 LLM prompt templates
    _core         — LLM singletons, session helpers, grader engine, assertions
    _context      — Unified tool-output pipeline + cleaners
    _multimodal   — Multimodal hallucination grounding + screenshot grader
    _trajectory   — Trajectory + safety trajectory graders
    _grading      — Correctness, hallucination (single-pass + MapReduce), relevance

Architectural notes (现行)：
  - Agent 上下文统一处理：``build_unified_entries`` 将 session 中所有工具调用合并、
    按工具类型派发 cleaner（browser snapshot / curl HTML / 通用 raw），并通过内容
    指纹跨工具去重。
  - Grader 共享同一份 context：correctness / search_relevance 走 100k 反向填充 →
    单次 grader（多 trial 中位）；hallucination 长度 > 30k 自动切到 MapReduce
    （Extract → Map 并发 → Reduce 多 trial），避免 LLM "lost in the middle"。
  - 多模态：``view_image`` 调用 *或* 用户消息含图片任一即触发 thinking 注入。
  - 轨迹：截断会话自动降分至 ≤ 2.0；安全轨迹支持外部 ``expected_behavior`` 注入。
  - 评分稳定性：所有 LLM grader 多 trial 取中位，trial 间分差 ≥ 2 触发
    ``_high_variance`` 标记提示人工复核。
"""

from openjudge.models.schema.prompt_template import (  # noqa: F401  (re-exported for convenience)
    LanguageEnum,
)

from ._core import (
    SAFETY_REFUSE_WARN_PATTERNS,
    assert_check,
    assert_grader_score,
    extract_assistant_thinking,
    extract_final_response,
    extract_tool_calls,
    get_llm_model,
    log_grader_score_line,
    read_session,
)
from ._grading import (
    evaluate_by_script,
    evaluate_correctness,
    evaluate_file_correctness,
    evaluate_search_hallucination,
    evaluate_search_relevance,
)
from ._multimodal import evaluate_screenshot_coherence
from ._pairwise import evaluate_pairwise_correctness
from ._trajectory import (
    evaluate_harmful_content,
    evaluate_safety_trajectory,
    evaluate_trajectory,
)

__all__ = [
    "SAFETY_REFUSE_WARN_PATTERNS",
    "assert_check",
    "assert_grader_score",
    "evaluate_by_script",
    "evaluate_correctness",
    "evaluate_file_correctness",
    "evaluate_harmful_content",
    "evaluate_safety_trajectory",
    "evaluate_screenshot_coherence",
    "evaluate_search_hallucination",
    "evaluate_search_relevance",
    "evaluate_pairwise_correctness",
    "evaluate_trajectory",
    "extract_assistant_thinking",
    "extract_final_response",
    "extract_tool_calls",
    "get_llm_model",
    "log_grader_score_line",
    "read_session",
]
