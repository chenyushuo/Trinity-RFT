"""Multimodal helpers + screenshot coherence grader.

Two responsibilities live here:

  1. Multimodal-search hallucination *grounding* helpers
     (``build_multimodal_search_hallucination_extra_context`` and friends),
     consumed by ``_grading.evaluate_search_hallucination`` to inject the
     visual-recognition fallback prompt when a session involved images.

  2. ``evaluate_screenshot_coherence`` and its supporting helpers
     (``extract_screenshot_paths``, ``_image_to_base64``,
     ``_evaluate_screenshot_coherence_once``).

Depends on ``_core`` (LLM model, session helpers, trial wrapper) and
``_prompts`` (multimodal grounding fragments). Does NOT depend on
``_grading`` so the dependency graph stays acyclic.
"""

import asyncio
import logging
import os
import re
from typing import Any

from openjudge.graders.schema import GraderError, GraderScore
from openjudge.models.schema.prompt_template import LanguageEnum

from ._core import (
    _DEFAULT_GRADING_TRIALS,
    _coerce_language,
    _trial_llm_grader,
    extract_assistant_thinking,
    extract_final_response,
    extract_tool_calls,
    get_vl_model,
)
from ._prompts import (
    _MM_SEARCH_HALLUCINATION_GROUNDING_FALLBACK_ZH,
    _MM_SEARCH_HALLUCINATION_GROUNDING_ZH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image-input detection across session formats
# ---------------------------------------------------------------------------

def _content_part_is_image_block(part: Any) -> bool:
    """user 消息 content 中单条是否表示图片（与多模态 API 多种块结构兼容）。"""
    if not isinstance(part, dict):
        return False
    t = part.get("type")
    if t in ("image", "image_url", "input_image"):
        return True
    if part.get("image_url"):
        return True
    # Anthropic / 部分编排：image 嵌在 source 里
    src = part.get("source") if isinstance(part.get("source"), dict) else None
    if src and src.get("type") in ("base64", "url", "file"):
        return True
    # 顶层 image 字段（非纯文本块）
    if part.get("image") is not None and t not in ("text", "thinking", "tool_use", "tool_result", None):
        return True
    return False


def _user_message_has_image(msg: dict) -> bool:
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, list):
        return any(_content_part_is_image_block(p) for p in content)
    if isinstance(content, str):
        # 极少数把路径写在纯文本里
        s = content
        return "image.jpg" in s or "/local_files/" in s or "attached image" in s.lower()
    return False


def _iter_session_user_messages(session: dict):
    """遍历 session 中可能出现的 user 消息（trajectory + memory）。"""
    for entry in session.get("agent", {}).get("_model_trajectory", []):
        for msg in entry.get("messages", []) or []:
            if isinstance(msg, dict):
                yield msg
    for turn in session.get("agent", {}).get("memory", {}).get("content", []):
        if not turn:
            continue
        for msg in turn:
            if isinstance(msg, dict) and msg.get("role") == "user":
                yield msg


def _session_user_message_has_attached_image(session: dict) -> bool:
    """首轮/任务中 user 侧是否包含图片块（模型可直接读图，无需 view_image）。"""
    for msg in _iter_session_user_messages(session):
        if _user_message_has_image(msg):
            return True
    return False


def _session_used_view_image_tool(session: dict) -> bool:
    """是否调用了 view_image 工具。"""
    for block in extract_tool_calls(session):
        if block.get("name") == "view_image":
            return True
    for entry in session.get("agent", {}).get("_model_trajectory", []):
        for item in entry.get("response", []) or []:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                if item.get("name") == "view_image":
                    return True
    return False


def session_has_multimodal_image_signal(session: dict) -> bool:
    """是否存在「用户传图」或「显式 view_image」任一信号。

    用于判断是否按多模态检索任务注入幻觉评测的 grounding 文案（不依赖像素输入 grader）。
    """
    return _session_user_message_has_attached_image(session) or _session_used_view_image_tool(session)


# ---------------------------------------------------------------------------
# Inline thinking extraction (some models emit </think> separator inline)
# ---------------------------------------------------------------------------

_THINK_TAG_RE = re.compile(r"</think>", re.IGNORECASE)


def _extract_inline_thinking(session: dict, max_chars: int = 12000) -> str:
    """从 content 中提取内嵌的 thinking（</think> 标记之前的文本）。

    部分模型将推理过程直接写在 assistant content 中，以 </think> 标记分隔，
    而非放在单独的 thinking 字段或 reasoning_content 中。
    本函数作为 extract_assistant_thinking 的补充 fallback。
    """
    full_response = extract_final_response(session)
    m = _THINK_TAG_RE.search(full_response)
    if m:
        raw = full_response[:m.start()].strip()
        if raw:
            return raw[:max_chars]
    return ""


def build_multimodal_search_hallucination_extra_context(
    session: dict,
    *,
    max_thinking_chars: int = 12000,
) -> str:
    """为「搜索幻觉」评测构造 context 补充段：无图片像素，仅注入 thinking 与说明。

    纯文本 HallucinationGrader 无法读图；注入后避免将「图中已识别的实体」误判为捏造。
    触发条件：**用户消息中带图**（直接读图）或 **调用 view_image**；二者满足其一即可。
    若有 thinking 则附上；若无 thinking 仍注入简短边界说明（避免纯文本检索任务误触发：无信号则返回空串）。
    """
    if not session_has_multimodal_image_signal(session):
        return ""
    thinking = extract_assistant_thinking(session).strip()
    if not thinking:
        thinking = _extract_inline_thinking(session, max_thinking_chars)
    if thinking:
        if len(thinking) > max_thinking_chars:
            thinking = thinking[:max_thinking_chars] + "\n[... thinking 已截断 ...]"
        return (
            _MM_SEARCH_HALLUCINATION_GROUNDING_ZH
            + "\n\n[Agent 视觉相关 thinking / 内部推理]\n"
            + thinking
        )
    return _MM_SEARCH_HALLUCINATION_GROUNDING_FALLBACK_ZH


# ---------------------------------------------------------------------------
# Screenshot coherence evaluation (multimodal: query + response + image)
# ---------------------------------------------------------------------------

def extract_screenshot_paths(session: dict) -> list[str]:
    """从 session 中提取 desktop_screenshot 工具产生的截图路径。"""
    paths: list[str] = []
    for turn in session.get("agent", {}).get("memory", {}).get("content", []):
        if not turn or len(turn) < 1:
            continue
        for msg in turn:
            if not isinstance(msg, dict):
                continue
            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result" and block.get("name") == "desktop_screenshot":
                    out = block.get("output", block.get("content", []))
                    if isinstance(out, list):
                        for item in out:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                m = re.search(r'"path"\s*:\s*"([^"]+)"', text)
                                if m:
                                    paths.append(m.group(1))
    # also check image blocks from view_image tool results
    for turn in session.get("agent", {}).get("memory", {}).get("content", []):
        if not turn:
            continue
        for msg in turn:
            if not isinstance(msg, dict):
                continue
            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    out = block.get("output", block.get("content", []))
                    if isinstance(out, list):
                        for item in out:
                            if isinstance(item, dict) and item.get("type") == "image":
                                src = item.get("source", {})
                                url = src.get("url", "")
                                if url and url not in paths:
                                    clean = url.replace("file://", "")
                                    if clean not in paths:
                                        paths.append(clean)
    return paths


def _image_to_base64(path: str) -> tuple[str, str] | None:
    """读取本地图片文件为 base64，返回 (b64_str, format)。"""
    import base64 as _b64
    if not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    fmt = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "png")
    with open(path, "rb") as f:
        data = _b64.standard_b64encode(f.read()).decode("ascii")
    return data, fmt


async def _evaluate_screenshot_coherence_once(
    session: dict,
    query: str,
    *,
    max_retries: int = 2,
    threshold: float = 3.0,
    language: LanguageEnum | str = LanguageEnum.ZH,
) -> GraderScore | GraderError:
    """单次截图-回复一致性评估：query + final_response 作为文本上下文，截图作为图片。"""
    from openjudge.graders.multimodal import ImageCoherenceGrader, MLLMImage

    language = _coerce_language(language)
    final_response = extract_final_response(session)
    screenshot_paths = extract_screenshot_paths(session)

    if not screenshot_paths:
        return GraderError(name="screenshot_coherence", error="未在 session 中找到截图")

    last_path = screenshot_paths[-1]
    img_data = _image_to_base64(last_path)
    if img_data is None:
        return GraderError(name="screenshot_coherence", error=f"截图文件不存在: {last_path}")

    b64_str, fmt = img_data
    image = MLLMImage(base64=b64_str, format=fmt)

    response_content = [
        f"用户的原始请求: {query}\n\n"
        f"请严格根据用户的原始请求判断截图是否完成了任务。"
        f"不要被 Agent 的自述所误导——即使 Agent 声称完成了任务，"
        f"如果截图内容与用户原始请求不符，也应给低分。",
        image,
        f"（注意：以下是 Agent 的自述，仅供参考，请以截图实际内容为准）\n"
        f"Agent 回复: {final_response}",
    ]

    grader = ImageCoherenceGrader(
        model=get_vl_model(),
        threshold=threshold / 5.0,
        language=language,
    )

    last_result: GraderScore | GraderError | None = None
    for attempt in range(max_retries + 1):
        try:
            result = await grader.aevaluate(response=response_content)
            return result
        except Exception as e:
            logger.warning("screenshot_coherence attempt %d error: %s", attempt, e)
            last_result = GraderError(name="screenshot_coherence", error=str(e))
            await asyncio.sleep(2 * (2 ** attempt))

    return last_result  # type: ignore[return-value]


async def evaluate_screenshot_coherence(
    session: dict,
    query: str,
    *,
    max_retries: int = 2,
    threshold: float = 3.0,
    trials: int = _DEFAULT_GRADING_TRIALS,
    language: LanguageEnum | str = LanguageEnum.ZH,
) -> GraderScore | GraderError:
    """评估截图与 query+response 的一致性（5 分制），多次 trial 取中位数。

    使用 OpenJudge 的 ImageCoherenceGrader：将 query 和 final_response 作为
    上下文，截图作为图片，评估图文一致性。
    """
    return await _trial_llm_grader(
        _evaluate_screenshot_coherence_once,
        trials=trials,
        session=session,
        query=query,
        max_retries=max_retries,
        threshold=threshold,
        language=language,
    )
