"""Correctness, hallucination (single-pass + MapReduce) and search relevance graders.

All three share the unified tool-output context built by ``_context``.
The hallucination grader auto-routes to MapReduce when total context length
exceeds ``_MAPREDUCE_CONTEXT_THRESHOLD`` (30 000 chars) to mitigate
``lost-in-the-middle`` issues with very long contexts.
"""

import asyncio
import json
import logging
import os
import statistics
from typing import Any

from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.graders.common.hallucination import HallucinationGrader
from openjudge.graders.common.relevance import RelevanceGrader
from openjudge.graders.schema import GraderError, GraderScore
from openjudge.models.schema.prompt_template import LanguageEnum

from ._context import (
    _get_context,
    _pack_entries_reverse_fill,
    build_unified_entries,
    extract_agent_capability_context,
)
from ._core import (
    _DEFAULT_GRADING_TRIALS,
    _HIGH_VARIANCE_RANGE_THRESHOLD,
    _coerce_language,
    _llm_raw_call,
    _parse_json_array,
    _parse_json_object,
    _run_grader_once,
    _trial_llm_grader,
    extract_final_response,
    get_llm_model,
    safe_grader_eval,
)
from ._multimodal import build_multimodal_search_hallucination_extra_context
from ._prompts import (
    _COPAW_HALLUCINATION_TEMPLATE,
    _MR_EXTRACT_CLAIMS_PROMPT,
    _MR_REDUCE_PROMPT,
    _MR_VERIFY_CHUNK_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Correctness core
# ---------------------------------------------------------------------------


async def _evaluate_correctness_core_once(
    session: dict,
    query: str,
    reference_response: str,
    *,
    max_retries: int = 2,
    threshold: int = 3,
    language: LanguageEnum | str = LanguageEnum.ZH,
) -> GraderScore | GraderError:
    """统一正确性评估核心（单次 + retry）。"""
    language = _coerce_language(language)
    grader = CorrectnessGrader(model=get_llm_model(), threshold=threshold, language=language)
    return await _run_grader_once(
        grader,
        dict(
            response=extract_final_response(session),
            query=query,
            reference_response=reference_response,
            context=_get_context(session),
        ),
        max_retries=max_retries,
        label="Correctness",
    )


async def _evaluate_correctness_core(
    session: dict,
    query: str,
    reference_response: str,
    *,
    max_retries: int = 2,
    threshold: int = 3,
    language: LanguageEnum | str = LanguageEnum.ZH,
    trials: int = _DEFAULT_GRADING_TRIALS,
) -> GraderScore | GraderError:
    """统一正确性评估（multi-trial 取中位数）。"""
    return await _trial_llm_grader(
        _evaluate_correctness_core_once,
        trials=trials,
        session=session,
        query=query,
        reference_response=reference_response,
        max_retries=max_retries,
        threshold=threshold,
        language=language,
    )


# ---------------------------------------------------------------------------
# Hallucination core (single-pass)
# ---------------------------------------------------------------------------

_MAPREDUCE_CONTEXT_THRESHOLD = 40_000
_MAPREDUCE_CHUNK_TARGET = 40_000
_MAPREDUCE_VERIFY_CONCURRENCY = max(
    1, int(os.environ.get("EVAL_MR_VERIFY_CONCURRENCY", "8"))
)
_mr_verify_semaphore: asyncio.Semaphore | None = None


def _get_mr_verify_semaphore() -> asyncio.Semaphore:
    """进程级 Semaphore，限制 MR-Verify 同时打 LLM 的 chunk 数。

    懒初始化以绑定到当前 event loop；env ``EVAL_MR_VERIFY_CONCURRENCY`` 可调。
    """
    global _mr_verify_semaphore
    if _mr_verify_semaphore is None:
        _mr_verify_semaphore = asyncio.Semaphore(_MAPREDUCE_VERIFY_CONCURRENCY)
    return _mr_verify_semaphore


async def _evaluate_hallucination_core_once(
    session: dict,
    query: str,
    reference_response: str = "",
    *,
    context_extra: str = "",
    max_retries: int = 2,
    threshold: int = 3,
    language: LanguageEnum | str = LanguageEnum.ZH,
    precomputed_tool_context: str | None = None,
) -> GraderScore | GraderError:
    """统一幻觉评估核心（单次 + retry）。

    所有过程中的工具输出经统一清洗/去重后作为 context，前置 Agent 能力说明，
    用自定义 _COPAW_HALLUCINATION_TEMPLATE（截断感知）评估幻觉。

    context_extra: 插在「能力说明」与「工具输出摘要」之间的补充文本（如多模态任务的
    thinking 摘录），便于纯文本 grader 获知视觉识别结论，无需传真实图片像素。

    precomputed_tool_context: 调用方已经构建好的工具上下文（避免与
    `_evaluate_hallucination_core` 阶段对 entries 的清洗去重重复劳动）。
    None 时退回到 `_get_context(session)` 现场计算。
    """
    language = _coerce_language(language)
    tool_ctx = (
        precomputed_tool_context if precomputed_tool_context is not None else _get_context(session)
    )
    parts: list[str] = [extract_agent_capability_context(session)]
    if context_extra.strip():
        parts.append(context_extra.strip())
    if tool_ctx.strip():
        parts.append(tool_ctx)
    context = "\n\n".join(parts)

    grader = HallucinationGrader(
        model=get_llm_model(),
        threshold=threshold,
        template=_COPAW_HALLUCINATION_TEMPLATE,
        language=language,
    )
    eval_kwargs: dict[str, Any] = dict(
        response=extract_final_response(session),
        query=query,
        context=context,
    )
    if reference_response.strip():
        eval_kwargs["reference_response"] = reference_response

    return await _run_grader_once(
        grader,
        eval_kwargs,
        max_retries=max_retries,
        label="Hallucination",
    )


async def _evaluate_hallucination_core(
    session: dict,
    query: str,
    reference_response: str = "",
    *,
    context_extra: str = "",
    max_retries: int = 2,
    threshold: int = 3,
    language: LanguageEnum | str = LanguageEnum.ZH,
    trials: int = _DEFAULT_GRADING_TRIALS,
) -> GraderScore | GraderError:
    """统一幻觉评估（multi-trial 取中位数）。

    **统一策略**（不按 domain 分支）：
      1. `build_unified_entries(session)` 收集所有过程中工具调用输出
         （browser_use snapshot、curl/wget HTML、read_file/shell 等），
         按工具类型派发清洗（DOM 噪音/HTML 标签）+ 跨工具内容指纹去重
      2. 总长 > `_MAPREDUCE_CONTEXT_THRESHOLD` 时自动切到 MapReduce
         (Extract → Map 并发 → Reduce 多 trial)，避免 LLM "lost in the middle"
      3. 否则反向填充 100k 后做单次 grader（多 trial 取中位）
    """
    entries = build_unified_entries(session)
    total_chars = sum(len(h) + len(o) for h, o in entries)
    if total_chars > _MAPREDUCE_CONTEXT_THRESHOLD:
        logger.info(
            "Hallucination context %d chars > threshold %d, using MapReduce",
            total_chars,
            _MAPREDUCE_CONTEXT_THRESHOLD,
        )
        return await _evaluate_hallucination_mapreduce(
            session,
            query,
            entries,
            max_retries=max_retries,
            threshold=threshold,
            trials=trials,
        )

    precomputed_tool_context = _pack_entries_reverse_fill(entries, 100_000)
    return await _trial_llm_grader(
        _evaluate_hallucination_core_once,
        trials=trials,
        session=session,
        query=query,
        reference_response=reference_response,
        context_extra=context_extra,
        max_retries=max_retries,
        threshold=threshold,
        language=language,
        precomputed_tool_context=precomputed_tool_context,
    )


# ---------------------------------------------------------------------------
# MapReduce hallucination (long-context fallback)
# ---------------------------------------------------------------------------


async def _mr_extract_claims(
    response: str,
    query: str,
    *,
    max_retries: int = 2,
) -> list[dict]:
    """提取声明。

    LLM 调用或 JSON 解析失败时退化为空列表，让上游走"未提取到声明，从宽给 5 分"
    的 fallback 路径，而不是把异常抛到 pytest 测试函数让那条 grader 缺失。
    """
    prompt = _MR_EXTRACT_CLAIMS_PROMPT.format(query=query, response=response[:12000])
    try:
        text = await _llm_raw_call(prompt, max_retries=max_retries, label="MR-Extract")
        claims = _parse_json_array(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MR-Extract failed (%s), falling back to empty claims list: %s",
            type(exc).__name__,
            exc,
        )
        return []
    for i, c in enumerate(claims):
        c.setdefault("id", i)
        c.setdefault("keywords", [])
    logger.info("MR-Extract: %d claims from response", len(claims))
    return claims


def _mr_keyword_search(
    claims: list[dict],
    entries: list[tuple[str, str]],
) -> dict[int, list[str]]:
    """对每条 claim 的 keywords 做字符串匹配，返回 {claim_id: [命中摘要]}。"""
    results: dict[int, list[str]] = {}
    for claim in claims:
        cid = claim["id"]
        hits: list[str] = []
        for kw in claim.get("keywords", []):
            if kw is None:
                continue
            kw = str(kw).strip()
            if not kw:
                continue
            for ei, (header, content) in enumerate(entries, 1):
                pos = content.find(kw)
                if pos < 0:
                    pos = content.lower().find(kw.lower())
                if pos >= 0:
                    snippet = content[max(0, pos - 80) : pos + len(kw) + 120]
                    hits.append(f"[Entry {ei}] ...{snippet.strip()}...")
                    break
        if hits:
            results[cid] = hits
    return results


def _mr_chunk_entries(
    entries: list[tuple[str, str]],
    target_size: int = _MAPREDUCE_CHUNK_TARGET,
) -> list[str]:
    """将 entries 按 target_size 装箱。

    通常 entries 已经在 ``build_unified_entries`` 阶段被 ``_truncate_entry_content``
    截到 ``_MAX_ENTRY_CHARS`` 以内。这里再做一层防御：若某个单条 entry 仍然超过
    ``target_size``（例如未来阈值不一致或上游被绕过），就把它切成多个 sub-chunk，
    保证**没有任何 chunk 会超过 target_size**——避免 MR-Verify prompt 拼出来直接
    撞模型 input length 上限触发 400。
    """
    if not entries:
        return []
    chunks: list[str] = []
    current_parts: list[str] = []
    current_size = 0
    for header, content in entries:
        entry_text = f"{header}\n{content}"
        if len(entry_text) > target_size:
            if current_parts:
                chunks.append("\n---\n".join(current_parts))
                current_parts = []
                current_size = 0
            num_parts = (len(entry_text) + target_size - 1) // target_size
            for i in range(num_parts):
                start = i * target_size
                segment = entry_text[start : start + target_size]
                if i == 0:
                    chunks.append(segment)
                else:
                    chunks.append(f"{header} [续 {i + 1}/{num_parts}]\n{segment}")
            continue
        if current_size + len(entry_text) > target_size and current_parts:
            chunks.append("\n---\n".join(current_parts))
            current_parts = []
            current_size = 0
        current_parts.append(entry_text)
        current_size += len(entry_text)
    if current_parts:
        chunks.append("\n---\n".join(current_parts))
    return chunks


async def _mr_verify_chunk(
    claims: list[dict],
    chunk: str,
    chunk_idx: int,
    total_chunks: int,
    *,
    max_retries: int = 2,
) -> list[dict]:
    """验证单个 chunk 中的 claim。

    LLM/JSON 解析失败时退化为空列表，让其他 chunk 仍可以贡献证据。
    """
    claims_for_prompt = [
        {"id": c["id"], "claim": c["claim"], "keywords": c.get("keywords", [])} for c in claims
    ]
    prompt = _MR_VERIFY_CHUNK_PROMPT.format(
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
        chunk=chunk,
        claims_json=json.dumps(claims_for_prompt, ensure_ascii=False, indent=2),
    )
    try:
        text = await _llm_raw_call(
            prompt,
            max_retries=max_retries,
            label=f"MR-Verify[{chunk_idx}/{total_chunks}]",
        )
        results = _parse_json_array(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MR-Verify chunk %d/%d failed (%s), treating as no verdicts: %s",
            chunk_idx,
            total_chunks,
            type(exc).__name__,
            exc,
        )
        return []
    found_ids = [r["id"] for r in results if r.get("supported")]
    logger.info("MR-Verify chunk %d/%d: supported=%s", chunk_idx, total_chunks, found_ids)
    return results


async def _mr_verify_all_chunks(
    claims: list[dict],
    chunks: list[str],
    *,
    max_retries: int = 2,
) -> list[list[dict]]:
    """并发验证所有 chunk（受 ``_MAPREDUCE_VERIFY_CONCURRENCY`` 节流）。

    使用 ``return_exceptions=True``：单个 chunk 即便绕过了内部 try/except 仍冒泡
    出异常，也只让该 chunk 退化为空 verdict，不会拖垮整批。

    通过模块级 Semaphore 限制同时打 LLM 的 chunk 数，避免长上下文场景下一次性
    把几十个请求灌给 dashscope 触发限流；可通过 env ``EVAL_MR_VERIFY_CONCURRENCY``
    覆盖默认值（8）。
    """
    sem = _get_mr_verify_semaphore()

    async def _guarded(idx: int, ch: str) -> list[dict]:
        async with sem:
            return await _mr_verify_chunk(
                claims, ch, idx + 1, len(chunks), max_retries=max_retries
            )

    tasks = [_guarded(i, ch) for i, ch in enumerate(chunks)]
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    out: list[list[dict]] = []
    for i, r in enumerate(raw):
        if isinstance(r, BaseException):
            logger.warning(
                "MR-Verify chunk %d/%d raised through inner guard (%s), skipping: %s",
                i + 1,
                len(chunks),
                type(r).__name__,
                r,
            )
            out.append([])
        else:
            out.append(r)
    return out


def _mr_merge_verdicts(
    claims: list[dict],
    chunk_results: list[list[dict]],
    keyword_hits: dict[int, list[str]],
) -> list[dict]:
    merged: list[dict] = []
    for claim in claims:
        cid = claim["id"]
        supported_by: list[str] = []
        contradicted_by: list[str] = []
        for ci, chunk_res in enumerate(chunk_results):
            for r in chunk_res:
                if r.get("id") != cid:
                    continue
                if r.get("supported"):
                    supported_by.append(f"chunk {ci + 1}: {r.get('evidence', '')[:200]}")
                if r.get("contradicted"):
                    contradicted_by.append(f"chunk {ci + 1}")
        kw = keyword_hits.get(cid, [])
        if supported_by or kw:
            verdict = "supported"
        elif contradicted_by:
            verdict = "contradicted"
        else:
            verdict = "unverified"
        merged.append(
            {
                "id": cid,
                "claim": claim["claim"],
                "supported_by": supported_by,
                "contradicted_by": contradicted_by,
                "keyword_hits": kw,
                "verdict": verdict,
            }
        )
    return merged


async def _mr_reduce_once(
    claims: list[dict],
    merged: list[dict],
    query: str,
    response: str,
    *,
    max_retries: int = 2,
) -> GraderScore | GraderError:
    verdict_lines: list[str] = []
    for v in merged:
        lines = [f"声明 {v['id']}: {v['claim']}"]
        lines.append(f"  综合判定: {v['verdict']}")
        if v["supported_by"]:
            lines.append(f"  支持证据 ({len(v['supported_by'])} 处):")
            for s in v["supported_by"][:3]:
                lines.append(f"    - {s[:200]}")
        if v["keyword_hits"]:
            lines.append(f"  关键词命中 ({len(v['keyword_hits'])} 处):")
            for h in v["keyword_hits"][:2]:
                lines.append(f"    - {h[:200]}")
        if v["contradicted_by"]:
            lines.append(f"  矛盾证据: {v['contradicted_by']}")
        if v["verdict"] == "unverified":
            lines.append("  (所有分块均未提及，按「无法验证」从宽)")
        verdict_lines.append("\n".join(lines))

    prompt = _MR_REDUCE_PROMPT.format(
        query=query,
        response=response[:8000],
        verdicts_text="\n\n".join(verdict_lines),
    )
    try:
        text = await _llm_raw_call(prompt, max_retries=max_retries, label="MR-Reduce")
        result = _parse_json_object(text)
        score = float(result.get("score", 3))
        reason = result.get("reason", text[:500])
        return GraderScore(name="hallucination", score=score, reason=reason)
    except Exception as exc:
        return GraderError(name="hallucination", error=f"MR-Reduce failed: {exc}")


async def _evaluate_hallucination_mapreduce(
    session: dict,
    query: str,
    entries: list[tuple[str, str]],
    *,
    max_retries: int = 2,
    threshold: int = 3,
    trials: int = _DEFAULT_GRADING_TRIALS,
) -> GraderScore | GraderError:
    """MapReduce 幻觉评估完整流程。

    Extract + Map 只执行一次（结果确定性可复用），仅 Reduce 做 multi-trial。
    """
    response = extract_final_response(session)

    claims = await _mr_extract_claims(response, query, max_retries=max_retries)
    if not claims:
        return GraderScore(
            name="hallucination",
            score=5.0,
            reason="[MapReduce] 未从回复中提取到可验证的事实性声明，视为无幻觉",
            metadata={"_strategy": "mapreduce", "_num_claims": 0},
        )

    keyword_hits = _mr_keyword_search(claims, entries)
    logger.info("MR keyword search: %d/%d claims have hits", len(keyword_hits), len(claims))

    chunks = _mr_chunk_entries(entries, target_size=_MAPREDUCE_CHUNK_TARGET)
    logger.info("MR: %d entries → %d chunks", len(entries), len(chunks))
    chunk_results = (
        await _mr_verify_all_chunks(claims, chunks, max_retries=max_retries) if chunks else []
    )

    merged = _mr_merge_verdicts(claims, chunk_results, keyword_hits)
    verdict_counts = {
        "supported": sum(1 for v in merged if v["verdict"] == "supported"),
        "contradicted": sum(1 for v in merged if v["verdict"] == "contradicted"),
        "unverified": sum(1 for v in merged if v["verdict"] == "unverified"),
    }
    logger.info("MR merged: %s", verdict_counts)

    results: list[GraderScore] = []
    last_error: GraderError | None = None
    for i in range(trials):
        r = await _mr_reduce_once(claims, merged, query, response, max_retries=max_retries)
        if isinstance(r, GraderScore):
            results.append(r)
        else:
            last_error = r
            logger.warning("MR-Reduce trial %d/%d error: %s", i + 1, trials, r.error)

    if not results:
        return last_error  # type: ignore[return-value]

    if len(results) == 1:
        selected = results[0]
    else:
        scores = [r.score for r in results]
        median = statistics.median(scores)
        selected = min(results, key=lambda r: abs(r.score - median))

    trial_scores = [r.score for r in results]
    high_var = (
        (max(trial_scores) - min(trial_scores)) >= _HIGH_VARIANCE_RANGE_THRESHOLD
        if len(trial_scores) >= 2
        else False
    )
    metadata = selected.metadata if isinstance(selected.metadata, dict) else {}
    metadata.update(
        {
            "_trial_scores": trial_scores,
            "_trial_errors": trials - len(results),
            "_high_variance": high_var,
            "_strategy": "mapreduce",
            "_num_claims": len(claims),
            "_num_chunks": len(chunks),
            "_verdicts_summary": verdict_counts,
            "threshold": threshold,
        }
    )
    selected.metadata = metadata
    return selected


# ---------------------------------------------------------------------------
# Search relevance
# ---------------------------------------------------------------------------


async def _evaluate_search_relevance_once(
    session: dict,
    query: str,
    reference_response: str = "",
    *,
    max_retries: int = 2,
    threshold: int = 3,
    language: LanguageEnum | str = LanguageEnum.ZH,
) -> GraderScore | GraderError:
    """单次搜索相关性评估（内置 retry on error）。

    context = 过程中所有工具调用的统一清洗输出（100k 反向填充），
    与 correctness/hallucination 共享同一份工具上下文。
    """
    language = _coerce_language(language)
    grader = RelevanceGrader(model=get_llm_model(), threshold=threshold, language=language)
    eval_kwargs: dict[str, Any] = dict(
        response=extract_final_response(session),
        query=query,
        context=_get_context(session),
    )
    if reference_response.strip():
        eval_kwargs["reference_response"] = reference_response
    return await _run_grader_once(
        grader, eval_kwargs, max_retries=max_retries, label="Search relevance"
    )


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


@safe_grader_eval("correctness")
async def evaluate_correctness(
    session: dict,
    query: str,
    reference_response: str,
    *,
    max_retries: int = 2,
    threshold: int = 3,
    language: LanguageEnum | str = LanguageEnum.ZH,
    trials: int = _DEFAULT_GRADING_TRIALS,
) -> GraderScore | GraderError:
    """LLM 语义评估答案正确性（5 分制），多次 trial 取中位数。

    context = 全部工具输出。
    """
    return await _evaluate_correctness_core(
        session,
        query,
        reference_response,
        max_retries=max_retries,
        threshold=threshold,
        language=language,
        trials=trials,
    )


@safe_grader_eval("file_correctness")
async def evaluate_file_correctness(
    session: dict,
    query: str,
    reference_response: str = "",
    *,
    max_retries: int = 2,
    threshold: int = 3,
    language: LanguageEnum | str = LanguageEnum.ZH,
    trials: int = _DEFAULT_GRADING_TRIALS,
) -> GraderScore | GraderError:
    """针对文件处理类任务的正确性评估，多次 trial 取中位数。

    reference_response 非空 → CorrectnessGrader (context=全部工具输出)。
    reference_response 为空 → HallucinationGrader (context=全部工具输出，截断感知)。
    """
    if reference_response.strip():
        return await _evaluate_correctness_core(
            session,
            query,
            reference_response,
            max_retries=max_retries,
            threshold=threshold,
            language=language,
            trials=trials,
        )
    return await _evaluate_hallucination_core(
        session,
        query,
        reference_response,
        max_retries=max_retries,
        threshold=threshold,
        language=language,
        trials=trials,
    )


@safe_grader_eval("search_hallucination")
async def evaluate_search_hallucination(
    session: dict,
    query: str,
    reference_response: str = "",
    *,
    max_retries: int = 2,
    threshold: int = 3,
    language: LanguageEnum | str = LanguageEnum.ZH,
    trials: int = _DEFAULT_GRADING_TRIALS,
) -> GraderScore | GraderError:
    """针对搜索/新闻/浏览类任务的幻觉评估，多次 trial 取中位数。

    context = Agent 能力说明 +（多模态时）thinking 补充 + 搜索/浏览器工具输出摘要。
    使用自定义截断感知 HallucinationGrader 模板。

    若会话中检测到 **用户消息含图片**（模型可直接读图）或 **曾调用 view_image**，会自动注入
    thinking + 边界说明；仅纯网页检索、且两者皆无时行为与原先一致。
    """
    extra = build_multimodal_search_hallucination_extra_context(session)
    return await _evaluate_hallucination_core(
        session,
        query,
        reference_response,
        context_extra=extra,
        max_retries=max_retries,
        threshold=threshold,
        language=language,
        trials=trials,
    )


@safe_grader_eval("search_relevance")
async def evaluate_search_relevance(
    session: dict,
    query: str,
    reference_response: str = "",
    *,
    max_retries: int = 2,
    threshold: int = 3,
    language: LanguageEnum | str = LanguageEnum.ZH,
    trials: int = _DEFAULT_GRADING_TRIALS,
) -> GraderScore | GraderError:
    """针对搜索/新闻/浏览类任务的相关性评估，多次 trial 取中位数。

    用 RelevanceGrader 评估 Agent 最终回答与用户 query 的契合程度，
    5 分量表（5=完全相关，1=完全无关）。
    """
    return await _trial_llm_grader(
        _evaluate_search_relevance_once,
        trials=trials,
        session=session,
        query=query,
        reference_response=reference_response,
        max_retries=max_retries,
        threshold=threshold,
        language=language,
    )
