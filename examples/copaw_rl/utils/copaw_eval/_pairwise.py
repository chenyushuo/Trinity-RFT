"""Pairwise correctness grader — tie-break when pointwise CorrectnessGrader scores match.

When two runs receive the same CorrectnessGrader score on the same task, this grader
compares the two final responses head-to-head against the reference answer, using
OpenJudge ``LLMGrader`` (same stack as ``CorrectnessGrader``).
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any

from openjudge.graders.base_grader import GraderMode
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderError, GraderScore
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

from ._context import _get_context
from ._core import (
    _coerce_language,
    _run_grader_once,
    _trial_llm_grader,
    get_llm_model,
    safe_grader_eval,
)

logger = logging.getLogger(__name__)

_PAIRWISE_CORRECTNESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=(
                    "你是专业的 AI Agent 轨迹质量评估专家。"
                    "前提：两份回复（A 与 B）在 pointwise CorrectnessGrader 上已获得**相同的事实分**，"
                    "说明两份 final response 在“字面上与参考答案的对齐度”上被判为等价。"
                    "你的任务不是重复该判定，而是通过比较**工具调用轨迹的质量**，"
                    "判断哪一份回复的“正确性”更**可信**、更**可验证**（即同样的正确结论，由真实数据得出还是凭空捏造）。"
                    '仅输出合法 JSON：{{"score": <float>, "reason": "..."}}。'
                    "score 取值仅限三档离散值：**1.0 = A 轨迹明显更可信，0.0 = B 轨迹明显更可信，0.5 = 两侧轨迹质量几乎相等**；"
                    "严禁输出 0.1/0.3/0.7/0.9 等中间值，只能从三档中选一。"
                    "reason 字段要求：必须是已经形成的最终结论，≤200 字；禁止出现 “等等”/“Wait”/“实际上”/“再看看” 等思考草稿、自我修正、计算过程。"
                    "【方向一致性硬约束（违反即无效）】reason 表达的胜方必须与 score 严格一致，不得自相矛盾："
                    "① 若 reason 出现 “A 更可信 / A 更优 / A 显著更高 / B 不可信 / B 伪造 / 相比之下 A 更…” 等指向 A 胜的措辞，score 必须 = 1.0；"
                    "② 若 reason 出现 “B 更可信 / B 更优 / B 显著更高 / A 不可信 / A 伪造 / 相比之下 B 更…” 等指向 B 胜的措辞，score 必须 = 0.0；"
                    "③ 仅当 reason 明确表述 “两者相当 / 等价 / 难分伯仲 / 同等程度” 且未褒贬任一方时，score 才可 = 0.5。"
                    "【自检流程】写完 reason 后**必须重读一遍**：先识别 reason 的胜方指向（A / B / 相当），再回写 score，确保两者方向一致；若不一致，**重写 reason 或修正 score 后再输出**，禁止提交方向矛盾的 JSON。"
                    "【禁止】禁止 reason 通篇贬 A 褒 B 却给 score=1.0；禁止 reason 通篇贬 B 褒 A 却给 score=0.0；这类输出会被判为无效评分。"
                ),
            ),
            ChatMessage(
                role="user",
                content=textwrap.dedent(
                    """
                    <背景与评估范式>
                    两份 final response 的 pointwise 正确性得分相同。请不要重复逐句对比 final response 文本，
                    重点沉到**工具调用上下文**中检查两侧“轨迹质量”，以互补于 CorrectnessGrader。

                    评估维度（按优先级，均锁在 correctness 范畴）：
                    1. 数据源真实性：工具日志中是否成功读取了任务指定的**原始文件**（路径、行数、字段与参考匹配），
                       还是读文件失败后**用 write_file/echo 等手段自行造数**、或者跳过读取直接虚构结果。
                    2. 计算路径可验证性：关键数值（均值/计数/排名/分组）是否能从工具输出中复现；
                       是否出现“工具返回 A，但 final response 却写 B”的轨迹与结论不一致。
                    3. 错误恢复与健壮性：遇到 file not found / 路径错误 / 解析异常时，是否主动纠正路径、换工具、glob 重试，
                       还是直接伪造输出。同样的结论下，有真实恢复证据的一方更可信。
                    4. 交付物落盘证据：任务要求生成 csv/md/图表时，轨迹中是否真有对应的 write_file/save 调用，
                       还是仅仅在 final response 里口头声称已生成。
                    5. 资源使用效率（仅作 tie-break，不得压过上述四项）：体现为重复失败、无意义重试、大量跳环思考。

                    判定原则：
                    - “轨迹可信 + 同样结论” > “轨迹虚构 + 同样结论”，前者明显更优。
                    - 两者都虚构/读错源时，选虚构程度更轻、中间产物与参考偏离更小的一方。
                    - 轨迹质量不变时，才回到 final response 文本看伤。
                    - 严格避免重复 CorrectnessGrader 已经在做的 “final response 与参考答案逐句对齐”，避免与 pointwise 出多重评估。
                    - 两侧可验证性几乎相等且无交付物差别才给 0.5；哪怕细节差异也要倾向 0.6/0.4。

                    <用户任务>
                    {query}
                    </用户任务>

                    <参考回答>
                    {reference_response}
                    </参考回答>

                    <回复A final response>
                    {response_a}
                    </回复A final response>

                    <回复A 工具调用轨迹>
                    {context_a}
                    </回复A 工具调用轨迹>

                    <回复B final response>
                    {response_b}
                    </回复B final response>

                    <回复B 工具调用轨迹>
                    {context_b}
                    </回复B 工具调用轨迹>

                    输出前自检：reason 文本指向 A 胜则 score=1.0；指向 B 胜则 score=0.0；明确表述“相当/等价”才给 0.5。
                    禁止出现 reason 偏 A 但 score=0.0、reason 偏 B 但 score=1.0 的方向矛盾。

                    输出 JSON（不要其它文字，不要思考草稿）：
                    {{
                      "score": <仅限三档：1.0=A轨迹更可信、0.0=B轨迹更可信、0.5=几乎相等；禁止其它值；必须与 reason 方向一致>,
                      "reason": "<≤2句、≤200 字；明确引用轨迹中的关键事件（如 read_file 是否成功、关键数值是否能复现）；结尾必须给出明确胜方判断（A 更可信 / B 更可信 / 两者相当），与 score 严格对应>"
                    }}
                    """
                ).strip(),
            ),
        ],
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=(
                    "You are an expert evaluator of AI agent trajectory quality. "
                    "PREMISE: both responses (A and B) already received the SAME pointwise CorrectnessGrader score, "
                    "meaning their final responses are deemed equivalent on surface alignment with the reference answer. "
                    "Do NOT repeat that judgment. Instead, compare the **tool-call trajectories** and decide which side's "
                    "correctness is more **trustworthy and verifiable** (same conclusion grounded in real data vs fabricated). "
                    'Output ONLY valid JSON: {{"score": <float>, "reason": "..."}}. '
                    "score MUST be one of three discrete values: **1.0 = A trajectory clearly more credible, 0.0 = B trajectory clearly more credible, 0.5 = roughly equal**. "
                    "Strictly forbid intermediate values like 0.1/0.3/0.7/0.9; pick exactly one of the three. "
                    "reason MUST be a finalized judgment, <=200 chars; NO 'wait' / 'actually' / self-correction / scratch reasoning. "
                    "[DIRECTION CONSISTENCY — HARD CONSTRAINT, violation invalidates the verdict] reason wording MUST strictly agree with score direction: "
                    "(1) If reason says 'A is more credible / A is better / B fabricated / compared to B, A ...' or any A-favoring phrasing, score MUST = 1.0; "
                    "(2) If reason says 'B is more credible / B is better / A fabricated / compared to A, B ...' or any B-favoring phrasing, score MUST = 0.0; "
                    "(3) score = 0.5 ONLY when reason explicitly states 'roughly equal / comparable / on par' WITHOUT praising either side. "
                    "[SELF-CHECK] After writing reason, RE-READ it: identify which side it favors (A / B / equal), then align score accordingly. If they disagree, REWRITE reason or FIX score before emitting. NEVER submit a JSON where reason and score point in opposite directions. "
                    "[FORBIDDEN] Forbid reason that bashes A and praises B with score=1.0; forbid reason that bashes B and praises A with score=0.0. Such outputs are treated as invalid."
                ),
            ),
            ChatMessage(
                role="user",
                content=textwrap.dedent(
                    """
                    <Setup>
                    Both final responses received the same pointwise correctness score. Do NOT re-litigate the
                    sentence-by-sentence alignment with the reference answer. Focus on **trajectory quality** to
                    complement (not duplicate) what CorrectnessGrader already measured.

                    Evaluation axes (priority order, all anchored on correctness):
                    1. Data-source authenticity: did tools actually read the specified raw files (matching path,
                       row count, fields), or did the agent fabricate via write_file/echo after a failed read?
                    2. Computation verifiability: are key numbers (means/counts/ranks/groupings) reproducible
                       from tool outputs? Any "tool returned A but final says B" mismatch?
                    3. Error recovery & robustness: on file-not-found / path / parse errors, did the agent
                       fix the path, switch tools, glob-retry — or fabricate? Same conclusion is more credible
                       when backed by real recovery evidence.
                    4. Deliverable evidence: when the task asks for csv/md/charts, do trajectories show real
                       write_file/save calls, or only verbal claims in the final response?
                    5. Resource efficiency (tie-break only, never overrides 1-4): repeated failures, useless
                       retries, large blocks of skipped reasoning.

                    Decision rules:
                    - "Verifiable trajectory + same conclusion" beats "fabricated trajectory + same conclusion".
                    - When both fabricate, pick the one with smaller deviation from reference and more honest
                       intermediate artifacts.
                    - Only fall back to final-response wording when trajectory quality is truly equal.
                    - Avoid duplicating the work CorrectnessGrader already did.
                    - Give 0.5 only if trajectories are essentially equal AND deliverables match; otherwise lean
                       to 0.6/0.4 even on subtle differences.

                    <Query>
                    {query}
                    </Query>

                    <Reference Response>
                    {reference_response}
                    </Reference Response>

                    <Response A final>
                    {response_a}
                    </Response A final>

                    <Response A trajectory>
                    {context_a}
                    </Response A trajectory>

                    <Response B final>
                    {response_b}
                    </Response B final>

                    <Response B trajectory>
                    {context_b}
                    </Response B trajectory>

                    Pre-output self-check: if reason favors A, score=1.0; if reason favors B, score=0.0; only when reason
                    explicitly says 'roughly equal' may score=0.5. Never emit reason-favors-A with score=0.0 or vice versa.

                    Output JSON only (no scratch reasoning):
                    {{
                      "score": <ONE OF {1.0, 0.5, 0.0}; 1.0=A more credible, 0.0=B more credible, 0.5=equal; no other values; MUST agree with reason direction>,
                      "reason": "<<=2 sentences, <=200 chars; cite concrete trajectory evidence; MUST end with an explicit verdict phrase (A more credible / B more credible / roughly equal) that strictly matches score>"
                    }}
                    """
                ).strip(),
            ),
        ],
    }
)


def _build_pairwise_grader(language: LanguageEnum) -> LLMGrader:
    return LLMGrader(
        name="PairwiseCorrectnessGrader",
        mode=GraderMode.POINTWISE,
        model=get_llm_model(),
        language=language,
        template=_PAIRWISE_CORRECTNESS_TEMPLATE,
    )


def _truncate(text: str, limit: int = 12_000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n...[truncated]..."


def _context_from_session(session: dict | None) -> str:
    if not session:
        return "(无工具上下文)"
    try:
        ctx = _get_context(session)
        return _truncate(ctx or "(无工具上下文)")
    except Exception as exc:  # noqa: BLE001
        logger.debug("failed to build context: %s", exc)
        return "(上下文提取失败)"


def _flip_score(score: float) -> float:
    return max(0.0, min(1.0, 1.0 - score))


def _aggregate_position_scores(score_ab: float, score_ba: float) -> float:
    """Combine original-order and swapped-order scores to reduce position bias."""
    return (score_ab + _flip_score(score_ba)) / 2.0


def _score_to_verdict(aggregated: float, *, margin: float = 0.1) -> str:
    if aggregated >= 0.5 + margin:
        return "A_better"
    if aggregated <= 0.5 - margin:
        return "B_better"
    return "tie"


async def _evaluate_pairwise_correctness_once(
    *,
    query: str,
    reference_response: str,
    response_a: str,
    response_b: str,
    context_a: str = "",
    context_b: str = "",
    max_retries: int = 2,
    language: LanguageEnum | str = LanguageEnum.ZH,
) -> GraderScore | GraderError:
    language = _coerce_language(language)
    grader = _build_pairwise_grader(language)
    return await _run_grader_once(
        grader,
        dict(
            query=query,
            reference_response=_truncate(reference_response, 20_000),
            response_a=_truncate(response_a, 20_000),
            response_b=_truncate(response_b, 20_000),
            context_a=_truncate(context_a or "(无)", 8_000),
            context_b=_truncate(context_b or "(无)", 8_000),
        ),
        max_retries=max_retries,
        label="PairwiseCorrectness",
    )


@safe_grader_eval("pairwise_correctness")
async def evaluate_pairwise_correctness(
    *,
    query: str,
    reference_response: str,
    response_a: str,
    response_b: str,
    session_a: dict | None = None,
    session_b: dict | None = None,
    context_a: str | None = None,
    context_b: str | None = None,
    max_retries: int = 2,
    language: LanguageEnum | str = LanguageEnum.ZH,
    trials: int = 1,
    debias_position: bool = True,
) -> GraderScore | GraderError:
    """Pairwise tie-break grader when two responses share the same CorrectnessGrader score.

    Returns ``GraderScore`` with:
    - ``score`` in [0, 1]: 1.0 = A better, 0.0 = B better, 0.5 = tie
    - ``metadata["verdict"]``: ``A_better`` | ``B_better`` | ``tie``
    - ``metadata["score_ab"]`` / ``score_ba`` / ``aggregated_score`` when debias enabled
    """
    ctx_a = context_a if context_a is not None else _context_from_session(session_a)
    ctx_b = context_b if context_b is not None else _context_from_session(session_b)

    result_ab = await _trial_llm_grader(
        _evaluate_pairwise_correctness_once,
        trials=trials,
        query=query,
        reference_response=reference_response,
        response_a=response_a,
        response_b=response_b,
        context_a=ctx_a,
        context_b=ctx_b,
        max_retries=max_retries,
        language=language,
    )
    if isinstance(result_ab, GraderError):
        return result_ab

    if not debias_position:
        metadata = dict(result_ab.metadata or {})
        metadata["verdict"] = _score_to_verdict(result_ab.score)
        metadata["debias_position"] = False
        return GraderScore(
            name="PairwiseCorrectnessGrader",
            score=result_ab.score,
            reason=result_ab.reason,
            metadata=metadata,
        )

    result_ba = await _trial_llm_grader(
        _evaluate_pairwise_correctness_once,
        trials=trials,
        query=query,
        reference_response=reference_response,
        response_a=response_b,
        response_b=response_a,
        context_a=ctx_b,
        context_b=ctx_a,
        max_retries=max_retries,
        language=language,
    )
    if isinstance(result_ba, GraderError):
        metadata = dict(result_ab.metadata or {})
        metadata["verdict"] = _score_to_verdict(result_ab.score)
        metadata["debias_position"] = True
        metadata["swap_error"] = result_ba.error
        return GraderScore(
            name="PairwiseCorrectnessGrader",
            score=result_ab.score,
            reason=f"{result_ab.reason} | swap_eval_failed: {result_ba.error}",
            metadata=metadata,
        )

    aggregated = _aggregate_position_scores(result_ab.score, result_ba.score)
    # 两次评分一致性检查：分别看单边判决（swap 后翻转回原方向）是否指向同一赢家。
    score_ba_flipped = _flip_score(result_ba.score)
    verdict_ab = _score_to_verdict(result_ab.score)
    verdict_ba_original = _score_to_verdict(score_ba_flipped)
    position_consistent = verdict_ab == verdict_ba_original

    # 降级裁决：position_consistent=False 时，aggregated 会被算术对消成 0.5（虚假 tie），
    # 改取“更决断”的那次打分（距离 0.5 更远）作为 final score；
    # 两次同样决断但方向相反（如 ab=0/ba_flipped=1）是位置偏见 100% 的信号，判 tie。
    if position_consistent:
        final_score = aggregated
        decision_path = "aggregated"
    else:
        decisive_ab = abs(result_ab.score - 0.5)
        decisive_ba = abs(score_ba_flipped - 0.5)
        if decisive_ab > decisive_ba:
            final_score = result_ab.score
            decision_path = "fallback_ab_decisive"
        elif decisive_ba > decisive_ab:
            final_score = score_ba_flipped
            decision_path = "fallback_ba_decisive"
        else:
            # 两次同等决断但方向相反 → 纯位置偏见，无方向证据，强制 tie。
            final_score = 0.5
            decision_path = "fallback_unresolved_tie"
        logger.warning(
            "PairwiseCorrectness position bias detected: ab=%.3f (%s) vs ba_flipped=%.3f (%s); "
            "fallback to %s, final_score=%.3f",
            result_ab.score,
            verdict_ab,
            score_ba_flipped,
            verdict_ba_original,
            decision_path,
            final_score,
        )

    verdict = _score_to_verdict(final_score)
    metadata: dict[str, Any] = dict(result_ab.metadata or {})  # type: ignore
    metadata.update(
        {
            "verdict": verdict,
            "debias_position": True,
            "score_ab": result_ab.score,
            "score_ba": result_ba.score,
            "aggregated_score": aggregated,
            "reason_ab": result_ab.reason,
            "reason_ba": result_ba.reason,
            "verdict_ab": verdict_ab,
            "verdict_ba_original": verdict_ba_original,
            "position_consistent": position_consistent,
            "decision_path": decision_path,
            "final_score": final_score,
        }
    )
    reason = (
        f"final={final_score:.3f} via {decision_path} "
        f"(ab={result_ab.score:.3f}/{verdict_ab}, ba_flipped={score_ba_flipped:.3f}/{verdict_ba_original}, "
        f"agg={aggregated:.3f}, consistent={position_consistent}) "
        f"→ {verdict} | ab: {result_ab.reason} | ba: {result_ba.reason}"
    )
    return GraderScore(
        name="PairwiseCorrectnessGrader",
        score=final_score,
        reason=reason,
        metadata=metadata,
    )
