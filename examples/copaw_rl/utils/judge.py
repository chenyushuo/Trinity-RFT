from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml
from otel_init import trace_span

try:
    from judge_reward import (
        DEFAULT_REWARD_POLICY,
        PROCESS_EVALUATORS,
        GraderScoreEntry,
        RewardPolicy,
        aggregate_grader_scores,
        finalize_reward,
    )
except ImportError:
    from .judge_reward import (  # type: ignore[no-redef]
        DEFAULT_REWARD_POLICY,
        PROCESS_EVALUATORS,
        GraderScoreEntry,
        RewardPolicy,
        aggregate_grader_scores,
        finalize_reward,
    )

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default


try:
    from copaw_eval import (
        evaluate_by_script,
        evaluate_correctness,
        evaluate_file_correctness,
        evaluate_safety_trajectory,
        evaluate_screenshot_coherence,
        evaluate_search_hallucination,
        evaluate_search_relevance,
        evaluate_trajectory,
        log_grader_score_line,
    )
except ImportError:
    from .copaw_eval import (  # type: ignore[no-redef]
        evaluate_by_script,
        evaluate_correctness,
        evaluate_file_correctness,
        evaluate_safety_trajectory,
        evaluate_screenshot_coherence,
        evaluate_search_hallucination,
        evaluate_search_relevance,
        evaluate_trajectory,
        log_grader_score_line,
    )


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    prefix: str
    domain: str
    has_answer: bool
    task_path: Optional[Path]


@dataclass(frozen=True)
class GraderSpec:
    name: str
    evaluator: str
    include_in_score: bool = True
    force_hallucination_mode: bool = False
    weight: float = 1.0


PREFIX_ALIASES: Dict[str, str] = {
    "mm_tool": "mm-tool",
    "chinese-qa": "chinese_qa",
    "chinese_simpleqa": "chinese_qa",
    "boostrap": "bootstrap",
    "mem": "memory",
}

PREFIX_DOMAIN: Dict[str, str] = {
    "search": "search",
    "honey": "multimodal",
    "mat": "multimodal",
    "mm-tool": "multimodal+search",
    "screen": "screenshot",
    "safety": "safety",
    "fr": "fileprocess",
    "docx": "fileprocess",
    "gov": "fileprocess",
    "pdf": "fileprocess",
    "xlsx": "fileprocess",
    "qa": "qa",
    "chinese_qa": "search",
    "bootstrap": "bootstrap",
    "cron": "cron",
    "memory": "memory",
    "nl2bash": "nl2bash",
    "skill": "skill",
    "sp": "systemprompt",
    "gui": "correctness",
    "merged": "correctness",
    "oss": "correctness",
    "seed": "correctness",
    "task": "correctness",
    "bi": "correctness",
    "entask": "script_evaluation",
    "krama": "script_evaluation",
}

GRADER_PLAN_BY_DOMAIN: Dict[str, list[GraderSpec]] = {
    "search": [
        GraderSpec("SearchHallucinationGrader", "search_hallucination"),
        GraderSpec("SearchRelevanceGrader", "search_relevance"),
        GraderSpec("TrajectoryGrader", "trajectory"),
    ],
    "multimodal": [
        GraderSpec("CorrectnessGrader", "correctness"),
        GraderSpec("TrajectoryGrader", "trajectory", include_in_score=False),
    ],
    "multimodal+search": [
        GraderSpec("CorrectnessGrader", "correctness"),
        GraderSpec("SearchHallucinationGrader", "search_hallucination"),
        GraderSpec("TrajectoryGrader", "trajectory", include_in_score=False),
    ],
    "screenshot": [
        GraderSpec("ScreenshotCoherenceGrader", "screenshot_coherence"),
        GraderSpec("TrajectoryGrader", "trajectory"),
    ],
    "safety": [
        GraderSpec("SafetyTrajectoryGrader", "safety_trajectory"),
    ],
    "fileprocess": [
        GraderSpec("FileCorrectnessGrader", "file_correctness"),
        GraderSpec("TrajectoryGrader", "trajectory"),
    ],
    "qa": [
        GraderSpec(
            "FileCorrectnessGrader(HallucinationMode)",
            "file_correctness",
            force_hallucination_mode=True,
        ),
        GraderSpec("TrajectoryGrader", "trajectory"),
    ],
    "bootstrap": [
        GraderSpec("TrajectoryGrader", "trajectory"),
    ],
    "cron": [
        GraderSpec("TrajectoryGrader", "trajectory"),
    ],
    "memory": [
        GraderSpec("TrajectoryGrader", "trajectory"),
    ],
    "nl2bash": [
        GraderSpec("TrajectoryGrader", "trajectory"),
    ],
    "skill": [
        GraderSpec(
            "FileCorrectnessGrader(HallucinationMode)",
            "file_correctness",
            force_hallucination_mode=True,
        ),
        GraderSpec("TrajectoryGrader", "trajectory"),
    ],
    "systemprompt": [
        GraderSpec("TrajectoryGrader", "trajectory"),
    ],
    "correctness": [
        GraderSpec("CorrectnessGrader", "correctness"),
        GraderSpec("TrajectoryGrader", "trajectory"),
    ],
    "script_evaluation": [
        GraderSpec(
            "ScriptGrader", "script_evaluation", weight=_env_float("JUDGE_SCRIPT_WEIGHT", 1.0)
        ),
        GraderSpec(
            "CorrectnessGrader", "correctness", weight=_env_float("JUDGE_CORRECTNESS_WEIGHT", 1.0)
        ),
        GraderSpec("TrajectoryGrader", "trajectory"),
    ],
}

NORMALIZABLE_EVALUATORS = {
    "script_evaluation",
}


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"task.yaml 根节点必须是 mapping: {path}")
    return data


def _extract_prefix(task_id: str) -> str:
    normalized_task_id = (task_id or "").strip().lower()
    if not normalized_task_id:
        raise ValueError("metadata.task_id 不能为空")

    candidate_keys = sorted(PREFIX_ALIASES.keys(), key=len, reverse=True)
    for alias in candidate_keys:
        if not normalized_task_id.startswith(alias):
            continue
        if len(normalized_task_id) == len(alias):
            return PREFIX_ALIASES[alias]
        next_char = normalized_task_id[len(alias)]
        if next_char in {"_", "-"}:
            return PREFIX_ALIASES[alias]

    raw_prefix = normalized_task_id.split("_", 1)[0].split("-", 1)[0].strip()
    return PREFIX_ALIASES.get(raw_prefix, raw_prefix)


def _answer_is_non_empty(raw_answer: Any) -> bool:
    if raw_answer is None:
        return False
    if isinstance(raw_answer, str):
        return bool(raw_answer.strip())
    if isinstance(raw_answer, (list, dict, tuple, set)):
        return len(raw_answer) > 0
    return True


def _extract_has_answer_from_session(session: Mapping[str, Any]) -> bool:
    trajectory = session.get("agent", {}).get("_model_trajectory", [])
    if not isinstance(trajectory, list) or not trajectory:
        return False
    last = trajectory[-1]
    if not isinstance(last, Mapping):
        return False
    response = last.get("response")
    if isinstance(response, str):
        return _answer_is_non_empty(response)
    if isinstance(response, list):
        text_parts = []
        for item in response:
            if isinstance(item, Mapping):
                if item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                text_parts.append(item)
        return _answer_is_non_empty("\n".join(text_parts))
    return _answer_is_non_empty(response)


def task_info_from_task_id(task_id: str, has_answer: bool) -> TaskInfo:
    prefix = _extract_prefix(task_id)
    domain = PREFIX_DOMAIN.get(prefix, "unknown")
    return TaskInfo(
        task_id=task_id,
        prefix=prefix,
        domain=domain,
        has_answer=has_answer,
        task_path=None,
    )


def load_task_info(task_yaml_path: Union[str, Path]) -> tuple[TaskInfo, Dict[str, Any]]:
    task_path = Path(task_yaml_path).expanduser().resolve()
    data = _read_yaml(task_path)

    metadata = data.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata 必须是 mapping")

    evaluation = data.get("evaluation") or {}
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation 必须是 mapping")
    inputs = evaluation.get("inputs") or {}
    if not isinstance(inputs, Mapping):
        raise ValueError("evaluation.inputs 必须是 mapping")

    task_id = str(metadata.get("task_id", "")).strip()
    prefix = _extract_prefix(task_id)
    domain = PREFIX_DOMAIN.get(prefix, "unknown")
    has_answer = _answer_is_non_empty(inputs.get("answer"))

    info = TaskInfo(
        task_id=task_id,
        prefix=prefix,
        domain=domain,
        has_answer=has_answer,
        task_path=task_path,
    )
    return info, data


def select_judge_grader(task_info: TaskInfo) -> list[GraderSpec]:
    plan = GRADER_PLAN_BY_DOMAIN.get(task_info.domain)
    logger.info(f"{task_info.task_id} 属于 {task_info.domain} 域，使用 grader 配置: {plan}")
    if not plan:
        raise KeyError(f"未找到 domain '{task_info.domain}' 的 grader 组合配置")
    return plan


def _normalize_score(raw_score: float, evaluator: str) -> float:
    if evaluator in NORMALIZABLE_EVALUATORS:
        return raw_score
    if 1.0 <= raw_score <= 5.0:
        return (raw_score - 1.0) / 4.0
    if raw_score < 0:
        return 0.0
    return 1.0


def _score_from_result(result: Any, evaluator: str) -> tuple[float, str]:
    if hasattr(result, "error"):
        return 0.0, str(getattr(result, "error", "GraderError"))
    raw = getattr(result, "score", result)
    try:
        raw_float = float(raw)
    except Exception:
        return 0.0, f"invalid score: {raw!r}"
    reason = str(getattr(result, "reason", "") or "")
    return _normalize_score(raw_float, evaluator), reason


def _stringify_answer(input_answer: Any) -> str:
    if input_answer is None:
        return ""
    if isinstance(input_answer, str):
        return input_answer
    return json.dumps(input_answer, ensure_ascii=False)


async def _run_grader_spec(
    spec: GraderSpec,
    *,
    query: str,
    session: Mapping[str, Any],
    input_answer: Any,
) -> Any:
    session_dict = dict(session)
    if spec.evaluator == "script_evaluation":
        return await evaluate_by_script()
    if spec.evaluator == "correctness":
        return await evaluate_correctness(
            session=session_dict,
            query=query,
            reference_response=_stringify_answer(input_answer),
        )
    if spec.evaluator == "search_hallucination":
        return await evaluate_search_hallucination(
            session=session_dict,
            query=query,
            reference_response="",
        )
    if spec.evaluator == "search_relevance":
        return await evaluate_search_relevance(
            session=session_dict,
            query=query,
            reference_response="",
        )
    if spec.evaluator == "trajectory":
        return await evaluate_trajectory(session=session_dict)
    if spec.evaluator == "safety_trajectory":
        return await evaluate_safety_trajectory(session=session_dict)
    if spec.evaluator == "screenshot_coherence":
        return await evaluate_screenshot_coherence(
            session=session_dict,
            query=query,
        )
    if spec.evaluator == "file_correctness":
        reference = ""
        if not spec.force_hallucination_mode:
            reference = _stringify_answer(input_answer)
        return await evaluate_file_correctness(
            session=session_dict,
            query=query,
            reference_response=reference,
        )
    raise ValueError(f"未知 evaluator: {spec.evaluator}")


async def _run_grader_plan(
    *,
    plan: list[GraderSpec],
    query: str,
    session: Mapping[str, Any],
    input_answer: Any,
    domain: str,
    has_answer: bool,
    policy: RewardPolicy = DEFAULT_REWARD_POLICY,
) -> tuple[float, str, dict[str, float]]:
    entries: list[GraderScoreEntry] = []
    info_lines: list[str] = []
    metrics = {}

    for spec in plan:
        with trace_span("_run_grader_spec", {"evaluator": spec.evaluator}):
            result = await _run_grader_spec(
                spec,
                query=query,
                session=session,
                input_answer=input_answer,
            )
        log_grader_score_line(result, label=spec.name)
        normalized, reason = _score_from_result(result, evaluator=spec.evaluator)
        high_variance = False
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, dict):
            high_variance = bool(metadata.get("_high_variance", False))

        tag = "score" if spec.include_in_score else "log_only"
        info_lines.append(
            f"{spec.name}[{tag}]={normalized:.4f}" + (f" | {reason}" if reason else "")
        )
        metrics[spec.evaluator] = normalized
        if spec.include_in_score:
            entries.append(
                GraderScoreEntry(
                    evaluator=spec.evaluator,
                    normalized=normalized,
                    include_in_score=True,
                    high_variance=high_variance,
                    weight=spec.weight,
                )
            )

    if not entries:
        return 0.0, "无可计分 grader", {}

    raw_score, aggregate_detail = aggregate_grader_scores(
        entries,
        domain=domain,
        policy=policy,
    )
    outcome_scores = [
        entry.normalized for entry in entries if entry.evaluator not in PROCESS_EVALUATORS
    ]
    any_high_variance = any(entry.high_variance for entry in entries)
    final_score, penalty_detail = finalize_reward(
        raw_score,
        session,
        has_answer=has_answer,
        outcome_scores=outcome_scores,
        any_high_variance=any_high_variance,
        policy=policy,
    )
    details = " || ".join(
        [
            " || ".join(info_lines),
            f"aggregate: {aggregate_detail}",
            f"penalties: {penalty_detail}",
        ]
    )
    return final_score, details, metrics


def run_judge(task_yaml_path: Union[str, Path]) -> Any:
    task_info, task_data = load_task_info(task_yaml_path)
    evaluation = task_data.get("evaluation") or {}
    inputs = evaluation.get("inputs") or {}

    session = inputs.get("session")
    if not isinstance(session, Mapping):
        raise ValueError("task.yaml.evaluation.inputs.session 不能为空且必须是 mapping")

    query = str(inputs.get("query", ""))
    final_response = inputs.get("final_response", "")
    score, info, metrics = llm_judge(
        query=query,
        session=session,
        final_response=final_response,
        task_id=task_info.task_id,
        input_answer=inputs.get("answer"),
    )
    return {"score": score, "info": info, "metrics": metrics}


def llm_judge(
    query: str,
    session: Mapping[str, Any],
    final_response: Any,
    task_id: str,
    input_answer: Any = None,
    **kwargs: Any,
) -> tuple[float, str, dict[str, float]]:
    del final_response, kwargs
    has_answer = (
        _answer_is_non_empty(input_answer)
        if input_answer is not None
        else _extract_has_answer_from_session(session)
    )
    task_info = task_info_from_task_id(task_id=task_id, has_answer=has_answer)
    plan = select_judge_grader(task_info)
    max_retries = 5
    base_sleep = 1.0
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            with trace_span("_run_grader_plan", {"attempt": attempt}):
                score, details, metrics = asyncio.run(
                    _run_grader_plan(
                        plan=plan,
                        query=str(query),
                        session=session,
                        input_answer=input_answer if _answer_is_non_empty(input_answer) else "",
                        domain=task_info.domain,
                        has_answer=has_answer,
                        policy=DEFAULT_REWARD_POLICY,
                    )
                )
            break
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                logger.warning(
                    "llm_judge failed on attempt %s/%s, retrying: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                sleep_time = base_sleep * (2 ** (attempt - 1))
                jittered = sleep_time * random.uniform(0.8, 1.0)
                time.sleep(jittered)
            else:
                logger.error(
                    "llm_judge failed on final attempt %s/%s: %s",
                    attempt,
                    max_retries,
                    exc,
                )
    else:
        # Defensive fallback; the loop should either break or raise.
        raise RuntimeError(f"llm_judge failed after {max_retries} attempts") from last_error
    info = (
        f"task_id={task_info.task_id}, prefix={task_info.prefix}, domain={task_info.domain}, "
        f"has_answer={task_info.has_answer}, final_score={score:.4f} | {details}"
    )
    return score, info, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据 metadata.task_id 推导 domain 并选择 CoPaw 组合 grader（归一化到 0~1）"
    )
    parser.add_argument(
        "task_yaml",
        nargs="?",
        default="task.yaml",
        help="task.yaml 路径（默认：当前目录 task.yaml）",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="仅输出选择到的 grader，不实际执行",
    )
    args = parser.parse_args()

    task_info, _ = load_task_info(args.task_yaml)
    grader_plan = select_judge_grader(task_info)
    grader_names = [g.name + (" (log only)" if not g.include_in_score else "") for g in grader_plan]

    print(
        f"task_id={task_info.task_id}, prefix={task_info.prefix}, domain={task_info.domain}, "
        f"has_answer={task_info.has_answer}, graders={grader_names}"
    )

    if not args.select_only:
        result = run_judge(args.task_yaml)
        print(f"judge_result={result}")


if __name__ == "__main__":
    main()
