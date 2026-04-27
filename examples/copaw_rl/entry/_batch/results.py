"""结果目录探查、状态分类、重试结果合并。

只读 / 写 result/{date}/{run_ts}/[<model>/]<task>/ 下的 summary.json / session.json。
"""

import json
import os
import shutil
from collections import Counter

from .constants import EVAL_API_FAILURE_PATTERNS, RETRYABLE_STATUSES, STATUS_LABELS


def _task_dir(result_dir: str, model_label: str, task_id: str) -> str:
    return (
        os.path.join(result_dir, model_label, task_id)
        if model_label
        else os.path.join(result_dir, task_id)
    )


def _read_json(path: str) -> dict | list | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _llm_graders_complete(ev: dict) -> bool:
    """LLM 评分器是否全部产出了有效 score（无 API 错误）。"""
    grader_results = ev.get("grader_results") or []
    llm = [
        g
        for g in grader_results
        if str(g.get("type", "")).lower() == "llm_grader" and not g.get("record_only")
    ]
    if not llm:
        return False
    for g in llm:
        if g.get("score") is None:
            return False
        ts = g.get("trial_scores")
        if isinstance(ts, list) and len(ts) == 0:
            return False
        err = g.get("error")
        if err and EVAL_API_FAILURE_PATTERNS.search(str(err)):
            return False
    rows = ev.get("llm_grader_scores")
    if isinstance(rows, list) and rows:
        if any(row.get("score") is None for row in rows):
            return False
    return True


def has_eval_api_failure(result_dir: str, model_label: str, task_id: str) -> bool:
    """summary.json 里是否包含评测阶段 API 错误。"""
    summary = _read_json(os.path.join(_task_dir(result_dir, model_label, task_id), "summary.json"))
    if not isinstance(summary, dict):
        return False
    for task in summary.get("tasks", []):
        ev = task.get("evaluation", {})
        if _llm_graders_complete(ev):
            continue
        details = ev.get("details") or ""
        if EVAL_API_FAILURE_PATTERNS.search(details):
            return True
        for gr in ev.get("grader_results") or []:
            err = gr.get("error") or ""
            if EVAL_API_FAILURE_PATTERNS.search(err):
                return True
    return False


def has_eval_crash(result_dir: str, model_label: str, task_id: str) -> bool:
    """评测中途崩溃：pytest 未跑完导致 tests.total == 0 但 steps > 0。"""
    summary = _read_json(os.path.join(_task_dir(result_dir, model_label, task_id), "summary.json"))
    if not isinstance(summary, dict):
        return False
    for task in summary.get("tasks", []):
        ev = task.get("evaluation", {})
        tests = ev.get("tests", {})
        if tests.get("total", -1) == 0 and task.get("steps", 0) > 0:
            return True
    return False


def _find_session_conversation(session) -> list | None:
    """从 session.json 里递归找 [[msg, meta], ...] 形式的会话数组。"""
    if isinstance(session, list) and session:
        if isinstance(session[0], list) and len(session[0]) >= 2:
            return session
    if isinstance(session, dict):
        for k, v in session.items():
            if k == "_model_trajectory":
                continue
            found = _find_session_conversation(v)
            if found:
                return found
    return None


def has_incomplete_execution(result_dir: str, model_label: str, task_id: str) -> bool:
    """轨迹被截断：session 最后一条消息是 tool_result。"""
    session = _read_json(os.path.join(_task_dir(result_dir, model_label, task_id), "session.json"))
    conv = _find_session_conversation(session) if session is not None else None
    if not conv:
        return False
    last = conv[-1]
    msg = last[0] if isinstance(last, list) else last
    if not isinstance(msg, dict):
        return False
    content = msg.get("content", "")
    if isinstance(content, list):
        types = {c.get("type", "") for c in content if isinstance(c, dict)}
        return "tool_result" in types
    return False


def reclassify_fail_results(
    all_results: list[dict],
    result_dir: str,
    *,
    include_eval_failures: bool = True,
) -> list[dict]:
    """把 FAIL 结果细分为 EMPTY_TRAJ / INCOMPLETE / EVAL_CRASH / EVAL_RETRY，并就地修改 status。

    返回所有可重试的条目（含原本 ERROR 的）。
    """
    retryable = [r for r in all_results if r["status"] == "ERROR"]
    seen = {(r.get("model", ""), r["task"]) for r in retryable}

    checks: list[tuple] = [
        (lambda r, _ml, _tid: r.get("steps", -1) <= 0, "EMPTY_TRAJ"),
        (lambda _r, ml, tid: has_incomplete_execution(result_dir, ml, tid), "INCOMPLETE"),
        (lambda _r, ml, tid: has_eval_crash(result_dir, ml, tid), "EVAL_CRASH"),
    ]
    if include_eval_failures:
        checks.append(
            (lambda _r, ml, tid: has_eval_api_failure(result_dir, ml, tid), "EVAL_RETRY"),
        )

    for check_fn, new_status in checks:
        for r in all_results:
            if r["status"] != "FAIL":
                continue
            ml, tid = r.get("model", ""), r["task"]
            if (ml, tid) in seen:
                continue
            if check_fn(r, ml, tid):
                r["status"] = new_status
                retryable.append(r)
                seen.add((ml, tid))
    return retryable


def merge_retry_results(
    original_results: list[dict], retry_results: list[dict]
) -> list[dict]:
    """用重试结果替换原始结果中对应 (model, task) 的条目。"""
    retry_map = {(r.get("model", ""), r["task"]): r for r in retry_results}
    return [retry_map.get((r.get("model", ""), r["task"]), r) for r in original_results]


def copy_non_retryable_results(
    all_results: list[dict],
    retryable_keys: set[tuple[str, str]],
    src_dir: str,
    dst_dir: str,
) -> int:
    """把不需要重试的任务结果目录从 src 拷到 dst，返回拷贝条目数。"""
    n_copied = 0
    for r in all_results:
        ml, tid = r.get("model", ""), r["task"]
        if (ml, tid) in retryable_keys:
            continue
        src = _task_dir(src_dir, ml, tid)
        dst = _task_dir(dst_dir, ml, tid)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            n_copied += 1
    return n_copied


def load_error_tasks_from_dir(
    result_dir: str,
    *,
    include_eval_failures: bool = False,
) -> tuple[dict, list[tuple[str, str]]]:
    """从 result_dir/_batch_summary.json 读出需要重跑的 (model, task) 列表。

    默认只读 ERROR；include_eval_failures=True 时还会扫 FAIL 任务的 summary.json
    把评测阶段 API 错误的任务也纳入重跑。
    """
    summary_path = os.path.join(result_dir, "_batch_summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"找不到 {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    retryable = reclassify_fail_results(
        summary.get("results", []),
        result_dir,
        include_eval_failures=include_eval_failures,
    )

    counts = Counter(r["status"] for r in retryable)
    for status, count in sorted(counts.items()):
        label = STATUS_LABELS.get(status, status)
        print(f"  发现 {count} 个{label}任务需要重跑")

    error_pairs = [(r.get("model", ""), r["task"]) for r in retryable]
    return summary, error_pairs
