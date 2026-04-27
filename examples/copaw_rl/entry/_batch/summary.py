"""指标计算 / 终端打印 / 写 _batch_summary.json / Trial 平均分汇总。

按"先纯计算 → 再渲染 → 最后持久化"分层；任何函数都不互相依赖输出顺序。
"""

import json
import os
from collections import defaultdict
from datetime import datetime

from .constants import STATUS_ICONS, TASK_TO_CATEGORY


def avg_valid(values, min_val: int = 0) -> float:
    """计算 >= min_val 的有效值的均值；无有效值返回 -1。"""
    valid = [v for v in values if v >= min_val]
    return sum(valid) / len(valid) if valid else -1


def build_result_maps(
    all_results: list[dict],
) -> tuple[dict, dict, dict, dict, dict]:
    """从结果列表构建 {model: {task: value}} 的 5 张表 (score/steps/length/duration/status)。"""
    score_map: dict[str, dict] = {}
    steps_map: dict[str, dict] = {}
    length_map: dict[str, dict] = {}
    duration_map: dict[str, dict] = {}
    status_map: dict[str, dict] = {}
    for r in all_results:
        m, t = r.get("model", ""), r["task"]
        score_map.setdefault(m, {})[t] = r["score"]
        steps_map.setdefault(m, {})[t] = r.get("steps", -1)
        length_map.setdefault(m, {})[t] = r.get("response_length", -1)
        duration_map.setdefault(m, {})[t] = r.get("duration_seconds", -1)
        status_map.setdefault(m, {})[t] = r["status"]
    return score_map, steps_map, length_map, duration_map, status_map


def compute_category_stats(results: list[dict]) -> dict[str, dict]:
    """按 PACKAGE_TASKS 类别分组，计算平均分 / 通过率。"""
    cat_results: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        cat_results[TASK_TO_CATEGORY.get(r["task"], "unknown")].append(r)

    stats: dict[str, dict] = {}
    for cat in sorted(cat_results):
        items = cat_results[cat]
        evaluated = [r for r in items if r["status"] != "ERROR"]
        passed = sum(1 for r in evaluated if r["status"] == "PASS")
        failed = sum(1 for r in evaluated if r["status"] == "FAIL")
        avg_score = avg_valid([r["score"] for r in evaluated])
        pass_rate = passed / len(evaluated) * 100 if evaluated else 0
        stats[cat] = {
            "total": len(items),
            "evaluated": len(evaluated),
            "passed": passed,
            "failed": failed,
            "errors": len(items) - len(evaluated),
            "avg_score": round(avg_score, 1) if avg_score >= 0 else -1,
            "pass_rate": round(pass_rate, 1),
        }
    return stats


def _model_aggregate(
    all_results: list[dict],
    model_keys: list[str],
    task_ids: list[str],
) -> dict[str, dict]:
    """每个模型的整体统计（写入 _batch_summary.json["model_stats"]）。"""
    score_map, steps_map, length_map, duration_map, status_map = build_result_maps(all_results)
    out: dict[str, dict] = {}
    for m in model_keys:
        scores = score_map.get(m, {})
        statuses = status_map.get(m, {})

        def _valid(d):
            return [v for t, v in d.get(m, {}).items() if statuses.get(t) != "ERROR" and v >= 0]

        vs, vst, vl, vd = _valid(score_map), _valid(steps_map), _valid(length_map), _valid(duration_map)
        n_errors = sum(1 for s in statuses.values() if s == "ERROR")
        avg_len = avg_valid(vl)
        out[m] = {
            "avg_score": avg_valid(vs),
            "avg_steps": avg_valid(vst),
            "avg_response_length": int(avg_len) if avg_len >= 0 else -1,
            "avg_duration_seconds": avg_valid(vd),
            "passed": sum(1 for s in vs if s == 100),
            "total": len(task_ids),
            "evaluated": len(task_ids) - n_errors,
            "errors": n_errors,
            "scores": scores,
            "steps": steps_map.get(m, {}),
            "response_lengths": length_map.get(m, {}),
            "duration_seconds": duration_map.get(m, {}),
        }
    return out


def print_single_model_summary(
    model_label: str, results: list[dict], max_parallel: int
) -> tuple[int, int, int]:
    """打印单模型结果摘要。ERROR（基础设施故障）不计入有效评测。"""
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    evaluated = passed + failed
    valid = [r for r in results if r["status"] != "ERROR"]

    avg_steps = avg_valid([r.get("steps", -1) for r in valid])
    avg_len_raw = avg_valid([r.get("response_length", -1) for r in valid])
    avg_len = int(avg_len_raw) if avg_len_raw >= 0 else -1
    avg_time = avg_valid([r.get("duration_seconds", -1) for r in valid])
    avg_score = avg_valid([r["score"] for r in valid])

    header = f"模型: {model_label}" if model_label else "运行结果"
    print(f"\n  {header}")
    error_note = f" | 错误(未计入): {errors}" if errors else ""
    print(
        f"  有效评测: {evaluated} | 通过: {passed} | 失败: {failed}{error_note} "
        f"| 平均分: {_fmt(avg_score)} | 平均步数: {_fmt(avg_steps)} "
        f"| 平均输出(计费): {avg_len if avg_len >= 0 else 'N/A'} "
        f"| 平均时延: {_fmt(avg_time, suffix='s')}"
    )
    print(f"  {'-' * 56}")
    for r in results:
        icon = STATUS_ICONS.get(r["status"], "?")
        steps_str = str(r["steps"]) if r.get("steps", -1) >= 0 else "N/A"
        len_str = str(r["response_length"]) if r.get("response_length", -1) >= 0 else "N/A"
        time_str = (
            f"{r['duration_seconds']:.1f}s" if r.get("duration_seconds", -1) >= 0 else "N/A"
        )
        retry_str = f" retry={r['retries']}" if r.get("retries", 0) > 0 else ""
        print(
            f"    [{icon}] {r['task']:<40s} score={r['score']:<6} "
            f"steps={steps_str} 输出={len_str} time={time_str}{retry_str}"
        )
    return passed, failed, errors


def _fmt(value: float, suffix: str = "") -> str:
    return f"{value:.1f}{suffix}" if value >= 0 else "N/A"


def print_category_stats(results: list[dict]) -> dict[str, dict]:
    """打印按类别的平均分和通过率，返回 stats（不再二次计算）。"""
    stats = compute_category_stats(results)
    if not stats:
        return stats
    print(f"\n{'=' * 70}")
    print("按类别统计")
    print(f"{'=' * 70}")
    print(f"  {'类别':<20s} {'评测':>4s} {'通过':>4s} {'失败':>4s} {'通过率':>7s} {'平均分':>7s}")
    print(f"  {'-' * 56}")
    for cat, s in stats.items():
        avg_str = f"{s['avg_score']:.1f}" if s["avg_score"] >= 0 else "N/A"
        err_mark = f" (+{s['errors']}err)" if s["errors"] else ""
        print(
            f"  {cat:<20s} {s['evaluated']:>4d} {s['passed']:>4d} {s['failed']:>4d} "
            f"{s['pass_rate']:>6.1f}% {avg_str:>7s}{err_mark}"
        )
    print(f"  {'-' * 56}")
    return stats


def print_multi_model_matrix(
    model_keys: list[str], task_ids: list[str], all_results: list[dict]
) -> None:
    """打印 模型×任务 对比矩阵。ERROR 任务排除在平均之外。"""
    score_map, steps_map, length_map, duration_map, status_map = build_result_maps(all_results)
    col_w = max(max((len(m) for m in model_keys), default=6), 14)

    print(f"\n{'=' * 60}")
    print("模型 × 任务 对比矩阵 (score / steps / 输出计费长度 / 时延s)")
    print(f"{'=' * 60}")
    header = f"{'任务':<40s}" + "".join(f" | {m:>{col_w}s}" for m in model_keys)
    print(header)
    print("-" * len(header))

    for task in task_ids:
        row = f"{task:<40s}"
        for m in model_keys:
            score = score_map.get(m, {}).get(task, -1)
            steps = steps_map.get(m, {}).get(task, -1)
            length = length_map.get(m, {}).get(task, -1)
            duration = duration_map.get(m, {}).get(task, -1)
            score_str = str(int(score)) if score >= 0 else "ERR"
            cell = (
                f"{score_str}/"
                f"{steps if steps >= 0 else '-'}/"
                f"{length if length >= 0 else '-'}/"
                f"{f'{duration:.1f}' if duration >= 0 else '-'}"
            )
            row += f" | {cell:>{col_w}s}"
        print(row)

    print("-" * len(header))
    avg_row = f"{'平均(排除ERR)':<40s}"
    for m in model_keys:
        statuses = status_map.get(m, {})
        valid = [t for t in task_ids if statuses.get(t) != "ERROR"]
        avg_score = avg_valid([score_map.get(m, {}).get(t, -1) for t in valid])
        avg_steps = avg_valid([steps_map.get(m, {}).get(t, -1) for t in valid])
        avg_len_raw = avg_valid([length_map.get(m, {}).get(t, -1) for t in valid])
        avg_len = int(avg_len_raw) if avg_len_raw >= 0 else -1
        avg_dur = avg_valid([duration_map.get(m, {}).get(t, -1) for t in valid])
        n_err = len(task_ids) - len(valid)
        cell = (
            f"{_fmt(avg_score)}/{_fmt(avg_steps)}/"
            f"{avg_len if avg_len >= 0 else 'N/A'}/{_fmt(avg_dur)}"
        )
        if n_err:
            cell += f"({n_err}err)"
        avg_row += f" | {cell:>{col_w}s}"
    print(avg_row)


def save_batch_summary(
    batch_dir: str,
    all_results: list[dict],
    model_keys: list[str],
    task_ids: list[str],
    max_parallel: int,
    run_ts: str,
    multi_model: bool,
) -> str:
    """把 _batch_summary.json 写到 batch_dir/_batch_summary.json，返回路径。"""
    os.makedirs(batch_dir, exist_ok=True)

    passed = sum(1 for r in all_results if r["status"] == "PASS")
    failed = sum(1 for r in all_results if r["status"] == "FAIL")
    errors = sum(1 for r in all_results if r["status"] == "ERROR")

    retried = [r for r in all_results if r.get("retries", 0) > 0]
    retry_stats = None
    if retried:
        retry_stats = {
            "total_retried_tasks": len(retried),
            "max_retries": max(r["retries"] for r in retried),
            "tasks": [
                {
                    "model": r.get("model", ""),
                    "task": r["task"],
                    "retries": r["retries"],
                    "final_status": r["status"],
                }
                for r in retried
            ],
        }

    batch_summary: dict = {
        "timestamp": run_ts,
        "models": model_keys,
        "parallel": max_parallel,
        "total": len(all_results),
        "evaluated": passed + failed,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "category_stats": compute_category_stats(all_results),
        "results": all_results,
    }
    if retry_stats:
        batch_summary["retry_stats"] = retry_stats
    if multi_model:
        batch_summary["model_stats"] = _model_aggregate(all_results, model_keys, task_ids)

    summary_path = os.path.join(batch_dir, "_batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存到: {summary_path}")
    return summary_path


def print_trial_summary(
    base_key: str,
    trial_summaries: list[dict],
    result_root: str,
    date_str: str | None = None,
) -> None:
    """打印同一模型 N 次 trial 的平均分，并写到 _trial_avg/{base_key}.json。"""
    n_trials = len(trial_summaries)
    task_trials: dict[str, list[dict]] = {}
    for summary in trial_summaries:
        for r in summary.get("results", []):
            task_trials.setdefault(r["task"], []).append(r)

    print(f"\n{'=' * 80}")
    print(f"模型 {base_key} — {n_trials} 次 Trial 平均分汇总")
    print(f"{'=' * 80}")

    per_trial_avgs: list[float] = []
    for t_idx, summary in enumerate(trial_summaries, 1):
        results = summary.get("results", [])
        valid = [
            r["score"] for r in results if r.get("status") != "ERROR" and r["score"] >= 0
        ]
        avg = avg_valid(valid)
        per_trial_avgs.append(avg)
        print(f"  Trial {t_idx}: 平均分={_fmt(avg)} (有效任务={len(valid)})")

    all_task_avgs: list[float] = []
    per_task_data: dict[str, dict] = {}
    for task in sorted(task_trials.keys()):
        scores = [t["score"] for t in task_trials[task]]
        avg = avg_valid(scores)
        per_task_data[task] = {"scores": scores, "avg": round(avg, 2) if avg >= 0 else -1}
        if avg >= 0:
            all_task_avgs.append(avg)

    overall = avg_valid(all_task_avgs)
    print(f"  总平均分: {_fmt(overall)}")

    avg_data = {
        "model": base_key,
        "n_trials": n_trials,
        "per_trial_avg": [round(a, 2) if a >= 0 else -1 for a in per_trial_avgs],
        "overall_avg": round(overall, 2) if overall >= 0 else -1,
        "per_task": per_task_data,
    }
    if not date_str:
        date_str = datetime.now().strftime("%Y%m%d")
    trial_dir = os.path.join(result_root, "_trial_avg")
    os.makedirs(trial_dir, exist_ok=True)
    avg_path = os.path.join(trial_dir, f"{base_key}.json")
    with open(avg_path, "w", encoding="utf-8") as f:
        json.dump(avg_data, f, ensure_ascii=False, indent=2)
    print(f"  Trial 平均分已保存到: {avg_path}")
