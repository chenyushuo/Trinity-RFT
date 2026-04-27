"""CLI 解析 + 模型/任务列表解析 + 三种运行模式。

三种模式:
    list_*       — 仅打印列表后退出
    run_full_batch       — 正常批量运行 (--package / --tasks / --range)
    run_retry_from_dir   — 从已有结果目录重跑失败任务 (--retry-from)

每种模式都遵循相同骨架: 解析 → 调度 → 收尾 (print + save + 可选自动重试)。
"""

import argparse
import asyncio
import os
import random
import re
from collections import Counter
from datetime import datetime

from .constants import (
    ALL_TASKS,
    DEFAULT_MODEL,
    MODELS,
    PACKAGE_CATEGORIES,
    PACKAGE_TASKS,
    RETRYABLE_STATUSES,
    STATUS_LABELS,
)
from .providers import build_provider_config, build_provider_config_from_json
from .results import (
    copy_non_retryable_results,
    load_error_tasks_from_dir,
    merge_retry_results,
    reclassify_fail_results,
)
from .runner import run_retry_tasks, run_single_task
from .summary import (
    print_category_stats,
    print_multi_model_matrix,
    print_single_model_summary,
    print_trial_summary,
    save_batch_summary,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量运行 benchmark 任务")

    p.add_argument(
        "tasks",
        nargs="*",
        help="位置参数：任务 ID（可与 --models 混用，建议优先用 --tasks 避免误解析）",
    )
    p.add_argument(
        "-t", "--tasks", nargs="+", dest="tasks_opt", metavar="TASK",
        help="要运行的任务 ID（与位置参数 tasks 二选一）",
    )
    p.add_argument(
        "--range", nargs=2, type=int, metavar=("START", "END"),
        help="按编号范围跑任务（如 --range 1 8）",
    )
    p.add_argument(
        "-p", "--parallel", type=int, default=8, metavar="N",
        help="最大并发数（默认 8，设为 1 即串行）",
    )
    p.add_argument("--serial", action="store_true", help="串行执行（等同 --parallel 1）")

    p.add_argument(
        "-m", "--models", nargs="+", metavar="MODEL",
        help="预设名 或 provider_id:model_name 自定义格式",
    )
    p.add_argument(
        "--models-file", metavar="FILE",
        help='JSON 模型列表，格式 [{"key":"...","provider_id":"...","model":"..."}]',
    )

    p.add_argument(
        "--package", nargs="+", metavar="PKG",
        help=f"按分类 package 跑任务，可选: {', '.join(PACKAGE_CATEGORIES)}",
    )

    p.add_argument(
        "--shuffle", action="store_true", default=True,
        help="随机打乱任务顺序（默认开启，避免固定顺序带来的偏差）",
    )
    p.add_argument(
        "--no-shuffle", dest="shuffle", action="store_false",
        help="保持任务原始顺序",
    )

    p.add_argument(
        "--retries", type=int, default=3, metavar="N",
        help="基础设施瞬态错误的最大重试次数（默认 3）",
    )
    p.add_argument(
        "--retry-from", metavar="RESULT_DIR",
        help="从指定结果目录重跑 ERROR 任务（读取 _batch_summary.json）",
    )
    p.add_argument(
        "--retry-errors", type=int, default=3, metavar="N",
        help="跑完后自动重试 ERROR 任务的最大轮次（默认 3）",
    )

    p.add_argument(
        "--trial", type=int, default=None, metavar="N",
        help="每个模型重复推理 N 次（也可在 --models-file 里单独配置）",
    )

    p.add_argument("--list-packages", action="store_true", help="列出所有可用 package 并退出")
    p.add_argument("--list-models", action="store_true", help="列出所有预设模型并退出")
    return p.parse_args()


def list_packages_and_exit() -> None:
    print("可用 package 列表:")
    for pkg in PACKAGE_CATEGORIES:
        tasks = PACKAGE_TASKS[pkg]
        print(f"  {pkg:<20s}  {len(tasks)} 个任务")
        for t in tasks:
            print(f"    - {t}")


def list_models_and_exit() -> None:
    print("预设模型列表:")
    for key, cfg in MODELS.items():
        print(f"  {key:<20s}  provider={cfg['provider_id']:<16s} model={cfg['model']}")


# ---------------------------------------------------------------------------
# 模型 / 任务列表解析
# ---------------------------------------------------------------------------


class ResolvedModels:
    """模型解析结果（含 trial 展开后的所有 key）。"""

    def __init__(
        self,
        keys: list[str],
        configs: dict[str, dict],
        trial_groups: dict[str, list[str]],
        multi_model: bool,
    ):
        self.keys = keys
        self.configs = configs
        self.trial_groups = trial_groups
        self.multi_model = multi_model

    @property
    def has_trials(self) -> bool:
        return bool(self.trial_groups)


def _resolve_base_models(args) -> tuple[list[str], dict[str, dict], dict[str, int]]:
    """解析 --models / --models-file（不展开 trial）。"""
    keys: list[str] = []
    configs: dict[str, dict] = {}
    trial_counts: dict[str, int] = {}

    if args.models_file:
        import json
        with open(args.models_file, "r", encoding="utf-8") as f:
            for item in json.load(f):
                key = item["key"]
                keys.append(key)
                configs[key] = build_provider_config_from_json(item)
                trial_counts[key] = item.get("inference_trials", item.get("trial", 1))
    elif args.models:
        for m in args.models:
            keys.append(m)
            configs[m] = build_provider_config(m)
            trial_counts[m] = 1

    if args.trial is not None:
        for k in trial_counts:
            trial_counts[k] = args.trial

    return keys, configs, trial_counts


def _expand_trials(
    keys: list[str], configs: dict[str, dict], trial_counts: dict[str, int]
) -> tuple[list[str], dict[str, dict], dict[str, list[str]]]:
    """trial > 1 时把 key 展开成 key_t1, key_t2, ..."""
    expanded_keys: list[str] = []
    expanded_configs: dict[str, dict] = {}
    groups: dict[str, list[str]] = {}
    for key in keys:
        tc = trial_counts.get(key, 1)
        if tc > 1:
            trial_keys = [f"{key}_t{t}" for t in range(1, tc + 1)]
            groups[key] = trial_keys
            for tk in trial_keys:
                expanded_keys.append(tk)
                expanded_configs[tk] = configs[key]
        else:
            expanded_keys.append(key)
            expanded_configs[key] = configs[key]
    return expanded_keys, expanded_configs, groups


def resolve_models(args) -> ResolvedModels:
    """把 CLI 参数解析成 ResolvedModels（含 trial 展开）。"""
    keys, configs, trial_counts = _resolve_base_models(args)
    multi_model = bool(args.models or args.models_file)

    if not keys:
        keys = [DEFAULT_MODEL]
        configs = {DEFAULT_MODEL: build_provider_config(DEFAULT_MODEL)}
        trial_counts = {DEFAULT_MODEL: args.trial or 1}

    keys, configs, groups = _expand_trials(keys, configs, trial_counts)
    return ResolvedModels(keys, configs, groups, multi_model)


def _resolve_task_ids(args) -> list[str]:
    if args.package:
        ids: list[str] = []
        for pkg in args.package:
            if pkg not in PACKAGE_TASKS:
                print(f"[WARN] Unknown package: {pkg}. Available: {', '.join(PACKAGE_CATEGORIES)}")
                continue
            ids.extend(PACKAGE_TASKS[pkg])
        return ids
    if args.range:
        start, end = args.range
        return [
            t
            for t in ALL_TASKS
            if t.split("-")[0].isdigit() and start <= int(t.split("-")[0]) <= end
        ]
    if args.tasks_opt:
        return args.tasks_opt
    if args.tasks:
        return args.tasks
    return ALL_TASKS


# ---------------------------------------------------------------------------
# 模式 1: 正常批量运行
# ---------------------------------------------------------------------------


async def run_full_batch(args, models: ResolvedModels) -> None:
    """正常批量运行：解析任务 → 跑全部 → 打印 → 保存 → 自动重试 → trial 汇总。"""
    task_ids = _resolve_task_ids(args)
    if not task_ids:
        print("[ERROR] 未解析出任何任务")
        return
    if args.shuffle:
        random.shuffle(task_ids)

    max_parallel = 1 if args.serial else args.parallel
    _print_run_plan(models, task_ids, max_parallel)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_prefix = os.environ.get("RESULT_DATE_PREFIX", datetime.now().strftime("%Y%m%d"))
    result_root = os.path.join("result", date_prefix)

    all_results = await _execute_all_jobs(
        models, task_ids, run_ts, result_root, args.shuffle, max_parallel, args.retries
    )
    all_results = _sort_results(all_results, models.keys, task_ids)

    _print_run_summary(all_results, models, max_parallel, run_ts)
    print_category_stats(all_results)

    batch_dir = os.path.join(result_root, run_ts)
    save_batch_summary(
        batch_dir, all_results, models.keys, task_ids, max_parallel, run_ts, models.multi_model
    )

    if args.retry_errors > 0:
        all_results = await _auto_retry_loop(
            all_results, batch_dir, run_ts, models, max_parallel,
            max_rounds=args.retry_errors, infra_retries=args.retries,
        )
        save_batch_summary(
            batch_dir, all_results, models.keys, task_ids, max_parallel, run_ts, models.multi_model
        )

    if models.has_trials:
        _emit_trial_summary(all_results, models.trial_groups, date_prefix)


def _print_run_plan(models: ResolvedModels, task_ids: list[str], max_parallel: int) -> None:
    if models.has_trials:
        info = ", ".join(f"{k}×{len(v)}" for k, v in models.trial_groups.items())
        print(f"模型(含trial展开): {', '.join(models.keys)}")
        print(f"Trial 配置: {info}")
    else:
        print(f"模型: {', '.join(models.keys)}")
    print(f"任务: {len(task_ids)} 个")
    print(f"总作业数: {len(models.keys) * len(task_ids)} (模型 {len(models.keys)} × 任务 {len(task_ids)})")
    print(f"最大并发: {max_parallel}")
    print(f"任务列表: {', '.join(task_ids)}")


async def _execute_all_jobs(
    models: ResolvedModels,
    task_ids: list[str],
    run_ts: str,
    result_root: str,
    shuffle: bool,
    max_parallel: int,
    infra_retries: int,
) -> list[dict]:
    jobs = [(sample_id, model_key) for sample_id in task_ids for model_key in models.keys]
    if shuffle:
        random.shuffle(jobs)

    if max_parallel <= 1:
        results = []
        for sample_id, model_key in jobs:
            r = await run_single_task(
                sample_id,
                run_ts,
                provider_config=models.configs[model_key],
                model_label=model_key if models.multi_model else "",
                result_root=result_root,
                max_retries=infra_retries,
            )
            results.append(r)
        return results

    sem = asyncio.Semaphore(max_parallel)
    coros = [
        run_single_task(
            sample_id,
            run_ts,
            provider_config=models.configs[model_key],
            model_label=model_key if models.multi_model else "",
            semaphore=sem,
            result_root=result_root,
            max_retries=infra_retries,
        )
        for sample_id, model_key in jobs
    ]
    return list(await asyncio.gather(*coros))


def _sort_results(
    all_results: list[dict], model_keys: list[str], task_ids: list[str]
) -> list[dict]:
    model_order = {m: i for i, m in enumerate(model_keys)}
    task_order = {t: i for i, t in enumerate(task_ids)}
    return sorted(
        all_results,
        key=lambda r: (
            model_order.get(r.get("model", ""), 999),
            task_order.get(r["task"], 999),
        ),
    )


def _print_run_summary(
    all_results: list[dict], models: ResolvedModels, max_parallel: int, run_ts: str
) -> None:
    print(f"\n{'=' * 60}")
    print(f"批量运行完成 — {run_ts}")
    print(f"{'=' * 60}")

    total_passed = total_failed = total_errors = 0
    for model_key in models.keys:
        label = model_key if models.multi_model else ""
        per_model = [r for r in all_results if r.get("model", "") == label]
        p, f_, e = print_single_model_summary(label, per_model, max_parallel)
        total_passed += p
        total_failed += f_
        total_errors += e

    if models.multi_model:
        task_ids = sorted({r["task"] for r in all_results})
        print_multi_model_matrix(models.keys, task_ids, all_results)

    error_note = f" | 错误(未计入): {total_errors}" if total_errors else ""
    print(
        f"\n有效评测: {total_passed + total_failed} | 通过: {total_passed} "
        f"| 失败: {total_failed}{error_note}"
    )
    print(f"并发: {max_parallel}")


# ---------------------------------------------------------------------------
# 自动重试循环（_run_full_batch 用）
# ---------------------------------------------------------------------------


async def _auto_retry_loop(
    all_results: list[dict],
    batch_dir: str,
    run_ts: str,
    models: ResolvedModels,
    max_parallel: int,
    *,
    max_rounds: int,
    infra_retries: int,
) -> list[dict]:
    """同目录原地重跑，最多 max_rounds 轮；返回更新后的 all_results。"""
    retry_tracker: dict[tuple[str, str], int] = {}

    for retry_round in range(1, max_rounds + 1):
        retryable = reclassify_fail_results(all_results, batch_dir)
        if not retryable:
            break

        _print_retry_round_header(retry_round, max_rounds, retryable, batch_dir)

        retry_jobs = []
        for r in retryable:
            ml = r.get("model", "")
            effective_key = ml if ml else models.keys[0]
            cfg = models.configs.get(effective_key)
            if not cfg:
                print(f"  [WARN] 跳过 {r['task']}: 模型 {effective_key} 无配置")
                continue
            retry_jobs.append((ml, r["task"], cfg))
            retry_tracker[(ml, r["task"])] = retry_tracker.get((ml, r["task"]), 0) + 1

        if not retry_jobs:
            break

        new_results = await run_retry_tasks(
            retry_jobs, run_ts,
            max_parallel=max_parallel,
            result_root=os.path.dirname(batch_dir),
            max_retries=infra_retries,
        )
        all_results = merge_retry_results(all_results, new_results)

        remaining = sum(1 for r in all_results if r["status"] in RETRYABLE_STATUSES)
        print(
            f"[AUTO-RETRY {retry_round}] 修复: {len(retryable) - remaining}/{len(retryable)}, "
            f"剩余需重试: {remaining}"
        )

    if retry_tracker:
        for r in all_results:
            key = (r.get("model", ""), r["task"])
            if key in retry_tracker:
                r["retries"] = retry_tracker[key]
        _print_retry_tracker_summary(retry_tracker, all_results)

    return all_results


def _print_retry_round_header(
    retry_round: int, max_rounds: int, retryable: list[dict], batch_dir: str
) -> None:
    counts = Counter(r["status"] for r in retryable)
    parts = [
        f"{counts[s]} 个 {STATUS_LABELS.get(s, s)}"
        for s in ("ERROR", "EMPTY_TRAJ", "INCOMPLETE", "EVAL_CRASH", "EVAL_RETRY")
        if counts.get(s)
    ]
    print(f"\n{'=' * 60}")
    print(f"[AUTO-RETRY {retry_round}/{max_rounds}] 重跑 {' + '.join(parts)}")
    print(f"  结果目录(原地覆盖): {batch_dir}")
    print(f"{'=' * 60}")


def _print_retry_tracker_summary(
    retry_tracker: dict[tuple[str, str], int], all_results: list[dict]
) -> None:
    total_retried = len(retry_tracker)
    max_used = max(retry_tracker.values())
    print(f"\n[RETRY 汇总] 共 {total_retried} 个任务经过重试, 最大重试轮次: {max_used}")
    for (ml, tid), cnt in sorted(retry_tracker.items()):
        match = next(
            (r for r in all_results if r.get("model", "") == ml and r["task"] == tid), None
        )
        final = match["status"] if match else "?"
        tag = f"[{ml}] {tid}" if ml else tid
        print(f"  {tag}: 重试 {cnt} 次, 最终状态: {final}")


# ---------------------------------------------------------------------------
# Trial 平均分汇总
# ---------------------------------------------------------------------------


def _emit_trial_summary(
    all_results: list[dict],
    trial_groups: dict[str, list[str]],
    date_prefix: str,
) -> None:
    for base_key, trial_keys in trial_groups.items():
        summaries = [
            {"results": [r for r in all_results if r.get("model", "") == tk]}
            for tk in trial_keys
        ]
        summaries = [s for s in summaries if s["results"]]
        if len(summaries) > 1:
            print_trial_summary(
                base_key, summaries,
                result_root=os.path.join("result", date_prefix),
                date_str=date_prefix,
            )


def _infer_trial_groups(model_keys: list[str]) -> dict[str, list[str]]:
    """从 ['pai-8b_t1','pai-8b_t2','other'] 推断 trial 分组。"""
    groups: dict[str, list[str]] = {}
    for key in model_keys:
        m = re.match(r"^(.+)_t(\d+)$", key)
        if m:
            groups.setdefault(m.group(1), []).append(key)
    return {base: sorted(keys) for base, keys in groups.items() if len(keys) > 1}


# ---------------------------------------------------------------------------
# 模式 2: --retry-from 已有结果目录
# ---------------------------------------------------------------------------


async def run_retry_from_dir(args, models: ResolvedModels) -> None:
    """从已有结果目录重跑失败任务，结果写到同级新建的 {run_ts}_retryN/。"""
    result_dir = os.path.normpath(args.retry_from)
    summary, error_pairs = load_error_tasks_from_dir(result_dir, include_eval_failures=True)
    if not error_pairs:
        print("没有需要重试的任务")
        return

    run_ts = summary["timestamp"]
    orig_model_keys = summary.get("models", [])
    orig_multi_model = "model_stats" in summary or len(orig_model_keys) > 1
    all_task_ids = sorted({r["task"] for r in summary["results"]})

    if not args.models and not args.models_file:
        models = ResolvedModels(
            keys=orig_model_keys,
            configs={k: build_provider_config(k) for k in orig_model_keys},
            trial_groups={},
            multi_model=orig_multi_model,
        )

    retry_jobs, retryable_keys, skipped = _build_retry_jobs(error_pairs, models)
    if skipped:
        print(f"  跳过 {len(skipped)} 个任务（模型配置缺失）: {skipped[:5]}...")
    if not retry_jobs:
        print("没有可执行的重试任务（所有 ERROR 模型均缺少配置）")
        return

    result_root = os.path.dirname(result_dir)
    retry_ts, retry_batch_dir = _allocate_retry_dir(result_root, result_dir)

    max_parallel = 1 if args.serial else args.parallel
    print(f"重试模式: {result_dir}")
    print(f"需重试: {len(retry_jobs)}, 跳过: {len(skipped)}")
    print(f"结果目录: {retry_batch_dir}")
    print(f"最大并发: {max_parallel}")

    all_results = summary["results"]
    n_copied = copy_non_retryable_results(all_results, retryable_keys, result_dir, retry_batch_dir)
    print(f"  已复制 {n_copied} 个不需要重试的任务结果")

    all_results = await _retry_until_stable(
        all_results, retry_jobs, retry_ts, error_pairs,
        max_parallel=max_parallel, result_root=result_root,
        infra_retries=args.retries, max_rounds=max(args.retry_errors, 1),
    )

    for mk in orig_model_keys if orig_multi_model else [""]:
        per = [r for r in all_results if r.get("model", "") == mk]
        if per:
            print_single_model_summary(mk, per, max_parallel)

    print_category_stats(all_results)
    save_batch_summary(
        retry_batch_dir, all_results, orig_model_keys, all_task_ids,
        max_parallel, retry_ts, orig_multi_model,
    )

    inferred = _infer_trial_groups(orig_model_keys)
    for base_key, trial_keys in inferred.items():
        summaries = [
            {"results": [r for r in all_results if r.get("model", "") == tk]}
            for tk in trial_keys
        ]
        summaries = [s for s in summaries if s["results"]]
        if len(summaries) > 1:
            print_trial_summary(
                base_key, summaries,
                result_root=result_root,
                date_str=os.path.basename(result_root),
            )


def _build_retry_jobs(
    error_pairs: list[tuple[str, str]], models: ResolvedModels
) -> tuple[list[tuple[str, str, dict]], set[tuple[str, str]], list[tuple[str, str]]]:
    retry_jobs: list[tuple[str, str, dict]] = []
    retryable_keys: set[tuple[str, str]] = set()
    skipped: list[tuple[str, str]] = []
    for ml, tid in error_pairs:
        effective_key = ml if ml else (models.keys[0] if models.keys else "")
        if effective_key in models.configs:
            retry_jobs.append((ml, tid, models.configs[effective_key]))
            retryable_keys.add((ml, tid))
        else:
            skipped.append((ml, tid))
    return retry_jobs, retryable_keys, skipped


def _allocate_retry_dir(result_root: str, result_dir: str) -> tuple[str, str]:
    """生成下一个可用的 {base}_retry{N+1} 目录。"""
    dir_basename = os.path.basename(result_dir)
    m = re.match(r"^(.+?)(_retry\d+)?$", dir_basename)
    base = m.group(1) if m else dir_basename
    existing_max = 0
    if os.path.isdir(result_root):
        for d in os.listdir(result_root):
            mm = re.match(re.escape(base) + r"_retry(\d+)$", d)
            if mm:
                existing_max = max(existing_max, int(mm.group(1)))
    retry_ts = f"{base}_retry{existing_max + 1}"
    return retry_ts, os.path.join(result_root, retry_ts)


async def _retry_until_stable(
    all_results: list[dict],
    retry_jobs: list[tuple[str, str, dict]],
    retry_ts: str,
    error_pairs: list[tuple[str, str]],
    *,
    max_parallel: int,
    result_root: str,
    infra_retries: int,
    max_rounds: int,
) -> list[dict]:
    """对 retry_jobs 反复重跑直到没有可重试任务或达到 max_rounds。"""
    for retry_round in range(1, max_rounds + 1):
        current = [
            (ml, tid, cfg)
            for ml, tid, cfg in retry_jobs
            if any(
                r.get("model", "") == ml and r["task"] == tid and r["status"] in RETRYABLE_STATUSES
                for r in all_results
            )
        ]
        if not current:
            break
        if max_rounds > 1:
            print(f"\n{'=' * 60}")
            print(f"[RETRY {retry_round}/{max_rounds}] {len(current)} 个任务")
            print(f"{'=' * 60}")

        new_results = await run_retry_tasks(
            current, retry_ts,
            max_parallel=max_parallel,
            result_root=result_root,
            max_retries=infra_retries,
        )
        all_results = merge_retry_results(all_results, new_results)

        remaining = sum(1 for r in all_results if r["status"] in RETRYABLE_STATUSES)
        fixed = len(error_pairs) - remaining
        print(f"\n[RETRY {retry_round}] 修复: {fixed}/{len(error_pairs)}, 剩余需重试: {remaining}")

    return all_results
