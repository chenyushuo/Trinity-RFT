#!/usr/bin/env python3
"""
batch_run.py — 批量运行 benchmark 任务的入口（薄壳）。

实际逻辑全部在 ./_batch/ 子包里。本文件只负责:
  1. 解析 CLI
  2. 处理 --list-* 仅打印型选项
  3. 把 stdout/stderr Tee 到日志文件
  4. 根据是否传 --retry-from 分派到对应的运行模式

用法示例:
  python batch_run.py                                      # 跑全部任务（默认模型）
  python batch_run.py --models qwen3.6-plus --tasks 178-news-sports
  python batch_run.py --package search safety --parallel 8
  python batch_run.py --models-file eval.json --trial 3 --retry-errors 2
  python batch_run.py --retry-from result/20260330/20260330_213935 --retry-errors 3

  nohup env PYTHONUNBUFFERED=1 python batch_run.py --models-file models.json --package search --parallel 16 &
"""

import asyncio

from _batch import (
    BatchStdoutTee,
    list_models_and_exit,
    list_packages_and_exit,
    parse_cli,
    resolve_models,
    run_full_batch,
    run_retry_from_dir,
)


async def main() -> None:
    args = parse_cli()

    if args.list_packages:
        list_packages_and_exit()
        return
    if args.list_models:
        list_models_and_exit()
        return

    with BatchStdoutTee() as log_path:
        print(f"[batch_run] 日志文件: {log_path}")
        models = resolve_models(args)
        if args.retry_from:
            await run_retry_from_dir(args, models)
        else:
            await run_full_batch(args, models)


if __name__ == "__main__":
    asyncio.run(main())
