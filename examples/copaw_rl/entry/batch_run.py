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

为请求注入采样参数（temperature / top_p / top_k / min_p / presence_penalty /
repetition_penalty 等），有两种方式（dashscope / 本地 vLLM 都适用）:

  方式 A — 手动 export env（适合 --models 直接跑预设模型）:
      export AUTO_EVAL_GENERATE_KWARGS='{"temperature":1.0,"top_p":0.95,
        "presence_penalty":1.5,"extra_body":{"top_k":20,"min_p":0.0,
        "repetition_penalty":1.0}}'
      python batch_run.py --models qwen3.6-plus --package search

  方式 B — 在 --models-file JSON 里加 sampling_params 字段（自动注入 env）:
      [{"key":"qwen3.6-plus","provider_id":"dashscope","model":"qwen3.6-plus",
        "sampling_params":{"temperature":1.0,"top_p":0.95,"top_k":20,
                           "min_p":0.0,"presence_penalty":1.5,
                           "repetition_penalty":1.0}}]
      python batch_run.py --models-file dashscope_eval.json

  规则:
    - OpenAI 标准字段（temperature/top_p/presence_penalty 等）会放顶层
    - 非标字段（top_k/min_p/repetition_penalty 等）自动塞 extra_body 透传
    - 若 --models-file 里多个模型的 sampling_params 不一致，按第一个非空为准
      （batch_run 用全局 env 透传，无法每模型独立采样）
    - 已设外部 env 时 --models-file 内的值不会再覆盖
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
