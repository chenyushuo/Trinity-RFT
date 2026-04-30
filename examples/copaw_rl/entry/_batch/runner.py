"""单任务执行 + 基础设施瞬态错误重试 + 批量重试调度。

关键入口:
    run_single_task(...)      — 在 sandbox 里跑一个任务，返回结果摘要 dict
    run_retry_tasks(jobs, ..) — 按 serial/parallel 调度跑一批重试任务
"""

import asyncio
import importlib.util
import os
import traceback


def _load_sandbox_utils():
    """ad-hoc 加载 ../workflows/sandbox_utils.py（保持原有的非常规 import 兼容）。"""
    spec = importlib.util.spec_from_file_location(
        "sandbox_utils",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "workflows",
            "sandbox_utils.py",
        ),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sandbox_utils = _load_sandbox_utils()
get_or_create_sandbox = _sandbox_utils.get_or_create_sandbox
run_eval_workflow = _sandbox_utils.run_eval_workflow
run_teacher_eval_workflow = _sandbox_utils.run_teacher_eval_workflow


def _is_infra_error(e: Exception) -> bool:
    """是否为基础设施瞬态错误（值得指数退避重试）。"""
    if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
        return True
    msg = str(e)
    if isinstance(e, RuntimeError) and "Failed to create environment" in msg:
        return True
    transient_markers = (
        "peer closed connection",
        "incomplete chunked read",
        "RemoteProtocolError",
    )
    if any(s in msg for s in transient_markers):
        return True
    if isinstance(e, KeyError) and str(e) == "'code'":
        return True
    return False


def _build_oss_config() -> dict:
    return {
        "access_key_id": os.environ.get("OSS_ACCESS_KEY_ID"),
        "access_key_secret": os.environ.get("OSS_ACCESS_KEY_SECRET"),
        "region": os.environ.get("OSS_REGION"),
        "endpoint": os.environ.get("OSS_ENDPOINT"),
        "bucket_name": os.environ.get("OSS_BUCKET_NAME"),
        "prefix": os.environ.get("OSS_PREFIX"),
    }


def _should_use_dashscope(
    provider_config: dict | None, api_server_url: str | None
) -> tuple[bool, str | None]:
    """判断是否走 teacher 评测路径（dashscope 直连，绕过 vLLM）。"""
    if not provider_config or api_server_url:
        return False, None
    active = provider_config.get("active_llm", {})
    pid = active.get("provider_id", "")
    mid = active.get("model", "")
    if pid == "dashscope" and mid:
        return True, mid
    return False, None


async def run_single_task(
    sample_id: str,
    run_ts: str,
    provider_config: dict | None = None,
    model_label: str = "",
    semaphore: asyncio.Semaphore | None = None,
    result_root: str = "result",
    max_retries: int = 3,
) -> dict:
    """跑单个任务，返回结果摘要；瞬态错误自动指数退避重试。"""
    tag = f"[{model_label}] {sample_id}" if model_label else sample_id

    def _inner():
        from trinity.utils.log import get_logger

        logger = get_logger()
        sandbox, _created = get_or_create_sandbox(
            "",
            os.environ.get("E2B_API_KEY"),
            os.environ.get("E2B_DOMAIN"),
            os.environ.get("E2B_TEMPLATE"),
            logger,
        )
        oss_config = _build_oss_config()
        dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
        api_server_url = os.environ.get("AUTO_EVAL_BASE_URL")
        model_path = os.environ.get("AUTO_EVAL_MODEL_ID")

        use_dashscope, dashscope_model_id = _should_use_dashscope(provider_config, api_server_url)

        try:
            if use_dashscope:
                metrics = run_teacher_eval_workflow(
                    sandbox,
                    sample_id,
                    oss_config,
                    dashscope_api_key,
                    dashscope_model_id,
                    model_label,
                    os.path.join(result_root, run_ts),
                    logger,
                )
            else:
                metrics = run_eval_workflow(
                    sandbox,
                    sample_id,
                    oss_config,
                    dashscope_api_key,
                    api_server_url,
                    model_path,
                    model_label,
                    os.path.join(result_root, run_ts),
                    logger,
                )
        finally:
            sandbox.kill()
        return metrics

    for attempt in range(max_retries + 1):
        try:
            if semaphore:
                async with semaphore:
                    return await asyncio.to_thread(_inner)
            return await asyncio.to_thread(_inner)
        except Exception as e:
            if attempt < max_retries and _is_infra_error(e):
                wait = min(2**attempt * 5, 60)
                print(
                    f"[RETRY] {tag} attempt {attempt + 1}/{max_retries}, "
                    f"waiting {wait}s... ({type(e).__name__}: {e})"
                )
                await asyncio.sleep(wait)
                continue
            print(f"[ERROR] {tag}: {e}")
            traceback.print_exc()

    return {
        "task": sample_id,
        "model": model_label,
        "score": -1,
        "status": "ERROR",
        "steps": -1,
        "response_length": -1,
        "latency_seconds": -1,
        "duration_seconds": -1,
    }


async def run_retry_tasks(
    jobs: list[tuple[str, str, dict]],
    run_ts: str,
    *,
    max_parallel: int,
    result_root: str,
    max_retries: int,
) -> list[dict]:
    """按 serial / parallel 调度跑一批 (model_label, task_id, provider_config) 重试任务。"""
    if max_parallel <= 1:
        results: list[dict] = []
        for ml, tid, cfg in jobs:
            r = await run_single_task(
                tid,
                run_ts,
                provider_config=cfg,
                model_label=ml,
                result_root=result_root,
                max_retries=max_retries,
            )
            results.append(r)
        return results

    sem = asyncio.Semaphore(max_parallel)
    coros = [
        run_single_task(
            tid,
            run_ts,
            provider_config=cfg,
            model_label=ml,
            semaphore=sem,
            result_root=result_root,
            max_retries=max_retries,
        )
        for ml, tid, cfg in jobs
    ]
    return list(await asyncio.gather(*coros))
