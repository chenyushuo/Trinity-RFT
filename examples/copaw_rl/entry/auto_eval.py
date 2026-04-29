#!/usr/bin/env python3
"""
auto_eval.py — vLLM 部署 + benchmark 评测流水线。

每个模型按顺序:
    启动 vLLM  →  健康检查  →  调用 batch_run.py  →  kill vLLM  →  下一个模型

用法:
    python auto_eval.py --config eval_models.json
    python auto_eval.py --config eval_models.json --package search safety --parallel 8
    python auto_eval.py --config eval_models.json --tasks 001-team-building-cities --gpus 2,3,4,5

后台运行:
    nohup env PYTHONUNBUFFERED=1 python auto_eval.py --config eval_qwen3.5_9b_base.json --parallel 16 &

终止全部相关进程:
    pkill -9 -f "auto_eval"; pkill -9 -f "VLLM::"; pkill -9 -f "batch_run"

日志/结果目录:
    logs/{date}/auto_eval_{HHMMSS}.log         — 主日志
    logs/{date}/vllm_logs/{model_key}_{ts}.log — vLLM 进程日志
    result/{date}/{run_ts}/...                  — batch_run 结果

配置文件 (eval_models.json) 字段:
    key             模型简称（用作目录名）
    model_path      模型本地路径
    tp / dp         tensor / data parallel size
    dtype           默认 bfloat16
    tool_call_parser 默认 qwen3_xml
    extra_args      vLLM serve 启动期额外 CLI 参数（如 --gpu-memory-utilization）
    sampling_params 采样参数（temperature / top_p / top_k / min_p /
                    presence_penalty / repetition_penalty 等）。
                    会同时走两条路径以确保完整生效：
                      ① 启动期 --override-generation-config 固化给 vLLM
                         （仅 HF GenerationConfig 标准字段被 vLLM 接受，即
                         temperature/top_p/top_k/min_p/repetition_penalty）
                      ② 通过 AUTO_EVAL_GENERATE_KWARGS env 透传到 sandbox
                         的 run.py，再写进 QwenPaw provider 配置，CoPaw 调
                         OpenAI 兼容接口时把这些值作为每请求 kwargs 下发
                         （覆盖 ① 中的服务端默认值，且能让 presence_penalty
                         这种非 HF 标准字段也生效）
                    仅作用于 vLLM 评测路径，不影响 dashscope 直连。
    inference_trials 同模型重复推理次数（默认 1，与 grading_trials 不同概念）
    max_model_len   默认 98304
    model_id        覆盖 model_path 作为 OpenAI 兼容 API 的 model 字段
"""

import argparse
import atexit
import ctypes
import errno
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

from _batch.providers import sampling_params_to_generate_kwargs
from _batch.summary import print_trial_summary

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_BASE = os.path.join(SCRIPT_DIR, "logs")
RESULT_BASE = os.path.join(SCRIPT_DIR, "result")
VLLM_BIN = os.path.join(sys.exec_prefix, "bin", "vllm")

DEFAULT_PORT_RANGE = (29000, 29100)
HEALTH_TIMEOUT = 1200
HEALTH_INTERVAL = 5
HOST_IP = subprocess.getoutput("hostname -I").strip().split()[0]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def _setup_logging(date_str: str, start_ts: str) -> str:
    """同时输出到 console 和 logs/{date}/auto_eval_{time}.log。"""
    day_dir = os.path.join(LOG_BASE, date_str)
    os.makedirs(day_dir, exist_ok=True)
    log_file = os.path.join(day_dir, f"auto_eval_{start_ts}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (logging.FileHandler(log_file, encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    return log_file


def _find_free_gpus() -> list[int]:
    """nvidia-smi 找出显存占用 < 1GiB 的空闲 GPU。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return []
    free = []
    for line in out.strip().splitlines():
        idx_str, mem_str = line.split(",")
        if int(mem_str.strip()) < 1024:
            free.append(int(idx_str.strip()))
    return sorted(free)


def _fetch_served_model_ids(port: int, timeout: float = 5.0) -> list[str] | None:
    """GET /v1/models，返回 data[].id 列表；任何错误返回 None。"""
    url = f"http://localhost:{port}/v1/models"
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    return [m.get("id", "") for m in data if isinstance(m, dict)]


def _wait_for_health(
    port: int,
    expected_model_id: str,
    timeout: int = HEALTH_TIMEOUT,
) -> bool:
    """轮询 /v1/models，必须返回 200 且 data[].id 中含 expected_model_id 才算就绪。

    严格匹配可避免端口被旧 vLLM 占用时把"别的模型 server"误认成自己人。
    """
    deadline = time.time() + timeout
    last_seen: list[str] | None = None
    while time.time() < deadline:
        ids = _fetch_served_model_ids(port)
        if ids is not None:
            last_seen = ids
            if expected_model_id in ids:
                return True
        time.sleep(HEALTH_INTERVAL)
    if last_seen is not None:
        log.warning("  健康检查超时；最后看到的 models=%s（期望含 %s）", last_seen, expected_model_id)
    return False


# ---------------------------------------------------------------------------
# 端口预检 / 残留 vLLM 清理
# ---------------------------------------------------------------------------


def _is_port_free(port: int) -> bool:
    """通过 0.0.0.0:port 试 bind 判断端口是否空闲。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, errno.EACCES):
            return False
        raise
    finally:
        s.close()
    return True


def _pick_free_port(start: int, end: int) -> int:
    """在 [start, end] 范围内顺序挑第一个空闲端口；都不空就 raise。"""
    for p in range(start, end + 1):
        if _is_port_free(p):
            return p
    raise RuntimeError(f"端口范围 {start}-{end} 内没有空闲端口")


def _list_listening_pids(port: int) -> list[int]:
    """列出当前监听 port 的所有 PID（用 ss）。"""
    try:
        out = subprocess.check_output(
            ["ss", "-tlnpH", f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []
    pids: set[int] = set()
    for line in out.strip().splitlines():
        for token in line.split():
            if token.startswith("users:"):
                for piece in token.split(","):
                    if piece.startswith("pid="):
                        try:
                            pids.add(int(piece.split("=", 1)[1]))
                        except ValueError:
                            pass
    return sorted(pids)


def _is_vllm_process(pid: int) -> bool:
    """严格判断 /proc/<pid> 是否为 vllm 进程，避免误杀普通 Python 脚本。

    判定标准（满足任一）:
      - /proc/<pid>/comm 是 'vllm' 或 'VLLM::*'（vLLM 改名后的子进程）
      - cmdline 的 argv 里有以 'vllm' 结尾的可执行路径，或某个 arg 以 'VLLM::' 开头
    """
    try:
        with open(f"/proc/{pid}/comm", "r", encoding="utf-8", errors="replace") as f:
            comm = f.read().strip()
    except OSError:
        return False
    if comm == "vllm" or comm.startswith("VLLM::"):
        return True

    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except OSError:
        return False
    argv = [a for a in raw.split(b"\x00") if a]
    if not argv:
        return False
    if os.path.basename(argv[0].decode("utf-8", errors="replace")) == "vllm":
        return True
    for a in argv:
        if a.startswith(b"VLLM::"):
            return True
    return False


def _free_port_or_raise(port: int) -> None:
    """端口被占用时只 kill 是 vllm/VLLM:: 的进程；其他占用直接 raise。"""
    if _is_port_free(port):
        return
    pids = _list_listening_pids(port)
    if not pids:
        raise RuntimeError(f"端口 {port} 被占用，但无法定位监听 PID（ss 可能不可用）")

    safe_to_kill, untouchable = [], []
    for pid in pids:
        (safe_to_kill if _is_vllm_process(pid) else untouchable).append(pid)

    if untouchable:
        raise RuntimeError(
            f"端口 {port} 被非 vLLM 进程占用 PID={untouchable}，拒绝 kill。"
            "请人工排查后重试。"
        )

    log.warning("  端口 %d 被残留 vLLM 进程占用 PID=%s，主动清理 ...", port, safe_to_kill)
    for pid in safe_to_kill:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pid in safe_to_kill:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    deadline = time.time() + 30
    while time.time() < deadline:
        if _is_port_free(port):
            log.info("  端口 %d 已释放", port)
            return
        time.sleep(1)
    raise RuntimeError(f"清理后端口 {port} 仍未释放")


# ---------------------------------------------------------------------------
# vLLM 进程生命周期
# ---------------------------------------------------------------------------


# 当前 auto_eval 进程持有的活跃 vLLM 子进程，供信号 handler / atexit 清理
_active_vllm_procs: list[subprocess.Popen] = []


# Linux prctl(PR_SET_PDEATHSIG, ...): 父进程死时 kernel 自动发信号给子进程
_PR_SET_PDEATHSIG = 1


def _preexec_setsid_and_pdeathsig() -> None:
    """子进程 fork 后、exec 前调用：自成进程组 + 父死随死（Linux only）。"""
    os.setsid()
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        except Exception:
            pass


def _launch_vllm(
    model_cfg: dict, gpus: list[int], port: int, date_str: str
) -> subprocess.Popen:
    """启动 vLLM serve；返回 Popen 对象。

    保护机制:
      1. 启动前确保端口空闲（残留 vLLM 自动清理，非 vLLM 占用直接 raise）
      2. 子进程绑定 PR_SET_PDEATHSIG=SIGKILL，父进程被 kill -9 也不会留孤儿（Linux）
      3. 注册到 _active_vllm_procs，让信号 handler / atexit 能找到它
    """
    tp = model_cfg.get("tp", 1)
    dp = model_cfg.get("dp", 1)
    needed = tp * dp
    if len(gpus) < needed:
        raise RuntimeError(f"需要 {needed} 张 GPU (TP={tp} × DP={dp})，但只有 {len(gpus)} 张可用: {gpus}")
    use_gpus = gpus[:needed]

    _free_port_or_raise(port)

    cmd = [
        VLLM_BIN, "serve", model_cfg["model_path"],
        "--dtype", model_cfg.get("dtype", "bfloat16"),
        "--enable-auto-tool-choice",
        "--tool-call-parser", model_cfg.get("tool_call_parser", "qwen3_xml"),
        "--tensor-parallel-size", str(tp),
        "--data-parallel-size", str(dp),
        "--port", str(port),
        "--max-model-len", str(model_cfg.get("max_model_len", 98304)),
        "--enable-prefix-caching",
    ]
    sampling_params = model_cfg.get("sampling_params") or {}
    if sampling_params:
        cmd += [
            "--override-generation-config",
            json.dumps(sampling_params, ensure_ascii=False),
        ]
    cmd += list(model_cfg.get("extra_args", []))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in use_gpus)
    env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", str(HEALTH_TIMEOUT))
    log.info("  CUDA_VISIBLE_DEVICES=%s", env["CUDA_VISIBLE_DEVICES"])
    log.info("  %s", " ".join(cmd))

    vllm_log_dir = os.path.join(LOG_BASE, date_str, "vllm_logs")
    os.makedirs(vllm_log_dir, exist_ok=True)
    log_file = os.path.join(vllm_log_dir, f"{model_cfg['key']}_{datetime.now().strftime('%H%M%S')}.log")
    fout = open(log_file, "w")

    proc = subprocess.Popen(
        cmd, env=env, stdout=fout, stderr=subprocess.STDOUT,
        preexec_fn=_preexec_setsid_and_pdeathsig,
    )
    _active_vllm_procs.append(proc)
    log.info("  PID: %d, 日志: %s", proc.pid, log_file)
    return proc


def _kill_vllm(proc: subprocess.Popen) -> None:
    """优雅终止 vLLM 进程组（SIGTERM → 30s 后 SIGKILL）。"""
    if proc in _active_vllm_procs:
        _active_vllm_procs.remove(proc)
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    log.info("  终止 vLLM 进程组 (PGID=%d) ...", pgid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log.info("  SIGTERM 超时，发送 SIGKILL ...")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    log.info("  vLLM 已停止")


def _kill_active_vllms() -> None:
    """杀掉所有当前持有的 vLLM 子进程组（不退出主进程）。"""
    for proc in list(_active_vllm_procs):
        if proc.poll() is not None:
            continue
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            continue
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _active_vllm_procs.clear()


def _signal_handler(signum, _frame) -> None:
    """收到 SIGTERM/SIGINT/SIGHUP：清理子进程后用 128+signum 退出。"""
    try:
        log.warning("收到信号 %d，清理 vLLM 子进程后退出", signum)
    except Exception:
        pass
    _kill_active_vllms()
    os._exit(128 + signum)


def _install_signal_handlers() -> None:
    """SIGTERM / SIGINT / SIGHUP 触发清理；正常退出走 atexit；SIGKILL 则靠 PR_SET_PDEATHSIG。"""
    atexit.register(_kill_active_vllms)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass


# ---------------------------------------------------------------------------
# 调用 batch_run.py
# ---------------------------------------------------------------------------


def _run_batch(
    model_cfg: dict, port: int, batch_args: list[str], date_str: str
) -> int:
    """调用 batch_run.py 跑评测，返回 exit code。"""
    key = model_cfg["key"]
    model_id = model_cfg.get("model_id", model_cfg["model_path"])

    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "batch_run.py"), "--models", key, *batch_args]

    env = os.environ.copy()
    if "AUTO_EVAL_BASE_URL" not in os.environ:
        env["AUTO_EVAL_BASE_URL"] = f"http://{HOST_IP}:{port}/v1"
    env["AUTO_EVAL_MODEL_ID"] = model_id
    env["AUTO_EVAL_MODEL_KEY"] = key
    env["RESULT_DATE_PREFIX"] = date_str

    sampling = model_cfg.get("sampling_params") or {}
    if sampling:
        gen_kwargs = sampling_params_to_generate_kwargs(sampling)
        env["AUTO_EVAL_GENERATE_KWARGS"] = json.dumps(gen_kwargs, ensure_ascii=False)

    log.info("  命令: %s", " ".join(cmd))
    log.info("  AUTO_EVAL_BASE_URL=%s", env["AUTO_EVAL_BASE_URL"])
    log.info("  AUTO_EVAL_MODEL_ID=%s", model_id)
    if "AUTO_EVAL_GENERATE_KWARGS" in env:
        log.info("  AUTO_EVAL_GENERATE_KWARGS=%s", env["AUTO_EVAL_GENERATE_KWARGS"])
    return subprocess.run(cmd, env=env).returncode


# ---------------------------------------------------------------------------
# 流水线主循环
# ---------------------------------------------------------------------------


def _build_batch_args(args: argparse.Namespace) -> list[str]:
    """把 CLI 参数翻译成传给 batch_run.py 的 argv。"""
    extra: list[str] = []
    if args.tasks:
        extra += args.tasks
    if args.package:
        extra += ["--package", *args.package]
    if args.parallel is not None:
        extra += ["--parallel", str(args.parallel)]
    return extra


def _collect_trial_summary(
    result_day_dir: str, existing_dirs: set[str], base_key: str, t: int,
    trial_data: dict[str, list[dict]],
) -> None:
    """vLLM 运行完一次后，找出新生成的 batch 目录读 _batch_summary.json。"""
    try:
        new_dirs = (
            set(os.listdir(result_day_dir)) - existing_dirs
            if os.path.isdir(result_day_dir)
            else set()
        )
        if not new_dirs:
            return
        latest_dir = sorted(new_dirs)[-1]
        summary_path = os.path.join(result_day_dir, latest_dir, "_batch_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                trial_data.setdefault(base_key, []).append(json.load(f))
            log.info("  Trial %d 结果已收集", t)
    except Exception as e:
        log.warning("  读取 trial %d 结果失败: %s", t, e)


def _run_one_model(
    model_cfg: dict,
    gpus: list[int],
    port: int,
    batch_extra: list[str],
    date_str: str,
    results: dict[str, str],
    trial_data: dict[str, list[dict]],
) -> None:
    """跑一个模型：启动 vLLM → trial 循环 → kill vLLM。"""
    key = model_cfg["key"]
    trial_count = model_cfg.get("inference_trials", model_cfg.get("trial", 1))
    log.info("  路径: %s", model_cfg["model_path"])
    log.info("-" * 60)

    log.info("[1/3] 启动 vLLM ...")
    try:
        proc = _launch_vllm(model_cfg, gpus, port, date_str)
    except RuntimeError as e:
        log.error("  %s", e)
        results[key] = f"SKIP ({e})"
        return

    log.info("[2/3] 等待 vLLM 就绪 ...")
    expected_id = model_cfg.get("model_id", model_cfg["model_path"])
    if not _wait_for_health(port, expected_id):
        log.error("  vLLM 启动超时或 model_id 不匹配，跳过该模型")
        _kill_vllm(proc)
        results[key] = "SKIP (启动超时)"
        return

    log.info("  vLLM 就绪 (model=%s)!", expected_id)
    log.info("[3/3] 运行 benchmark ...")

    result_day_dir = os.path.join(RESULT_BASE, date_str)
    os.makedirs(result_day_dir, exist_ok=True)

    for t in range(1, trial_count + 1):
        trial_key = f"{key}_t{t}" if trial_count > 1 else key
        if trial_count > 1:
            log.info("  === Trial %d/%d (key=%s) ===", t, trial_count, trial_key)

        trial_cfg = dict(model_cfg, key=trial_key)
        existing = set(os.listdir(result_day_dir)) if os.path.isdir(result_day_dir) else set()

        exit_code = _run_batch(trial_cfg, port, batch_extra, date_str)
        results[trial_key] = "OK" if exit_code == 0 else f"FAIL (exit={exit_code})"

        if trial_count > 1:
            _collect_trial_summary(result_day_dir, existing, key, t, trial_data)

    log.info("  评测完成，关闭 vLLM ...")
    _kill_vllm(proc)
    time.sleep(5)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="自动化 vLLM 部署 + benchmark 评测流水线")
    p.add_argument("--config", required=True, help="模型配置 JSON 文件路径")
    p.add_argument(
        "--gpus", type=str, default=None,
        help="指定 GPU 编号，逗号分隔（如 2,3,4,5）。不指定则自动检测空闲 GPU",
    )
    p.add_argument(
        "--port", type=int, default=None,
        help=f"vLLM 服务端口；不指定则在 {DEFAULT_PORT_RANGE[0]}-{DEFAULT_PORT_RANGE[1]} "
             "范围内自动挑第一个空闲端口（方便同机并发跑多个 auto_eval）",
    )
    p.add_argument("--package", nargs="+", metavar="PKG", help="传给 batch_run.py 的 --package")
    p.add_argument("--tasks", nargs="+", metavar="TASK", help="传给 batch_run.py 的任务列表")
    p.add_argument("-p", "--parallel", type=int, default=None, help="传给 batch_run.py 的并发数")
    return p.parse_args()


def _resolve_gpus(args: argparse.Namespace) -> list[int] | None:
    if args.gpus:
        return [int(g) for g in args.gpus.split(",")]
    free = _find_free_gpus()
    if not free:
        log.error("没有检测到空闲 GPU")
        return None
    return free


def _print_overall_summary(
    results: dict[str, str],
    trial_data: dict[str, list[dict]],
    date_str: str,
) -> None:
    log.info("=" * 60)
    log.info("全部评测完成！汇总:")
    log.info("=" * 60)
    for key, status in results.items():
        log.info("  %-30s %s", key, status)

    for base_key, summaries in trial_data.items():
        print_trial_summary(
            base_key, summaries,
            result_root=os.path.join(RESULT_BASE, date_str),
            date_str=date_str,
        )


def main() -> None:
    args = _parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        models: list[dict] = json.load(f)
    if not models:
        log.error("配置文件中没有模型")
        return

    date_str = _today()
    start_ts = datetime.now().strftime("%H%M%S")
    log_file = _setup_logging(date_str, start_ts)
    log.info("日志文件: %s", log_file)

    _install_signal_handlers()

    gpus = _resolve_gpus(args)
    if gpus is None:
        return

    if args.port is None:
        args.port = _pick_free_port(*DEFAULT_PORT_RANGE)
        log.info("自动挑选 vLLM 端口: %d", args.port)
    else:
        log.info("使用指定 vLLM 端口: %d", args.port)

    log.info("可用 GPU: %s", gpus)
    log.info("待评测模型: %d 个", len(models))
    log.info("=" * 60)

    batch_extra = _build_batch_args(args)
    results: dict[str, str] = {}
    trial_data: dict[str, list[dict]] = {}

    for i, model_cfg in enumerate(models, 1):
        log.info(
            "[%d/%d] 模型: %s (inference_trials=%d)",
            i, len(models), model_cfg["key"],
            model_cfg.get("inference_trials", model_cfg.get("trial", 1)),
        )
        _run_one_model(model_cfg, gpus, args.port, batch_extra, date_str, results, trial_data)

    _print_overall_summary(results, trial_data, date_str)


if __name__ == "__main__":
    main()
