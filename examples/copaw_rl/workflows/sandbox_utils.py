import argparse
import hashlib
import json
import os
import pickle
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import httpx
import numpy as np
from e2b import (
    CommandExitException,
    NotFoundException,
    Sandbox,
    SandboxException,
    TimeoutException,
)


# ANSI color codes
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_sandbox_info(sandbox_id, token, domain, logger):
    """Get sandbox info via API to retrieve dashboard URL"""
    try:
        url = f"https://api.{domain}/sandboxes/{sandbox_id}"
        headers = {"X-API-KEY": token, "X-Generate-Dashboard-Url": "true"}

        resp = httpx.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            data = resp.json()

            dashboard_url = None

            if "metadata" in data:
                metadata = data["metadata"]
                if isinstance(metadata, dict):
                    if "sandbox.alicloud.com/dashboard-url" in metadata:
                        dashboard_url = metadata["sandbox.alicloud.com/dashboard-url"]
                    elif "dashboard_url" in metadata:
                        dashboard_url = metadata["dashboard_url"]

            if not dashboard_url and "dashboard_url" in data:
                dashboard_url = data["dashboard_url"]

            if dashboard_url:
                dashboard_url = dashboard_url.replace(".vpc.", ".")
                logger.info(f"\n{Colors.HEADER}{Colors.BOLD}🌐 Dashboard URL:{Colors.ENDC}")
                logger.info(f"{Colors.OKGREEN}{Colors.UNDERLINE}{dashboard_url}{Colors.ENDC}\n")
            else:
                logger.warning(f"{Colors.WARNING}Dashboard URL not found in metadata{Colors.ENDC}")

            return data
        else:
            logger.error(
                f"{Colors.FAIL}Failed to get sandbox info: {resp.status_code}{Colors.ENDC}"
            )
            return None

    except Exception as e:
        logger.error(f"{Colors.WARNING}Could not fetch dashboard URL: {e}{Colors.ENDC}")
        return None


def connect_sandbox(sandbox_id, token, domain, logger):
    """Connect to existing sandbox"""
    if domain:
        os.environ["E2B_DOMAIN"] = domain
    os.environ["E2B_API_KEY"] = token

    sandbox = Sandbox.connect(sandbox_id)

    logger.info(f"    {Colors.OKGREEN}✓ Connected to sandbox{Colors.ENDC}")

    return sandbox


def create_sandbox(token, domain, template, logger) -> Sandbox:
    """Create sandbox with authorization header"""
    if domain:
        os.environ["E2B_DOMAIN"] = domain
    os.environ["E2B_API_KEY"] = token

    sandbox = Sandbox.create(
        template=template,
        timeout=3600,  # 1 hour ; TODO: make this configurable
        headers={
            "template": template,
        },
    )

    logger.info(
        f"    {Colors.OKGREEN}✓ Sandbox created{Colors.ENDC} (ID: {Colors.BOLD}{sandbox.sandbox_id}{Colors.ENDC})"
    )

    return sandbox


def update_sandbox_files(sandbox: Sandbox, template, logger):
    """递归同步 utils/ 下所有 .py 和 .sh 文件到 /root/，保留目录结构。

    - 顶层模块（如 bench_client.py）→ /root/bench_client.py
    - 包目录（如 copaw_eval/__init__.py）→ /root/copaw_eval/__init__.py
    - md5_maps.json 中的 key 用 POSIX 相对路径（如 "copaw_eval/__init__.py"）。
    - 跳过 __pycache__ 等运行时产物。
    """
    utils_dir = Path(__file__).parent.parent / "utils"
    local_md5_map = {}
    for file in utils_dir.rglob("*"):
        if not file.is_file() or file.suffix not in {".py", ".sh"}:
            continue
        rel_parts = file.relative_to(utils_dir).parts
        if any(p.startswith((".", "__pycache__")) for p in rel_parts):
            continue
        rel_path = file.relative_to(utils_dir).as_posix()
        with open(file, "rb") as f:
            file_md5 = hashlib.file_digest(f, "md5").hexdigest()
        local_md5_map[rel_path] = file_md5

    # get md5 map in sandbox
    files = " ".join(local_md5_map.keys())
    try:
        result = sandbox.commands.run(f"md5sum {files}")
        stdout = result.stdout.strip()
    except CommandExitException as e:
        stdout = e.stdout.strip()
    md5_map = {}
    for line in stdout.splitlines():
        line = line.strip()
        if "No such file or directory" in line:
            rel_path = line.split(": ")[1]
            md5_map[rel_path] = None
        else:
            md5, rel_path = line.split("  ")
            md5_map[rel_path] = md5

    # upload files that are missing or different
    for rel_path, md5 in local_md5_map.items():
        if md5_map.get(rel_path, None) != md5:
            logger.info(f"Uploading {rel_path} to sandbox")
            with open(utils_dir / rel_path, "r") as f:
                sandbox.files.write(f"/root/{rel_path}", f)


def get_or_create_sandbox(sandbox_id, token, domain, template, logger) -> Tuple[Sandbox, bool]:
    """Get existing sandbox or create new one"""
    if sandbox_id:
        logger.info(
            f"\n{Colors.OKCYAN}[1] Connecting to existing sandbox:{Colors.ENDC} {Colors.BOLD}{sandbox_id}{Colors.ENDC}"
        )
        sandbox = connect_sandbox(sandbox_id, token, domain, logger)
        get_sandbox_info(sandbox_id, token, domain, logger)
        update_sandbox_files(sandbox, template, logger)
        return sandbox, False
    else:
        logger.info(
            f"\n{Colors.OKCYAN}[1] Creating sandbox with template:{Colors.ENDC} {Colors.BOLD}{template}{Colors.ENDC}"
        )
        sandbox = create_sandbox(token, domain, template, logger)
        logger.info(f"\n{Colors.OKCYAN}[2] Waiting for sandbox to be ready...{Colors.ENDC}")
        max_attempts = 60
        for attempt in range(1, max_attempts + 1):
            try:
                is_running = sandbox.is_running()
                status_str = (
                    f"{Colors.OKGREEN}Running{Colors.ENDC}"
                    if is_running
                    else f"{Colors.WARNING}Not Running{Colors.ENDC}"
                )
                logger.info(f"    [{attempt}/{max_attempts}] Sandbox status: {status_str}")
                if is_running:
                    logger.info(f"    {Colors.OKGREEN}✓ Sandbox is now running!{Colors.ENDC}")
                    get_sandbox_info(sandbox.sandbox_id, token, domain, logger)
                    update_sandbox_files(sandbox, template, logger)
                    return sandbox, True
            except Exception as e:
                logger.error(
                    f"    [{attempt}/{max_attempts}] {Colors.WARNING}Failed to check status:{Colors.ENDC} {e}"
                )

            if attempt < max_attempts:
                time.sleep(2)

        logger.warning(
            f"    {Colors.WARNING}Warning: Sandbox did not reach Running state within timeout{Colors.ENDC}"
        )
        get_sandbox_info(sandbox.sandbox_id, token, domain, logger)
        update_sandbox_files(sandbox, template, logger)
        return sandbox, True


def run_with_reconnect(sandbox: Sandbox, cmd, envs, logger, max_retries=5):
    # 1. 后台启动命令
    handle = sandbox.commands.run(
        cmd,
        background=True,
        envs=envs,
        timeout=1440,  # about 24 min; was 3600
        request_timeout=1800,
    )
    pid = handle.pid

    for attempt in range(max_retries):
        try:
            # 2. 等待命令完成（此处维持 streaming 接收输出）
            result = handle.wait(
                on_stdout=lambda data: logger.info(f"[stdout]: {data.rstrip()}"),
                on_stderr=lambda data: logger.info(f"[stderr]: {data.rstrip()}"),
            )
            return result
        except TimeoutException as e:
            logger.warning(f"连接断开 (attempt {attempt + 1}): {e}")
            time.sleep(3)
            if attempt != max_retries - 1:
                # 3. 重连到 sandbox 和进程
                handle = sandbox.commands.connect(
                    pid,
                    timeout=1440,
                    request_timeout=1800,
                )
            else:
                raise e
    return None


def launch_run_py(
    sandbox: Sandbox, cmd: str, oss_config, dashscope_api_key, logger, envs={}, raise_error=False
):
    assert dashscope_api_key, "DASHSCOPE_API_KEY is required to run the workflow"
    dashscope_api_keys = dashscope_api_key.split(",")
    dashscope_api_key = np.random.choice(dashscope_api_keys).item()
    t0 = time.perf_counter()
    envs.update(
        {
            "OSS_ACCESS_KEY_ID": oss_config["access_key_id"],
            "OSS_ACCESS_KEY_SECRET": oss_config["access_key_secret"],
            "OSS_REGION": oss_config["region"],
            "OSS_ENDPOINT": oss_config["endpoint"],
            "OSS_BUCKET_NAME": oss_config["bucket_name"],
            "DASHSCOPE_API_KEY": dashscope_api_key,
        }
    )
    # auto_eval.py 注入的每请求 sampling kwargs（JSON 字符串），透传给沙箱里的 run.py
    gen_kwargs = os.environ.get("AUTO_EVAL_GENERATE_KWARGS")
    if gen_kwargs:
        envs["AUTO_EVAL_GENERATE_KWARGS"] = gen_kwargs
    try:
        logger.info(f"Running command in sandbox: {cmd}")
        logger.info(f"Sandbox envs: {envs}")
        result = run_with_reconnect(sandbox, cmd, envs, logger)
        run_outputs = result.stdout + "\n" + result.stderr
    except CommandExitException as e:
        logger.info("run.py exited with non-zero exit code: %s", e.exit_code)
        logger.info("Error stdout: %s", e.stdout.strip())
        logger.info("Error stderr: %s", e.stderr.strip())
        run_outputs = e.stdout + "\n" + e.stderr
        if raise_error:
            raise e
    latency_seconds = time.perf_counter() - t0
    return latency_seconds, run_outputs


def _download_file(sandbox: Sandbox, remote_path: str, logger, format: Optional[str] = None):
    for attempt in range(30):
        try:
            content = sandbox.files.read(remote_path, format=format)
            return content
        except SandboxException as e:
            logger.warning(f"Attempt {attempt + 1}/30: {remote_path} not ready, retrying... ({e})")
            time.sleep(2)
    raise FileNotFoundError(f"{remote_path} not found in sandbox after multiple attempts")


def _setup_otel_envs(otel_config: dict = {}):
    key_map = {
        "endpoint": "OTEL_EXPORTER_OTLP_ENDPOINT",
        "service_name": "OTEL_SERVICE_NAME",
        "arms_license_key": "OTEL_ARMS_LICENSE_KEY",
        "arms_project": "OTEL_ARMS_PROJECT",
        "cms_workspace": "OTEL_CMS_WORKSPACE",
    }
    envs = {}
    for k, v in key_map.items():
        if v in os.environ:
            envs[v] = os.environ[v]
        if otel_config.get(k, None) is not None:
            envs[v] = str(otel_config[k])
    return envs


def _str_to_bool(s):
    if isinstance(s, str):
        s = s.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
    if isinstance(s, bool):
        return s
    raise ValueError(f"无法转换为布尔值: {s!r}")


def run_workflow(
    sandbox: Sandbox,
    task_id,
    oss_config,
    otel_config,
    dashscope_api_key,
    api_server_url,
    model_path,
    logger,
):
    cmd = (
        f"python run.py --task-id {task_id} --oss-prefix {oss_config['prefix']} "
        f"--provider-base-url {api_server_url} --provider-model-id {model_path} "
        f"--agent-timeout-seconds 1200"
    )
    envs = {key: value for key, value in os.environ.items() if key.startswith("JUDGE_")}
    enable_otel = _str_to_bool(otel_config.pop("enable", False))
    if enable_otel:
        logger.info("Restarting qwenpaw app with LOONGSUITE_PYTHON_SITE_BOOTSTRAP=True")
        sandbox.commands.run("pkill -f '[q]wenpaw app' || true", timeout=30)
        qwenpaw_envs = dict(envs)
        qwenpaw_envs["LOONGSUITE_PYTHON_SITE_BOOTSTRAP"] = "True"
        result = sandbox.commands.run(
            "qwenpaw app |& tee /app/qwenpaw-app.log",
            background=True,
            envs=qwenpaw_envs,
        )
        for stdout, stderr, _ in result:
            if stdout:
                logger.debug(f"[qwenpaw app stdout]: {stdout.strip()}")
                if "http://127.0.0.1:8088" in stdout:
                    logger.info("qwenpaw app restarted with pid %d", result.pid)
                    break
            if stderr:
                logger.debug(f"[qwenpaw app stderr]: {stderr.strip()}")
        # 注入 OTEL 相关环境变量
        cmd += " --enable-otel"
        envs.update(_setup_otel_envs(otel_config))
    launch_duration_seconds, _ = launch_run_py(
        sandbox, cmd, oss_config, dashscope_api_key, logger, envs=envs, raise_error=True
    )
    logger.info(f"Launch duration: {launch_duration_seconds}")

    output = _download_file(sandbox, "/root/output.pkl", logger, format="bytes")
    output = pickle.loads(output)
    output["launch_duration_seconds"] = launch_duration_seconds
    return output


def _save_and_extract_zip(sandbox: Sandbox, remote_path: str, task_dir: str, logger):
    """Download a zip from the sandbox, extract it, and delete the zip."""
    filename = os.path.basename(remote_path)
    zip_path = os.path.join(task_dir, filename)
    try:
        data = sandbox.files.read(remote_path, format="bytes")
        with open(zip_path, "wb") as f:
            f.write(data)
        extract_dir = os.path.join(task_dir, filename.replace(".zip", ""))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        os.remove(zip_path)
        logger.info("Extracted %s -> %s", filename, extract_dir)
    except NotFoundException as e:
        logger.warning(f"{Colors.WARNING}{filename} not found: {e}{Colors.ENDC}")
    except zipfile.BadZipFile as e:
        logger.warning(f"{Colors.WARNING}{filename} is not a valid zip: {e}{Colors.ENDC}")


def run_eval_workflow(
    sandbox: Sandbox,
    task_id: str,
    oss_config: dict,
    dashscope_api_key: str,
    api_server_url: str,
    model_path: str,
    model_label: str,
    checkpoint_job_dir: str,
    logger,
):
    cmd = (
        f"python run.py --task-id {task_id} --oss-prefix {oss_config['prefix']} "
        f"--provider-base-url {api_server_url} --provider-model-id {model_path} --evaluation"
    )
    latency_seconds, _ = launch_run_py(sandbox, cmd, oss_config, dashscope_api_key, logger)

    if model_label:
        task_dir = os.path.join(checkpoint_job_dir, model_label, task_id)
    else:
        task_dir = os.path.join(checkpoint_job_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    logger.info("Saving summary to %s", task_dir)

    summary = sandbox.files.read("/root/summary.json")
    with open(os.path.join(task_dir, "summary.json"), "w") as f:
        f.write(summary)
    summary_data = json.loads(summary)
    logger.info("summary.json content: %s", summary_data)

    session = sandbox.files.read("/root/session.json")
    with open(os.path.join(task_dir, "session.json"), "w") as f:
        f.write(session)

    _save_and_extract_zip(sandbox, "/root/screenshots.zip", task_dir, logger)
    _save_and_extract_zip(sandbox, "/root/workspace_files.zip", task_dir, logger)
    _save_and_extract_zip(sandbox, "/root/qwenpaw_log.zip", task_dir, logger)

    score = summary_data.get("summary", {}).get("avg_score", -1) if summary_data else -1
    status = "PASS" if score == 100 else "FAIL" if score >= 0 else "ERROR"
    steps = -1
    duration_seconds = -1.0
    response_length = -1
    if summary_data and summary_data.get("tasks"):
        task_steps = [t.get("steps", -1) for t in summary_data["tasks"] if t.get("steps", -1) >= 0]
        if task_steps:
            steps = sum(task_steps)
        durs = [
            t.get("duration_seconds")
            for t in summary_data["tasks"]
            if t.get("duration_seconds") is not None
        ]
        if durs:
            duration_seconds = sum(durs)
        else:
            duration_seconds = latency_seconds
        total_chars = 0
        for t in summary_data["tasks"]:
            for step in t.get("trajectory") or []:
                total_chars += len((step.get("thought") or ""))
            total_chars += len((t.get("final_text") or ""))
        response_length = total_chars
    if duration_seconds < 0:
        duration_seconds = latency_seconds
    tag = f"[{model_label or model_path}] {task_id}" if (model_label or model_path) else task_id
    print(
        f"[{status}] {tag} — score: {score}, steps: {steps}, 输出(计费): {response_length}, time: {duration_seconds:.1f}s"
    )
    return {
        "task": task_id,
        "model": model_label or model_path,
        "score": score,
        "status": status,
        "steps": steps,
        "response_length": response_length,
        "latency_seconds": round(latency_seconds, 2),
        "duration_seconds": round(duration_seconds, 2) if duration_seconds >= 0 else -1,
    }


def run_teacher_workflow(
    sandbox: Sandbox, task_id, oss_config, dashscope_api_key, model_id, logger
):
    cmd = (
        f"python run.py --task-id {task_id} --oss-prefix {oss_config['prefix']} "
        f"--provider-name dashscope --provider-model-id {model_id} "
        f"--provider-api-key {dashscope_api_key} --evaluation"
    )
    latency_seconds, run_outputs = launch_run_py(
        sandbox, cmd, oss_config, dashscope_api_key, logger
    )

    trajectory_file = sandbox.files.read("/root/tests/traj.json")
    trajectory = json.loads(trajectory_file)
    return trajectory_file, trajectory, run_outputs


def run_teacher_eval_workflow(
    sandbox: Sandbox,
    task_id: str,
    oss_config: dict,
    dashscope_api_key: str,
    model_id: str,
    model_label: str,
    checkpoint_job_dir: str,
    logger,
):
    """Teacher model evaluation: uses DashScope built-in provider instead of
    --provider-base-url, and collects the same summary/session/screenshots
    artifacts as run_eval_workflow."""
    cmd = (
        f"python run.py --task-id {task_id} --oss-prefix {oss_config['prefix']} "
        f"--provider-name dashscope --provider-model-id {model_id} "
        f"--provider-api-key {dashscope_api_key} --evaluation"
    )
    latency_seconds, _ = launch_run_py(sandbox, cmd, oss_config, dashscope_api_key, logger)

    if model_label:
        task_dir = os.path.join(checkpoint_job_dir, model_label, task_id)
    else:
        task_dir = os.path.join(checkpoint_job_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    logger.info("Saving summary to %s", task_dir)

    summary = sandbox.files.read("/root/summary.json")
    with open(os.path.join(task_dir, "summary.json"), "w") as f:
        f.write(summary)
    summary_data = json.loads(summary)
    logger.info("summary.json content: %s", summary_data)

    session = sandbox.files.read("/root/session.json")
    with open(os.path.join(task_dir, "session.json"), "w") as f:
        f.write(session)

    _save_and_extract_zip(sandbox, "/root/screenshots.zip", task_dir, logger)
    _save_and_extract_zip(sandbox, "/root/workspace_files.zip", task_dir, logger)
    _save_and_extract_zip(sandbox, "/root/qwenpaw_log.zip", task_dir, logger)

    score = summary_data.get("summary", {}).get("avg_score", -1) if summary_data else -1
    status = "PASS" if score == 100 else "FAIL" if score >= 0 else "ERROR"
    steps = -1
    duration_seconds = -1.0
    response_length = -1
    if summary_data and summary_data.get("tasks"):
        task_steps = [t.get("steps", -1) for t in summary_data["tasks"] if t.get("steps", -1) >= 0]
        if task_steps:
            steps = sum(task_steps)
        durs = [
            t.get("duration_seconds")
            for t in summary_data["tasks"]
            if t.get("duration_seconds") is not None
        ]
        if durs:
            duration_seconds = sum(durs)
        else:
            duration_seconds = latency_seconds
        total_chars = 0
        for t in summary_data["tasks"]:
            for step in t.get("trajectory") or []:
                total_chars += len((step.get("thought") or ""))
            total_chars += len((t.get("final_text") or ""))
        response_length = total_chars
    if duration_seconds < 0:
        duration_seconds = latency_seconds
    tag = f"[{model_label or model_id}] {task_id}" if (model_label or model_id) else task_id
    print(
        f"[{status}] {tag} — score: {score}, steps: {steps}, 输出(计费): {response_length}, time: {duration_seconds:.1f}s"
    )
    return {
        "task": task_id,
        "model": model_label or model_id,
        "score": score,
        "status": status,
        "steps": steps,
        "response_length": response_length,
        "latency_seconds": round(latency_seconds, 2),
        "duration_seconds": round(duration_seconds, 2) if duration_seconds >= 0 else -1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=os.environ.get("E2B_API_KEY", None))
    parser.add_argument("--domain", type=str, default=os.environ.get("E2B_DOMAIN", None))
    parser.add_argument("--template", type=str, default=os.environ.get("E2B_TEMPLATE", None))
    parser.add_argument(
        "--enable-otel", action="store_true", help="Whether to set up OpenTelemetry in the sandbox"
    )
    args = parser.parse_args()

    from trinity.utils.log import get_logger

    logger = get_logger()
    sandbox, created = get_or_create_sandbox(None, args.token, args.domain, args.template, logger)

    utils_dir = Path(__file__).parent.parent / "utils"

    sandbox.commands.run("mkdir patch")
    patch_dir = utils_dir / "patch"
    for patch_file in patch_dir.glob("*.patch"):
        logger.info(f"Uploading patch {patch_file.name} to sandbox...")
        with open(patch_file, "r") as f:
            sandbox.files.write(f"/root/patch/{patch_file.name}", f)

    if args.enable_otel:
        otel_envs = _setup_otel_envs()
        logger.info(f"OTEL envs for sandbox: {otel_envs}")
    else:
        otel_envs = {}
    try:
        result = sandbox.commands.run(
            "pip uninstall qwenpaw -y && "
            "pip install qwenpaw==v1.1.9 && "
            "pip install oss2 pytest py-openjudge pytest-asyncio && "
            "patch /app/venv/lib/python3.11/site-packages/qwenpaw/agents/react_agent.py < /root/patch/react_agent.patch && "
            "patch /app/venv/lib/python3.11/site-packages/qwenpaw/agents/tools/browser_control.py < /root/patch/browser_control.patch && "
            "patch /app/venv/lib/python3.11/site-packages/agentscope/model/_openai_model.py < /root/patch/openai_model.patch && "
            "patch /app/venv/lib/python3.11/site-packages/agentscope/model/_model_response.py < /root/patch/model_response.patch && "
            "apt-get update && "
            "apt-get install -y xfce4 xfce4-goodies x11vnc openbox xvfb novnc websockify supervisor dbus-x11 && "
            "rm -rf /var/lib/apt/lists/* && "
            "echo '100.118.58.9    copaw-dataset.oss-cn-beijing-internal.aliyuncs.com' >> /etc/hosts && "
            "bash /root/setup_otel.sh && "
            "pip install --no-cache-dir 'wrapt<2' && "
            "pip install opentelemetry-instrumentation-openai && "
            "qwenpaw init --defaults --accept-security",
            envs=otel_envs,
            on_stdout=lambda data: logger.info(f"[stdout]: {data.rstrip()}"),
            on_stderr=lambda data: logger.info(f"[stderr]: {data.rstrip()}"),
            timeout=3600,
        )
    except CommandExitException as e:
        logger.info("Error stdout: %s", e.stdout.strip())
        logger.info("Error stderr: %s", e.stderr.strip())
        raise e

    try:
        result = sandbox.commands.run(
            "qwenpaw app &> /app/qwenpaw-app.log",
            background=True,
        )
        logger.info("qwenpaw app started with pid %d", result.pid)
    except CommandExitException as e:
        logger.info("Error starting qwenpaw app. stdout: %s", e.stdout.strip())
        logger.info("Error starting qwenpaw app. stderr: %s", e.stderr.strip())
        raise e

    try:
        result = sandbox.commands.run(
            "bash /root/start-vnc.sh &> /root/start-vnc.log",
            background=True,
        )
        logger.info("VNC server started with pid %d", result.pid)
    except CommandExitException as e:
        logger.info("Error starting VNC server. stdout: %s", e.stdout.strip())
        logger.info("Error starting VNC server. stderr: %s", e.stderr.strip())
        raise e
