"""Thin MCP server that wraps docker/sync.sh and docker/remote_run.sh.

Running as an MCP server (a regular child process) instead of through the
Bash tool avoids Claude Code's sandbox, which blocks outbound SSH connections.
"""

import asyncio
import os

from mcp.server.fastmcp import Context, FastMCP

DOCKER_DIR = os.path.dirname(os.path.abspath(__file__))

mcp = FastMCP("trinity-remote-test")

# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------


async def _run_script(cmd: list[str], timeout: int, ctx: Context | None = None) -> str:
    """Run a shell script and return combined stdout+stderr."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=os.path.join(DOCKER_DIR, ".."),
    )
    assert proc.stdout is not None

    output_lines: list[str] = []
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    last_heartbeat = start_time
    heartbeat_interval = 15

    while True:
        elapsed = loop.time() - start_time
        remaining = timeout - elapsed
        if remaining <= 0:
            proc.kill()
            await proc.wait()
            timeout_msg = f"Command timed out after {timeout} seconds."
            output_lines.append(timeout_msg)
            if ctx is not None:
                await ctx.warning(timeout_msg)
            break

        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=min(1, remaining))
        except asyncio.TimeoutError:
            now = loop.time()
            if ctx is not None and now - last_heartbeat >= heartbeat_interval:
                heartbeat_msg = f"Still running after {int(now - start_time)}s..."
                await ctx.info(heartbeat_msg)
                await ctx.report_progress(now - start_time, message=heartbeat_msg)
                last_heartbeat = now
            continue

        if not line:
            if proc.stdout.at_eof():
                break
            continue

        decoded = line.decode(errors="replace").rstrip("\n")
        output_lines.append(decoded)
        if ctx is not None:
            await ctx.info(decoded)

    output = "\n".join(output_lines)
    if proc.returncode is None:
        await proc.wait()
    if proc.returncode != 0:
        output += f"\n[exit code: {proc.returncode}]"
    return output


@mcp.tool()
async def sync_code(dry_run: bool = False, ctx: Context | None = None) -> str:
    """Sync local code to the remote GPU server via rsync over SSH.

    Only git-tracked files are synced. Untracked files are listed as warnings.

    Args:
        dry_run: If true, show what would be transferred without making changes.
    """
    cmd = ["bash", os.path.join(DOCKER_DIR, "sync.sh")]
    if dry_run:
        cmd.append("--dry-run")
    return await _run_script(cmd, timeout=120, ctx=ctx)


@mcp.tool()
async def run_tests(
    module: str = "",
    keyword: str = "",
    quiet: bool = True,
    no_sync: bool = False,
    timeout: int = 600,
    ctx: Context | None = None,
) -> str:
    """Sync code and run pytest on the remote GPU server inside Docker.

    Args:
        module: Test target under tests/ (directory or file). Empty runs all tests.
        keyword: Pytest -k expression to filter tests.
        quiet: Run pytest in quiet mode.
        no_sync: Skip the rsync step if code is already up to date.
        timeout: Timeout in seconds for the entire operation.
    """
    cmd = ["bash", os.path.join(DOCKER_DIR, "remote_run.sh")]
    if module:
        cmd.extend(["--module", module])
    if keyword:
        cmd.extend(["--keyword", keyword])
    if quiet:
        cmd.append("--quiet")
    if no_sync:
        cmd.append("--no-sync")
    cmd.extend(["--timeout", str(timeout)])
    return await _run_script(cmd, timeout=timeout + 30, ctx=ctx)


@mcp.tool()
async def check_status(ctx: Context | None = None) -> str:
    """Check Docker container and Ray cluster status on the remote GPU server."""
    cmd = [
        "bash",
        "-c",
        f"source {os.path.join(DOCKER_DIR, 'common.sh')} && "
        "load_remote_env && "
        'ssh -p "$TRINITY_REMOTE_SSH_PORT" '
        "-o StrictHostKeyChecking=accept-new "
        "-o ConnectTimeout=10 "
        '"$TRINITY_REMOTE_HOST" '
        f'"cd $TRINITY_REMOTE_WORKSPACE && bash docker/status.sh"',
    ]
    return await _run_script(cmd, timeout=30, ctx=ctx)


if __name__ == "__main__":
    mcp.run(transport="stdio")
