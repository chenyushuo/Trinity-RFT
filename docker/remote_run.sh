#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
DEFAULT_TIMEOUT=600

print_help() {
    cat <<EOF
Usage: bash docker/remote_run.sh [options]

Sync code to the remote server and run pytest inside its Docker container.

Options:
  -m, --module <path>    Test target under tests/ (directory or file)
  -k, --keyword <expr>   Pytest -k expression used to filter tests
  -q, --quiet            Run pytest in quiet mode
  -t, --timeout <secs>   SSH command timeout in seconds. Default: ${DEFAULT_TIMEOUT}
  --no-sync              Skip the rsync step (code is already up to date)
  -h, --help             Show this help message and exit

Examples:
  bash docker/remote_run.sh --module common
  bash docker/remote_run.sh --module trainer/trainer_test
  bash docker/remote_run.sh --module common --keyword test_config --quiet
  bash docker/remote_run.sh --module explorer --no-sync --timeout 300
EOF
}

fail() {
    echo "$1" >&2
    exit 1
}

require_arg() {
    local option="$1"
    local value="$2"

    if [[ -z "$value" ]]; then
        fail "Missing value for ${option}. Use --help for usage information."
    fi
}

module_name=""
keyword_expr=""
quiet_mode=false
timeout_secs="$DEFAULT_TIMEOUT"
skip_sync=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--module)
            require_arg "$1" "$2"
            module_name="$2"
            shift 2
            ;;
        -k|--keyword)
            require_arg "$1" "$2"
            keyword_expr="$2"
            shift 2
            ;;
        -q|--quiet)
            quiet_mode=true
            shift
            ;;
        -t|--timeout)
            require_arg "$1" "$2"
            timeout_secs="$2"
            shift 2
            ;;
        --no-sync)
            skip_sync=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            fail "Unknown option: $1. Use --help for usage information."
            ;;
        *)
            fail "Unexpected positional argument: $1. Use --help for usage information."
            ;;
    esac
done

if ! load_remote_env; then
    exit 1
fi

ssh_port="${TRINITY_REMOTE_SSH_PORT}"

if [[ "$skip_sync" == false ]]; then
    echo "=== Syncing code to remote ==="
    if ! bash "$SCRIPT_DIR/sync.sh"; then
        fail "Code sync failed. Aborting."
    fi
    echo ""
fi

run_cmd="cd ${TRINITY_REMOTE_WORKSPACE} && bash docker/run.sh"
if [[ -n "$module_name" ]]; then
    run_cmd="${run_cmd} --module ${module_name}"
fi
if [[ -n "$keyword_expr" ]]; then
    run_cmd="${run_cmd} --keyword $(printf '%q' "$keyword_expr")"
fi
if [[ "$quiet_mode" == true ]]; then
    run_cmd="${run_cmd} --quiet"
fi

echo "=== Running tests on remote ==="
echo "Host: ${TRINITY_REMOTE_HOST}"
echo "Command: ${run_cmd}"
echo "Timeout: ${timeout_secs}s"
echo ""

timeout_cmd=""
if command -v timeout >/dev/null 2>&1; then
    timeout_cmd="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
    timeout_cmd="gtimeout"
fi

if [[ -n "$timeout_cmd" ]]; then
    "$timeout_cmd" "$timeout_secs" \
        ssh -p "$ssh_port" \
            -o StrictHostKeyChecking=accept-new \
            -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=3 \
            "$TRINITY_REMOTE_HOST" \
            "$run_cmd"
    exit_code=$?

    if [[ $exit_code -eq 124 ]]; then
        fail "Remote test timed out after ${timeout_secs} seconds."
    fi
else
    ssh -p "$ssh_port" \
        -o StrictHostKeyChecking=accept-new \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o "ConnectTimeout=${timeout_secs}" \
        "$TRINITY_REMOTE_HOST" \
        "$run_cmd"
    exit_code=$?
fi

exit $exit_code
