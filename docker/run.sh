#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
SERVICE_NAME="trinity-node-1"

print_help() {
    cat <<EOF
Usage: bash docker/run.sh [options]

Run pytest inside a Trinity Docker container.

Options:
  -m, --module <path>    Test target under tests/ (directory or file). Default: tests/
  -k, --keyword <expr>   Pytest -k expression used to filter tests
  -q, --quiet            Run pytest in quiet mode
  -h, --help             Show this help message and exit

Examples:
  bash docker/run.sh
  bash docker/run.sh --module buffer
  bash docker/run.sh --module trainer/trainer_test
  bash docker/run.sh --module common --keyword test_config
  bash docker/run.sh --module common --keyword test_config --quiet
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
quiet_mode=0

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
            quiet_mode=1
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

if [[ $# -gt 0 ]]; then
    fail "Unexpected positional arguments: $*. Use --help for usage information."
fi

if ! init_docker_compose; then
    exit 1
fi

test_target="tests"
if [[ -n "$module_name" ]]; then
    candidate_target="tests/${module_name}"
    if [[ -d "$SCRIPT_DIR/../${candidate_target}" || -f "$SCRIPT_DIR/../${candidate_target}" ]]; then
        test_target="$candidate_target"
    elif [[ -f "$SCRIPT_DIR/../${candidate_target}.py" ]]; then
        test_target="${candidate_target}.py"
    else
        fail "Test target '${candidate_target}' does not exist."
    fi
fi

container_id="$("${COMPOSE_CMD[@]}" ps -a -q "$SERVICE_NAME" 2>/dev/null)"
if [[ -z "$container_id" ]]; then
    fail "Container '${SERVICE_NAME}' does not exist. Run 'bash docker/start.sh' first."
fi

running_id="$("${COMPOSE_CMD[@]}" ps -q "$SERVICE_NAME" 2>/dev/null)"
if [[ -z "$running_id" ]]; then
    fail "Container '${SERVICE_NAME}' exists but is not running. Start it before running tests."
fi

pytest_args=(pytest "$test_target" -s)
if [[ "$quiet_mode" -eq 1 ]]; then
    pytest_args+=(-q)
else
    pytest_args+=(-v)
fi

if [[ -n "$keyword_expr" ]]; then
    pytest_args+=(-k "$keyword_expr")
fi

printf -v pytest_cmd '%q ' "${pytest_args[@]}"
pytest_cmd="source /opt/venv/bin/activate && ${pytest_cmd% }"

echo "Running tests in ${SERVICE_NAME}: ${pytest_cmd}"
"${COMPOSE_CMD[@]}" exec "$SERVICE_NAME" bash -c "$pytest_cmd"
