# Docker Test Environment

This document focuses on two things:

1. how to prepare the Docker-based test environment
2. how to run Trinity unit tests inside that environment

## Hardware Requirements

- `trinity-node-1` requires at least 2 GPUs.
- `trinity-node-2` requires at least 2 GPUs.
- Each container requires `64G` shared memory.

Default GPU assignment:

- `trinity-node-1`: GPU `0` and GPU `1`
- `trinity-node-2`: GPU `2` and GPU `3`

If the current machine uses different GPU indices, override them in `docker/env`.

## Test Environment Setup

Before running any Docker test command:

1. Copy `docker/env.example` to `docker/env`.
2. Update `docker/env` for the current machine.
3. Make sure `TRINITY_MOUNT_DIR` points to a host directory that contains the required models, datasets, and checkpoints.

The helper scripts do not read `env.example` automatically. If `docker/env` is missing, they will stop and ask you to create it first.

Required settings in `docker/env`:

- `TRINITY_DOCKER_IMAGE`: Docker image used by both containers.
- `TRINITY_MOUNT_DIR`: Host directory mounted to `/mnt` inside the containers.
- `TRINITY_NODE1_GPU_0`, `TRINITY_NODE1_GPU_1`: GPU indices for `trinity-node-1`.
- `TRINITY_NODE2_GPU_0`, `TRINITY_NODE2_GPU_1`: GPU indices for `trinity-node-2`.
- `TRINITY_RAY_DASHBOARD_PORT`: Host port mapped to the Ray dashboard.

Optional settings in `docker/env`:

- `TRINITY_HF_ENDPOINT`: Hugging Face mirror or endpoint. Leave it empty to use the container default.
- `TRINITY_PYPI_INDEX_URL`: Python package index used inside containers. Leave it empty to use the container default.

## Start And Check The Environment

Start the Docker test environment:

```bash
bash docker/start.sh
```

Check whether both containers are up and whether Ray is healthy:

```bash
bash docker/status.sh
```

Expected interpretation:

- If a container does not exist, run `bash docker/start.sh` first.
- If a container exists but is stopped, start the environment again before running tests.
- If a container is running but Ray is unhealthy, resolve the container startup problem before running tests.

## Run Tests

Use `bash docker/run.sh` to execute pytest inside `trinity-node-1`.

Run one narrow test module:

```bash
bash docker/run.sh --module common
```

Run a filtered subset when a smaller slice is known:

```bash
bash docker/run.sh --module common --keyword test_config
```

Rules for test execution:

- Always prefer the smallest viable `--module`.
- Add `--keyword` whenever you know the failing test name, keyword, or a smaller slice.
- Do not widen the test scope unless the narrower check is insufficient.
- `run.sh` always executes tests inside `trinity-node-1`.

## Stop The Environment

Stop the Docker test environment after use:

```bash
bash docker/stop.sh
```

## Remote Testing

When working from a machine that cannot run the Docker environment (e.g. macOS without GPU), use `sync.sh` and `remote_run.sh` to sync code and run tests on a remote GPU server.

### Prepare the GPU Server

SSH into the remote GPU server and follow the [Test Environment Setup](#test-environment-setup) section above to set up `docker/env`, then start the Docker environment:

```bash
bash docker/start.sh
bash docker/status.sh   # verify containers and Ray are healthy
```

The Docker environment should stay running on the GPU server. You do not need to restart it for each test run.

### Local Setup

1. Copy `docker/remote.env.example` to `docker/remote.env`.
2. Fill in the remote server details (`TRINITY_REMOTE_HOST`, `TRINITY_REMOTE_WORKSPACE`).
3. Make sure SSH key authentication is configured for the remote host.

`docker/remote.env` is gitignored and is not synced to the remote

### Sync Code

```bash
bash docker/sync.sh              # sync local code to remote workspace
bash docker/sync.sh --dry-run    # preview what would be synced
```

### Run Tests Remotely

```bash
bash docker/remote_run.sh --module common
bash docker/remote_run.sh --module common --keyword test_config
bash docker/remote_run.sh --module explorer --no-sync --timeout 300
```

`remote_run.sh` syncs code first (unless `--no-sync` is passed), then runs `docker/run.sh` on the remote machine via SSH.

## Script Roles

- [start.sh](start.sh): starts the test environment.
- [status.sh](status.sh): checks container state and Ray health.
- [run.sh](run.sh): runs pytest in the Docker test environment.
- [stop.sh](stop.sh): shuts the test environment down cleanly.
- [sync.sh](sync.sh): syncs local code to a remote workspace via rsync.
- [remote_run.sh](remote_run.sh): syncs code and runs tests on a remote server.
