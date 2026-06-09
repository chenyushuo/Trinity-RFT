# -*- coding: utf-8 -*-
"""Checkpoint coordination for verl trainer.

Provides CheckpointCoordinator — the single entry point for all checkpoint
operations. It wraps the CheckpointMonitor Ray actor and background-thread
management so that callers (VERLTrainer, workers, engine helpers) never need
to interact with CheckpointMonitor directly.

Flow:
    1. Main thread collects state dicts (requires FSDP/Megatron context)
    2. CheckpointCoordinator.save_async() offloads torch.save to a background thread
    3. The thread calls notify_started → save → notify_finished on CheckpointMonitor
    4. CheckpointMonitor only updates latest_state_dict_iteration.txt after ALL
       registered saves for a given step complete
"""

import threading
from typing import Callable

import ray
from ray.actor import ActorHandle

from trinity.utils.log import get_logger

logger = get_logger(__name__)


class CheckpointCoordinator:
    """Manages checkpoint saves and coordinates with CheckpointMonitor.

    This is the **only** interface callers should use for checkpoint
    operations.  CheckpointMonitor (a Ray actor) is an internal
    implementation detail — all Monitor RPCs are routed through this class.

    Each save operation runs in a named background thread. Threads with the same
    name are serialized (the previous one is joined before a new one starts).
    The CheckpointMonitor is notified before and after each save, which gates
    the iteration file update.

    Usage::

        coordinator = CheckpointCoordinator(checkpoint_monitor)
        coordinator.save_async("model", lambda: torch.save(sd, path), step, is_state_dict=True)
        # ... training continues while save runs in background ...
        coordinator.wait_all()  # block until all saves complete
    """

    def __init__(self, checkpoint_monitor: ActorHandle):
        self._monitor = checkpoint_monitor
        self._threads: dict[str, threading.Thread] = {}

    # ------------------------------------------------------------------
    # Worker-level: background save with Monitor notifications
    # ------------------------------------------------------------------

    def save_async(
        self,
        name: str,
        save_fn: Callable[[], None],
        global_step: int,
        is_state_dict: bool = False,
    ) -> None:
        """Run save_fn in a background thread with CheckpointMonitor coordination.

        Joins any previous thread with the same ``name`` before starting, so
        concurrent writes to the same destination are impossible.

        Args:
            name: Logical name for this save slot (e.g. "model_state_dict").
            save_fn: Zero-arg callable that does the actual I/O.
            global_step: Training step, passed to CheckpointMonitor.
            is_state_dict: True for state-dict-only saves (weight sync),
                False for full checkpoint components.
        """
        self._join(name)

        def _run():
            try:
                ctx = ray.get_runtime_context()
                ray.get(
                    self._monitor.notify_started.remote(
                        node_id=ctx.get_node_id(), job_id=ctx.get_job_id()
                    )
                )
                save_fn()
                ray.get(self._monitor.notify_finished.remote(global_step, is_state_dict))
            except Exception:
                logger.error(
                    f"Background save '{name}' failed at step {global_step}", exc_info=True
                )
                raise

        t = threading.Thread(target=_run, name=f"ckpt-{name}-step{global_step}")
        t.start()
        self._threads[name] = t

    def save_sync(
        self,
        save_fn: Callable[[], None],
        global_step: int,
        is_state_dict: bool = False,
    ) -> None:
        """Run save_fn synchronously with CheckpointMonitor notifications.

        Use this for saves that must run on the main thread (e.g. because they
        involve distributed barriers).
        """
        ctx = ray.get_runtime_context()
        ray.get(
            self._monitor.notify_started.remote(node_id=ctx.get_node_id(), job_id=ctx.get_job_id())
        )
        save_fn()
        ray.get(self._monitor.notify_finished.remote(global_step, is_state_dict))

    def _join(self, name: str) -> None:
        t = self._threads.get(name)
        if t is not None:
            t.join()

    def wait_all(self) -> None:
        """Block until all background save threads complete."""
        for t in self._threads.values():
            t.join()
        self._threads.clear()

    # ------------------------------------------------------------------
    # Trainer-level: register expected save counts and commit step
    # ------------------------------------------------------------------

    async def register_and_monitor(
        self,
        global_step: int,
        is_state_dict: bool = False,
        state_dict_thread_count: int = 0,
        checkpoint_thread_count: int = 0,
    ) -> None:
        """Register expected save thread counts and commit the step.

        Called by VERLTrainer after dispatching save operations to workers.
        This wraps the two Monitor RPCs (register_thread_count + monitor_step)
        into a single call so the trainer never talks to Monitor directly.

        Args:
            global_step: Training step.
            is_state_dict: Whether this is a state-dict-only save.
            state_dict_thread_count: Number of state dict save threads to expect.
            checkpoint_thread_count: Number of full checkpoint save threads to expect.
        """
        if state_dict_thread_count or checkpoint_thread_count:
            ray.get(
                self._monitor.register_thread_count.remote(
                    global_step,
                    state_dict_thread_count=state_dict_thread_count,
                    checkpoint_thread_count=checkpoint_thread_count,
                )
            )
        await self._monitor.monitor_step.remote(global_step, is_state_dict=is_state_dict)
