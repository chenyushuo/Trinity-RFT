"""Helpers for TensorBoard metric parsing and aggregation."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TASK_EXECUTION_METRIC_NAME = "rollout/time/task_execution/mean"
RUN_EXECUTION_METRIC_NAME = "rollout/time/run_execution/mean"
FINISHED_TASK_METRIC_NAME = "rollout/finished_task_count"


class TensorBoardScalarReader:
    """Read scalar metrics from TensorBoard event files."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics = self._load_metrics(log_dir)

    def _load_metrics(self, log_dir: str) -> dict[str, dict[int, float]]:
        metric_map: dict[str, dict[int, float]] = defaultdict(dict)
        for event_file in self._find_event_files(log_dir):
            accumulator = EventAccumulator(event_file)
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                for scalar in accumulator.Scalars(tag):
                    prior_value = metric_map[tag].get(scalar.step)
                    if prior_value is None or scalar.value > prior_value:
                        metric_map[tag][scalar.step] = scalar.value
        return dict(metric_map)

    def _find_event_files(self, log_dir: str) -> list[str]:
        event_files: list[str] = []
        for root, _, files in os.walk(log_dir):
            for file_name in files:
                if file_name.startswith("events.out.tfevents."):
                    event_files.append(os.path.join(root, file_name))
        return sorted(event_files)


def extract_raw_metrics_for_step(
    metric_map: dict[str, dict[int, float]], step: int
) -> dict[str, float]:
    """Extract all scalar metrics that were logged for one step."""
    return {
        metric_name: float(step_values[step])
        for metric_name, step_values in metric_map.items()
        if step in step_values
    }


def collect_step_metrics(metric_map: dict[str, dict[int, float]]) -> list[dict[str, Any]]:
    """Build per-step metrics from TensorBoard scalars."""
    step_numbers = sorted(metric_map.get(FINISHED_TASK_METRIC_NAME, {}).keys())
    step_metrics: list[dict[str, Any]] = []
    for step in step_numbers:
        metrics = extract_raw_metrics_for_step(metric_map, step)
        metrics["step"] = step
        step_metrics.append(metrics)
    return step_metrics
