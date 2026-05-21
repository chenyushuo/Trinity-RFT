"""Unified metrics aggregation utilities for Trinity-RFT.

Metric keys may carry an aggregation-type suffix in the form ``name:agg``.
Supported suffixes: ``:mean``, ``:sum``, ``:max``, ``:min``, ``:last``.
Keys without a suffix default to mean aggregation.
"""

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


class AggType(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    LAST = "last"


def take_last(values: List[float]) -> float:
    return values[-1]


AGG_REDUCERS: Dict[AggType, Callable[[List[float]], float]] = {
    AggType.SUM: sum,
    AggType.MAX: max,
    AggType.MIN: min,
    AggType.LAST: take_last,
}


STAT_FNS: Dict[str, Callable[[np.ndarray], np.generic]] = {
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
    "std": np.std,
}


def group_numeric_metrics(
    metric_dicts: List[Dict[str, float]]
) -> Dict[Tuple[str, AggType], List[float]]:
    grouped: Dict[Tuple[str, AggType], List[float]] = defaultdict(list)
    for metric_dict in metric_dicts:
        for key, value in metric_dict.items():
            if not isinstance(value, (int, float)):
                continue
            name, agg = parse_metric_key(key)
            grouped[(name, agg)].append(float(value))
    return grouped


def group_metrics_by_canonical_key(
    metric_dicts: List[Dict[str, float]],
) -> Dict[str, Tuple[AggType, List[float]]]:
    grouped: Dict[str, Tuple[AggType, List[float]]] = {}
    for metric_dict in metric_dicts:
        for key, value in metric_dict.items():
            if not isinstance(value, (int, float)):
                continue
            name, agg = parse_metric_key(key)
            canonical_key = f"{name}:{agg.value}" if agg != AggType.MEAN else name
            if canonical_key not in grouped:
                grouped[canonical_key] = (agg, [])
            grouped[canonical_key][1].append(float(value))
    return grouped


def parse_metric_key(key: str) -> Tuple[str, AggType]:
    """Parse a metric key into (name, aggregation_type).

    Examples:
        "reward"               -> ("reward", AggType.MEAN)
        "experience_count:sum" -> ("experience_count", AggType.SUM)
        "model_version:last"   -> ("model_version", AggType.LAST)
        "some:unknown_suffix"  -> ("some:unknown_suffix", AggType.MEAN)
    """
    if ":" in key:
        name, agg_str = key.rsplit(":", 1)
        try:
            return name, AggType(agg_str)
        except ValueError:
            return key, AggType.MEAN
    return key, AggType.MEAN


def aggregate_metrics(
    metric_dicts: List[Dict[str, float]],
    prefix: str = "",
    default_output_stats: List[str] | None = None,
) -> Dict[str, float]:
    """Aggregate a list of metric dictionaries respecting per-key aggregation types.

    For keys with AggType.MEAN, outputs ``{prefix}/{name}/mean``, ``/max``, ``/min``
    (controlled by *default_output_stats*).
    For AggType.SUM, outputs ``{prefix}/{name}/sum``.
    For AggType.MAX, outputs ``{prefix}/{name}/max``.
    For AggType.MIN, outputs ``{prefix}/{name}/min``.
    For AggType.LAST, outputs ``{prefix}/{name}/last``.

    Args:
        metric_dicts: List of flat metric dictionaries (values must be numeric).
        prefix: Optional prefix prepended as ``{prefix}/{name}/...``.
        default_output_stats: Stats to output for MEAN metrics. Defaults to ["mean", "max", "min"].

    Returns:
        Flat dictionary of aggregated metrics ready for monitor logging.
    """
    if not metric_dicts:
        return {}

    if default_output_stats is None:
        default_output_stats = ["mean", "max", "min"]

    grouped = group_numeric_metrics(metric_dicts)

    result: Dict[str, float] = {}
    prefix_str = f"{prefix}/" if prefix else ""

    for (name, agg), values in grouped.items():
        if agg == AggType.MEAN:
            arr = np.array(values)
            for stat in default_output_stats:
                fn = STAT_FNS.get(stat)
                if fn is not None:
                    result[f"{prefix_str}{name}/{stat}"] = fn(arr).item()
            continue

        reducer = AGG_REDUCERS.get(agg)
        if reducer is not None:
            result[f"{prefix_str}{name}/{agg.value}"] = reducer(values)

    return result


def aggregate_eval_metrics(
    metric_dicts: List[Dict[str, float]],
    prefix: str = "",
    output_stats: List[str] | None = None,
    detailed_stats: bool = False,
) -> Dict[str, float]:
    """Aggregate eval metrics with optional detailed statistics.

    For MEAN metrics:
      - If detailed_stats=True: output mean/max/min/std per the output_stats list.
      - If detailed_stats=False: output only the mean value as ``{prefix}/{name}``.
    For non-MEAN metrics: same behavior as aggregate_metrics.
    """
    if not metric_dicts:
        return {}

    if output_stats is None:
        output_stats = ["mean", "max", "min", "std"]

    grouped = group_numeric_metrics(metric_dicts)

    result: Dict[str, float] = {}
    prefix_str = f"{prefix}/" if prefix else ""

    for (name, agg), values in grouped.items():
        if agg == AggType.MEAN:
            arr = np.array(values)
            if detailed_stats:
                for stat in output_stats:
                    fn = STAT_FNS.get(stat)
                    if fn is not None:
                        result[f"{prefix_str}{name}/{stat}"] = fn(arr).item()
            else:
                result[f"{prefix_str}{name}"] = np.mean(arr).item()
            continue

        reducer = AGG_REDUCERS.get(agg)
        if reducer is not None:
            result[f"{prefix_str}{name}/{agg.value}"] = reducer(values)

    return result


def aggregate_run_level_metrics(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate experience-level metrics into a single run-level metric dict.

    Unlike batch-level aggregation, this preserves the original key format (with ``:agg``
    suffix if present) so that downstream task/batch aggregation can still see the
    aggregation type annotation.

    Aggregation rules:
      - MEAN keys: averaged across experiences
      - SUM keys: summed across experiences
      - MAX keys: max across experiences
      - MIN keys: min across experiences
      - LAST keys: last value
    """
    if not metric_dicts:
        return {}

    grouped = group_metrics_by_canonical_key(metric_dicts)

    result: Dict[str, float] = {}
    for key, (agg, values) in grouped.items():
        if agg == AggType.MEAN:
            result[key] = sum(values) / len(values)
            continue

        reducer = AGG_REDUCERS.get(agg)
        if reducer is not None:
            result[key] = reducer(values)

    return result


# adapted from verl/trainer/ppo/metric_utils.py
def bootstrap_metric(
    data: List[Any],
    subset_size: int,
    reduce_fns: List[Callable[[List[Any]], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> List[Tuple[float, float]]:
    """Estimate metric statistics with bootstrap resampling."""
    np.random.seed(seed)

    bootstrap_metric_lists = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lists[i].append(reduce_fn(bootstrap_data))

    return [(float(np.mean(lst)), float(np.std(lst))) for lst in bootstrap_metric_lists]


def calculate_task_level_metrics(
    metrics: List[Dict[str, float]], is_eval: bool
) -> Dict[str, float]:
    """Calculate task-level metrics from multiple runs of the same task."""
    if not metrics:
        return {}

    grouped = group_metrics_by_canonical_key(metrics)

    if not is_eval:
        return aggregate_run_level_metrics(metrics)

    result: Dict[str, float] = {}
    for key, (agg, values) in grouped.items():
        name, _ = parse_metric_key(key)

        if agg != AggType.MEAN:
            reducer = AGG_REDUCERS.get(agg)
            if reducer is not None:
                result[f"{name}:{agg.value}"] = reducer(values)
            continue

        if "time/task_execution" in name or "time/run_execution" in name:
            result[key] = sum(values) / len(values)
            continue

        n_values = len(values)
        result[f"{key}/mean@{n_values}"] = float(np.mean(values))
        result[f"{key}/std@{n_values}"] = float(np.std(values))

        if n_values <= 1:
            continue

        ns = []
        n = 2
        while n < n_values:
            ns.append(n)
            n *= 2
        ns.append(n_values)

        for subset_size in ns:
            [(best_mean, _), (worst_mean, _)] = bootstrap_metric(
                data=values,
                subset_size=subset_size,
                reduce_fns=[max, min],
                seed=42,
            )
            result[f"{key}/best@{subset_size}"] = best_mean
            result[f"{key}/worst@{subset_size}"] = worst_mean

    return result
