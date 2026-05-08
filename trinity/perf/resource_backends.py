"""System resource collection backends for performance tooling."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Iterable

import psutil
from pynvml import (
    NVMLError,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


@dataclass
class GPUSample:
    """One GPU sample at one point in time."""

    gpu_id: int
    name: str
    gpu_util_percent: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float

    def to_dict(self) -> dict:
        """Serialize the GPU sample to a dictionary."""
        return asdict(self)


@dataclass
class ResourceSample:
    """One system resource sample at one point in time."""

    timestamp: float
    cpu_percent: float
    memory_rss_mb: float
    memory_percent: float
    gpu_metrics: list[GPUSample]

    def to_dict(self) -> dict:
        """Serialize the resource sample to a dictionary."""
        payload = asdict(self)
        payload["gpu_metrics"] = [gpu_sample.to_dict() for gpu_sample in self.gpu_metrics]
        return payload


class SystemResourceBackend:
    """Collect system-level CPU, memory and per-GPU metrics."""

    def __init__(
        self,
        gpu_subsample_count: int = 5,
        gpu_subsample_interval_seconds: float = 0.2,
    ) -> None:
        if gpu_subsample_count <= 0:
            raise ValueError("gpu_subsample_count must be greater than 0.")
        if gpu_subsample_interval_seconds < 0:
            raise ValueError("gpu_subsample_interval_seconds must be non-negative.")
        self._process = psutil.Process()
        self._initialized = False
        self._gpu_count = 0
        self._gpu_subsample_count = gpu_subsample_count
        self._gpu_subsample_interval_seconds = gpu_subsample_interval_seconds

    def open(self) -> None:
        """Initialize the GPU management library and validate the environment."""
        if self._initialized:
            return
        try:
            nvmlInit()
            self._gpu_count = nvmlDeviceGetCount()
        except NVMLError as error:
            raise RuntimeError(f"Failed to initialize NVML: {error}") from error
        if self._gpu_count <= 0:
            self.close()
            raise RuntimeError("No GPU devices detected by NVML.")

        self._process.cpu_percent(interval=None)
        self._initialized = True

    def close(self) -> None:
        """Release NVML resources."""
        if not self._initialized:
            return
        try:
            nvmlShutdown()
        except NVMLError:
            pass
        self._initialized = False
        self._gpu_count = 0

    def sample(self) -> ResourceSample:
        """Collect one resource sample."""
        if not self._initialized:
            raise RuntimeError("SystemResourceBackend must be opened before sampling.")

        timestamp = time.time()
        memory_info = self._process.memory_info()
        gpu_metrics = self._collect_gpu_metrics()
        for _ in range(1, self._gpu_subsample_count):
            if self._gpu_subsample_interval_seconds > 0:
                time.sleep(self._gpu_subsample_interval_seconds)
            gpu_metrics = self._merge_gpu_metrics(gpu_metrics, self._collect_gpu_metrics())

        return ResourceSample(
            timestamp=timestamp,
            cpu_percent=float(self._process.cpu_percent(interval=None)),
            memory_rss_mb=float(memory_info.rss) / (1024 * 1024),
            memory_percent=float(self._process.memory_percent()),
            gpu_metrics=gpu_metrics,
        )

    def _collect_gpu_metrics(self) -> list[GPUSample]:
        """Collect one instantaneous GPU snapshot from NVML."""
        gpu_metrics: list[GPUSample] = []
        for gpu_index in range(self._gpu_count):
            gpu_handle = nvmlDeviceGetHandleByIndex(gpu_index)
            utilization = nvmlDeviceGetUtilizationRates(gpu_handle)
            gpu_memory = nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_name = nvmlDeviceGetName(gpu_handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")
            gpu_metrics.append(
                GPUSample(
                    gpu_id=gpu_index,
                    name=str(gpu_name),
                    gpu_util_percent=float(utilization.gpu),
                    gpu_memory_used_mb=float(gpu_memory.used) / (1024 * 1024),
                    gpu_memory_total_mb=float(gpu_memory.total) / (1024 * 1024),
                )
            )
        return gpu_metrics

    def _merge_gpu_metrics(
        self,
        base_metrics: Iterable[GPUSample],
        next_metrics: Iterable[GPUSample],
    ) -> list[GPUSample]:
        """Merge GPU snapshots by keeping peak utilization and memory usage."""
        merged_metrics = {gpu_metric.gpu_id: gpu_metric for gpu_metric in base_metrics}
        for gpu_metric in next_metrics:
            prior_metric = merged_metrics.get(gpu_metric.gpu_id)
            if prior_metric is None:
                merged_metrics[gpu_metric.gpu_id] = gpu_metric
                continue
            merged_metrics[gpu_metric.gpu_id] = GPUSample(
                gpu_id=gpu_metric.gpu_id,
                name=gpu_metric.name,
                gpu_util_percent=max(prior_metric.gpu_util_percent, gpu_metric.gpu_util_percent),
                gpu_memory_used_mb=max(
                    prior_metric.gpu_memory_used_mb, gpu_metric.gpu_memory_used_mb
                ),
                gpu_memory_total_mb=gpu_metric.gpu_memory_total_mb,
            )
        return [merged_metrics[gpu_index] for gpu_index in sorted(merged_metrics)]
