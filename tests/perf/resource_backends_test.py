"""Tests for NVML-backed perf resource sampling."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from trinity.perf.resource_backends import SystemResourceBackend


class FakeProcess:
    def __init__(self):
        self._cpu_values = iter([0.0, 12.5])

    def cpu_percent(self, interval=None):
        return next(self._cpu_values)

    def memory_info(self):
        return SimpleNamespace(rss=256 * 1024 * 1024)

    def memory_percent(self):
        return 1.25


class SystemResourceBackendTest(unittest.TestCase):
    @patch("trinity.perf.resource_backends.time.sleep")
    @patch("trinity.perf.resource_backends.nvmlDeviceGetName", return_value="GPU-0")
    @patch("trinity.perf.resource_backends.nvmlDeviceGetHandleByIndex", return_value=object())
    @patch("trinity.perf.resource_backends.nvmlDeviceGetCount", return_value=1)
    @patch("trinity.perf.resource_backends.nvmlShutdown")
    @patch("trinity.perf.resource_backends.nvmlInit")
    @patch("trinity.perf.resource_backends.psutil.Process", return_value=FakeProcess())
    def test_sample_keeps_peak_gpu_utilization_within_one_outer_sample(
        self,
        _mock_process,
        _mock_nvml_init,
        _mock_nvml_shutdown,
        _mock_gpu_count,
        _mock_gpu_handle,
        _mock_gpu_name,
        _mock_sleep,
    ):
        utilization_side_effect = [
            SimpleNamespace(gpu=0.0),
            SimpleNamespace(gpu=35.0),
            SimpleNamespace(gpu=80.0),
        ]
        memory_side_effect = [
            SimpleNamespace(used=100 * 1024 * 1024, total=500 * 1024 * 1024),
            SimpleNamespace(used=120 * 1024 * 1024, total=500 * 1024 * 1024),
            SimpleNamespace(used=110 * 1024 * 1024, total=500 * 1024 * 1024),
        ]

        with patch(
            "trinity.perf.resource_backends.nvmlDeviceGetUtilizationRates",
            side_effect=utilization_side_effect,
        ), patch(
            "trinity.perf.resource_backends.nvmlDeviceGetMemoryInfo",
            side_effect=memory_side_effect,
        ):
            backend = SystemResourceBackend(
                gpu_subsample_count=3,
                gpu_subsample_interval_seconds=0.0,
            )
            backend.open()
            sample = backend.sample()
            backend.close()

        self.assertEqual(sample.cpu_percent, 12.5)
        self.assertEqual(sample.gpu_metrics[0].gpu_util_percent, 80.0)
        self.assertEqual(sample.gpu_metrics[0].gpu_memory_used_mb, 120.0)
