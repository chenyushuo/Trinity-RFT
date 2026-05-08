"""Tests for perf resource timeline helpers."""

import itertools
import time
import unittest
from typing import cast

from trinity.perf.resource_backends import (
    GPUSample,
    ResourceSample,
    SystemResourceBackend,
)
from trinity.perf.resource_sampler import ResourceSampler


class FakeBackend:
    def __init__(self):
        self.opened = False
        self.closed = False
        self.sample_index = itertools.count()

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def sample(self):
        index = next(self.sample_index)
        return ResourceSample(
            timestamp=1000.0 + index,
            cpu_percent=50.0 + index,
            memory_rss_mb=1024.0 + index,
            memory_percent=20.0 + index,
            gpu_metrics=[
                GPUSample(
                    gpu_id=0,
                    name="GPU-0",
                    gpu_util_percent=70.0 + index,
                    gpu_memory_used_mb=16000.0 + index,
                    gpu_memory_total_mb=24000.0,
                ),
                GPUSample(
                    gpu_id=1,
                    name="GPU-1",
                    gpu_util_percent=75.0 + index,
                    gpu_memory_used_mb=15000.0 + index,
                    gpu_memory_total_mb=24000.0,
                ),
            ],
        )


class ResourceSamplerTest(unittest.TestCase):
    def test_resource_sampler_collects_samples(self):
        backend = FakeBackend()
        sampler = ResourceSampler(
            interval_seconds=0.01,
            backend=cast(SystemResourceBackend, backend),
        )

        sampler.start()
        time.sleep(0.03)
        samples = sampler.stop()

        self.assertTrue(backend.opened)
        self.assertTrue(backend.closed)
        self.assertGreaterEqual(len(samples), 2)
        self.assertEqual(samples[0].gpu_metrics[0].gpu_id, 0)

    def test_resource_samples_serialize_cpu_single_line_and_gpu_per_device(self):
        samples = [FakeBackend().sample(), FakeBackend().sample()]

        payload = {"resource_timeline": [sample.to_dict() for sample in samples]}

        self.assertEqual(len(payload["resource_timeline"]), 2)
        self.assertEqual(payload["resource_timeline"][0]["cpu_percent"], 50.0)
        self.assertEqual(len(payload["resource_timeline"][0]["gpu_metrics"]), 2)
        self.assertEqual(
            payload["resource_timeline"][0]["gpu_metrics"][0]["name"],
            "GPU-0",
        )
