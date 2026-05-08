"""Threaded resource sampler for performance tooling."""

from __future__ import annotations

import threading
import time
from typing import Optional

from trinity.perf.resource_backends import ResourceSample, SystemResourceBackend


class ResourceSampler:
    """Periodically collect system resource samples in a background thread."""

    def __init__(
        self,
        interval_seconds: float,
        backend: Optional[SystemResourceBackend] = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than 0.")
        self.interval_seconds = interval_seconds
        self.backend = backend or SystemResourceBackend()
        self._samples: list[ResourceSample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def start(self) -> None:
        """Start sampling in the background."""
        if self._started:
            raise RuntimeError("ResourceSampler has already been started.")
        self.backend.open()
        self._started = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> list[ResourceSample]:
        """Stop sampling and return the collected samples."""
        if not self._started:
            return self.samples()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.backend.close()
        self._started = False
        return self.samples()

    def samples(self) -> list[ResourceSample]:
        """Return a snapshot of all collected samples."""
        with self._lock:
            return list(self._samples)

    def _run(self) -> None:
        next_sample_time = time.monotonic()
        while not self._stop_event.is_set():
            self._collect_once()
            next_sample_time += self.interval_seconds
            remaining_time = max(0.0, next_sample_time - time.monotonic())
            if self._stop_event.wait(remaining_time):
                break

    def _collect_once(self) -> None:
        sample = self.backend.sample()
        with self._lock:
            self._samples.append(sample)
