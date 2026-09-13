"""Shape-bucketed CUDA Graph capture/replay helper for static GPU workloads."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class CudaGraphBucket:
    """One captured graph for one exact static shape.

    ``factory`` must close over stable CUDA input/output buffers.  A caller that
    serves changing values should copy new values into those buffers before
    calling :meth:`replay`; the graph itself never receives changing tensor
    objects or dimensions.
    """

    torch: Any
    shape: tuple[int, ...]
    warmups: int = 2
    graph: Any = None
    output: Any = None
    capture_ms: float | None = None

    def capture(self, factory: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        self.graph = self.torch.cuda.CUDAGraph()
        capture_stream = self.torch.cuda.Stream()
        self.torch.cuda.synchronize()
        with self.torch.cuda.stream(capture_stream):
            for _ in range(max(2, self.warmups)):
                factory()
        capture_stream.synchronize()
        self.torch.cuda.synchronize()
        with self.torch.cuda.graph(self.graph, stream=capture_stream):
            self.output = factory()
        self.torch.cuda.synchronize()
        self.capture_ms = (time.perf_counter() - started) * 1000.0
        return self.output

    def replay(self) -> Any:
        if self.graph is None:
            raise RuntimeError(f"CUDA Graph bucket {self.shape} has not been captured")
        self.graph.replay()
        return self.output


class CudaGraphBucketCache:
    """In-process cache that refuses to reuse a graph for another shape."""

    def __init__(self, torch_module: Any, warmups: int = 2):
        self.torch = torch_module
        self.warmups = warmups
        self._buckets: dict[tuple[int, ...], CudaGraphBucket] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._captures = 0

    def get_or_capture(self, shape: tuple[int, ...], factory: Callable[[], Any]) -> CudaGraphBucket:
        key = tuple(int(value) for value in shape)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is not None:
                self._hits += 1
                return bucket
            bucket = CudaGraphBucket(self.torch, key, self.warmups)
            bucket.capture(factory)
            self._buckets[key] = bucket
            self._captures += 1
            return bucket

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "buckets": len(self._buckets),
                "captures": self._captures,
                "hits": self._hits,
            }
