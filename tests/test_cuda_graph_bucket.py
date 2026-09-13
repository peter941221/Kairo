import unittest
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

from scripts.wsl.cuda_graph_bucket import CudaGraphBucketCache


class _FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class _FakeCuda:
    def __init__(self):
        self.graphs = []

    def CUDAGraph(self):
        graph = _FakeGraph()
        self.graphs.append(graph)
        return graph

    def Stream(self):
        return _FakeStream()

    def synchronize(self):
        return None

    def stream(self, _stream):
        return nullcontext()

    def graph(self, _graph, stream=None):
        return nullcontext()


class _FakeStream:
    def synchronize(self):
        return None


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()


class CudaGraphBucketTests(unittest.TestCase):
    def test_same_shape_captures_once_and_hits(self):
        torch = _FakeTorch()
        cache = CudaGraphBucketCache(torch, warmups=2)
        calls = []

        def factory():
            calls.append(True)
            return "output"

        first = cache.get_or_capture((32, 4096, 4096), factory)
        second = cache.get_or_capture((32, 4096, 4096), factory)
        self.assertIs(first, second)
        self.assertEqual(cache.stats(), {"buckets": 1, "captures": 1, "hits": 1})
        first.replay()
        self.assertEqual(torch.cuda.graphs[0].replays, 1)
        self.assertEqual(len(calls), 3)  # two warmups plus capture body

    def test_concurrent_first_requests_share_capture(self):
        torch = _FakeTorch()
        cache = CudaGraphBucketCache(torch, warmups=2)

        def factory():
            return "output"

        with ThreadPoolExecutor(max_workers=8) as pool:
            buckets = list(
                pool.map(
                    lambda _index: cache.get_or_capture((128, 4096, 4096), factory),
                    range(8),
                )
            )
        self.assertEqual(len({id(bucket) for bucket in buckets}), 1)
        self.assertEqual(cache.stats(), {"buckets": 1, "captures": 1, "hits": 7})

    def test_namespace_prevents_cross_operation_reuse(self):
        torch = _FakeTorch()
        cache = CudaGraphBucketCache(torch, warmups=0)
        first = cache.get_or_capture((32, 4096, 4096), lambda: "a", namespace="cutlass")
        second = cache.get_or_capture((32, 4096, 4096), lambda: "b", namespace="b12x")
        self.assertIsNot(first, second)
        self.assertEqual(cache.stats(), {"buckets": 2, "captures": 2, "hits": 0})
