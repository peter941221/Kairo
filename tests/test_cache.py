import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kairo_lab.cache import KernelCacheKey, RuntimeKernelCache


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.cache = RuntimeKernelCache(Path(self.directory.name))
        self.key = KernelCacheKey("blueprint-a", (32, 4096, 4096), "555.1", "sm120")

    def tearDown(self):
        self.directory.cleanup()

    def test_get_or_build_then_hits(self):
        builds = []

        def build():
            builds.append(True)
            return b"cubin"

        self.assertEqual(self.cache.get_or_build(self.key, build), (b"cubin", False))
        self.assertEqual(self.cache.get_or_build(self.key, build), (b"cubin", True))
        self.assertEqual(len(builds), 1)
        self.assertEqual(self.cache.stats()["hits"], 1)

    def test_hardware_fingerprint_change_is_explainable(self):
        self.cache.store(self.key, b"ptx")
        changed = KernelCacheKey("blueprint-a", (32, 4096, 4096), "555.2", "sm120")
        self.assertIsNone(self.cache.load(changed))
        self.assertEqual(self.cache.stats()["miss_reasons"], {"driver_version_changed": 1})

    def test_concurrent_misses_single_flight_build(self):
        builds = []
        builds_lock = threading.Lock()

        def build():
            with builds_lock:
                builds.append(True)
            time.sleep(0.03)
            return b"shared-cubin"

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: self.cache.get_or_build(self.key, build), range(6)))
        self.assertEqual(len(builds), 1)
        self.assertEqual({payload for payload, _ in results}, {b"shared-cubin"})
        self.assertEqual(sum(not hit for _, hit in results), 1)
        self.assertEqual(self.cache.stats()["stores"], 1)

    def test_integrity_failure_invalidates(self):
        path = self.cache.store(self.key, b"ptx")
        path.write_bytes(b"tampered")
        self.assertIsNone(self.cache.load(self.key))
        stats = self.cache.stats()
        self.assertEqual(stats["invalidations"], 1)
        self.assertEqual(stats["miss_reasons"], {"integrity_failure": 1})

    def test_inspect_reports_valid_and_tampered_entries(self):
        self.cache.store(self.key, b"ptx")
        report = self.cache.inspect()
        self.assertEqual(report["valid_entries"], 1)
        self.assertEqual(report["invalid_entries"], 0)
        self.cache._artifact_path(self.key).write_bytes(b"tampered")
        report = self.cache.inspect()
        self.assertEqual(report["valid_entries"], 0)
        self.assertEqual(report["entries"][0]["reason"], "integrity_failure")
