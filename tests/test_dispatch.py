import tempfile
import unittest
from pathlib import Path

from kairo_lab.dispatch import DispatchError, RuntimeDispatcher
from kairo_lab.cache import RuntimeKernelCache

from test_blueprint import valid_blueprint


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.builds = 0

    def dispatcher(self, launcher):
        def builder(report, shape):
            self.builds += 1
            return f"artifact:{report['blueprint_hash']}:{shape}".encode()

        return RuntimeDispatcher(
            RuntimeKernelCache(Path(self.directory.name)), builder, launcher
        )

    def kwargs(self):
        return {
            "driver_version": "596.36",
            "gpu_capability": "sm120",
        }

    def test_dispatch_builds_once_then_reuses_cache(self):
        dispatcher = self.dispatcher(lambda artifact, report: artifact.decode())
        first = dispatcher.dispatch(valid_blueprint(), (128, 128, 64), **self.kwargs())
        second = dispatcher.dispatch(valid_blueprint(), (128, 128, 64), **self.kwargs())
        self.assertEqual(first.backend, "kairo")
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(self.builds, 1)
        self.assertIsNotNone(first.artifact_ms)
        self.assertIsNotNone(first.launch_ms)
        self.assertGreaterEqual(first.artifact_ms, 0.0)
        self.assertGreaterEqual(first.launch_ms, 0.0)
        self.assertIn("run_correctness_gate", first.plan["steps"])

    def test_launch_failure_uses_explainable_fallback(self):
        dispatcher = self.dispatcher(lambda _artifact, _report: (_ for _ in ()).throw(RuntimeError("bad launch")))
        result = dispatcher.dispatch(
            valid_blueprint(),
            (128, 128, 64),
            fallback=lambda exc: f"reference:{exc}",
            **self.kwargs(),
        )
        self.assertEqual(result.backend, "fallback")
        self.assertEqual(result.output, "reference:bad launch")
        self.assertIn("launch:RuntimeError", result.fallback_reason)

    def test_build_failure_requires_or_uses_fallback(self):
        dispatcher = RuntimeDispatcher(
            RuntimeKernelCache(Path(self.directory.name)),
            lambda _report, _shape: (_ for _ in ()).throw(ValueError("compile rejected")),
            lambda _artifact, _report: "unreachable",
        )
        with self.assertRaises(DispatchError):
            dispatcher.dispatch(valid_blueprint(), (128, 128, 64), **self.kwargs())
        result = dispatcher.dispatch(
            valid_blueprint(),
            (128, 128, 64),
            fallback=lambda exc: f"reference:{exc}",
            **self.kwargs(),
        )
        self.assertEqual(result.backend, "fallback")
        self.assertIn("build:ValueError", result.fallback_reason)
