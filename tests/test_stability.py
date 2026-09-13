import json
import tempfile
import unittest
from pathlib import Path

from kairo_lab import stability


class StabilityTests(unittest.TestCase):
    def test_summarizes_repeated_json_and_ansi(self):
        payload = (
            '{"summary":{"passed":4,"total":4}}\n'
            '{"summary":{"ok":2,"failed":0,"output_tokens_per_s":100.0}}\n'
            '\x1b[4C{"summary":{"ok":2,"failed":0,"output_tokens_per_s":120.0}}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.out"
            path.write_text(payload, encoding="utf-8")
            result = stability.summarize(path)
        self.assertEqual(result["bench_repeats"], 2)
        self.assertEqual(result["throughput_median_tok_s"], 110.0)
        self.assertTrue(result["all_requests_successful"])

    def test_summarizes_latency_across_repeats(self):
        payload = "\n".join(
            json.dumps(
                {
                    "config": {"concurrency": 2},
                    "summary": {
                        "ok": 2,
                        "failed": 0,
                        "output_tokens_per_s": 100.0,
                        "ttft_p50_ms": value,
                        "ttft_p99_ms": value + 10,
                        "total_p50_ms": value + 100,
                        "total_p99_ms": value + 200,
                    },
                }
            )
            for value in (20.0, 30.0, 40.0)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.out"
            path.write_text(payload, encoding="utf-8")
            result = stability.summarize(path)
        self.assertEqual(result["ttft_p50_ms_per_repeat"], [20.0, 30.0, 40.0])
        self.assertEqual(result["ttft_p50_ms_median"], 30.0)
        self.assertEqual(result["total_p99_ms_median"], 230.0)

    def test_missing_correctness_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.out"
            path.write_text('{"summary":{"ok":1,"failed":0,"output_tokens_per_s":1}}', encoding="utf-8")
            result = stability.summarize(path)
        self.assertIsNone(result["correctness"])

    def test_workload_consistency_is_reported_per_repeat(self):
        payload = (
            '{"config":{"concurrency":2,"prompt_tokens_requested":512},'
            '"summary":{"ok":2,"failed":0,"output_tokens_per_s":100.0}}\n'
            '{"config":{"concurrency":2,"prompt_tokens_requested":1024},'
            '"summary":{"ok":2,"failed":0,"output_tokens_per_s":110.0}}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.out"
            path.write_text(payload, encoding="utf-8")
            result = stability.summarize(path)
        self.assertEqual(result["workload"], {"concurrency": 2, "prompt_tokens_requested": 512})
        self.assertFalse(result["workload_consistent"])

    def test_missing_repeat_config_is_not_considered_consistent(self):
        payload = (
            '{"config":{"concurrency":2},'
            '"summary":{"ok":2,"failed":0,"output_tokens_per_s":100.0}}\n'
            '{"summary":{"ok":2,"failed":0,"output_tokens_per_s":110.0}}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.out"
            path.write_text(payload, encoding="utf-8")
            result = stability.summarize(path)
        self.assertFalse(result["workload_consistent"])
