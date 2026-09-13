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
