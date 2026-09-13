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
