import json
import tempfile
import unittest
from pathlib import Path

from kairo_lab import cli


class CliTests(unittest.TestCase):
    def test_environment_has_reproducibility_fields(self):
        data = cli.environment()
        self.assertTrue(data["timestamp_utc"])
        self.assertIn("python", data)

    def test_init_run_writes_contract(self):
        original_root = cli.ROOT
        with tempfile.TemporaryDirectory() as temporary_directory:
            cli.ROOT = Path(temporary_directory)
            try:
                output = cli.init_run("decode")
                data = json.loads(output.read_text(encoding="utf-8"))
            finally:
                cli.ROOT = original_root
        self.assertEqual(data["lane"], "decode")
        self.assertEqual(
            data["result_contract"]["correctness_status"],
            "required_before_performance_claim",
        )

    def test_profile_recommender_stays_inside_measured_shapes(self):
        high = cli.recommend_profile("qwen38", concurrency=16, prompt_tokens=512)
        self.assertEqual(high["mamba_full_memory_ratio"], 8.0)
        self.assertEqual(high["confidence"], "measured_c16")

        long_prompt = cli.recommend_profile("qwen38", concurrency=16, prompt_tokens=2048)
        self.assertEqual(long_prompt["mamba_full_memory_ratio"], 4.59)
        self.assertEqual(long_prompt["confidence"], "baseline_or_unvalidated")

    def test_runtime_policy_only_selects_measured_cells(self):
        high = cli.recommend_runtime("qwen38", concurrency=16, prompt_tokens=2048)
        self.assertEqual(high["backend"], "vllm-nightly")
        low = cli.recommend_runtime("qwen38", concurrency=4, prompt_tokens=512)
        self.assertEqual(low["backend"], "sglang")
        unknown = cli.recommend_runtime("qwen38", concurrency=8, prompt_tokens=512)
        self.assertEqual(unknown["backend"], "manual")
        self.assertEqual(unknown["confidence"], "unvalidated")
