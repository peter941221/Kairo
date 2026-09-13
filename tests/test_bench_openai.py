import argparse
import unittest
from unittest.mock import patch

from scripts.wsl import bench_openai


class BenchOpenAiTests(unittest.TestCase):
    def test_records_actual_token_counts_in_workload_config(self):
        args = argparse.Namespace(
            warmup=0,
            concurrency=2,
            requests=2,
            base_url="http://unused",
            model="smoke",
            lane="decode",
            prompt_tokens=512,
            generation_tokens=128,
            context_tokens=1024,
            model_revision="abc123",
            disable_thinking=True,
            ignore_eos=True,
            timeout=1.0,
        )
        samples = [
            bench_openai.Sample(0, True, prompt_tokens=510, completion_tokens=128),
            bench_openai.Sample(1, True, prompt_tokens=510, completion_tokens=128),
        ]
        with patch.object(bench_openai, "_one", side_effect=samples):
            result = bench_openai.run(args)
        self.assertEqual(result["config"]["prompt_tokens_actual"], 510)
        self.assertEqual(result["config"]["generation_tokens_actual"], 128)
        self.assertEqual(result["config"]["context_tokens"], 1024)
        self.assertEqual(result["config"]["model_revision"], "abc123")


if __name__ == "__main__":
    unittest.main()
