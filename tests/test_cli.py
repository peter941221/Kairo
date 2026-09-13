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
        graph = cli.recommend_runtime("qwen38", concurrency=8, prompt_tokens=256)
        self.assertEqual(graph["linear_backend"], "cutlass")
        self.assertEqual(graph["cudagraph_mode"], "FULL_DECODE_ONLY")
        self.assertEqual(graph["confidence"], "measured_pilot")
        graph_c32 = cli.recommend_runtime("qwen38", concurrency=32, prompt_tokens=256)
        self.assertEqual(graph_c32["profile"], "qwen38-vllm-nightly-cutlass-full-decode-graph-c32")
        self.assertEqual(graph_c32["max_num_seqs"], 32)
        self.assertEqual(graph_c32["confidence"], "measured_repeated")
        graph_c32_p512 = cli.recommend_runtime(
            "qwen38", concurrency=32, prompt_tokens=512,
            context_tokens=1024, generation_tokens=128,
        )
        self.assertIn("graph-c32-p512", graph_c32_p512["profile"])
        graph_c16 = cli.recommend_runtime("qwen38", concurrency=16, prompt_tokens=256)
        self.assertEqual(graph_c16["profile"], "qwen38-vllm-nightly-cutlass-full-decode-graph-c16")
        self.assertEqual(graph_c16["cudagraph_mode"], "FULL_DECODE_ONLY")
        self.assertEqual(graph_c16["confidence"], "measured_repeated")
        graph_p512 = cli.recommend_runtime(
            "qwen38", concurrency=16, prompt_tokens=512,
            context_tokens=1024, generation_tokens=128,
        )
        self.assertIn("graph-c16-p512", graph_p512["profile"])
        eager_p512 = cli.recommend_runtime(
            "qwen38", concurrency=16, prompt_tokens=512,
            context_tokens=4096, generation_tokens=256,
        )
        self.assertEqual(eager_p512["profile"], "qwen38-vllm-nightly-cutlass-c16")
        unknown_graph = cli.recommend_runtime(
            "qwen38", concurrency=32, prompt_tokens=256,
            context_tokens=4096, generation_tokens=128,
        )
        self.assertEqual(unknown_graph["backend"], "manual")
        with self.assertRaises(ValueError):
            cli.recommend_runtime("qwen38", 16, 512, context_tokens=1024)
        short = cli.recommend_runtime("qwen38", concurrency=16, prompt_tokens=512)
        self.assertEqual(short["backend"], "vllm-nightly")
        self.assertEqual(short["linear_backend"], "cutlass")
        self.assertEqual(short["confidence"], "measured_repeated")
        high = cli.recommend_runtime("qwen38", concurrency=16, prompt_tokens=2048)
        self.assertEqual(high["backend"], "vllm-nightly")
        self.assertEqual(high["linear_backend"], "cutlass")
        self.assertEqual(high["confidence"], "measured_repeated")
        low = cli.recommend_runtime("qwen38", concurrency=4, prompt_tokens=512)
        self.assertEqual(low["backend"], "sglang")
        unknown = cli.recommend_runtime("qwen38", concurrency=8, prompt_tokens=512)
        self.assertEqual(unknown["backend"], "manual")
        self.assertEqual(unknown["confidence"], "unvalidated")

    def test_runtime_policy_supports_measured_qwen3_8b_control(self):
        graph = cli.recommend_runtime(
            "qwen3_8b", concurrency=16, prompt_tokens=512,
            context_tokens=1024, generation_tokens=128,
        )
        self.assertEqual(graph["backend"], "vllm-nightly")
        self.assertEqual(graph["cudagraph_mode"], "FULL_DECODE_ONLY")
        self.assertEqual(graph["confidence"], "measured_repeated")
        graph_c32 = cli.recommend_runtime(
            "qwen3_8b", concurrency=32, prompt_tokens=512,
            context_tokens=1024, generation_tokens=128,
        )
        self.assertIn("graph-c32-p512", graph_c32["profile"])
        eager = cli.recommend_runtime(
            "qwen3_8b", concurrency=16, prompt_tokens=512,
            context_tokens=4096, generation_tokens=256,
        )
        self.assertEqual(eager["profile"], "qwen3-8b-vllm-nightly-cutlass-c16")
        eager_c32 = cli.recommend_runtime(
            "qwen3_8b", concurrency=32, prompt_tokens=512,
            context_tokens=4096, generation_tokens=256,
        )
        self.assertEqual(eager_c32["profile"], "qwen3-8b-vllm-nightly-cutlass-c32")
        long_graph = cli.recommend_runtime(
            "qwen3_8b", concurrency=16, prompt_tokens=2048,
            context_tokens=4096, generation_tokens=128,
        )
        self.assertIn("p2048-4k", long_graph["profile"])
        self.assertEqual(long_graph["max_model_len"], 4096)
        unknown = cli.recommend_runtime("qwen3_8b", concurrency=8, prompt_tokens=512)
        self.assertEqual(unknown["backend"], "manual")

    def test_gemm_policy_promotes_only_measured_winners(self):
        measured = cli.recommend_gemm(4096, 4096, 4096)
        self.assertEqual(measured["variant"], "m256")
        self.assertEqual(measured["confidence"], "measured_shape")
        expanded = cli.recommend_gemm(512, 1024, 1024)
        self.assertEqual(expanded["variant"], "m128")
        larger = cli.recommend_gemm(2048, 2048, 2048)
        self.assertEqual(larger["variant"], "m256")
        unknown = cli.recommend_gemm(3072, 2048, 1024)
        self.assertEqual(unknown["variant"], "single")
        self.assertEqual(unknown["confidence"], "aligned_control_unvalidated_shape")
        unaligned = cli.recommend_gemm(1000, 1024, 1024)
        self.assertEqual(unaligned["variant"], "fallback")

    def test_nvfp4_policy_only_promotes_measured_graph_buckets(self):
        measured = cli.recommend_nvfp4_pipeline(128, 4096, 4096)
        self.assertEqual(measured["strategy"], "cuda_graph_shape_bucket")
        self.assertEqual(measured["confidence"], "measured_repeated")
        aligned_unknown = cli.recommend_nvfp4_pipeline(64, 4096, 4096)
        self.assertEqual(aligned_unknown["strategy"], "regular_nvfp4_pipeline")
        unaligned = cli.recommend_nvfp4_pipeline(32, 4100, 4096)
        self.assertEqual(unaligned["strategy"], "fallback")
