import unittest

from scripts.wsl.analyze_nvfp4_graph import summarize


class Nvfp4GraphTests(unittest.TestCase):
    def test_recommends_graph_only_when_correct(self):
        records = [
            {
                "shape": [32, 4096, 4096],
                "quantized_pipeline_ms": 10.0,
                "cuda_graph_pipeline_ms": 7.0,
                "cuda_graph_max_abs_error_vs_pipeline": 0.0,
            },
            {
                "shape": [32, 4096, 4096],
                "quantized_pipeline_ms": 12.0,
                "cuda_graph_pipeline_ms": 8.0,
                "cuda_graph_max_abs_error_vs_pipeline": 0.0,
            },
        ]
        result = summarize(records)[0]
        self.assertEqual(result["recommendation"], "cuda_graph")
        self.assertAlmostEqual(result["pipeline_reduction_percent"], 31.8181818)

    def test_rejects_incorrect_graph(self):
        result = summarize(
            [
                {
                    "shape": [1, 4096, 4096],
                    "quantized_pipeline_ms": 10.0,
                    "cuda_graph_pipeline_ms": 1.0,
                    "cuda_graph_max_abs_error_vs_pipeline": 0.1,
                }
            ]
        )[0]
        self.assertFalse(result["graph_correct"])
        self.assertEqual(result["recommendation"], "regular_pipeline")
