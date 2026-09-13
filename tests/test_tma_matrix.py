import unittest

from scripts.wsl.analyze_tma_wmma_matrix import summarize


class TmaMatrixTests(unittest.TestCase):
    def test_winner_uses_median_and_correctness(self):
        result = summarize(
            [
                {"shape": [1024, 1024, 1024], "variant": "single", "tma_gflops": 40, "tma_ok": True, "max_abs_error_vs_cublas": 0},
                {"shape": [1024, 1024, 1024], "variant": "single", "tma_gflops": 60, "tma_ok": True, "max_abs_error_vs_cublas": 0},
                {"shape": [1024, 1024, 1024], "variant": "m128", "tma_gflops": 50, "tma_ok": True, "max_abs_error_vs_cublas": 0},
                {"shape": [1024, 1024, 1024], "variant": "m128", "tma_gflops": 55, "tma_ok": False, "max_abs_error_vs_cublas": 0.5},
            ]
        )
        self.assertEqual(result["winners"][0]["winner"], "single")
        single = next(item for item in result["variants"] if item["variant"] == "single")
        self.assertEqual(single["median_gflops"], 50)


if __name__ == "__main__":
    unittest.main()
