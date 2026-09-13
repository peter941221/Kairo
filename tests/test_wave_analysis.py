import unittest

from scripts.wsl.analyze_waves import analyze


class WaveAnalysisTests(unittest.TestCase):
    def test_groups_large_ttft_gaps(self):
        result = analyze(
            {
                "config": {"concurrency": 4},
                "samples": [
                    {"request_id": 0, "ok": True, "ttft_ms": 100},
                    {"request_id": 1, "ok": True, "ttft_ms": 150},
                    {"request_id": 2, "ok": True, "ttft_ms": 2500},
                    {"request_id": 3, "ok": True, "ttft_ms": 2600},
                    {"request_id": 4, "ok": False, "ttft_ms": 8000},
                ]
            },
            gap_ms=1000,
        )
        self.assertEqual(result["summary"]["wave_count"], 2)
        self.assertEqual(result["summary"]["wave_sizes"], [2, 2])
        self.assertEqual(result["summary"]["inter_wave_gaps_ms"], [2350])

    def test_empty_result_is_safe(self):
        result = analyze({"samples": []})
        self.assertEqual(result["summary"]["wave_count"], 0)
        self.assertIsNone(result["summary"]["median_ttft_ms"])


if __name__ == "__main__":
    unittest.main()
