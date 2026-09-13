import unittest

from scripts.wsl.analyze_memory_bandwidth import summarize


class MemoryBandwidthTests(unittest.TestCase):
    def test_median_and_range_use_repeated_records(self):
        result = summarize(
            [
                {"read_write_gbps": 1400.0},
                {"read_write_gbps": 1500.0},
                {"read_write_gbps": 1600.0},
            ]
        )
        self.assertEqual(result["read_write_gbps_median"], 1500.0)
        self.assertEqual(result["relative_range_percent_of_median"], 13.33)
        self.assertEqual(result["gate"], "passed")

    def test_insufficient_repeats_fail_closed(self):
        with self.assertRaises(ValueError):
            summarize([{"read_write_gbps": 1500.0}])


if __name__ == "__main__":
    unittest.main()
