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
