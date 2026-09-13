import json
import tempfile
import unittest
from pathlib import Path

from scripts.wsl.validate_serving_protocol import validate


class ServingProtocolTests(unittest.TestCase):
    def _write_protocol(
        self,
        root: Path,
        candidate_values=(200.0, 220.0),
        baseline_values=(100.0, 110.0),
    ) -> Path:
        raw = root / ".kairo-local"
        raw.mkdir()
        records = [{"summary": {"passed": 4, "total": 4}}]
        for value in candidate_values:
            records.append(
                {
                    "config": {
                        "concurrency": 2,
                        "requests": 2,
                        "prompt_tokens_requested": 512,
                        "prompt_tokens_actual": 510.0,
                        "generation_tokens": 128,
                        "generation_tokens_actual": 128.0,
                        "warmup": 1,
                        "disable_thinking": True,
                        "ignore_eos": True,
                        "context_tokens": 1024,
                    },
                    "summary": {"ok": 2, "failed": 0, "output_tokens_per_s": value},
                }
            )
        for name, values in (
            ("candidate.out", candidate_values),
            ("baseline.out", baseline_values),
        ):
            lane_records = [records[0]] + [
                {**record, "summary": {**record["summary"], "output_tokens_per_s": value}}
                for record, value in zip(records[1:], values)
            ]
            (raw / name).write_text(
                "\n".join(json.dumps(record) for record in lane_records), encoding="utf-8"
            )
        protocol = root / "experiments" / "protocols"
        protocol.mkdir(parents=True)
        path = protocol / "serving.yaml"
        path.write_text(
            """
pinned_current_validation:
  workload:
    concurrency: 2
    requests: 2
    prompt_tokens_requested: 512
    prompt_tokens_actual_mean: 510.0
    context_tokens: 1024
    generation_tokens: 128
    generation_tokens_actual: 128.0
    warmup: 1
  graph:
    raw_output: .kairo-local/candidate.out
    throughput_tok_s: [200.0, 220.0]
    median_tok_s: 210.0
    correctness: 4/4
    successful_requests_per_repeat: [2, 2]
  eager_control:
    raw_output: .kairo-local/baseline.out
    throughput_tok_s: [100.0, 110.0]
    median_tok_s: 105.0
    correctness: 4/4
    successful_requests_per_repeat: [2, 2]
  comparison:
    median_throughput_speedup: 2.0
    promotion_gate: true
""",
            encoding="utf-8",
        )
        return path

    def test_protocol_matches_raw_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = validate(self._write_protocol(Path(directory)))
        self.assertTrue(result["valid"])

    def test_protocol_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_protocol(Path(directory), candidate_values=(201.0, 220.0))
            result = validate(path)
        self.assertFalse(result["valid"])
        self.assertFalse(result["lanes"][0]["checks"]["throughput_repeats"])

    def test_pinned_revision_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_protocol(Path(directory))
            text = path.read_text(encoding="utf-8").replace(
                "  workload:\n", "  source_revision: expected-source\n  workload:\n"
            )
            path.write_text(text, encoding="utf-8")
            result = validate(path)
        self.assertFalse(result["valid"])
        self.assertFalse(result["lanes"][0]["checks"]["workload"])

    def test_missing_raw_log_returns_structured_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_protocol(Path(directory))
            (Path(directory) / ".kairo-local" / "candidate.out").unlink()
            result = validate(path)
        self.assertFalse(result["valid"])
        self.assertFalse(result["lanes"][0]["checks"]["raw_exists"])


if __name__ == "__main__":
    unittest.main()
