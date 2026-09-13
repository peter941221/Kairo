import json
import tempfile
import unittest
from pathlib import Path

from kairo_lab.comparison import compare_logs, render_markdown


def write_log(path: Path, values, *, prompt=512, context=None, model_revision=None, prompts=None, passed=4):
    records = [{"summary": {"passed": passed, "total": 4}}]
    for index, value in enumerate(values):
        repeat_prompt = prompts[index] if prompts is not None else prompt
        config = {
            "concurrency": 16,
            "requests": 16,
            "prompt_tokens_requested": repeat_prompt,
            "generation_tokens": 128,
            "warmup": 1,
            "disable_thinking": True,
            "ignore_eos": True,
        }
        if context is not None:
            config["context_tokens"] = context
        if model_revision is not None:
            config["model_revision"] = model_revision
        records.append(
            {
                "config": config,
                "summary": {"ok": 16, "failed": 0, "output_tokens_per_s": value},
            }
        )
    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")


class ComparisonTests(unittest.TestCase):
    def test_promotion_requires_matching_workload_and_correctness(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.out"
            baseline = Path(directory) / "baseline.out"
            write_log(candidate, [200.0, 220.0])
            write_log(baseline, [100.0, 110.0])
            result = compare_logs(candidate, baseline)
        self.assertTrue(result["workload_match"])
        self.assertEqual(result["throughput_ratio"], 2.0)
        self.assertTrue(result["promotion_gate"])

    def test_drop_first_is_explicit_and_failed_correctness_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.out"
            baseline = Path(directory) / "baseline.out"
            write_log(candidate, [100.0, 220.0], passed=3)
            write_log(baseline, [100.0, 110.0])
            result = compare_logs(candidate, baseline, drop_first=True)
        self.assertTrue(result["drop_first"])
        self.assertFalse(result["correctness_and_success_ok"])
        self.assertFalse(result["promotion_gate"])

    def test_workload_mismatch_blocks_even_with_speedup(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.out"
            baseline = Path(directory) / "baseline.out"
            write_log(candidate, [300.0], prompt=1024)
            write_log(baseline, [100.0], prompt=512)
            result = compare_logs(candidate, baseline)
        self.assertFalse(result["workload_match"])
        self.assertFalse(result["promotion_gate"])

    def test_inconsistent_repeats_block_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.out"
            baseline = Path(directory) / "baseline.out"
            write_log(candidate, [200.0, 220.0], prompts=[512, 1024])
            write_log(baseline, [100.0, 110.0])
            result = compare_logs(candidate, baseline)
        self.assertFalse(result["candidate"]["workload_consistent"])
        self.assertFalse(result["workload_match"])
        self.assertFalse(result["promotion_gate"])

    def test_context_length_is_part_of_workload_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.out"
            baseline = Path(directory) / "baseline.out"
            write_log(candidate, [200.0], context=1024)
            write_log(baseline, [100.0], context=4096)
            result = compare_logs(candidate, baseline)
        self.assertFalse(result["workload_match"])
        self.assertFalse(result["promotion_gate"])

    def test_single_repeat_is_exploratory_not_promotable_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.out"
            baseline = Path(directory) / "baseline.out"
            write_log(candidate, [200.0])
            write_log(baseline, [100.0])
            result = compare_logs(candidate, baseline)
            exploratory = compare_logs(candidate, baseline, minimum_repeats=1)
        self.assertFalse(result["correctness_and_success_ok"])
        self.assertFalse(result["promotion_gate"])
        self.assertTrue(exploratory["correctness_and_success_ok"])
        self.assertTrue(exploratory["promotion_gate"])

    def test_drop_first_requires_two_remaining_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.out"
            baseline = Path(directory) / "baseline.out"
            write_log(candidate, [200.0, 210.0])
            write_log(baseline, [100.0, 105.0])
            result = compare_logs(candidate, baseline, drop_first=True)
        self.assertEqual(result["effective_minimum_repeats"], 3)
        self.assertFalse(result["correctness_and_success_ok"])
        self.assertFalse(result["promotion_gate"])

    def test_markdown_report_contains_gate_and_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.out"
            baseline = Path(directory) / "baseline.out"
            write_log(candidate, [200.0, 220.0])
            write_log(baseline, [100.0, 110.0])
            report = render_markdown(compare_logs(candidate, baseline))
        self.assertIn("Promotion gate: PASS", report)
        self.assertIn("Throughput median (tok/s)", report)
        self.assertIn("TTFT P50 median (ms)", report)
        self.assertIn("Workload identity", report)

    def test_model_revision_is_part_of_workload_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.out"
            baseline = Path(directory) / "baseline.out"
            write_log(candidate, [200.0, 220.0], model_revision="model-a")
            write_log(baseline, [100.0, 110.0], model_revision="model-b")
            result = compare_logs(candidate, baseline)
        self.assertFalse(result["workload_match"])
        self.assertFalse(result["promotion_gate"])
