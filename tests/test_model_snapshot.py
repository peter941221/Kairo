import tempfile
import unittest
from pathlib import Path

from scripts.wsl.verify_model_snapshot import verify


class ModelSnapshotTests(unittest.TestCase):
    def test_verifies_single_revision_across_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            metadata = model / ".cache" / "huggingface" / "download"
            metadata.mkdir(parents=True)
            (metadata / "config.json.metadata").write_text("abc123\netag-a\n", encoding="utf-8")
            (metadata / "weights.metadata").write_text("abc123\netag-b\n", encoding="utf-8")
            result = verify(model, "abc123")
        self.assertTrue(result["verified"])
        self.assertEqual(result["found_revisions"], ["abc123"])

    def test_revision_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            metadata = model / ".cache" / "huggingface" / "download"
            metadata.mkdir(parents=True)
            (metadata / "config.json.metadata").write_text("actual\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(model, "expected")


if __name__ == "__main__":
    unittest.main()
