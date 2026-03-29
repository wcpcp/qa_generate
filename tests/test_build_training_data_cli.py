from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_training_data.py"
EXAMPLE_SCENE = ROOT / "examples" / "scene_metadata_minimal.json"


class BuildTrainingDataCliRobustnessTestCase(unittest.TestCase):
    def test_directory_build_skips_empty_metadata_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_root = Path(tmpdir) / "input"
            output_root = Path(tmpdir) / "output"
            good_scene_dir = input_root / "scene_ok" / "0001"
            bad_scene_dir = input_root / "scene_bad" / "0002"
            good_scene_dir.mkdir(parents=True, exist_ok=True)
            bad_scene_dir.mkdir(parents=True, exist_ok=True)

            (good_scene_dir / "metadata.json").write_text(EXAMPLE_SCENE.read_text(encoding="utf-8"), encoding="utf-8")
            (bad_scene_dir / "metadata.json").write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(input_root),
                    "--output-dir",
                    str(output_root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["scene_count"], 1)
            self.assertEqual(summary["skipped_input_count"], 1)
            self.assertEqual(summary["failed_scene_count"], 0)

            manifest_lines = (output_root / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
            manifest = [json.loads(line) for line in manifest_lines if line.strip()]
            statuses = {item["status"] for item in manifest}
            self.assertIn("prepared", statuses)
            self.assertIn("skipped", statuses)


if __name__ == "__main__":
    unittest.main()
