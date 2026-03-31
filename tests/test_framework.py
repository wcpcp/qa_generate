from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from erp_data_generation.builders import build_canonical_samples
from erp_data_generation.orchestrator import build_corpus_bundle, build_scene_bundle
from erp_data_generation.pipeline import build_scene_plan, load_scene_metadata
from erp_data_generation.postprocess import build_postprocess_jobs
from erp_data_generation.postprocess_execution import _merge_counting, _merge_text_repackage, derive_execution_context


EXAMPLE_SCENE = ROOT / "examples" / "scene_metadata_minimal.json"
REAL_SCENE = ROOT / "dataset" / "metadata.json"


class FrameworkTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = load_scene_metadata(str(EXAMPLE_SCENE))
        self.plan = build_scene_plan(self.scene)
        self.samples = build_canonical_samples(self.scene, self.plan)

    def test_scene_plan_builds_canonical_samples(self) -> None:
        self.assertGreater(len(self.samples), 0)
        task_families = {sample["task_family"] for sample in self.samples}
        self.assertIn("relative_direction", task_families)
        dense_caption_available = any(
            (entity.semantic.caption_dense or "").strip() and len((entity.semantic.caption_dense or "").split()) >= 12
            for entity in self.scene.entities
        )
        if dense_caption_available:
            self.assertIn("caption", task_families)

    def test_global_tasks_do_not_generate_zero_count(self) -> None:
        for sample in self.samples:
            self.assertNotEqual(sample.get("generation_mode"), "zero_instance_count")

    def test_postprocess_jobs_cover_counting_and_sample_other_tasks(self) -> None:
        scene = self.scene
        samples = self.samples
        if REAL_SCENE.exists():
            scene = load_scene_metadata(str(REAL_SCENE))
            plan = build_scene_plan(scene)
            samples = build_canonical_samples(scene, plan)
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "postprocess_policy.json"
            policy = {
                "default_repackage_probability": 1.0,
                "task_policies": {
                    task_family: {
                        "mode": "counting_visual_correct" if task_family == "counting" else "grounding_repackage",
                        "repackage_probability": 1.0,
                        "requires_visual": task_family == "counting",
                        "fallback_policy": "filter" if task_family == "counting" else "use_canonical",
                    }
                    for task_family in {sample["task_family"] for sample in samples}
                },
            }
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            postprocess = build_postprocess_jobs(scene, samples, policy_path=str(policy_path))
            covered_task_families = {job["task_family"] for job in postprocess["jobs"]}
            covered_task_families.update(item["task_family"] for item in postprocess["skipped_samples"])
            if any(sample["task_family"] == "counting" for sample in samples):
                self.assertIn("counting", covered_task_families)

    def test_text_repackage_accepts_question_and_full_answer(self) -> None:
        sample = next(sample for sample in self.samples if sample["task_family"] == "relative_direction")
        merged, validation = _merge_text_repackage(
            sample,
            {
                "question": "Where is object A relative to object B in the panorama?",
                "full_answer": "It is to the left of the reference object.",
            },
        )
        self.assertEqual(validation["status"], "accepted")
        self.assertIn("final_question", merged)

    def test_counting_merge_can_correct_answer(self) -> None:
        if REAL_SCENE.exists():
            scene = load_scene_metadata(str(REAL_SCENE))
            plan = build_scene_plan(scene)
            samples = build_canonical_samples(scene, plan)
        else:
            samples = self.samples
        sample = next((sample for sample in samples if sample["task_family"] == "counting"), None)
        if sample is None:
            sample = next((sample for sample in self.samples if sample["task_family"] == "counting"), None)
        if sample is None:
            self.skipTest("No counting sample available in the current test metadata.")
        merged, validation = _merge_counting(
            sample,
            {
                "decision": "correct",
                "verified_count": "5",
                "question": "How many chairs are visible in the panorama?",
                "full_answer": "After checking the full panorama carefully, there are 5 chairs visible.",
            },
        )
        self.assertEqual(validation["status"], "accepted")
        self.assertEqual(merged["final_answer"], "After checking the full panorama carefully, there are 5 chairs visible.")
        self.assertTrue(merged["llm_repackaged"])

    def test_scene_bundle_contains_postprocess_plan(self) -> None:
        bundle = build_scene_bundle(str(EXAMPLE_SCENE))
        self.assertIn("postprocess_plan", bundle)
        self.assertIn("canonical_samples", bundle)
        self.assertIn("prepared_canonical_samples", bundle)

    def test_prepared_canonical_samples_can_restore_execution_context(self) -> None:
        bundle = build_scene_bundle(str(EXAMPLE_SCENE))
        passthrough_ids, filtered_ids = derive_execution_context(bundle["prepared_canonical_samples"])
        dispositions = {sample["postprocess_disposition"] for sample in bundle["prepared_canonical_samples"]}
        self.assertIn("passthrough", dispositions)
        self.assertEqual(sorted(filtered_ids), sorted(bundle["postprocess_plan"]["filtered_sample_ids"]))
        self.assertEqual(sorted(passthrough_ids), sorted(bundle["postprocess_plan"]["passthrough_sample_ids"]))

    def test_corpus_bundle_builds(self) -> None:
        bundle = build_corpus_bundle([str(EXAMPLE_SCENE)])
        self.assertIn("scenes", bundle)
        self.assertEqual(bundle["summary"]["scene_count"], 1)

    def test_seam_continuity_uses_new_subtypes(self) -> None:
        seam_samples = [sample for sample in self.samples if sample["task_family"] == "seam_continuity"]
        self.assertTrue(seam_samples)
        allowed_modes = {
            "nearest_neighbor",
            "relative_direction",
            "dedup_count",
            "structure_continuity",
            "same_entity_judgement",
        }
        for sample in seam_samples:
            self.assertIn(sample.get("generation_mode"), allowed_modes)
            metadata = sample.get("metadata", {})
            self.assertEqual(metadata.get("seam_mode"), sample.get("generation_mode"))


if __name__ == "__main__":
    unittest.main()
