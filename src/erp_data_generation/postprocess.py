from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .pipeline import NEGATIVE_EXISTENCE_CANDIDATES
from .schemas import Entity, SceneMetadata
from .visual_context import (
    build_entity_visual_context,
    build_entity_visual_context_from_spec,
    build_four_face_visual_context,
    build_four_face_visual_context_from_path,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSTPROCESS_POLICY_PATH = ROOT / "config" / "postprocess_policy.json"


def load_postprocess_policy(policy_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(policy_path) if policy_path else DEFAULT_POSTPROCESS_POLICY_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def build_postprocess_jobs(
    scene: SceneMetadata,
    canonical_samples: Iterable[Dict[str, Any]],
    *,
    policy_path: Optional[str] = None,
    repackage_probability: Optional[float] = None,
) -> Dict[str, Any]:
    policy = load_postprocess_policy(policy_path)
    task_policies = policy.get("task_policies", {})
    entity_by_id = {entity.entity_id: entity for entity in scene.entities}

    jobs: List[Dict[str, Any]] = []
    passthrough_samples: List[str] = []
    filtered_samples: List[str] = []
    skipped_samples: List[Dict[str, Any]] = []

    for sample in canonical_samples:
        task_family = sample["task_family"]
        task_policy = task_policies.get(task_family)
        if not task_policy:
            passthrough_samples.append(sample["sample_id"])
            continue

        effective_probability = (
            float(repackage_probability)
            if repackage_probability is not None and task_family != "counting"
            else float(task_policy.get("repackage_probability", policy.get("default_repackage_probability", 0.4)))
        )
        if task_family != "counting" and not _should_select_sample(sample["sample_id"], effective_probability):
            passthrough_samples.append(sample["sample_id"])
            skipped_samples.append(
                {
                    "sample_id": sample["sample_id"],
                    "task_family": task_family,
                    "reason": "not_selected_for_postprocess",
                }
            )
            continue

        entities = [entity_by_id[entity_id] for entity_id in sample.get("entity_ids", []) if entity_id in entity_by_id]
        job = _build_single_job(scene, sample, entities, task_policy)
        if job["requires_visual"] and not job["visual_assets"]["image_available"]:
            skipped_samples.append(
                {
                    "sample_id": sample["sample_id"],
                    "task_family": task_family,
                    "reason": "missing_visual_asset",
                    "fallback_policy": job["fallback_policy"],
                }
            )
            if job["fallback_policy"] == "use_canonical":
                passthrough_samples.append(sample["sample_id"])
            elif job["fallback_policy"] == "filter":
                filtered_samples.append(sample["sample_id"])
            continue
        jobs.append(job)

    return {
        "policy": policy,
        "summary": {
            "scene_id": scene.scene_id,
            "job_count": len(jobs),
            "passthrough_count": len(passthrough_samples),
            "filtered_count": len(filtered_samples),
            "skipped_count": len(skipped_samples),
        },
        "jobs": jobs,
        "passthrough_sample_ids": passthrough_samples,
        "filtered_sample_ids": filtered_samples,
        "skipped_samples": skipped_samples,
    }


def _build_single_job(
    scene: SceneMetadata,
    sample: Dict[str, Any],
    entities: List[Entity],
    task_policy: Dict[str, Any],
) -> Dict[str, Any]:
    mode = _resolve_mode(sample, task_policy["mode"])
    visual_assets = _visual_assets(scene, sample, entities, mode)
    facts = _postprocess_facts(scene, sample, entities, mode)
    requires_visual = bool(task_policy.get("requires_visual", False))
    prompt = _render_prompt(mode, sample, facts, visual_assets)
    schema = _expected_output_schema(mode)
    return {
        "job_id": f"{sample['sample_id']}:postprocess",
        "scene_id": scene.scene_id,
        "sample_id": sample["sample_id"],
        "task_family": sample["task_family"],
        "mode": mode,
        "requires_visual": requires_visual,
        "fallback_policy": task_policy.get("fallback_policy", "use_canonical"),
        "facts": facts,
        "visual_assets": visual_assets,
        "prompt_text": prompt,
        "expected_output": schema,
    }


def _resolve_mode(sample: Dict[str, Any], base_mode: str) -> str:
    if base_mode == "existence_repackage":
        if sample.get("generation_mode") == "negative_existence_label":
            return "existence_negative_repackage"
        return "existence_positive_repackage"
    return base_mode


def _should_select_sample(sample_id: str, probability: float) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return bucket < probability


def _postprocess_facts(scene: SceneMetadata, sample: Dict[str, Any], entities: List[Entity], mode: str) -> Dict[str, Any]:
    metadata = dict(sample.get("metadata", {}))
    facts: Dict[str, Any] = {"current_question": sample["canonical_question"]}

    if mode == "caption_visual_refine":
        entity = entities[0]
        facts.update(
            {
                "target": {
                    "bfov": metadata.get("bfov"),
                    "reground_query": entity.semantic.reground_query,
                    "caption_brief": entity.semantic.caption_brief,
                    "caption_dense": entity.semantic.caption_dense,
                }
            }
        )
    elif mode == "counting_visual_correct":
        facts.update(
            {
                "query_target": metadata.get("query_target"),
                "canonical_count": sample["canonical_answer"],
                "scan_hint": (
                    "Scan the ERP image from left to right once, be careful at the left/right seam, "
                    "and avoid double-counting partially visible duplicates."
                ),
            }
        )
    elif mode == "existence_positive_repackage":
        facts.update({"query_target": metadata.get("query_target"), "canonical_truth": "yes"})
    elif mode == "existence_negative_repackage":
        present_labels = sorted({entity.label for entity in scene.entities})
        facts.update(
            {
                "present_labels": present_labels,
                "negative_candidate_pool": [item for item in NEGATIVE_EXISTENCE_CANDIDATES if item not in present_labels],
                "seed_negative_target": metadata.get("query_target"),
                "canonical_truth": "no",
            }
        )
    elif mode == "grounding_repackage":
        target: Dict[str, Any] = {"label": metadata.get("target_label"), "entity_ref": metadata.get("entity_ref")}
        if sample.get("generation_mode") != "angles_only":
            target["bbox_norm_1000"] = metadata.get("bbox_norm_1000")
        if sample.get("generation_mode") != "bbox_only":
            target["bfov"] = metadata.get("bfov")
        facts.update({"target": target, "grounding_mode": sample.get("generation_mode")})
    elif mode == "direct_direction_repackage":
        target: Dict[str, Any] = {"label": metadata.get("target_label"), "entity_ref": metadata.get("entity_ref")}
        if sample.get("generation_mode") == "precise_bfov":
            target["bfov"] = metadata.get("bfov")
        facts.update(
            {
                "target": target,
                "direction_mode": sample.get("generation_mode"),
                "canonical_truth": sample["canonical_answer"],
            }
        )
    elif mode == "relative_direction_repackage":
        facts.update(
            {
                "reference": _entity_relation_stub(entities[0], metadata.get("entity_a_ref"), metadata.get("yaw_a_deg"), metadata.get("bfov_a")),
                "target": _entity_relation_stub(entities[1], metadata.get("entity_b_ref"), metadata.get("yaw_b_deg"), metadata.get("bfov_b")),
                "delta_yaw_deg": round(
                    ((entities[1].lon_deg % 360.0) - (entities[0].lon_deg % 360.0) + 180.0) % 360.0 - 180.0,
                    2,
                ),
                "relation_space": ["right", "back-right", "opposite", "back-left", "left"],
                "canonical_relation": sample["canonical_answer"],
            }
        )
    elif mode == "view_transform_repackage":
        if sample.get("generation_mode") == "object_conditioned_reorientation":
            facts.update(
                {
                    "facing_target": _entity_relation_stub(
                        entities[0],
                        metadata.get("facing_ref"),
                        metadata.get("facing_yaw_deg"),
                        metadata.get("facing_bfov"),
                    ),
                    "query_target": _entity_relation_stub(
                        entities[1],
                        metadata.get("target_ref"),
                        metadata.get("target_yaw_deg"),
                        metadata.get("target_bfov"),
                    ),
                    "delta_yaw_deg": metadata.get("delta_yaw_deg"),
                    "relation_space": ["right", "back-right", "behind", "back-left", "left"],
                    "transform_mode": "object_conditioned_reorientation",
                    "canonical_direction": sample["canonical_answer"],
                }
            )
        else:
            facts.update(
                {
                    "target": {
                        "label": metadata.get("target_label"),
                        "entity_ref": metadata.get("entity_ref"),
                        "bfov": metadata.get("bfov"),
                    },
                    "rotation": {
                        "angle_deg": metadata.get("turn_angle_deg"),
                        "direction": metadata.get("rotation_direction"),
                    },
                    "delta_after_rotation_deg": metadata.get("delta_after_rotation_deg"),
                    "original_absolute_sector": _coarse_direction_from_yaw(float(metadata.get("yaw_deg"))) if metadata.get("yaw_deg") is not None else None,
                    "relation_space": ["right", "back-right", "behind", "back-left", "left"],
                    "transform_mode": "camera_rotation_transform",
                    "canonical_direction": sample["canonical_answer"],
                }
            )
    elif mode == "distance_estimation_repackage":
        if sample.get("generation_mode") == "candidate_nearest_choice":
            facts.update(
                {
                    "reference_label": metadata.get("reference_label"),
                    "reference_ref": metadata.get("reference_ref"),
                    "candidate_objects": _candidate_ref_list(metadata.get("candidate_labels", []), metadata.get("candidate_refs", {})),
                    "distance_measure": metadata.get("distance_measure"),
                    "candidate_distances_m": metadata.get("candidate_distances_m"),
                    "nearest_candidate": sample["canonical_answer"],
                }
            )
        elif sample.get("generation_mode") == "observer_nearest_choice":
            facts.update(
                {
                    "reference_label": "observer",
                    "candidate_objects": _candidate_ref_list(metadata.get("candidate_labels", []), metadata.get("candidate_refs", {})),
                    "distance_measure": metadata.get("distance_measure"),
                    "candidate_depths_m": metadata.get("candidate_depths_m"),
                    "nearest_candidate": sample["canonical_answer"],
                }
            )
        else:
            facts.update(
                {
                    "target_label": metadata.get("target_label"),
                    "target_ref": metadata.get("entity_ref"),
                    "target_bfov": metadata.get("bfov"),
                    "depth_m": metadata.get("depth_m"),
                }
            )
    elif mode == "relative_3d_position_repackage":
        entity_a, entity_b = entities
        xyz_a = entity_a.resolved_xyz_camera
        xyz_b = entity_b.resolved_xyz_camera
        delta_xyz = None
        if xyz_a is not None and xyz_b is not None:
            delta_xyz = [
                round(float(xyz_a[0]) - float(xyz_b[0]), 3),
                round(float(xyz_a[1]) - float(xyz_b[1]), 3),
                round(float(xyz_a[2]) - float(xyz_b[2]), 3),
            ]
        facts.update(
            {
                "entity_a": _entity_3d_stub(entity_a, metadata.get("entity_a_ref")),
                "entity_b": _entity_3d_stub(entity_b, metadata.get("entity_b_ref")),
                "delta_xyz_m": delta_xyz,
                "canonical_relation": sample["canonical_answer"],
                "geometry_source": sample.get("geometry_source", "explicit_xyz"),
            }
        )
    elif mode == "seam_continuity_repackage":
        entity = entities[0]
        seam_mode = sample.get("generation_mode")
        facts.update({"target_label": entity.label, "target_bfov": metadata.get("bfov"), "seam_mode": seam_mode})
        if seam_mode == "counterpart_boundary_side":
            facts["fragment_side"] = metadata.get("hinted_boundary_side")
            facts["canonical_side"] = sample["canonical_answer"]
        elif seam_mode == "wrap_explanation":
            facts["canonical_explanation"] = sample["canonical_answer"]
        else:
            facts["canonical_truth"] = sample["canonical_answer"]
    elif mode == "polar_distortion_awareness_repackage":
        entity = entities[0]
        abs_lat = abs(float(entity.lat_deg))
        facts.update(
            {
                "target_label": entity.label,
                "target_bfov": metadata.get("bfov"),
                "latitude_band": "strong_polar" if abs_lat >= 60.0 else "high_latitude",
                "true_shape": sample["canonical_answer"],
            }
        )
    else:
        facts["metadata"] = metadata
    return facts


def _entity_relation_stub(
    entity: Entity,
    entity_ref: Optional[str] = None,
    yaw_deg: Optional[float] = None,
    bfov: Optional[Any] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"label": entity.label}
    if entity_ref:
        payload["entity_ref"] = entity_ref
    if yaw_deg is not None:
        payload["yaw_deg"] = yaw_deg
    if bfov is not None:
        payload["bfov"] = bfov
    return payload


def _entity_3d_stub(entity: Entity, entity_ref: Optional[str] = None) -> Dict[str, Any]:
    payload = {"label": entity.label}
    if entity_ref:
        payload["entity_ref"] = entity_ref
    xyz = entity.resolved_xyz_camera
    if xyz is not None:
        payload["xyz_camera_m"] = [round(float(value), 3) for value in xyz]
    return payload


def _candidate_ref_list(candidate_labels: List[str], candidate_refs: Dict[str, str]) -> List[Dict[str, Any]]:
    return [
        {
            "label": label,
            "entity_ref": candidate_refs.get(label, label),
        }
        for label in candidate_labels
    ]


def _coarse_direction_from_yaw(yaw_deg: float) -> str:
    sectors = [
        "front",
        "front-right",
        "right",
        "back-right",
        "back",
        "back-left",
        "left",
        "front-left",
    ]
    normalized = yaw_deg % 360.0
    index = int(((normalized + 22.5) % 360.0) // 45.0)
    return sectors[index]


def _visual_assets(scene: SceneMetadata, sample: Dict[str, Any], entities: List[Entity], mode: str) -> Dict[str, Any]:
    erp_path = Path(scene.erp_image_path) if scene.erp_image_path else None
    image_available = bool(erp_path and erp_path.exists())
    assets: Dict[str, Any] = {
        "erp_image_path": str(erp_path) if erp_path else "",
        "image_available": image_available,
        "preferred_visual_context": "erp_full",
        "entity_regions": [
            {
                "entity_id": entity.entity_id,
                "label": entity.label,
                "bbox_erp": entity.bbox_erp,
                "representative_view_id": entity.representative_view_id,
            }
            for entity in entities
        ],
        "task_family": sample["task_family"],
    }
    if mode == "caption_visual_refine" and entities:
        visual_context = build_entity_visual_context(scene, entities[0])
        assets.update(visual_context)
    elif mode == "counting_visual_correct":
        assets.update(
            {
                "mode": "erp_four_faces_deferred",
                "four_face_spec": {
                    "scene_id": scene.scene_id,
                    "erp_image_path": str(erp_path) if erp_path else "",
                },
            }
        )
    return assets


def _render_prompt(mode: str, sample: Dict[str, Any], facts: Dict[str, Any], visual_assets: Dict[str, Any]) -> str:
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
    if mode == "caption_visual_refine":
        return (
            "You are refining an ERP caption-style QA sample for one target object.\n\n"
            f"Structured facts:\n{facts_json}\n\n"
            "Think step by step internally:\n"
            "1. Use the provided context_view with the thin target box to locate the object.\n"
            "2. Confirm the target identity and attributes from the boxed object and its local surroundings.\n"
            "3. Compare the current QA with the actual target appearance.\n"
            "4. If the current question uses an awkward attribute-stacked hint, rewrite it into a natural referring expression while keeping the same target.\n"
            "5. Keep only visually supported details.\n"
            "6. Rewrite the QA into a more natural and complete form.\n\n"
            "Important rules:\n"
            "- The target is anchored by the provided BFOV and local description cues.\n"
            "- If the current answer is slightly incomplete but still correct, improve completeness conservatively.\n"
            "- Do not hallucinate hidden attributes, actions, or materials.\n"
            "- full_answer should be the final polished answer text, and it may include a short grounded rationale before the main description.\n\n"
            "Return JSON with keys decision, question, and full_answer.\n"
            "Valid decisions: keep, correct, filter.\n"
        )

    if mode == "counting_visual_correct":
        return (
            "You are verifying and rewriting an ERP counting QA sample.\n\n"
            f"Structured facts:\n{facts_json}\n\n"
            "Visual setup:\n"
            "- You are given four perspective views derived from the same ERP panorama.\n"
            "- The four images are ordered as front, right, back, and left.\n"
            "- Each view uses the same 100-degree perspective field of view.\n"
            "- Together they cover the full 360 scene.\n\n"
            "Think step by step internally:\n"
            "1. Inspect all four perspective views together as one 360 scene.\n"
            "2. Locate all instances of the queried category across the four views.\n"
            "3. Be careful not to double-count objects that appear near the overlapping boundaries between adjacent views.\n"
            "4. Check partial occlusions and small instances before deciding the final count.\n"
            "5. Compare the visual count with the canonical count.\n"
            "6. If the scene is too ambiguous, choose filter.\n"
            "7. After deciding the final count, rewrite the QA naturally.\n\n"
            "Important rules:\n"
            "- verified_count must be the final integer count.\n"
            "- full_answer should contain a brief grounded counting rationale and then the final count answer.\n"
            "- The final numeric answer in full_answer must match verified_count.\n\n"
            "Return JSON with keys decision, question, verified_count, and full_answer.\n"
            "Valid decisions: keep, correct, filter.\n"
        )

    if mode == "existence_positive_repackage":
        return (
            "You are rewriting a positive ERP existence QA sample.\n\n"
            f"Structured facts:\n{facts_json}\n\n"
            "Think step by step internally:\n"
            "1. Read the queried category.\n"
            "2. Confirm from the structured facts that it is present in the metadata.\n"
            "3. Rewrite the question and answer more naturally without changing the truth.\n\n"
            "Rules:\n"
            "- full_answer should directly answer yes, and may add one short supporting clause.\n"
            "Return JSON with keys question and full_answer.\n"
        )

    if mode == "existence_negative_repackage":
        return (
            "You are rewriting a negative ERP existence QA sample.\n\n"
            f"Structured facts:\n{facts_json}\n\n"
            "Think step by step internally:\n"
            "1. Read the present_labels list.\n"
            "2. Choose one absent category from negative_candidate_pool.\n"
            "3. Prefer a plausible household or indoor category that is clearly not listed as present.\n"
            "4. Rewrite the negative existence QA around that absent category.\n\n"
            "Rules:\n"
            "- Do not choose a category from present_labels.\n"
            "- Do not invent a new category outside negative_candidate_pool.\n"
            "- full_answer should directly answer no, and may add one short supporting clause.\n"
            "Return JSON with keys question and full_answer.\n"
        )

    if mode == "grounding_repackage":
        return _deterministic_prompt(
            facts_json,
            "grounding",
            [
                "Read the target label, entity_ref, and grounding_mode first.",
                "Interpret what the question expects: bbox only, BFOV only, or both.",
                "Treat BFOV as the object's full spherical localization footprint rather than a single center point.",
                "Preserve the localization truth exactly.",
                "Rewrite the QA more naturally, but do not alter any numeric target values or the target identity.",
            ],
            "full_answer must preserve the same grounding truth exactly, including BFOV and bbox when they are required.",
            allow_reasoning=False,
        )

    if mode == "direct_direction_repackage":
        return _deterministic_prompt(
            facts_json,
            "absolute directional localization",
            [
                "Read the target label, entity_ref, and direction_mode first.",
                "Decide whether the target answer is precise BFOV or a coarse absolute 8-way panorama sector.",
                "If the answer is precise, treat BFOV as the full spherical localization target.",
                "Preserve the direction truth exactly.",
                "Rewrite the QA in a more varied style.",
            ],
            "If the canonical answer is BFOV, keep the same BFOV. If it is directional, keep the same directional meaning and do not expand it into a different localization format.",
            allow_reasoning=False,
        )

    if mode == "relative_direction_repackage":
        return _deterministic_prompt(
            facts_json,
            "panoramic angular relation",
            [
                "Read the reference target and query target, together with their referring phrases and yaw or BFOV cues first.",
                "Interpret the task as a viewer-centered angular relation on the 360 panorama ring rather than a true 3D front or back relation.",
                "Use the per-object yaw or BFOV cues first, and then use delta_yaw_deg only as a support fact rather than the whole task definition.",
                "Stay within the allowed relation space right / back-right / opposite / back-left / left.",
                "Keep the relative direction truth unchanged.",
                "full_answer may include one short explanatory clause about the angular offset around the panorama ring before or after the final relation.",
            ],
            "full_answer must preserve the exact relative direction label.",
            allow_reasoning=True,
        )

    if mode == "view_transform_repackage":
        if sample.get("generation_mode") == "object_conditioned_reorientation":
            return _deterministic_prompt(
                facts_json,
                "object-conditioned reorientation",
                [
                    "Read the facing_target, query_target, and their yaw or BFOV cues first.",
                    "Treat the facing_target as the new forward direction in the reoriented view.",
                    "Interpret the answer in the new observer-centered view, not as a true 3D relation between objects.",
                    "Stay within the allowed relation space right / back-right / behind / back-left / left.",
                    "Use delta_yaw_deg only as a supporting geometric fact.",
                    "Rewrite the QA clearly so it is obvious that the observer first turns to face the reference target.",
                ],
                "full_answer must preserve the same reoriented-view direction label exactly.",
                allow_reasoning=True,
            )
        return _deterministic_prompt(
            facts_json,
            "camera rotation transform",
            [
                "Read the target label, entity_ref, original BFOV cue, original_absolute_sector, turn direction, and turn angle carefully.",
                "Use the provided explicit camera rotation instead of inventing a new transform.",
                "Interpret the answer in the reoriented observer view, not as a world-coordinate relation.",
                "Stay within the allowed relation space right / back-right / behind / back-left / left.",
                "You may briefly mention the original absolute sector and the shifted view, but preserve the canonical direction truth exactly.",
                "Rewrite the QA clearly so there is no left or right turn ambiguity.",
            ],
            "The question must explicitly mention turning left or turning right.",
            allow_reasoning=True,
        )

    if mode == "distance_estimation_repackage":
        if sample.get("generation_mode") == "candidate_nearest_choice":
            return _deterministic_prompt(
                facts_json,
                "candidate distance comparison",
                [
                    "Read the reference_label, reference_ref, candidate_objects, distance_measure, and candidate_distances_m carefully.",
                    "Treat the provided candidate_distances_m as deterministic ground-truth support facts for the answer.",
                    "Do not reorder or replace the candidate set, and do not change which candidate is the nearest one.",
                    "Use the provided referring phrases to make the target objects easier to identify than plain labels alone.",
                    "Rewrite the question naturally as a multi-choice nearest-distance comparison in 3D space.",
                    "full_answer may briefly mention that the chosen object is the nearest under the stated metric.",
                ],
                "Keep the notion of distance explicit in the question, and preserve that the metric is center-based 3D distance rather than a vague semantic notion of closeness.",
                allow_reasoning=True,
            )
        if sample.get("generation_mode") == "observer_nearest_choice":
            return _deterministic_prompt(
                facts_json,
                "observer-referenced distance comparison",
                [
                    "Read the candidate_objects, candidate_depths_m, and the observer reference carefully.",
                    "Treat the provided candidate_depths_m as deterministic ground-truth support facts for the answer.",
                    "Do not reorder or replace the candidate set, and do not change which candidate is nearest to the observer.",
                    "Use the candidate referring phrases to make the question more specific than plain object labels alone.",
                    "Rewrite the question naturally as a multi-choice nearest-to-observer comparison.",
                    "full_answer may briefly mention that the chosen object has the smallest observer depth.",
                ],
                "Keep the observer or camera-center reference explicit in the question, and preserve that the metric is depth to the camera center.",
                allow_reasoning=True,
            )
        return _deterministic_prompt(
            facts_json,
            "distance estimation",
            [
                "Read the target label, target_ref, target BFOV cue, and the absolute depth value in meters.",
                "Keep the numeric answer stable.",
                "Use the referring phrase to keep the question specific when there may be multiple objects of the same class.",
                "Rewrite the QA naturally without changing the unit or magnitude.",
            ],
            "full_answer must preserve the same meter value.",
            allow_reasoning=False,
        )

    if mode == "relative_3d_position_repackage":
        return _deterministic_prompt(
            facts_json,
            "relative 3D position",
            [
                "Read the two target entities, their referring phrases, and the already-computed geometric support facts first.",
                "Use the already-computed geometry rather than recomputing from scratch.",
                "Read the rule-derived 3D relation carefully.",
                "Keep the 3D relation truth unchanged.",
                "If multiple axis relations are present, preserve all of them and do not drop any axis.",
                "Do not expose raw xyz coordinates in the final answer unless the question explicitly asks for them.",
                "If you include a short analysis, express it as natural relative-axis reasoning that stays consistent with the canonical relation.",
                "full_answer may include a short, correct geometric analysis before giving the final relation.",
            ],
            "full_answer must preserve the same 3D relation label, including every active axis relation.",
            allow_reasoning=True,
        )

    if mode == "seam_continuity_repackage":
        seam_mode = sample.get("generation_mode")
        if seam_mode == "counterpart_boundary_side":
            return _deterministic_prompt(
                facts_json,
                "seam continuity boundary-side matching",
                [
                    "Read the target label, BFOV cue, fragment_side, and canonical_side carefully.",
                    "Interpret the task as ERP wrap-around matching between the left and right panorama boundaries.",
                    "Preserve exactly which boundary contains the continuation fragment.",
                    "Rewrite the question so it is clearly about left-right wrap-around continuation rather than a generic spatial relation.",
                ],
                "full_answer must preserve the same boundary-side truth exactly.",
                allow_reasoning=True,
            )
        if seam_mode == "wrap_explanation":
            return _deterministic_prompt(
                facts_json,
                "seam continuity explanation",
                [
                    "Read the target label, BFOV cue, and canonical_explanation carefully.",
                    "Interpret the split appearance as ERP wrap-around across the left-right boundary.",
                    "Rewrite the question so it asks for the wrap-around explanation rather than a generic imaging artifact explanation.",
                    "full_answer may include one short clarifying clause, but it must still state ERP wrap-around as the core explanation.",
                ],
                "full_answer must preserve ERP left-right wrap-around as the explanation.",
                allow_reasoning=True,
            )
        if seam_mode == "rotation_continuity":
            return _deterministic_prompt(
                facts_json,
                "seam continuity under boundary shift",
                [
                    "Read the target label, BFOV cue, and canonical_truth carefully.",
                    "Interpret the task as asking whether shifting the ERP boundary would make the object visually continuous.",
                    "Preserve the wrap-around continuity truth exactly.",
                    "Rewrite the question so the boundary-shift thought experiment is explicit.",
                ],
                "full_answer must preserve the same yes/no continuity truth exactly.",
                allow_reasoning=True,
            )
        return _deterministic_prompt(
            facts_json,
            "seam continuity same-instance judgement",
            [
                "Read the target label, BFOV cue, and canonical_truth carefully.",
                "Interpret the two fragments as possible ERP wrap-around views of the same object.",
                "Preserve the same-instance truth exactly.",
                "Rewrite the question so it is explicitly about ERP left-right wrap-around rather than a normal image split.",
            ],
            "full_answer must preserve the same yes/no same-instance truth exactly.",
            allow_reasoning=True,
        )

    if mode == "polar_distortion_awareness_repackage":
        if sample.get("generation_mode") == "shape_recovery_distortion_aware":
            return (
                "You are rewriting an ERP polar distortion awareness QA sample.\n\n"
                f"Structured facts:\n{facts_json}\n\n"
                "Background:\n"
                "- This is an ERP panorama, not a standard perspective image.\n"
                "- Near the poles, ERP projection can stretch or warp object appearance in 2D.\n"
                "- The goal is to recover the object's true shape rather than the distorted visual impression.\n\n"
                "Think step by step internally:\n"
                "1. Read the target BFOV, latitude_band, and true_shape first.\n"
                "2. Preserve that this is a distortion-aware shape recovery question.\n"
                "3. Rewrite the question so the pole-distortion challenge is explicit, not hidden.\n"
                "4. full_answer may include one short phrase that the ERP appearance can be distorted, followed by the true shape.\n\n"
                "Return JSON with keys question and full_answer.\n"
            )
        return (
            "You are rewriting an ERP high-latitude shape QA sample.\n\n"
            f"Structured facts:\n{facts_json}\n\n"
            "Background:\n"
            "- This is an ERP panorama.\n"
            "- The target is at high latitude, so shape understanding should remain grounded in the true object shape.\n\n"
            "Think step by step internally:\n"
            "1. Read the target BFOV, latitude_band, and true_shape first.\n"
            "2. Preserve that the answer is the object's true shape.\n"
            "3. Rewrite the question naturally without over-explaining distortion if the prompt is the direct version.\n"
            "4. full_answer should end with the true shape and may include at most one short supporting clause.\n\n"
            "Return JSON with keys question and full_answer.\n"
        )

    return _deterministic_prompt(
        facts_json,
        sample["task_family"],
        [
            "Read the canonical truth carefully.",
            "Keep the truth unchanged.",
            "Rewrite the QA in a more varied style.",
        ],
        "full_answer must stay semantically identical to the canonical truth.",
        allow_reasoning=False,
    )


def _deterministic_prompt(
    facts_json: str,
    task_name: str,
    steps: List[str],
    extra_rule: str,
    *,
    allow_reasoning: bool,
) -> str:
    step_lines = "\n".join(f"{idx}. {step}" for idx, step in enumerate(steps, start=1))
    reasoning_rule = (
        "You may include one short rationale clause in full_answer.\n"
        if allow_reasoning
        else "Prefer a direct concise answer.\n"
    )
    return (
        f"You are rewriting a deterministic ERP {task_name} QA sample.\n\n"
        f"Structured facts:\n{facts_json}\n\n"
        "Think step by step internally:\n"
        f"{step_lines}\n\n"
        "Rules:\n"
        f"- {extra_rule}\n"
        f"- {reasoning_rule}"
        "- Do not invent new entities, new relations, or new numbers.\n"
        "- Put any short rationale directly into full_answer instead of a separate notes field.\n"
        "Return JSON with keys question and full_answer.\n"
    )


def _build_messages(prompt_text: str, visual_assets: Dict[str, Any], requires_visual: bool) -> List[Dict[str, Any]]:
    if requires_visual and visual_assets.get("image_available"):
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        perspective_images = visual_assets.get("perspective_images", [])
        if perspective_images:
            for item in perspective_images:
                image_url = item.get("data_url") or _image_data_url(item["path"])
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            if visual_assets.get("overlay_image_path"):
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(visual_assets["overlay_image_path"])}})
        else:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(visual_assets["erp_image_path"])}})
        return [{"role": "user", "content": content}]
    return [{"role": "user", "content": prompt_text}]


def build_messages_for_job(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    visual_assets = dict(job.get("visual_assets", {}))
    if (
        bool(job.get("requires_visual"))
        and visual_assets.get("mode") == "erp_four_faces_deferred"
        and visual_assets.get("image_available")
    ):
        spec = visual_assets.get("four_face_spec", {})
        visual_assets = build_four_face_visual_context_from_path(
            spec.get("erp_image_path", ""),
            spec.get("scene_id", job.get("scene_id", "scene")),
        )
    elif (
        bool(job.get("requires_visual"))
        and visual_assets.get("mode") == "erp_entity_context_deferred"
        and visual_assets.get("image_available")
    ):
        spec = visual_assets.get("entity_context_spec", {})
        visual_assets = build_entity_visual_context_from_spec(
            visual_assets.get("erp_image_path", ""),
            spec,
        )
    return _build_messages(job["prompt_text"], visual_assets, bool(job.get("requires_visual")))


def _image_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _expected_output_schema(mode: str) -> Dict[str, Any]:
    if mode in {"counting_visual_correct", "caption_visual_refine"}:
        schema = {"question": "string", "full_answer": "string"}
        if mode == "counting_visual_correct":
            schema["decision"] = "string"
            schema["verified_count"] = "string"
        else:
            schema["decision"] = "string"
        return schema
    return {
        "question": "string",
        "full_answer": "string",
    }
