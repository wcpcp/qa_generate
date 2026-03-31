from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schemas import Entity, SceneMetadata


ROOT = Path(__file__).resolve().parents[2]
QUESTION_TEMPLATE_PATH = ROOT / "templates" / "question_templates.json"
ANSWER_TEMPLATE_PATH = ROOT / "templates" / "answer_templates.json"


def load_question_templates(template_path: Optional[str] = None) -> Dict[str, List[str]]:
    path = Path(template_path) if template_path else QUESTION_TEMPLATE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def load_answer_templates(template_path: Optional[str] = None) -> Dict[str, List[str]]:
    path = Path(template_path) if template_path else ANSWER_TEMPLATE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def build_canonical_samples(
    scene: SceneMetadata,
    scene_plan: Dict[str, Any],
    template_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    templates = load_question_templates(template_path)
    answer_templates = load_answer_templates()
    entity_by_id = {entity.entity_id: entity for entity in scene.entities}
    samples: List[Dict[str, Any]] = []

    def _append_from_task(task: Dict[str, Any], task_origin: str, origin_id: Optional[str] = None) -> None:
        sample = _build_sample_from_task(
            scene=scene,
            task=task,
            task_origin=task_origin,
            origin_id=origin_id,
            entity_by_id=entity_by_id,
            templates=templates,
            answer_templates=answer_templates,
            sample_index=len(samples) + 1,
        )
        if sample is not None:
            samples.append(sample)

    for task in scene_plan.get("task_groups", {}).get("global_tasks", []):
        _append_from_task(task, "scene_global")

    for task in scene_plan.get("task_groups", {}).get("rotation_tasks", []):
        _append_from_task(task, "scene_rotation")

    for anchor in scene_plan.get("anchors", []):
        anchor_id = anchor["entity_id"]
        for task in anchor.get("tasks", []):
            _append_from_task(task, "anchor_or_pair", origin_id=anchor_id)

    return _cap_scene_samples(samples, max_samples=20)


def _build_sample_from_task(
    *,
    scene: SceneMetadata,
    task: Dict[str, Any],
    task_origin: str,
    origin_id: Optional[str],
    entity_by_id: Dict[str, Entity],
    templates: Dict[str, List[str]],
    answer_templates: Dict[str, List[str]],
    sample_index: int,
) -> Optional[Dict[str, Any]]:
    task_family = task["task_family"]
    entities = [entity_by_id[entity_id] for entity_id in task.get("entity_ids", []) if entity_id in entity_by_id]
    question, answer_value, answer_text, metadata, authoring, template_key, template_index = _realize_task(
        scene=scene,
        task=task,
        entities=entities,
        templates=templates,
        answer_templates=answer_templates,
        template_seed=f"{scene.scene_id}:{task_family}:{origin_id or 'scene'}:{sample_index}",
    )
    if not question or answer_value is None:
        return None

    sample_id = f"{scene.scene_id}:{task_family}:{sample_index:04d}"
    sample: Dict[str, Any] = {
        "sample_id": sample_id,
        "scene_id": scene.scene_id,
        "task_family": task_family,
        "ability": task["ability"],
        "difficulty": task["difficulty"],
        "phase": task["phase"],
        "metadata_tier": task["metadata_tier"],
        "erp_robustness": task.get("erp_robustness", False),
        "task_origin": task_origin,
        "origin_anchor_id": origin_id,
        "entity_ids": task.get("entity_ids", []),
        "canonical_question": question,
        "canonical_answer": answer_value,
        "answer_text": answer_text,
        "answer_type": task["answer_type"],
        "answer_space": task.get("answer_space"),
        "evidence_fields": task.get("evidence_fields", []),
        "gt_source": task.get("gt_source", []),
        "generation_mode": task.get("generation_mode"),
        "metadata": {
            **metadata,
            "template_key": template_key,
            "template_variant_index": template_index,
        },
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_text},
        ],
    }
    if task.get("geometry_source"):
        sample["geometry_source"] = task["geometry_source"]
    if authoring:
        sample["authoring_strategy"] = authoring["strategy"]
        sample["authoring_prompt_facts"] = authoring["facts"]
        sample["authoring_prompt_variants"] = authoring["prompt_variants"]
    return sample


def _realize_task(
    *,
    scene: SceneMetadata,
    task: Dict[str, Any],
    entities: Sequence[Entity],
    templates: Dict[str, List[str]],
    answer_templates: Dict[str, List[str]],
    template_seed: str,
) -> Tuple[str, Any, str, Dict[str, Any], Optional[Dict[str, Any]], str, int]:
    task_family = task["task_family"]
    generation_mode = task.get("generation_mode", "")

    if task_family == "caption":
        entity = entities[0]
        return _realize_caption(scene, entity, generation_mode, templates, answer_templates, template_seed)

    if task_family == "existence":
        query_target = task.get("query_target", "object")
        present = any(_normalize_phrase(entity.label) == _normalize_phrase(query_target) for entity in scene.entities)
        template_key = "existence"
        template, index = _pick_template(templates, template_key, template_seed)
        question = template.format(category_name=_display_label(query_target))
        answer_key = "existence.true" if present else "existence.false"
        answer_text = _pick_answer_template(
            answer_templates,
            answer_key,
            template_seed,
            truth="yes" if present else "no",
        )
        return question, present, answer_text, {"query_target": query_target}, None, template_key, index

    if task_family == "counting":
        query_target = task.get("query_target", "object")
        count = sum(1 for entity in scene.entities if _normalize_phrase(entity.label) == _normalize_phrase(query_target))
        template_key = "counting"
        template, index = _pick_template(templates, template_key, template_seed)
        question = template.format(category_name=_pluralize_label(query_target))
        metadata = {
            "query_target": query_target,
            "count_source": "metadata_provisional",
            "requires_visual_verification": True,
        }
        answer_text = _pick_answer_template(answer_templates, "counting", template_seed, count=str(count))
        return question, count, answer_text, metadata, None, template_key, index

    if task_family == "grounding":
        entity = entities[0]
        answer_mode = generation_mode or "full"
        template_key = f"grounding.{answer_mode}"
        template, index = _pick_template(templates, template_key, template_seed)
        question = template.format(entity_ref=_grounding_ref(entity), target_label=_display_label(entity.label))
        bbox_norm = _normalized_bbox_1000(scene, entity)
        bfov = _entity_bfov(scene, entity)
        answer_value = {"bbox_norm_1000": bbox_norm, "bfov": bfov}
        answer_template_key = f"grounding.{answer_mode}"
        answer_text = _pick_answer_template(
            answer_templates,
            answer_template_key,
            template_seed,
            bbox=bbox_norm,
            bfov=_bfov_text(bfov),
        )
        return question, answer_value, answer_text, _entity_loc_metadata(scene, entity), None, template_key, index

    if task_family == "scene_understanding":
        return _build_scene_understanding_stub(task)

    if task_family == "direct_direction":
        entity = entities[0]
        return _realize_direct_direction(scene, entity, generation_mode, templates, answer_templates, template_seed)

    if task_family == "relative_direction":
        entity_a, entity_b = entities
        template_key = "relative_direction"
        template, index = _pick_template(templates, template_key, template_seed)
        question = template.format(
            reference_ref=_grounding_ref(entity_a),
            target_ref=_grounding_ref(entity_b),
            reference_label=_display_label(entity_a.label),
            target_label=_display_label(entity_b.label),
        )
        answer_value = _panoramic_ring_relation(entity_a, entity_b, opposite_label="opposite") or "right"
        answer_text = _pick_answer_template(answer_templates, "relative_direction", template_seed, relation=answer_value)
        return question, answer_value, answer_text, _pair_metadata(scene, entity_a, entity_b), None, template_key, index

    if task_family == "view_transform":
        return _realize_view_transform(scene, entities, generation_mode, task, templates, answer_templates, template_seed)

    if task_family == "distance_estimation":
        return _realize_distance_estimation(scene, entities, generation_mode, templates, answer_templates, template_seed)

    if task_family == "relative_3d_position":
        entity_a, entity_b = entities
        return _realize_relative_3d_position(scene, entity_a, entity_b, generation_mode, templates, answer_templates, template_seed, task.get("geometry_source"))

    if task_family == "seam_continuity":
        entity = entities[0]
        return _realize_seam_continuity(scene, task, entity, generation_mode, templates, answer_templates, template_seed)

    if task_family == "polar_distortion_awareness":
        entity = entities[0]
        return _realize_polar_distortion_awareness(scene, entity, generation_mode, templates, answer_templates, template_seed)

    if task_family == "cognitive_map_understanding":
        return _build_cognitive_map_understanding_stub(task)
    if task_family == "hps_path_selection":
        return _build_hps_path_selection_stub(task)
    if task_family == "rotation_consistency":
        return _build_rotation_consistency_stub(task)

    return "", None, "", {}, None, "", 0


def _realize_caption(
    scene: SceneMetadata,
    entity: Entity,
    generation_mode: str,
    templates: Dict[str, List[str]],
    answer_templates: Dict[str, List[str]],
    template_seed: str,
) -> Tuple[str, Any, str, Dict[str, Any], Optional[Dict[str, Any]], str, int]:
    bbox_text = _bbox_text(_normalized_bbox_1000(scene, entity))
    yaw_deg = _fmt_float(_yaw_deg_360(entity))
    pitch_deg = _fmt_float(_pitch_deg_180(entity))
    if generation_mode == "attribute_like":
        attribute_name, attribute_value = _choose_attribute(entity)
        entity_hint = _entity_hint(entity, exclude_attribute=attribute_name, allow_label=True)
        template_key = "caption.attribute"
        template, index = _pick_template(templates, template_key, template_seed)
        question = template.format(
            attribute_name=_display_label(attribute_name),
            entity_hint=entity_hint,
        )
        metadata = {
            **_entity_loc_metadata(scene, entity),
            "caption_mode": "attribute",
            "attribute_name": attribute_name,
            "entity_hint": entity_hint,
        }
        answer_text = _pick_answer_template(
            answer_templates,
            "caption.attribute",
            template_seed,
            attribute_name=_display_label(attribute_name),
            attribute_value=_display_value(attribute_value),
        )
        return question, attribute_value, answer_text, metadata, None, template_key, index

    if generation_mode == "description_like":
        answer = entity.semantic.caption_brief or f"A {_display_label(entity.label)}."
        entity_hint = _entity_hint(entity, allow_label=True)
        template_key = "caption.description"
        template, index = _pick_template(templates, template_key, template_seed)
        question = template.format(entity_hint=entity_hint)
        metadata = {**_entity_loc_metadata(scene, entity), "caption_mode": "description", "entity_hint": entity_hint}
        answer_text = _pick_answer_template(answer_templates, "caption.description", template_seed, description=answer)
        return question, answer, answer_text, metadata, None, template_key, index

    if generation_mode == "dense_description_like":
        answer = entity.semantic.caption_dense or entity.semantic.caption_brief or f"A {_display_label(entity.label)}."
        entity_hint = _caption_entity_hint(entity)
        template_key = "caption.dense_description"
        template, index = _pick_template(templates, template_key, template_seed)
        question = template.format(entity_hint=entity_hint)
        metadata = {
            **_entity_loc_metadata(scene, entity),
            "caption_mode": "dense_description",
            "entity_hint": entity_hint,
        }
        answer_text = _pick_answer_template(
            answer_templates,
            "caption.dense_description",
            template_seed,
            description=answer,
        )
        return question, answer, answer_text, metadata, None, template_key, index

    entity_hint = _entity_hint(entity, allow_label=False)
    template_key = "caption.identify"
    template, index = _pick_template(templates, template_key, template_seed)
    question = template.format(entity_hint=entity_hint)
    metadata = {**_entity_loc_metadata(scene, entity), "caption_mode": "identify", "entity_hint": entity_hint}
    answer_text = _pick_answer_template(answer_templates, "caption.identify", template_seed, label=_display_label(entity.label))
    return question, entity.label, answer_text, metadata, None, template_key, index


def _caption_entity_hint(entity: Entity) -> str:
    # caption 题面优先使用 reground_query 作为 referring expression。
    # 它本身就是为“重新定位该物体”设计的短语，通常比属性硬拼更自然、
    # 也更有区分度。只有 reground_query 不可用时，才退回其他语义字段。
    reground = (entity.semantic.reground_query or "").strip()
    if reground:
        phrase = _display_label(reground)
        if not phrase.startswith("the "):
            phrase = f"the {phrase}"
        return phrase

    brief = (entity.semantic.caption_brief or "").strip()
    if brief:
        phrase = _display_label(brief).rstrip(".")
        if not phrase.startswith("the "):
            phrase = f"the {phrase}"
        return phrase

    return _entity_hint(entity, allow_label=True)


def _realize_direct_direction(
    scene: SceneMetadata,
    entity: Entity,
    generation_mode: str,
    templates: Dict[str, List[str]],
    answer_templates: Dict[str, List[str]],
    template_seed: str,
) -> Tuple[str, Any, str, Dict[str, Any], Optional[Dict[str, Any]], str, int]:
    if generation_mode == "absolute_sector_8way":
        template_key = "direct_direction.cardinal"
        template, index = _pick_template(templates, template_key, template_seed)
        answer = _cardinal_direction_from_yaw(_yaw_deg_360(entity))
        question = template.format(entity_ref=_grounding_ref(entity), target_label=_display_label(entity.label))
        metadata = {**_entity_loc_metadata(scene, entity), "direction_mode": "absolute_sector_8way"}
        answer_text = _pick_answer_template(answer_templates, "direct_direction.cardinal", template_seed, direction=answer)
        return question, answer, answer_text, metadata, None, template_key, index

    template_key = "direct_direction.precise"
    template, index = _pick_template(templates, template_key, template_seed)
    bfov = _entity_bfov(scene, entity)
    question = template.format(entity_ref=_grounding_ref(entity), target_label=_display_label(entity.label))
    answer_value = {"bfov": bfov}
    metadata = {**_entity_loc_metadata(scene, entity), "direction_mode": "precise_bfov"}
    answer_text = _pick_answer_template(
        answer_templates,
        "direct_direction.precise",
        template_seed,
        bfov=_bfov_text(bfov),
    )
    return question, answer_value, answer_text, metadata, None, template_key, index


def _realize_view_transform(
    scene: SceneMetadata,
    entities: Sequence[Entity],
    generation_mode: str,
    task: Dict[str, Any],
    templates: Dict[str, List[str]],
    answer_templates: Dict[str, List[str]],
    template_seed: str,
) -> Tuple[str, Any, str, Dict[str, Any], Optional[Dict[str, Any]], str, int]:
    mode = generation_mode or "camera_rotation_transform"
    template_key = f"view_transform.{mode}"
    template, index = _pick_template(templates, template_key, template_seed)

    if mode == "object_conditioned_reorientation":
        facing, target = entities
        answer = _panoramic_ring_relation(facing, target, opposite_label="behind") or "right"
        metadata = {
            "facing_label": facing.label,
            "facing_ref": _grounding_ref(facing),
            "facing_yaw_deg": round(_yaw_deg_360(facing), 1),
            "facing_bfov": _entity_bfov(scene, facing),
            "target_label": target.label,
            "target_ref": _grounding_ref(target),
            "target_yaw_deg": round(_yaw_deg_360(target), 1),
            "target_bfov": _entity_bfov(scene, target),
            "delta_yaw_deg": round(_wrapped_delta_deg(_yaw_deg_360(target) - _yaw_deg_360(facing)), 2),
            "transform_mode": "object_conditioned_reorientation",
        }
        question = template.format(
            facing_ref=_grounding_ref(facing),
            facing_label=_display_label(facing.label),
            target_ref=_grounding_ref(target),
            target_label=_display_label(target.label),
        )
        answer_text = _pick_answer_template(answer_templates, "view_transform", template_seed, direction=answer)
        return question, answer, answer_text, metadata, None, template_key, index

    entity = entities[0]
    turn_angle_deg = int(task.get("rotation_angle_deg", 90))
    rotation_direction = str(task.get("rotation_direction") or "right")
    if rotation_direction == "left":
        delta_after_rotation = _wrapped_delta_deg(_yaw_deg_360(entity) + turn_angle_deg)
    else:
        delta_after_rotation = _wrapped_delta_deg(_yaw_deg_360(entity) - turn_angle_deg)
    new_direction = _panoramic_relation_from_delta(delta_after_rotation, opposite_label="behind") or "right"
    question = template.format(
        turn_angle_deg=turn_angle_deg,
        entity_ref=_grounding_ref(entity),
        target_label=_display_label(entity.label),
        rotation_direction=rotation_direction,
    )
    metadata = {
        **_entity_loc_metadata(scene, entity),
        "turn_angle_deg": turn_angle_deg,
        "rotation_direction": rotation_direction,
        "delta_after_rotation_deg": round(delta_after_rotation, 2),
        "transform_mode": "camera_rotation_transform",
    }
    answer_text = _pick_answer_template(answer_templates, "view_transform", template_seed, direction=new_direction)
    return question, new_direction, answer_text, metadata, None, template_key, index


def _realize_distance_estimation(
    scene: SceneMetadata,
    entities: Sequence[Entity],
    generation_mode: str,
    templates: Dict[str, List[str]],
    answer_templates: Dict[str, List[str]],
    template_seed: str,
) -> Tuple[str, Any, str, Dict[str, Any], Optional[Dict[str, Any]], str, int]:
    if generation_mode == "candidate_nearest_choice":
        reference = entities[0]
        candidates = list(entities[1:])
        candidate_distances = [
            (_center_distance_3d(reference, candidate), candidate)
            for candidate in candidates
        ]
        candidate_distances.sort(key=lambda item: item[0])
        answer_entity = candidate_distances[0][1]
        template_key = "distance_estimation.choice"
        template, index = _pick_template(templates, template_key, template_seed)
        candidate_list = ", ".join(_display_label(candidate.label) for candidate in candidates)
        question = template.format(
            reference_label=_display_label(reference.label),
            candidate_list=candidate_list,
        )
        metadata = {
            "reference_label": reference.label,
            "reference_ref": _grounding_ref(reference),
            "candidate_labels": [candidate.label for candidate in candidates],
            "candidate_refs": {candidate.label: _grounding_ref(candidate) for candidate in candidates},
            "distance_measure": "center_to_center_3d_distance",
            "candidate_distances_m": {
                candidate.label: round(distance, 3)
                for distance, candidate in candidate_distances
            },
        }
        answer_text = _pick_answer_template(
            answer_templates,
            "distance_estimation.choice",
            template_seed,
            label=_display_label(answer_entity.label),
        )
        return question, answer_entity.label, answer_text, metadata, None, template_key, index

    if generation_mode == "observer_nearest_choice":
        candidates = [
            entity for entity in entities
            if entity.entity_center_depth is not None
        ]
        candidate_depths = sorted(
            [(float(candidate.entity_center_depth), candidate) for candidate in candidates],
            key=lambda item: item[0],
        )
        answer_entity = candidate_depths[0][1]
        template_key = "distance_estimation.observer_choice"
        template, index = _pick_template(templates, template_key, template_seed)
        candidate_list = ", ".join(_display_label(candidate.label) for _, candidate in candidate_depths)
        question = template.format(candidate_list=candidate_list)
        metadata = {
            "reference_label": "observer",
            "candidate_labels": [candidate.label for _, candidate in candidate_depths],
            "candidate_refs": {candidate.label: _grounding_ref(candidate) for _, candidate in candidate_depths},
            "distance_measure": "depth_to_camera_center",
            "candidate_depths_m": {
                candidate.label: round(depth, 3)
                for depth, candidate in candidate_depths
            },
        }
        answer_text = _pick_answer_template(
            answer_templates,
            "distance_estimation.observer_choice",
            template_seed,
            label=_display_label(answer_entity.label),
        )
        return question, answer_entity.label, answer_text, metadata, None, template_key, index

    entity = entities[0]
    depth_m = round(float(entity.entity_center_depth), 2)
    template_key = "distance_estimation"
    template, index = _pick_template(templates, template_key, template_seed)
    question = template.format(target_label=_display_label(entity.label), entity_ref=_grounding_ref(entity))
    metadata = {**_entity_loc_metadata(scene, entity), "depth_m": depth_m}
    answer_text = _pick_answer_template(answer_templates, "distance_estimation", template_seed, depth_m=str(depth_m))
    return question, depth_m, answer_text, metadata, None, template_key, index


def _realize_relative_3d_position(
    scene: SceneMetadata,
    entity_a: Entity,
    entity_b: Entity,
    generation_mode: str,
    templates: Dict[str, List[str]],
    answer_templates: Dict[str, List[str]],
    template_seed: str,
    geometry_source: Optional[str],
) -> Tuple[str, Any, str, Dict[str, Any], Optional[Dict[str, Any]], str, int]:
    metadata = _pair_metadata(scene, entity_a, entity_b)
    metadata["geometry_source"] = geometry_source
    answer = _relative_3d_relation(entity_a, entity_b)
    if generation_mode == "camera_centric_choice":
        template_key = "relative_3d_position.choice"
        template, index = _pick_template(templates, template_key, template_seed)
        choices = _build_relative_3d_choices(entity_a, entity_b, answer)
        metadata["relative_3d_mode"] = "camera_centric_choice"
        metadata["choice_candidates"] = choices
        question = template.format(
            entity_a=_grounding_ref(entity_a),
            entity_b=_grounding_ref(entity_b),
            choice_list=", ".join(choices),
        )
        answer_text = _pick_answer_template(answer_templates, "relative_3d_position.choice", template_seed, relation=answer)
        return question, answer, answer_text, metadata, None, template_key, index

    template_key = "relative_3d_position.open"
    template, index = _pick_template(templates, template_key, template_seed)
    question = template.format(entity_a=_grounding_ref(entity_a), entity_b=_grounding_ref(entity_b))
    metadata["relative_3d_mode"] = "camera_centric_open"
    answer_text = _pick_answer_template(answer_templates, "relative_3d_position", template_seed, relation=answer)
    return question, answer, answer_text, metadata, None, template_key, index


def _realize_polar_distortion_awareness(
    scene: SceneMetadata,
    entity: Entity,
    generation_mode: str,
    templates: Dict[str, List[str]],
    answer_templates: Dict[str, List[str]],
    template_seed: str,
) -> Tuple[str, Any, str, Dict[str, Any], Optional[Dict[str, Any]], str, int]:
    mode = generation_mode or "shape_recovery_direct"
    if mode == "shape_recovery_distortion_aware":
        template_key = "polar_distortion_awareness.shape_distortion_aware"
    else:
        template_key = "polar_distortion_awareness.shape_direct"
    answer = _display_value((entity.semantic.attributes or {}).get("shape", "unknown"))
    template, index = _pick_template(templates, template_key, template_seed)
    question = template.format(
        target_label=_display_label(entity.label),
        bbox_norm_1000=_bbox_text(_normalized_bbox_1000(scene, entity)),
        pitch_deg=_fmt_float(_pitch_deg_180(entity)),
        yaw_deg=_fmt_float(_yaw_deg_360(entity)),
        bfov=_bfov_text(_entity_bfov(scene, entity)),
    )
    metadata = {**_entity_loc_metadata(scene, entity), "polar_mode": mode}
    answer_text = _pick_answer_template(answer_templates, "polar_distortion_awareness.shape", template_seed, shape=answer)
    return question, answer, answer_text, metadata, None, template_key, index


def _realize_seam_continuity(
    scene: SceneMetadata,
    task: Dict[str, Any],
    entity: Entity,
    generation_mode: str,
    templates: Dict[str, List[str]],
    answer_templates: Dict[str, List[str]],
    template_seed: str,
) -> Tuple[str, Any, str, Dict[str, Any], Optional[Dict[str, Any]], str, int]:
    entity_by_id = {item.entity_id: item for item in scene.entities}
    valid_modes = list(task.get("seam_valid_modes", []) or ["same_entity_judgement", "dedup_count"])
    mode = generation_mode or valid_modes[0]
    if mode not in valid_modes:
        digest = hashlib.md5(f"{template_seed}:seam_fallback".encode("utf-8")).hexdigest()
        mode = valid_modes[int(digest[:8], 16) % len(valid_modes)]
    template_key = f"seam_continuity.{mode}"
    template, index = _pick_template(templates, template_key, template_seed)
    metadata = {
        **_entity_loc_metadata(scene, entity),
        "seam_crossing_flag": bool(entity.seam_crossing_flag),
        "seam_mode": mode,
        "seam_subtype": mode,
    }
    target_ref = _grounding_ref(entity)
    target_side = str(task.get("seam_target_side", _preferred_seam_target_side(scene, entity)))
    metadata["target_side"] = target_side

    if mode == "nearest_neighbor":
        choice_entities = _resolve_seam_choice_entities(task, entity_by_id)
        correct_entity = entity_by_id.get(str(task.get("seam_partner_id", "")))
        if correct_entity is None or len(choice_entities) < 3:
            return "", None, "", {}, None, "", 0
        choice_refs = [_grounding_ref(item) for item in choice_entities]
        answer = _grounding_ref(correct_entity)
        metadata.update(
            {
                "choice_candidates": choice_refs,
                "neighbor_ref": answer,
                "neighbor_entity_id": correct_entity.entity_id,
                "wrap_gap_deg": task.get("seam_wrap_gap_deg"),
            }
        )
        question = template.format(target_ref=target_ref, target_side=target_side, choice_list=", ".join(choice_refs))
        answer_text = _pick_answer_template(answer_templates, "seam_continuity.choice", template_seed, answer=answer)
        return question, answer, answer_text, metadata, None, template_key, index

    if mode == "relative_direction":
        partner = entity_by_id.get(str(task.get("seam_partner_id", "")))
        relation = str(task.get("seam_relation", "")).strip()
        if partner is None or not relation:
            return "", None, "", {}, None, "", 0
        choices = [
            "immediately across the seam on the left",
            "immediately across the seam on the right",
            "roughly opposite in the panorama",
            "not actually seam-adjacent",
        ]
        metadata.update(
            {
                "neighbor_ref": _grounding_ref(partner),
                "neighbor_entity_id": partner.entity_id,
                "choice_candidates": choices,
                "canonical_relation": relation,
            }
        )
        question = template.format(
            target_ref=target_ref,
            neighbor_ref=_grounding_ref(partner),
            choice_list=", ".join(choices),
        )
        answer_text = _pick_answer_template(answer_templates, "seam_continuity.choice", template_seed, answer=relation)
        return question, relation, answer_text, metadata, None, template_key, index

    if mode == "dedup_count":
        choices = [
            "one continuous object",
            "two separate objects",
            "cannot determine from the panorama",
        ]
        answer = "one continuous object"
        metadata["choice_candidates"] = choices
        question = template.format(target_ref=target_ref, choice_list=", ".join(choices))
        answer_text = _pick_answer_template(answer_templates, "seam_continuity.choice", template_seed, answer=answer)
        return question, answer, answer_text, metadata, None, template_key, index

    if mode == "structure_continuity":
        target_kind = _display_label(entity.label)
        choices = [
            f"one continuous {target_kind} structure across the seam",
            f"two unrelated {target_kind} parts that only look similar",
            "a mirrored or reflective duplication rather than a continuous structure",
            "a local fragment that should terminate at the edge",
        ]
        answer = choices[0]
        metadata["choice_candidates"] = choices
        question = template.format(target_ref=target_ref, choice_list=", ".join(choices))
        answer_text = _pick_answer_template(answer_templates, "seam_continuity.choice", template_seed, answer=answer)
        return question, answer, answer_text, metadata, None, template_key, index

    choices = [
        "two views of the same continuous object across the seam",
        "two different objects of the same category",
        "one object and one unrelated background fragment",
        "cannot determine from the panorama",
    ]
    answer = choices[0]
    metadata["choice_candidates"] = choices
    question = template.format(target_ref=target_ref, choice_list=", ".join(choices))
    answer_text = _pick_answer_template(answer_templates, "seam_continuity.choice", template_seed, answer=answer)
    return question, answer, answer_text, metadata, None, template_key, index


def _build_scene_understanding_stub(task: Dict[str, Any]) -> Tuple[str, Any, str, Dict[str, Any], Optional[Dict[str, Any]], str, int]:
    # scene_understanding 当前按用户要求只保留任务名，不直接生成。
    return "", None, "", {"reserved_task_family": task["task_family"]}, None, "scene_understanding", 0


def _build_cognitive_map_understanding_stub(task: Dict[str, Any]) -> Tuple[str, Any, str, Dict[str, Any], Optional[Dict[str, Any]], str, int]:
    return "", None, "", {"reserved_task_family": task["task_family"]}, None, "cognitive_map_understanding", 0


def _build_hps_path_selection_stub(task: Dict[str, Any]) -> Tuple[str, Any, str, Dict[str, Any], Optional[Dict[str, Any]], str, int]:
    return "", None, "", {"reserved_task_family": task["task_family"]}, None, "hps_path_selection", 0


def _build_rotation_consistency_stub(task: Dict[str, Any]) -> Tuple[str, Any, str, Dict[str, Any], Optional[Dict[str, Any]], str, int]:
    return "", None, "", {"reserved_task_family": task["task_family"]}, None, "rotation_consistency", 0


def _pick_template(templates: Dict[str, List[str]], key: str, seed: str) -> Tuple[str, int]:
    variants = templates.get(key)
    if not variants:
        raise KeyError(f"missing template key: {key}")
    digest = hashlib.md5(f"{key}:{seed}".encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(variants)
    return variants[index], index


def _pick_answer_template(answer_templates: Dict[str, List[str]], key: str, seed: str, **kwargs: Any) -> str:
    variants = answer_templates.get(key)
    if not variants:
        if "truth" in kwargs:
            return str(kwargs["truth"])
        if "count" in kwargs:
            return str(kwargs["count"])
        if "label" in kwargs:
            return str(kwargs["label"])
        if "relation" in kwargs:
            return str(kwargs["relation"])
        return ""
    digest = hashlib.md5(f"answer:{key}:{seed}".encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(variants)
    return variants[index].format(**kwargs)

def _normalized_bbox_1000(scene: SceneMetadata, entity: Entity) -> List[int]:
    width = max(scene.erp_width, 1)
    height = max(scene.erp_height, 1)
    x1, y1, x2, y2 = entity.bbox_erp
    return [
        int(round((float(x1) / width) * 1000.0)),
        int(round((float(y1) / height) * 1000.0)),
        int(round((float(x2) / width) * 1000.0)),
        int(round((float(y2) / height) * 1000.0)),
    ]


def _entity_loc_metadata(scene: SceneMetadata, entity: Entity) -> Dict[str, Any]:
    return {
        "target_label": entity.label,
        "entity_ref": _grounding_ref(entity),
        "bbox_norm_1000": _normalized_bbox_1000(scene, entity),
        "yaw_deg": round(_yaw_deg_360(entity), 1),
        "pitch_deg": round(_pitch_deg_180(entity), 1),
        "bfov": _entity_bfov(scene, entity),
        "lon_deg": round(entity.lon_deg, 2),
        "lat_deg": round(entity.lat_deg, 2),
    }


def _pair_metadata(scene: SceneMetadata, entity_a: Entity, entity_b: Entity) -> Dict[str, Any]:
    return {
        "entity_a": entity_a.entity_id,
        "entity_b": entity_b.entity_id,
        "entity_a_label": entity_a.label,
        "entity_a_ref": _grounding_ref(entity_a),
        "entity_b_label": entity_b.label,
        "entity_b_ref": _grounding_ref(entity_b),
        "depth_a_m": round(float(entity_a.entity_center_depth), 2) if entity_a.entity_center_depth is not None else None,
        "depth_b_m": round(float(entity_b.entity_center_depth), 2) if entity_b.entity_center_depth is not None else None,
        "yaw_a_deg": round(_yaw_deg_360(entity_a), 1),
        "yaw_b_deg": round(_yaw_deg_360(entity_b), 1),
        "pitch_a_deg": round(_pitch_deg_180(entity_a), 1),
        "pitch_b_deg": round(_pitch_deg_180(entity_b), 1),
        "bfov_a": _entity_bfov(scene, entity_a),
        "bfov_b": _entity_bfov(scene, entity_b),
        "bbox_a_norm_1000": _normalized_bbox_1000(scene, entity_a),
        "bbox_b_norm_1000": _normalized_bbox_1000(scene, entity_b),
    }


def _entity_bfov(scene: SceneMetadata, entity: Entity) -> List[float]:
    resolved = entity.resolved_bfov
    if resolved is not None:
        return [round(resolved[0], 1), round(resolved[1], 1), round(resolved[2], 1), round(resolved[3], 1)]
    bbox = entity.bbox_erp
    if len(bbox) == 4 and scene.erp_width and scene.erp_height:
        x1, y1, x2, y2 = bbox
        width = abs(float(x2) - float(x1))
        height = abs(float(y2) - float(y1))
        return [
            round(_yaw_deg_360(entity), 1),
            round(_pitch_deg_180(entity), 1),
            round((width / max(scene.erp_width, 1)) * 360.0, 1),
            round((height / max(scene.erp_height, 1)) * 180.0, 1),
        ]
    return [round(_yaw_deg_360(entity), 1), round(_pitch_deg_180(entity), 1), 0.0, 0.0]


def _bfov_text(bfov: Sequence[float]) -> str:
    return f"[{bfov[0]}, {bfov[1]}, {bfov[2]}, {bfov[3]}]"


def _resolve_seam_choice_entities(task: Dict[str, Any], entity_by_id: Dict[str, Entity]) -> List[Entity]:
    entities: List[Entity] = []
    for entity_id in task.get("seam_choice_entity_ids", []) or []:
        entity = entity_by_id.get(str(entity_id))
        if entity is None:
            continue
        entities.append(entity)
    return entities


def _preferred_seam_target_side(scene: SceneMetadata, entity: Entity) -> str:
    contacts = _entity_boundary_contacts(scene, entity)
    if contacts == {"left"}:
        return "left"
    if contacts == {"right"}:
        return "right"
    return "left" if _yaw_deg_360(entity) >= 180.0 else "right"


def _entity_boundary_contacts(scene: SceneMetadata, entity: Entity) -> set[str]:
    contacts: set[str] = set()
    if len(entity.bbox_erp) == 4 and scene.erp_width:
        x1, _, x2, _ = [float(v) for v in entity.bbox_erp]
        margin = max(12.0, scene.erp_width * 0.03)
        if x1 <= margin:
            contacts.add("left")
        if x2 >= (scene.erp_width - margin):
            contacts.add("right")
    if not contacts and bool(entity.seam_crossing_flag):
        contacts.add("left" if _yaw_deg_360(entity) >= 180.0 else "right")
    return contacts


def _choose_attribute(entity: Entity) -> Tuple[str, Any]:
    preferred = ["color", "material", "pattern", "state", "size", "condition", "shape"]
    for key in preferred:
        value = entity.semantic.attributes.get(key)
        if value not in (None, "", [], {}):
            return key, value
    return "category", entity.label


def _grounding_ref(entity: Entity) -> str:
    return entity.semantic.reground_query or f"the {_display_label(entity.label)}"


def _entity_hint(
    entity: Entity,
    *,
    exclude_attribute: Optional[str] = None,
    allow_label: bool = True,
) -> str:
    attrs = entity.semantic.attributes or {}
    parts: List[str] = []

    ordered_keys = [
        "shape",
        "material",
        "pattern",
        "condition",
        "surface",
        "legs",
        "frame_color",
        "parts",
        "objects_on_top",
        "view",
        "text",
    ]
    if exclude_attribute != "color":
        ordered_keys.insert(0, "color")

    for key in ordered_keys:
        if key == exclude_attribute:
            continue
        value = attrs.get(key)
        text = _attribute_value_to_hint(key, value)
        if text and text not in parts:
            parts.append(text)
        if len(parts) >= 3:
            break

    if allow_label:
        label_text = _display_label(entity.label)
        label_piece = _non_overlapping_label_phrase(label_text, parts)
        if label_piece:
            parts.append(label_piece)

    if not parts:
        reground = (entity.semantic.reground_query or "").strip()
        if reground:
            parts.append(_display_label(reground))

    if not parts:
        return "the target object"

    phrase = " ".join(parts[:4]).strip()
    if not phrase.startswith("the "):
        phrase = f"the {phrase}"
    return phrase


def _non_overlapping_label_phrase(label_text: str, parts: Sequence[str]) -> str:
    if not label_text:
        return ""
    existing_tokens = set()
    for part in parts:
        for token in _display_label(part).split():
            existing_tokens.add(token)
    label_tokens = _display_label(label_text).split()
    remaining = [token for token in label_tokens if token not in existing_tokens]
    if remaining:
        return " ".join(remaining)
    return label_text if label_text not in parts else ""


def _attribute_value_to_hint(key: str, value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, list):
        items = [_display_label(str(item)) for item in value if str(item).strip()]
        if not items:
            return ""
        joined = ", ".join(items[:2])
        if key == "parts":
            return f"with {joined}"
        if key == "objects_on_top":
            return f"with {joined} on top"
        return joined
    text = _display_label(str(value))
    if not text:
        return ""
    if key in {"legs", "surface", "view", "text", "frame_color"}:
        return text
    if key == "parts":
        return f"with {text}"
    if key == "objects_on_top":
        return f"with {text} on top"
    return text


def _yaw_deg_360(entity: Entity) -> float:
    return entity.lon_deg % 360.0


def _pitch_deg_180(entity: Entity) -> float:
    return max(0.0, min(180.0, 90.0 - entity.lat_deg))


def _cardinal_direction_from_yaw(yaw_deg: float) -> str:
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
    idx = int(((yaw_deg % 360.0) + 22.5) // 45.0) % 8
    return sectors[idx]


def _panoramic_ring_relation(reference: Entity, target: Entity, *, opposite_label: str) -> Optional[str]:
    delta_yaw = _wrapped_delta_deg(_yaw_deg_360(target) - _yaw_deg_360(reference))
    return _panoramic_relation_from_delta(delta_yaw, opposite_label=opposite_label)


def _panoramic_relation_from_delta(delta_yaw: float, *, opposite_label: str) -> Optional[str]:
    if abs(delta_yaw) < 15.0:
        return None
    if 15.0 <= delta_yaw < 90.0:
        return "right"
    if 90.0 <= delta_yaw < 150.0:
        return "back-right"
    if delta_yaw >= 150.0 or delta_yaw < -150.0:
        return opposite_label
    if -150.0 <= delta_yaw < -90.0:
        return "back-left"
    return "left"


def _depth_relation(entity_a: Entity, entity_b: Entity) -> str:
    depth_a = float(entity_a.entity_center_depth)
    depth_b = float(entity_b.entity_center_depth)
    if abs(depth_a - depth_b) < 0.35:
        return "similar_depth"
    return "closer" if depth_a < depth_b else "farther"


def _relative_3d_relation(entity_a: Entity, entity_b: Entity) -> str:
    xyz_a = entity_a.erp_consistent_xyz_camera
    xyz_b = entity_b.erp_consistent_xyz_camera
    if xyz_a is None or xyz_b is None:
        return _depth_relation(entity_a, entity_b)
    dx = float(xyz_a[0]) - float(xyz_b[0])
    dy = float(xyz_a[1]) - float(xyz_b[1])
    dz = float(xyz_a[2]) - float(xyz_b[2])
    relations: List[str] = []
    if _axis_clear_x(entity_a, entity_b, dx):
        relations.append("right of" if dx > 0 else "left of")
    if _axis_clear_y(entity_a, entity_b, dy):
        relations.append("above" if dy > 0 else "below")
    if abs(dz) >= 0.6:
        relations.append("in front of" if dz > 0 else "behind")
    if not relations:
        return ""
    if len(relations) == 1:
        return relations[0]
    if len(relations) == 2:
        return f"{relations[0]} and {relations[1]}"
    return f"{relations[0]}, {relations[1]}, and {relations[2]}"


def _build_relative_3d_choices(entity_a: Entity, entity_b: Entity, answer: str) -> List[str]:
    xyz_a = entity_a.erp_consistent_xyz_camera
    xyz_b = entity_b.erp_consistent_xyz_camera
    if xyz_a is None or xyz_b is None:
        return [answer, "left of", "right of", "in front of"]

    dx = float(xyz_a[0]) - float(xyz_b[0])
    dy = float(xyz_a[1]) - float(xyz_b[1])
    dz = float(xyz_a[2]) - float(xyz_b[2])

    axis_pairs = [
        ("left of", "right of") if dx < 0 else ("right of", "left of"),
        ("below", "above") if dy < 0 else ("above", "below"),
        ("in front of", "behind") if dz > 0 else ("behind", "in front of"),
    ]
    active = []
    if _axis_clear_x(entity_a, entity_b, dx):
        active.append(axis_pairs[0])
    if _axis_clear_y(entity_a, entity_b, dy):
        active.append(axis_pairs[1])
    if abs(dz) >= 0.6:
        active.append(axis_pairs[2])

    choices = [answer]
    if active:
        # 先对每个激活轴做一次“翻转”，得到最自然的干扰项。
        for idx in range(len(active)):
            parts = [pair[0] for pair in active]
            parts[idx] = active[idx][1]
            choices.append(_join_relations(parts))
    # 再补一些原子关系，保证至少 4 个候选项。
    fallback_pool = [
        "left of",
        "right of",
        "above",
        "below",
        "in front of",
        "behind",
    ]
    for item in fallback_pool:
        if item not in choices:
            choices.append(item)
        if len(choices) >= 4:
            break
    return choices[:4]


def _join_relations(relations: Sequence[str]) -> str:
    relations = [item for item in relations if item]
    if not relations:
        return ""
    if len(relations) == 1:
        return relations[0]
    if len(relations) == 2:
        return f"{relations[0]} and {relations[1]}"
    return f"{relations[0]}, {relations[1]}, and {relations[2]}"


def _axis_clear_x(entity_a: Entity, entity_b: Entity, dx: float) -> bool:
    base = 0.35
    radius_sum = _approx_axis_radius(entity_a, axis="x") + _approx_axis_radius(entity_b, axis="x")
    return abs(dx) >= max(base, radius_sum)


def _axis_clear_y(entity_a: Entity, entity_b: Entity, dy: float) -> bool:
    base = 0.25
    radius_sum = _approx_axis_radius(entity_a, axis="y") + _approx_axis_radius(entity_b, axis="y")
    return abs(dy) >= max(base, radius_sum)


def _approx_axis_radius(entity: Entity, axis: str) -> float:
    bfov = entity.resolved_bfov
    depth = entity.entity_center_depth
    if bfov is None or depth is None:
        return 0.0
    import math

    fov = float(bfov[2] if axis == "x" else bfov[3])
    if fov <= 0:
        return 0.0
    return float(depth) * math.tan(math.radians(fov / 2.0))


def _center_distance_3d(entity_a: Entity, entity_b: Entity) -> float:
    xyz_a = entity_a.resolved_xyz_camera
    xyz_b = entity_b.resolved_xyz_camera
    if xyz_a is None or xyz_b is None:
        if entity_a.entity_center_depth is None or entity_b.entity_center_depth is None:
            return 999.0
        return abs(float(entity_a.entity_center_depth) - float(entity_b.entity_center_depth))
    return (
        (float(xyz_a[0]) - float(xyz_b[0])) ** 2
        + (float(xyz_a[1]) - float(xyz_b[1])) ** 2
        + (float(xyz_a[2]) - float(xyz_b[2])) ** 2
    ) ** 0.5


def _safe_atan2(y: float, x: float) -> float:
    import math

    return math.atan2(y, x)


def degrees_from_radians(value: float) -> float:
    import math

    return math.degrees(value)


def _wrapped_delta_deg(delta: float) -> float:
    delta = ((delta + 180.0) % 360.0) - 180.0
    if delta == -180.0:
        return 180.0
    return delta


def _bbox_text(bbox: Sequence[int]) -> str:
    return f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"


def _fmt_float(value: float) -> str:
    return f"{round(float(value), 1):.1f}"


def _display_label(text: str) -> str:
    return str(text).replace("_", " ")


def _display_value(value: Any) -> str:
    if isinstance(value, str):
        return _display_label(value)
    if isinstance(value, list):
        return ", ".join(_display_value(item) for item in value)
    return str(value)


def _normalize_phrase(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def _pluralize_label(label: str) -> str:
    label = _display_label(label)
    if label.endswith("s"):
        return label
    if label.endswith("y") and len(label) > 1 and label[-2] not in "aeiou":
        return label[:-1] + "ies"
    return label + "s"


def _cap_scene_samples(samples: List[Dict[str, Any]], max_samples: int = 20) -> List[Dict[str, Any]]:
    family_budget = {
        "caption": 2,
        "existence": 2,
        "counting": 1,
        "grounding": 2,
        "direct_direction": 2,
        "distance_estimation": 3,
        "relative_direction": 2,
        "relative_3d_position": 2,
        "view_transform": 2,
        "seam_continuity": 1,
        "polar_distortion_awareness": 1,
    }
    per_origin_limit = {
        "caption": 1,
        "grounding": 1,
        "direct_direction": 1,
        "distance_estimation": 1,
        "relative_direction": 1,
        "relative_3d_position": 1,
    }
    mode_bonus = {
        ("caption", "dense_description_like"): 0.30,
    }

    def _score(sample: Dict[str, Any]) -> float:
        digest = hashlib.md5(sample["sample_id"].encode("utf-8")).hexdigest()
        base = int(digest[:8], 16) / 0xFFFFFFFF
        return base + mode_bonus.get((sample["task_family"], sample.get("generation_mode") or ""), 0.0)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["task_family"]].append(sample)

    selected: List[Dict[str, Any]] = []
    for family, items in grouped.items():
        budget = family_budget.get(family, 0)
        if budget <= 0:
            continue
        ranked = sorted(items, key=_score, reverse=True)
        chosen: List[Dict[str, Any]] = []
        origin_counts: Dict[str, int] = defaultdict(int)
        limit = per_origin_limit.get(family)
        for sample in ranked:
            origin = sample.get("origin_anchor_id") or sample.get("task_origin") or sample["sample_id"]
            if limit is not None and origin_counts[origin] >= limit:
                continue
            chosen.append(sample)
            origin_counts[origin] += 1
            if len(chosen) >= budget:
                break
        selected.extend(chosen)

    selected = sorted(selected, key=_score, reverse=True)[:max_samples]
    selected.sort(key=lambda sample: sample["sample_id"])
    return selected
