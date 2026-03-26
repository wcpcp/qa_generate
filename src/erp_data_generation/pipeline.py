from __future__ import annotations

import json
from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .entity_selector import (
    choose_relation_partners,
    infer_pole_proximity,
    infer_seam_adjacency,
    select_anchor_entities,
    summarize_label_distribution,
)
from .schemas import Entity, SceneMetadata
from .task_registry import TASK_REGISTRY, TaskDefinition


BASE_SCENE_FIELDS = ["scene_id", "erp_image_path", "erp_width", "erp_height", "entities"]
BASE_ENTITY_FIELDS = [
    "entity_id",
    "bbox_erp",
    "mask_rle",
    "lon_lat",
    "area_ratio",
    "confidence",
    "support_views",
    "semantic.identify",
    "semantic.attributes",
    "semantic.caption_brief",
    "semantic.reground_query",
]
RAW_BASE_SCENE_FIELDS = ["scene_id", "image_path", "viewpoint_id", "entities", "quality_stats"]
RAW_BASE_ENTITY_FIELDS = [
    "entity_id",
    "representative_view_id",
    "confidence",
    "bbox_erp",
    "mask_rle",
    "lon_lat",
    "area_ratio",
    "source_views",
    "support_views",
    "best_score",
    "semantic.identify",
    "semantic.attributes",
    "semantic.caption_brief",
    "semantic.reground_query",
    "semantic.confidence",
    "local_reground.status",
    "local_reground.query",
    "local_reground.consistency_iou",
    "local_reground.passed",
    "depth.status",
    "depth.median_m",
    "depth.valid_ratio",
    "spatial.yaw_deg",
    "spatial.pitch_deg",
    "spatial.xyz_camera_m",
    "spatial.range_m",
]
OPTIONAL_SCENE_FIELDS = ["depth_map_path", "scene_global_tags", "depth_source"]
OPTIONAL_ENTITY_FIELDS = [
    "entity_center_depth",
    "semantic_verification_passed",
    "semantic_verification_iou",
    "projection_iou",
    "semantic_quality_score",
    "entity_uniqueness_score",
    "occlusion_flag",
    "truncation_flag",
    "seam_crossing_flag",
    "pole_proximity_flag",
]
ADVANCED_SCENE_FIELDS = ["camera_convention", "room_layout_proxy", "free_space_map", "openings"]
ADVANCED_ENTITY_FIELDS = ["entity_xyz_camera", "entity_extent_3d"]
NEGATIVE_EXISTENCE_CANDIDATES = [
    "microwave",
    "bathtub",
    "toilet",
    "stove",
    "washing machine",
    "television",
    "laptop",
    "keyboard",
    "mouse",
    "printer",
    "desk lamp",
    "bookshelf",
    "wardrobe",
    "nightstand",
    "ceiling fan",
    "air conditioner",
    "shoe rack",
    "guitar",
    "bicycle",
    "car",
    "motorcycle",
    "dog",
    "cat",
    "person",
    "baby stroller",
    "sink",
    "oven",
    "dishwasher",
    "trash can",
    "whiteboard",
]
PANORAMIC_RELATION_MIN_DELTA_DEG = 15.0
CAMERA_ROTATION_OPTIONS = [
    ("right", 90),
    ("left", 90),
    ("right", 135),
    ("left", 135),
    ("right", 180),
    ("left", 180),
]


def load_scene_metadata(input_path: str) -> SceneMetadata:
    # 从磁盘读取一个 scene metadata JSON，并归一化为内部结构。
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return SceneMetadata.from_dict(data)


def _field_present(value: Any) -> bool:
    # 统一判断一个字段是否“可用”。
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _nested_value(obj: Any, path: str) -> Any:
    # 按 a.b.c 的路径形式读取嵌套字段，兼容 dict 和 dataclass 对象。
    current = obj
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def _entity_missing(scene: SceneMetadata, field_path: str) -> List[str]:
    # 统计归一化后的运行时实体字段里，哪些 entity 缺这个字段。
    missing = []
    for entity in scene.entities:
        if not _field_present(_nested_value(entity, field_path)):
            missing.append(entity.entity_id)
    return missing


def _raw_entity_missing(scene: SceneMetadata, field_path: str) -> List[str]:
    # 统计原始输入 JSON 中，哪些 entity 缺这个字段。
    missing = []
    raw_entities = scene.raw.get("entities", []) if isinstance(scene.raw, dict) else []
    for index, entity in enumerate(raw_entities):
        if not _field_present(_nested_value(entity, field_path)):
            missing.append(str(entity.get("entity_id", f"entity_{index:04d}")))
    return missing


def _scene_rotation_supported(scene: SceneMetadata) -> bool:
    # 只要 scene 中有实体具备 lon_lat，就可以构造 ERP 内部旋转任务。
    return any(_field_present(entity.lon_lat) for entity in scene.entities)


def _task_feasible_for_scene(task_family: str, scene: SceneMetadata) -> Tuple[bool, List[str], int]:
    # 判断一个 scene-global / scene-rotation 任务是否可行。
    definition = TASK_REGISTRY[task_family]
    if definition.phase == "future":
        return False, ["deferred_to_future_phase"], 0

    missing = [field for field in definition.required_fields if not _field_present(_nested_value(scene, field))]
    return (not missing), missing, int(not missing)


def _task_feasible_for_entity(task_family: str, scene: SceneMetadata, entity: Entity) -> Tuple[bool, List[str]]:
    # 判断一个单实体任务对当前 entity 是否可行。
    definition = TASK_REGISTRY[task_family]
    if definition.phase == "future":
        return False, ["deferred_to_future_phase"]

    if task_family == "seam_continuity":
        # seam_continuity 现在只保留“真正跨 seam 的物体”，
        # 不再把单纯靠近边界的实体也拿来出题。
        supported = bool(entity.seam_crossing_flag)
        blockers = [] if supported else ["seam_crossing_flag"]
        return supported, blockers

    if task_family == "polar_distortion_awareness":
        has_shape = bool((entity.semantic.attributes or {}).get("shape"))
        supported = (bool(entity.pole_proximity_flag) or abs(entity.lat_deg) >= 45.0) and has_shape
        blockers = [] if supported else (["shape_attribute"] if not has_shape else ["polar_distortion_threshold"])
        return supported, blockers

    if task_family == "distance_estimation":
        supported = entity.has_depth
        blockers = [] if supported else ["entity_center_depth"]
        return supported, blockers

    missing = [field for field in definition.required_fields if not _field_present(_nested_value(entity, field))]
    return (not missing), missing


def _task_feasible_for_pair(
    task_family: str,
    scene: SceneMetadata,
    entity_a: Entity,
    entity_b: Entity,
) -> Tuple[bool, List[str], str]:
    # 判断一个 pair 任务在两个实体之间是否可行，并返回几何来源说明。
    definition = TASK_REGISTRY[task_family]
    if definition.phase == "future":
        return False, ["deferred_to_future_phase"], "future"

    if task_family == "relative_direction":
        supported = _field_present(entity_a.lon_lat) and _field_present(entity_b.lon_lat)
        blockers = [] if supported else ["lon_lat"]
        return supported, blockers, "erp_angular"

    if task_family == "relative_3d_position":
        xyz_a = entity_a.resolved_xyz_camera
        xyz_b = entity_b.resolved_xyz_camera
        supported = (xyz_a is not None and xyz_b is not None) or (entity_a.has_depth and entity_b.has_depth)
        blockers = [] if supported else ["entity_center_depth_or_entity_xyz_camera"]
        geometry_source = "explicit_xyz"
        if entity_a.entity_xyz_camera is None or entity_b.entity_xyz_camera is None:
            geometry_source = "depth_relation_only"
        return supported, blockers, geometry_source

    missing = [
        field
        for field in definition.required_fields
        if not _field_present(_nested_value(entity_a, field)) or not _field_present(_nested_value(entity_b, field))
    ]
    return (not missing), missing, "pair"


def assess_task_feasibility(scene: SceneMetadata) -> Dict[str, Any]:
    # 对整个 scene 做逐 task family 的 feasibility 扫描。
    # 输出会被 inspect_metadata 和 scene planning 同时复用。
    report: Dict[str, Any] = {}
    pairs: List[Tuple[Entity, Entity]] = []
    for idx, entity_a in enumerate(scene.entities):
        for entity_b in scene.entities[idx + 1 :]:
            pairs.append((entity_a, entity_b))

    for task_family, definition in TASK_REGISTRY.items():
        if definition.unit == "scene":
            supported, blockers, eligible_count = _task_feasible_for_scene(task_family, scene)
            status = "supported" if supported else ("deferred" if definition.phase == "future" else "blocked")
            report[task_family] = {
                "status": status,
                "phase": definition.phase,
                "metadata_tier": definition.metadata_tier,
                "eligible_count": eligible_count,
                "blocked_by": blockers,
            }
            continue

        if definition.unit == "anchor_entity":
            eligible = []
            blockers = set()
            for entity in scene.entities:
                supported, entity_blockers = _task_feasible_for_entity(task_family, scene, entity)
                if supported:
                    eligible.append(entity.entity_id)
                else:
                    blockers.update(entity_blockers)
            status = "supported" if eligible else ("deferred" if definition.phase == "future" else "blocked")
            report[task_family] = {
                "status": status,
                "phase": definition.phase,
                "metadata_tier": definition.metadata_tier,
                "eligible_count": len(eligible),
                "sample_entity_ids": eligible[:5],
                "blocked_by": [] if eligible else sorted(blockers),
            }
            continue

        if definition.unit == "entity_pair":
            eligible = []
            blockers = set()
            geometry_sources = set()
            for entity_a, entity_b in pairs:
                supported, pair_blockers, geometry_source = _task_feasible_for_pair(task_family, scene, entity_a, entity_b)
                if supported:
                    eligible.append([entity_a.entity_id, entity_b.entity_id])
                    geometry_sources.add(geometry_source)
                else:
                    blockers.update(pair_blockers)
            status = "supported" if eligible else ("deferred" if definition.phase == "future" else "blocked")
            report[task_family] = {
                "status": status,
                "phase": definition.phase,
                "metadata_tier": definition.metadata_tier,
                "eligible_count": len(eligible),
                "sample_pairs": eligible[:5],
                "geometry_sources": sorted(geometry_sources),
                "blocked_by": [] if eligible else sorted(blockers),
            }
            continue

        if definition.unit == "scene_rotation":
            supported = _scene_rotation_supported(scene)
            blockers = [] if supported else ["lon_lat"]
            status = "supported" if supported else ("deferred" if definition.phase == "future" else "blocked")
            report[task_family] = {
                "status": status,
                "phase": definition.phase,
                "metadata_tier": definition.metadata_tier,
                "eligible_count": int(supported),
                "blocked_by": blockers,
            }
            continue

        report[task_family] = {
            "status": "blocked",
            "phase": definition.phase,
            "metadata_tier": definition.metadata_tier,
            "eligible_count": 0,
            "blocked_by": ["unsupported_unit"],
        }

    return report


def audit_metadata(scene: SceneMetadata) -> Dict[str, Any]:
    # 输出一个完整的 metadata 审计报告。
    # 包括原始 contract、归一化 contract、coverage 和 task readiness。
    raw_missing_scene = [field for field in RAW_BASE_SCENE_FIELDS if not _field_present(_nested_value(scene.raw, field))]
    raw_missing_entity = {
        field: _raw_entity_missing(scene, field)[:10]
        for field in RAW_BASE_ENTITY_FIELDS
        if _raw_entity_missing(scene, field)
    }
    base_missing_scene = [field for field in BASE_SCENE_FIELDS if not _field_present(_nested_value(scene, field))]
    base_missing_entity = {
        field: _entity_missing(scene, field)[:10]
        for field in BASE_ENTITY_FIELDS
        if _entity_missing(scene, field)
    }
    optional_missing_scene = [field for field in OPTIONAL_SCENE_FIELDS if not _field_present(_nested_value(scene, field))]
    optional_missing_entity = {
        field: _entity_missing(scene, field)[:10]
        for field in OPTIONAL_ENTITY_FIELDS
        if _entity_missing(scene, field)
    }
    advanced_missing_scene = [field for field in ADVANCED_SCENE_FIELDS if not _field_present(_nested_value(scene, field))]
    advanced_missing_entity = {
        field: _entity_missing(scene, field)[:10]
        for field in ADVANCED_ENTITY_FIELDS
        if _entity_missing(scene, field)
    }

    entities_with_depth = sum(1 for entity in scene.entities if entity.has_depth)
    entities_with_verified_semantics = sum(1 for entity in scene.entities if entity.verified_semantics)
    seam_entities = sum(1 for entity in scene.entities if infer_seam_adjacency(entity, scene))
    pole_entities = sum(1 for entity in scene.entities if infer_pole_proximity(entity))
    explicit_xyz_entities = sum(1 for entity in scene.entities if entity.entity_xyz_camera is not None)
    derived_xyz_entities = sum(1 for entity in scene.entities if entity.resolved_xyz_camera is not None)
    task_readiness = assess_task_feasibility(scene)

    return {
        "scene_id": scene.scene_id,
        "entity_count": len(scene.entities),
        "labels": summarize_label_distribution(scene.entities),
        "raw_input_contract": {
            "scene_fields": {
                "required": RAW_BASE_SCENE_FIELDS,
                "missing": raw_missing_scene,
            },
            "entity_fields": {
                "required": RAW_BASE_ENTITY_FIELDS,
                "missing": raw_missing_entity,
            },
        },
        "metadata_tiers": {
            "level_a_base": {
                "missing_scene_fields": base_missing_scene,
                "missing_entity_fields": base_missing_entity,
            },
            "level_b_optional": {
                "missing_scene_fields": optional_missing_scene,
                "missing_entity_fields": optional_missing_entity,
            },
            "level_c_advanced": {
                "missing_scene_fields": advanced_missing_scene,
                "missing_entity_fields": advanced_missing_entity,
            },
        },
        "coverage": {
            "entities_with_depth": entities_with_depth,
            "entities_with_verified_semantics": entities_with_verified_semantics,
            "seam_adjacent_entities": seam_entities,
            "pole_adjacent_entities": pole_entities,
            "entities_with_explicit_xyz": explicit_xyz_entities,
            "entities_with_resolved_xyz": derived_xyz_entities,
        },
        "phase_readiness": {
            "phase1_base_ready": not base_missing_scene and not base_missing_entity,
            "phase1_depth_ready": entities_with_depth > 0,
            "future_geometry_ready": explicit_xyz_entities > 0 and bool(scene.room_layout_proxy),
        },
        "task_feasibility": task_readiness,
    }


def _make_task(
    task_family: str,
    difficulty: str,
    *,
    entity_ids: Optional[Sequence[str]] = None,
    evidence_fields: Optional[Iterable[str]] = None,
    geometry_source: Optional[str] = None,
    partner_role: Optional[str] = None,
    generation_mode: Optional[str] = None,
    answer_space: Optional[str] = None,
    rotation_angle_deg: Optional[int] = None,
    rotation_direction: Optional[str] = None,
    query_target: Optional[str] = None,
) -> Dict[str, Any]:
    # scene plan 里的每个 task 都统一走这个函数组装。
    # 它会把 task registry 里的静态定义和当前 scene/entity 上下文拼成结构化 task payload。
    definition: TaskDefinition = TASK_REGISTRY[task_family]
    payload: Dict[str, Any] = {
        "task_family": definition.task_family,
        "ability": definition.ability,
        "unit": definition.unit,
        "phase": definition.phase,
        "metadata_tier": definition.metadata_tier,
        "answer_type": definition.answer_type,
        "gt_source": definition.gt_source,
        "required_fields": definition.required_fields,
        "difficulty": difficulty,
        "erp_robustness": definition.erp_robustness,
    }
    if entity_ids:
        payload["entity_ids"] = list(entity_ids)
    if evidence_fields:
        payload["evidence_fields"] = sorted(set(evidence_fields))
    if geometry_source:
        payload["geometry_source"] = geometry_source
    if partner_role:
        payload["partner_role"] = partner_role
    if generation_mode:
        payload["generation_mode"] = generation_mode
    if answer_space:
        payload["answer_space"] = answer_space
    if rotation_angle_deg is not None:
        payload["rotation_angle_deg"] = rotation_angle_deg
    if rotation_direction:
        payload["rotation_direction"] = rotation_direction
    if query_target:
        payload["query_target"] = query_target
    return payload


def _build_global_tasks(scene: SceneMetadata) -> List[Dict[str, Any]]:
    # global task 面向整张 ERP scene，而不是单个实体。
    # 当前严格按照用户定义的 scene-level 子功能：
    # existence / counting / scene_understanding
    # 其中：
    # 1. existence 保留正例和一个 metadata-level 负例。
    #    负例本身不是强真值，后面交给 postprocess prompt 继续重构成更自然的负例问答。
    # 2. counting 只保留 count>=2 的正例，避免把大量 0/1 的弱样本塞进主数据流
    # 3. scene_understanding 暂时只保留任务名，不实际生成
    tasks: List[Dict[str, Any]] = []
    labels = Counter(entity.label for entity in scene.entities)
    if not labels:
        return tasks

    positive_existence_target = sorted(labels.items(), key=lambda item: (item[1], item[0]))[0][0]
    tasks.append(
        _make_task(
            "existence",
            "easy",
            query_target=positive_existence_target,
            evidence_fields=["entities"],
            generation_mode="positive_existence_label",
        )
    )
    common_existence_target = labels.most_common(1)[0][0]
    if common_existence_target != positive_existence_target:
        tasks.append(
            _make_task(
                "existence",
                "easy",
                query_target=common_existence_target,
                evidence_fields=["entities"],
                generation_mode="positive_existence_label",
            )
        )
    for candidate in NEGATIVE_EXISTENCE_CANDIDATES:
        if candidate not in labels:
            tasks.append(
                _make_task(
                    "existence",
                    "medium",
                    query_target=candidate,
                    evidence_fields=["entities"],
                    generation_mode="negative_existence_label",
                )
            )
            break

    multi_count_labels = [(label, count) for label, count in labels.items() if count >= 2]
    multi_count_labels.sort(key=lambda item: (-item[1], item[0]))
    seen_counts = set()
    for label, count in multi_count_labels:
        if count in seen_counts:
            continue
        tasks.append(
            _make_task(
                "counting",
                "medium" if count <= 3 else "hard",
                query_target=label,
                evidence_fields=["entities"],
                generation_mode="multi_instance_count",
            )
        )
        seen_counts.add(count)
        if len(seen_counts) >= 2:
            break
    return tasks


def _anchor_base_difficulty(anchor_item: Dict[str, Any]) -> str:
    # 这是 anchor 级别的粗难度估计，后面一些任务会基于它做 easy / medium / hard 调整。
    entity = anchor_item["entity"]
    if anchor_item["seam_adjacent"] or anchor_item["pole_adjacent"]:
        return "hard"
    if entity.area_ratio < 0.003 or not entity.verified_semantics:
        return "medium"
    return "easy"


def _build_anchor_tasks(scene: SceneMetadata, anchor_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 围绕单个 anchor entity 构建任务。
    # 这里只生成单实体任务，不处理 entity-pair 和 scene-rotation。
    entity = anchor_item["entity"]
    anchor_id = entity.entity_id
    tasks: List[Dict[str, Any]] = []

    # caption 当前收敛成 dense_description_like 主模式。
    # 原因是用户当前更需要高质量、长描述的 object-level caption，
    # 而不是大量 identify / attribute / brief description 的简单问答。
    supported, _ = _task_feasible_for_entity("caption", scene, entity)
    if supported:
        if _should_generate_dense_caption(scene, entity):
            tasks.append(
                _make_task(
                    "caption",
                    "medium",
                    entity_ids=[anchor_id],
                    evidence_fields=["semantic.caption_dense", "semantic.caption_brief", "semantic.attributes", "semantic.identify"],
                    generation_mode="dense_description_like",
                )
            )

    supported, _ = _task_feasible_for_entity("grounding", scene, entity)
    if supported:
        grounding_mode = _sample_grounding_mode(scene.scene_id, anchor_id)
        tasks.append(
            _make_task(
                "grounding",
                "medium",
                entity_ids=[anchor_id],
                evidence_fields=["bbox_erp", "mask_rle", "semantic.reground_query"],
                answer_space="normalized_bbox_and_precise_angles",
                generation_mode=grounding_mode,
            )
        )

    supported, _ = _task_feasible_for_entity("direct_direction", scene, entity)
    if supported:
        mode = _sample_direct_direction_mode(scene.scene_id, anchor_id)
        difficulty = "medium" if mode == "precise_bfov" else "easy"
        tasks.append(
            _make_task(
                "direct_direction",
                difficulty,
                entity_ids=[anchor_id],
                evidence_fields=["lon_lat"],
                answer_space="bfov_or_absolute_sector_label",
                generation_mode=mode,
            )
        )

    supported, _ = _task_feasible_for_entity("distance_estimation", scene, entity)
    if supported:
        distance_mode = _sample_distance_estimation_mode(scene.scene_id, anchor_id, _select_distance_choice_entities(scene, entity) is not None)
        if distance_mode == "absolute_depth_meter":
            tasks.append(
                _make_task(
                    "distance_estimation",
                    "medium" if _anchor_base_difficulty(anchor_item) != "hard" else "hard",
                    entity_ids=[anchor_id],
                    evidence_fields=["entity_center_depth"],
                    answer_space="depth_meter_value",
                    generation_mode=distance_mode,
                )
            )

    supported, _ = _task_feasible_for_entity("seam_continuity", scene, entity)
    if supported:
        seam_mode = _sample_seam_continuity_mode(scene.scene_id, anchor_id)
        tasks.append(
            _make_task(
                "seam_continuity",
                "hard",
                entity_ids=[anchor_id],
                evidence_fields=["bbox_erp", "mask_rle", "seam_crossing_flag"],
                generation_mode=seam_mode,
            )
        )

    supported, _ = _task_feasible_for_entity("polar_distortion_awareness", scene, entity)
    if supported:
        polar_mode = _sample_polar_shape_mode(scene.scene_id, anchor_id, abs(entity.lat_deg))
        tasks.append(
            _make_task(
                "polar_distortion_awareness",
                "hard",
                entity_ids=[anchor_id],
                evidence_fields=["bfov", "lon_lat", "semantic.attributes.shape"],
                generation_mode=polar_mode,
            )
        )

    return tasks


def _should_generate_dense_caption(scene: SceneMetadata, entity: Entity) -> bool:
    # dense caption 只对“更有辨识度”的物体生成：
    # 1. 必须有可用的 caption_dense
    # 2. 当前 scene 内同类数量尽量少，默认只对单例类别开放
    # 3. caption_dense 需要有一定长度，避免只是换皮 brief caption
    # 4. 目标不要过小，避免仅靠一个模糊小框要求详细描述
    dense = (entity.semantic.caption_dense or "").strip()
    if not dense:
        return False
    if len(dense.split()) < 12:
        return False
    label_counts = Counter(item.label for item in scene.entities)
    if label_counts.get(entity.label, 0) > 1:
        return False
    if entity.area_ratio < 0.002:
        return False
    return True


def _sample_direct_direction_mode(scene_id: str, entity_id: str) -> str:
    # 对每个实体稳定随机地在两种绝对定位模式中选一种：
    # 1. precise_bfov
    # 2. absolute_sector_8way
    digest = hashlib.md5(f"{scene_id}:direct_direction:{entity_id}".encode("utf-8")).hexdigest()
    return "precise_bfov" if int(digest[:8], 16) % 2 == 0 else "absolute_sector_8way"


def _sample_grounding_mode(scene_id: str, entity_id: str) -> str:
    # grounding 也采用“先平等抽问题种类，再在该种类下抽模板”的策略。
    # 这样 full / bbox_only / angles_only 的出现概率由题种控制，
    # 不会被模板数量偷偷带偏。
    modes = ["full", "bbox_only", "angles_only"]
    digest = hashlib.md5(f"{scene_id}:grounding:{entity_id}".encode("utf-8")).hexdigest()
    return modes[int(digest[:8], 16) % len(modes)]


def _sample_distance_estimation_mode(scene_id: str, entity_id: str, has_choice_mode: bool) -> str:
    # distance_estimation 统一采用“先抽题种、再抽模板”的逻辑。
    modes = ["absolute_depth_meter"]
    if has_choice_mode:
        modes.extend(["observer_nearest_choice", "observer_nearest_choice", "candidate_nearest_choice"])
    digest = hashlib.md5(f"{scene_id}:distance_estimation:{entity_id}".encode("utf-8")).hexdigest()
    return modes[int(digest[:8], 16) % len(modes)]


def _sample_seam_continuity_mode(scene_id: str, entity_id: str) -> str:
    modes = [
        "same_instance_yesno",
        "counterpart_boundary_side",
        "wrap_explanation",
        "rotation_continuity",
    ]
    digest = hashlib.md5(f"{scene_id}:seam_continuity:{entity_id}".encode("utf-8")).hexdigest()
    return modes[int(digest[:8], 16) % len(modes)]


def _sample_polar_shape_mode(scene_id: str, entity_id: str, abs_lat_deg: float) -> str:
    # 高纬度强畸变区域同时支持“带畸变提示”和“不带提示”的形状题；
    # 45-60 度之间只做普通真实形状题，避免把畸变说得过强。
    if abs_lat_deg < 60.0:
        return "shape_recovery_direct"
    modes = ["shape_recovery_direct", "shape_recovery_distortion_aware"]
    digest = hashlib.md5(f"{scene_id}:polar_shape:{entity_id}".encode("utf-8")).hexdigest()
    return modes[int(digest[:8], 16) % len(modes)]


def _build_contextual_anchor_tasks(scene: SceneMetadata, anchor_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 围绕 anchor 再构造需要额外上下文的任务：
    # 1. 多候选最近物体选择型 distance_estimation
    anchor = anchor_item["entity"]
    tasks: List[Dict[str, Any]] = []

    candidate_group = _select_distance_choice_entities(scene, anchor)
    observer_group = _select_observer_distance_choice_entities(scene, anchor)
    has_choice_mode = candidate_group is not None or observer_group is not None
    distance_mode = _sample_distance_estimation_mode(scene.scene_id, anchor.entity_id, has_choice_mode)
    if candidate_group is not None and distance_mode == "candidate_nearest_choice":
        reference, candidates = candidate_group
        tasks.append(
            _make_task(
                "distance_estimation",
                "hard",
                entity_ids=[reference.entity_id] + [entity.entity_id for entity in candidates],
                evidence_fields=["entity_center_depth", "lon_lat", "entity_xyz_camera"],
                answer_space="candidate_label",
                generation_mode="candidate_nearest_choice",
            )
        )
    elif observer_group is not None and distance_mode == "observer_nearest_choice":
        tasks.append(
            _make_task(
                "distance_estimation",
                "medium",
                entity_ids=[entity.entity_id for entity in observer_group],
                evidence_fields=["entity_center_depth"],
                answer_space="candidate_label",
                generation_mode="observer_nearest_choice",
            )
        )

    return tasks


def _pair_task_difficulty(anchor_item: Dict[str, Any], partner_payload: Dict[str, Any], task_family: str) -> str:
    # pair 任务的难度由几类 hard case 触发：
    # 同类干扰、跨 seam、小目标、角度太近、深度差太小等。
    if partner_payload["role"] == "same_category_distractor":
        return "hard"
    if anchor_item["seam_adjacent"] or partner_payload["entity"].area_ratio < 0.003:
        return "hard"
    if task_family == "depth_ordering" and partner_payload.get("depth_gap", 0.0) < 0.75:
        return "hard"
    if partner_payload.get("angular_gap_deg", 999.0) < 18.0:
        return "hard"
    return "medium"


def _build_pair_tasks(scene: SceneMetadata, anchor_item: Dict[str, Any], partners: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # pairwise 任务当前包含：
    # 1. relative_direction: panoramic angular relation on the 360 ring
    # 2. relative_3d_position
    # 3. view_transform.object_conditioned_reorientation
    anchor = anchor_item["entity"]
    tasks: List[Dict[str, Any]] = []
    for index, partner_payload in enumerate(partners[:2]):
        partner = partner_payload["entity"]
        pair_ids = [anchor.entity_id, partner.entity_id]
        relation = _panoramic_ring_relation_from_yaws(_yaw_deg_360(anchor), _yaw_deg_360(partner), opposite_label="opposite")
        if index == 0 and relation is not None:
            pair_difficulty = _pair_task_difficulty(anchor_item, partner_payload, "relative_direction")
            tasks.append(
                _make_task(
                    "relative_direction",
                    pair_difficulty,
                    entity_ids=pair_ids,
                    evidence_fields=["lon_lat"],
                    partner_role=partner_payload["role"],
                    answer_space="panoramic_ring_relation_label",
                    generation_mode="panoramic_angular_relation",
                )
            )
            reoriented_relation = _panoramic_ring_relation_from_yaws(
                _yaw_deg_360(anchor),
                _yaw_deg_360(partner),
                opposite_label="behind",
            )
            if reoriented_relation is not None:
                tasks.append(
                    _make_task(
                        "view_transform",
                        "hard",
                        entity_ids=pair_ids,
                        evidence_fields=["bfov", "lon_lat"],
                        partner_role=partner_payload["role"],
                        answer_space="reoriented_view_direction_label",
                        generation_mode="object_conditioned_reorientation",
                    )
                )

        supported, _, geometry_source = _task_feasible_for_pair("relative_3d_position", scene, anchor, partner)
        if supported and _has_clear_relative_3d_relation(anchor, partner):
            evidence_fields = ["entity_center_depth", "lon_lat"]
            if geometry_source == "explicit_xyz":
                evidence_fields.append("entity_xyz_camera")
            relation_mode = _sample_relative_3d_mode(scene.scene_id, anchor.entity_id, partner.entity_id)
            tasks.append(
                _make_task(
                    "relative_3d_position",
                    "hard" if geometry_source == "explicit_xyz" else "medium",
                    entity_ids=pair_ids,
                    evidence_fields=evidence_fields,
                    partner_role=partner_payload["role"],
                    geometry_source=geometry_source,
                    answer_space="distance_or_camera_centric_relation",
                    generation_mode=relation_mode,
                )
            )
    return tasks


def _has_clear_relative_3d_relation(entity_a: Entity, entity_b: Entity) -> bool:
    # relative_3d_position 现在采用“多轴保留”的关系表达：
    # 只要某个轴差值超过阈值，就把该轴对应的相对关系写进答案。
    # 如果三个轴都不明显，才视为近似重合，这类样本直接不出题。
    xyz_a = entity_a.resolved_xyz_camera
    xyz_b = entity_b.resolved_xyz_camera
    if xyz_a is None or xyz_b is None:
        return False

    dx = abs(float(xyz_a[0]) - float(xyz_b[0]))
    dy = abs(float(xyz_a[1]) - float(xyz_b[1]))
    dz = abs(float(xyz_a[2]) - float(xyz_b[2]))
    return _axis_clear_x(entity_a, entity_b, dx) or _axis_clear_y(entity_a, entity_b, dy) or dz >= 0.6


def _sample_relative_3d_mode(scene_id: str, entity_a_id: str, entity_b_id: str) -> str:
    # relative_3d_position 在开放问答和选择题之间平等随机抽取题种。
    modes = ["camera_centric_open", "camera_centric_choice"]
    digest = hashlib.md5(f"{scene_id}:relative_3d_position:{entity_a_id}:{entity_b_id}".encode("utf-8")).hexdigest()
    return modes[int(digest[:8], 16) % len(modes)]


def _build_rotation_tasks(scene: SceneMetadata, anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 相机旋转型 view_transform：
    # 显式给定观察者旋转角度后，要求判断目标在新视角中的方向。
    if not anchors or not _scene_rotation_supported(scene):
        return []
    candidate_tasks: List[Tuple[str, int, str, Entity]] = []
    for anchor_item in anchors:
        entity = anchor_item["entity"]
        for rotation_direction, angle_deg in CAMERA_ROTATION_OPTIONS:
            relation = _camera_rotation_relation(entity, rotation_direction, angle_deg)
            if relation is None:
                continue
            digest = hashlib.md5(
                f"{scene.scene_id}:camera_rotation:{entity.entity_id}:{rotation_direction}:{angle_deg}".encode("utf-8")
            ).hexdigest()
            candidate_tasks.append((digest, angle_deg, rotation_direction, entity))
    if not candidate_tasks:
        return []
    _, angle_deg, rotation_direction, entity = sorted(candidate_tasks, key=lambda item: item[0])[0]
    return [
        _make_task(
            "view_transform",
            "hard",
            entity_ids=[entity.entity_id],
            evidence_fields=["bfov", "lon_lat"],
            answer_space="reoriented_view_direction_label",
            rotation_angle_deg=angle_deg,
            rotation_direction=rotation_direction,
            generation_mode="camera_rotation_transform",
        )
    ]


def _select_distance_choice_entities(
    scene: SceneMetadata,
    reference: Entity,
    *,
    max_candidates: int = 4,
) -> Optional[Tuple[Entity, List[Entity]]]:
    # 为多候选最近距离题选择 reference 和若干候选物体。
    # 当前使用 center-to-center 3D distance proxy。
    reference_xyz = reference.resolved_xyz_camera
    if reference_xyz is None:
        return None

    ranked: List[Tuple[float, Entity]] = []
    for entity in scene.entities:
        if entity.entity_id == reference.entity_id:
            continue
        xyz = entity.resolved_xyz_camera
        if xyz is None:
            continue
        distance = _euclidean_distance(reference_xyz, xyz)
        if distance < 0.25:
            continue
        ranked.append((distance, entity))
    ranked.sort(key=lambda item: item[0])
    if len(ranked) < 3:
        return None

    unique_ranked: List[Tuple[float, Entity]] = []
    used_labels = {reference.label}
    for distance, entity in ranked:
        if entity.label in used_labels:
            continue
        unique_ranked.append((distance, entity))
        used_labels.add(entity.label)
    if len(unique_ranked) < 3:
        return None

    nearest_distance, nearest_entity = unique_ranked[0]
    selected: List[Tuple[float, Entity]] = [(nearest_distance, nearest_entity)]
    for distance, entity in unique_ranked[1:]:
        # 避免次近候选和最近候选距离过于接近导致 argmin 题含糊。
        if abs(distance - nearest_distance) < 0.15:
            continue
        selected.append((distance, entity))
        if len(selected) >= max_candidates:
            break
    if len(selected) < 3:
        return None
    return reference, [entity for _, entity in selected]


def _select_observer_distance_choice_entities(
    scene: SceneMetadata,
    focus_entity: Entity,
    *,
    max_candidates: int = 4,
) -> Optional[List[Entity]]:
    # observer/camera reference 的选择题直接用深度排序，不依赖 xyz。
    # 候选里固定包含当前 focus_entity，再补若干深度差异足够明显的不同类别物体。
    if focus_entity.entity_center_depth is None:
        return None

    ranked: List[Tuple[float, Entity]] = []
    for entity in scene.entities:
        if entity.entity_center_depth is None:
            continue
        ranked.append((float(entity.entity_center_depth), entity))
    ranked.sort(key=lambda item: item[0])

    selected: List[Tuple[float, Entity]] = []
    used_labels = set()
    focus_added = False
    for depth, entity in ranked:
        if entity.label in used_labels:
            continue
        if entity.entity_id == focus_entity.entity_id and not focus_added:
            selected.append((depth, entity))
            used_labels.add(entity.label)
            focus_added = True
            continue
        if selected and abs(depth - selected[0][0]) < 0.2 and entity.entity_id != focus_entity.entity_id:
            continue
        selected.append((depth, entity))
        used_labels.add(entity.label)
        if len(selected) >= max_candidates:
            break

    if len(selected) < 3:
        return None
    return [entity for _, entity in selected[:max_candidates]]


def _euclidean_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 + (float(a[2]) - float(b[2])) ** 2) ** 0.5


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


def _panoramic_ring_relation_from_yaws(reference_yaw: float, target_yaw: float, *, opposite_label: str) -> Optional[str]:
    delta_yaw = _wrapped_delta_deg(target_yaw - reference_yaw)
    if abs(delta_yaw) < PANORAMIC_RELATION_MIN_DELTA_DEG:
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


def _camera_rotation_relation(entity: Entity, rotation_direction: str, angle_deg: int) -> Optional[str]:
    yaw = _yaw_deg_360(entity)
    observer_forward = float(angle_deg) if rotation_direction == "right" else -float(angle_deg)
    return _panoramic_ring_relation_from_yaws(observer_forward % 360.0, yaw, opposite_label="behind")


def _yaw_deg_360(entity: Entity) -> float:
    return float(entity.lon_deg) % 360.0


def _wrapped_delta_deg(delta: float) -> float:
    delta = ((delta + 180.0) % 360.0) - 180.0
    if delta == -180.0:
        return 180.0
    return delta


def _build_cognitive_map_understanding_tasks(scene: SceneMetadata) -> List[Dict[str, Any]]:
    # 预留：后续接入 10x10 Cognitive Map 数据时，从这里生成 scene-level 任务。
    return []


def _build_hps_path_selection_tasks(scene: SceneMetadata) -> List[Dict[str, Any]]:
    # 预留：后续接入 HPS / 路径选择 supervision 时，从这里生成任务。
    return []


def _build_rotation_consistency_augmentation_tasks(scene: SceneMetadata) -> List[Dict[str, Any]]:
    # 预留：最终 ERP 旋转增强阶段的任务入口。
    # 当前不在 canonical QA 里直接生成。
    return []


def _task_inventory_summary(task_groups: Dict[str, Any], anchors: List[Dict[str, Any]]) -> Dict[str, int]:
    # 统计 scene plan 里不同能力维度下大概生成了多少任务，
    # 方便快速看这个 scene 的任务覆盖分布。
    counter = Counter()

    def _add_task(task: Dict[str, Any]) -> None:
        counter[task["ability"]] += 1
        if task.get("erp_robustness"):
            counter["erp_projection_robustness"] += 1

    for task in task_groups.get("global_tasks", []):
        _add_task(task)
    for task in task_groups.get("rotation_tasks", []):
        _add_task(task)
    for anchor in anchors:
        for task in anchor.get("tasks", []):
            _add_task(task)
    return dict(counter)


def build_scene_plan(scene: SceneMetadata, max_anchors: int = 6) -> Dict[str, Any]:
    # 这是 scene planning 的主入口。
    # 它输出的是“任务规划结果”，不是最终样本：
    # 先判断这个 scene 能做什么，再决定围绕哪些 anchor / pair / rotation 去出题。
    task_feasibility = assess_task_feasibility(scene)
    anchors = select_anchor_entities(scene, max_anchors=max_anchors)
    anchor_payload = []
    for item in anchors:
        entity = item["entity"]
        # 每个 anchor 最多配若干 partner，用于构造 pairwise task。
        partners = choose_relation_partners(entity, scene, max_partners=3)
        anchor_tasks = _build_anchor_tasks(scene, item)
        contextual_tasks = _build_contextual_anchor_tasks(scene, item)
        pair_tasks = _build_pair_tasks(scene, item, partners)
        anchor_payload.append(
            {
                "entity_id": entity.entity_id,
                "label": entity.label,
                "selection_score": item["selection_score"],
                "yaw_bin": item["yaw_bin"],
                "pitch_bin": item["pitch_bin"],
                "lon_deg": round(entity.lon_deg, 2),
                "lat_deg": round(entity.lat_deg, 2),
                "depth_bucket": item["depth_bucket"],
                "verified_semantics": entity.verified_semantics,
                "seam_adjacent": item["seam_adjacent"],
                "pole_adjacent": item["pole_adjacent"],
                "relation_partners": [
                    {
                        "entity_id": partner["entity"].entity_id,
                        "label": partner["entity"].label,
                        "role": partner["role"],
                        "angular_gap_deg": partner["angular_gap_deg"],
                        "depth_gap": partner["depth_gap"],
                    }
                    for partner in partners
                ],
                "tasks": anchor_tasks + contextual_tasks + pair_tasks,
            }
        )

    task_groups = {
        "global_tasks": _build_global_tasks(scene),
        "rotation_tasks": _build_rotation_tasks(scene, anchors),
        "reserved_future_tasks": {
            "cognitive_map_understanding": _build_cognitive_map_understanding_tasks(scene),
            "hps_path_selection": _build_hps_path_selection_tasks(scene),
            "rotation_consistency_augmentation": _build_rotation_consistency_augmentation_tasks(scene),
        },
    }

    return {
        "scene_id": scene.scene_id,
        "readiness_audit": audit_metadata(scene),
        "scene_summary": {
            "entity_count": len(scene.entities),
            "label_distribution": summarize_label_distribution(scene.entities),
            "has_depth_source": bool(scene.depth_map_path or scene.depth_source),
            "has_scene_global_tags": bool(scene.scene_global_tags),
            "has_room_layout_proxy": bool(scene.room_layout_proxy),
            "has_free_space_map": scene.free_space_map is not None,
        },
        "task_feasibility": task_feasibility,
        "anchors": anchor_payload,
        "task_groups": task_groups,
        "task_inventory_summary": _task_inventory_summary(task_groups, anchor_payload),
    }
