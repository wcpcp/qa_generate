from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class TaskDefinition:
    # Task registry 中每个任务的静态定义。
    # 这里描述的是“一个任务在框架中的身份”，不是某一条具体样本。
    task_family: str
    ability: str
    unit: str
    answer_type: str
    gt_source: List[str]
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    metadata_tier: str = "base"
    phase: str = "phase1_strong"
    recommended_budget: int = 1
    erp_robustness: bool = False


# 当前框架严格按用户定义的四大能力组织。
# 已实现的任务只保留当前要真正出数的部分；
# 未来任务保留名称和入口，但 phase 设为 future，不参与当前生成。
TASK_REGISTRY: Dict[str, TaskDefinition] = {
    "caption": TaskDefinition(
        task_family="caption",
        ability="basic_understanding",
        unit="anchor_entity",
        answer_type="label_or_attribute_or_caption_text",
        gt_source=["semantic.identify", "semantic.attributes", "semantic.caption_brief"],
        required_fields=["semantic.identify"],
        optional_fields=["semantic.attributes", "semantic.caption_brief", "semantic.reground_query", "bbox_erp"],
        metadata_tier="base",
        phase="phase1_strong",
        recommended_budget=2,
    ),
    "existence": TaskDefinition(
        task_family="existence",
        ability="basic_understanding",
        unit="scene",
        answer_type="boolean",
        gt_source=["entity_set"],
        required_fields=["entities"],
        metadata_tier="base",
        phase="phase1_strong",
        recommended_budget=2,
    ),
    "counting": TaskDefinition(
        task_family="counting",
        ability="basic_understanding",
        unit="scene",
        answer_type="integer",
        gt_source=["entity_set"],
        required_fields=["entities"],
        metadata_tier="base",
        phase="phase1_strong",
        recommended_budget=1,
    ),
    "grounding": TaskDefinition(
        task_family="grounding",
        ability="basic_understanding",
        unit="anchor_entity",
        answer_type="normalized_bbox_and_precise_angles",
        gt_source=["bbox_erp", "mask_rle", "semantic.reground_query"],
        required_fields=["bbox_erp", "mask_rle", "semantic.reground_query"],
        metadata_tier="base",
        phase="phase1_strong",
        recommended_budget=1,
    ),
    "scene_understanding": TaskDefinition(
        task_family="scene_understanding",
        ability="basic_understanding",
        unit="scene",
        answer_type="scene_text",
        gt_source=["scene_global_tags", "entity_summary"],
        required_fields=["entities"],
        optional_fields=["scene_global_tags"],
        metadata_tier="optional",
        phase="future",
        recommended_budget=0,
    ),
    "direct_direction": TaskDefinition(
        task_family="direct_direction",
        ability="omnidirectional_understanding",
        unit="anchor_entity",
        answer_type="bfov_or_absolute_sector_label",
        gt_source=["bfov", "lon_lat"],
        required_fields=["lon_lat"],
        metadata_tier="base",
        phase="phase1_strong",
        recommended_budget=2,
    ),
    "relative_direction": TaskDefinition(
        task_family="relative_direction",
        ability="omnidirectional_understanding",
        unit="entity_pair",
        answer_type="panoramic_ring_relation_label",
        gt_source=["lon_lat"],
        required_fields=["lon_lat"],
        metadata_tier="base",
        phase="phase1_strong",
        recommended_budget=1,
    ),
    "view_transform": TaskDefinition(
        task_family="view_transform",
        ability="omnidirectional_understanding",
        unit="scene_rotation",
        answer_type="reoriented_view_direction_label",
        gt_source=["lon_lat", "rotation_rule", "bfov"],
        required_fields=["lon_lat"],
        metadata_tier="base",
        phase="phase1_strong",
        recommended_budget=2,
        erp_robustness=True,
    ),
    "distance_estimation": TaskDefinition(
        task_family="distance_estimation",
        ability="spatial_3d_understanding",
        unit="anchor_entity",
        answer_type="depth_meter_value",
        gt_source=["entity_center_depth"],
        required_fields=["entity_center_depth"],
        metadata_tier="optional",
        phase="phase1_strong",
        recommended_budget=1,
    ),
    "relative_3d_position": TaskDefinition(
        task_family="relative_3d_position",
        ability="spatial_3d_understanding",
        unit="entity_pair",
        answer_type="distance_or_camera_centric_relation",
        gt_source=["entity_center_depth", "lon_lat"],
        required_fields=["entity_center_depth", "lon_lat"],
        optional_fields=["entity_xyz_camera"],
        metadata_tier="optional",
        phase="phase1_conservative",
        recommended_budget=1,
    ),
    "cognitive_map_understanding": TaskDefinition(
        task_family="cognitive_map_understanding",
        ability="spatial_3d_understanding",
        unit="scene",
        answer_type="grid_map_or_layout_text",
        gt_source=["room_layout_proxy", "free_space_map"],
        required_fields=["room_layout_proxy", "free_space_map"],
        metadata_tier="advanced",
        phase="future",
        recommended_budget=0,
    ),
    "hps_path_selection": TaskDefinition(
        task_family="hps_path_selection",
        ability="spatial_3d_understanding",
        unit="scene",
        answer_type="path_choice_or_turn_bias",
        gt_source=["free_space_map", "openings"],
        required_fields=["free_space_map", "openings"],
        metadata_tier="advanced",
        phase="future",
        recommended_budget=0,
    ),
    "seam_continuity": TaskDefinition(
        task_family="seam_continuity",
        ability="erp_specific",
        unit="anchor_entity",
        answer_type="boolean_or_identity",
        gt_source=["seam_crossing_flag", "bbox_erp", "mask_rle"],
        required_fields=["bbox_erp"],
        optional_fields=["seam_crossing_flag", "mask_rle"],
        metadata_tier="optional",
        phase="phase1_strong",
        recommended_budget=1,
        erp_robustness=True,
    ),
    "polar_distortion_awareness": TaskDefinition(
        task_family="polar_distortion_awareness",
        ability="erp_specific",
        unit="anchor_entity_or_pair",
        answer_type="distortion_aware_shape_size_or_volume_judgement",
        gt_source=["lon_lat", "bbox_erp", "semantic.attributes"],
        required_fields=["lon_lat", "bbox_erp"],
        optional_fields=["semantic.attributes", "entity_xyz_camera"],
        metadata_tier="optional",
        phase="phase1_conservative",
        recommended_budget=1,
        erp_robustness=True,
    ),
    "rotation_consistency": TaskDefinition(
        task_family="rotation_consistency",
        ability="erp_specific",
        unit="scene_rotation",
        answer_type="augmentation_rule",
        gt_source=["lon_lat", "rotation_rule"],
        required_fields=["lon_lat"],
        metadata_tier="base",
        phase="future",
        recommended_budget=0,
        erp_robustness=True,
    ),
}
