# Task / Template Matrix

This table summarizes the current active training tasks in `qa_generate`, including:

- ability grouping
- canonical generation modes
- question / answer template families
- LLM postprocess mode
- current ambiguity and consistency controls

## Global Rules

| Item | Current Rule |
| --- | --- |
| Canonical truth source | Metadata, ERP geometry, or deterministic rule logic |
| LLM role | Repackage wording for sampled tasks; only `counting` may correct truth |
| Choice answer balancing | Choice-like tasks now use deterministic balanced option ordering instead of always placing the correct answer first |
| Ambiguity handling | Distance-choice tasks now use `candidate_objects` with referring phrases, and duplicate references are disambiguated with `BFOV` or normalized `box` cues when needed |
| Response-style diversity | Canonical answer templates and postprocess prompts allow bare labels, short phrases, and concise sentences instead of forcing a single answer style |

## Active Tasks

| Ability Group | Task Family | Generation Modes | Question Template Keys | Answer Template Keys | LLM Mode | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `basic_understanding` | `caption` | `dense_description_like` | `caption.dense_description` | `caption.dense_description` | not enabled by default | Current main path only emits dense object descriptions; identify / attribute / brief-caption templates remain available but are not the dominant generation path. |
| `basic_understanding` | `existence` | `positive_existence_label`, `negative_existence_label` | `existence` | `existence.true`, `existence.false` | `existence_positive_repackage`, `existence_negative_repackage` | Metadata-driven yes/no supervision; negative examples are category-level negatives, not visual hard negatives. |
| `basic_understanding` | `counting` | `multi_instance_count` | `counting` | `counting` | `counting_visual_correct` | The only task that always performs visual verification; canonical count is not passed as a fact to the LLM. |
| `basic_understanding` | `grounding` | `full`, `bbox_only`, `angles_only` | `grounding.full`, `grounding.bbox_only`, `grounding.angles_only` | `grounding.full`, `grounding.bbox_only`, `grounding.angles_only` | `grounding_repackage` | Stable ERP localization supervision. Ground-truth localization remains unchanged in postprocess. |
| `omnidirectional_understanding` | `direct_direction` | `precise_bfov`, `absolute_sector_8way` | `direct_direction.precise`, `direct_direction.cardinal` | `direct_direction.precise`, `direct_direction.cardinal` | `direct_direction_repackage` | `precise_bfov` stays structured; `absolute_sector_8way` teaches coarse absolute panorama direction. |
| `omnidirectional_understanding` | `relative_direction` | `panoramic_angular_relation` | `relative_direction` | `relative_direction` | `relative_direction_repackage` | Ring relation on the 360 horizontal panorama only; not a true 3D front / back task. |
| `omnidirectional_understanding` | `view_transform` | `camera_rotation_transform`, `object_conditioned_reorientation` | `view_transform.camera_rotation_transform`, `view_transform.object_conditioned_reorientation` | `view_transform` | `view_transform_repackage` | Trains reference-frame updating after observer rotation or facing-target reorientation. |
| `spatial_3d_understanding` | `distance_estimation` | `absolute_depth_meter`, `candidate_nearest_choice`, `observer_nearest_choice` | `distance_estimation`, `distance_estimation.choice`, `distance_estimation.observer_choice` | `distance_estimation`, `distance_estimation.choice`, `distance_estimation.observer_choice` | `distance_estimation_repackage` | Choice variants now use referring phrases rather than only coarse labels; candidate order is balanced. |
| `spatial_3d_understanding` | `relative_3d_position` | `camera_centric_open`, `camera_centric_choice` | `relative_3d_position.open`, `relative_3d_position.choice` | `relative_3d_position`, `relative_3d_position.choice` | `relative_3d_position_repackage` | Camera-centered multi-axis relation. Choice order is balanced; postprocess keeps every active axis relation. |
| `erp_specific` | `seam_continuity` | `nearest_neighbor`, `relative_direction`, `dedup_count`, `structure_continuity`, `same_entity_judgement` | `seam_continuity.nearest_neighbor`, `seam_continuity.relative_direction`, `seam_continuity.dedup_count`, `seam_continuity.structure_continuity`, `seam_continuity.same_entity_judgement` | `seam_continuity.choice` | `seam_continuity_repackage` | Only generated for true seam-crossing entities. Harder seam modes are only emitted when a real seam partner context exists. |
| `erp_specific` | `polar_distortion_awareness` | `shape_recovery_direct`, `shape_recovery_distortion_aware`, `shape_matching`, `cross_latitude_matching` | `polar_distortion_awareness.shape_direct`, `polar_distortion_awareness.shape_distortion_aware`, `polar_distortion_awareness.shape_matching`, `polar_distortion_awareness.cross_latitude_matching` | `polar_distortion_awareness.shape`, `polar_distortion_awareness.candidate` | `polar_distortion_awareness_repackage` | High-latitude target is identified by coarse label plus `BFOV` or normalized `box`, avoiding shape leakage through referring phrases. |

## Current Choice-Balanced Tasks

The following generation modes now use deterministic option balancing:

- `distance_estimation.candidate_nearest_choice`
- `distance_estimation.observer_nearest_choice`
- `relative_3d_position.camera_centric_choice`
- `seam_continuity.*` choice-like modes
- `polar_distortion_awareness.shape_matching`
- `polar_distortion_awareness.cross_latitude_matching`

This avoids repeatedly placing the correct answer in the first slot while keeping generation reproducible.

## Current Ambiguity Controls

| Task | Control |
| --- | --- |
| `distance_estimation.choice` / `observer_choice` | Candidates are stored as `candidate_objects` with `entity_ref`, and duplicate references are disambiguated with `BFOV` or normalized `box` locators. |
| `polar_distortion_awareness` | High-latitude target uses coarse label + locator instead of a rich referring phrase that could leak shape. |
| `seam_continuity` | Only seam-crossing entities are eligible; seam-neighbor tasks require a valid wrap-around partner context. |
| `relative_3d_position` | Camera-centered relation is preserved exactly in postprocess; no silent dropping of active axes. |

## Inactive / Reserved Families

These task families are registered or reserved but are not part of the main generation flow:

- `scene_understanding`
- `cognitive_map_understanding`
- `hps_path_selection`
- `rotation_consistency`

They remain placeholders for future expansion rather than active training supervision.
