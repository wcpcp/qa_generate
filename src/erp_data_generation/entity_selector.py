from __future__ import annotations

from collections import Counter, defaultdict
from math import pi
from typing import Dict, Iterable, List

from .schemas import Entity, SceneMetadata


POLE_ABS_LAT_DEG_MIN = 55.0
SEAM_MARGIN_RATIO = 0.03


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def infer_seam_adjacency(entity: Entity, scene: SceneMetadata) -> bool:
    # 优先使用显式 seam 标记；如果上游没有提供，就退化为 bbox 是否靠近 ERP 左右边界，
    # 再不行就根据经度是否接近 wrap-around 边界做启发式判断。
    if entity.seam_crossing_flag is not None:
        return bool(entity.seam_crossing_flag)
    if len(entity.bbox_erp) == 4 and scene.erp_width:
        x1, _, x2, _ = entity.bbox_erp
        margin = max(12.0, scene.erp_width * SEAM_MARGIN_RATIO)
        return x1 <= margin or x2 >= (scene.erp_width - margin)
    return abs(entity.lon_deg) >= 165.0


def infer_pole_proximity(entity: Entity) -> bool:
    # 优先使用显式 pole 标记；否则根据纬度是否过高来近似判断是否接近极区。
    if entity.pole_proximity_flag is not None:
        return bool(entity.pole_proximity_flag)
    return abs(entity.lat_deg) >= POLE_ABS_LAT_DEG_MIN


def angular_gap_deg(entity_a: Entity, entity_b: Entity) -> float:
    # ERP 上的相对方向比较主要依赖经纬差，这里对经度差做了 wrap-around 处理，
    # 避免跨 seam 的实体被误判成角度很远。
    raw_lon_gap = abs(entity_a.lon_deg - entity_b.lon_deg) % 360.0
    lon_gap = min(raw_lon_gap, 360.0 - raw_lon_gap)
    lat_gap = abs(entity_a.lat_deg - entity_b.lat_deg)
    return lon_gap + lat_gap


def _area_score(area_ratio: float) -> float:
    if area_ratio <= 0:
        return 0.0
    if area_ratio < 0.001:
        return 0.15
    if area_ratio < 0.005:
        return 0.5
    if area_ratio < 0.05:
        return 1.0
    if area_ratio < 0.2:
        return 0.8
    return 0.4


def _uniqueness_score(entity: Entity, label_counts: Counter) -> float:
    if entity.entity_uniqueness_score is not None:
        return float(entity.entity_uniqueness_score)
    count = max(label_counts[entity.label], 1)
    return 1.0 / count


def _geometry_usefulness(entity: Entity, scene: SceneMetadata) -> float:
    score = 0.45
    if infer_seam_adjacency(entity, scene):
        score += 0.2
    if infer_pole_proximity(entity):
        score += 0.15
    if abs(entity.lon_deg) > 120:
        score += 0.1
    if abs(entity.lat_deg) > 35:
        score += 0.1
    return _clamp(score, 0.0, 1.0)


def _semantic_verification_score(entity: Entity) -> float:
    return 1.0 if entity.verified_semantics else 0.35


def _depth_quality_score(entity: Entity) -> float:
    if entity.depth_quality_score is not None:
        return _clamp(float(entity.depth_quality_score), 0.0, 1.0)
    return 1.0 if entity.has_depth else 0.4


def score_entity(entity: Entity, label_counts: Counter, scene: SceneMetadata) -> float:
    # 这个分数用于挑选 anchor entity。
    # 它综合考虑语义质量、检测置信度、跨视角支持、面积、唯一性、几何价值和深度质量。
    semantic_conf = (
        float(entity.semantic.confidence)
        if entity.semantic.confidence is not None
        else float(entity.semantic_quality_score or 0.7)
    )
    confidence = float(entity.best_score or entity.confidence or 0.0)
    support_views = _clamp(entity.support_views / 3.0, 0.0, 1.0)
    area_score = _area_score(entity.area_ratio)
    uniqueness = _uniqueness_score(entity, label_counts)
    geometry = _geometry_usefulness(entity, scene)
    semantic_verification = _semantic_verification_score(entity)
    depth_quality = _depth_quality_score(entity)
    score = (
        0.22 * semantic_conf
        + 0.14 * confidence
        + 0.10 * support_views
        + 0.14 * area_score
        + 0.12 * uniqueness
        + 0.10 * geometry
        + 0.10 * semantic_verification
        + 0.08 * depth_quality
    )
    return round(score, 4)


def _yaw_bin(entity: Entity, bins: int = 4) -> int:
    lon = entity.lon_lat[0]
    normalized = (lon + pi) / (2 * pi)
    return min(bins - 1, max(0, int(normalized * bins)))


def _pitch_bin(entity: Entity, bins: int = 3) -> int:
    lat = entity.lon_lat[1]
    normalized = (lat + (pi / 2.0)) / pi
    return min(bins - 1, max(0, int(normalized * bins)))


def select_anchor_entities(scene: SceneMetadata, max_anchors: int = 6) -> List[Dict[str, object]]:
    # 先给每个实体打分并记录它在 yaw / pitch 上的 bin。
    # 选择时优先覆盖不同方向区域，避免所有 anchor 都集中在 panorama 的同一侧。
    label_counts = Counter(entity.label for entity in scene.entities)
    scored = []
    for entity in scene.entities:
        selection_score = score_entity(entity, label_counts, scene)
        scored.append(
            {
                "entity": entity,
                "selection_score": selection_score,
                "yaw_bin": _yaw_bin(entity),
                "pitch_bin": _pitch_bin(entity),
                "seam_adjacent": infer_seam_adjacency(entity, scene),
                "pole_adjacent": infer_pole_proximity(entity),
                "depth_bucket": entity.depth_bucket,
            }
        )
    scored.sort(key=lambda item: item["selection_score"], reverse=True)

    selected: List[Dict[str, object]] = []
    used_bins = set()

    for item in scored:
        bin_key = (item["yaw_bin"], item["pitch_bin"])
        if bin_key not in used_bins:
            selected.append(item)
            used_bins.add(bin_key)
        if len(selected) >= max_anchors:
            return selected

    for item in scored:
        if any(sel["entity"].entity_id == item["entity"].entity_id for sel in selected):
            continue
        selected.append(item)
        if len(selected) >= max_anchors:
            break

    return selected


def choose_relation_partners(
    anchor: Entity,
    scene: SceneMetadata,
    max_partners: int = 3,
) -> List[Dict[str, object]]:
    # partner 不是简单取最近的几个，而是显式构造三种角色：
    # 最近邻、同类干扰项、远距对比项。这样 pair task 会更有结构。
    candidate_payloads: List[Dict[str, object]] = []
    for entity in scene.entities:
        if entity.entity_id == anchor.entity_id:
            continue
        gap = angular_gap_deg(anchor, entity)
        depth_gap = 0.0
        if anchor.entity_center_depth is not None and entity.entity_center_depth is not None:
            depth_gap = abs(anchor.entity_center_depth - entity.entity_center_depth)
        candidate_payloads.append(
            {
                "entity": entity,
                "angular_gap_deg": round(gap, 2),
                "depth_gap": round(depth_gap, 3),
                "same_label": entity.label == anchor.label,
                "verified_semantics": entity.verified_semantics,
            }
        )

    selected: List[Dict[str, object]] = []
    used_ids = set()

    def _take_best(candidates: List[Dict[str, object]], role: str) -> None:
        for item in candidates:
            entity = item["entity"]
            if entity.entity_id in used_ids:
                continue
            enriched = dict(item)
            enriched["role"] = role
            selected.append(enriched)
            used_ids.add(entity.entity_id)
            return

    nearest = sorted(
        candidate_payloads,
        key=lambda item: (
            item["angular_gap_deg"],
            -int(item["verified_semantics"]),
            item["depth_gap"],
        ),
    )
    _take_best(nearest, "nearest_neighbor")

    same_category = sorted(
        [item for item in candidate_payloads if item["same_label"]],
        key=lambda item: (
            item["angular_gap_deg"],
            -int(item["verified_semantics"]),
        ),
    )
    _take_best(same_category, "same_category_distractor")

    far_contrast = sorted(
        [item for item in candidate_payloads if not item["same_label"]],
        key=lambda item: (
            -(item["angular_gap_deg"] + item["depth_gap"] * 10.0),
            -int(item["verified_semantics"]),
        ),
    )
    _take_best(far_contrast, "far_contrast")

    fallback = sorted(
        candidate_payloads,
        key=lambda item: (
            -int(item["verified_semantics"]),
            item["angular_gap_deg"],
            -item["depth_gap"],
        ),
    )
    for item in fallback:
        if len(selected) >= max_partners:
            break
        if item["entity"].entity_id in used_ids:
            continue
        enriched = dict(item)
        enriched["role"] = "fallback"
        selected.append(enriched)
        used_ids.add(item["entity"].entity_id)

    return selected[:max_partners]


def summarize_label_distribution(entities: Iterable[Entity]) -> Dict[str, int]:
    return dict(Counter(entity.label for entity in entities))


def bucket_entities_by_label(scene: SceneMetadata) -> Dict[str, List[Entity]]:
    grouped: Dict[str, List[Entity]] = defaultdict(list)
    for entity in scene.entities:
        grouped[entity.label].append(entity)
    return dict(grouped)
