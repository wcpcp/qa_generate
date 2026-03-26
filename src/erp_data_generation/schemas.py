from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, degrees, radians, sin
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EntitySemantic:
    # 这是实体语义层的统一结构，承接上游多模态模型生成的 identify / attribute / caption 等信息。
    identify: str = "unknown"
    attributes: Dict[str, Any] = field(default_factory=dict)
    event_status: str = ""
    caption_brief: str = ""
    caption_dense: str = ""
    reground_query: str = ""
    confidence: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntitySemantic":
        # 将原始 semantic 字典归一化为内部结构，缺失字段做安全默认值填充。
        return cls(
            identify=data.get("identify", "unknown"),
            attributes=data.get("attributes", {}) or {},
            event_status=data.get("event_status", ""),
            caption_brief=data.get("caption_brief", ""),
            caption_dense=data.get("caption_dense", ""),
            reground_query=data.get("reground_query", ""),
            confidence=data.get("confidence"),
        )


@dataclass
class Entity:
    # 这是 scene 内单个实体的核心结构。
    # 它同时保存原始 ERP 区域、语义信息、深度/几何信息，以及下游规划需要的质量字段。
    entity_id: str
    confidence: float
    bbox_erp: List[float]
    mask_rle: Dict[str, Any]
    lon_lat: Tuple[float, float]
    area_ratio: float
    support_views: int
    semantic: EntitySemantic
    source_views: List[str] = field(default_factory=list)
    representative_view_id: str = ""
    best_score: Optional[float] = None
    vote_score: Optional[float] = None
    member_count: Optional[int] = None
    projection_iou: Optional[float] = None
    semantic_verification_iou: Optional[float] = None
    semantic_verification_passed: Optional[bool] = None
    entity_center_depth: Optional[float] = None
    depth_quality_score: Optional[float] = None
    depth_source: str = ""
    entity_xyz_camera: Optional[List[float]] = None
    entity_bfov: Optional[List[float]] = None
    entity_extent_3d: Optional[List[float]] = None
    semantic_quality_score: Optional[float] = None
    entity_uniqueness_score: Optional[float] = None
    occlusion_flag: Optional[bool] = None
    truncation_flag: Optional[bool] = None
    seam_crossing_flag: Optional[bool] = None
    pole_proximity_flag: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        # 这里做一层关键的“原始 metadata -> 运行时字段”映射。
        # 例如 depth.median_m -> entity_center_depth，spatial.xyz_camera_m -> entity_xyz_camera。
        lon_lat = data.get("lon_lat", [0.0, 0.0])
        semantic = EntitySemantic.from_dict(data.get("semantic", {}))
        depth_info = data.get("depth", {}) or {}
        spatial_info = data.get("spatial", {}) or {}
        entity_center_depth = data.get("entity_center_depth")
        if entity_center_depth is None and depth_info.get("status") == "ok":
            entity_center_depth = depth_info.get("median_m")
        entity_xyz_camera = data.get("entity_xyz_camera")
        if entity_xyz_camera is None and spatial_info.get("xyz_camera_m") is not None:
            entity_xyz_camera = spatial_info.get("xyz_camera_m")
        entity_bfov = data.get("entity_bfov")
        if entity_bfov is None:
            bfov_info = data.get("bfov", {}) or {}
            if {"yaw_deg", "pitch_deg", "x_fov_deg", "y_fov_deg"} <= set(bfov_info.keys()):
                entity_bfov = [
                    bfov_info.get("yaw_deg"),
                    bfov_info.get("pitch_deg"),
                    bfov_info.get("x_fov_deg"),
                    bfov_info.get("y_fov_deg"),
                ]
        depth_quality_score = data.get("depth_quality_score")
        if depth_quality_score is None and depth_info.get("valid_ratio") is not None:
            depth_quality_score = depth_info.get("valid_ratio")
        semantic_verification_passed = data.get("semantic_verification_passed")
        if semantic_verification_passed is None:
            semantic_verification_passed = True
        semantic_quality_score = data.get("semantic_quality_score")
        if semantic_quality_score is None:
            semantic_quality_score = semantic.confidence
        return cls(
            entity_id=data["entity_id"],
            confidence=float(data.get("confidence", 0.0)),
            bbox_erp=list(data.get("bbox_erp", [])),
            mask_rle=data.get("mask_rle", {}) or {},
            lon_lat=(float(lon_lat[0]), float(lon_lat[1])),
            area_ratio=float(data.get("area_ratio", 0.0)),
            support_views=int(data.get("support_views", 0)),
            semantic=semantic,
            source_views=list(data.get("source_views", []) or []),
            representative_view_id=data.get("representative_view_id", ""),
            best_score=data.get("best_score"),
            vote_score=data.get("vote_score"),
            member_count=data.get("member_count"),
            projection_iou=data.get("projection_iou"),
            semantic_verification_iou=data.get("semantic_verification_iou"),
            semantic_verification_passed=semantic_verification_passed,
            entity_center_depth=entity_center_depth,
            depth_quality_score=depth_quality_score,
            depth_source=data.get("depth_source", "metadata_depth" if entity_center_depth is not None else ""),
            entity_xyz_camera=entity_xyz_camera,
            entity_bfov=entity_bfov,
            entity_extent_3d=data.get("entity_extent_3d"),
            semantic_quality_score=semantic_quality_score,
            entity_uniqueness_score=data.get("entity_uniqueness_score"),
            occlusion_flag=data.get("occlusion_flag"),
            truncation_flag=data.get("truncation_flag"),
            seam_crossing_flag=data.get("seam_crossing_flag"),
            pole_proximity_flag=data.get("pole_proximity_flag"),
        )

    @property
    def label(self) -> str:
        # 对下游统一暴露实体类别，默认使用 semantic.identify。
        return self.semantic.identify or "unknown"

    @property
    def center_xy(self) -> Tuple[float, float]:
        # 从 ERP bbox 推导一个简单的二维中心点，便于后续做可视化或轻量几何处理。
        if len(self.bbox_erp) != 4:
            return (0.0, 0.0)
        x1, y1, x2, y2 = self.bbox_erp
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def lon_deg(self) -> float:
        # 内部 lon_lat 用弧度，这里暴露角度制便于 planning 和调试阅读。
        return degrees(self.lon_lat[0])

    @property
    def lat_deg(self) -> float:
        return degrees(self.lon_lat[1])

    @property
    def has_depth(self) -> bool:
        # 判断这个实体是否具备可用深度。
        return self.entity_center_depth is not None

    @property
    def depth_bucket(self) -> Optional[str]:
        # 这里是一个非常粗粒度的距离分桶，主要服务 phase-1 的 near / medium / far 任务。
        if self.entity_center_depth is None:
            return None
        if self.entity_center_depth < 2.0:
            return "near"
        if self.entity_center_depth < 5.0:
            return "medium"
        return "far"

    @property
    def verified_semantics(self) -> bool:
        # 统一判断“这个实体的语义是否可信”。
        # 优先使用显式验证结果，否则退化到语义置信度阈值。
        if self.semantic_verification_passed is not None:
            return bool(self.semantic_verification_passed)
        if self.semantic.confidence is not None:
            return float(self.semantic.confidence) >= 0.7
        if self.semantic_quality_score is not None:
            return float(self.semantic_quality_score) >= 0.7
        return True

    @property
    def resolved_xyz_camera(self) -> Optional[Tuple[float, float, float]]:
        # 优先使用显式 xyz；没有的话，只要有 lon_lat + depth，就推导一个粗略 camera-space 点。
        # 这使得 relative_3d_position 能在缺少显式 xyz 时退化运行。
        if self.entity_xyz_camera is not None and len(self.entity_xyz_camera) == 3:
            return (
                float(self.entity_xyz_camera[0]),
                float(self.entity_xyz_camera[1]),
                float(self.entity_xyz_camera[2]),
            )
        if self.entity_center_depth is None:
            return None
        depth = float(self.entity_center_depth)
        lon, lat = self.lon_lat
        x = depth * cos(lat) * sin(lon)
        y = depth * sin(lat)
        z = depth * cos(lat) * cos(lon)
        return (x, y, z)

    @property
    def resolved_bfov(self) -> Optional[Tuple[float, float, float, float]]:
        if self.entity_bfov is not None and len(self.entity_bfov) == 4:
            return (
                float(self.entity_bfov[0]),
                float(self.entity_bfov[1]),
                float(self.entity_bfov[2]),
                float(self.entity_bfov[3]),
            )
        return None

    @property
    def erp_consistent_xyz_camera(self) -> Optional[Tuple[float, float, float]]:
        # 这是专门给 ERP 几何任务使用的“球面一致坐标”。
        # 它优先使用 BFOV 中心 + median depth 推导，避免 seam-crossing 物体的显式 xyz
        # 因为普通 bbox center 计算中心角而发生偏转。
        if self.entity_center_depth is None:
            return None

        depth = float(self.entity_center_depth)
        bfov = self.resolved_bfov
        if bfov is not None:
            yaw_rad = radians(float(bfov[0]))
            pitch_rad = radians(float(bfov[1]))
            x = depth * cos(pitch_rad) * sin(yaw_rad)
            y = depth * sin(-pitch_rad)
            z = depth * cos(pitch_rad) * cos(yaw_rad)
            return (x, y, z)

        lon, lat = self.lon_lat
        x = depth * cos(lat) * sin(lon)
        y = depth * sin(lat)
        z = depth * cos(lat) * cos(lon)
        return (x, y, z)


@dataclass
class SceneMetadata:
    # 这是单个 ERP scene 的统一运行时结构。
    # raw 字段保留原始 JSON，便于审计时同时查看“原始 contract”和“归一化 contract”。
    scene_id: str
    erp_image_path: str = ""
    depth_map_path: str = ""
    erp_width: int = 0
    erp_height: int = 0
    depth_source: str = ""
    metadata_pipeline_version: str = ""
    camera_convention: Dict[str, Any] = field(default_factory=dict)
    scene_global_tags: Dict[str, Any] = field(default_factory=dict)
    room_layout_proxy: Dict[str, Any] = field(default_factory=dict)
    free_space_map: Optional[Any] = None
    openings: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneMetadata":
        # 场景级别也做归一化：例如 image_path -> erp_image_path，
        # mask_rle.size -> erp_width / erp_height。
        entities = [Entity.from_dict(item) for item in data.get("entities", [])]
        width = int(data.get("erp_width", 0))
        height = int(data.get("erp_height", 0))
        if (not width or not height) and entities:
            size = entities[0].mask_rle.get("size")
            if isinstance(size, list) and len(size) == 2:
                height = int(size[0])
                width = int(size[1])
        erp_image_path = _resolve_erp_image_path(
            data.get("erp_image_path", data.get("image_path", "")),
            data.get("viewpoint_id", ""),
        )
        return cls(
            scene_id=str(data.get("scene_id", data.get("image_id", "unknown_scene"))),
            erp_image_path=erp_image_path,
            depth_map_path=data.get("depth_map_path", ""),
            erp_width=width,
            erp_height=height,
            depth_source=data.get("depth_source", ""),
            metadata_pipeline_version=data.get("metadata_pipeline_version", ""),
            camera_convention=data.get("camera_convention", {}) or {},
            scene_global_tags=data.get("scene_global_tags", {}) or {},
            room_layout_proxy=data.get("room_layout_proxy", {}) or {},
            free_space_map=data.get("free_space_map"),
            openings=list(data.get("openings", []) or []),
            entities=entities,
            raw=data,
        )


def _resolve_erp_image_path(raw_path: str, viewpoint_id: str) -> str:
    # 优先保留原始 image_path。
    # 如果原路径在当前机器不可访问，则尝试映射到仓库内的本地 dataset 目录。
    if raw_path:
        raw = Path(raw_path)
        if raw.exists():
            return str(raw)
        repo_root = Path(__file__).resolve().parents[2]
        local_candidate = repo_root / "dataset" / raw.name
        if local_candidate.exists():
            return str(local_candidate)
        if viewpoint_id:
            by_viewpoint = repo_root / "dataset" / str(viewpoint_id) / raw.name
            if by_viewpoint.exists():
                return str(by_viewpoint)
    return raw_path
