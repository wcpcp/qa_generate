from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    np = None
    Image = None
    ImageDraw = None

from .schemas import Entity, SceneMetadata


def build_entity_visual_context(scene: SceneMetadata, entity: Entity) -> Dict[str, Any]:
    # 为 caption 等需要更强视觉上下文的任务准备局部透视视图。
    # 当前只保留一张带目标细框的 context view。
    # 视角中心对准目标，视野大小按目标角尺寸的约 3 倍放大，并做最大范围裁剪。
    assets = {
        "erp_image_path": scene.erp_image_path,
        "image_available": bool(scene.erp_image_path and Path(scene.erp_image_path).exists()),
        "mode": "erp_full_only",
        "perspective_images": [],
    }
    if not assets["image_available"] or np is None or Image is None:
        return assets

    try:
        out_dir = Path(tempfile.gettempdir()) / "erp_visual_context"
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{scene.scene_id}_{entity.entity_id}"
        yaw_deg = entity.lon_deg % 360.0
        pitch_deg = max(0.0, min(180.0, 90.0 - entity.lat_deg))
        bbox = entity.bbox_erp
        bbox_w = max(1.0, float(bbox[2] - bbox[0]))
        bbox_h = max(1.0, float(bbox[3] - bbox[1]))
        angular_w = (bbox_w / max(scene.erp_width, 1)) * 360.0
        angular_h = (bbox_h / max(scene.erp_height, 1)) * 180.0
        context_fov = float(max(80.0, min(120.0, max(angular_w, angular_h) * 3.0)))

        context_path = out_dir / f"{base_name}_context.jpg"

        _save_context_view(scene, entity, context_path, yaw_deg, pitch_deg, context_fov, 896, 896)

        assets["mode"] = "erp_to_perspective"
        assets["perspective_images"] = [
            {"path": str(context_path), "kind": "context_view", "fov_deg": round(context_fov, 1)},
        ]
        return assets
    except Exception:
        return assets


def build_four_face_visual_context(scene: SceneMetadata, entity: Entity | None = None) -> Dict[str, Any]:
    # 为 counting 等需要 360 全局视觉核查的任务准备四面透视图，而不是直接把整张 ERP 原图发给模型。
    # 这四张图固定覆盖 front / right / back / left 四个水平朝向，
    # 如果给了 entity，则在目标可见时绘制细框。
    assets = {
        "erp_image_path": scene.erp_image_path,
        "image_available": bool(scene.erp_image_path and Path(scene.erp_image_path).exists()),
        "mode": "erp_four_faces",
        "perspective_images": [],
    }
    if not assets["image_available"] or np is None or Image is None:
        return assets

    try:
        out_dir = Path(tempfile.gettempdir()) / "erp_grounding_context"
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{scene.scene_id}_{entity.entity_id}" if entity is not None else f"{scene.scene_id}_global"
        faces = [
            ("front", 0.0),
            ("right", 90.0),
            ("back", 180.0),
            ("left", 270.0),
        ]
        pitch_deg = 90.0
        fov_deg = 100.0
        generated = []
        for face_name, yaw_deg in faces:
            output_path = out_dir / f"{base_name}_{face_name}.jpg"
            _save_context_view(scene, entity, output_path, yaw_deg, pitch_deg, fov_deg, 768, 768)
            generated.append(
                {
                    "path": str(output_path),
                    "kind": "erp_face",
                    "face_name": face_name,
                    "yaw_deg": round(yaw_deg, 1),
                    "pitch_deg": round(pitch_deg, 1),
                    "fov_deg": round(fov_deg, 1),
                }
            )
        assets["perspective_images"] = generated
        return assets
    except Exception:
        return assets


def _save_context_view(
    scene: SceneMetadata,
    entity: Entity | None,
    output_path: Path,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_w: int,
    out_h: int,
) -> None:
    src = Image.open(scene.erp_image_path).convert("RGB")
    arr = np.asarray(src)
    persp = _equirectangular_to_perspective(arr, yaw_deg, pitch_deg, fov_deg, out_w, out_h)
    image = Image.fromarray(persp)
    if entity is not None:
        box = _project_bbox_to_perspective(
            scene.erp_width,
            scene.erp_height,
            entity.bbox_erp,
            yaw_deg,
            pitch_deg,
            fov_deg,
            out_w,
            out_h,
        )
        if box is not None:
            draw = ImageDraw.Draw(image)
            draw.rectangle(box, outline=(255, 64, 64), width=2)
    image.save(output_path, quality=92)


def _equirectangular_to_perspective(
    src: "np.ndarray",
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_w: int,
    out_h: int,
) -> "np.ndarray":
    h, w = src.shape[:2]
    hfov = math.radians(fov_deg)
    vfov = hfov * (out_h / max(out_w, 1))
    xs = np.linspace(-math.tan(hfov / 2.0), math.tan(hfov / 2.0), out_w)
    ys = np.linspace(-math.tan(vfov / 2.0), math.tan(vfov / 2.0), out_h)
    grid_x, grid_y = np.meshgrid(xs, ys)

    z = np.ones_like(grid_x)
    x = grid_x
    y = -grid_y
    norm = np.sqrt(x * x + y * y + z * z)
    x /= norm
    y /= norm
    z /= norm

    yaw = math.radians(yaw_deg)
    lat = math.radians(90.0 - pitch_deg)

    x1 = np.cos(yaw) * x + np.sin(yaw) * z
    z1 = -np.sin(yaw) * x + np.cos(yaw) * z
    y1 = y

    y2 = np.cos(lat) * y1 - np.sin(lat) * z1
    z2 = np.sin(lat) * y1 + np.cos(lat) * z1
    x2 = x1

    lon = np.arctan2(x2, z2)
    lat2 = np.arcsin(np.clip(y2, -1.0, 1.0))

    src_x = (lon / (2.0 * math.pi) + 0.5) * w
    src_y = (0.5 - lat2 / math.pi) * h
    src_x = np.mod(src_x, w)
    src_y = np.clip(src_y, 0, h - 1)

    x0 = np.floor(src_x).astype(np.int32)
    x1i = (x0 + 1) % w
    y0 = np.floor(src_y).astype(np.int32)
    y1i = np.clip(y0 + 1, 0, h - 1)

    wa = (x1i - src_x) * (y1i - src_y)
    wb = (src_x - x0) * (y1i - src_y)
    wc = (x1i - src_x) * (src_y - y0)
    wd = (src_x - x0) * (src_y - y0)

    out = (
        src[y0, x0] * wa[..., None]
        + src[y0, x1i] * wb[..., None]
        + src[y1i, x0] * wc[..., None]
        + src[y1i, x1i] * wd[..., None]
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def _project_bbox_to_perspective(
    erp_w: int,
    erp_h: int,
    bbox: List[float],
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_w: int,
    out_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    samples = []
    x1, y1, x2, y2 = [float(v) for v in bbox]
    xs = np.linspace(x1, x2, 5)
    ys = np.linspace(y1, y2, 5)
    for x in xs:
        for y in ys:
            pt = _erp_point_to_perspective(erp_w, erp_h, x, y, yaw_deg, pitch_deg, fov_deg, out_w, out_h)
            if pt is not None:
                samples.append(pt)
    if not samples:
        return None
    min_x = max(0.0, min(p[0] for p in samples))
    min_y = max(0.0, min(p[1] for p in samples))
    max_x = min(float(out_w - 1), max(p[0] for p in samples))
    max_y = min(float(out_h - 1), max(p[1] for p in samples))
    if max_x <= min_x or max_y <= min_y:
        return None
    return (min_x, min_y, max_x, max_y)


def _erp_point_to_perspective(
    erp_w: int,
    erp_h: int,
    x: float,
    y: float,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_w: int,
    out_h: int,
) -> Optional[Tuple[float, float]]:
    lon = ((x / max(erp_w, 1)) - 0.5) * 2.0 * math.pi
    lat = (0.5 - (y / max(erp_h, 1))) * math.pi

    xw = math.cos(lat) * math.sin(lon)
    yw = math.sin(lat)
    zw = math.cos(lat) * math.cos(lon)

    yaw = math.radians(yaw_deg)
    tilt = math.radians(90.0 - pitch_deg)

    y1 = math.cos(-tilt) * yw - math.sin(-tilt) * zw
    z1 = math.sin(-tilt) * yw + math.cos(-tilt) * zw
    x1 = xw

    xc = math.cos(-yaw) * x1 + math.sin(-yaw) * z1
    zc = -math.sin(-yaw) * x1 + math.cos(-yaw) * z1
    yc = y1

    if zc <= 1e-6:
        return None

    hfov = math.radians(fov_deg)
    vfov = hfov * (out_h / max(out_w, 1))
    px = xc / zc
    py = -yc / zc
    max_x = math.tan(hfov / 2.0)
    max_y = math.tan(vfov / 2.0)
    if abs(px) > max_x or abs(py) > max_y:
        return None

    u = ((px / max_x) + 1.0) * 0.5 * out_w
    v = ((py / max_y) + 1.0) * 0.5 * out_h
    return (u, v)
