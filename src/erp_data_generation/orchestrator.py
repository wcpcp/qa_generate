from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .builders import build_canonical_samples
from .pipeline import build_scene_plan, load_scene_metadata
from .postprocess import build_postprocess_jobs
from .postprocess_execution import execute_postprocess_jobs
from .providers import OpenAIResponsesProvider


def discover_scene_inputs(input_path: str, *, metadata_filename: str = "metadata.json") -> List[str]:
    # 输入既可以是单个 JSON，也可以是一个目录。
    # 目录模式下默认只递归找到名为 metadata.json 的基础元数据文件，
    # 避免把中间产物 json、summary.json 等误当成输入。
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    return sorted(str(item) for item in path.rglob(metadata_filename) if item.is_file())


def build_scene_bundle(
    scene_input_path: str,
    *,
    max_anchors: int = 6,
    template_path: Optional[str] = None,
    postprocess_policy_path: Optional[str] = None,
    repackage_probability: Optional[float] = None,
) -> Dict[str, Any]:
    # 新版单 scene 主入口：
    # metadata -> scene plan -> canonical samples -> postprocess jobs。
    scene = load_scene_metadata(scene_input_path)
    scene_plan = build_scene_plan(scene, max_anchors=max_anchors)
    canonical_samples = build_canonical_samples(scene, scene_plan, template_path=template_path)
    postprocess_plan = build_postprocess_jobs(
        scene,
        canonical_samples,
        policy_path=postprocess_policy_path,
        repackage_probability=repackage_probability,
    )
    prepared_canonical_samples = _annotate_canonical_samples(canonical_samples, postprocess_plan)
    return {
        "scene_id": scene.scene_id,
        "input_path": scene_input_path,
        "scene_plan": scene_plan,
        "canonical_samples": canonical_samples,
        "prepared_canonical_samples": prepared_canonical_samples,
        "postprocess_plan": postprocess_plan,
        "summary": {
            "canonical_sample_count": len(canonical_samples),
            "postprocess_job_count": len(postprocess_plan["jobs"]),
            "passthrough_count": len(postprocess_plan["passthrough_sample_ids"]),
            "filtered_postprocess_count": len(postprocess_plan["filtered_sample_ids"]),
            "skipped_postprocess_count": len(postprocess_plan["skipped_samples"]),
        },
    }


def build_corpus_bundle(
    input_paths: Iterable[str],
    *,
    max_anchors: int = 6,
    template_path: Optional[str] = None,
    postprocess_policy_path: Optional[str] = None,
    repackage_probability: Optional[float] = None,
) -> Dict[str, Any]:
    # 多 scene 版本：对每个 scene 重复 build_scene_bundle，不再夹杂 QC/balancing 等旧流程。
    scenes = [
        build_scene_bundle(
            input_path,
            max_anchors=max_anchors,
            template_path=template_path,
            postprocess_policy_path=postprocess_policy_path,
            repackage_probability=repackage_probability,
        )
        for input_path in input_paths
    ]
    return {
        "summary": {
            "scene_count": len(scenes),
            "canonical_sample_count": sum(scene["summary"]["canonical_sample_count"] for scene in scenes),
            "postprocess_job_count": sum(scene["summary"]["postprocess_job_count"] for scene in scenes),
            "passthrough_count": sum(scene["summary"]["passthrough_count"] for scene in scenes),
            "skipped_postprocess_count": sum(scene["summary"]["skipped_postprocess_count"] for scene in scenes),
        },
        "scenes": scenes,
    }


def execute_scene_bundle(
    scene_bundle: Dict[str, Any],
    *,
    provider: OpenAIResponsesProvider,
) -> Dict[str, Any]:
    # 对单 scene bundle 执行统一后的 LLM 后处理。
    execution = execute_postprocess_jobs(
        scene_bundle["postprocess_plan"]["jobs"],
        scene_bundle["canonical_samples"],
        scene_bundle["postprocess_plan"]["passthrough_sample_ids"],
        scene_bundle["postprocess_plan"]["filtered_sample_ids"],
        provider=provider,
    )
    return {
        **scene_bundle,
        "postprocess_execution": execution,
        "summary": {
            **scene_bundle["summary"],
            "final_sample_count": execution["summary"]["final_sample_count"],
            "processed_sample_count": execution["summary"]["processed_count"],
            "unresolved_job_count": execution["summary"]["unresolved_count"],
        },
    }


def execute_corpus_bundle(
    corpus_bundle: Dict[str, Any],
    *,
    provider: OpenAIResponsesProvider,
) -> Dict[str, Any]:
    # 对多 scene bundle 执行统一后的 LLM 后处理。
    executed_scenes = [execute_scene_bundle(scene_bundle, provider=provider) for scene_bundle in corpus_bundle["scenes"]]
    return {
        "summary": {
            **corpus_bundle["summary"],
            "final_sample_count": sum(
                scene["postprocess_execution"]["summary"]["final_sample_count"] for scene in executed_scenes
            ),
            "processed_sample_count": sum(
                scene["postprocess_execution"]["summary"]["processed_count"] for scene in executed_scenes
            ),
            "unresolved_job_count": sum(
                scene["postprocess_execution"]["summary"]["unresolved_count"] for scene in executed_scenes
            ),
        },
        "scenes": executed_scenes,
    }


def _annotate_canonical_samples(
    canonical_samples: Iterable[Dict[str, Any]],
    postprocess_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    # 为第二阶段“只读取现有快照执行 LLM”准备最小可复用状态：
    # 直接把每条 canonical sample 标成 job / passthrough / filtered，
    # 这样后续无需再读 metadata 或 postprocess_plan 明细就能恢复执行上下文。
    jobs_by_sample_id = {job["sample_id"]: job for job in postprocess_plan.get("jobs", [])}
    filtered_ids = set(postprocess_plan.get("filtered_sample_ids", []))
    skipped_by_sample_id = {
        item["sample_id"]: item for item in postprocess_plan.get("skipped_samples", []) if item.get("sample_id")
    }

    prepared_samples: List[Dict[str, Any]] = []
    for sample in canonical_samples:
        prepared = copy.deepcopy(sample)
        sample_id = prepared["sample_id"]
        if sample_id in jobs_by_sample_id:
            job = jobs_by_sample_id[sample_id]
            prepared["postprocess_disposition"] = "job"
            prepared["postprocess_job_id"] = job["job_id"]
            prepared["postprocess_mode"] = job["mode"]
            prepared["postprocess_requires_visual"] = job.get("requires_visual", False)
            prepared["postprocess_fallback_policy"] = job.get("fallback_policy", "use_canonical")
        elif sample_id in filtered_ids:
            prepared["postprocess_disposition"] = "filtered"
            prepared["postprocess_reason"] = skipped_by_sample_id.get(sample_id, {}).get(
                "reason", "filtered_before_execution"
            )
        else:
            prepared["postprocess_disposition"] = "passthrough"
            if sample_id in skipped_by_sample_id:
                prepared["postprocess_reason"] = skipped_by_sample_id[sample_id].get("reason")
        prepared_samples.append(prepared)
    return prepared_samples
