from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def write_json(data: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(input_path: str) -> List[Dict[str, Any]]:
    path = Path(input_path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def export_scene_bundle(bundle: Dict[str, Any], output_dir: str) -> List[str]:
    # 导出单 scene 的统一产物。
    # 这里默认只保留“这一步新增且真正有用”的文件，
    # 不再重复保存已经可以通过其他独立脚本单独导出的 scene_plan / canonical_samples / postprocess_plan。
    output_path = Path(output_dir)
    scene_dir = output_path / bundle["scene_id"]
    scene_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for name, payload in [
        ("summary.json", bundle["summary"]),
    ]:
        target = scene_dir / name
        write_json(payload, str(target))
        written.append(str(target))

    for name, payload in [
        ("postprocess_jobs.jsonl", bundle["postprocess_plan"]["jobs"]),
    ]:
        target = scene_dir / name
        write_jsonl(payload, str(target))
        written.append(str(target))

    execution = bundle.get("postprocess_execution")
    if execution:
        for name, payload in [
            ("postprocess_execution.json", execution),
        ]:
            target = scene_dir / name
            write_json(payload, str(target))
            written.append(str(target))
        for name, payload in [
            ("final_samples.jsonl", execution["final_samples"]),
        ]:
            target = scene_dir / name
            write_jsonl(payload, str(target))
            written.append(str(target))

    return written


def export_scene_bundle_to_path(bundle: Dict[str, Any], scene_dir: str) -> List[str]:
    # 导出到指定 scene 目录。
    # 这个函数主要给“大目录流式处理”模式使用，允许输出完整镜像输入目录树。
    scene_path = Path(scene_dir)
    scene_path.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for name, payload in [
        ("summary.json", bundle["summary"]),
    ]:
        target = scene_path / name
        write_json(payload, str(target))
        written.append(str(target))

    target = scene_path / "postprocess_jobs.jsonl"
    write_jsonl(bundle["postprocess_plan"]["jobs"], str(target))
    written.append(str(target))

    execution = bundle.get("postprocess_execution")
    if execution:
        target = scene_path / "postprocess_execution.json"
        write_json(execution, str(target))
        written.append(str(target))
        target = scene_path / "final_samples.jsonl"
        write_jsonl(execution["final_samples"], str(target))
        written.append(str(target))

    return written


def export_corpus_bundle(bundle: Dict[str, Any], output_dir: str) -> List[str]:
    # 导出多 scene 统一产物，并在根目录额外导出聚合结果。
    # 同样只保留本步骤真正新增的核心文件。
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    summary_path = output_path / "summary.json"
    write_json(bundle["summary"], str(summary_path))
    written.append(str(summary_path))

    all_jobs: List[Dict[str, Any]] = []
    all_final: List[Dict[str, Any]] = []
    all_unresolved: List[Dict[str, Any]] = []

    for scene_bundle in bundle["scenes"]:
        written.extend(export_scene_bundle(scene_bundle, output_dir))
        all_jobs.extend(scene_bundle["postprocess_plan"]["jobs"])
        if "postprocess_execution" in scene_bundle:
            all_final.extend(scene_bundle["postprocess_execution"]["final_samples"])
            all_unresolved.extend(scene_bundle["postprocess_execution"]["unresolved_jobs"])

    for name, payload in [
        ("postprocess_jobs.jsonl", all_jobs),
    ]:
        target = output_path / name
        write_jsonl(payload, str(target))
        written.append(str(target))

    if all_final:
        final_path = output_path / "final_samples.jsonl"
        write_jsonl(all_final, str(final_path))
        written.append(str(final_path))
        execution_path = output_path / "postprocess_execution.json"
        write_json(
            {
                "summary": {
                    "final_sample_count": len(all_final),
                    "unresolved_job_count": len(all_unresolved),
                },
                "unresolved_jobs": all_unresolved,
            },
            str(execution_path),
        )
        written.append(str(execution_path))

    return written
