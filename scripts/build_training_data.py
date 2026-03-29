#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from erp_data_generation.exporters import export_corpus_bundle, export_scene_bundle, export_scene_bundle_to_path, write_json, write_jsonl
from erp_data_generation.orchestrator import (
    build_corpus_bundle,
    build_scene_bundle,
    discover_scene_inputs,
    execute_corpus_bundle,
    execute_scene_bundle,
    iter_scene_inputs,
)
from erp_data_generation.pipeline import SceneMetadataLoadError
from erp_data_generation.providers import OpenAIResponsesProvider


def main() -> int:
    # 统一入口：
    # 1. scene plan
    # 2. canonical samples
    # 3. task-aware postprocess jobs
    # 4. 可选执行 LLM 后处理
    parser = argparse.ArgumentParser(description="Build ERP training data with unified canonical generation and task-aware LLM postprocess.")
    parser.add_argument("--input", required=True, help="Path to one scene metadata JSON file or a directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--max-anchors", type=int, default=6, help="Maximum anchor entities.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 4)),
        help="Number of scene-level workers for directory input.",
    )
    parser.add_argument("--template-path", help="Optional question template JSON path.")
    parser.add_argument("--postprocess-policy-path", help="Optional postprocess policy JSON path.")
    parser.add_argument(
        "--repackage-probability",
        type=float,
        default=0.4,
        help="Sampling probability for eligible non-counting rule tasks to enter LLM repackaging.",
    )
    parser.add_argument("--run-llm", action="store_true", help="Execute LLM postprocess immediately after bundle generation.")
    parser.add_argument("--model", help="Provider model override.")
    parser.add_argument("--base-url", help="Provider base URL override.")
    parser.add_argument("--timeout-seconds", type=int, default=180, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry count for provider calls.")
    parser.add_argument("--cache-dir", help="Optional provider cache directory.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable provider response caching.")
    parser.add_argument(
        "--metadata-filename",
        default="metadata.json",
        help="Metadata filename to discover recursively under directory input.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        try:
            bundle = build_scene_bundle(
                str(input_path),
                max_anchors=args.max_anchors,
                template_path=args.template_path,
                postprocess_policy_path=args.postprocess_policy_path,
                repackage_probability=args.repackage_probability,
            )
        except SceneMetadataLoadError as exc:
            output_root = Path(args.output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            summary = {
                "scene_count": 0,
                "canonical_sample_count": 0,
                "postprocess_job_count": 0,
                "passthrough_count": 0,
                "filtered_postprocess_count": 0,
                "skipped_postprocess_count": 0,
                "final_sample_count": 0,
                "processed_sample_count": 0,
                "unresolved_job_count": 0,
                "skipped_input_count": 1,
                "failed_scene_count": 0,
            }
            manifest = [
                {
                    "input_path": str(input_path),
                    "status": "skipped",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ]
            write_json(summary, str(output_root / "summary.json"))
            write_jsonl(manifest, str(output_root / "manifest.jsonl"))
            print(f"[skip] {input_path}: {exc}", flush=True)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if args.run_llm:
            provider = _build_provider(args)
            bundle = execute_scene_bundle(bundle, provider=provider)
        export_scene_bundle(bundle, args.output_dir)
        print(json.dumps(bundle["summary"], ensure_ascii=False, indent=2))
        return 0

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[start] preparing into {output_root}", flush=True)
    provider = _build_provider(args) if args.run_llm else None
    summary = _stream_directory_build(
        input_root=input_path,
        input_paths=iter_scene_inputs(str(input_path), metadata_filename=args.metadata_filename),
        output_root=output_root,
        max_anchors=args.max_anchors,
        workers=max(1, int(args.workers)),
        template_path=args.template_path,
        postprocess_policy_path=args.postprocess_policy_path,
        repackage_probability=args.repackage_probability,
        provider=provider,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _build_provider(args: argparse.Namespace) -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        cache_dir=args.cache_dir,
        use_cache=not args.disable_cache,
    )


def _stream_directory_build(
    *,
    input_root: Path,
    input_paths: Iterable[str],
    output_root: Path,
    max_anchors: int,
    workers: int,
    template_path: str | None,
    postprocess_policy_path: str | None,
    repackage_probability: float | None,
    provider: OpenAIResponsesProvider | None,
) -> dict:
    # 面向服务器大目录的流式主入口：
    # 1. 只递归处理 metadata.json
    # 2. 逐个 scene build/export，避免把整批场景都放进内存
    # 3. 输出目录镜像输入目录树，例如：
    #    metadatabfov/scene_00001/1753781394/metadata.json
    #    -> output_root/scene_00001/1753781394/
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "scene_count": 0,
        "canonical_sample_count": 0,
        "postprocess_job_count": 0,
        "passthrough_count": 0,
        "filtered_postprocess_count": 0,
        "skipped_postprocess_count": 0,
        "final_sample_count": 0,
        "processed_sample_count": 0,
        "unresolved_job_count": 0,
        "skipped_input_count": 0,
        "failed_scene_count": 0,
    }
    manifest_records = []

    if provider is not None and workers > 1:
        print("[info] --run-llm mode uses a shared provider; falling back to workers=1 for safety.", flush=True)
        workers = 1

    if workers <= 1:
        for index, input_item in enumerate(input_paths, start=1):
            result = _prepare_single_scene(
                input_item=input_item,
                input_root=input_root,
                output_root=output_root,
                max_anchors=max_anchors,
                template_path=template_path,
                postprocess_policy_path=postprocess_policy_path,
                repackage_probability=repackage_probability,
                provider=provider,
            )
            _merge_scene_result(aggregate, manifest_records, output_root, result)
            _print_scene_result(index, result)
    else:
        max_pending = max(workers * 2, workers)
        future_to_index: dict[concurrent.futures.Future, int] = {}
        next_index = 1
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for input_item in input_paths:
                future = executor.submit(
                    _prepare_single_scene,
                    input_item=input_item,
                    input_root=input_root,
                    output_root=output_root,
                    max_anchors=max_anchors,
                    template_path=template_path,
                    postprocess_policy_path=postprocess_policy_path,
                    repackage_probability=repackage_probability,
                    provider=provider,
                )
                future_to_index[future] = next_index
                next_index += 1

                if len(future_to_index) >= max_pending:
                    done, _ = concurrent.futures.wait(
                        list(future_to_index.keys()),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for done_future in done:
                        index = future_to_index.pop(done_future)
                        result = done_future.result()
                        completed += 1
                        _merge_scene_result(aggregate, manifest_records, output_root, result)
                        _print_scene_result(index, result)

            for done_future in concurrent.futures.as_completed(list(future_to_index.keys())):
                index = future_to_index[done_future]
                result = done_future.result()
                completed += 1
                _merge_scene_result(aggregate, manifest_records, output_root, result)
                _print_scene_result(index, result)

    write_json(aggregate, str(output_root / "summary.json"))
    write_jsonl(manifest_records, str(output_root / "manifest.jsonl"))
    return aggregate


def _scene_output_dir(input_metadata_path: Path, input_root: Path, output_root: Path) -> Path:
    rel_parent = input_metadata_path.relative_to(input_root).parent
    return output_root / rel_parent


def _prepare_single_scene(
    *,
    input_item: str,
    input_root: Path,
    output_root: Path,
    max_anchors: int,
    template_path: str | None,
    postprocess_policy_path: str | None,
    repackage_probability: float | None,
    provider: OpenAIResponsesProvider | None,
) -> dict:
    try:
        bundle = build_scene_bundle(
            input_item,
            max_anchors=max_anchors,
            template_path=template_path,
            postprocess_policy_path=postprocess_policy_path,
            repackage_probability=repackage_probability,
        )
        if provider is not None:
            bundle = execute_scene_bundle(bundle, provider=provider)

        scene_output_dir = _scene_output_dir(Path(input_item), input_root, output_root)
        export_scene_bundle_to_path(bundle, str(scene_output_dir))
        return {
            "status": "prepared",
            "input_path": str(input_item),
            "scene_output_dir": str(scene_output_dir),
            "scene_id": bundle["scene_id"],
            "summary": bundle["summary"],
        }
    except SceneMetadataLoadError as exc:
        return {
            "status": "skipped",
            "input_path": str(input_item),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "input_path": str(input_item),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _merge_scene_result(
    aggregate: dict,
    manifest_records: list,
    output_root: Path,
    result: dict,
) -> None:
    status = result.get("status", "prepared")
    if status == "skipped":
        aggregate["skipped_input_count"] += 1
        manifest_records.append(
            {
                "input_path": result["input_path"],
                "status": "skipped",
                "error_type": result.get("error_type", "SceneMetadataLoadError"),
                "error": result.get("error", ""),
            }
        )
        return
    if status == "failed":
        aggregate["failed_scene_count"] += 1
        manifest_records.append(
            {
                "input_path": result["input_path"],
                "status": "failed",
                "error_type": result.get("error_type", "Exception"),
                "error": result.get("error", ""),
            }
        )
        return

    aggregate["scene_count"] += 1
    for key in [
        "canonical_sample_count",
        "postprocess_job_count",
        "passthrough_count",
        "filtered_postprocess_count",
        "skipped_postprocess_count",
    ]:
        aggregate[key] += int(result["summary"].get(key, 0))
    for key in ["final_sample_count", "processed_sample_count", "unresolved_job_count"]:
        aggregate[key] += int(result["summary"].get(key, 0))

    manifest_records.append(
        {
            "input_path": result["input_path"],
            "status": "prepared",
            "relative_output_dir": str(Path(result["scene_output_dir"]).relative_to(output_root)),
            "scene_id": result["scene_id"],
            "summary": result["summary"],
        }
    )


def _print_scene_result(index: int, result: dict) -> None:
    status = result.get("status", "prepared")
    if status == "prepared":
        print(f"[prepared {index}] {result['scene_output_dir']}", flush=True)
        return
    if status == "skipped":
        print(f"[skip {index}] {result['input_path']}: {result.get('error', '')}", flush=True)
        return
    print(
        f"[error {index}] {result['input_path']}: {result.get('error_type', 'Exception')}: {result.get('error', '')}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
