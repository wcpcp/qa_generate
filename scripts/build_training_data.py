#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
)
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
        bundle = build_scene_bundle(
            str(input_path),
            max_anchors=args.max_anchors,
            template_path=args.template_path,
            postprocess_policy_path=args.postprocess_policy_path,
            repackage_probability=args.repackage_probability,
        )
        if args.run_llm:
            provider = _build_provider(args)
            bundle = execute_scene_bundle(bundle, provider=provider)
        export_scene_bundle(bundle, args.output_dir)
        print(json.dumps(bundle["summary"], ensure_ascii=False, indent=2))
        return 0

    inputs = discover_scene_inputs(str(input_path), metadata_filename=args.metadata_filename)
    provider = _build_provider(args) if args.run_llm else None
    summary = _stream_directory_build(
        input_root=input_path,
        input_paths=inputs,
        output_root=Path(args.output_dir),
        max_anchors=args.max_anchors,
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
    input_paths: list[str],
    output_root: Path,
    max_anchors: int,
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
    }
    manifest_records = []

    for input_item in input_paths:
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

        aggregate["scene_count"] += 1
        for key in [
            "canonical_sample_count",
            "postprocess_job_count",
            "passthrough_count",
            "filtered_postprocess_count",
            "skipped_postprocess_count",
        ]:
            aggregate[key] += int(bundle["summary"].get(key, 0))
        for key in ["final_sample_count", "processed_sample_count", "unresolved_job_count"]:
            aggregate[key] += int(bundle["summary"].get(key, 0))

        manifest_records.append(
            {
                "input_path": str(input_item),
                "relative_output_dir": str(scene_output_dir.relative_to(output_root)),
                "scene_id": bundle["scene_id"],
                "summary": bundle["summary"],
            }
        )

    write_json(aggregate, str(output_root / "summary.json"))
    write_jsonl(manifest_records, str(output_root / "manifest.jsonl"))
    return aggregate


def _scene_output_dir(input_metadata_path: Path, input_root: Path, output_root: Path) -> Path:
    rel_parent = input_metadata_path.relative_to(input_root).parent
    return output_root / rel_parent


if __name__ == "__main__":
    raise SystemExit(main())
