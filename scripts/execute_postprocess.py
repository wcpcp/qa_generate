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

from erp_data_generation.exporters import export_scene_execution, read_jsonl, write_json, write_jsonl
from erp_data_generation.postprocess_execution import derive_execution_context, execute_postprocess_jobs
from erp_data_generation.providers import OpenAIResponsesProvider


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute LLM postprocess from previously prepared canonical_samples.jsonl and postprocess_jobs.jsonl."
    )
    parser.add_argument("--input", required=True, help="Prepared root directory or one prepared scene directory.")
    parser.add_argument("--output-dir", help="Optional output directory. Defaults to in-place execution under --input.")
    parser.add_argument("--model", help="Provider model override.")
    parser.add_argument("--base-url", help="Provider base URL override.")
    parser.add_argument("--timeout-seconds", type=int, default=180, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry count for provider calls.")
    parser.add_argument("--cache-dir", help="Optional provider cache directory.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable provider response caching.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scene dirs that already contain final_samples.jsonl.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_dir) if args.output_dir else input_path
    provider = OpenAIResponsesProvider(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        cache_dir=args.cache_dir,
        use_cache=not args.disable_cache,
    )

    scene_dirs = discover_prepared_scene_dirs(input_path)
    aggregate = {
        "scene_count": 0,
        "executed_scene_count": 0,
        "skipped_scene_count": 0,
        "final_sample_count": 0,
        "processed_sample_count": 0,
        "unresolved_job_count": 0,
    }
    manifest_records = []
    all_final_samples = []
    all_unresolved_jobs = []

    for scene_dir in scene_dirs:
        aggregate["scene_count"] += 1
        target_scene_dir = _map_scene_dir(scene_dir, input_path, output_root)
        if args.skip_existing and (target_scene_dir / "final_samples.jsonl").exists():
            aggregate["skipped_scene_count"] += 1
            continue

        canonical_samples = read_jsonl(str(scene_dir / "canonical_samples.jsonl"))
        jobs = read_jsonl(str(scene_dir / "postprocess_jobs.jsonl"))
        passthrough_ids, filtered_ids = derive_execution_context(canonical_samples)
        execution = execute_postprocess_jobs(
            jobs,
            canonical_samples,
            passthrough_ids,
            filtered_ids,
            provider=provider,
        )

        scene_summary = {}
        summary_path = scene_dir / "summary.json"
        if summary_path.exists():
            scene_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        scene_summary = {
            **scene_summary,
            "final_sample_count": execution["summary"]["final_sample_count"],
            "processed_sample_count": execution["summary"]["processed_count"],
            "unresolved_job_count": execution["summary"]["unresolved_count"],
        }
        export_scene_execution(execution, scene_summary, str(target_scene_dir))

        aggregate["executed_scene_count"] += 1
        aggregate["final_sample_count"] += int(execution["summary"]["final_sample_count"])
        aggregate["processed_sample_count"] += int(execution["summary"]["processed_count"])
        aggregate["unresolved_job_count"] += int(execution["summary"]["unresolved_count"])
        all_final_samples.extend(execution["final_samples"])
        all_unresolved_jobs.extend(execution["unresolved_jobs"])
        manifest_records.append(
            {
                "scene_dir": str(scene_dir),
                "relative_output_dir": str(target_scene_dir.relative_to(output_root)),
                "summary": scene_summary,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(aggregate, str(output_root / "execution_summary.json"))
    write_jsonl(manifest_records, str(output_root / "execution_manifest.jsonl"))
    if all_final_samples:
        write_jsonl(all_final_samples, str(output_root / "final_samples.jsonl"))
    write_json(
        {
            "summary": {
                "final_sample_count": aggregate["final_sample_count"],
                "unresolved_job_count": aggregate["unresolved_job_count"],
            },
            "unresolved_jobs": all_unresolved_jobs,
        },
        str(output_root / "postprocess_execution.json"),
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


def discover_prepared_scene_dirs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        raise ValueError("--input must point to a prepared scene directory or a prepared root directory, not a file.")
    if (input_path / "canonical_samples.jsonl").exists() and (input_path / "postprocess_jobs.jsonl").exists():
        return [input_path]
    return sorted(
        path.parent
        for path in input_path.rglob("canonical_samples.jsonl")
        if path.is_file() and (path.parent / "postprocess_jobs.jsonl").exists()
    )


def _map_scene_dir(scene_dir: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_dir() and (input_root / "canonical_samples.jsonl").exists():
        return output_root
    return output_root / scene_dir.relative_to(input_root)


if __name__ == "__main__":
    raise SystemExit(main())
