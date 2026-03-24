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

from erp_data_generation.builders import build_canonical_samples
from erp_data_generation.exporters import write_json, write_jsonl
from erp_data_generation.pipeline import build_scene_plan, load_scene_metadata


def main() -> int:
    # 这个脚本负责从一个 scene metadata 生成 canonical QA。
    # 它内部会先得到 scene plan，再把 task 转成 rule-based sample。
    parser = argparse.ArgumentParser(
        description="Generate rule-based canonical QA samples from ERP scene metadata."
    )
    parser.add_argument("--input", required=True, help="Path to one scene metadata JSON file.")
    parser.add_argument("--output", help="Optional output path. Use .jsonl for JSONL export.")
    parser.add_argument("--max-anchors", type=int, default=6, help="Maximum anchor entities.")
    parser.add_argument(
        "--template-path",
        help="Optional path to a question template JSON file.",
    )
    args = parser.parse_args()

    # 和 scene-plan 脚本一样，这里先把原始 metadata 读成统一结构。
    scene = load_scene_metadata(args.input)
    # canonical sample 不是直接从 metadata 平铺出来的，而是先依赖 scene plan。
    plan = build_scene_plan(scene, max_anchors=args.max_anchors)
    # 再把 planning task 逐个落成 canonical QA sample。
    samples = build_canonical_samples(scene, plan, template_path=args.template_path)

    if args.output:
        if args.output.endswith(".jsonl"):
            write_jsonl(samples, args.output)
        else:
            write_json(
                {
                    "scene_id": scene.scene_id,
                    "sample_count": len(samples),
                    "samples": samples,
                },
                args.output,
            )
    else:
        print(json.dumps(samples, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
