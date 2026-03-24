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

from erp_data_generation.exporters import write_json
from erp_data_generation.pipeline import build_scene_plan, load_scene_metadata


def main() -> int:
    # 这个脚本只负责“读入一个 scene metadata -> 生成 scene-level task plan”。
    # 它不会生成最终 QA，只会输出后续数据生成要围绕哪些单元出题。
    parser = argparse.ArgumentParser(description="Generate a v2 scene-level ERP task plan with task-gated outputs.")
    parser.add_argument("--input", required=True, help="Path to one scene metadata JSON file.")
    parser.add_argument("--output", help="Optional output JSON path.")
    parser.add_argument("--max-anchors", type=int, default=6, help="Maximum anchor entities.")
    args = parser.parse_args()

    # 先把原始 metadata 归一化为内部统一的 SceneMetadata / Entity 结构。
    scene = load_scene_metadata(args.input)
    # 再基于 task feasibility、anchor 选择、pair mining 等逻辑构建 scene plan。
    plan = build_scene_plan(scene, max_anchors=args.max_anchors)

    if args.output:
        write_json(plan, args.output)
    else:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
