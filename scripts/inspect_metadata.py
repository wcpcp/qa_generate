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

from erp_data_generation.pipeline import audit_metadata, load_scene_metadata


def main() -> int:
    # 这是一个只读检查脚本。
    # 它不会保存文件，只负责把 scene 的 metadata 审计结果打印出来。
    parser = argparse.ArgumentParser(description="Inspect one ERP metadata file and report tier coverage plus task feasibility.")
    parser.add_argument("--input", required=True, help="Path to one scene metadata JSON file.")
    args = parser.parse_args()

    scene = load_scene_metadata(args.input)
    report = audit_metadata(scene)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
