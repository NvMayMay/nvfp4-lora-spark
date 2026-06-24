#!/usr/bin/env python3
"""nybbloris inspect (standalone shim).

Thin wrapper over nybbloris.plan.serve_plan so the inspect pre-flight runs from a
repo checkout without a pip install. The packaged entry point is `nybbloris inspect`
(nybbloris/cli.py); the analysis lives in nybbloris/plan.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for the nybbloris package
from nybbloris.plan import render_plan, serve_plan  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="pre-flight serve plan for an NVFP4 base + LoRA adapter")
    ap.add_argument("--base-model-dir", required=True, type=Path)
    ap.add_argument("--adapter-dir", required=True, type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    plan = serve_plan(args.base_model_dir, args.adapter_dir)
    print(render_plan(plan))
    if args.json_out:
        args.json_out.write_text(json.dumps(plan, indent=2))
        print(f"\n[inspect] wrote plan object -> {args.json_out}")


if __name__ == "__main__":
    main()
