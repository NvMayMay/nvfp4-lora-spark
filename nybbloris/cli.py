"""nybbloris CLI: inspect / serve / train for NVFP4 LoRA on consumer Blackwell."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import __version__
from .plan import render_plan, serve_plan

REPO_ROOT = Path(__file__).resolve().parent.parent  # nvfp4-lora-spark/


def cmd_inspect(args):
    plan = serve_plan(args.base_model_dir, args.adapter_dir, allow_fp8_targets=args.allow_fp8_targets)
    print(render_plan(plan))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(plan, indent=2))
        print(f"\n[inspect] wrote plan object -> {args.json_out}")
    # non-zero only when the adapter would not bind, so this can gate a serve script / CI.
    return 0 if plan["verdict"] != "FAIL" else 1


def cmd_serve(args):
    plan = serve_plan(args.base_model_dir, args.adapter_dir)
    print(render_plan(plan))
    print()
    if plan["verdict"] == "FAIL":
        print("[serve] refusing: adapter does not bind to this base (see UNRESOLVED above).")
        return 1
    if args.launcher:
        print(f"[serve] pre-flight OK; launching {args.launcher} ...")
        return subprocess.call(["bash", args.launcher])
    print("[serve] pre-flight only. Pass --launcher serve/run_*.sh to start the vLLM serve.")
    return 0


def cmd_train(args):
    trainer = REPO_ROOT / "scripts" / "train_nvfp4_lora.py"
    if not trainer.exists():
        print(f"[train] trainer not found at {trainer}")
        return 1
    return subprocess.call([sys.executable, str(trainer), *args.passthrough])


def build_parser():
    p = argparse.ArgumentParser(prog="nybbloris",
                                description="NVFP4 LoRA fit / serve on consumer Blackwell (GB10)")
    p.add_argument("--version", action="version", version=f"nybbloris {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="pre-flight serve plan for a base + adapter")
    pi.add_argument("--base-model-dir", required=True)
    pi.add_argument("--adapter-dir", required=True)
    pi.add_argument("--allow-fp8-targets", action="store_true",
                    help="count FP8 targets as live (only if the serve loader enables FP8 LoRA)")
    pi.add_argument("--json-out", default=None)
    pi.set_defaults(func=cmd_inspect)

    ps = sub.add_parser("serve", help="pre-flight gate, then start the dynamic-LoRA vLLM serve")
    ps.add_argument("--base-model-dir", required=True)
    ps.add_argument("--adapter-dir", required=True)
    ps.add_argument("--launcher", default=None, help="path to a serve/run_*.sh docker launcher")
    ps.set_defaults(func=cmd_serve)

    pt = sub.add_parser("train", help="LoRA fine-tune (pass-through to the unified trainer)")
    pt.add_argument("passthrough", nargs=argparse.REMAINDER,
                    help="arguments forwarded to scripts/train_nvfp4_lora.py")
    pt.set_defaults(func=cmd_train)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
