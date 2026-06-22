"""nybbloris CLI: inspect / serve / train for NVFP4 LoRA on consumer Blackwell."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import __version__
from .plan import lm_head_status, render_plan, serve_plan

REPO_ROOT = Path(__file__).resolve().parent.parent  # nvfp4-lora-spark/


def cmd_inspect(args):
    plan = serve_plan(args.base_model_dir, args.adapter_dir, allow_fp8_targets=args.allow_fp8_targets)
    print(render_plan(plan))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(plan, indent=2))
        print(f"\n[inspect] wrote plan object -> {args.json_out}")
    # non-zero only when the adapter would not bind, so this can gate a serve script / CI.
    return 0 if plan["verdict"] != "FAIL" else 1


def _parse_adapter(spec):
    """'name=path' or 'path' (name defaults to the dir basename)."""
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        path, name = spec, Path(spec.rstrip("/")).name
    return name, path


def cmd_serve(args):
    base = args.base_model_dir

    # 1) Checkpoint-compat pre-flight: vLLM can't load a quantized lm_head.
    st = lm_head_status(base)
    print(f"[serve] lm_head: {st['note']}")
    if st["quantized"]:
        fixer = REPO_ROOT / "scripts" / "fix_nvfp4_lm_head.py"
        if args.fix_lm_head:
            print(f"[serve] --fix-lm-head: applying {fixer.name} (backs up shard + index) ...")
            if subprocess.call([sys.executable, str(fixer), "--model-dir", str(base), "--apply"]) != 0:
                return 1
        else:
            print("[serve] REFUSING: vLLM keeps lm_head bf16; a quantized lm_head crashes it. Fix first:")
            print(f"        python {fixer.relative_to(REPO_ROOT)} --model-dir {base}            # dry-run")
            print(f"        python {fixer.relative_to(REPO_ROOT)} --model-dir {base} --apply     # then write")
            print("        ...or re-run `nybbloris serve` with --fix-lm-head.")
            return 1

    # 2) Binding pre-flight per adapter: refuse no-binds, auto-re-key silent no-ops.
    lora_modules, max_rank = [], 0
    for spec in (args.adapter or []):
        name, path = _parse_adapter(spec)
        plan = serve_plan(base, path)
        print()
        print(render_plan(plan))
        print()
        v = plan["verdict"]
        if v in ("FAIL", "EMPTY"):
            print(f"[serve] REFUSING '{name}': verdict {v} (see above).")
            return 1
        if v in ("NO-OP", "NEEDS-REKEY"):
            if args.rekey == "off":
                print(f"[serve] REFUSING '{name}': verdict {v}; re-key it or pass --rekey auto.")
                return 1
            out = path.rstrip("/") + "_vllm_rekey"
            print(f"[serve] {v}: auto-re-keying '{name}' -> {out}")
            rekeyer = REPO_ROOT / "scripts" / "rekey_lora_for_vllm.py"
            if subprocess.call([sys.executable, str(rekeyer), "--in-dir", path, "--out-dir", out]) != 0:
                return 1
            plan = serve_plan(base, out)
            if plan["verdict"] != "PASS":
                print(f"[serve] re-key did not yield PASS (got {plan['verdict']}); aborting.")
                return 1
            path = out
        lora_modules.append((name, path))
        max_rank = max(max_rank, int(plan["adapter"].get("r") or 0))

    # 3a) Escape hatch: an explicit hand-written launcher (e.g. the NGC-docker recipe).
    if args.launcher:
        print(f"[serve] pre-flight OK; handing off to launcher {args.launcher} ...")
        return subprocess.call(["bash", args.launcher])

    # 3b) Build + launch the host-venv vLLM serve (the proven canonical recipe).
    # Run vllm THROUGH a python interpreter rather than exec'ing the `vllm` script
    # directly: a copied/relocated venv keeps a stale shebang (points at the source
    # box's python), so `bin/python bin/vllm` is the portable invocation. Prefer an
    # explicit --python, else the interpreter beside the vllm script, else exec direct.
    py = args.python
    if py is None:
        sibling = Path(args.vllm).parent / "python"
        py = str(sibling) if sibling.exists() else None
    served = args.served_model_name or Path(str(base).rstrip("/")).name
    cmd = ([py, args.vllm] if py else [args.vllm]) + [
           "serve", str(base),
           "--served-model-name", served,
           "--host", args.host, "--port", str(args.port),
           "--max-model-len", str(args.max_model_len),
           "--gpu-memory-utilization", str(args.gpu_memory_utilization),
           "--enforce-eager"]
    if lora_modules:
        rank = args.max_lora_rank or max(max_rank, 16)
        cmd += ["--enable-lora", "--max-lora-rank", str(rank),
                "--max-loras", str(max(args.max_loras, len(lora_modules))),
                "--lora-modules", *(f"{n}={p}" for n, p in lora_modules)]
    env = dict(os.environ)
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    env["CUDA_HOME"] = env.get("CUDA_HOME") or "/usr/local/cuda"
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    print("[serve] launch:\n  " + " ".join(cmd))
    if args.dry_run:
        print("[serve] --dry-run: not launching.")
        return 0
    return subprocess.call(cmd, env=env)


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

    ps = sub.add_parser("serve",
                        help="pre-flight gate (lm_head + binding), then start the dynamic-LoRA vLLM serve")
    ps.add_argument("--base-model-dir", required=True)
    ps.add_argument("--adapter", action="append", metavar="NAME=PATH",
                    help="adapter to register (repeatable); NAME defaults to the dir basename")
    ps.add_argument("--rekey", choices=["auto", "off"], default="auto",
                    help="auto: re-key a silent-no-op adapter to the serve layout (default); off: refuse")
    ps.add_argument("--fix-lm-head", action="store_true",
                    help="auto-run scripts/fix_nvfp4_lm_head.py --apply if the base has a quantized lm_head")
    ps.add_argument("--vllm", default="vllm",
                    help="vllm entrypoint (e.g. /path/to/qwen-serve/bin/vllm for the host venv)")
    ps.add_argument("--python", default=None,
                    help="interpreter to run vllm through (default: the python beside --vllm, "
                         "for a relocated venv with a stale shebang; else exec vllm directly)")
    ps.add_argument("--served-model-name", default=None, help="defaults to the base dir basename")
    ps.add_argument("--host", default="0.0.0.0")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--max-model-len", type=int, default=8192)
    ps.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ps.add_argument("--max-lora-rank", type=int, default=0, help="0 = auto from the adapters (min 16)")
    ps.add_argument("--max-loras", type=int, default=2)
    ps.add_argument("--launcher", default=None,
                    help="escape hatch: hand off to a serve/run_*.sh launcher after the pre-flight")
    ps.add_argument("--dry-run", action="store_true", help="print the vLLM command, do not launch")
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
