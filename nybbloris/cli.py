"""nybbloris CLI: inspect / serve / train for NVFP4 LoRA on consumer Blackwell."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import __version__
from .manifest import MANIFEST_NAME, check_compat
from .plan import (LORA_BLOCKED_MOE_BACKENDS, LORA_CAPABLE_MOE_BACKENDS,
                   lm_head_status, render_plan, serve_plan)

REPO_ROOT = Path(__file__).resolve().parent.parent  # nvfp4-lora-spark/


def _require_script(path: Path) -> bool:
    """True if a shell-out helper is on disk; else print a clear reason and return False.

    The script-backed subcommands resolve helpers under REPO_ROOT/scripts, which exists for
    an editable install from a clone but NOT for a bare wheel. Without this, a missing helper
    surfaces as an opaque `python: can't open file ...`. Callers should abort cleanly instead.
    """
    if not path.exists():
        print(f"[error] required helper not found: {path}\n"
              f"        Script-backed subcommands need the source checkout "
              f"(`pip install -e .` from a clone), not a bare wheel.")
        return False
    return True


# Documented `inspect` exit codes, so a CI step / serve script can branch on the verdict:
#   0  PASS            binds + serves live as-is
#   1  FAIL / EMPTY    does not bind to this base (or no LoRA tensors found)
#   3  NO-OP/NEEDS-REKEY  binds only after re-keying (`serve --rekey auto`)
#   4  BLOCKED-ROUTED  binds, but targets are routed-expert MoE: served live ONLY on a
#                      LoRA-capable MoE backend (`--moe-backend emulation`, or marlin),
#                      NOT on the cutlass/flashinfer fast backends (supports_lora=False).
#                      Not "merge-only"; `serve` handles it by auto-forcing
#                      --moe-backend emulation rather than aborting.
VERDICT_EXIT = {"PASS": 0, "FAIL": 1, "EMPTY": 1, "NO-OP": 3, "NEEDS-REKEY": 3, "BLOCKED-ROUTED": 4}


def cmd_inspect(args):
    plan = serve_plan(args.base_model_dir, args.adapter_dir)
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(render_plan(plan))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(plan, indent=2))
        if not args.json:
            print(f"\n[inspect] wrote plan object -> {args.json_out}")
    return VERDICT_EXIT.get(plan["verdict"], 1)


def _parse_adapter(spec):
    """'name=path' or 'path' (name defaults to the dir basename)."""
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        path, name = spec, Path(spec.rstrip("/")).name
    return name, path


def _wait_ready(base_url, timeout):
    """Poll /v1/models until the server answers, or timeout (seconds)."""
    import time
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return False


def _terminate(proc):
    """Stop the vLLM process group: SIGINT (so vLLM shuts its EngineCore child down
    gracefully, avoiding the orphan that holds GPU memory), then SIGKILL as a fallback."""
    import signal
    import time
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except Exception:  # noqa: BLE001
        proc.terminate()
    for _ in range(30):
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        proc.kill()


def _derive_probe_prompt(val_file, max_chars=4000):
    """A raw probe prompt for the apply-check, built from the first val row (or None).

    The apply-check scores prompt tokens via echo-logprobs, so it wants a plain string;
    we stitch the first row's message contents. Falls back to the checker's own default
    probe when the file is missing/unreadable.
    """
    try:
        with open(val_file) as f:
            row = json.loads(f.readline())
        msgs = row.get("messages") or []
        text = "\n".join(m.get("content", "") for m in msgs
                         if isinstance(m.get("content"), str)).strip()
        return text[:max_chars] or None
    except Exception:  # noqa: BLE001
        return None


def _apply_check(base_url, base_name, adapter_names, env, prompt=None):
    """DECISIVE runtime-apply gate: prompt-echo logprob delta, base vs each adapter.

    The parity probe below is advisory: greedy text can match base even when the adapter
    IS applied (saturated prompts) or when it is a silent NO-OP, and vLLM greedy is
    nondeterministic. Identical prompt logprobs, by contrast, PROVE the adapter delta is
    not in the forward pass. Returns True iff EVERY adapter moves the logprobs past the
    threshold. Shells out to scripts/serve_apply_check.py (exit 0 = applies, else not).
    """
    import tempfile
    checker = REPO_ROOT / "scripts" / "serve_apply_check.py"
    if not _require_script(checker):
        return False
    ok = True
    print("\n[verify] runtime APPLY check (prompt-echo logprob delta, decisive):")
    for a in adapter_names:
        out = str(Path(tempfile.gettempdir()) / f"nybbloris_apply_{a}.json")
        cmd = [sys.executable, str(checker), "--base-url", base_url,
               "--base-model", base_name, "--adapter-model", a, "--out", out]
        if prompt:
            cmd += ["--prompt", prompt]
        applies = subprocess.call(cmd, env=env) == 0
        try:
            v = json.load(open(out))
            detail = f"max|delta|={v['max_abs_delta']:.3e} (sum {v['sum_delta']:+.3f})"
        except Exception:  # noqa: BLE001
            detail = "(no verdict json)"
        print(f"  {a}: {'APPLIES' if applies else 'NO-OP / error'}  {detail}")
        ok = ok and applies
    print(f"[verify] APPLY VERDICT: {'PASS' if ok else 'FAIL'} -- "
          + ("every adapter changes the forward pass." if ok
             else "an adapter loaded but did NOT apply (logprobs identical): silent no-op."))
    return ok


def _verify(base_url, base_name, adapter_names, val_file, n, max_new, threshold, env,
            max_prompt_chars=24000):
    """Runtime behavioral check: base vs each adapter, via the serve-parity probe.

    The static contract (`inspect`) proves an adapter BINDS; this proves it actually
    CHANGED behavior at runtime, catching anything that binds-but-no-ops. Uses
    char-prefix agreement vs base as a divergence proxy: low = the adapter diverged
    (applied), ~base = a possible silent no-op. Advisory, not a correctness proof --
    vLLM greedy is non-deterministic here, so it also prints a sample opening pair so
    the divergence is visible. Returns True iff every adapter diverged.
    """
    import tempfile
    probe = REPO_ROOT / "scripts" / "serve_parity_vllm.py"
    if not _require_script(probe):
        return False
    out = str(Path(tempfile.gettempdir()) / "nybbloris_verify.json")
    cmd = [sys.executable, str(probe), "--base-url", base_url, "--val-file", val_file,
           "--models", base_name, *adapter_names,
           "--n", str(n), "--max-new-tokens", str(max_new),
           "--max-prompt-chars", str(max_prompt_chars), "--out", out]
    if subprocess.call(cmd, env=env) != 0:
        print("[verify] probe failed.")
        return False
    data = json.load(open(out))
    pairs = data["summary"]["pairs"]
    examples = data.get("examples", [])
    ok = True
    print("\n[verify] runtime behavioral check (base vs adapter):")
    for a in adapter_names:
        agree = pairs.get(f"{base_name}__vs__{a}", {}).get("mean_char_prefix_agreement", 1.0)
        diverged = agree < threshold
        verdict = "DIVERGED (adapter applied)" if diverged else "~BASE (possible SILENT NO-OP)"
        print(f"  {a}: char-prefix agreement vs base = {agree:.2f} (< {threshold} => diverged) -> {verdict}")
        if examples:
            ex = examples[0]
            gb = ex["generations"].get(base_name, "")[:90].replace("\n", " ")
            ga = ex["generations"].get(a, "")[:90].replace("\n", " ")
            print(f"      base : {gb!r}")
            print(f"      {a:<8}: {ga!r}")
        ok = ok and diverged
    print(f"[verify] PARITY (advisory): {'diverged' if ok else 'WARN'} -- "
          + ("all adapters changed behavior vs base." if ok
             else "an adapter looks behaviorally identical to base (greedy can mislead; "
                  "trust the APPLY check above)."))
    return ok


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
    #    A routed-expert (MoE) adapter is NOT a hard abort: it serves live on a
    #    LoRA-capable MoE backend (emulation/marlin), so we pick that backend
    #    instead of refusing. `needs_emulation` records whether any adapter binds
    #    routed-expert deltas so the vLLM command below forces --moe-backend emulation.
    lora_modules, max_rank = [], 0
    needs_emulation = False
    base_wrapped = False
    for spec in (args.adapter or []):
        name, path = _parse_adapter(spec)
        plan = serve_plan(base, path)
        base_wrapped = base_wrapped or bool(plan["base"].get("wrapped"))
        print()
        print(render_plan(plan))
        print()

        # 2a) Provenance gate: if a manifest sits next to the adapter, REFUSE when
        #     its base fingerprint does not match the base we're about to serve on
        #     (a wrong-base load is the silent-no-op / garbage-output class).
        manifest = Path(path) / MANIFEST_NAME
        if manifest.exists():
            ok, reasons = check_compat(str(manifest), base)
            if not ok:
                print(f"[serve] REFUSING '{name}': manifest base fingerprint does not match "
                      f"the serve base:")
                for r in reasons:
                    print(f"          - {r}")
                print("        Serve the adapter on the base it was trained against, or "
                      "re-fingerprint if you are sure the bases are equivalent.")
                return 1
            print(f"[serve] manifest: base fingerprint matches (compat OK).")

        v = plan["verdict"]
        if v in ("FAIL", "EMPTY"):
            print(f"[serve] REFUSING '{name}': verdict {v} (see above).")
            return 1
        if v in ("NO-OP", "NEEDS-REKEY"):
            if args.rekey == "off":
                print(f"[serve] REFUSING '{name}': verdict {v}; re-key it or pass --rekey auto.")
                return 1
            # Pick the CORRECT rekey script by adapter layout: expert/fused adapters
            # carry stacked expert tensors the dense rekey cannot handle; only the
            # expert rekey un-stacks them into vLLM's per-expert / fused-3D layout.
            layout = plan["adapter"].get("expert_layout", "none")
            out = path.rstrip("/") + "_vllm_rekey"
            if layout == "native":
                print(f"[serve] {v}: auto-re-keying expert adapter '{name}' -> {out}")
                rekeyer = REPO_ROOT / "scripts" / "rekey_expert_lora_for_vllm.py"
                cmd = [sys.executable, str(rekeyer), "--in", path, "--out", out,
                       "--base-model", str(base)]
            else:
                print(f"[serve] {v}: auto-re-keying '{name}' -> {out}")
                rekeyer = REPO_ROOT / "scripts" / "rekey_lora_for_vllm.py"
                cmd = [sys.executable, str(rekeyer), "--in-dir", path, "--out-dir", out]
            if subprocess.call(cmd) != 0:
                return 1
            plan = serve_plan(base, out)
            # A re-keyed routed-expert adapter lands on BLOCKED-ROUTED (backend-gated),
            # which is a valid serve state on a LoRA-capable backend -- accept it.
            if plan["verdict"] not in ("PASS", "BLOCKED-ROUTED"):
                print(f"[serve] re-key did not bind (got {plan['verdict']}); aborting.")
                return 1
            path = out
        # Routed-expert deltas are backend-gated, not merge-only: serve them live on
        # the LoRA-capable emulation backend instead of aborting.
        if plan["targets"]["blocked_routed"]:
            needs_emulation = True
            live_on = "/".join(plan.get("routed", {}).get("live_on", ["emulation"]))
            print(f"[serve] '{name}': {plan['targets']['blocked_routed']} routed-expert deltas "
                  f"are backend-gated -> serving via --moe-backend emulation (live on {live_on}).")
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
    # A vision-wrapped base (*ForConditionalGeneration, e.g. Qwen3.5-122B) must serve
    # text-only, else vLLM loads the vision tower (and may need an image processor it
    # will not find). Serve the decoder alone.
    if base_wrapped:
        cmd += ["--language-model-only"]
    # MoE backend. Guard the silent-no-op trap FIRST: a routed-expert adapter on an
    # explicitly-chosen LoRA-blocked kernel (cutlass/flashinfer, supports_lora=False)
    # would load and silently do nothing -- refuse rather than let the explicit
    # backend win, unless the user passes --unsafe-moe-backend to override.
    if (needs_emulation and args.moe_backend in LORA_BLOCKED_MOE_BACKENDS
            and not args.unsafe_moe_backend):
        print(f"[serve] REFUSING: --moe-backend {args.moe_backend} cannot apply routed-expert "
              f"LoRA (supports_lora=False) and would silently NO-OP the adapter. Use a "
              f"LoRA-capable MoE backend ({'/'.join(LORA_CAPABLE_MOE_BACKENDS)}), drop "
              f"--moe-backend to auto-select emulation, or pass --unsafe-moe-backend if you "
              f"intend to serve the BASE (no adapter effect).")
        return 1
    # An explicit --moe-backend wins; else force emulation when any adapter binds
    # routed-expert deltas (the LoRA-capable MoE backend on sm_121).
    moe_backend = args.moe_backend or ("emulation" if needs_emulation else None)
    if moe_backend:
        cmd += ["--moe-backend", moe_backend]
    if lora_modules:
        rank = args.max_lora_rank or max(max_rank, 16)
        cmd += ["--enable-lora", "--max-lora-rank", str(rank),
                "--max-loras", str(max(args.max_loras, len(lora_modules))),
                "--lora-modules", *(f"{n}={p}" for n, p in lora_modules)]
    env = dict(os.environ)
    if args.allow_runtime_lora_updates:
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    else:
        # Don't let an inherited/baked VLLM_ALLOW_RUNTIME_LORA_UPDATING enable
        # runtime LoRA updates without the explicit --allow-runtime-lora-updates flag.
        env.pop("VLLM_ALLOW_RUNTIME_LORA_UPDATING", None)
    env["CUDA_HOME"] = env.get("CUDA_HOME") or "/usr/local/cuda"
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    print("[serve] launch:\n  " + " ".join(cmd))
    if args.dry_run:
        print("[serve] --dry-run: not launching.")
        return 0

    if not args.verify:
        return subprocess.call(cmd, env=env)

    # --verify: launch in the background, wait for READY, run the DECISIVE apply-check
    # (logprob delta) plus the advisory behavioral parity, then keep serving (or tear
    # down + exit on the apply verdict if --verify-only).
    if not args.val_file:
        print("[serve] --verify requires --val-file (prompts to diff base vs adapter).")
        return 2
    if not lora_modules:
        print("[serve] --verify needs at least one --adapter.")
        return 2
    base_url = f"http://localhost:{args.port}"
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        print(f"[serve] launched (pid {proc.pid}); waiting for READY (<= {args.verify_timeout}s) ...")
        if not _wait_ready(base_url, args.verify_timeout):
            print("[serve] server did not become READY in time; aborting.")
            _terminate(proc)
            return 1
        adapter_names = [n for n, _ in lora_modules]
        # Decisive gate: does the adapter actually change the forward pass?
        ok = _apply_check(base_url, served, adapter_names, env,
                          _derive_probe_prompt(args.val_file))
        # Advisory: behavioral divergence on generated text (does not change the verdict).
        try:
            _verify(base_url, served, adapter_names,
                    args.val_file, args.verify_n, args.verify_max_new_tokens,
                    args.verify_threshold, env, args.verify_max_prompt_chars)
        except Exception as e:  # noqa: BLE001
            print(f"[verify] (advisory parity probe skipped: {type(e).__name__}: {e})")
        if args.verify_only:
            print("[serve] --verify-only: stopping the server.")
            _terminate(proc)
            return 0 if ok else 1
        print("[serve] verify complete; serving continues. Ctrl-C to stop.")
        proc.wait()
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[serve] interrupted; stopping the server.")
        _terminate(proc)
        return 0


def _sniff(passthrough, name):
    """Pull '--name VALUE' or '--name=VALUE' out of a passthrough arg list."""
    for i, tok in enumerate(passthrough):
        if tok == name and i + 1 < len(passthrough):
            return passthrough[i + 1]
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return None


def cmd_train(args):
    trainer = REPO_ROOT / "scripts" / "train_nvfp4_lora.py"
    if not trainer.exists():
        print(f"[train] trainer not found at {trainer}")
        return 1
    rc = subprocess.call([sys.executable, str(trainer), *args.passthrough])
    if rc != 0:
        return rc

    # Post-train serve pre-flight: close the train -> serve loop. A freshly trained
    # adapter has flat PEFT keys, which silently NO-OP on a multimodal-wrapped base
    # at serve (the whole reason `inspect`/`serve --rekey` exist) -- so surface that
    # the moment training finishes, not at deploy time. The binding/quant verdict is
    # layout-based (target-module names + base quant), so it holds for best/ and final
    # alike. Best-effort: never let an inspect hiccup mask a successful train.
    try:
        model_dir = _sniff(args.passthrough, "--model-dir")
        out_dir = _sniff(args.passthrough, "--output-dir")
        if not (model_dir and out_dir):
            print("[train] (post-train inspect skipped: --model-dir/--output-dir not found in args)")
            return 0
        adapter = next((c for c in (Path(out_dir), Path(out_dir) / "best")
                        if (c / "adapter_config.json").exists()), None)
        if adapter is None:
            print(f"[train] (post-train inspect skipped: no adapter_config.json under {out_dir})")
            return 0
        print("\n[train] post-train serve pre-flight (will this adapter bind + apply at serve?):")
        plan = serve_plan(model_dir, str(adapter))
        print(render_plan(plan))
        v = plan["verdict"]
        if v in ("NO-OP", "NEEDS-REKEY"):
            print("[train] -> flat keys on a wrapped base; `nybbloris serve --rekey auto` re-keys it for you.")
        elif v in ("FAIL", "EMPTY"):
            print("[train] -> WARNING: does not bind to this base (see UNRESOLVED above).")
        # Stamp the provenance manifest next to the adapter (and best/ if present) so
        # the serve-time check_compat gate fires on the normal train -> serve flow,
        # not only when a manifest was written by hand. Best-effort.
        try:
            from .manifest import MANIFEST_NAME, write_manifest
            stamped = []
            for cand in {adapter, Path(out_dir), Path(out_dir) / "best"}:
                if (cand / "adapter_config.json").exists():
                    write_manifest(model_dir, cand)
                    stamped.append(str(cand / MANIFEST_NAME))
            if stamped:
                print(f"[train] wrote provenance manifest: {', '.join(stamped)}")
        except Exception as e:  # noqa: BLE001
            print(f"[train] (manifest stamp skipped: {type(e).__name__}: {e})")
    except Exception as e:  # noqa: BLE001
        print(f"[train] (post-train inspect skipped: {type(e).__name__}: {e})")
    return 0


def cmd_doctor(args):
    """Environment pre-flight: are the train/serve deps present, and which versions?

    Pure metadata + PATH probing (no torch/vllm import, no CUDA init) so it is fast
    and safe to run while a GPU job holds the device. Exits non-zero only if a CORE
    dep is missing; serve/train-specific gaps (vllm, fla, ninja, nvcc) are warnings.
    """
    import importlib.metadata as im
    import shutil

    def ver(pkg):
        try:
            return im.version(pkg)
        except Exception:  # noqa: BLE001
            return None

    rows = [("python", "OK", sys.version.split()[0])]
    for pkg, critical, note in [
        ("torch", True, "core"),
        ("transformers", True, "core"),
        ("safetensors", True, "core"),
        ("peft", False, "bf16 LoRA path"),
        ("vllm", False, "runtime-LoRA serve"),
        ("flash-linear-attention", False, "GDN training (Qwen3.x)"),
    ]:
        v = ver(pkg)
        rows.append((pkg, "OK", v) if v else (pkg, "FAIL" if critical else "WARN", f"missing ({note})"))
    for tool, note in [("ninja", "flashinfer JIT at serve"), ("nvcc", "CUDA toolchain")]:
        path = shutil.which(tool)
        rows.append((tool, "OK" if path else "WARN", path or f"not on PATH ({note})"))
    if args.base_model_dir:
        st = lm_head_status(args.base_model_dir)
        rows.append(("lm_head", "OK" if not st["quantized"] else "WARN", st["note"]))

    print("=== nybbloris doctor ===")
    for label, status, detail in rows:
        print(f"  [{status:<4}] {label:<24} {detail}")
    n_ok = sum(1 for r in rows if r[1] == "OK")
    n_warn = sum(1 for r in rows if r[1] == "WARN")
    n_fail = sum(1 for r in rows if r[1] == "FAIL")
    print(f"doctor: {'FAIL' if n_fail else 'OK'} ({n_ok} ok, {n_warn} warn, {n_fail} fail)")
    return 1 if n_fail else 0


def _passthrough_script(script_name, passthrough):
    script = REPO_ROOT / "scripts" / script_name
    if not script.exists():
        print(f"[{script_name}] not found at {script}")
        return 1
    return subprocess.call([sys.executable, str(script), *passthrough])


def cmd_data_check(args):
    """Training-data preflight (mask coverage / truncation drops / length histogram /
    tokenizer hashes). Forwards to scripts/data_check.py."""
    return _passthrough_script("data_check.py", args.passthrough)


def cmd_contamination(args):
    """Train<->eval overlap report (exact-match + n-gram + db_id). Forwards to
    scripts/check_contamination.py."""
    return _passthrough_script("check_contamination.py", args.passthrough)


# Subcommands that forward all remaining args to a script (parsed loosely in main()).
_PASSTHROUGH_CMDS = {"train", "data-check", "contamination"}


def build_parser():
    p = argparse.ArgumentParser(prog="nybbloris",
                                description="NVFP4 LoRA fit / serve on consumer Blackwell (GB10)")
    p.add_argument("--version", action="version", version=f"nybbloris {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="pre-flight serve plan for a base + adapter")
    pi.add_argument("--base-model-dir", required=True)
    pi.add_argument("--adapter-dir", required=True)
    pi.add_argument("--json-out", default=None, help="also write the plan object to this file")
    pi.add_argument("--json", action="store_true",
                    help="emit the plan as JSON on stdout (machine-readable; suppresses the human report)")
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
    ps.add_argument("--host", default="127.0.0.1",
                    help="bind address (default: localhost; pass 0.0.0.0 to bind all interfaces)")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--max-model-len", type=int, default=8192)
    ps.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ps.add_argument("--moe-backend", default=None,
                    help="force a vLLM MoE backend (e.g. emulation/marlin/cutlass); "
                         "default: emulation is auto-selected when an adapter binds "
                         "routed-expert deltas, else vLLM's own default")
    ps.add_argument("--unsafe-moe-backend", action="store_true",
                    help="allow an explicit --moe-backend that cannot apply routed-expert "
                         "LoRA (cutlass/flashinfer); the adapter will NOT affect output "
                         "(base-only serve). Off by default so the no-op is refused.")
    ps.add_argument("--max-lora-rank", type=int, default=0, help="0 = auto from the adapters (min 16)")
    ps.add_argument("--max-loras", type=int, default=2)
    ps.add_argument("--allow-runtime-lora-updates", action="store_true",
                    help="opt in to vLLM runtime LoRA load/unload via the API "
                         "(sets VLLM_ALLOW_RUNTIME_LORA_UPDATING=1; off by default)")
    ps.add_argument("--launcher", default=None,
                    help="escape hatch: hand off to a serve/run_*.sh launcher after the pre-flight")
    ps.add_argument("--dry-run", action="store_true", help="print the vLLM command, do not launch")
    ps.add_argument("--verify", action="store_true",
                    help="after READY, run a runtime base-vs-adapter behavioral check (needs --val-file)")
    ps.add_argument("--verify-only", action="store_true",
                    help="with --verify: stop the server after the check (CI gate; non-zero exit on WARN)")
    ps.add_argument("--val-file", default=None, help="jsonl of chat rows ({messages:[...]}) for --verify")
    ps.add_argument("--verify-n", type=int, default=6, help="prompts to diff for --verify")
    ps.add_argument("--verify-max-new-tokens", type=int, default=200)
    ps.add_argument("--verify-threshold", type=float, default=0.3,
                    help="char-prefix agreement vs base below which an adapter counts as DIVERGED (applied)")
    ps.add_argument("--verify-max-prompt-chars", type=int, default=24000,
                    help="skip --verify prompts longer than this (keep base+adapter+output within context)")
    ps.add_argument("--verify-timeout", type=int, default=600, help="seconds to wait for READY")
    ps.set_defaults(func=cmd_serve)

    pt = sub.add_parser("train",
                        help="LoRA fine-tune (unified trainer) + a post-train serve pre-flight; "
                             "all other args forward to scripts/train_nvfp4_lora.py "
                             "(e.g. --model-dir ... --output-dir ... --target-modules ...)")
    pt.set_defaults(func=cmd_train)

    pd = sub.add_parser("doctor", help="environment pre-flight: train/serve deps + versions")
    pd.add_argument("--base-model-dir", default=None, help="also check this base's lm_head serve-compat")
    pd.set_defaults(func=cmd_doctor)

    pdc = sub.add_parser("data-check",
                         help="training-data preflight (mask coverage, truncation drops, length "
                              "histogram, tokenizer hashes); forwards to scripts/data_check.py "
                              "(--data ... --tokenizer ... --max-length ...)")
    pdc.set_defaults(func=cmd_data_check)

    pcc = sub.add_parser("contamination",
                         help="train<->eval overlap report (exact-match + n-gram + db_id); forwards "
                              "to scripts/check_contamination.py (--train ... --eval ...)")
    pcc.set_defaults(func=cmd_contamination)
    return p


def main(argv=None):
    # `train` forwards arbitrary args to the unified trainer; parse_known_args lets
    # them through (argparse.REMAINDER drops a *leading* optional like --model-dir).
    # Other subcommands stay strict.
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if getattr(args, "cmd", None) in _PASSTHROUGH_CMDS:
        args.passthrough = extra
    elif extra:
        parser.error("unrecognized arguments: " + " ".join(extra))
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
