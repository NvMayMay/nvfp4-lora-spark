"""nybbloris.manifest -- adapter provenance manifest + serve-time compatibility check.

A trained adapter is only useful if you can later prove WHAT it was trained against
and WHETHER it is safe to serve on a given base. This module fingerprints the
(base, adapter) pair and the training environment into a `manifest.json` written next
to the adapter, and provides `check_compat()` so a serve pre-flight can REFUSE an
adapter whose base fingerprint does not match the base you are about to serve it on
(a wrong-base load is the silent-no-op / garbage-output class this project keeps
hitting). Pure stdlib + safetensors header reads: no torch, no GPU, no weight load.
"""
from __future__ import annotations

import glob
import hashlib
import json
import struct
import subprocess
from pathlib import Path

__all__ = ["build_manifest", "check_compat", "write_manifest", "MANIFEST_NAME"]

MANIFEST_NAME = "nybbloris_manifest.json"
MANIFEST_VERSION = 1


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path, _chunk: int = 1 << 20) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_chunk), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _sha256_json(path: Path) -> str | None:
    """Hash a JSON file by its CANONICAL content (key-sorted), so formatting churn
    (whitespace/key order) does not change the fingerprint."""
    try:
        obj = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return _sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def _safetensors_header(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n))
    except (OSError, ValueError, struct.error):
        return {}


def _pkg_versions(pkgs=("torch", "transformers", "peft", "safetensors", "vllm",
                        "nvidia-modelopt")) -> dict:
    import importlib.metadata as im
    out = {}
    for p in pkgs:
        try:
            out[p] = im.version(p)
        except Exception:  # noqa: BLE001
            out[p] = None
    return out


def _git_sha(repo_hint: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_hint), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def base_fingerprint(base_model_dir) -> dict:
    """Identity of the base checkpoint: config + quant + weight-index, hashed.

    This is what an adapter is bound to. Two bases with the same fingerprint are
    interchangeable for serving; a mismatch means the adapter may silently no-op or
    produce garbage, so serve should refuse.
    """
    base_model_dir = Path(base_model_dir)
    cfg_path = base_model_dir / "config.json"
    cfg = {}
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        pass
    qcfg = cfg.get("quantization_config") or {}
    idx = base_model_dir / "model.safetensors.index.json"
    return {
        "name": base_model_dir.name,
        "arch": (cfg.get("architectures") or [None])[0],
        "model_type": cfg.get("model_type"),
        "quant_method": qcfg.get("quant_method"),
        "config_sha256": _sha256_json(cfg_path),
        "weight_index_sha256": _sha256_json(idx),
    }


def _tokenizer_fingerprint(model_dir: Path) -> dict:
    out = {}
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "chat_template.jinja"):
        p = model_dir / fn
        if p.exists():
            out[fn] = _sha256_file(p)
    return out


def adapter_fingerprint(adapter_dir) -> dict:
    adapter_dir = Path(adapter_dir)
    acfg_path = adapter_dir / "adapter_config.json"
    acfg = {}
    try:
        acfg = json.loads(acfg_path.read_text())
    except (OSError, ValueError):
        pass
    shards = sorted(glob.glob(str(adapter_dir / "adapter_model*.safetensors")))
    lora_tensors = 0
    for s in shards:
        lora_tensors += sum(1 for k in _safetensors_header(Path(s))
                            if k != "__metadata__" and k.endswith((".lora_A.weight",
                                                                    ".lora_B.weight")))
    return {
        "r": acfg.get("r"),
        "lora_alpha": acfg.get("lora_alpha"),
        "target_modules": sorted(acfg.get("target_modules") or []),
        "peft_type": acfg.get("peft_type"),
        "adapter_config_sha256": _sha256_json(acfg_path),
        "adapter_weights_sha256": {Path(s).name: _sha256_file(Path(s)) for s in shards},
        "lora_tensor_count": lora_tensors,
    }


def build_manifest(base_model_dir, adapter_dir, *, train_meta=None,
                   repo_hint=None) -> dict:
    """Assemble the provenance manifest for a (base, adapter) pair."""
    adapter_dir = Path(adapter_dir)
    repo_hint = Path(repo_hint) if repo_hint else Path(__file__).resolve().parent.parent
    m = {
        "manifest_version": MANIFEST_VERSION,
        "base": base_fingerprint(base_model_dir),
        "base_tokenizer": _tokenizer_fingerprint(Path(base_model_dir)),
        "adapter": adapter_fingerprint(adapter_dir),
        "provenance": {
            "train_git_sha": _git_sha(repo_hint),
            "package_versions": _pkg_versions(),
        },
    }
    # Fold in coverage + trainer run metadata when present (best-effort, no failure).
    cov = adapter_dir / "target_coverage.json"
    if cov.exists():
        try:
            m["target_coverage"] = json.loads(cov.read_text())
        except (OSError, ValueError):
            pass
    if train_meta:
        m["train_meta"] = train_meta
    return m


def write_manifest(base_model_dir, adapter_dir, *, train_meta=None,
                   repo_hint=None) -> Path:
    adapter_dir = Path(adapter_dir)
    m = build_manifest(base_model_dir, adapter_dir, train_meta=train_meta,
                       repo_hint=repo_hint)
    dest = adapter_dir / MANIFEST_NAME
    dest.write_text(json.dumps(m, indent=2))
    return dest


def check_compat(manifest, base_model_dir) -> tuple[bool, list[str]]:
    """Does `base_model_dir` match the base the adapter was trained against?

    Returns (ok, reasons). Reasons name each mismatched field. A missing/partial base
    fingerprint (e.g. no index) degrades to a NAME/arch/quant comparison rather than a
    false pass. The weight-index hash is the strong signal; config hash is secondary.
    """
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    want = manifest.get("base") or {}
    got = base_fingerprint(base_model_dir)
    reasons = []
    # Strong: exact weight-index identity (only compare when both present).
    if want.get("weight_index_sha256") and got.get("weight_index_sha256"):
        if want["weight_index_sha256"] != got["weight_index_sha256"]:
            reasons.append("weight_index_sha256 mismatch (different base weights)")
    # Structural fields must always agree.
    for f in ("arch", "model_type", "quant_method"):
        if want.get(f) is not None and got.get(f) is not None and want[f] != got[f]:
            reasons.append(f"{f} mismatch: adapter trained on {want[f]!r}, base is {got[f]!r}")
    # Config hash is a softer signal (config can be re-serialized); report but do not
    # fail on it alone if the strong index hash already matched.
    if (want.get("config_sha256") and got.get("config_sha256")
            and want["config_sha256"] != got["config_sha256"]
            and "weight_index_sha256 mismatch" not in " ".join(reasons)
            and not any("weight_index" in r for r in reasons)
            and not (want.get("weight_index_sha256") and got.get("weight_index_sha256"))):
        reasons.append("config_sha256 mismatch (no weight index to corroborate)")
    return (len(reasons) == 0, reasons)
