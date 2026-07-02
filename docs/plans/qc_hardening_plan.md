# QC Hardening Plan: from research stack to trustable tool

Goal: make the repo's promise precise and dependable. "Any supported NVFP4 model
topology can be fine-tuned on Spark, and unsupported layouts fail early with a
useful report." The stack (Triton dequant kernel, `NVFP4LoRALinear`, fused-3D MoE)
is already family-agnostic; the gaps are in detection, validation, and the public
surface around it.

All file references verified against the working tree on 2026-06-12.

---

## Findings recap

| # | Severity | Finding | Evidence |
|---|----------|---------|----------|
| 1 | High | LoRA target detection is suffix-only; a checkpoint with `o_proj` quantized in some layers but BF16/FP8 in others can be partially trained without a hard failure | `scripts/train_nvfp4_lora.py:129` (`detect_lora_mode` collapses module names to suffixes), `nvfp4_lora/loader.py:39` (`list_quantized_modules` keys on `.weight` + `.weight_scale` presence, no dtype check) |
| 2 | High | Loader warns and continues on missing paths / failed assignments; under `init_empty_weights` this can leave meta tensors alive until first forward | `nvfp4_lora/loader.py:757` and `:778` (`WARN: path not found` / `WARN: failed to load`, both `continue`) |
| 3 | Medium | Fused-3D MoE support is family-shape-specific: exact HF class-name map, gate/up global scales required equal | `nvfp4_lora/experts.py:474` (class-name map), `:444` (`gate_gscale == up_gscale` hard requirement) |
| 4 | Medium | Merge scripts hardcode per-family key mapping instead of sharing the trainer's family registry | `scripts/merge_lora_into_nvfp4.py:56` (`BASE_PREFIX = "backbone."`), `scripts/merge_lora_into_ct_nvfp4.py:73` (`LM_PREFIX = "model.language_model."`) |
| 5 | Medium | Public polish mixed with local runbook state: 27 untracked files, hardcoded `/home/veritan-spark-01/...` paths in serve launchers, README quickstart still leads with `train/*.py` constant-editing | `git status` (27 untracked), `serve/run_qwen35_122b_rh_ct_dynamic_lora.sh:47-48`, `serve/run_mistral_small4_rh_lora.sh:29-32`, `README.md` Quickstart; no `pyproject.toml` |

---

## Phase 1: Public-ready hygiene

Clean the repo surface before deeper engineering.

- Make `scripts/train_nvfp4_lora.py` the primary README quickstart. The current
  Quickstart tells users to edit path constants in `train/train_super_nvfp4.py`;
  the unified trainer with a CLI should be the front door, with `train/*.py`
  documented as the original per-family research scripts.
- Triage the 27 untracked files: promote the multi-family scripts and smoke tests
  that belong on the public branch (`nvfp4_lora/training_utils.py`, the
  `smoke_tests/test_phase0_*` ladder, Mistral/Qwen3.5 scripts), move one-off
  runbooks to `docs/archive/` or label them as local research artifacts.
- Replace hardcoded `/home/veritan-spark-01/...` defaults in `serve/*.sh` with
  required env vars plus a documented example block. A user-facing launcher that
  silently defaults to someone else's home directory is a trust killer.
- Add `pyproject.toml` with the package, a `[test]` extra, and pinned
  GB10-critical deps, so `pip install -e '.[test]' && pytest tests/ -q` is
  one-command reproducible.
- Add a release checklist doc: `compileall`, `bash -n` on all shell scripts,
  CPU pytest, one trainer `--dry-run`, one save/resume smoke on real hardware.

Acceptance: fresh clone on a clean machine reaches a passing `pytest tests/ -q`
with no manual edits; README quickstart references only commands that take
arguments rather than requiring source edits.

## Phase 2: Checkpoint inspector (biggest leverage)

Add `scripts/inspect_nvfp4_checkpoint.py`, the first command a user runs on any
checkpoint. Reads only `config.json` + `model.safetensors.index.json` (cheap, no
weights touched) and reports:

- model type and resolved family (or "unknown, here is what we see")
- quant format per module: ModelOpt NVFP4, compressed-tensors NVFP4, FP8, BF16
- attention/MLP target coverage per layer, including mixed-layer cases
- MoE topology: per-expert linears vs fused-3D, expert count, dims
- for a proposed `--target-modules` list: exactly which modules match, in which
  format, and whether the run would be native NVFP4-LoRA, PEFT, or a hard error
- known unsupported assumptions (e.g. mismatched gate/up global scales, finding 3)

Output human-readable text plus `--json` for tooling. This converts "porting to
a new family" from grepping safetensors indexes by hand into running one command.

Acceptance: inspector gives correct verdicts on all four known families
(Nemotron ModelOpt, Qwen3.5/3.6 ModelOpt, Mistral-Small-4 compressed-tensors)
plus a synthetic mixed-format fixture.

## Phase 3: Fail-fast target coverage (fixes finding 1)

Replace suffix-collapse detection with a full module inventory.

- `detect_lora_mode` enumerates every module matching each target suffix and
  classifies each one individually (native NVFP4 / PEFT-able BF16 / FP8 / absent),
  instead of collapsing to a suffix set.
- Hard error when one suffix matches a mix of quantized and unquantized modules,
  unless `--allow-partial-targets` is passed explicitly.
- Hard error when target modules are FP8-demoted (the current Nemotron checkpoints
  demote some attention layers), unless explicitly allowed.
- Log exact counts: requested, matched, native, PEFT, FP8-demoted, skipped, missing.
- Write the coverage report as JSON into the adapter output dir, so every adapter
  records exactly what was and was not trained.

Acceptance: CPU tests with synthetic safetensors indexes covering the
mixed-layer `o_proj` case, the FP8-demotion case, and the all-clean case; the
mixed case must exit nonzero without `--allow-partial-targets`.

## Phase 4: Loader hardening (fixes finding 2)

Remove "load succeeded but the model is broken later" behavior.

- Turn the two `WARN ... continue` sites in `load_non_nvfp4_weights`
  (`nvfp4_lora/loader.py:757`, `:778`) into errors by default; keep a
  `--permissive-load` flag for bring-up of new families only.
- Maintain an explicit per-family allowlist of intentionally skipped branches
  (vision tower, projector, MTP head) in the family registry, not as silent skips.
- After loading, assert `no_meta_params_or_buffers()` across the model, with
  only allowlisted modules exempt. A meta tensor surviving load is a load bug,
  full stop.
- Emit a loader summary JSON: tensors loaded, tensors skipped (and under which
  allowlist entry), modules replaced with NVFP4LoRALinear / NVFP4Experts3D,
  any remaining meta tensors.

Acceptance: a deliberately corrupted index (one renamed key) fails at load time
with the offending key named, not at first forward with a meta-tensor error.

## Phase 5: Shared family registry (enables findings 3 and 4 fixes)

Trainer, loader, merge, and serve each carry overlapping family knowledge today.

- Create `nvfp4_lora/families.py` holding, per family: `FAMILIES` metadata
  (auto class, expert prefixes, PEFT scope, frozen towers), safetensors key
  translation, merge adapter-key mapping, skipped-branch allowlist, and quant
  quirks (e.g. QKV shared-scale grouping, gate/up scale assumptions).
- Port `scripts/train_nvfp4_lora.py`, `nvfp4_lora/loader.py`, and both merge
  scripts onto it.
- Tests asserting train-side and merge-side key translation are identical for
  Nemotron, Qwen3.5, and Mistral4.

Acceptance: a new family is added by writing one registry entry plus tests; no
edits inside loader/trainer/merge bodies.

## Phase 6: Merge generalization (fixes finding 4)

- Refactor `merge_lora_into_nvfp4.py` and `merge_lora_into_ct_nvfp4.py` around
  the registry's adapter-key translation; family-specific requant rules stay,
  but declared in the registry rather than as module-level constants.
- Add `--dry-run` to both: validate that every adapter target maps to a real
  quantized base tensor and print the coverage table, writing nothing.
- Produce a merge manifest JSON: family, quant format, target coverage,
  per-tensor worst cosine similarity stats.

Acceptance: merge dry-run tests on tiny synthetic indexes + adapters for all
three key layouts; a stale adapter against the wrong base fails in dry-run with
the unmapped keys listed.

## Phase 7: Documented topology contract (addresses finding 3 honestly)

Make "any NVFP4 model" true by narrowing the contract, then expanding it.

- Document supported topology v1: ModelOpt or compressed-tensors NVFP4 linear
  weights, group size 16, standard E2M1 packing, the known fused-3D MoE layouts
  (exact class names in `nvfp4_lora/experts.py:474`).
- For fused-3D MoE outside that set, fail with a message naming exactly which
  assumption broke (class name, dims, key names, gate/up scale equality).
- Backlog item: support separate gate/up global scales (store two scales per
  expert instead of asserting equality at `experts.py:444`).
- Add a porting checklist doc whose step 1 is the Phase 2 inspector.

## Phase 8: Test ladder and CI

- CPU tests (no GPU, synthetic fixtures): full target inventory, partial
  quantization detection, FP8-demotion failure, no-meta assertion, merge
  dry-run key mapping. Extends the existing `tests/` suite
  (`test_lora_mode_detection.py`, `test_key_translation.py`, fixtures dir).
- GPU smokes standardized as a validation ladder, run in order on real
  hardware: inspect, trainer `--dry-run`, 3-step train, save/resume,
  merge dry-run, merge validate.
- CI: `compileall` + `bash -n` + CPU pytest on every push.

---

## Suggested order

1. Phase 1 (hygiene, README, packaging) - unblocks everything public-facing
2. Phase 2 (inspector) - biggest user-facing leverage, no behavior changes
3. Phase 3 (fail-fast targets) - closes the High-severity silent-partial-train hole
4. Phase 4 (loader no-meta) - closes the High-severity silent-load hole
5. Phase 5 (family registry) - refactor, sets up 6
6. Phase 6 (merge refactor)
7. Phase 7 + 8 (contract docs, expanded topology, CI) - ongoing

Phases 3 and 4 are the two that change failure behavior; everything else is
additive or organizational.
