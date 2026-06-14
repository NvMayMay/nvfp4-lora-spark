# Scope: attention LoRA on Nemotron-3

Status: scoping only (no code). Investigated against the on-disk Nano-30B and
Super-120B NVFP4 checkpoints by reading config + safetensors index; no weights
loaded, no GPU.

## Headline finding (changes the shape of the work)

Nemotron-3 attention projections are **BF16, not NVFP4**. So attention LoRA is
a **PEFT-path** job, not a native-NVFP4-LoRA build. The unified trainer already
has that path (it is how Mistral-Small-4 MLA attention trains). This is much
smaller than "teach `NVFP4LoRALinear` to target Mamba2-attention" — most of it
likely already runs; the work is validation plus a couple of correctness fixes.

### What the checkpoints actually contain

Nemotron-3 is a hybrid stack; the `hybrid_override_pattern` marks layer types
(`M` = Mamba2, `E` = MoE/MLP, `*` = attention):

| | attention layers (`*` in `hybrid_override_pattern`) | attention projection storage | naming |
|---|---|---|---|
| Nano-30B | 6 (idx 5,12,19,26,33,42) | q/k/v/o all **bf16** (24/24) | `backbone.layers.N.mixer.{q,k,v,o}_proj` |
| Super-120B | 8 (idx 7,16,25,36,47,58,69,78) | 30 bf16 + **2 fp8** (`layers.69` & `layers.78` `o_proj`) | `model.layers.N.mixer.{q,k,v,o}_proj` |

(Mamba2 layers — e.g. Super layer 0 — also live under `mixer.` but expose
`in_proj`/`out_proj`/`conv1d`, not q/k/v/o, so attention targeting never hits
them. The `mtp.*` Multi-Token Prediction head also carries an attention block
with q/k/v/o, but it is on the family skip-list and never trains. Super's
in-memory backbone prefix is `model.`, Nano's is `backbone.`.)

Notes that matter downstream:
- Projections sit under `mixer.`, not `self_attn.`. The `nemotron_h`
  `peft_scope` (`^(model|backbone)\.layers\.`) composed by `attach_peft_lora`
  becomes `^(model|backbone)\.layers\..*\.(q_proj|k_proj|v_proj|o_proj)$`,
  which DOES match `backbone.layers.5.mixer.q_proj`. The Mamba2 layers expose
  `in_proj`/`out_proj` (not q/k/v/o), so targeting q/k/v/o hits only the
  attention layers — no accidental Mamba targeting.
- GQA: 32 query heads, 2 KV heads (Nano), so q_proj and k/v_proj have
  different output dims. LoRA is shape-agnostic; not an issue.

## What almost certainly already works

`scripts/train_nvfp4_lora.py --model-dir <nano> --target-modules q_proj,k_proj,v_proj,o_proj`
resolves to `lora_mode=peft` (all targets bf16) and attaches PEFT LoRA to the
attention layers, with the NVFP4 experts and Mamba2 layers loaded frozen.

**Phase A is VALIDATED end-to-end on hardware (2026-06-14).**
- Nano: dry-run + 3-step ICH train. PEFT attached to all 6 attention layers
  (48 adapter tensors = 24 modules x A/B, all on `mixer.{q,k,v,o}_proj`, no
  expert/Mamba leakage), trainable 1.87M, losses 1.66 -> 1.53 -> 1.31, 39 GB
  peak. The real risk (PEFT silently matching zero modules on the hybrid
  graph) is cleared.
- Super: 3-step ICH train, `lora_mode=peft` with NO override flags. PEFT
  attached to all 8 attention layers (64 tensors = 32 modules), trainable
  3.21M, losses 1.27 -> 1.31 -> 1.08, ~90 GB peak, zero NVRM. Both FP8
  `o_proj` (layers 69, 78) received `lora_A`/`lora_B` — the on-hardware
  confirmation of item 1 below.

## Resolved / open items

0. **Target inventory now excludes skip-listed modules — DONE.** Validation
   surfaced that `build_target_inventory` (and thus `decide_lora_mode`, the
   saved `target_coverage.json`, and the inspector verdict) counted the
   `mtp.*` attention head as a trainable target, even though the loader skips
   it — Super reported 9 q_proj when only 8 train. Fixed: the inventory drops
   prefixes on the resolved family's `skip_st_prefixes`, so counts reflect
   reality (verdict now `q_proj: 8`). The inspector's raw storage census still
   shows all 9 as a layout fact. Two CPU tests added; unknown-family /
   config-less checkpoints inventory in full as before.

1. **FP8-target semantics for the PEFT path — DONE (shipped + hardware-validated).**
   The fail-fast guard used to treat any FP8-matching target as "demoted to
   frozen → hard error unless `--allow-fp8-targets`". Correct for the native
   path, wrong for PEFT: the loader converts FP8 to a frozen bf16 `nn.Linear`,
   which PEFT *can* wrap. `decide_lora_mode` now fires the FP8 guard only for
   NATIVE suffixes (those with NVFP4 modules); a suffix with no NVFP4 resolves
   to `peft` and trains its bf16 AND fp8 modules with no override. Confirmed on
   Super (`o_proj` = bf16 + 2 fp8 → peft, all 8 layers incl. the 2 fp8 trained
   without flags). Shipped in PR #5 with a `peft_fp8_mix` fixture and three CPU
   tests (suite 79 → 81).

2. **Merging a BF16 attention adapter is unsupported (decide serve path).**
   Both merge scripts requantize into NVFP4 base tensors and reject targets
   without NVFP4 scales (added in the merge-hardening PR). A bf16 attention
   LoRA has no NVFP4 base to requant into. Two options:
   - serve as a **dynamic PEFT adapter** (Nano already supports
     `--enable-lora --lora-modules` per the README) — no merge needed; or
   - extend a merge path to fold a bf16 delta into the bf16 base weight
     in place (no requant). Lower priority; the dynamic path covers Nano.
   Decision needed before claiming "attention LoRA" end-to-end on Super
   (whose serve path is merge-then-serve, not dynamic).

3. **Attention + expert LoRA in one run is currently impossible (own phase).**
   Native (experts, NVFP4) and PEFT (attention, bf16) are mutually exclusive
   in a single run — mixed targets are a hard error by design. Training both
   the routed experts and the attention in one adapter needs the two LoRA
   mechanisms to coexist on one model (PEFT wrapping bf16 attention while
   `NVFP4LoRALinear` carries expert LoRA). This is the actual engineering
   lift and should be its own plan/PR, gated on items 1-2 landing first.

4. **Interaction checks (validation, not code).** Gradient checkpointing +
   Mamba2 fast path + PEFT wrapping; `enable_input_require_grads` on the
   hybrid graph; adapter save/resume with `mixer.*` key names. All expected
   to work; confirm in the smoke.

## Proposed sequencing

- **Phase A — DONE.** Item 1 shipped (PR #5); Nano and Super attention LoRA
  validated end-to-end on hardware (training only — a `dynamic-serve` smoke is
  still nice-to-have but the train/save path is proven).
- **Phase B:** decide item 2; if extending merge, do the bf16-in-place fold +
  validate Super attention LoRA through merge-then-serve.
- **Phase C (the real feature):** item 3, combined attention+expert LoRA in one
  run. Separate plan.

## Effort estimate

- Phase A code: ~half a day (one `decide_lora_mode` branch + fixture/test).
  Validation: one GPU smoke session (~1h, Nano loads fast).
- Phase B: ~1 day if extending merge; ~0 if dynamic-serve is accepted for Nano
  and Super attention LoRA is deferred.
- Phase C: multi-day; coexisting LoRA mechanisms on one model is the crux.

## One-line takeaway

"Nemotron attention LoRA" is not a kernel/loader build — it is a PEFT-path
validation plus a small FP8-semantics fix (Phase A), with combined
attention+expert training as the genuine follow-on engineering (Phase C).
