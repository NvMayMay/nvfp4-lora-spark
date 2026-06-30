# Cross-architecture expert-LoRA testing plan

GOAL: honestly substantiate "NVFP4 LoRA train -> rekey -> runtime-LoRA serve works across model
architectures" for the **MoE expert-LoRA** capability. Today that loop is proven END-TO-END on
**GLM-4.5-Air only** (glm4_moe fused-3D). Everything else is either dense (proven via the Spider
cross-family table: llama 8B, qwen3 32B, mistral 24B) or untested at the expert path. Per the user's
reframe: we are testing **functionality across architectures**, NOT chasing a quality/EM number; data
choice is left to users. The bar is "the machinery works on this arch", not "this beats a benchmark".

## Two distinct expert-LoRA mechanisms (the thing we are proving generalizes)
The repo has TWO ways experts get a LoRA, decided by how the checkpoint stores routed experts:

1. **Fused-3D expert LoRA** (`expert_lora_r` path; `replace_moe_experts_with_nvfp4_3d` ->
   `NVFP4Experts3D`; per-expert `lora_A [E,r,in]` / `lora_B [E,out,r]`). Families with
   `moe_experts_class` set: `glm4_moe` (Glm4MoeNaiveMoe), `qwen3_5_moe` (Qwen3_5MoeExperts),
   `mistral4` (Mistral4NaiveMoe). Serve = emulation backend + `rekey_expert_lora_for_vllm.py`.
   **PROVEN on glm4_moe; UNTESTED on qwen3_5_moe and mistral4.**

2. **Per-module expert LoRA** (`moe_experts_class=None`; experts are per-expert NVFP4 `nn.Linear`;
   `replace_nvfp4_modules` handles them; LoRA is applied as ordinary `--target-modules` suffixes on
   the expert projections). Family: `nemotron_h`. **UNTESTED as an expert-LoRA target** (only attention
   LoRA proven on Nemotron historically).

Proving BOTH mechanisms on a 2nd+ model is what makes "works across architectures" honest.

## Targets on disk (no downloads required)
| # | Model | Family | Mechanism | Size | Cost | Priority |
|---|-------|--------|-----------|------|------|----------|
| T1 | Nemotron-3-Nano-30B-A3B-NVFP4 | nemotron_h | per-module | 30B-A3B | LOW (fast) | **first** |
| T2 | RedHatAI-Qwen3.5-122B-A10B-NVFP4 | qwen3_5_moe | fused-3D | 122B-A10B | HIGH | **second** |
| T3 | RedHatAI-Mistral-Small-4-119B-NVFP4 | mistral4 | fused-3D | 119B | HIGH | third |
| T4 | Nemotron-3-Super-120B-A12B-NVFP4 | nemotron_h | per-module | 120B-A12B | HIGH | optional |
| -- | Llama-4-Scout-109B-A17B-NVFP4 | (no family entry) | -- | 109B | -- | **SKIP** (UMA-fail, abandoned; see memory) |
| -- | Mistral-Small-3.2-24B-NVFP4 | mistral3 | (dense; no per-expert keys) | 24B | -- | not an expert target |

Rationale for order: T1 is small/fast and validates the *other* mechanism (per-module) cheaply -> highest
value/cost. T2 is the cleanest "different fused-3D arch, same binding contract" proof. T3 adds a 3rd
fused-3D family (different vision-text wrapper + MTP straggler). T4 only if we want a large per-module data point.

NOTE on the `mistral3` registry entry (codex review #8): `mistral3` declares
`moe_experts_class="Mistral4NaiveMoe"` (the family is generic), but the specific checkpoint on disk,
Mistral-Small-3.2-24B, is DENSE -- the trainer's `detect_moe_expert_storage` finds no per-expert keys and
takes the "declared MoE class but no per-expert keys -> treat as dense" branch
(`train_nvfp4_lora.py:322-325`). So it is not a registry bug; the entry is reusable for a future
Mistral-3 MoE checkpoint. It is simply not an expert-LoRA target with the weights we have. Verify this is
still the branch taken (log it) so the "dense" label is not assumed.

## What "works" means -- the functional pass bar (per target)
The bar must prove the **expert LoRA tensors specifically were trained, saved, rekeyed, bound, and
affect generation** -- NOT merely that "some adapter changed some output" (codex review #1,#3,#7: a raw
`base != adapter` delta can come from nondeterminism, prompt formatting, or accidentally-included
attention LoRA, and would be a dishonest pass). To isolate the expert path, **these functional smokes
use an EXPERT-ONLY LoRA config (no attention/MLP targets)** so any measured effect is attributable to
experts. A target PASSES iff ALL of:

1. **Load**: NVFP4 base loads; the right expert path engages (fused-3D: `replace_moe_experts_with_nvfp4_3d`
   swaps the declared `moe_experts_class`; per-module: expert linears appear as NVFP4 targets). Log the
   trainable-param count and assert expert tensors are in it and attention tensors are NOT (expert-only
   config -- guards the arm-B silent-no-op trap).
2. **Train + expert tensors actually move** (loss alone is insufficient -- codex #3):
   - **train loss decreases** by >=X% (gate: smoothed final < smoothed initial by a set margin) over a
     few hundred updates; periodic `best/` checkpoints write (needs the trainer-hang fix -- see Gating);
   - **expert LoRA `grad_norm > 0`** logged at >=1 step (the trainer already logs grad norm; assert it);
   - **>= N expert LoRA tensors differ from init** (snapshot a sample of `lora_B` at step 0 -- init zero --
     and assert post-train L2 delta > eps; `lora_B` is zero-init so any nonzero norm proves an update).
3. **Save**: `best/` adapter non-empty; **key/shape round-trip inspection** (codex #5,#9): sample the
   saved adapter keys, assert expert keys present, and assert per-projection `lora_A/lora_B` shapes match
   `[E,r,in]`/`[E,out,r]` (fused-3D) or the per-module suffix shapes (Nemotron). Record attention vs
   shared vs expert param split (should be expert-only here).
4. **Rekey** (fused-3D only): `rekey_expert_lora_for_vllm.py` -> vLLM per-expert layout. Assert NOT just
   non-zero count but **shape/layout correctness** (codex #5): expected #layers x #experts x #projections,
   per-tensor shapes, expert-index ordering preserved, gate/up split correct, no dropped expert tensors
   except explicitly-allowed stragglers. Fail-fast if zero match.
5. **Bind**: binding-contract check (`VERDICT PASS`) against the served base -- 0 blocked-routed.
   **Per-module (Nemotron) needs an explicit expert-key binding assertion** (codex #4,#7): confirm the
   served runtime actually accepts+applies the Nemotron expert module paths (sample the expert keys in the
   verdict), not just attention paths; fail if expert keys are ignored/renamed/treated as unsupported.
6. **Serve + DETERMINISTIC, EXPERT-ATTRIBUTED delta** (codex #1,#10): emulation serve READY; eval with
   **temperature 0 / fixed seed / fixed max tokens / identical prompt construction both arms** (codex
   #13). Run the control ladder on the SAME prompts: **(a) base, (b) zeroed-expert adapter, (c) real
   expert adapter**. PASS requires `base == zeroed-expert` (proves no spurious delta) AND
   `real-expert != base` on >= M/N prompts (proves the trained experts change generation). A delta that
   does NOT vanish when experts are zeroed = a bug (something other than experts is moving the output).
7. **Routing coverage** (codex #2): confirm the eval prompts actually route tokens through adapted
   experts -- log routed expert-id coverage during the serve eval (or, cheaper, assert via the train-time
   router stats that the trained experts are reachable). If eval prompts never route through any adapted
   expert, the delta is meaningless -> not a valid pass.
8. **GLM regression control** (codex #6): because all targets share the trainer-save + rekey machinery
   (just changed for the hang fix), re-run a SHORT GLM-4.5-Air expert-only smoke through the same updated
   pipeline first. A failure there is a shared-machinery regression, not an arch-specific finding.

FAIL of any step is a finding (file an issue + notebook entry). The point is to surface arch-specific
breaks (key translation, straggler tensors, expert-class mismatch, silent serve-side skip), not a leaderboard.

Concrete thresholds (codex #12), tune on the GLM control run: loss drop >= 5% smoothed; expert update
L2 > 1e-4; >= 90% of sampled expert tensors moved; deterministic delta on >= M=ceil(0.5*N) prompts.

Tiny **deterministic** data is preferred over arbitrary user data for these smokes (codex #11): the goal
is to force expert updates and a reproducible delta, not benchmark quality. A small fixed Spider/ICH
subset with a fixed seed is fine; record the exact subset + seed.

## Per-target configs and known risks
- **T1 Nemotron-3-Nano (per-module)**: seq2048, add expert-proj suffixes to `--target-modules`
  (confirm the Nemotron expert linear names), r16/a32, ~200 updates. Risk: dynamic key translation
  (`st_to_model=None`, `backbone.*` vs `model.*`); `mtp.*` skipped. Cheapest end-to-end -> run first to
  shake out the per-module path before spending a 120B box-day.
- **T2 Qwen3.5-122B (fused-3D)**: `--expert-lora-r 4 --expert-lora-alpha 8`, seq2048, ~200 updates,
  single box. Risk: the **MTP/speculation BF16 straggler** per 3.x target suffix (global-mode +
  partial-drop already handles it -- assert it triggers, don't regress). Adapter already exists for
  *attention* LoRA (`qwen3_5_122b_a10b_rh_nvfp4_lora_ich_v3_5`) so load/serve path is known-good; this run
  adds the EXPERT tensors. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **T3 Mistral-Small-4-119B (fused-3D)**: **seq2048 NOT 4096** (known UMA OOM at 4096 ~step 11-12, see
  memory `mistral_119b_uma_oom`), `expandable_segments:True`, ~150-200 updates. Risk: image_text_to_text
  wrapper (vision_tower/multi_modal_projector frozen + skipped); st_to_model `language_model.*` mapping.
- **T4 Nemotron-3-Super-120B (per-module, optional)**: as T1 but large; only if T1 passes and we want a
  big per-module data point. seq2048, expandable_segments.

## Gating / sequencing
1. **BLOCKER: land the trainer final-save-hang fix first.** Every target above ends in the same
   `save_to(output_dir)` root save that hung on all 3 GLM arms. Without the fix, each run needs manual
   babysitting + a kill-after-best/. Fix -> then run targets unattended.
2. Run **T1** (cheap) to validate the per-module expert path + the fixed save path end-to-end.
3. Run **T2** then **T3** (one box each; can parallelize box A + box B). Each is a SELF-CONTAINED
   single-box run (proving single-box capability), not distributed.
4. T4 optional.
5. Update the family table in README/docs + the notebook with PASS/FAIL per target. Encode
   **mechanism-specific status, not just model names** (codex #15): e.g. "fused-3D expert LoRA: GLM +
   Qwen3.5 (passed), Mistral-4 (untested); per-module expert LoRA: Nemotron (passed)". Do NOT generalize
   from T1 alone -- if T1 passes and T2/T3 fail, the honest claim is "per-module works; fused-3D remains
   GLM-only".

**OOM policy** (codex #14): for T2/T3, OOM at the planned config (seq2048 + expandable_segments) is a
**functional FAIL recorded as a finding**, not silently rescued by shrinking config. We MAY then retry at
a smaller seq/batch and record it as "passes at seq<=N" -- but the headline status must state the config,
never claim unqualified success after a post-hoc shrink.

## Out of scope (explicitly)
- Quality/EM/benchmark numbers per arch (we proved the value question already: NO-GO on EM for typical
  format tasks; data choice left to users).
- Distributed/2-box training (each target is single-box).
- Llama-4-Scout (abandoned), DeepSeek-V4 (LoRA-FT infeasible on 2x GB10 -- see memory).
- Engine breadth beyond vLLM 0.22.1 emulation (Phase 3, traction-gated).

## Validation artifacts (so it is credible, not anecdotal)
Per target, commit under `results/cross_arch/<model>/`: the train metrics.jsonl tail (loss curve), the
rekey PASS log, the binding-contract verdict, and the base-vs-adapter delta JSON from eval_retention.
Wire a one-line PASS/FAIL row into a `docs/cross_arch_status.md` table.
